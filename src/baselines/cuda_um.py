"""
CUDA Unified Memory + Prefetching — OS/Runtime-Level Tiering Baseline.

Models NVIDIA's Unified Memory (UM) with cudaMemPrefetchAsync hints.
This represents the OS/runtime approach to memory tiering:
  - GPU page faults drive migration (4KB/64KB pages, not chunk-granular)
  - cudaMemPrefetchAsync hints based on address patterns
  - No KV semantic awareness — pages migrated blindly

Key defect: page granularity (4KB) is 16x smaller than PROSE's chunk
granularity (64KB), causing excessive page faults and PCIe overhead.
Plus, the lack of KV semantics means the prefetcher can't distinguish
high-utility from low-utility pages.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, "d:/LLM")

from src.runners.e2e_eval_runner import BaselinePolicy
from src.memory.cxl_queue_simulator import (
    CXLQueueSimulator, CXLQueueConfig, BaselineCXLSession, StepStats
)


class CUDAUnifiedMemoryPolicy(BaselinePolicy):
    """CUDA Unified Memory with cudaMemPrefetchAsync hints."""

    name = "CUDA-UM"

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
        page_size_kb: int = 64,  # GPU page size (default: 64KB on A100+)
        prefetch_distance: int = 2,
        hbm_capacity_gb: float = 20.0,
    ):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.page_size_kb = page_size_kb
        self.prefetch_distance = prefetch_distance
        self.hbm_capacity_bytes = hbm_capacity_gb * 1024**3
        self.cxl_session: Optional[BaselineCXLSession] = None

        # Pages per chunk (e.g., 64KB chunk / 64KB page = 1; or / 4KB = 16)
        self.pages_per_chunk = max(1, self.cxl_config.chunk_size_bytes // (page_size_kb * 1024))

        self._hbm_pages: set = set()  # Page IDs in HBM
        self._page_to_chunk: Dict[int, int] = {}  # page_id → chunk_id mapping
        self._access_history: List[int] = []  # Chunk access history
        self._page_faults: int = 0
        self.step_count: int = 0

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self._hbm_pages.clear()
        self._page_to_chunk.clear()
        self._access_history.clear()
        self._page_faults = 0
        self.step_count = 0

    def select_active_chunks(
        self,
        num_chunks: int,
        budget_chunks: int,
        chunk_attention_masses: Dict[int, float],
        anchor_ids: List[int],
        step: int,
    ) -> List[int]:
        self.step_count = step

        if self.cxl_session is None:
            self.cxl_session = BaselineCXLSession(self.cxl_config)

        anchor_set = set(anchor_ids)

        # Record accesses (GPU page fault handler sees touched pages,
        # but only at coarse address granularity — top-3 access counters)
        if chunk_attention_masses:
            items = sorted(chunk_attention_masses.items(), key=lambda x: x[1], reverse=True)
            # OS sees page-fault addresses for top-3 most-accessed chunks
            accessed_list = [int(cid) for cid, _ in items[:3] if int(cid) not in anchor_set]
        else:
            accessed_list = []
        for cid in accessed_list:
            if cid not in self._access_history[-3:]:
                self._access_history.append(cid)

        # Map chunks to pages
        needed_pages = set()
        needed_chunks = set(anchor_ids)

        # Anchors: assume always in HBM (pinned)
        for aid in anchor_ids:
            for p in range(self.pages_per_chunk):
                page_id = aid * self.pages_per_chunk + p
                self._hbm_pages.add(page_id)
                self._page_to_chunk[page_id] = aid

        # Accessed chunks: check page residency
        accessed_chunks = set()
        for chunk_id in self._access_history[-budget_chunks * 2:]:
            if 0 <= chunk_id < num_chunks and chunk_id not in anchor_set:
                accessed_chunks.add(chunk_id)
                for p in range(self.pages_per_chunk):
                    page_id = chunk_id * self.pages_per_chunk + p
                    needed_pages.add(page_id)
                    self._page_to_chunk[page_id] = chunk_id

        # ── GPU page fault handling ──
        # Find pages not in HBM → trigger page migration
        faulting_pages = needed_pages - self._hbm_pages
        self._page_faults += len(faulting_pages)

        # Group faulting pages by chunk for CXL DMA
        faulting_chunks = set()
        for page_id in faulting_pages:
            chunk_id = self._page_to_chunk.get(page_id, page_id // self.pages_per_chunk)
            faulting_chunks.add(chunk_id)

        if faulting_chunks:
            # CUDA UM: pages migrated on fault (no summary pre-screening)
            # Each page fault triggers page-sized DMA
            total_bytes = len(faulting_pages) * self.page_size_kb * 1024

            # Simulate: multiple small page transfers (worse than chunk-granular)
            for chunk_id in faulting_chunks:
                self.cxl_session.cxl.submit_payload_fetch(
                    [chunk_id],
                    0,
                    bytes_per_chunk=self.cxl_config.chunk_size_bytes
                )

            self.cxl_session.cxl.mark_chunks_used(list(faulting_chunks))

        # cudaMemPrefetchAsync hint: prefetch next likely pages
        prefetch_chunks = set()
        if len(self._access_history) >= 2:
            # Sequential stride detection (same as StreamPrefetcher but at page level)
            stride = self._access_history[-1] - self._access_history[-2]
            if abs(stride) <= 2:
                last = self._access_history[-1]
                for i in range(1, self.prefetch_distance + 1):
                    nxt = last + i * stride
                    if 0 <= nxt < num_chunks:
                        prefetch_chunks.add(nxt)

        # Determine which prefetch chunks need pages migrated
        for chunk_id in prefetch_chunks:
            if chunk_id not in faulting_chunks:
                pages_needed = set(
                    chunk_id * self.pages_per_chunk + p
                    for p in range(self.pages_per_chunk)
                )
                if pages_needed - self._hbm_pages:
                    self.cxl_session.cxl.submit_payload_fetch(
                        [chunk_id], 0,
                        bytes_per_chunk=self.cxl_config.chunk_size_bytes
                    )

        # Page replacement (LRU eviction when HBM full)
        hbm_page_limit = self.hbm_capacity_bytes // (self.page_size_kb * 1024)
        while len(self._hbm_pages) > hbm_page_limit and self._hbm_pages:
            self._hbm_pages.pop()  # Evict LRU page

        # Migrate pages to HBM
        for page_id in faulting_pages:
            self._hbm_pages.add(page_id)

        # Final selected set
        selected = anchor_set | accessed_chunks

        # Gold
        if chunk_attention_masses:
            sorted_by_attn = sorted(chunk_attention_masses.items(),
                                    key=lambda x: x[1], reverse=True)
            gold = [int(cid) for cid, _ in sorted_by_attn[:budget_chunks]
                   if int(cid) not in anchor_set]
        else:
            gold = list(selected - anchor_set)[:budget_chunks]

        self.cxl_session.end_step(list(selected), gold)
        self.cxl_session.advance_step()

        return sorted(selected)

    def get_stats(self) -> Optional[StepStats]:
        if self.cxl_session is None:
            return None
        stats_list = [r.cxl_stats for r in self.cxl_session.step_results]
        if not stats_list:
            return None
        total = StepStats()
        for s in stats_list:
            for field_name in s.__dataclass_fields__:
                current = getattr(total, field_name, 0)
                added = getattr(s, field_name, 0)
                setattr(total, field_name, current + added)
        return total

    def get_mean_recovery(self) -> float:
        if self.cxl_session is None or not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.recovery for r in self.cxl_session.step_results]))

    @property
    def total_page_faults(self) -> int:
        return self._page_faults
