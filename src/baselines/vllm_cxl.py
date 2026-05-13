"""
vLLM PagedAttention + CXL — Software-Managed Tiering Baseline.

This models the industry-standard approach: vLLM's block-level page manager
extended to manage CXL-expanded memory.  Page granularity = chunk size.

Key characteristics (making it a "fetch-then-decide" baseline):
  - Page manager allocates blocks in HBM (hot) and CXL (cold)
  - On page fault (chunk not in HBM): DMA full page from CXL
  - No metadata pre-screening — entire page transferred before decision
  - LRU eviction from HBM when capacity exceeded
  - cudaMemPrefetchAsync hints can be added but lack KV semantic awareness

This represents the most natural industry approach: take what works in
software (vLLM's PagedAttention) and extend the address space to CXL.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, "d:/LLM")

from src.runners.e2e_eval_runner import BaselinePolicy
from src.memory.cxl_queue_simulator import (
    CXLQueueSimulator, CXLQueueConfig, BaselineCXLSession, StepStats
)


class VLLMCXLPolicy(BaselinePolicy):
    """vLLM PagedAttention extended to CXL memory.

    Models:
      - Block manager: allocates pages (chunks) in HBM or CXL
      - Page fault: on access to CXL page, DMA full page
      - LRU eviction: when HBM is full
      - Prefetch hint: cudaMemPrefetchAsync based on sequential pattern
    """

    name = "vLLM-CXL"

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
        hbm_capacity_chunks: int = 16,
        enable_prefetch_hint: bool = True,
        prefetch_depth: int = 2,
    ):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.hbm_capacity = hbm_capacity_chunks
        self.enable_prefetch_hint = enable_prefetch_hint
        self.prefetch_depth = prefetch_depth
        self.cxl_session: Optional[BaselineCXLSession] = None

        # vLLM page table: chunk_id → location (HBM or CXL)
        self.hbm_pages: OrderedDict[int, bool] = OrderedDict()  # LRU ordered
        self.cxl_pages: Dict[int, bool] = {}
        self.access_history: List[int] = []
        self.step_count: int = 0

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self.hbm_pages.clear()
        self.cxl_pages.clear()
        self.access_history.clear()
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

        # ── HW page manager only sees addresses, not scores ──
        # Model: GPU touches top-3 chunks (page-level access counters)
        if chunk_attention_masses:
            items = sorted(chunk_attention_masses.items(), key=lambda x: x[1], reverse=True)
            accessed = [int(cid) for cid, _ in items[:3] if int(cid) not in anchor_set]
        else:
            accessed = []

        self.access_history.extend(accessed)

        # ── Page fault handling ──
        hbm_resident = set(self.hbm_pages.keys()) | anchor_set
        faults = [c for c in accessed if c not in hbm_resident]

        # ── HW prefetch hint: sequential neighbors ──
        prefetched = []
        if self.enable_prefetch_hint and len(self.access_history) >= 3:
            stride = self.access_history[-1] - self.access_history[-2]
            prev_stride = self.access_history[-2] - self.access_history[-3]
            if stride == prev_stride and abs(stride) <= 2:
                last = self.access_history[-1]
                for i in range(1, self.prefetch_depth + 1):
                    nxt = last + i * stride
                    if 0 <= nxt < num_chunks and nxt not in hbm_resident and nxt not in faults:
                        prefetched.append(nxt)

        # ── DMA faulted + prefetched pages from CXL ──
        all_fetches = list(dict.fromkeys(faults + prefetched))

        if all_fetches:
            self.cxl_session.cxl.submit_payload_fetch(all_fetches, 0)

            # GPU actually uses only faulted chunks; prefetched may be wasted
            used_set = set(faults)
            invalid = [c for c in prefetched if c not in used_set]
            if invalid:
                self.cxl_session.cxl.mark_chunks_invalid(invalid)
            self.cxl_session.cxl.mark_chunks_used(faults if faults else all_fetches[:1])
        else:
            # No faults — everything hit in HBM/anchors
            # Still need to account for zero-fetch step in CXL stats
            pass

        # ── LRU page management ──
        for chunk in all_fetches:
            if chunk in self.cxl_pages:
                del self.cxl_pages[chunk]

            # Evict LRU if HBM is full
            while len(self.hbm_pages) >= self.hbm_capacity and self.hbm_pages:
                evicted, _ = self.hbm_pages.popitem(last=False)
                self.cxl_pages[evicted] = True

            self.hbm_pages[chunk] = True
            self.hbm_pages.move_to_end(chunk)

        # Refresh LRU order for already-resident accessed chunks
        for c in accessed:
            if c in self.hbm_pages:
                self.hbm_pages.move_to_end(c)

        # ── Final selection ──
        selected = anchor_set | set(self.hbm_pages.keys())

        # ── Gold: only used for internal accounting ──
        if chunk_attention_masses:
            sorted_by_attn = sorted(chunk_attention_masses.items(),
                                    key=lambda x: x[1], reverse=True)
            gold = [int(cid) for cid, _ in sorted_by_attn[:budget_chunks]
                   if int(cid) not in anchor_set]
        else:
            gold = faults[:budget_chunks]

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
