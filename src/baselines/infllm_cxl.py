"""
InfLLM-CXL — Retrieval-Based KV Selection with Fetch-Then-Decide.

InfLLM uses retrieval to select which KV entries to keep locally.
This baseline models:
  - Local cache: top-20% retrieved KV entries
  - CXL: remaining 80% (offloaded)
  - On retrieval miss: fetch full chunks from CXL, then verify locally

Demonstrates that retrieval-based KV reduction has the same ordering
defect as other fetch-then-decide approaches when extended to CXL.
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


class InfLLMCXLPolicy(BaselinePolicy):
    """InfLLM retrieval-based KV selection with fetch-then-decide on CXL."""

    name = "InfLLM-CXL"

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
        top_k_retrieve: int = 16,
        local_cache_ratio: float = 0.20,
        retrieval_overfetch: float = 2.0,  # Fetch 2x more than needed
    ):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.top_k_retrieve = top_k_retrieve
        self.local_cache_ratio = local_cache_ratio
        self.retrieval_overfetch = retrieval_overfetch
        self.cxl_session: Optional[BaselineCXLSession] = None

        self._query_signatures: List[np.ndarray] = []
        self._chunk_signatures: Dict[int, np.ndarray] = {}
        self._local_cache: List[int] = []  # FIFO-ordered local cache
        self._local_cache_set: set = set()  # Fast lookup
        self.step_count: int = 0
        self._rng = np.random.RandomState(42)

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self._query_signatures.clear()
        self._chunk_signatures.clear()
        self._local_cache.clear()
        self.step_count = 0
        self._local_cache_set: set = set()  # Fast lookup

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

        # Simulate query signature from attention distribution
        attn_arr = np.zeros(num_chunks)
        for cid, mass in chunk_attention_masses.items():
            if 0 <= cid < num_chunks:
                attn_arr[cid] = mass

        # Query signature: top-K attention values as proxy embedding
        query_sig = attn_arr.copy()
        self._query_signatures.append(query_sig)

        # Build/update chunk signatures lazily (simulating retrieval index)
        for cid in range(num_chunks):
            if cid not in self._chunk_signatures:
                self._chunk_signatures[cid] = self._rng.randn(16).astype(np.float32)
                self._chunk_signatures[cid] /= np.linalg.norm(self._chunk_signatures[cid])

        # Retrieval: cosine similarity between query and chunk signatures
        # (simulated: attention mass is the true relevance signal)
        num_retrieve = int(self.top_k_retrieve * self.retrieval_overfetch)

        # Simulated retrieval: top-K by current attention mass (with noise)
        noisy_attn = attn_arr + self._rng.normal(0, 0.01, num_chunks)
        retrieved = list(np.argsort(noisy_attn)[::-1][:num_retrieve])
        retrieved = [int(c) for c in retrieved if int(c) not in anchor_set]

        # ── Fetch-then-decide: DMA ALL retrieved chunks from CXL ──
        # InfLLM fetches full chunks, then verifies locally
        chunks_to_fetch = [c for c in retrieved if c not in self._local_cache_set]

        if chunks_to_fetch:
            self.cxl_session.cxl.submit_payload_fetch(chunks_to_fetch, 0)

            # Some retrieved chunks are false positives (retrieval noise)
            # These become invalid traffic
            true_top = set(np.argsort(attn_arr)[::-1][:self.top_k_retrieve])
            false_positives = [c for c in chunks_to_fetch if c not in true_top]
            true_positives = [c for c in chunks_to_fetch if c in true_top]

            if false_positives:
                self.cxl_session.cxl.mark_chunks_invalid(false_positives)
            self.cxl_session.cxl.mark_chunks_used(true_positives)

        # Update local cache (FIFO, keep most recent)
        for c in chunks_to_fetch:
            if c in self._local_cache_set:
                continue
            cache_limit = max(1, int(num_chunks * self.local_cache_ratio))
            while len(self._local_cache) >= cache_limit and self._local_cache:
                removed = self._local_cache.pop(0)
                self._local_cache_set.discard(removed)
            self._local_cache.append(c)
            self._local_cache_set.add(c)

        # Final selection: anchors + local cache + just-fetched chunks
        # Include freshly fetched chunks in this step's visible set
        # (they were DMA'd from CXL THIS step, so they are available for attention)
        just_fetched = set(chunks_to_fetch) if chunks_to_fetch else set()
        selected = anchor_set | self._local_cache_set | just_fetched

        # Gold
        sorted_by_attn = sorted(chunk_attention_masses.items(),
                                key=lambda x: x[1], reverse=True)
        gold = [int(cid) for cid, _ in sorted_by_attn[:budget_chunks]
               if int(cid) not in anchor_set]

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
