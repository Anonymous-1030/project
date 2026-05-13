"""
FreqRec-PF + Meta — FreqRec with 64B Summary Budget but Fetch-Then-Score.

Key question (Tier 2 #6): Does the benefit come from metadata itself
or from how metadata is USED (ordering)?

This baseline gives FreqRec-PF the SAME 64B summary budget as PROSE,
but keeps fetch-then-score ordering.  The metadata is used only for
post-fetch verification, not for admission gating.

Contrast with FreqRec-PF (#1): same frequency counters, but now has
64B summaries to read BEFORE deciding what to fetch.
However, since ordering is still FTS, the summaries are read but the
full chunks are still fetched before ranking — wasting the summary's
gating potential.
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


class FreqRecMetaPolicy(BaselinePolicy):
    """FreqRec-PF with 64B metadata summary access, but fetch-then-score."""

    name = "FreqRec-PF+Meta"

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
        freq_budget_frac: float = 0.60,
        max_counter: int = 15,
        decay_period: int = 8,
        hbm_capacity_chunks: int = 16,
        summary_window: int = 20,  # Up to 20 chunks we'll read summaries for
    ):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.freq_budget_frac = freq_budget_frac
        self.max_counter = max_counter
        self.decay_period = decay_period
        self.hbm_capacity = hbm_capacity_chunks
        self.summary_window = summary_window
        self.cxl_session: Optional[BaselineCXLSession] = None

        self.freq_counters: Dict[int, int] = {}
        self.recency_fifo: List[int] = []
        self.recency_capacity: int = 16
        self._hbm_set: set = set()
        self.step_count: int = 0
        self._summary_cache: Dict[int, float] = {}  # Simulated 64B summary data

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self.freq_counters.clear()
        self.recency_fifo.clear()
        self._hbm_set.clear()
        self._summary_cache.clear()
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

        # Record accesses
        if chunk_attention_masses:
            items = sorted(chunk_attention_masses.items(), key=lambda x: x[1], reverse=True)
            accessed_list = [int(cid) for cid, _ in items[:3]]
        else:
            accessed_list = [0]

        # Decay
        if self.step_count % self.decay_period == 0:
            for cid in list(self.freq_counters.keys()):
                self.freq_counters[cid] = self.freq_counters[cid] // 2
                if self.freq_counters[cid] <= 0:
                    del self.freq_counters[cid]

        # Update counters
        for cid in accessed_list:
            self.freq_counters[cid] = min(
                self.freq_counters.get(cid, 0) + 1, self.max_counter
            )
            if cid in self.recency_fifo:
                self.recency_fifo.remove(cid)
            self.recency_fifo.append(cid)
            if len(self.recency_fifo) > self.recency_capacity:
                self.recency_fifo.pop(0)

        # ── KEY DIFFERENCE: read 64B summaries for top candidates ──
        freq_sorted = sorted(self.freq_counters.items(), key=lambda x: x[1], reverse=True)
        top_freq_candidates = [cid for cid, _ in freq_sorted[:self.summary_window]
                               if cid not in anchor_set]

        recency_candidates = [cid for cid in reversed(self.recency_fifo)
                             if cid not in anchor_set and cid not in top_freq_candidates]

        summary_candidates = (top_freq_candidates + recency_candidates)[:self.summary_window]

        # Read summaries from CXL (64B each)
        if summary_candidates:
            self.cxl_session.cxl.submit_summary_fetch(summary_candidates, 0)

            # Simulate reading summary content: we get an attention-proxy score
            for cid in summary_candidates:
                if cid < num_chunks:
                    mass = chunk_attention_masses.get(cid, 0.0)
                    self._summary_cache[cid] = float(mass)

        # ── BUT: still fetch-then-score! ──
        # Even though we HAVE the summaries, the ordering is still FTS.
        # We fetch full chunks for all candidates, then use summaries only
        # for post-hoc verification.

        selected = set(anchor_ids)
        freq_budget = max(1, int(budget_chunks * self.freq_budget_frac))
        recency_budget = budget_chunks - freq_budget

        # Select by frequency (augmented by summary scores)
        freq_scored = []
        for cid, freq in freq_sorted:
            if cid not in anchor_set:
                summary_boost = self._summary_cache.get(cid, 0.0) * 2.0
                freq_scored.append((cid, freq + summary_boost))

        freq_scored.sort(key=lambda x: x[1], reverse=True)
        for cid, _ in freq_scored:
            if len(selected) - len(anchor_set) < freq_budget:
                selected.add(cid)

        # Select by recency
        for cid in reversed(self.recency_fifo):
            if cid not in anchor_set and cid not in selected:
                if len(selected) - len(anchor_set) >= budget_chunks:
                    break
                selected.add(cid)

        # ── CXL: fetch full payloads for all selected (FTS!) ──
        chunks_to_fetch = [c for c in selected
                          if c not in self._hbm_set and c not in anchor_set]

        if chunks_to_fetch:
            self.cxl_session.cxl.submit_payload_fetch(chunks_to_fetch, 0)

            # Even with summaries, we still fetched ALL candidates as full
            # payloads.  Some will be wasted (invalid traffic).
            actually_useful = set(accessed_list)
            invalid = [c for c in chunks_to_fetch if c not in actually_useful]
            if invalid:
                self.cxl_session.cxl.mark_chunks_invalid(invalid)

            self.cxl_session.cxl.mark_chunks_used(
                [c for c in chunks_to_fetch if c in actually_useful]
            )

        # HBM management
        for c in chunks_to_fetch:
            if len(self._hbm_set) >= self.hbm_capacity and self._hbm_set:
                self._hbm_set.pop()
            self._hbm_set.add(c)

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
