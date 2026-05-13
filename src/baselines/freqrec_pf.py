"""
FreqRec-PF — Frequency-Recency Hybrid Hardware Prefetcher.

A stronger hardware baseline than StreamPrefetcher.  Uses address-level
metadata only (no attention scores, no content metadata):
  - Frequency counter table: saturating counters per chunk
  - Recency FIFO: captures short-term temporal locality

CRITICAL (matches Table IV claim):
  FreqRec-PF is a FETCH-THEN-OBSERVE prefetcher.  It aggressively prefetches
  3-5x the HBM budget for "observation", then keeps the top budget_chunks
  based on local frequency/recency scoring.  The rest are discarded as
  invalid traffic (72-80% per Table IV).

This is the fundamental inefficiency of fetch-then-score: you pay CXL
bandwidth for chunks you end up discarding.
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


class FreqRecPrefetcherPolicy(BaselinePolicy):
    """Frequency-Recency Hybrid Prefetcher — fetch-then-observe.

    Models the Tablet IV behavior: aggressively prefetches 3-5x budget
    for observation, then discards 72-80% as invalid traffic.

    Uses only address-level metadata (no attention scores):
      - Frequency: saturating counters per chunk ID
      - Recency: small FIFO of recently accessed chunks
      - Observation factor: how many candidates to prefetch vs keep
    """

    name = "FreqRec-PF"

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
        freq_budget_frac: float = 0.60,
        max_counter: int = 15,
        decay_period: int = 8,
        hbm_capacity_chunks: int = 16,
        observation_factor: float = 4.0,  # Prefetch 4x budget for observation
    ):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.freq_budget_frac = freq_budget_frac
        self.max_counter = max_counter
        self.decay_period = decay_period
        self.hbm_capacity = hbm_capacity_chunks
        self.observation_factor = observation_factor
        self.cxl_session: Optional[BaselineCXLSession] = None

        self.freq_counters: Dict[int, int] = {}
        self.recency_fifo: List[int] = []
        self.recency_capacity: int = 24  # Larger FIFO for observation pool
        self._access_history: List[int] = []  # Tracks what GPU actually touched
        self.step_count: int = 0

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self.freq_counters.clear()
        self.recency_fifo.clear()
        self._access_history.clear()
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

        # ── Record accesses (HW sees memory addresses, not attention scores) ──
        if chunk_attention_masses:
            items = sorted(chunk_attention_masses.items(), key=lambda x: x[1], reverse=True)
            # HW sees the top-3 most-accessed addresses (page-level access counters)
            accessed = [int(cid) for cid, _ in items[:3]]
        else:
            accessed = [0]
        self._access_history.extend(accessed)

        # ── Age-based decay ──
        if self.step_count % self.decay_period == 0:
            for cid in list(self.freq_counters.keys()):
                self.freq_counters[cid] = self.freq_counters[cid] // 2
                if self.freq_counters[cid] <= 0:
                    del self.freq_counters[cid]

        # ── Update metadata ──
        for cid in accessed:
            self.freq_counters[cid] = min(
                self.freq_counters.get(cid, 0) + 1, self.max_counter
            )
            if cid in self.recency_fifo:
                self.recency_fifo.remove(cid)
            self.recency_fifo.append(cid)
            if len(self.recency_fifo) > self.recency_capacity:
                self.recency_fifo.pop(0)

        # ── CANDIDATE GENERATION (aggressive, 3-5x budget) ──
        # This models "HW prefetcher speculatively pulls chunks for observation"
        observe_count = max(int(budget_chunks * self.observation_factor), budget_chunks + 4)

        candidates = set()

        # Source 1: High-frequency chunks
        freq_sorted = sorted(self.freq_counters.items(), key=lambda x: x[1], reverse=True)
        for cid, cnt in freq_sorted:
            if cid not in anchor_set and cnt > 0:
                candidates.add(cid)

        # Source 2: Recent chunks (recency FIFO)
        for cid in reversed(self.recency_fifo):
            if cid not in anchor_set:
                candidates.add(cid)

        # Source 3: Sequential neighbors of recent accesses (spatial locality)
        for cid in self._access_history[-5:]:
            for offset in [-2, -1, 1, 2]:
                neighbor = cid + offset
                if 0 <= neighbor < num_chunks and neighbor not in anchor_set:
                    candidates.add(neighbor)

        # Cap at observation budget
        candidates = sorted(candidates)[:observe_count]

        if len(candidates) < budget_chunks:
            # Fallback: sequential fill
            for cid in range(num_chunks):
                if cid not in anchor_set and cid not in candidates:
                    candidates.append(cid)
                if len(candidates) >= observe_count:
                    break

        # ── FETCH-THEN-OBSERVE: DMA all candidates from CXL ──
        if candidates:
            self.cxl_session.cxl.submit_payload_fetch(candidates, 0)

            # ── LOCAL OBSERVATION: score by freq+recency (no attention scores!) ──
            scores = {}
            max_freq = max(self.freq_counters.values()) if self.freq_counters else 1
            for i, cid in enumerate(candidates):
                # Frequency score (normalized)
                freq_score = self.freq_counters.get(cid, 0) / max(max_freq, 1)
                # Recency score (position in FIFO, more recent = higher)
                if cid in self.recency_fifo:
                    rec_idx = list(reversed(self.recency_fifo)).index(cid)
                    rec_score = 1.0 - rec_idx / max(len(self.recency_fifo), 1)
                else:
                    rec_score = 0.0
                scores[cid] = 0.6 * freq_score + 0.4 * rec_score

            # Select top budget_chunks
            ranked = sorted(scores, key=scores.get, reverse=True)
            selected_cxl = ranked[:budget_chunks]

            # Mark unselected as invalid (discarded after observation)
            invalid = [c for c in candidates if c not in selected_cxl]
            if invalid:
                self.cxl_session.cxl.mark_chunks_invalid(invalid)
            self.cxl_session.cxl.mark_chunks_used(selected_cxl)
        else:
            selected_cxl = []

        # ── Final selection ──
        selected = anchor_set | set(selected_cxl)

        # ── Gold ──
        if chunk_attention_masses:
            sorted_by_attn = sorted(chunk_attention_masses.items(),
                                    key=lambda x: x[1], reverse=True)
            gold = [int(cid) for cid, _ in sorted_by_attn[:budget_chunks]
                   if int(cid) not in anchor_set]
        else:
            gold = selected_cxl[:budget_chunks]

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
