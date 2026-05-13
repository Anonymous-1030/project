"""
Oracle-Policy — Ordering-Independent Upper Bound.

This baseline has PERFECT utility knowledge: it knows exactly which chunks
will have the highest attention at the NEXT decode step.  However, it is
STILL subject to CXL queue constraints (bandwidth, queue depth, serialization
latency).

Key insight: Even with perfect knowledge, the CXL bandwidth limit creates a
hard latency floor that PROSE approaches.  This separates the "oracle gap"
from the "ordering gap."

Two variants:
  1. Oracle-SBFI (score-before-fetch): uses 64B summaries to gate admission
     (same ordering as PROSE, but perfect predictor)
  2. Oracle-FTS (fetch-then-score): fetches ALL chunks, then keeps top-K
     (same ordering as baseline, but perfect knowledge)

The gap between Oracle-SBFI and Oracle-FTS IS the ordering benefit —
it cannot be closed by a better predictor.

Configurations tested:
  - CXL bandwidth: 4-64 GB/s
  - Offload ratio: 70-98%
  - Context length: 8K-128K
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


class OraclePolicy(BaselinePolicy):
    """Perfect-utility oracle with CXL queue constraints.

    Knows the TRUE future attention distribution (cheating from the paper's
    perspective) but still respects hardware limits:
      - CXL bandwidth: serialization delay per byte
      - Queue depth: bounded in-flight requests
      - DRAM access: row buffer hit/miss latency
    """

    name = "Oracle-Policy"

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
        use_sbfi: bool = True,
        oracle_lookahead: int = 1,
    ):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.use_sbfi = use_sbfi  # True = SBFI ordering, False = FTS ordering
        self.oracle_lookahead = oracle_lookahead
        self.cxl_session: Optional[BaselineCXLSession] = None
        self._future_attention: Optional[np.ndarray] = None
        self.step_count: int = 0

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self._future_attention = None
        self.step_count = 0

    def set_future_attention(self, future_attn: np.ndarray):
        """Set the oracle's future knowledge (called by runner each step).

        In real evaluation, the runner provides the TRUE next-step attention
        that no real policy can see.
        """
        self._future_attention = future_attn

    def select_active_chunks(
        self,
        num_chunks: int,
        budget_chunks: int,
        chunk_attention_masses: Dict[int, float],
        anchor_ids: List[int],
        step: int,
    ) -> List[int]:
        """Oracle selection: perfect ranking with CXL queue constraints."""
        self.step_count = step

        if self.cxl_session is None:
            self.cxl_session = BaselineCXLSession(self.cxl_config)

        anchor_set = set(anchor_ids)

        # Use current attention as fallback if no future provided
        if self._future_attention is None or len(self._future_attention) != num_chunks:
            attn_arr = np.zeros(num_chunks)
            for cid, mass in chunk_attention_masses.items():
                if 0 <= cid < num_chunks:
                    attn_arr[cid] = mass
            self._future_attention = attn_arr

        # Oracle ranking: perfect knowledge of which chunks matter
        oracle_ranking = list(np.argsort(self._future_attention)[::-1])

        # Remove anchors from oracle ranking
        candidates = [int(c) for c in oracle_ranking if int(c) not in anchor_set]

        if self.use_sbfi:
            # Oracle-SBFI: perfect scoring of summaries
            selected = candidates[:budget_chunks]
            # Simulate: fetch summaries → oracle says "these exact ones" → fetch payloads
            _ = self.cxl_session.cxl.submit_summary_fetch(candidates[:budget_chunks * 2], 0)
            _ = self.cxl_session.cxl.submit_payload_fetch(selected, 0)
            self.cxl_session.cxl.mark_chunks_used(selected)
        else:
            # Oracle-FTS: fetch ALL candidates → keep perfect top-K
            # Even oracle wastes bandwidth on rejected chunks
            selected, _ = self.cxl_session.fetch_then_score(
                candidate_ids=candidates,
                scorer_fn=lambda ids: ids,  # Oracle already knows the order
                budget_chunks=budget_chunks,
            )

        gold = candidates[:budget_chunks]
        self.cxl_session.end_step(selected, gold)
        self.cxl_session.advance_step()

        return sorted(set(selected) | anchor_set)

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

    def get_cxl_queue_rho(self) -> float:
        if self.cxl_session is None or not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.cxl_stats.queue_utilization_rho
                              for r in self.cxl_session.step_results]))


class OracleCandidateOnlyPolicy(BaselinePolicy):
    """Oracle-Candidate-Only: perfect exposure + ODUS-X ranking.

    Separates exposure gap (which candidates ULF generates) from ranking gap
    (how well ODUS-X scores them).

    - Generator = Oracle (perfect exposure: all truly useful chunks are candidates)
    - Ranker = ODUS-X (same as PROSE)
    """

    name = "Oracle-Candidate"

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
    ):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.cxl_session: Optional[BaselineCXLSession] = None
        self._future_attention: Optional[np.ndarray] = None
        self._ewma: Optional[np.ndarray] = None
        self._decay = 0.3
        self.step_count: int = 0

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self._future_attention = None
        self._ewma = None
        self.step_count = 0

    def set_future_attention(self, future_attn: np.ndarray):
        self._future_attention = future_attn

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

        attn_arr = np.zeros(num_chunks)
        for cid, mass in chunk_attention_masses.items():
            if 0 <= cid < num_chunks:
                attn_arr[cid] = mass

        if self._ewma is None:
            self._ewma = attn_arr.copy()
        else:
            self._ewma = self._decay * attn_arr + (1 - self._decay) * self._ewma

        # Perfect exposure: all high-attention chunks are candidates
        if self._future_attention is not None and len(self._future_attention) == num_chunks:
            oracle_ranking = list(np.argsort(self._future_attention)[::-1])
            candidates = [int(c) for c in oracle_ranking[:budget_chunks * 3]
                         if int(c) not in anchor_set]
        else:
            candidates = [int(c) for c in np.argsort(attn_arr)[::-1][:budget_chunks * 3]
                         if int(c) not in anchor_set]

        # ODUS-X scoring (same as PROSE, not oracle)
        def odus_score(ids):
            scores = {}
            for cid in ids:
                if cid in anchor_set:
                    continue
                s = 0.4 * float(attn_arr[cid]) if cid < len(attn_arr) else 0
                if self._ewma is not None and cid < len(self._ewma):
                    s += 0.3 * float(self._ewma[cid])
                scores[cid] = s
            return sorted(scores, key=scores.get, reverse=True)

        selected, _, _ = self.cxl_session.score_before_fetch(
            candidate_ids=candidates,
            scorer_fn=odus_score,
            budget_chunks=budget_chunks,
        )

        gold = [c for c in oracle_ranking if c not in anchor_set][:budget_chunks] if self._future_attention is not None else selected
        self.cxl_session.end_step(selected, gold)
        self.cxl_session.advance_step()

        return sorted(set(selected) | anchor_set)

    def get_mean_recovery(self) -> float:
        if self.cxl_session is None or not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.recovery for r in self.cxl_session.step_results]))
