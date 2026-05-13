"""
Perfect Oracle Bounds — Separating Prediction Quality from Ordering Cost.

Two oracle bounds that decompose the PROSE contribution:

1. Perfect-FTS Oracle:
   Fetches ONLY the chunks that will eventually be useful, but still pays
   full payload DMA cost for each. This gives the upper bound on what
   perfect prediction can achieve WITHOUT SBFI ordering.

   Contribution isolated: prediction quality (how good is the scorer?)

2. Perfect-SBFI Oracle:
   Knows the correct useful chunks before fetch AND uses SBFI ordering
   (metadata read → perfect admission → payload DMA only for useful).
   This gives the absolute upper bound on what ANY SBFI mechanism can achieve.

   Contribution isolated: hardware placement (is CE necessary?)

Comparison ladder:
  Tuned-FTS < PROSE-FTS < SW-SBFI < PROSE < Perfect-FTS Oracle < Perfect-SBFI Oracle

The gaps between adjacent entries isolate:
  - Tuned-FTS → PROSE-FTS: value of PROSE's ranker (same ordering)
  - PROSE-FTS → SW-SBFI: value of SBFI ordering (software)
  - SW-SBFI → PROSE: value of hardware CE acceleration
  - PROSE → Perfect-FTS: remaining prediction gap (scorer imperfection)
  - Perfect-FTS → Perfect-SBFI: ordering benefit with perfect knowledge
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, "d:/LLM")

from src.runners.e2e_eval_runner import BaselinePolicy
from src.memory.cxl_queue_simulator import (
    CXLQueueConfig, BaselineCXLSession, StepStats
)


class PerfectFTSOraclePolicy(BaselinePolicy):
    """Perfect Fetch-Then-Score Oracle.

    Has perfect knowledge of which chunks are useful, but still uses
    FTS ordering: fetches all candidates as full payloads, then keeps
    only the useful ones.

    This is DIFFERENT from the existing OraclePolicy because:
    - OraclePolicy fetches only top-K (it cheats on ordering too)
    - PerfectFTSOracle fetches a realistic candidate set, then filters

    The key insight: even with perfect prediction, FTS ordering wastes
    bandwidth on the candidate-set expansion needed for recall.
    """

    name = "Perfect-FTS-Oracle"

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
        candidate_expansion: float = 3.0,  # Fetch 3x budget as candidates
    ):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.candidate_expansion = candidate_expansion
        self.cxl_session: Optional[BaselineCXLSession] = None
        self._future_attention: Optional[np.ndarray] = None
        self.step_count: int = 0

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self._future_attention = None
        self.step_count = 0

    def set_future_attention(self, future_attn: np.ndarray):
        """Provide oracle future attention (called by runner)."""
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

        # Use future attention if available, else current
        if self._future_attention is not None and len(self._future_attention) == num_chunks:
            attn_arr = self._future_attention
        else:
            attn_arr = np.zeros(num_chunks)
            for cid, mass in chunk_attention_masses.items():
                if 0 <= cid < num_chunks:
                    attn_arr[cid] = mass

        # Oracle knows the true ranking
        oracle_ranking = np.argsort(attn_arr)[::-1]
        non_anchor_ranking = [int(c) for c in oracle_ranking if int(c) not in anchor_set]

        # Gold set: the truly useful chunks
        gold = non_anchor_ranking[:budget_chunks]

        # FTS ordering: must fetch a candidate set BEFORE knowing which are useful
        # Even with perfect knowledge, a realistic system needs candidate expansion
        # to achieve high recall (you can't know the exact set without scoring)
        num_candidates = min(
            int(budget_chunks * self.candidate_expansion),
            len(non_anchor_ranking)
        )
        candidates = non_anchor_ranking[:num_candidates]

        # Fetch ALL candidates as full payloads (FTS ordering)
        payload_result = self.cxl_session.cxl.submit_payload_fetch(
            candidates, self.cxl_session._time_ns
        )

        # Perfect scoring: keep only the truly useful ones
        selected = gold[:budget_chunks]

        # Mark invalid traffic (fetched but not used)
        invalid = [c for c in candidates if c not in selected]
        self.cxl_session.cxl.mark_chunks_invalid(invalid)
        self.cxl_session.cxl.mark_chunks_used(selected)

        self.cxl_session.end_step(selected, gold)
        self.cxl_session.advance_step()

        return sorted(set(selected) | anchor_set)

    def get_mean_recovery(self) -> float:
        if self.cxl_session is None or not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.recovery for r in self.cxl_session.step_results]))

    def get_invalid_traffic_ratio(self) -> float:
        if self.cxl_session is None or not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.invalid_traffic_ratio for r in self.cxl_session.step_results]))

    def get_cxl_stats(self):
        if self.cxl_session is None:
            return None
        stats_list = [r.cxl_stats for r in self.cxl_session.step_results]
        if not stats_list:
            return None
        total = type(stats_list[0])()
        for s in stats_list:
            for f in s.__dataclass_fields__:
                setattr(total, f, getattr(total, f) + getattr(s, f))
        return total


class PerfectSBFIOraclePolicy(BaselinePolicy):
    """Perfect Score-Before-Fetch Oracle.

    Has perfect knowledge AND uses SBFI ordering:
      1. Read 64B summaries for candidates
      2. Perfect admission: admit ONLY truly useful chunks
      3. Fetch payloads ONLY for admitted (all useful, zero waste)

    This is the absolute upper bound on what SBFI can achieve.
    The gap between PROSE and this oracle = remaining scorer imperfection.
    The gap between Perfect-FTS and this = pure ordering benefit.
    """

    name = "Perfect-SBFI-Oracle"

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
        candidate_expansion: float = 3.0,
    ):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.candidate_expansion = candidate_expansion
        self.cxl_session: Optional[BaselineCXLSession] = None
        self._future_attention: Optional[np.ndarray] = None
        self.step_count: int = 0

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self._future_attention = None
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

        if self._future_attention is not None and len(self._future_attention) == num_chunks:
            attn_arr = self._future_attention
        else:
            attn_arr = np.zeros(num_chunks)
            for cid, mass in chunk_attention_masses.items():
                if 0 <= cid < num_chunks:
                    attn_arr[cid] = mass

        oracle_ranking = np.argsort(attn_arr)[::-1]
        non_anchor_ranking = [int(c) for c in oracle_ranking if int(c) not in anchor_set]

        gold = non_anchor_ranking[:budget_chunks]

        # SBFI ordering: read summaries for candidate set first
        num_candidates = min(
            int(budget_chunks * self.candidate_expansion),
            len(non_anchor_ranking)
        )
        candidates = non_anchor_ranking[:num_candidates]

        # Step 1: Fetch summaries (cheap: 64B each)
        summary_result = self.cxl_session.cxl.submit_summary_fetch(
            candidates, self.cxl_session._time_ns
        )

        # Step 2: Perfect admission — admit ONLY truly useful chunks
        selected = gold[:budget_chunks]

        # Mark rejected summaries
        rejected = [c for c in candidates if c not in selected]
        self.cxl_session.cxl._step_stats.invalid_summary_bytes += (
            len(rejected) * self.cxl_session.cxl.cfg.summary_size_bytes
        )

        # Step 3: Fetch payloads ONLY for admitted chunks (zero waste)
        payload_result = self.cxl_session.cxl.submit_payload_fetch(
            selected, self.cxl_session._time_ns
        )
        self.cxl_session.cxl.mark_chunks_used(selected)

        self.cxl_session.end_step(selected, gold)
        self.cxl_session.advance_step()

        return sorted(set(selected) | anchor_set)

    def get_mean_recovery(self) -> float:
        if self.cxl_session is None or not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.recovery for r in self.cxl_session.step_results]))

    def get_invalid_traffic_ratio(self) -> float:
        if self.cxl_session is None or not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.invalid_traffic_ratio for r in self.cxl_session.step_results]))

    def get_cxl_stats(self):
        if self.cxl_session is None:
            return None
        stats_list = [r.cxl_stats for r in self.cxl_session.step_results]
        if not stats_list:
            return None
        total = type(stats_list[0])()
        for s in stats_list:
            for f in s.__dataclass_fields__:
                setattr(total, f, getattr(total, f) + getattr(s, f))
        return total

    def get_metadata_overhead_ratio(self) -> float:
        """Ratio of summary bytes to total bytes (cost of SBFI metadata)."""
        if self.cxl_session is None or not self.cxl_session.step_results:
            return 0.0
        total_summary = sum(r.cxl_stats.summary_bytes_fetched for r in self.cxl_session.step_results)
        total_all = sum(r.cxl_stats.total_bytes_fetched for r in self.cxl_session.step_results)
        if total_all == 0:
            return 0.0
        return total_summary / total_all
