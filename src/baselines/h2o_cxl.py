"""
H2O-CXL — H2O Heavy-Hitter Retention with Blind CXL Paging.

H2O keeps "heavy hitters" (high cumulative attention) in local memory
and evicts the rest.  This baseline extends H2O to CXL:
  - Local HBM: top-20% heavy hitters (H2O's selection)
  - CXL: remaining 80% blindly paged to CXL
  - On access to CXL page: full DMA fetch (fetch-then-decide)

Shows that retention policy alone (without promotion ordering) cannot
solve the remote memory access problem.
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


class H2OCXLPolicy(BaselinePolicy):
    """H2O heavy-hitter retention with blind CXL paging."""

    name = "H2O-CXL"

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
        hbm_budget_ratio: float = 0.20,
        recent_window_ratio: float = 0.10,
    ):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.hbm_budget_ratio = hbm_budget_ratio
        self.recent_window_ratio = recent_window_ratio
        self.cxl_session: Optional[BaselineCXLSession] = None

        self.cumulative_attention: Dict[int, float] = {}
        self._hbm_set: set = set()
        self.step_count: int = 0

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self.cumulative_attention.clear()
        self._hbm_set.clear()
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

        # Update cumulative attention (H2O's heavy hitter tracking)
        for cid, mass in chunk_attention_masses.items():
            self.cumulative_attention[cid] = (
                self.cumulative_attention.get(cid, 0.0) + mass
            )

        # H2O selection: heavy hitters + recent window
        hh_budget = max(1, int(budget_chunks * 0.7))
        recent_budget = budget_chunks - hh_budget

        sorted_hh = sorted(self.cumulative_attention.items(),
                          key=lambda x: x[1], reverse=True)

        selected = set(anchor_ids)

        # Heavy hitters
        for cid, _ in sorted_hh:
            if cid not in anchor_set and len(selected) - len(anchor_set) < hh_budget:
                selected.add(cid)

        # Recent window (last few accessed)
        if chunk_attention_masses:
            sorted_recent = sorted(chunk_attention_masses.items(),
                                  key=lambda x: x[1], reverse=True)
            for cid, _ in sorted_recent:
                if cid not in anchor_set and cid not in selected:
                    if len(selected) - len(anchor_set) >= budget_chunks:
                        break
                    selected.add(cid)

        # ── CXL paging: fetch any selected chunk not in HBM ──
        # H2O-CXL blindly pages evicted chunks to CXL.
        # On re-access, a full chunk DMA is triggered (no summary pre-filter).
        chunks_to_fetch = [c for c in selected
                          if c not in self._hbm_set and c not in anchor_set]

        if chunks_to_fetch:
            self.cxl_session.cxl.submit_payload_fetch(chunks_to_fetch, 0)

            # Fetch-then-decide: ALL fetched as full payloads
            # Invalid traffic = any chunk eventually not in final selection
            self.cxl_session.cxl.mark_chunks_used(chunks_to_fetch)

        # Update HBM residency
        for c in chunks_to_fetch:
            hbm_limit = max(1, int(num_chunks * self.hbm_budget_ratio))
            if len(self._hbm_set) >= hbm_limit and self._hbm_set:
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
