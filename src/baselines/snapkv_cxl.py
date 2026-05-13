"""
SnapKV + CXL Paging — SnapKV Local Retention with Blind CXL Offloading.

SnapKV selects KV pairs for local retention based on an observation window.
Remaining KV pairs are blindly offloaded to CXL.  This baseline models:
  - Local HBM: SnapKV-selected pairs (top-20%)
  - CXL: remaining 80% blind-offloaded
  - On access: full DMA fetch, no metadata pre-screening

Contrasts metadata-gated admission (PROSE) vs. local retention + blind paging.
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


class SnapKVCXLPolicy(BaselinePolicy):
    """SnapKV local retention + blind CXL paging."""

    name = "SnapKV-CXL"

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
        observation_window: int = 5,
        hbm_budget_ratio: float = 0.20,
    ):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.observation_window = observation_window
        self.hbm_budget_ratio = hbm_budget_ratio
        self.cxl_session: Optional[BaselineCXLSession] = None

        self._observation_buffer: List[Dict[int, float]] = []
        self._hbm_set: set = set()
        self.step_count: int = 0

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self._observation_buffer.clear()
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

        # SnapKV observation window
        self._observation_buffer.append(dict(chunk_attention_masses))
        if len(self._observation_buffer) > self.observation_window:
            self._observation_buffer.pop(0)

        # Aggregate observation scores
        obs_scores: Dict[int, float] = {}
        for obs in self._observation_buffer:
            for cid, mass in obs.items():
                obs_scores[cid] = obs_scores.get(cid, 0.0) + mass

        # Select top-K by observation score
        sorted_obs = sorted(obs_scores.items(), key=lambda x: x[1], reverse=True)

        selected = set(anchor_ids)
        for cid, _ in sorted_obs:
            if cid not in anchor_set and len(selected) - len(anchor_set) < budget_chunks:
                selected.add(cid)

        # ── CXL blind paging ──
        # Any chunk not in HBM but in the selected set triggers a full DMA
        chunks_to_fetch = [c for c in selected
                          if c not in self._hbm_set and c not in anchor_set]

        if chunks_to_fetch:
            self.cxl_session.cxl.submit_payload_fetch(chunks_to_fetch, 0)
            self.cxl_session.cxl.mark_chunks_used(chunks_to_fetch)

        # Update HBM with blind offload: evicted chunks go to CXL
        for c in chunks_to_fetch:
            hbm_limit = max(1, int(num_chunks * self.hbm_budget_ratio))
            if len(self._hbm_set) >= hbm_limit and self._hbm_set:
                self._hbm_set.pop()
            self._hbm_set.add(c)

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
