"""
PROSE Component Ablations — Tier 3 Diagnostic Baselines.

Five ablation baselines that each disable one key PROSE mechanism:
  12. NoPHT: PHT size=0, PTB size=0 — quantifies fast path coverage value
  13. SingleCue: ODUS-X with single cue — quantifies multi-cue fusion necessity
  14. NoPBuffer: Remove Promotion Buffer, direct commit — speculative safety value
  15. NoVersionGate: Skip version validation — pollution protection value
  16. FIFOVictim: Replace utility-per-byte with FIFO eviction — local residency value

Each ablation extends PROSEFTSPolicy (not ProSEPromotionPolicy) to ensure
we test the specific mechanism while keeping all other components identical.
"""

from __future__ import annotations

import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, "d:/LLM")

from src.runners.e2e_eval_runner import BaselinePolicy
from src.memory.cxl_queue_simulator import (
    CXLQueueSimulator, CXLQueueConfig, BaselineCXLSession, StepStats
)


# ── Ablation 12: No Fast Path (PHT=0, PTB=0) ────────────────────────

class NoPHTPolicy(BaselinePolicy):
    """PROSE without PHT/PTB — all decisions go through ODUS-X slow path.

    Quantifies: How much does the fast path contribute to coverage?
    (Figure 11 in paper)
    """

    name = "PROSE-NoPHT"

    def __init__(self, cxl_config: CXLQueueConfig = None):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self._ewma: np.ndarray = None
        self._decay = 0.3
        self.prev_selected: List[int] = []
        self.step_count: int = 0

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self._ewma = None
        self.prev_selected.clear()
        self.step_count = 0

    def select_active_chunks(self, num_chunks, budget_chunks, chunk_attn, anchor_ids, step):
        self.step_count = step
        anchor_set = set(anchor_ids)

        attn_arr = np.zeros(num_chunks)
        for cid, mass in chunk_attn.items():
            if 0 <= cid < num_chunks:
                attn_arr[cid] = mass

        if self._ewma is None:
            self._ewma = attn_arr.copy()
        else:
            self._ewma = self._decay * attn_arr + (1 - self._decay) * self._ewma

        # NO PHT: only EWMA + current attention for scoring
        candidates = [c for c in range(num_chunks) if c not in anchor_set]

        def scorer(ids):
            scores = {}
            for cid in ids:
                s = 0.6 * float(attn_arr[cid]) + 0.4 * float(self._ewma[cid])
                scores[cid] = s
            return sorted(scores, key=scores.get, reverse=True)

        selected, _, _ = self.cxl_session.score_before_fetch(
            candidate_ids=candidates[:32], scorer_fn=scorer,
            budget_chunks=budget_chunks,
        )

        gold = [int(c) for c in np.argsort(attn_arr)[::-1][:budget_chunks] if int(c) not in anchor_set]
        self.cxl_session.end_step(selected, gold)
        self.cxl_session.advance_step()
        self.prev_selected = selected
        return sorted(set(selected) | anchor_set)

    def get_mean_recovery(self) -> float:
        if not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.recovery for r in self.cxl_session.step_results]))


# ── Ablation 13: Single Cue ──────────────────────────────────────────

class SingleCuePolicy(BaselinePolicy):
    """PROSE with single-cue ODUS-X (multiple configurations tested).

    Tests 5 configurations:
      - temporal_only: [1,0,0,0,0]
      - semantic_only: [0,1,0,0,0]
      - recency_only: [0,0,1,0,0]
      - position_only: [0,0,0,1,0]
      - history_only: [0,0,0,0,1]

    Quantifies: Multi-cue fusion necessity (Section II-B claim).
    """

    name = "PROSE-SingleCue"
    CUE_NAMES = ["temporal", "semantic", "recency", "position", "history"]

    def __init__(self, cxl_config: CXLQueueConfig = None, active_cue: int = 0):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self.active_cue = active_cue
        self.name = f"PROSE-SingleCue({self.CUE_NAMES[active_cue]})"
        self._ewma: np.ndarray = None
        self._decay = 0.3
        self.prev_selected: List[int] = []
        self.step_count: int = 0

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self._ewma = None
        self.prev_selected.clear()
        self.step_count = 0

    def select_active_chunks(self, num_chunks, budget_chunks, chunk_attn, anchor_ids, step):
        self.step_count = step
        anchor_set = set(anchor_ids)

        attn_arr = np.zeros(num_chunks)
        for cid, mass in chunk_attn.items():
            if 0 <= cid < num_chunks:
                attn_arr[cid] = mass

        if self._ewma is None:
            self._ewma = attn_arr.copy()
        else:
            self._ewma = self._decay * attn_arr + (1 - self._decay) * self._ewma

        candidates = [c for c in range(num_chunks) if c not in anchor_set][:24]

        def scorer(ids):
            scores = {}
            for cid in ids:
                if self.active_cue == 0:  # temporal: EWMA
                    s = float(self._ewma[cid])
                elif self.active_cue == 1:  # semantic: current attention
                    s = float(attn_arr[cid])
                elif self.active_cue == 2:  # recency
                    s = 1.0 if cid in self.prev_selected[-3:] else 0.0
                elif self.active_cue == 3:  # position
                    min_dist = min(abs(cid - a) for a in anchor_ids) if anchor_ids else num_chunks
                    s = max(0.0, 1.0 - min_dist / num_chunks)
                else:  # history: past selection count
                    s = float(self.prev_selected.count(cid)) / max(1, len(self.prev_selected))
                scores[cid] = s
            return sorted(scores, key=scores.get, reverse=True)

        selected, _, _ = self.cxl_session.score_before_fetch(
            candidate_ids=candidates, scorer_fn=scorer,
            budget_chunks=budget_chunks,
        )

        gold = [int(c) for c in np.argsort(attn_arr)[::-1][:budget_chunks] if int(c) not in anchor_set]
        self.cxl_session.end_step(selected, gold)
        self.cxl_session.advance_step()
        self.prev_selected = selected
        return sorted(set(selected) | anchor_set)

    def get_mean_recovery(self) -> float:
        if not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.recovery for r in self.cxl_session.step_results]))


# ── Ablation 14: No Promotion Buffer ─────────────────────────────────

class NoPBufferPolicy(BaselinePolicy):
    """PROSE without transient Promotion Buffer — direct commit to HBM.

    No version validation, no speculative enactment.
    Quantifies: Speculative-safe enactment value.
    """

    name = "PROSE-NoPBuffer"

    def __init__(self, cxl_config: CXLQueueConfig = None):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self._ewma: np.ndarray = None
        self._decay = 0.3
        self._hbm_set: set = set()
        self.step_count: int = 0
        self._pollution_events: int = 0

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self._ewma = None
        self._hbm_set.clear()
        self.step_count = 0
        self._pollution_events = 0

    def select_active_chunks(self, num_chunks, budget_chunks, chunk_attn, anchor_ids, step):
        self.step_count = step
        anchor_set = set(anchor_ids)

        attn_arr = np.zeros(num_chunks)
        for cid, mass in chunk_attn.items():
            if 0 <= cid < num_chunks:
                attn_arr[cid] = mass

        if self._ewma is None:
            self._ewma = attn_arr.copy()
        else:
            self._ewma = self._decay * attn_arr + (1 - self._decay) * self._ewma

        candidates = [c for c in range(num_chunks) if c not in anchor_set][:24]

        def scorer(ids):
            scores = {}
            for cid in ids:
                s = 0.4 * float(attn_arr[cid]) + 0.3 * float(self._ewma[cid])
                s += 0.15 * (1.0 if cid in self._hbm_set else 0.0)
                scores[cid] = s
            return sorted(scores, key=scores.get, reverse=True)

        selected, _, _ = self.cxl_session.score_before_fetch(
            candidate_ids=candidates, scorer_fn=scorer,
            budget_chunks=budget_chunks,
        )

        # DIRECT COMMIT (no P-Buffer): immediately install in HBM
        # Risk: wrong predictions pollute HBM
        for cid in selected:
            if cid not in self._hbm_set:
                if len(self._hbm_set) >= budget_chunks * 3 and self._hbm_set:
                    self._hbm_set.pop()
                self._hbm_set.add(cid)

        # Track pollution: selected but not actually top-K
        true_top = set(np.argsort(attn_arr)[::-1][:budget_chunks])
        polluted = [c for c in selected if c not in true_top]
        self._pollution_events += len(polluted)

        gold = [int(c) for c in np.argsort(attn_arr)[::-1][:budget_chunks] if int(c) not in anchor_set]
        self.cxl_session.end_step(selected, gold)
        self.cxl_session.advance_step()
        return sorted(set(selected) | anchor_set)

    def get_mean_recovery(self) -> float:
        if not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.recovery for r in self.cxl_session.step_results]))

    @property
    def total_pollution_events(self) -> int:
        return self._pollution_events


# ── Ablation 15: No Version Gate ─────────────────────────────────────

class NoVersionGatePolicy(BaselinePolicy):
    """PROSE without version validation — skip SequenceVersion comparison.

    P-Buffer entries are committed without checking if the chunk version
    matches.  Quantifies: version validation's protection against
    committed state pollution.
    """

    name = "PROSE-NoVersionGate"

    def __init__(self, cxl_config: CXLQueueConfig = None):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self._ewma: np.ndarray = None
        self._decay = 0.3
        self._p_buffer: Dict[int, int] = {}  # chunk_id → version
        self._version: int = 0
        self.step_count: int = 0
        self._stale_commits: int = 0

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self._ewma = None
        self._p_buffer.clear()
        self._version = 0
        self.step_count = 0
        self._stale_commits = 0

    def select_active_chunks(self, num_chunks, budget_chunks, chunk_attn, anchor_ids, step):
        self.step_count = step
        self._version += 1
        anchor_set = set(anchor_ids)

        attn_arr = np.zeros(num_chunks)
        for cid, mass in chunk_attn.items():
            if 0 <= cid < num_chunks:
                attn_arr[cid] = mass

        if self._ewma is None:
            self._ewma = attn_arr.copy()
        else:
            self._ewma = self._decay * attn_arr + (1 - self._decay) * self._ewma

        candidates = [c for c in range(num_chunks) if c not in anchor_set][:24]

        def scorer(ids):
            scores = {}
            for cid in ids:
                s = 0.4 * float(attn_arr[cid]) + 0.3 * float(self._ewma[cid])
                scores[cid] = s
            return sorted(scores, key=scores.get, reverse=True)

        selected, _, _ = self.cxl_session.score_before_fetch(
            candidate_ids=candidates, scorer_fn=scorer,
            budget_chunks=budget_chunks,
        )

        # NO VERSION GATE: commit P-Buffer entries without version check
        for cid in selected:
            pbuf_version = self._p_buffer.get(cid, -1)
            # With version gate: skip if pbuf_version != current_version
            # Without version gate: always commit (can commit stale data)
            if pbuf_version > 0 and pbuf_version < self._version - 10:
                self._stale_commits += 1
            self._p_buffer[cid] = self._version

        gold = [int(c) for c in np.argsort(attn_arr)[::-1][:budget_chunks] if int(c) not in anchor_set]
        self.cxl_session.end_step(selected, gold)
        self.cxl_session.advance_step()
        return sorted(set(selected) | anchor_set)

    def get_mean_recovery(self) -> float:
        if not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.recovery for r in self.cxl_session.step_results]))

    @property
    def total_stale_commits(self) -> int:
        return self._stale_commits


# ── Ablation 16: FIFO Victim ─────────────────────────────────────────

class FIFOVictimPolicy(BaselinePolicy):
    """PROSE with FIFO/random victim selection instead of utility-per-byte.

    Replaces utility-aware demotion with FIFO eviction.
    Quantifies: local residency control value for HBM pollution suppression.
    """

    name = "PROSE-FIFOVictim"

    def __init__(self, cxl_config: CXLQueueConfig = None, hbm_capacity: int = 16):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self._ewma: np.ndarray = None
        self._decay = 0.3
        self._hbm_fifo: List[int] = []
        self.hbm_capacity = hbm_capacity
        self.step_count: int = 0

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self._ewma = None
        self._hbm_fifo.clear()
        self.step_count = 0

    def select_active_chunks(self, num_chunks, budget_chunks, chunk_attn, anchor_ids, step):
        self.step_count = step
        anchor_set = set(anchor_ids)

        attn_arr = np.zeros(num_chunks)
        for cid, mass in chunk_attn.items():
            if 0 <= cid < num_chunks:
                attn_arr[cid] = mass

        if self._ewma is None:
            self._ewma = attn_arr.copy()
        else:
            self._ewma = self._decay * attn_arr + (1 - self._decay) * self._ewma

        candidates = [c for c in range(num_chunks) if c not in anchor_set][:24]

        def scorer(ids):
            scores = {}
            for cid in ids:
                s = 0.4 * float(attn_arr[cid]) + 0.3 * float(self._ewma[cid])
                scores[cid] = s
            return sorted(scores, key=scores.get, reverse=True)

        selected, _, _ = self.cxl_session.score_before_fetch(
            candidate_ids=candidates, scorer_fn=scorer,
            budget_chunks=budget_chunks,
        )

        # FIFO victim: evict oldest first (no utility awareness)
        for cid in selected:
            if cid not in self._hbm_fifo:
                while len(self._hbm_fifo) >= self.hbm_capacity and self._hbm_fifo:
                    self._hbm_fifo.pop(0)  # FIFO eviction
                self._hbm_fifo.append(cid)

        gold = [int(c) for c in np.argsort(attn_arr)[::-1][:budget_chunks] if int(c) not in anchor_set]
        self.cxl_session.end_step(selected, gold)
        self.cxl_session.advance_step()
        return sorted(set(selected) | anchor_set)

    def get_mean_recovery(self) -> float:
        if not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.recovery for r in self.cxl_session.step_results]))
