"""
PROSE-SBFI (Score-Before-Fetch-Initiative) — Full PROSE System Baseline.

This is THE core contribution policy. It implements the complete PROSE pipeline:
  - MQR-ULF candidate generation (5-queue recall)
  - ODUS-X multi-cue scoring (5 signal cues)
  - PHT/PTB fast-path prediction
  - Promotion Buffer with version-gate validation
  - Burst expansion for spatial locality
  - Sticky TTL for temporal persistence
  - Utility-per-byte eviction

CRITICALLY, it uses Score-Before-Fetch-Initiative (SBFI) ordering:
  PROSE-SBFI:  Summaries from CXL (64B each) → Score locally → Fetch only validated
  PROSE-FTS:   Fetch full chunks from CXL (64KB each) → Score locally → Keep top-K

The ONLY difference from PROSE-FTS is the ordering of fetch vs. score.
This isolates the value of SBFI from the quality of the ranker/predictor.

Expected result (HPCA claim):
  - 0% invalid payload traffic (all fetched chunks are used)
  - Lower CXL queue utilization ρ vs. PROSE-FTS (only fetch validated subset)
  - Same or better Recovery@K (same ranker, but without bandwidth-induced throttling)
  - Significantly lower latency than PROSE-FTS (no bulk DMA of rejected chunks)
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, "d:/LLM")

from src.runners.e2e_eval_runner import BaselinePolicy
from src.memory.cxl_queue_simulator import CXLQueueSimulator, CXLQueueConfig, BaselineCXLSession


class PROSEPolicy(BaselinePolicy):
    """PROSE with Score-Before-Fetch-Initiative ordering.

    Same predictor/ranker/buffer as PROSE-FTS, but:
      - Step 1: Fetch 64B metadata summaries for ALL candidates from CXL
      - Step 2: Score summaries locally using ODUS-X
      - Step 3: Fetch full 64KB payloads ONLY for top-K validated chunks
      - Step 4: Apply PHT update, burst expansion, sticky TTL

    This is the SBFI (Score-Before-Fetch-Initiative) that defines PROSE.
    """

    name = "PROSE"

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
        enable_pht: bool = True,
        enable_burst: bool = True,
        enable_sticky: bool = True,
        lookahead_depth: int = 3,
    ):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.cxl_session: Optional[BaselineCXLSession] = None
        self.enable_pht = enable_pht
        self.enable_burst = enable_burst
        self.enable_sticky = enable_sticky
        self.lookahead_depth = lookahead_depth

        # PROSE state
        self.pht_ema: Dict[int, float] = {}
        self.pht_anchor: Dict[int, bool] = {}
        self.prev_selected: List[int] = []
        self.step_count: int = 0
        self._window_buffer: List[np.ndarray] = []
        self._sticky_ttl: Dict[int, int] = {}
        self._ewma: Optional[np.ndarray] = None
        self._decay = 0.3

    def reset(self):
        """Reset state for a new trace."""
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self.pht_ema.clear()
        self.pht_anchor.clear()
        self.prev_selected.clear()
        self.step_count = 0
        self._window_buffer.clear()
        self._sticky_ttl.clear()
        self._ewma = None

    def select_active_chunks(
        self,
        num_chunks: int,
        budget_chunks: int,
        chunk_attention_masses: Dict[int, float],
        anchor_ids: List[int],
        step: int,
    ) -> List[int]:
        """Score-before-fetch: fetch 64B summaries, score, then fetch only validated."""
        self.step_count = step

        if self.cxl_session is None:
            self.cxl_session = BaselineCXLSession(self.cxl_config)

        anchor_set = set(anchor_ids)

        # Convert attention masses to array for scoring
        attn_arr = np.zeros(num_chunks)
        for cid, mass in chunk_attention_masses.items():
            if isinstance(cid, int) and 0 <= cid < num_chunks:
                attn_arr[cid] = mass

        # Update EWMA (same as PROSE-FTS)
        if self._ewma is None:
            self._ewma = attn_arr.copy()
        else:
            self._ewma = self._decay * attn_arr + (1 - self._decay) * self._ewma

        # Maintain window buffer for burst expansion
        self._window_buffer.append(attn_arr.copy())
        if len(self._window_buffer) > 8:
            self._window_buffer.pop(0)

        # ── CANDIDATE GENERATION (MQR-ULF: 5-queue recall) ──
        candidates = self._generate_candidates(num_chunks, attn_arr, anchor_ids)

        # ── SCORE-BEFORE-FETCH: fetch 64B summaries, score, fetch only validated ──
        # This is the KEY difference from PROSE-FTS
        selected, summary_result, payload_result = self.cxl_session.score_before_fetch(
            candidate_ids=candidates,
            scorer_fn=lambda ids: self._score_chunks(ids, attn_arr, anchor_ids),
            budget_chunks=budget_chunks,
        )

        # ── POST-FETCH: Apply PHT update, sticky TTL, burst ──
        if self.enable_pht:
            self._update_pht(selected, attn_arr)

        if self.enable_burst:
            selected = self._apply_burst(selected, num_chunks, anchor_set)

        if self.enable_sticky:
            selected = self._apply_sticky(selected, anchor_set)

        # Finalize step
        self.cxl_session.end_step(selected, self._get_gold(attn_arr, budget_chunks, anchor_ids))
        self.cxl_session.advance_step()
        self.prev_selected = list(selected)

        return sorted(set(selected) | anchor_set)

    # ── Internal: Candidate Generation (MQR-ULF 5-queue recall) ──

    def _generate_candidates(
        self, num_chunks: int, attn_arr: np.ndarray, anchor_ids: List[int]
    ) -> List[int]:
        """Generate candidate set mirroring MQR-ULF's 5 queues."""
        anchor_set = set(anchor_ids)
        candidates = set()

        # Queue 1: Top-K by EWMA (anchor-neighbor equivalent)
        if self._ewma is not None:
            ewma_order = np.argsort(self._ewma)[::-1]
            for cid in ewma_order[:max(5, num_chunks // 4)]:
                if int(cid) not in anchor_set:
                    candidates.add(int(cid))

        # Queue 2: Top-K by current attention (lexical overlap equivalent)
        top_attn = np.argsort(attn_arr)[::-1][:5]
        for cid in top_attn:
            if int(cid) not in anchor_set:
                candidates.add(int(cid))

        # Queue 3: Recently accessed (structural/recency queue)
        for cid in self.prev_selected[-5:]:
            if 0 <= cid < num_chunks and cid not in anchor_set:
                candidates.add(cid)

        # Queue 4: High PHT-EMA chunks (historical success queue)
        if self.enable_pht and self.pht_ema:
            pht_sorted = sorted(self.pht_ema.items(), key=lambda x: x[1], reverse=True)
            for cid, _ in pht_sorted[:5]:
                if 0 <= cid < num_chunks and cid not in anchor_set:
                    candidates.add(cid)

        # Queue 5: Lookahead speculative (neighbors of current top chunks)
        if self._ewma is not None:
            top_idx = int(np.argmax(self._ewma))
            for offset in range(-self.lookahead_depth, self.lookahead_depth + 1):
                neighbor = top_idx + offset
                if 0 <= neighbor < num_chunks and neighbor not in anchor_set:
                    candidates.add(neighbor)

        # Fallback: if too few candidates, add sequential neighbors
        if len(candidates) < 8:
            for cid in range(num_chunks):
                if cid not in anchor_set and len(candidates) < 16:
                    candidates.add(cid)

        return sorted(candidates)

    # ── Internal: ODUS-X Multi-Cue Scoring ──

    def _score_chunks(
        self, candidate_ids: List[int], attn_arr: np.ndarray, anchor_ids: List[int]
    ) -> List[int]:
        """ODUS-X equivalent: multi-cue scoring on 64B summaries.

        In PROSE-SBFI this runs on 64B metadata summaries BEFORE payload DMA.
        In PROSE-FTS this runs on full chunks AFTER payload DMA (same logic).
        The scoring quality is identical — only the TIMING differs.
        """
        scores = {}
        anchor_set = set(anchor_ids)

        for cid in candidate_ids:
            if cid in anchor_set or cid < 0 or cid >= len(attn_arr):
                continue

            score = 0.0

            # Cue 1: Current attention mass (40% weight)
            score += 0.40 * float(attn_arr[cid])

            # Cue 2: EWMA (30% weight)
            if self._ewma is not None and cid < len(self._ewma):
                score += 0.30 * float(self._ewma[cid])

            # Cue 3: PHT history (15% weight)
            pht_val = self.pht_ema.get(cid, 0.0)
            score += 0.15 * pht_val

            # Cue 4: Recency (10% weight)
            if cid in self.prev_selected:
                recency_idx = self.prev_selected[::-1].index(cid)
                score += 0.10 * max(0.0, 1.0 - recency_idx / 10.0)

            # Cue 5: Position proximity to anchors (5% weight)
            n_chunks = len(attn_arr)
            min_dist = min(abs(cid - a) for a in anchor_ids) if anchor_ids else n_chunks
            score += 0.05 * max(0.0, 1.0 - min_dist / n_chunks)

            scores[cid] = score

        return sorted(scores, key=scores.get, reverse=True)

    # ── Internal: PHT, Burst, Sticky ──

    def _update_pht(self, selected: List[int], attn_arr: np.ndarray):
        """Update PHT EMA (mirrors PHT engine behavior)."""
        alpha = 0.15
        for cid in selected:
            if cid < len(attn_arr):
                self.pht_ema[cid] = (
                    alpha * float(attn_arr[cid]) +
                    (1 - alpha) * self.pht_ema.get(cid, 0.0)
                )

    def _apply_burst(self, selected: List[int], num_chunks: int,
                     anchor_set: set) -> List[int]:
        """Burst expansion: add neighbors of selected chunks."""
        expanded = set(selected)
        for cid in selected:
            for offset in [-1, 1]:
                neighbor = cid + offset
                if 0 <= neighbor < num_chunks and neighbor not in anchor_set:
                    expanded.add(neighbor)
        return sorted(expanded)

    def _apply_sticky(self, selected: List[int], anchor_set: set) -> List[int]:
        """Sticky TTL: keep recently promoted chunks alive."""
        result = set(selected)

        # Decrement TTL for all sticky chunks
        expired = []
        for cid, ttl in list(self._sticky_ttl.items()):
            self._sticky_ttl[cid] = ttl - 1
            if self._sticky_ttl[cid] <= 0:
                expired.append(cid)
            elif cid not in anchor_set:
                result.add(cid)

        for cid in expired:
            del self._sticky_ttl[cid]

        # Set TTL for newly selected (non-anchor) chunks
        for cid in selected:
            if cid not in anchor_set:
                self._sticky_ttl[cid] = 4

        return sorted(result)

    @staticmethod
    def _get_gold(attn_arr: np.ndarray, budget_chunks: int,
                  anchor_ids: List[int]) -> List[int]:
        """Oracle gold set: top-K by true attention."""
        anchor_set = set(anchor_ids)
        ranked = np.argsort(attn_arr)[::-1]
        gold = []
        for cid in ranked:
            if int(cid) not in anchor_set and len(gold) < budget_chunks:
                gold.append(int(cid))
        return gold

    # ── Statistics accessors (for runner) ──

    def get_cxl_stats(self):
        """Return accumulated CXL statistics from the session."""
        if self.cxl_session is None:
            return None
        stats_list = [r.cxl_stats for r in self.cxl_session.step_results]
        if not stats_list:
            return None
        total = type(stats_list[0])()
        for s in stats_list:
            for field in s.__dataclass_fields__:
                setattr(total, field, getattr(total, field) + getattr(s, field))
        return total

    def get_invalid_traffic_ratio(self) -> float:
        """Mean invalid traffic ratio across all steps."""
        if self.cxl_session is None or not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.invalid_traffic_ratio for r in self.cxl_session.step_results]))

    def get_mean_recovery(self) -> float:
        """Mean gold recovery across all steps."""
        if self.cxl_session is None or not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.recovery for r in self.cxl_session.step_results]))
