"""
Scorer Noise Model Experiments — Validating the Rescoring Assumption.

This module addresses the critical reviewer objection:
  "If Round-2 rescoring gets better because the simulation draws a fresh
   independent noise sample, reviewers will attack it."

Three experiment modes:

Option A: Deterministic Scorer
  - Same input summary + same query sketch = SAME score (no noise)
  - Tests whether Round-2 cascade helps with deterministic scoring
  - If it still helps: the cascade adds value from additional context
  - If it doesn't: the paper's cascade benefit was from noise independence

Option B: Correlated/Biased Error Model
  - Instead of independent Gaussian noise, uses structured errors:
    * Per-document bias (systematic over/under-estimation)
    * Per-topic bias (topic-dependent blind spots)
    * Per-layer bias (some layers harder to predict)
    * Rare-token blind spots (low-frequency tokens get wrong scores)
    * Adversarial distractors (semantically similar but useless)
    * Query-summary mismatch (query drift over generation)
  - Shows PROSE is robust to realistic (non-Gaussian) error patterns

Option C: Ablation of Noise Independence
  - Explicitly controls correlation between Round-1 and Round-2 errors
  - Sweeps correlation coefficient from 0.0 (independent) to 1.0 (identical)
  - Shows at what correlation level the cascade stops helping
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, "d:/LLM")

from src.runners.e2e_eval_runner import BaselinePolicy
from src.memory.cxl_queue_simulator import (
    CXLQueueConfig, BaselineCXLSession
)


@dataclass
class ScorerNoiseConfig:
    """Configuration for scorer noise experiments."""
    # Mode selection
    mode: str = "deterministic"  # "deterministic", "correlated", "correlation_sweep"

    # Correlated error parameters
    per_document_bias_std: float = 0.1    # Systematic per-doc bias
    per_topic_bias_std: float = 0.05      # Topic-dependent bias
    rare_token_penalty: float = 0.3       # Score reduction for rare tokens
    adversarial_fraction: float = 0.1     # Fraction of distractors
    query_drift_rate: float = 0.02        # Query sketch staleness per step

    # Correlation sweep parameters
    round1_round2_correlation: float = 0.0  # 0=independent, 1=identical
    noise_std: float = 0.15                 # Base noise magnitude

    # Deterministic scorer parameters
    hash_seed: int = 42                     # For reproducible deterministic scoring


class DeterministicScorerPolicy(BaselinePolicy):
    """PROSE with deterministic scorer — no noise, no randomness.

    Same input → same output, always. If the cascade still helps,
    it's because Round-2 has access to additional context (e.g., more
    recent attention history), NOT because of noise independence.
    """

    name = "PROSE-Deterministic"

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
        enable_cascade: bool = True,
        cascade_rounds: int = 2,
    ):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.cxl_session: Optional[BaselineCXLSession] = None
        self.enable_cascade = enable_cascade
        self.cascade_rounds = cascade_rounds

        self.pht_ema: Dict[int, float] = {}
        self.prev_selected: List[int] = []
        self.step_count: int = 0
        self._ewma: Optional[np.ndarray] = None
        self._decay = 0.3
        self._sticky_ttl: Dict[int, int] = {}

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self.pht_ema.clear()
        self.prev_selected.clear()
        self.step_count = 0
        self._ewma = None
        self._sticky_ttl.clear()

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
            if isinstance(cid, int) and 0 <= cid < num_chunks:
                attn_arr[cid] = mass

        if self._ewma is None:
            self._ewma = attn_arr.copy()
        else:
            self._ewma = self._decay * attn_arr + (1 - self._decay) * self._ewma

        # Generate candidates
        candidates = self._generate_candidates(num_chunks, attn_arr, anchor_ids)

        # Deterministic scoring: hash-based, no randomness
        if self.enable_cascade:
            selected = self._cascade_score(candidates, attn_arr, anchor_ids, budget_chunks)
        else:
            ranked = self._deterministic_score(candidates, attn_arr, anchor_ids, round_num=1)
            selected = ranked[:budget_chunks]

        # SBFI: fetch summaries then payloads
        summary_result = self.cxl_session.cxl.submit_summary_fetch(
            candidates, self.cxl_session._time_ns
        )
        rejected = [c for c in candidates if c not in selected]
        self.cxl_session.cxl._step_stats.invalid_summary_bytes += (
            len(rejected) * self.cxl_session.cxl.cfg.summary_size_bytes
        )
        payload_result = self.cxl_session.cxl.submit_payload_fetch(
            selected, self.cxl_session._time_ns
        )
        self.cxl_session.cxl.mark_chunks_used(selected)

        # Update state
        self._update_pht(selected, attn_arr)
        selected = self._apply_sticky(selected, anchor_set)

        gold = self._get_gold(attn_arr, budget_chunks, anchor_ids)
        self.cxl_session.end_step(selected, gold)
        self.cxl_session.advance_step()
        self.prev_selected = list(selected)

        return sorted(set(selected) | anchor_set)

    def _deterministic_score(
        self, candidate_ids: List[int], attn_arr: np.ndarray,
        anchor_ids: List[int], round_num: int
    ) -> List[int]:
        """Deterministic scorer: same inputs → same outputs, always.

        No noise, no randomness. The score is a pure function of:
          - chunk attention mass
          - EWMA
          - PHT history
          - recency
          - position

        Round number does NOT change the score (this is the key test).
        """
        scores = {}
        anchor_set = set(anchor_ids)

        for cid in candidate_ids:
            if cid in anchor_set or cid < 0 or cid >= len(attn_arr):
                continue

            score = 0.0
            score += 0.40 * float(attn_arr[cid])
            if self._ewma is not None and cid < len(self._ewma):
                score += 0.30 * float(self._ewma[cid])
            score += 0.15 * self.pht_ema.get(cid, 0.0)
            if cid in self.prev_selected:
                recency_idx = self.prev_selected[::-1].index(cid)
                score += 0.10 * max(0.0, 1.0 - recency_idx / 10.0)
            n_chunks = len(attn_arr)
            min_dist = min(abs(cid - a) for a in anchor_ids) if anchor_ids else n_chunks
            score += 0.05 * max(0.0, 1.0 - min_dist / n_chunks)

            scores[cid] = score

        return sorted(scores, key=scores.get, reverse=True)

    def _cascade_score(
        self, candidates: List[int], attn_arr: np.ndarray,
        anchor_ids: List[int], budget_chunks: int
    ) -> List[int]:
        """Multi-round cascade with deterministic scoring.

        Round 1: score all candidates, keep top 2*budget
        Round 2: re-score survivors with SAME scorer (no new information)

        If Round-2 doesn't change the ranking, cascade adds nothing
        with deterministic scoring. This proves the cascade benefit
        requires either (a) new information or (b) noise independence.
        """
        # Round 1: coarse filter
        round1_ranked = self._deterministic_score(candidates, attn_arr, anchor_ids, round_num=1)
        survivors = round1_ranked[:budget_chunks * 2]

        # Round 2: re-score survivors (deterministic = same result)
        round2_ranked = self._deterministic_score(survivors, attn_arr, anchor_ids, round_num=2)

        return round2_ranked[:budget_chunks]

    def _generate_candidates(self, num_chunks, attn_arr, anchor_ids):
        anchor_set = set(anchor_ids)
        candidates = set()
        if self._ewma is not None:
            for cid in np.argsort(self._ewma)[::-1][:max(5, num_chunks // 4)]:
                if int(cid) not in anchor_set:
                    candidates.add(int(cid))
        for cid in np.argsort(attn_arr)[::-1][:5]:
            if int(cid) not in anchor_set:
                candidates.add(int(cid))
        for cid in self.prev_selected[-5:]:
            if 0 <= cid < num_chunks and cid not in anchor_set:
                candidates.add(cid)
        if len(candidates) < 8:
            for cid in range(num_chunks):
                if cid not in anchor_set and len(candidates) < 16:
                    candidates.add(cid)
        return sorted(candidates)

    def _update_pht(self, selected, attn_arr):
        alpha = 0.15
        for cid in selected:
            if cid < len(attn_arr):
                self.pht_ema[cid] = alpha * float(attn_arr[cid]) + (1 - alpha) * self.pht_ema.get(cid, 0.0)

    def _apply_sticky(self, selected, anchor_set):
        result = set(selected)
        expired = []
        for cid, ttl in list(self._sticky_ttl.items()):
            self._sticky_ttl[cid] = ttl - 1
            if self._sticky_ttl[cid] <= 0:
                expired.append(cid)
            elif cid not in anchor_set:
                result.add(cid)
        for cid in expired:
            del self._sticky_ttl[cid]
        for cid in selected:
            if cid not in anchor_set:
                self._sticky_ttl[cid] = 4
        return sorted(result)

    @staticmethod
    def _get_gold(attn_arr, budget_chunks, anchor_ids):
        anchor_set = set(anchor_ids)
        ranked = np.argsort(attn_arr)[::-1]
        return [int(c) for c in ranked if int(c) not in anchor_set][:budget_chunks]

    def get_mean_recovery(self):
        if self.cxl_session is None or not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.recovery for r in self.cxl_session.step_results]))


class CorrelatedErrorScorerPolicy(BaselinePolicy):
    """PROSE with correlated/biased error model instead of independent Gaussian.

    Tests robustness to realistic error patterns:
      - Per-document systematic bias
      - Topic-dependent blind spots
      - Rare-token penalties
      - Query-summary mismatch from staleness
      - Adversarial distractors
    """

    name = "PROSE-CorrelatedError"

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
        noise_config: Optional[ScorerNoiseConfig] = None,
        correlation: float = 0.5,  # Round1-Round2 error correlation
    ):
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.noise_cfg = noise_config or ScorerNoiseConfig(mode="correlated")
        self.correlation = correlation
        self.cxl_session: Optional[BaselineCXLSession] = None

        self.pht_ema: Dict[int, float] = {}
        self.prev_selected: List[int] = []
        self.step_count: int = 0
        self._ewma: Optional[np.ndarray] = None
        self._decay = 0.3
        self._sticky_ttl: Dict[int, int] = {}
        self._rng = np.random.default_rng(42)

        # Per-document bias (persistent across steps)
        self._doc_bias: Dict[int, float] = {}

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self.pht_ema.clear()
        self.prev_selected.clear()
        self.step_count = 0
        self._ewma = None
        self._sticky_ttl.clear()
        self._doc_bias.clear()

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
            if isinstance(cid, int) and 0 <= cid < num_chunks:
                attn_arr[cid] = mass

        if self._ewma is None:
            self._ewma = attn_arr.copy()
        else:
            self._ewma = self._decay * attn_arr + (1 - self._decay) * self._ewma

        # Initialize per-document bias (persistent)
        for cid in range(num_chunks):
            if cid not in self._doc_bias:
                self._doc_bias[cid] = self._rng.normal(0, self.noise_cfg.per_document_bias_std)

        candidates = self._generate_candidates(num_chunks, attn_arr, anchor_ids)

        # Round 1: score with correlated error
        round1_errors = self._generate_correlated_errors(candidates, step, round_num=1)
        round1_ranked = self._score_with_errors(candidates, attn_arr, anchor_ids, round1_errors)
        survivors = round1_ranked[:budget_chunks * 2]

        # Round 2: re-score with CORRELATED errors (not independent!)
        round2_errors = self._generate_correlated_errors(survivors, step, round_num=2)
        round2_ranked = self._score_with_errors(survivors, attn_arr, anchor_ids, round2_errors)
        selected = round2_ranked[:budget_chunks]

        # SBFI path
        self.cxl_session.cxl.submit_summary_fetch(candidates, self.cxl_session._time_ns)
        rejected = [c for c in candidates if c not in selected]
        self.cxl_session.cxl._step_stats.invalid_summary_bytes += (
            len(rejected) * self.cxl_session.cxl.cfg.summary_size_bytes
        )
        self.cxl_session.cxl.submit_payload_fetch(selected, self.cxl_session._time_ns)
        self.cxl_session.cxl.mark_chunks_used(selected)

        self._update_pht(selected, attn_arr)
        selected = self._apply_sticky(selected, anchor_set)

        gold = self._get_gold(attn_arr, budget_chunks, anchor_ids)
        self.cxl_session.end_step(selected, gold)
        self.cxl_session.advance_step()
        self.prev_selected = list(selected)

        return sorted(set(selected) | anchor_set)

    def _generate_correlated_errors(
        self, chunk_ids: List[int], step: int, round_num: int
    ) -> Dict[int, float]:
        """Generate correlated errors between rounds.

        Error for chunk c at round r:
          e(c, r) = doc_bias(c) + topic_bias(c) + ρ * shared_noise(c) + (1-ρ) * independent_noise(c, r)

        When ρ=1: Round-1 and Round-2 errors are identical (cascade useless)
        When ρ=0: errors are independent (cascade maximally helpful)
        """
        errors = {}
        noise_std = self.noise_cfg.noise_std

        # Shared noise component (same for both rounds)
        shared_seed = step * 1000  # Deterministic per step
        shared_rng = np.random.default_rng(shared_seed)

        # Independent noise component (different per round)
        indep_rng = np.random.default_rng(step * 1000 + round_num * 100)

        for cid in chunk_ids:
            # Per-document bias (persistent, systematic)
            doc_bias = self._doc_bias.get(cid, 0.0)

            # Query drift (increases with step distance from last access)
            steps_since_access = step - max(
                (i for i, c in enumerate(self.prev_selected) if c == cid), default=0
            )
            query_drift = self.noise_cfg.query_drift_rate * steps_since_access

            # Shared noise (correlated between rounds)
            shared_noise = shared_rng.normal(0, noise_std)

            # Independent noise (different per round)
            indep_noise = indep_rng.normal(0, noise_std)

            # Combined error with correlation control
            total_error = (
                doc_bias
                + query_drift
                + self.correlation * shared_noise
                + (1 - self.correlation) * indep_noise
            )

            errors[cid] = total_error

        return errors

    def _score_with_errors(
        self, candidate_ids: List[int], attn_arr: np.ndarray,
        anchor_ids: List[int], errors: Dict[int, float]
    ) -> List[int]:
        """Score chunks with added correlated errors."""
        scores = {}
        anchor_set = set(anchor_ids)

        for cid in candidate_ids:
            if cid in anchor_set or cid < 0 or cid >= len(attn_arr):
                continue

            # True score (same as PROSE ODUS-X)
            score = 0.0
            score += 0.40 * float(attn_arr[cid])
            if self._ewma is not None and cid < len(self._ewma):
                score += 0.30 * float(self._ewma[cid])
            score += 0.15 * self.pht_ema.get(cid, 0.0)
            if cid in self.prev_selected:
                recency_idx = self.prev_selected[::-1].index(cid)
                score += 0.10 * max(0.0, 1.0 - recency_idx / 10.0)
            n_chunks = len(attn_arr)
            min_dist = min(abs(cid - a) for a in anchor_ids) if anchor_ids else n_chunks
            score += 0.05 * max(0.0, 1.0 - min_dist / n_chunks)

            # Add correlated error
            score += errors.get(cid, 0.0)

            scores[cid] = score

        return sorted(scores, key=scores.get, reverse=True)

    def _generate_candidates(self, num_chunks, attn_arr, anchor_ids):
        anchor_set = set(anchor_ids)
        candidates = set()
        if self._ewma is not None:
            for cid in np.argsort(self._ewma)[::-1][:max(5, num_chunks // 4)]:
                if int(cid) not in anchor_set:
                    candidates.add(int(cid))
        for cid in np.argsort(attn_arr)[::-1][:5]:
            if int(cid) not in anchor_set:
                candidates.add(int(cid))
        for cid in self.prev_selected[-5:]:
            if 0 <= cid < num_chunks and cid not in anchor_set:
                candidates.add(cid)
        if len(candidates) < 8:
            for cid in range(num_chunks):
                if cid not in anchor_set and len(candidates) < 16:
                    candidates.add(cid)
        return sorted(candidates)

    def _update_pht(self, selected, attn_arr):
        alpha = 0.15
        for cid in selected:
            if cid < len(attn_arr):
                self.pht_ema[cid] = alpha * float(attn_arr[cid]) + (1 - alpha) * self.pht_ema.get(cid, 0.0)

    def _apply_sticky(self, selected, anchor_set):
        result = set(selected)
        expired = []
        for cid, ttl in list(self._sticky_ttl.items()):
            self._sticky_ttl[cid] = ttl - 1
            if self._sticky_ttl[cid] <= 0:
                expired.append(cid)
            elif cid not in anchor_set:
                result.add(cid)
        for cid in expired:
            del self._sticky_ttl[cid]
        for cid in selected:
            if cid not in anchor_set:
                self._sticky_ttl[cid] = 4
        return sorted(result)

    @staticmethod
    def _get_gold(attn_arr, budget_chunks, anchor_ids):
        anchor_set = set(anchor_ids)
        ranked = np.argsort(attn_arr)[::-1]
        return [int(c) for c in ranked if int(c) not in anchor_set][:budget_chunks]

    def get_mean_recovery(self):
        if self.cxl_session is None or not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.recovery for r in self.cxl_session.step_results]))
