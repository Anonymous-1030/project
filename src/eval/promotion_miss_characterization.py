"""Promotion Miss Characterization Study for HPCA.

This is the MOTIVATION FIGURE generator (Figure 1 / Figure 2).

Core experiment: For each decode step, classify every accuracy loss into
one of four categories:
  1. Retention Miss  — the chunk was evicted entirely (not in tail)
  2. Promotion Miss  — the chunk IS in tail but was NOT promoted
  3. Scoring Miss    — the chunk was a ULF candidate but scored too low
  4. Budget Miss     — the chunk scored high enough but budget ran out

The key insight we want to demonstrate:
  "At 10% KV budget, 60%+ of accuracy loss comes from promotion miss,
   not retention miss."

This module provides:
  - PromotionMissProfiler: runs oracle vs. policy comparison per step
  - MissBreakdownAggregator: aggregates across steps into stacked bar data
  - CrossMethodComparison: runs H2O/SnapKV/StreamingLLM/ProSE side by side
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
from enum import Enum


class MissCategory(str, Enum):
    """Categories of accuracy loss."""
    RETENTION_MISS = "retention_miss"
    PROMOTION_MISS = "promotion_miss"
    SCORING_MISS = "scoring_miss"
    BUDGET_MISS = "budget_miss"
    RECOVERED = "recovered"  # Successfully promoted and used


@dataclass
class StepMissProfile:
    """Miss profile for a single decode step."""
    step: int
    # Oracle info
    oracle_useful_chunks: Set[str]       # Chunks with attention > threshold in full-KV
    oracle_total_attention_mass: float
    # Policy info
    retained_chunks: Set[str]            # Chunks still in tail (not evicted)
    ulf_candidates: Set[str]             # Chunks selected by ULF
    scored_above_threshold: Set[str]     # Chunks that passed scoring
    promoted_chunks: Set[str]            # Chunks actually promoted
    # Classification
    retention_misses: Set[str]
    promotion_misses: Set[str]
    scoring_misses: Set[str]
    budget_misses: Set[str]
    recovered: Set[str]
    # Attention mass breakdown
    retention_miss_attention: float = 0.0
    promotion_miss_attention: float = 0.0
    scoring_miss_attention: float = 0.0
    budget_miss_attention: float = 0.0
    recovered_attention: float = 0.0


@dataclass
class MissBreakdown:
    """Aggregated miss breakdown across all steps."""
    method_name: str
    budget_ratio: float
    total_steps: int
    # Count-based breakdown
    total_retention_misses: int = 0
    total_promotion_misses: int = 0
    total_scoring_misses: int = 0
    total_budget_misses: int = 0
    total_recovered: int = 0
    # Attention-mass-based breakdown (more meaningful)
    retention_miss_mass_pct: float = 0.0
    promotion_miss_mass_pct: float = 0.0
    scoring_miss_mass_pct: float = 0.0
    budget_miss_mass_pct: float = 0.0
    recovered_mass_pct: float = 0.0
    # Per-step data for plotting
    per_step_profiles: List[Dict[str, float]] = field(default_factory=list)


class PromotionMissProfiler:
    """Profiles promotion misses at each decode step.

    Requires oracle (full-KV) attention weights to classify misses.
    This is an OFFLINE analysis tool — never used at inference time.
    """

    def __init__(self, attention_threshold: float = 0.01):
        """
        Args:
            attention_threshold: Minimum attention mass to consider a chunk
                                 "useful" in the oracle run.
        """
        self.attention_threshold = attention_threshold

    def profile_step(
        self,
        step: int,
        # Oracle data (from full-KV run)
        oracle_attention_per_chunk: Dict[str, float],
        # System state
        all_chunk_ids: Set[str],          # All chunks that exist
        retained_chunk_ids: Set[str],     # Chunks in tail (not evicted)
        ulf_candidate_ids: Set[str],      # ULF output
        scored_above_ids: Set[str],       # Passed scoring threshold
        promoted_ids: Set[str],           # Actually promoted
    ) -> StepMissProfile:
        """Classify every oracle-useful chunk into a miss category."""

        # Oracle useful chunks
        oracle_useful = {
            cid for cid, mass in oracle_attention_per_chunk.items()
            if mass >= self.attention_threshold
        }
        total_oracle_mass = sum(oracle_attention_per_chunk.values())

        # Classification (waterfall: earliest failure stage)
        retention_misses: Set[str] = set()
        promotion_misses: Set[str] = set()
        scoring_misses: Set[str] = set()
        budget_misses: Set[str] = set()
        recovered: Set[str] = set()

        ret_mass = 0.0
        promo_mass = 0.0
        score_mass = 0.0
        budget_mass = 0.0
        rec_mass = 0.0

        for cid in oracle_useful:
            mass = oracle_attention_per_chunk.get(cid, 0.0)

            if cid not in retained_chunk_ids:
                # Chunk was evicted entirely
                retention_misses.add(cid)
                ret_mass += mass
            elif cid not in ulf_candidate_ids:
                # Chunk is in tail but ULF didn't select it
                promotion_misses.add(cid)
                promo_mass += mass
            elif cid not in scored_above_ids:
                # ULF selected it but scorer ranked it too low
                scoring_misses.add(cid)
                score_mass += mass
            elif cid not in promoted_ids:
                # Scored high but budget ran out
                budget_misses.add(cid)
                budget_mass += mass
            else:
                # Successfully promoted
                recovered.add(cid)
                rec_mass += mass

        return StepMissProfile(
            step=step,
            oracle_useful_chunks=oracle_useful,
            oracle_total_attention_mass=total_oracle_mass,
            retained_chunks=retained_chunk_ids,
            ulf_candidates=ulf_candidate_ids,
            scored_above_threshold=scored_above_ids,
            promoted_chunks=promoted_ids,
            retention_misses=retention_misses,
            promotion_misses=promotion_misses,
            scoring_misses=scoring_misses,
            budget_misses=budget_misses,
            recovered=recovered,
            retention_miss_attention=ret_mass,
            promotion_miss_attention=promo_mass,
            scoring_miss_attention=score_mass,
            budget_miss_attention=budget_mass,
            recovered_attention=rec_mass,
        )


class MissBreakdownAggregator:
    """Aggregates per-step miss profiles into paper-ready data."""

    def aggregate(
        self,
        profiles: List[StepMissProfile],
        method_name: str,
        budget_ratio: float,
    ) -> MissBreakdown:
        """Aggregate step profiles into a single breakdown."""
        breakdown = MissBreakdown(
            method_name=method_name,
            budget_ratio=budget_ratio,
            total_steps=len(profiles),
        )

        total_miss_mass = 0.0
        total_rec_mass = 0.0

        for p in profiles:
            breakdown.total_retention_misses += len(p.retention_misses)
            breakdown.total_promotion_misses += len(p.promotion_misses)
            breakdown.total_scoring_misses += len(p.scoring_misses)
            breakdown.total_budget_misses += len(p.budget_misses)
            breakdown.total_recovered += len(p.recovered)

            step_total = (
                p.retention_miss_attention + p.promotion_miss_attention
                + p.scoring_miss_attention + p.budget_miss_attention
                + p.recovered_attention
            )
            total_miss_mass += (
                p.retention_miss_attention + p.promotion_miss_attention
                + p.scoring_miss_attention + p.budget_miss_attention
            )
            total_rec_mass += p.recovered_attention

            if step_total > 0:
                breakdown.per_step_profiles.append({
                    "step": p.step,
                    "retention_pct": p.retention_miss_attention / step_total * 100,
                    "promotion_pct": p.promotion_miss_attention / step_total * 100,
                    "scoring_pct": p.scoring_miss_attention / step_total * 100,
                    "budget_pct": p.budget_miss_attention / step_total * 100,
                    "recovered_pct": p.recovered_attention / step_total * 100,
                })

        grand_total = total_miss_mass + total_rec_mass
        if grand_total > 0:
            # Compute percentages of TOTAL attention (including recovered)
            breakdown.recovered_mass_pct = total_rec_mass / grand_total * 100
            # Percentages of MISSED attention only
            if total_miss_mass > 0:
                # Recompute from accumulated per-step masses
                ret_total = sum(p.retention_miss_attention for p in profiles)
                promo_total = sum(p.promotion_miss_attention for p in profiles)
                score_total = sum(p.scoring_miss_attention for p in profiles)
                budget_total = sum(p.budget_miss_attention for p in profiles)

                breakdown.retention_miss_mass_pct = ret_total / grand_total * 100
                breakdown.promotion_miss_mass_pct = promo_total / grand_total * 100
                breakdown.scoring_miss_mass_pct = score_total / grand_total * 100
                breakdown.budget_miss_mass_pct = budget_total / grand_total * 100

        return breakdown

    def to_stacked_bar_data(
        self,
        breakdowns: List[MissBreakdown],
    ) -> Dict[str, Any]:
        """Convert multiple breakdowns to stacked bar chart data.

        Output format suitable for matplotlib / pgfplots.
        """
        methods = [b.method_name for b in breakdowns]
        categories = [
            "retention_miss", "promotion_miss", "scoring_miss",
            "budget_miss", "recovered",
        ]

        data: Dict[str, List[float]] = {cat: [] for cat in categories}
        for b in breakdowns:
            data["retention_miss"].append(b.retention_miss_mass_pct)
            data["promotion_miss"].append(b.promotion_miss_mass_pct)
            data["scoring_miss"].append(b.scoring_miss_mass_pct)
            data["budget_miss"].append(b.budget_miss_mass_pct)
            data["recovered"].append(b.recovered_mass_pct)

        return {
            "methods": methods,
            "categories": categories,
            "data": data,
            "ylabel": "Attention Mass (%)",
            "title": "Miss Category Breakdown by Method",
        }


class CrossMethodComparison:
    """Run miss characterization across multiple methods.

    Simulates H2O / SnapKV / StreamingLLM / ProSE promotion behavior
    and classifies misses for each.
    """

    def __init__(self, profiler: Optional[PromotionMissProfiler] = None):
        self.profiler = profiler or PromotionMissProfiler()
        self.aggregator = MissBreakdownAggregator()

    def simulate_method_behavior(
        self,
        method_name: str,
        budget_ratio: float,
        num_chunks: int,
        num_steps: int,
        oracle_attention_traces: List[Dict[str, float]],
        anchor_ratio: float = 0.1,
        seed: int = 42,
    ) -> MissBreakdown:
        """Simulate a method's promotion behavior and profile misses.

        This uses synthetic but realistic attention patterns to demonstrate
        the miss breakdown. For the real paper, replace with actual model
        attention weights.
        """
        import numpy as np
        rng = np.random.RandomState(seed)

        all_chunk_ids = {f"chunk_{i}" for i in range(num_chunks)}
        num_anchors = max(1, int(num_chunks * anchor_ratio))
        anchor_ids = {f"chunk_{i}" for i in range(num_anchors)}
        tail_ids = all_chunk_ids - anchor_ids

        budget_chunks = max(1, int(num_chunks * budget_ratio))

        profiles: List[StepMissProfile] = []

        for step in range(min(num_steps, len(oracle_attention_traces))):
            oracle_attn = oracle_attention_traces[step]

            # Simulate method-specific behavior
            if method_name == "H2O":
                retained, ulf_cands, scored, promoted = self._simulate_h2o(
                    tail_ids, oracle_attn, budget_chunks, step, rng
                )
            elif method_name == "SnapKV":
                retained, ulf_cands, scored, promoted = self._simulate_snapkv(
                    tail_ids, oracle_attn, budget_chunks, step, rng
                )
            elif method_name == "StreamingLLM":
                retained, ulf_cands, scored, promoted = self._simulate_streaming(
                    tail_ids, num_chunks, budget_chunks, step
                )
            elif method_name == "ProSE":
                retained, ulf_cands, scored, promoted = self._simulate_prose(
                    tail_ids, oracle_attn, budget_chunks, step, rng
                )
            else:
                retained, ulf_cands, scored, promoted = tail_ids, set(), set(), set()

            profile = self.profiler.profile_step(
                step=step,
                oracle_attention_per_chunk=oracle_attn,
                all_chunk_ids=all_chunk_ids,
                retained_chunk_ids=retained | anchor_ids,
                ulf_candidate_ids=ulf_cands,
                scored_above_ids=scored,
                promoted_ids=promoted | anchor_ids,
            )
            profiles.append(profile)

        return self.aggregator.aggregate(profiles, method_name, budget_ratio)

    # ------------------------------------------------------------------ #
    # Method-specific simulation
    # ------------------------------------------------------------------ #

    def _simulate_h2o(self, tail_ids, oracle_attn, budget, step, rng):
        """H2O: keeps top-k by cumulative attention. No promotion mechanism."""
        retained = set(tail_ids)
        # H2O selects top-k by cumulative attention (it has access to attention)
        sorted_chunks = sorted(
            tail_ids, key=lambda c: oracle_attn.get(c, 0.0), reverse=True
        )
        promoted = set(sorted_chunks[:budget])
        # H2O has no ULF/scoring pipeline — it directly selects
        return retained, promoted, promoted, promoted

    def _simulate_snapkv(self, tail_ids, oracle_attn, budget, step, rng):
        """SnapKV: observation window based. Good retention, weak promotion."""
        retained = set(tail_ids)
        # SnapKV uses observation window — approximates attention
        noisy_attn = {
            c: oracle_attn.get(c, 0.0) + rng.normal(0, 0.05)
            for c in tail_ids
        }
        sorted_chunks = sorted(
            tail_ids, key=lambda c: noisy_attn.get(c, 0.0), reverse=True
        )
        candidates = set(sorted_chunks[:budget * 3])
        scored = set(sorted_chunks[:budget * 2])
        promoted = set(sorted_chunks[:budget])
        return retained, candidates, scored, promoted

    def _simulate_streaming(self, tail_ids, num_chunks, budget, step):
        """StreamingLLM: sink + sliding window. High retention miss."""
        # StreamingLLM only keeps recent tokens — many tail chunks evicted
        window_size = budget
        recent_ids = {
            f"chunk_{i}" for i in range(max(0, num_chunks - window_size), num_chunks)
        }
        retained = recent_ids & tail_ids
        # No promotion mechanism
        return retained, retained, retained, retained

    def _simulate_prose(self, tail_ids, oracle_attn, budget, step, rng):
        """ProSE: multi-queue recall + utility scoring + budget scheduling."""
        retained = set(tail_ids)  # ProSE retains all in tail (compressed)

        # MQR-ULF: cast wide net (3x budget candidates)
        # Mix of attention-based and structural signals
        attn_sorted = sorted(
            tail_ids, key=lambda c: oracle_attn.get(c, 0.0), reverse=True
        )
        # Queue 1: top by attention proxy
        q1 = set(attn_sorted[:budget])
        # Queue 2: random structural
        q2_list = list(tail_ids)
        rng.shuffle(q2_list)
        q2 = set(q2_list[:budget // 2])
        # Queue 3: neighbors of high-attention chunks
        q3 = set()
        for c in list(q1)[:3]:
            idx = int(c.split("_")[1])
            for delta in [-1, 0, 1]:
                neighbor = f"chunk_{idx + delta}"
                if neighbor in tail_ids:
                    q3.add(neighbor)

        candidates = q1 | q2 | q3

        # ODUS scoring: noisy utility prediction
        scored_chunks = sorted(
            candidates,
            key=lambda c: oracle_attn.get(c, 0.0) + rng.normal(0, 0.02),
            reverse=True,
        )
        scored = set(scored_chunks[:budget * 2])

        # EABS: exploit + explore
        exploit = set(scored_chunks[:int(budget * 0.8)])
        explore_pool = [c for c in scored_chunks[int(budget * 0.8):] if c not in exploit]
        if explore_pool:
            n_explore = min(budget - len(exploit), len(explore_pool))
            explore_idx = rng.choice(len(explore_pool), size=max(0, n_explore), replace=False)
            explore = {explore_pool[i] for i in explore_idx}
        else:
            explore = set()
        promoted = exploit | explore

        return retained, candidates, scored, promoted

    # ------------------------------------------------------------------ #
    # Generate synthetic oracle traces
    # ------------------------------------------------------------------ #

    @staticmethod
    def generate_synthetic_traces(
        num_chunks: int = 100,
        num_steps: int = 50,
        sparsity: float = 0.9,
        locality_strength: float = 0.7,
        seed: int = 42,
    ) -> List[Dict[str, float]]:
        """Generate synthetic but realistic attention traces.

        Models:
        - Power-law attention distribution (few chunks get most mass)
        - Temporal locality (recently attended chunks likely re-attended)
        - Spatial locality (neighboring chunks co-attended)
        """
        import numpy as np
        rng = np.random.RandomState(seed)

        traces: List[Dict[str, float]] = []
        prev_hot: Set[int] = set()

        for step in range(num_steps):
            attn: Dict[str, float] = {}
            num_useful = max(1, int(num_chunks * (1 - sparsity)))

            # Temporal locality: some chunks from previous step
            hot_from_prev = set()
            if prev_hot:
                for c in prev_hot:
                    if rng.random() < locality_strength:
                        hot_from_prev.add(c)

            # New hot chunks
            remaining = num_useful - len(hot_from_prev)
            if remaining > 0:
                available = [i for i in range(num_chunks) if i not in hot_from_prev]
                if available:
                    # Power-law selection
                    weights = np.array([1.0 / (i + 1) for i in range(len(available))])
                    weights /= weights.sum()
                    new_hot = rng.choice(
                        available,
                        size=min(remaining, len(available)),
                        replace=False,
                        p=weights[:len(available)] / weights[:len(available)].sum(),
                    )
                    hot_from_prev.update(new_hot)

            # Assign attention masses (Zipf-like)
            hot_list = sorted(hot_from_prev)
            for rank, cid in enumerate(hot_list):
                mass = 1.0 / (rank + 1)
                # Spatial locality: neighbors get some mass too
                attn[f"chunk_{cid}"] = mass
                for delta in [-1, 1]:
                    neighbor = cid + delta
                    if 0 <= neighbor < num_chunks:
                        key = f"chunk_{neighbor}"
                        attn[key] = attn.get(key, 0.0) + mass * 0.2

            # Normalize
            total = sum(attn.values()) or 1.0
            attn = {k: v / total for k, v in attn.items()}

            traces.append(attn)
            prev_hot = hot_from_prev

        return traces

    # ------------------------------------------------------------------ #
    # Full characterization study
    # ------------------------------------------------------------------ #

    def run_full_study(
        self,
        num_chunks: int = 100,
        num_steps: int = 50,
        budget_ratios: Optional[List[float]] = None,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the complete promotion miss characterization study.

        Returns data for Figure 1 (stacked bar) and Figure 2 (budget sweep).
        """
        if budget_ratios is None:
            budget_ratios = [0.05, 0.10, 0.20, 0.40]
        if methods is None:
            methods = ["H2O", "SnapKV", "StreamingLLM", "ProSE"]

        traces = self.generate_synthetic_traces(
            num_chunks=num_chunks, num_steps=num_steps
        )

        # Figure 1: Fixed budget (10%), compare methods
        fig1_breakdowns = []
        for method in methods:
            bd = self.simulate_method_behavior(
                method_name=method,
                budget_ratio=0.10,
                num_chunks=num_chunks,
                num_steps=num_steps,
                oracle_attention_traces=traces,
            )
            fig1_breakdowns.append(bd)

        fig1_data = self.aggregator.to_stacked_bar_data(fig1_breakdowns)

        # Figure 2: Fixed method (ProSE), sweep budgets
        fig2_breakdowns = []
        for ratio in budget_ratios:
            bd = self.simulate_method_behavior(
                method_name="ProSE",
                budget_ratio=ratio,
                num_chunks=num_chunks,
                num_steps=num_steps,
                oracle_attention_traces=traces,
            )
            fig2_breakdowns.append(bd)

        fig2_data = {
            "budget_ratios": budget_ratios,
            "retention_miss_pct": [b.retention_miss_mass_pct for b in fig2_breakdowns],
            "promotion_miss_pct": [b.promotion_miss_mass_pct for b in fig2_breakdowns],
            "scoring_miss_pct": [b.scoring_miss_mass_pct for b in fig2_breakdowns],
            "budget_miss_pct": [b.budget_miss_mass_pct for b in fig2_breakdowns],
            "recovered_pct": [b.recovered_mass_pct for b in fig2_breakdowns],
        }

        # Summary statistics
        summary = {}
        for bd in fig1_breakdowns:
            total_miss = (
                bd.retention_miss_mass_pct + bd.promotion_miss_mass_pct
                + bd.scoring_miss_mass_pct + bd.budget_miss_mass_pct
            )
            summary[bd.method_name] = {
                "total_miss_pct": total_miss,
                "promotion_miss_share_of_total_miss": (
                    bd.promotion_miss_mass_pct / max(total_miss, 1e-9) * 100
                ),
                "retention_miss_share_of_total_miss": (
                    bd.retention_miss_mass_pct / max(total_miss, 1e-9) * 100
                ),
                "recovered_pct": bd.recovered_mass_pct,
            }

        return {
            "figure_1_stacked_bar": fig1_data,
            "figure_2_budget_sweep": fig2_data,
            "summary": summary,
            "config": {
                "num_chunks": num_chunks,
                "num_steps": num_steps,
                "budget_ratios": budget_ratios,
                "methods": methods,
            },
        }
