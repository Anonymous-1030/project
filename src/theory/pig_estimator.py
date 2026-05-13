"""
Practical PIG (Promotion Information Gain) estimation engine.

Provides runtime-usable PIG computation from attention distributions,
and evaluation tools to compare promotion decisions against PIG-optimal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from src.theory.promotion_information_theory import (
    PIGResult,
    PromotionInformationGain,
)


@dataclass
class PIGOptimalResult:
    """Result of PIG-optimal promotion set computation."""

    selected_indices: List[int]
    selected_pig_values: List[float]
    total_pig: float
    budget_used: int
    budget_total: int


@dataclass
class PIGRegretResult:
    """Per-step PIG regret measurement."""

    step: int
    pig_optimal: float
    pig_achieved: float
    pig_regret: float          # optimal - achieved
    relative_efficiency: float  # achieved / optimal


class PIGEstimator:
    """Runtime PIG estimation from attention distributions.

    Uses the attention-mass approximation (Theorem 1, Corollary 1):
      PIG(c, W) ≈ a(c) · (1 - R(c, W))
    """

    def __init__(self, vocab_size: int = 32000):
        self._pig = PromotionInformationGain(vocab_size=vocab_size)

    def estimate_pig_from_attention(
        self,
        attention_masses: np.ndarray,
        chunk_index: int,
        working_set_indices: List[int],
        chunk_embeddings: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate PIG for a single chunk. Returns scalar PIG value."""
        result = self._pig.compute_pig(
            attention_masses, chunk_index, working_set_indices, chunk_embeddings
        )
        return result.pig_value

    def rank_by_pig(
        self,
        attention_masses: np.ndarray,
        candidate_indices: List[int],
        working_set_indices: List[int],
        chunk_embeddings: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, float]]:
        """Rank candidates by PIG value (descending).

        Returns list of (chunk_index, pig_value) sorted by PIG.
        """
        results = self._pig.compute_pig_set(
            attention_masses, candidate_indices,
            working_set_indices, chunk_embeddings,
        )
        ranked = [
            (int(r.chunk_id.split("_")[1]), r.pig_value) for r in results
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def compute_pig_optimal_set(
        self,
        attention_masses: np.ndarray,
        candidate_indices: List[int],
        working_set_indices: List[int],
        budget_chunks: int,
        chunk_embeddings: Optional[np.ndarray] = None,
    ) -> PIGOptimalResult:
        """Compute the greedy PIG-optimal promotion set.

        Greedy algorithm: iteratively select chunk with highest marginal PIG.
        By Theorem 3, this achieves (1-1/e) of optimal.
        """
        selected = []
        selected_pigs = []
        total_pig = 0.0
        current_ws = list(working_set_indices)
        remaining = set(candidate_indices)

        for _ in range(min(budget_chunks, len(candidate_indices))):
            best_idx = -1
            best_pig = -1.0

            for idx in remaining:
                pig = self._pig.compute_pig(
                    attention_masses, idx, current_ws, chunk_embeddings
                ).pig_value
                if pig > best_pig:
                    best_pig = pig
                    best_idx = idx

            if best_idx < 0 or best_pig <= 0:
                break

            selected.append(best_idx)
            selected_pigs.append(best_pig)
            total_pig += best_pig
            current_ws.append(best_idx)
            remaining.discard(best_idx)

        return PIGOptimalResult(
            selected_indices=selected,
            selected_pig_values=selected_pigs,
            total_pig=total_pig,
            budget_used=len(selected),
            budget_total=budget_chunks,
        )


class PIGEvaluator:
    """Evaluates on decisions against PIG-optimal.

    Computes per-step and aggregate PIG regret metrics.
    """

    def __init__(self, vocab_size: int = 32000):
        self._estimator = PIGEstimator(vocab_size=vocab_size)
        self._step_results: List[PIGRegretResult] = []

    def evaluate_step(
        self,
        step: int,
        attention_masses: np.ndarray,
        actual_promoted_indices: List[int],
        candidate_indices: List[int],
        working_set_indices: List[int],
        budget_chunks: int,
        chunk_embeddings: Optional[np.ndarray] = None,
    ) -> PIGRegretResult:
        """Evaluate a single step's promotion decision against PIG-optimal."""
        # Compute PIG-optimal
        optimal = self._estimator.compute_pig_optimal_set(
            attention_masses, candidate_indices,
            working_set_indices, budget_chunks, chunk_embeddings,
        )

        # Compute PIG achieved by actual promotion
        pig_achieved = 0.0
        current_ws = list(working_set_indices)
        for idx in actual_promoted_indices:
            pig = self._estimator.estimate_pig_from_attention(
                attention_masses, idx, current_ws, chunk_embeddings,
            )
            pig_achieved += pig
            current_ws.append(idx)

        regret = optimal.total_pig - pig_achieved
        efficiency = pig_achieved / max(1e-10, optimal.total_pig)

        result = PIGRegretResult(
            step=step,
            pig_optimal=optimal.total_pig,
            pig_achieved=pig_achieved,
            pig_regret=max(0.0, regret),
            relative_efficiency=min(1.0, efficiency),
        )
        self._step_results.append(result)
        return result

    def aggregate(self) -> Dict[str, Any]:
        """Compute aggregate PIG metrics across all evaluated steps."""
        if not self._step_results:
            return {
                "num_steps": 0,
                "total_pig_optimal": 0.0,
                "total_pig_achieved": 0.0,
                "total_regret": 0.0,
                "avg_efficiency": 0.0,
            }

        total_optimal = sum(r.pig_optimal for r in self._step_results)
        total_achieved = sum(r.pig_achieved for r in self._step_results)
        total_regret = sum(r.pig_regret for r in self._step_results)
        avg_efficiency = np.mean(
            [r.relative_efficiency for r in self._step_results]
        )

        return {
            "num_steps": len(self._step_results),
            "total_pig_optimal": total_optimal,
            "total_pig_achieved": total_achieved,
            "total_regret": total_regret,
            "cumulative_efficiency": total_achieved / max(1e-10, total_optimal),
            "avg_step_efficiency": float(avg_efficiency),
            "min_step_efficiency": min(
                r.relative_efficiency for r in self._step_results
            ),
            "max_step_regret": max(
                r.pig_regret for r in self._step_results
            ),
        }

    def reset(self) -> None:
        self._step_results.clear()
