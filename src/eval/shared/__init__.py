"""
Shared Evaluator Module for ProSE-X 2.0

All metrics are computed through this module to ensure consistency.
No runner-specific metric logic allowed.
"""

from .evaluator import SharedEvaluator, EvaluationResult
from .evaluator_v2 import SharedEvaluatorV2, EvaluationResult as EvaluationResultV2
from .metrics import (
    compute_conditional_recovery,
    compute_no_miss_rate,
    compute_useful_promote_ratio,
    compute_burst_gain,
    compute_candidate_recall_at_k,
)

__all__ = [
    "SharedEvaluator",
    "EvaluationResult",
    "SharedEvaluatorV2",
    "EvaluationResultV2",
    "compute_conditional_recovery",
    "compute_no_miss_rate",
    "compute_useful_promote_ratio",
    "compute_burst_gain",
    "compute_candidate_recall_at_k",
]
