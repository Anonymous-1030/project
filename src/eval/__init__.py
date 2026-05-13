"""Evaluation modules for ProSE-X 2.0"""

from .shared import SharedEvaluator, EvaluationResult
from .accounting import UsefulBytesAccountant, AccountingMode
from .failure_attribution import FailureAttributor
from .neighbor_recall_report import NeighborRecallReporter

from .causal import (
    # Evaluator
    CausalVerificationEvaluator,
    # Metrics
    CEIMetrics,
    compute_intervention_effect,
    compute_qcdr,
    classify_quadrant,
    compute_cvi,
    compute_cross_architecture_consistency,
    compute_sbfi_violation_rate,
    # Layer 1
    EvidenceDecomposer,
    CounterfactualIntervention,
    CEILayerRunner,
    # Layer 2
    ReuseMetricCalculator,
    QueryUtilityCalculator,
    QuadrantPartitioner,
    QUDMLayerRunner,
    # Layer 3
    EvidenceBudgetRepartitioner,
    QueryableEndpoint,
    EBQPTLayerRunner,
    # Layer 4
    VariationalEncoder,
    InformationBottleneck,
    ITLBPLayerRunner,
    # Layer 5
    AdversarialPromptGenerator,
    ScoringRobustnessComparator,
    ACSLayerRunner,
    # Layer 6
    ArchitectureAdapter,
    CACTLayerRunner,
    # Layer 7
    LinUCBBandit,
    ThompsonSamplingBandit,
    EpsilonGreedyScorer,
    DistributionDriftDetector,
    OCALayerRunner,
)

__all__ = [
    "SharedEvaluator",
    "EvaluationResult",
    "UsefulBytesAccountant",
    "AccountingMode",
    "FailureAttributor",
    "NeighborRecallReporter",
    # Causal
    "CausalVerificationEvaluator",
    "CEIMetrics",
    "compute_intervention_effect",
    "compute_qcdr",
    "classify_quadrant",
    "compute_cvi",
    "compute_cross_architecture_consistency",
    "compute_sbfi_violation_rate",
    "EvidenceDecomposer",
    "CounterfactualIntervention",
    "CEILayerRunner",
    "ReuseMetricCalculator",
    "QueryUtilityCalculator",
    "QuadrantPartitioner",
    "QUDMLayerRunner",
    "EvidenceBudgetRepartitioner",
    "QueryableEndpoint",
    "EBQPTLayerRunner",
    "VariationalEncoder",
    "InformationBottleneck",
    "ITLBPLayerRunner",
    "AdversarialPromptGenerator",
    "ScoringRobustnessComparator",
    "ACSLayerRunner",
    "ArchitectureAdapter",
    "CACTLayerRunner",
    "LinUCBBandit",
    "ThompsonSamplingBandit",
    "EpsilonGreedyScorer",
    "DistributionDriftDetector",
    "OCALayerRunner",
]
