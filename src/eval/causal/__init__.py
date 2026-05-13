"""
Causal Verification Framework for ODUS-X Scoring System.

Seven-layer framework that verifies the causal validity, query-conditional
expressivity, information-theoretic sufficiency, adversarial robustness,
architectural generalizability, and online adaptability of the ODUS-X scorer.

Layers:
  1. CEI  — Counterfactual Evidence Intervention
  2. QUDM — Query-Utility Disentanglement Matrix
  3. EB-QPT — Evidence Budget-Query Projection Tradeoff
  4. ITLBP — Information-Theoretic Lower Bound Probing
  5. ACS  — Adversarial Causal Spoofing
  6. CACT — Cross-Architectural Causal Transfer
  7. OCA  — Online Causal Adaptation

Usage:
    from src.eval.causal import CEILayerRunner
    runner = CEILayerRunner(config)
    results = runner.run(evidence_vectors, attention_data)
"""

from src.eval.causal.causal_metrics import (
    CEIMetrics,
    compute_intervention_effect,
    compute_qcdr,
    classify_quadrant,
    compute_budget_quality,
    compute_mutual_information_binned,
    compute_saturation_entropy,
    compute_cvi,
    compute_spoof_success_rate,
    compute_cross_architecture_consistency,
    compute_sbfi_violation_rate,
    compute_cumulative_regret,
    detect_distribution_drift,
    compute_recovery_stability,
)

from src.eval.causal.causal_evaluator import CausalVerificationEvaluator
from src.eval.causal.layer1_cei import (
    EvidenceDecomposer,
    CounterfactualIntervention,
    CEILayerRunner,
)
from src.eval.causal.layer2_qudm import (
    ReuseMetricCalculator,
    QueryUtilityCalculator,
    QuadrantPartitioner,
    QUDMLayerRunner,
)
from src.eval.causal.layer3_ebqpt import (
    EvidenceBudgetRepartitioner,
    QueryableEndpoint,
    EBQPTLayerRunner,
)
from src.eval.causal.layer4_itlbp import (
    VariationalEncoder,
    InformationBottleneck,
    ITLBPLayerRunner,
)
from src.eval.causal.layer5_acs import (
    AdversarialPromptGenerator,
    ScoringRobustnessComparator,
    ACSLayerRunner,
)
from src.eval.causal.layer6_cact import (
    ArchitectureAdapter,
    CACTLayerRunner,
)
from src.eval.causal.layer7_oca import (
    LinUCBBandit,
    ThompsonSamplingBandit,
    EpsilonGreedyScorer,
    DistributionDriftDetector,
    OCALayerRunner,
)

__all__ = [
    # Metrics
    "CEIMetrics",
    "compute_intervention_effect",
    "compute_qcdr",
    "classify_quadrant",
    "compute_budget_quality",
    "compute_mutual_information_binned",
    "compute_saturation_entropy",
    "compute_cvi",
    "compute_spoof_success_rate",
    "compute_cross_architecture_consistency",
    "compute_sbfi_violation_rate",
    "compute_cumulative_regret",
    "detect_distribution_drift",
    "compute_recovery_stability",
    # Evaluator
    "CausalVerificationEvaluator",
    # Layer 1
    "EvidenceDecomposer",
    "CounterfactualIntervention",
    "CEILayerRunner",
    # Layer 2
    "ReuseMetricCalculator",
    "QueryUtilityCalculator",
    "QuadrantPartitioner",
    "QUDMLayerRunner",
    # Layer 3
    "EvidenceBudgetRepartitioner",
    "QueryableEndpoint",
    "EBQPTLayerRunner",
    # Layer 4
    "VariationalEncoder",
    "InformationBottleneck",
    "ITLBPLayerRunner",
    # Layer 5
    "AdversarialPromptGenerator",
    "ScoringRobustnessComparator",
    "ACSLayerRunner",
    # Layer 6
    "ArchitectureAdapter",
    "CACTLayerRunner",
    # Layer 7
    "LinUCBBandit",
    "ThompsonSamplingBandit",
    "EpsilonGreedyScorer",
    "DistributionDriftDetector",
    "OCALayerRunner",
]
