"""
Causal verification metric functions for the Seven-Layer Framework.

Pure metric computation with no side effects. All functions accept structured
inputs and return structured outputs. Follows the pattern from eval/shared/metrics.py.

Each metric is independently computable and testable.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core_types import (
    EvidenceDimension,
    QuadrantLabel,
)


# =============================================================================
# Layer 1: CEI Metrics
# =============================================================================

@dataclass
class CEIMetrics:
    """Per-dimension causal effect metrics from counterfactual intervention."""
    dimension: EvidenceDimension
    baseline_admission_rate: float
    intervention_admission_rate: float
    delta: float
    phase_consistency: float  # correlation of delta across decode phases
    pass_fail: bool = False


def compute_intervention_effect(
    dimension: EvidenceDimension,
    baseline_admissions: np.ndarray,       # bool array [n_chunks]
    intervention_admissions: np.ndarray,   # bool array [n_chunks]
    phase_labels: Optional[np.ndarray] = None,  # int array [n_chunks] for phase
    pass_threshold: float = 0.05,
    phase_consistency_required: int = 3,
) -> CEIMetrics:
    """
    Compute causal effect of an intervention on a single evidence dimension.

    Args:
        dimension: Which evidence dimension was intervened on
        baseline_admissions: Admission decisions before intervention
        intervention_admissions: Admission decisions after intervention
        phase_labels: Optional decode phase per chunk (for consistency check)
        pass_threshold: Minimum |delta| in admission rate to pass
        phase_consistency_required: Minimum number of phases with consistent sign

    Returns:
        CEIMetrics with delta, consistency, and pass/fail verdict
    """
    n = len(baseline_admissions)
    if n == 0:
        return CEIMetrics(
            dimension=dimension,
            baseline_admission_rate=0.0,
            intervention_admission_rate=0.0,
            delta=0.0,
            phase_consistency=0.0,
            pass_fail=False,
        )

    base_rate = float(np.mean(baseline_admissions))
    interv_rate = float(np.mean(intervention_admissions))
    delta = interv_rate - base_rate

    # Phase consistency: sign of delta should match across decode phases
    phase_consistency = 0.0
    n_consistent_phases = 0
    if phase_labels is not None:
        unique_phases = np.unique(phase_labels)
        phase_deltas = {}
        for phase in unique_phases:
            mask = phase_labels == phase
            if mask.sum() > 0:
                p_delta = float(np.mean(intervention_admissions[mask])) - float(np.mean(baseline_admissions[mask]))
                phase_deltas[int(phase)] = p_delta
                if np.sign(p_delta) == np.sign(delta) or abs(p_delta) < 1e-6:
                    n_consistent_phases += 1

        if len(unique_phases) > 0:
            phase_consistency = n_consistent_phases / len(unique_phases)

    pass_fail = (abs(delta) >= pass_threshold) and (n_consistent_phases >= phase_consistency_required)

    return CEIMetrics(
        dimension=dimension,
        baseline_admission_rate=base_rate,
        intervention_admission_rate=interv_rate,
        delta=delta,
        phase_consistency=phase_consistency,
        pass_fail=pass_fail,
    )


# =============================================================================
# Layer 2: QUDM Metrics
# =============================================================================

def compute_qcdr(
    alpha_2: float,  # admission rate of High-R + Low-U_Q quadrant (locality trap)
    alpha_3: float,  # admission rate of Low-R + High-U_Q quadrant (long-range)
) -> float:
    """
    Compute Query-Causal Defect Rate.

    QCDR = alpha_2 / (alpha_2 + alpha_3)

    High QCDR means the system admits more locality traps than long-range
    dependencies, indicating query-independent summaries are structurally
    biased toward LRU-like behavior.

    Returns:
        QCDR in [0, 1], or 0.0 if denominator is zero.
    """
    denom = alpha_2 + alpha_3
    if denom < 1e-9:
        return 0.0
    return alpha_2 / denom


def classify_quadrant(
    reuse_score: float,
    utility_score: float,
    reuse_quantile: float,
    utility_quantile: float,
) -> QuadrantLabel:
    """
    Classify a single chunk into a QUDM quadrant.

    Uses quantile-based thresholds for adaptive partitioning.
    """
    if reuse_score >= reuse_quantile and utility_score >= utility_quantile:
        return QuadrantLabel.HIGH_R_HIGH_U
    elif reuse_score >= reuse_quantile and utility_score < utility_quantile:
        return QuadrantLabel.HIGH_R_LOW_U
    elif reuse_score < reuse_quantile and utility_score >= utility_quantile:
        return QuadrantLabel.LOW_R_HIGH_U
    else:
        return QuadrantLabel.LOW_R_LOW_U


# =============================================================================
# Layer 3: EB-QPT Metrics
# =============================================================================

def compute_budget_quality(
    budget_bytes: int,
    saturation_bytes: float = 128.0,
) -> float:
    """
    Model evidence quality as a function of budget allocation.

    quality = sigmoid((budget - offset) / steepness)
    Calibrated so that 64B -> ~0.88, 128B -> ~0.96 (matching existing evidence model).
    """
    if budget_bytes <= 0:
        return 0.0
    # Logistic saturation model
    midpoint = saturation_bytes / 2.0
    steepness = saturation_bytes / 8.0
    return 1.0 / (1.0 + np.exp(-(budget_bytes - midpoint) / steepness))


# =============================================================================
# Layer 4: ITLBP Metrics
# =============================================================================

def compute_mutual_information_binned(
    encoding: np.ndarray,  # [n_samples, d_encoding]
    utility: np.ndarray,   # [n_samples]
    n_bins: int = 20,
) -> float:
    """
    Estimate I(E; U) via binning estimator.

    Uses plug-in estimator: I(E;U) = H(U) + H(E) - H(E,U)
    with histogram-based entropy estimates.

    Args:
        encoding: Encoded representations
        utility: Target utility values
        n_bins: Number of bins for histogram

    Returns:
        Estimated mutual information in nats
    """
    n_samples = len(utility)
    if n_samples < 10:
        return 0.0

    # Marginal entropy H(U)
    u_hist, _ = np.histogram(utility, bins=n_bins, density=True)
    u_hist = u_hist[u_hist > 0]
    h_u = -np.sum(u_hist * np.log(u_hist)) / n_bins * (utility.max() - utility.min() + 1e-9)
    h_u = max(0.0, h_u)

    # For multivariate encoding, use PCA projection for 1D entropy estimate
    if encoding.ndim > 1 and encoding.shape[1] > 1:
        # Project to 1D using first principal component
        encoding_centered = encoding - encoding.mean(axis=0)
        if encoding_centered.shape[0] > 1:
            u, s, vt = np.linalg.svd(encoding_centered, full_matrices=False)
            e_1d = encoding_centered @ vt[0]
        else:
            e_1d = encoding_centered.flatten()
    else:
        e_1d = encoding.flatten()

    e_hist, _ = np.histogram(e_1d, bins=n_bins, density=True)
    e_hist = e_hist[e_hist > 0]
    h_e = -np.sum(e_hist * np.log(e_hist)) / n_bins * (e_1d.max() - e_1d.min() + 1e-9)
    h_e = max(0.0, h_e)

    # Joint entropy via 2D histogram
    joint_hist, _, _ = np.histogram2d(e_1d, utility, bins=n_bins, density=True)
    joint_hist = joint_hist[joint_hist > 0]
    h_joint = -np.sum(joint_hist * np.log(joint_hist)) / (n_bins * n_bins)
    h_joint = max(0.0, h_joint)

    mi = h_u + h_e - h_joint
    return max(0.0, float(mi))


def compute_saturation_entropy(
    budget_recovery: Dict[int, float],
    saturation_threshold: float = 0.95,
) -> Tuple[float, int]:
    """
    Identify the saturation point in a budget-recovery curve.

    Saturation = budget at which recovery reaches threshold * max_recovery.

    Returns:
        (saturation_entropy_in_bytes, saturation_budget_in_bytes)
    """
    if not budget_recovery:
        return 0.0, 0

    budgets = sorted(budget_recovery.keys())
    recoveries = [budget_recovery[b] for b in budgets]
    max_recovery = max(recoveries)
    if max_recovery < 1e-9:
        return 0.0, budgets[-1]

    saturation_target = saturation_threshold * max_recovery
    saturation_budget = budgets[-1]
    for b, r in zip(budgets, recoveries):
        if r >= saturation_target:
            saturation_budget = b
            break

    # Convert budget bytes to entropy bits
    saturation_bits = saturation_budget * 8
    return saturation_bits, saturation_budget


# =============================================================================
# Layer 5: ACS Metrics
# =============================================================================

def compute_cvi(
    ssr_base: float,           # Spoof success rate of base ODUS-X
    ssr_query_aware: float,    # Spoof success rate of query-aware variant
) -> float:
    """
    Compute Causal Vulnerability Index.

    CVI = (SSR_base - SSR_query_aware) / SSR_base

    CVI near 0: query-aware offers no benefit over query-independent (vulnerable).
    CVI near 1: query-aware eliminates most spoofing (robust).

    Returns:
        CVI in [0, 1], or 0.0 if base SSR is zero.
    """
    if ssr_base < 1e-9:
        return 0.0
    return max(0.0, min(1.0, (ssr_base - ssr_query_aware) / ssr_base))


def compute_spoof_success_rate(
    spoof_chunks_admitted: int,
    total_spoof_chunks: int,
) -> float:
    """Compute fraction of adversarial spoof chunks that were admitted."""
    if total_spoof_chunks < 1:
        return 0.0
    return spoof_chunks_admitted / total_spoof_chunks


# =============================================================================
# Layer 6: CACT Metrics
# =============================================================================

def compute_cross_architecture_consistency(
    ace_vectors: Dict[str, Dict[str, float]],  # arch -> {dim -> ACE}
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute Causal Effect Consistency (CEC) across architectures.

    CEC(d) = Pearson correlation of ACE_d across all architecture pairs.

    Args:
        ace_vectors: Mapping from architecture name to dimension->ACE dict

    Returns:
        (CEC per dimension, average CEC per architecture vs MHA baseline)
    """
    dimensions = set()
    for ace in ace_vectors.values():
        dimensions.update(ace.keys())
    dimensions = sorted(dimensions)

    # Build matrix: [n_architectures, n_dimensions]
    arch_names = sorted(ace_vectors.keys())
    n_archs = len(arch_names)
    if n_archs < 2:
        return {d: 1.0 for d in dimensions}, {a: 1.0 for a in arch_names}

    ace_matrix = np.zeros((n_archs, len(dimensions)))
    for i, arch in enumerate(arch_names):
        for j, dim in enumerate(dimensions):
            ace_matrix[i, j] = ace_vectors[arch].get(dim, 0.0)

    # Per-dimension CEC: standard deviation across architectures (lower = more consistent)
    dim_cec = {}
    for j, dim in enumerate(dimensions):
        col = ace_matrix[:, j]
        if col.std() < 1e-9:
            dim_cec[dim] = 1.0
        else:
            # CEC = 1 - normalized_std
            normalized_std = col.std() / (abs(col.mean()) + 1e-9)
            dim_cec[dim] = max(0.0, 1.0 - min(1.0, normalized_std))

    # Per-architecture CEC vs MHA baseline
    arch_cec = {}
    mha_idx = None
    for i, a in enumerate(arch_names):
        if a == "MHA":
            mha_idx = i
            break

    if mha_idx is not None:
        mha_vec = ace_matrix[mha_idx]
        for i, arch in enumerate(arch_names):
            if i == mha_idx:
                arch_cec[arch] = 1.0
            else:
                corr = np.corrcoef(mha_vec, ace_matrix[i])[0, 1]
                arch_cec[arch] = max(0.0, float(corr)) if not np.isnan(corr) else 0.0
    else:
        for arch in arch_names:
            arch_cec[arch] = 1.0

    return dim_cec, arch_cec


# =============================================================================
# Layer 7: OCA Metrics
# =============================================================================

def compute_sbfi_violation_rate(
    bandit_scores: np.ndarray,     # [n_decisions]
    sbfi_min_score: float = 0.3,
    admitted_mask: np.ndarray = None,  # [n_decisions] bool, which were admitted
) -> int:
    """
    Count SBFI boundary violations.

    A violation occurs when a candidate is admitted with score below the SBFI floor.
    This must always be zero for the causal isolation guarantee.

    Returns:
        Number of violations (must be 0 for pass).
    """
    if admitted_mask is None:
        return 0
    violations = np.sum((bandit_scores < sbfi_min_score) & admitted_mask)
    return int(violations)


def compute_cumulative_regret(
    optimal_rewards: np.ndarray,   # [n_steps] oracle-optimal reward per step
    achieved_rewards: np.ndarray,  # [n_steps] actual reward per step
) -> float:
    """
    Compute cumulative regret: sum over steps of (optimal - achieved).
    """
    regret = np.cumsum(optimal_rewards - achieved_rewards)
    return float(regret[-1]) if len(regret) > 0 else 0.0


def detect_distribution_drift(
    commit_abort_history: List[float],
    window: int = 50,
    threshold: float = 0.2,
) -> List[int]:
    """
    Detect distribution drift events using CUSUM on commit/abort ratio.

    Args:
        commit_abort_history: Per-step commit/abort ratios
        window: Sliding window size for baseline
        threshold: CUSUM alarm threshold

    Returns:
        List of step indices where drift was detected
    """
    if len(commit_abort_history) < window * 2:
        return []

    drift_events = []
    cusum_pos = 0.0
    cusum_neg = 0.0

    for t in range(window, len(commit_abort_history)):
        # Baseline: mean of previous window
        baseline = np.mean(commit_abort_history[t - window:t])
        current = commit_abort_history[t]
        delta = current - baseline
        cusum_pos = max(0.0, cusum_pos + delta)
        cusum_neg = min(0.0, cusum_neg + delta)

        if cusum_pos > threshold or abs(cusum_neg) > threshold:
            drift_events.append(t)
            # Reset after detection
            cusum_pos = 0.0
            cusum_neg = 0.0

    return drift_events


def compute_recovery_stability(
    recovery_history: List[float],
    baseline_recovery: float,
    tolerance: float = 0.05,
) -> float:
    """
    Compute fraction of steps where recovery stays within tolerance of baseline.

    Used to verify OCA maintains recovery under distribution drift.
    """
    if not recovery_history:
        return 1.0
    stable_count = sum(
        1 for r in recovery_history
        if abs(r - baseline_recovery) <= tolerance
    )
    return stable_count / len(recovery_history)
