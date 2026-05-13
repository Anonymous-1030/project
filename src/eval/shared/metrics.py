"""
Core metric computation functions for ProSE-X 2.0

All metrics follow strict definitions from docs/metrics_spec.md
No implicit assumptions. Explicit computation only.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class RecoveryMetrics:
    """Recovery-related metrics."""
    conditional_recovery: float
    no_miss_rate: float
    total_steps: int
    steps_with_gold: int
    steps_with_recovery: int
    steps_with_miss: int


@dataclass
class UsefulnessMetrics:
    """Useful promotion metrics."""
    total_promoted_bytes: int
    useful_bytes_attention_based: int
    useful_bytes_gold_based: int
    useful_bytes_recovery_based: int
    upr_attention_based: float
    upr_gold_based: float
    upr_recovery_based: float


@dataclass
class BurstMetrics:
    """Burst ablation metrics."""
    recovery_with_burst: float
    recovery_without_burst: float
    bytes_with_burst: int
    bytes_without_burst: int
    burst_gain: float  # per MB


def compute_conditional_recovery(
    step_results: List[Dict[str, Any]],
    gold_accessor: callable,
    visibility_accessor: callable,
) -> RecoveryMetrics:
    """
    Compute Conditional Recovery and No-Miss Rate.
    
    Args:
        step_results: List of per-step results
        gold_accessor: Function(step_result) -> Set[chunk_ids] that are gold
        visibility_accessor: Function(step_result) -> Set[chunk_ids] that are visible
        
    Returns:
        RecoveryMetrics with all computed values
        
    Definition (from metrics_spec.md):
    - Conditional Recovery: Among cases where gold exists in universe, 
      how often is it made visible?
    - No-Miss Rate: Fraction of steps where no evidence miss occurs
    """
    total_steps = len(step_results)
    steps_with_gold = 0
    steps_with_recovery = 0
    steps_with_miss = 0
    
    for step in step_results:
        gold_ids = gold_accessor(step)
        visible_ids = visibility_accessor(step)
        
        # Check if gold exists in system
        has_gold = len(gold_ids) > 0
        
        if has_gold:
            steps_with_gold += 1
            
            # Check if any gold is visible
            recovered = len(gold_ids & visible_ids) > 0
            
            if recovered:
                steps_with_recovery += 1
            else:
                steps_with_miss += 1
    
    # Conditional Recovery = P(visible | gold_exists)
    conditional_recovery = (
        steps_with_recovery / steps_with_gold if steps_with_gold > 0 else 0.0
    )
    
    # No-Miss Rate = 1 - P(miss)
    no_miss_rate = (
        (total_steps - steps_with_miss) / total_steps if total_steps > 0 else 0.0
    )
    
    return RecoveryMetrics(
        conditional_recovery=conditional_recovery,
        no_miss_rate=no_miss_rate,
        total_steps=total_steps,
        steps_with_gold=steps_with_gold,
        steps_with_recovery=steps_with_recovery,
        steps_with_miss=steps_with_miss,
    )


def compute_no_miss_rate(
    step_results: List[Dict[str, Any]],
    gold_needed_accessor: callable,
    visibility_accessor: callable,
) -> float:
    """
    Compute No-Miss Rate directly.
    
    A "miss" occurs when gold is needed but not visible.
    """
    total_steps = len(step_results)
    miss_count = 0
    
    for step in step_results:
        gold_needed = gold_needed_accessor(step)
        visible_ids = visibility_accessor(step)
        
        if gold_needed and not visible_ids:
            miss_count += 1
    
    return (total_steps - miss_count) / total_steps if total_steps > 0 else 0.0


def compute_useful_promote_ratio(
    promoted_units: List[Dict[str, Any]],
    usefulness_criteria: str = "attention_access_based",
    attention_threshold: float = 0.01,
    attention_window: int = 10,
) -> UsefulnessMetrics:
    """
    Compute Useful Promote Ratio (UPR).
    
    Args:
        promoted_units: List of promotion records with metadata
        usefulness_criteria: One of ["attention_access_based", "gold_overlap_based", "recovery_event_based"]
        attention_threshold: Minimum attention weight to count as useful
        attention_window: Number of steps to check for attention
        
    Returns:
        UsefulnessMetrics with all accounting modes
        
    Definition (from metrics_spec.md):
    UPR = useful_promoted_bytes / total_promoted_bytes
    
    A promoted byte is useful ONLY if it satisfies explicit criteria:
    1. Attention access based: chunk receives attention weight > threshold
    2. Gold overlap based: chunk overlaps gold region (evaluation only)
    3. Recovery event based: chunk participates in successful recovery
    """
    total_bytes = 0
    useful_attention = 0
    useful_gold = 0
    useful_recovery = 0
    
    for unit in promoted_units:
        bytes_count = unit.get("bytes_transferred", 0)
        total_bytes += bytes_count
        
        # Criterion 1: Attention access based
        future_accesses = unit.get("future_access_count", 0)
        max_attention = unit.get("max_attention_weight", 0.0)
        
        if future_accesses > 0 and max_attention >= attention_threshold:
            useful_attention += bytes_count
        
        # Criterion 2: Gold overlap based (evaluation only)
        gold_overlap_tokens = unit.get("gold_overlap_tokens", 0)
        if gold_overlap_tokens > 0:
            useful_gold += bytes_count
        
        # Criterion 3: Recovery event based
        contributed_to_recovery = unit.get("contributed_to_recovery", False)
        if contributed_to_recovery:
            useful_recovery += bytes_count
    
    # Compute ratios
    upr_attention = useful_attention / total_bytes if total_bytes > 0 else 0.0
    upr_gold = useful_gold / total_bytes if total_bytes > 0 else 0.0
    upr_recovery = useful_recovery / total_bytes if total_bytes > 0 else 0.0
    
    return UsefulnessMetrics(
        total_promoted_bytes=total_bytes,
        useful_bytes_attention_based=useful_attention,
        useful_bytes_gold_based=useful_gold,
        useful_bytes_recovery_based=useful_recovery,
        upr_attention_based=upr_attention,
        upr_gold_based=upr_gold,
        upr_recovery_based=upr_recovery,
    )


def compute_burst_gain(
    results_with_burst: Dict[str, Any],
    results_without_burst: Dict[str, Any],
) -> BurstMetrics:
    """
    Compute Burst Gain metric.
    
    Args:
        results_with_burst: Metrics from run with burst enabled
        results_without_burst: Metrics from run with burst disabled
        
    Returns:
        BurstMetrics with gain calculation
        
    Definition (from metrics_spec.md):
    Burst Gain = (recovery_with_burst - recovery_without_burst) / 
                 (bytes_with_burst - bytes_without_burst)
                 
    Unit: Recovery percentage points per additional MB
    """
    recovery_with = results_with_burst.get("conditional_recovery", 0.0)
    recovery_without = results_without_burst.get("conditional_recovery", 0.0)
    
    bytes_with = results_with_burst.get("total_promoted_bytes", 0)
    bytes_without = results_without_burst.get("total_promoted_bytes", 0)
    
    recovery_delta = recovery_with - recovery_without
    bytes_delta_mb = (bytes_with - bytes_without) / 1_000_000
    
    # Burst gain per MB
    burst_gain = recovery_delta / bytes_delta_mb if bytes_delta_mb > 0 else 0.0
    
    return BurstMetrics(
        recovery_with_burst=recovery_with,
        recovery_without_burst=recovery_without,
        bytes_with_burst=bytes_with,
        bytes_without_burst=bytes_without,
        burst_gain=burst_gain,
    )


def compute_candidate_recall_at_k(
    candidate_ids: List[str],
    gold_ids: Set[str],
    k_values: List[int] = [1, 5, 10, 20],
) -> Dict[int, float]:
    """
    Compute Candidate Recall@K.
    
    Args:
        candidate_ids: Ordered list of candidate chunk IDs (by score/rank)
        gold_ids: Set of gold chunk IDs
        k_values: K values to compute recall at
        
    Returns:
        Dict mapping k -> recall@k
        
    Definition (from metrics_spec.md):
    Candidate Recall@K = |gold ∩ top_k_candidates| / |gold|
    """
    results = {}
    candidate_set = set(candidate_ids)
    
    for k in k_values:
        top_k = set(candidate_ids[:k])
        recalled = len(gold_ids & top_k)
        recall = recalled / len(gold_ids) if gold_ids else 0.0
        results[k] = recall
    
    return results


def compute_budget_utilization(
    used_bytes: int,
    budget_bytes: int,
) -> float:
    """
    Compute budget utilization ratio.
    
    Returns:
        used_bytes / budget_bytes (clamped to [0, 1])
    """
    if budget_bytes <= 0:
        return 0.0
    return min(1.0, used_bytes / budget_bytes)


def compute_latency_statistics(
    latency_values: List[float],
) -> Dict[str, float]:
    """
    Compute latency statistics.
    
    Args:
        latency_values: List of latency measurements (in microseconds)
        
    Returns:
        Dict with mean, p50, p95, p99 in milliseconds
    """
    if not latency_values:
        return {
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }
    
    arr = np.array(latency_values)
    
    return {
        "mean_ms": np.mean(arr) / 1000.0,
        "p50_ms": np.percentile(arr, 50) / 1000.0,
        "p95_ms": np.percentile(arr, 95) / 1000.0,
        "p99_ms": np.percentile(arr, 99) / 1000.0,
    }
