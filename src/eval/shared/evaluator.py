"""
Shared Evaluator for ProSE-X 2.0

Single source of truth for all metric computation.
All experiments must use this evaluator.
"""

import json
import logging
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from .metrics import (
    RecoveryMetrics,
    UsefulnessMetrics,
    BurstMetrics,
    compute_conditional_recovery,
    compute_useful_promote_ratio,
    compute_burst_gain,
    compute_candidate_recall_at_k,
    compute_budget_utilization,
    compute_latency_statistics,
)

logger = logging.getLogger(__name__)


@dataclass
class QueueContributionMetrics:
    """Per-queue contribution metrics."""
    queue_name: str
    raw_output_count: int = 0
    post_dedup_count: int = 0
    entering_scorer: int = 0
    surviving_scheduler: int = 0
    burst_expanded: int = 0
    ultimately_useful: int = 0
    total_bytes_promoted: int = 0
    useful_bytes: int = 0
    empty_reason: Optional[str] = None


@dataclass
class FailureAttributionMetrics:
    """Failure attribution counts."""
    candidate_miss: int = 0
    scorer_rank_miss: int = 0
    scheduler_budget_drop: int = 0
    scheduler_threshold_drop: int = 0
    burst_boundary_miss: int = 0
    sticky_eviction_miss: int = 0
    promoted_but_unused: int = 0
    retention_miss: int = 0
    total_misses: int = 0


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    
    # Experiment info
    experiment_id: str
    config: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Recovery metrics
    conditional_recovery: float = 0.0
    no_miss_rate: float = 0.0
    total_steps: int = 0
    steps_with_gold: int = 0
    steps_with_recovery: int = 0
    steps_with_miss: int = 0
    
    # Useful promote ratio (all modes)
    total_promoted_bytes: int = 0
    upr_attention_based: float = 0.0
    upr_gold_based: float = 0.0
    upr_recovery_based: float = 0.0
    
    # Queue contributions
    queue_contributions: Dict[str, QueueContributionMetrics] = field(default_factory=dict)
    
    # Failure attribution
    failure_attribution: FailureAttributionMetrics = field(default_factory=FailureAttributionMetrics)
    
    # Budget and bytes
    budget_utilization: float = 0.0
    total_budget_bytes: int = 0
    total_used_bytes: int = 0
    
    # Latency
    latency_mean_ms: float = 0.0
    latency_p95_ms: float = 0.0

    # PHT/PTB metrics (populated when PHT/PTB enabled)
    pht_prediction_accuracy: float = 0.0
    ptb_prefetch_hit_rate: float = 0.0
    pig_efficiency: float = 0.0  # achieved PIG / optimal PIG

    # Per-step results for detailed analysis
    per_step_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, output_path: str) -> None:
        """Save to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Evaluation result saved to {output_path}")


class SharedEvaluator:
    """
    Shared evaluator for all ProSE-X 2.0 experiments.
    
    This is the ONLY allowed source of metric computation.
    All runners must use this evaluator.
    """
    
    def __init__(self, config: Dict[str, Any], experiment_id: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            config: Experiment configuration
            experiment_id: Unique identifier for this experiment
        """
        self.config = config
        self.experiment_id = experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Storage for step-level data
        self.step_results: List[Dict[str, Any]] = []
        self.promotion_records: List[Dict[str, Any]] = []
        
        # Metric spec version for validation
        self.metric_spec_version = "1.0.0"
        
        logger.info(f"SharedEvaluator initialized: {self.experiment_id}")
    
    def add_step_result(
        self,
        step: int,
        gold_chunk_ids: Set[str],
        visible_chunk_ids: Set[str],
        candidate_chunk_ids: List[str],
        selected_chunk_ids: List[str],
        promoted_chunk_ids: List[str],
        budget_bytes: int,
        used_bytes: int,
        latency_us: float,
        queue_contributions: Optional[Dict[str, Any]] = None,
        failure_reason: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a step result for evaluation.
        
        Args:
            step: Step number
            gold_chunk_ids: Set of gold chunk IDs (evaluation only)
            visible_chunk_ids: Set of visible chunk IDs
            candidate_chunk_ids: Ordered list of candidates from ULF
            selected_chunk_ids: Chunks selected by scheduler
            promoted_chunk_ids: Final promoted chunks after burst/sticky
            budget_bytes: Budget for this step
            used_bytes: Actual bytes used
            latency_us: Total latency in microseconds
            queue_contributions: Per-queue contribution info
            failure_reason: If miss occurred, the reason
            extra: Additional step-specific data
        """
        result = {
            "step": step,
            "gold_chunk_ids": list(gold_chunk_ids),
            "visible_chunk_ids": list(visible_chunk_ids),
            "candidate_chunk_ids": candidate_chunk_ids,
            "selected_chunk_ids": selected_chunk_ids,
            "promoted_chunk_ids": promoted_chunk_ids,
            "budget_bytes": budget_bytes,
            "used_bytes": used_bytes,
            "latency_us": latency_us,
            "queue_contributions": queue_contributions or {},
            "failure_reason": failure_reason,
            "extra": extra or {},
        }
        self.step_results.append(result)
    
    def add_promotion_record(
        self,
        step: int,
        chunk_id: str,
        bytes_transferred: int,
        center_unit_id: Optional[str] = None,
        burst_expanded_ids: Optional[List[str]] = None,
        queue_of_origin: Optional[str] = None,
        score: Optional[float] = None,
        rank: Optional[int] = None,
        selection_type: Optional[str] = None,  # "exploit" or "explore"
        first_visible_step: Optional[int] = None,
        future_access_count: int = 0,
        max_attention_weight: float = 0.0,
        gold_overlap_tokens: int = 0,
        contributed_to_recovery: bool = False,
        ttl_original: int = 0,
        eviction_reason: Optional[str] = None,
    ) -> None:
        """
        Add a promotion record for usefulness tracking.
        
        This creates the promote-to-use chain for each promoted unit.
        """
        record = {
            "step": step,
            "chunk_id": chunk_id,
            "bytes_transferred": bytes_transferred,
            "center_unit_id": center_unit_id,
            "burst_expanded_ids": burst_expanded_ids or [],
            "queue_of_origin": queue_of_origin,
            "score": score,
            "rank": rank,
            "selection_type": selection_type,
            "first_visible_step": first_visible_step,
            "future_access_count": future_access_count,
            "max_attention_weight": max_attention_weight,
            "gold_overlap_tokens": gold_overlap_tokens,
            "contributed_to_recovery": contributed_to_recovery,
            "ttl_original": ttl_original,
            "eviction_reason": eviction_reason,
        }
        self.promotion_records.append(record)
    
    def evaluate(self) -> EvaluationResult:
        """
        Run full evaluation and return results.
        
        This is the main entry point for metric computation.
        """
        if not self.step_results:
            logger.warning("No step results to evaluate")
            return EvaluationResult(experiment_id=self.experiment_id, config=self.config)
        
        # Compute recovery metrics
        recovery = self._compute_recovery_metrics()
        
        # Compute usefulness metrics
        usefulness = self._compute_usefulness_metrics()
        
        # Compute queue contributions
        queue_contribs = self._compute_queue_contributions()
        
        # Compute failure attribution
        failure_attr = self._compute_failure_attribution()
        
        # Compute budget metrics
        budget_util, total_budget, total_used = self._compute_budget_metrics()
        
        # Compute latency metrics
        latency_stats = self._compute_latency_metrics()
        
        return EvaluationResult(
            experiment_id=self.experiment_id,
            config=self.config,
            conditional_recovery=recovery.conditional_recovery,
            no_miss_rate=recovery.no_miss_rate,
            total_steps=recovery.total_steps,
            steps_with_gold=recovery.steps_with_gold,
            steps_with_recovery=recovery.steps_with_recovery,
            steps_with_miss=recovery.steps_with_miss,
            total_promoted_bytes=usefulness.total_promoted_bytes,
            upr_attention_based=usefulness.upr_attention_based,
            upr_gold_based=usefulness.upr_gold_based,
            upr_recovery_based=usefulness.upr_recovery_based,
            queue_contributions=queue_contribs,
            failure_attribution=failure_attr,
            budget_utilization=budget_util,
            total_budget_bytes=total_budget,
            total_used_bytes=total_used,
            latency_mean_ms=latency_stats["mean_ms"],
            latency_p95_ms=latency_stats["p95_ms"],
            per_step_results=self.step_results,
        )
    
    def _compute_recovery_metrics(self) -> RecoveryMetrics:
        """Compute conditional recovery and no-miss rate."""
        def gold_accessor(step):
            return set(step.get("gold_chunk_ids", []))
        
        def visibility_accessor(step):
            return set(step.get("visible_chunk_ids", []))
        
        return compute_conditional_recovery(
            self.step_results,
            gold_accessor,
            visibility_accessor,
        )
    
    def _compute_usefulness_metrics(self) -> UsefulnessMetrics:
        """Compute useful promote ratio (all modes)."""
        return compute_useful_promote_ratio(
            self.promotion_records,
            usefulness_criteria="attention_access_based",
        )
    
    def _compute_queue_contributions(self) -> Dict[str, QueueContributionMetrics]:
        """Compute per-queue contribution metrics."""
        contributions = {}
        
        for step in self.step_results:
            queue_contribs = step.get("queue_contributions", {})
            
            for queue_name, data in queue_contribs.items():
                if queue_name not in contributions:
                    contributions[queue_name] = QueueContributionMetrics(
                        queue_name=queue_name
                    )
                
                contrib = contributions[queue_name]
                contrib.raw_output_count += len(data.get("raw_output", [])) if isinstance(data.get("raw_output"), list) else data.get("raw_output", 0)
                contrib.post_dedup_count += len(data.get("post_dedup", [])) if isinstance(data.get("post_dedup"), list) else data.get("post_dedup", 0)
                contrib.entering_scorer += len(data.get("entering_scorer", [])) if isinstance(data.get("entering_scorer"), list) else data.get("entering_scorer", 0)
                contrib.surviving_scheduler += len(data.get("surviving_scheduler", [])) if isinstance(data.get("surviving_scheduler"), list) else data.get("surviving_scheduler", 0)
                contrib.burst_expanded += len(data.get("burst_expanded", [])) if isinstance(data.get("burst_expanded"), list) else data.get("burst_expanded", 0)
                contrib.ultimately_useful += len(data.get("ultimately_useful", [])) if isinstance(data.get("ultimately_useful"), list) else data.get("ultimately_useful", 0)
                
                if data.get("empty_reason") and not contrib.empty_reason:
                    contrib.empty_reason = data.get("empty_reason")
        
        return contributions
    
    def _compute_failure_attribution(self) -> FailureAttributionMetrics:
        """Compute failure attribution histogram."""
        attr = FailureAttributionMetrics()
        
        for step in self.step_results:
            reason = step.get("failure_reason")
            if reason:
                attr.total_misses += 1
                
                if reason == "candidate_miss":
                    attr.candidate_miss += 1
                elif reason == "scorer_rank_miss":
                    attr.scorer_rank_miss += 1
                elif reason == "scheduler_budget_drop":
                    attr.scheduler_budget_drop += 1
                elif reason == "scheduler_threshold_drop":
                    attr.scheduler_threshold_drop += 1
                elif reason == "burst_boundary_miss":
                    attr.burst_boundary_miss += 1
                elif reason == "sticky_eviction_miss":
                    attr.sticky_eviction_miss += 1
                elif reason == "promoted_but_unused":
                    attr.promoted_but_unused += 1
                elif reason == "retention_miss":
                    attr.retention_miss += 1
        
        return attr
    
    def _compute_budget_metrics(self) -> tuple:
        """Compute budget utilization metrics."""
        total_budget = 0
        total_used = 0
        
        for step in self.step_results:
            total_budget += step.get("budget_bytes", 0)
            total_used += step.get("used_bytes", 0)
        
        utilization = compute_budget_utilization(total_used, total_budget)
        
        return utilization, total_budget, total_used
    
    def _compute_latency_metrics(self) -> Dict[str, float]:
        """Compute latency statistics."""
        latencies = [step.get("latency_us", 0.0) for step in self.step_results]
        return compute_latency_statistics(latencies)
    
    def compare_with(
        self,
        other: "SharedEvaluator",
        comparison_name: str = "comparison",
    ) -> Dict[str, Any]:
        """
        Compare this evaluator's results with another.
        
        Returns a structured comparison report.
        """
        self_results = self.evaluate()
        other_results = other.evaluate()
        
        comparison = {
            "comparison_name": comparison_name,
            "timestamp": datetime.now().isoformat(),
            "baseline_experiment": self.experiment_id,
            "comparison_experiment": other.experiment_id,
            "metrics": {
                "conditional_recovery": {
                    "baseline": self_results.conditional_recovery,
                    "comparison": other_results.conditional_recovery,
                    "delta": other_results.conditional_recovery - self_results.conditional_recovery,
                },
                "no_miss_rate": {
                    "baseline": self_results.no_miss_rate,
                    "comparison": other_results.no_miss_rate,
                    "delta": other_results.no_miss_rate - self_results.no_miss_rate,
                },
                "upr_attention_based": {
                    "baseline": self_results.upr_attention_based,
                    "comparison": other_results.upr_attention_based,
                    "delta": other_results.upr_attention_based - self_results.upr_attention_based,
                },
                "budget_utilization": {
                    "baseline": self_results.budget_utilization,
                    "comparison": other_results.budget_utilization,
                    "delta": other_results.budget_utilization - self_results.budget_utilization,
                },
            },
            "config_diff": self._compute_config_diff(self.config, other.config),
        }
        
        return comparison
    
    def _compute_config_diff(self, config1: Dict, config2: Dict) -> List[Dict]:
        """Compute differences between two configs."""
        diffs = []
        
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)
            
            if val1 != val2:
                diffs.append({
                    "key": key,
                    "baseline": val1,
                    "comparison": val2,
                })
        
        return diffs
