"""
Master Evaluation Orchestrator for ProSE-X 2.0

Coordinates all evaluation components:
1. Shared Evaluator for metric computation
2. Useful Bytes Accounting
3. Neighbor Recall Reporting
4. Failure Attribution
5. Burst Ablation
6. Sticky Ablation
7. Fair Comparison Runner

This is the main entry point for comprehensive evaluation.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.config import ProSEXv2Config
from src.eval.shared import SharedEvaluator, EvaluationResult
from src.eval.accounting import UsefulBytesAccountant, AccountingMode
from src.eval.failure_attribution.attributor import FailureAttributor
from src.eval.neighbor_recall_report import NeighborRecallReporter
from src.runners.comparison_runner import FairComparisonRunner
from src.runners.burst_ablation import create_burst_ablation_runner, generate_burst_ablation_report
from src.runners.sticky_ablation import create_sticky_ablation_runner, generate_sticky_ablation_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationContext:
    """Context object passed through evaluation pipeline."""
    config: ProSEXv2Config
    step: int = 0
    gold_chunk_ids: set = field(default_factory=set)
    visible_chunk_ids: set = field(default_factory=set)
    candidate_chunk_ids: List[str] = field(default_factory=list)
    selected_chunk_ids: List[str] = field(default_factory=list)
    promoted_chunk_ids: List[str] = field(default_factory=list)
    queue_contributions: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


class EvaluationOrchestrator:
    """
    Master orchestrator for ProSE-X 2.0 evaluation.
    
    Coordinates all evaluation components to answer:
    Q1. Does neighbor recall produce useful promotions?
    Q2. Does promotion improve Conditional Recovery and No-Miss Rate?
    Q3. What fraction of promoted bytes are useful?
    Q4. Is burst truly efficient?
    """
    
    def __init__(
        self,
        config: ProSEXv2Config,
        output_dir: str = "outputs",
        experiment_id: Optional[str] = None,
    ):
        """
        Initialize orchestrator.
        
        Args:
            config: ProSE-X 2.0 configuration
            output_dir: Base output directory
            experiment_id: Unique experiment identifier
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.experiment_id = experiment_id or f"prose_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize all evaluation components
        self.shared_evaluator = SharedEvaluator(
            config=config.to_dict(),
            experiment_id=self.experiment_id,
        )
        self.useful_bytes_accountant = UsefulBytesAccountant()
        self.failure_attributor = FailureAttributor()
        self.neighbor_reporter = NeighborRecallReporter()
        
        # Storage
        self.step_contexts: List[EvaluationContext] = []
        
        logger.info(f"EvaluationOrchestrator initialized: {self.experiment_id}")
    
    def record_step(
        self,
        context: EvaluationContext,
        pipeline_result: Optional[Any] = None,
    ) -> None:
        """
        Record a step for evaluation.
        
        Args:
            context: Evaluation context for this step
            pipeline_result: Optional pipeline result object
        """
        # 1. Record in shared evaluator
        self.shared_evaluator.add_step_result(
            step=context.step,
            gold_chunk_ids=context.gold_chunk_ids,
            visible_chunk_ids=context.visible_chunk_ids,
            candidate_chunk_ids=context.candidate_chunk_ids,
            selected_chunk_ids=context.selected_chunk_ids,
            promoted_chunk_ids=context.promoted_chunk_ids,
            budget_bytes=context.extra.get("budget_bytes", 0),
            used_bytes=context.extra.get("used_bytes", 0),
            latency_us=context.extra.get("latency_us", 0.0),
            queue_contributions=context.queue_contributions,
            failure_reason=self._determine_failure_reason(context),
        )
        
        # 2. Record neighbor recall data
        anchor_data = context.queue_contributions.get("anchor_neighbor", {})
        if anchor_data:
            self.neighbor_reporter.record_step(
                step=context.step,
                raw_output=anchor_data.get("raw_output", []),
                post_dedup=anchor_data.get("post_dedup", []),
                entering_scorer=anchor_data.get("entering_scorer", []),
                surviving_scheduler=anchor_data.get("surviving_scheduler", []),
                burst_expanded=anchor_data.get("burst_expanded", []),
                ultimately_useful=anchor_data.get("ultimately_useful", []),
                empty_reason=anchor_data.get("empty_reason"),
                chunk_bytes=context.extra.get("chunk_bytes", {}),
            )
        
        # 3. Record useful bytes for promoted units
        for chunk_id in context.promoted_chunk_ids:
            chunk_info = context.extra.get("chunk_info", {}).get(chunk_id, {})
            
            self.useful_bytes_accountant.record_promotion(
                chunk_id=chunk_id,
                step=context.step,
                request_id=context.extra.get("request_id", ""),
                bytes_transferred=chunk_info.get("bytes", 0),
                queue_of_origin=chunk_info.get("queue_of_origin"),
                score=chunk_info.get("score", 0.0),
                rank=chunk_info.get("rank", 0),
                selection_type=chunk_info.get("selection_type"),
                ttl_original=chunk_info.get("ttl", 0),
            )
            
            # Check if useful (evaluation only)
            is_useful = chunk_id in context.gold_chunk_ids
            if is_useful:
                unit = self.useful_bytes_accountant.units.get(chunk_id)
                if unit:
                    unit.gold_overlap_tokens = chunk_info.get("bytes", 0) // 100  # Approx
                    unit.overlaps_gold_region = True
        
        # 4. Record failure attribution for missed gold
        for gold_id in context.gold_chunk_ids:
            if gold_id not in context.visible_chunk_ids:
                self.failure_attributor.attribute_failure(
                    step=context.step,
                    gold_chunk_id=gold_id,
                    gold_exists_in_system=True,  # Assumed
                    candidate_ids=context.candidate_chunk_ids,
                    scored_candidates=context.extra.get("scored_candidates", []),
                    selected_ids=context.selected_chunk_ids,
                    burst_expanded_ids=context.extra.get("burst_expanded", []),
                    sticky_promoted_ids=context.promoted_chunk_ids,
                )
        
        self.step_contexts.append(context)
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize evaluation and generate all reports.
        
        Returns:
            Summary of all generated reports
        """
        logger.info("Finalizing evaluation...")
        
        reports = {}
        
        # 1. Shared evaluator report
        eval_result = self.shared_evaluator.evaluate()
        eval_path = self.output_dir / "reports" / "evaluation_result.json"
        eval_result.save(str(eval_path))
        reports["evaluation"] = str(eval_path)
        
        # 2. Useful bytes report
        useful_report = self.useful_bytes_accountant.get_accounting_report()
        useful_path = self.output_dir / "reports" / "useful_bytes_report.json"
        with open(useful_path, 'w') as f:
            json.dump(useful_report, f, indent=2, default=str)
        reports["useful_bytes"] = str(useful_path)
        
        # 3. Neighbor recall report
        neighbor_path = self.output_dir / "reports" / "neighbor_recall_report.json"
        self.neighbor_reporter.save(str(neighbor_path))
        reports["neighbor_recall"] = str(neighbor_path)
        
        # 4. Failure attribution report
        failure_report = self.failure_attributor.generate_report(
            total_steps=len(self.step_contexts)
        )
        failure_path = self.output_dir / "reports" / "failure_attribution.json"
        with open(failure_path, 'w') as f:
            json.dump(failure_report.to_dict(), f, indent=2)
        reports["failure_attribution"] = str(failure_path)
        
        logger.info(f"Evaluation complete. Reports saved to {self.output_dir}/reports/")
        
        return {
            "experiment_id": self.experiment_id,
            "reports": reports,
            "summary": {
                "conditional_recovery": eval_result.conditional_recovery,
                "no_miss_rate": eval_result.no_miss_rate,
                "upr_attention": eval_result.upr_attention_based,
                "total_steps": eval_result.total_steps,
                "steps_with_miss": eval_result.steps_with_miss,
            }
        }
    
    def run_burst_ablation(
        self,
        run_fn: Callable[[ProSEXv2Config], SharedEvaluator],
    ) -> Dict[str, Any]:
        """
        Run burst ablation study.
        
        Args:
            run_fn: Function to run a single configuration
            
        Returns:
            Ablation report
        """
        logger.info("Running burst ablation...")
        
        runner = create_burst_ablation_runner(
            self.config,
            workload=self.experiment_id,
            seed=self.config.seed,
        )
        
        results = runner.run_all(run_fn)
        report = generate_burst_ablation_report(
            results,
            output_dir=str(self.output_dir / "reports"),
        )
        
        return report
    
    def run_sticky_ablation(
        self,
        run_fn: Callable[[ProSEXv2Config], SharedEvaluator],
    ) -> Dict[str, Any]:
        """
        Run sticky TTL ablation study.
        
        Args:
            run_fn: Function to run a single configuration
            
        Returns:
            Ablation report
        """
        logger.info("Running sticky ablation...")
        
        runner = create_sticky_ablation_runner(
            self.config,
            workload=self.experiment_id,
            seed=self.config.seed,
        )
        
        results = runner.run_all(run_fn)
        report = generate_sticky_ablation_report(
            results,
            output_dir=str(self.output_dir / "reports"),
        )
        
        return report
    
    def _determine_failure_reason(self, context: EvaluationContext) -> Optional[str]:
        """Determine failure reason for this step."""
        # Check if any gold was missed
        for gold_id in context.gold_chunk_ids:
            if gold_id not in context.visible_chunk_ids:
                # Determine why
                if gold_id not in context.candidate_chunk_ids:
                    return "candidate_miss"
                elif gold_id not in context.selected_chunk_ids:
                    return "scheduler_budget_drop"
                elif gold_id not in context.promoted_chunk_ids:
                    return "sticky_eviction_miss"
                else:
                    return "promoted_but_unused"
        return None
    
    def clear(self) -> None:
        """Clear all evaluation state."""
        self.shared_evaluator = SharedEvaluator(
            config=self.config.to_dict(),
            experiment_id=self.experiment_id,
        )
        self.useful_bytes_accountant.clear()
        self.failure_attributor.clear()
        self.neighbor_reporter.clear()
        self.step_contexts.clear()
        logger.info("EvaluationOrchestrator cleared")


def create_orchestrator_from_baseline(
    baseline_path: str = "outputs/reports/baseline_after_scheduler_fix.json",
    output_dir: str = "outputs",
) -> EvaluationOrchestrator:
    """
    Create orchestrator from saved baseline config.
    
    Args:
        baseline_path: Path to baseline config
        output_dir: Output directory
        
    Returns:
        Configured EvaluationOrchestrator
    """
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    config = ProSEXv2Config.from_dict(baseline.get("config", {}))
    
    return EvaluationOrchestrator(
        config=config,
        output_dir=output_dir,
        experiment_id=f"from_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
