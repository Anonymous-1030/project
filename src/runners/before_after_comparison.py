"""
Before/After Comparison Runner for ProSE-X 2.0

Runs fair comparison under identical conditions:
- workload
- seed
- budget
- retention settings
- scorer mode
- transfer granularity

Compares:
1. Old baseline (v1)
2. Current patched version (v2)
3. Current patched + no burst
4. Current patched + burst radius 1
5. Current patched + burst radius 2
"""

import json
import logging
import csv
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.config import ProSEXv2Config
from src.eval.shared.evaluator_v2 import SharedEvaluatorV2, EvaluationResult
from src.eval.fairness_guard import preflight_check

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonConfig:
    """Configuration for a single comparison run."""
    name: str
    config: ProSEXv2Config
    description: str = ""


@dataclass
class ComparisonRunResult:
    """Result of a comparison run."""
    run_name: str
    evaluator_result: EvaluationResult
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "timestamp": self.timestamp,
            "metrics": {
                "conditional_recovery": self.evaluator_result.conditional_recovery,
                "no_miss_rate": self.evaluator_result.no_miss_rate,
                "upr_attention_based": self.evaluator_result.upr_attention_based,
                "upr_gold_based": self.evaluator_result.upr_gold_based,
                "upr_recovery_based": self.evaluator_result.upr_recovery_based,
                "promotion_miss_count": self.evaluator_result.failure_attribution.total_misses,
                "total_miss_count": self.evaluator_result.steps_with_miss,
                "promoted_bytes": self.evaluator_result.total_promoted_bytes,
                "active_bytes": self.evaluator_result.total_used_bytes,
                "budget_utilization": self.evaluator_result.budget_utilization,
                "latency_mean_ms": self.evaluator_result.latency_mean_ms,
                "latency_p95_ms": self.evaluator_result.latency_p95_ms,
                "steps_with_gold": self.evaluator_result.steps_with_gold,
                "steps_with_recovery": self.evaluator_result.steps_with_recovery,
                "gold_step_misses": self.evaluator_result.gold_step_misses,
            },
            "consistency_warnings": self.evaluator_result.consistency_warnings,
        }


class BeforeAfterComparisonRunner:
    """
    Runner for before/after comparison.
    
    Ensures fair comparison by using FairnessGuard for validation.
    """
    
    def __init__(
        self,
        base_config: ProSEXv2Config,
        workload_name: str = "needle_in_haystack",
        seed: int = 42,
    ):
        self.base_config = base_config
        self.workload_name = workload_name
        self.seed = seed
        self.runs: List[ComparisonConfig] = []
        self.results: List[ComparisonRunResult] = []
        
        logger.info(f"BeforeAfterComparisonRunner initialized: workload={workload_name}, seed={seed}")
    
    def add_run(
        self,
        name: str,
        config_modifications: Dict[str, Any],
        description: str = "",
    ) -> "BeforeAfterComparisonRunner":
        """Add a run to the comparison."""
        # Deep copy base config
        config_dict = self.base_config.to_dict()
        
        # Apply modifications
        self._deep_update(config_dict, config_modifications)
        
        # Ensure critical parameters are preserved
        config_dict["seed"] = self.seed
        config_dict["workload"] = self.workload_name
        
        # Create config
        run_config = ProSEXv2Config.from_dict(config_dict)
        
        run = ComparisonConfig(
            name=name,
            config=run_config,
            description=description,
        )
        
        self.runs.append(run)
        logger.info(f"Added comparison run: {name}")
        
        return self
    
    def run_all(
        self,
        run_fn: Callable[[ProSEXv2Config], SharedEvaluatorV2],
    ) -> List[ComparisonRunResult]:
        """
        Run all comparisons.
        
        Args:
            run_fn: Function that takes a config and returns an evaluator
            
        Returns:
            List of comparison results
        """
        self.results = []
        
        for i, run in enumerate(self.runs):
            logger.info(f"Running comparison {i+1}/{len(self.runs)}: {run.name}")
            
            # Run preflight fairness check
            base_config_dict = self.base_config.to_dict()
            run_config_dict = run.config.to_dict()
            
            fairness_report = preflight_check(
                baseline_config=base_config_dict,
                comparison_config=run_config_dict,
                comparison_name=run.name,
            )
            
            if not fairness_report.can_compare:
                logger.error(f"Fairness check failed for {run.name}, skipping...")
                continue
            
            # Execute
            try:
                evaluator = run_fn(run.config)
                result = evaluator.evaluate()
                
                comparison_result = ComparisonRunResult(
                    run_name=run.name,
                    evaluator_result=result,
                )
                
                self.results.append(comparison_result)
                logger.info(f"Completed {run.name}: CR={result.conditional_recovery:.2%}")
                
            except Exception as e:
                logger.error(f"Failed to run {run.name}: {e}")
                raise
        
        return self.results
    
    def generate_markdown_report(self) -> str:
        """Generate markdown comparison report."""
        if not self.results:
            return "No results to display."
        
        lines = [
            "# Before/After Comparison Report",
            "",
            f"**Workload:** {self.workload_name}",
            f"**Seed:** {self.seed}",
            f"**Timestamp:** {datetime.now().isoformat()}",
            "",
            "## Summary Table",
            "",
            "| Run | Cond. Rec. | No-Miss Rate | UPR (Attn) | UPR (Gold) | UPR (Rec) | Promoted Bytes | Latency (ms) |",
            "|-----|-----------|--------------|------------|------------|-----------|---------------|--------------|",
        ]
        
        for result in self.results:
            m = result.to_dict()["metrics"]
            lines.append(
                f"| {result.run_name} | "
                f"{m['conditional_recovery']:.2%} | "
                f"{m['no_miss_rate']:.2%} | "
                f"{m['upr_attention_based']:.2%} | "
                f"{m['upr_gold_based']:.2%} | "
                f"{m['upr_recovery_based']:.2%} | "
                f"{m['promoted_bytes']:,} | "
                f"{m['latency_mean_ms']:.2f} |"
            )
        
        lines.extend([
            "",
            "## Detailed Metrics",
            "",
        ])
        
        for result in self.results:
            lines.append(f"### {result.run_name}")
            lines.append("")
            
            m = result.to_dict()["metrics"]
            lines.append(f"- **Conditional Recovery:** {m['conditional_recovery']:.2%}")
            lines.append(f"- **No-Miss Rate:** {m['no_miss_rate']:.2%}")
            lines.append(f"- **UPR (Attention-based):** {m['upr_attention_based']:.2%}")
            lines.append(f"- **UPR (Gold-based):** {m['upr_gold_based']:.2%}")
            lines.append(f"- **UPR (Recovery-based):** {m['upr_recovery_based']:.2%}")
            lines.append(f"- **Gold-Step Misses:** {m['gold_step_misses']}")
            lines.append(f"- **Promoted Bytes:** {m['promoted_bytes']:,}")
            lines.append(f"- **Active Bytes:** {m['active_bytes']:,}")
            lines.append(f"- **Budget Utilization:** {m['budget_utilization']:.2%}")
            lines.append(f"- **Mean Latency:** {m['latency_mean_ms']:.2f} ms")
            lines.append(f"- **P95 Latency:** {m['latency_p95_ms']:.2f} ms")
            
            warnings = result.to_dict().get("consistency_warnings", [])
            if warnings:
                lines.append("")
                lines.append("**Consistency Warnings:**")
                for warning in warnings:
                    lines.append(f"- ⚠️ {warning}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def save_csv(self, output_path: str) -> None:
        """Save comparison as CSV."""
        if not self.results:
            return
        
        fieldnames = [
            "run_name",
            "conditional_recovery",
            "no_miss_rate",
            "upr_attention_based",
            "upr_gold_based",
            "upr_recovery_based",
            "promotion_miss_count",
            "total_miss_count",
            "promoted_bytes",
            "active_bytes",
            "budget_utilization",
            "latency_mean_ms",
            "latency_p95_ms",
            "steps_with_gold",
            "steps_with_recovery",
            "gold_step_misses",
        ]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {"run_name": result.run_name}
                row.update(result.to_dict()["metrics"])
                writer.writerow(row)
        
        logger.info(f"CSV report saved to {output_path}")
    
    def save_json(self, output_path: str) -> None:
        """Save comparison as JSON."""
        summary = {
            "workload": self.workload_name,
            "seed": self.seed,
            "timestamp": datetime.now().isoformat(),
            "runs": [result.to_dict() for result in self.results],
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"JSON report saved to {output_path}")
    
    def save_all(self, output_dir: str, name: str = "before_after_comparison") -> Dict[str, str]:
        """Save all report formats."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        saved = {}
        
        # Markdown
        md_path = Path(output_dir) / f"{name}.md"
        with open(md_path, 'w') as f:
            f.write(self.generate_markdown_report())
        saved["markdown"] = str(md_path)
        
        # CSV
        csv_path = Path(output_dir) / f"{name}.csv"
        self.save_csv(str(csv_path))
        saved["csv"] = str(csv_path)
        
        # JSON
        json_path = Path(output_dir) / f"{name}.json"
        self.save_json(str(json_path))
        saved["json"] = str(json_path)
        
        logger.info(f"All reports saved to {output_dir}")
        return saved
    
    def _deep_update(self, d: Dict, u: Dict) -> Dict:
        """Deep update dictionary d with values from u."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
        return d


def create_standard_before_after_comparison(
    base_config: ProSEXv2Config,
    workload_name: str = "needle_in_haystack",
    seed: int = 42,
) -> BeforeAfterComparisonRunner:
    """
    Create standard before/after comparison with 5 runs.
    
    Runs:
    1. Old baseline (simulated - would use v1 evaluator)
    2. Current patched (v2 evaluator)
    3. Current patched + no burst
    4. Current patched + burst radius 1
    5. Current patched + burst radius 2
    """
    runner = BeforeAfterComparisonRunner(base_config, workload_name, seed)
    
    # 1. Old baseline (v1)
    runner.add_run(
        name="old_baseline_v1",
        config_modifications={
            "evaluation": {"version": "1.0.0"},
        },
        description="Old baseline with v1 evaluator (for reference)",
    )
    
    # 2. Current patched (v2)
    runner.add_run(
        name="patched_v2_baseline",
        config_modifications={},
        description="Current patched system with v2 evaluator",
    )
    
    # 3. No burst
    runner.add_run(
        name="patched_no_burst",
        config_modifications={
            "burst": {
                "enabled": False,
                "radius": 0,
            }
        },
        description="Current patched with burst disabled",
    )
    
    # 4. Burst radius 1
    runner.add_run(
        name="patched_burst_r1",
        config_modifications={
            "burst": {
                "enabled": True,
                "radius": 1,
            }
        },
        description="Current patched with burst radius 1",
    )
    
    # 5. Burst radius 2
    runner.add_run(
        name="patched_burst_r2",
        config_modifications={
            "burst": {
                "enabled": True,
                "radius": 2,
            }
        },
        description="Current patched with burst radius 2",
    )
    
    return runner


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run before/after comparison")
    parser.add_argument("--output-dir", default="outputs/reports", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workload", default="needle_in_haystack", help="Workload name")
    
    args = parser.parse_args()
    
    # Create base config
    config = ProSEXv2Config(seed=args.seed, experiment_name=args.workload)
    
    # Create runner
    runner = create_standard_before_after_comparison(
        base_config=config,
        workload_name=args.workload,
        seed=args.seed,
    )
    
    logger.info(f"Created comparison with {len(runner.runs)} runs")
    logger.info("Runs: " + ", ".join([r.name for r in runner.runs]))
    
    # Note: Actual run requires a workload runner function
    # This would integrate with the actual simulation or real system
    logger.info("To execute comparison, provide a run_fn that takes ProSEXv2Config and returns SharedEvaluatorV2")


if __name__ == "__main__":
    main()
