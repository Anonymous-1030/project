"""
Fair Comparison Runner for ProSE-X 2.0

Ensures fair, apples-to-apples comparisons across experiments.
Guarantees same workload, seed, budget, retention settings.
"""

import json
import logging
import copy
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.config import ProSEXv2Config
from src.eval.shared import SharedEvaluator

logger = logging.getLogger(__name__)


@dataclass
class ComparisonRun:
    """Single run configuration in a comparison."""
    name: str
    config: ProSEXv2Config
    description: str = ""
    modifications: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of a comparison run."""
    run_name: str
    evaluator_result: Any  # EvaluationResult
    latency_ms: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


class FairComparisonRunner:
    """
    Runner for fair comparisons across configurations.
    
    GUARANTEES:
    - Same workload
    - Same seed
    - Same budget
    - Same retention settings (unless explicitly ablated)
    - Same transfer-unit granularity (unless explicitly ablated)
    
    WARNINGS:
    - If denominator changes
    - If granularity changes
    """
    
    def __init__(
        self,
        base_config: ProSEXv2Config,
        workload_name: str = "default",
        seed: int = 42,
    ):
        """
        Initialize comparison runner.
        
        Args:
            base_config: Base configuration (frozen for comparison)
            workload_name: Name of workload (must be same across runs)
            seed: Random seed (must be same across runs)
        """
        self.base_config = base_config
        self.workload_name = workload_name
        self.seed = seed
        
        self.runs: List[ComparisonRun] = []
        self.results: List[ComparisonResult] = []
        
        logger.info(
            f"FairComparisonRunner initialized: "
            f"workload={workload_name}, seed={seed}"
        )
    
    def add_run(
        self,
        name: str,
        config_modifications: Dict[str, Any],
        description: str = "",
    ) -> "FairComparisonRunner":
        """
        Add a run to the comparison.
        
        Args:
            name: Name of this run (e.g., "baseline", "burst_disabled")
            config_modifications: Dict of config changes from base
            description: Human-readable description
            
        Returns:
            Self for chaining
        """
        # Deep copy base config
        config_dict = self.base_config.to_dict()
        
        # Apply modifications
        self._deep_update(config_dict, config_modifications)
        
        # Ensure critical parameters are preserved
        config_dict["seed"] = self.seed
        
        # Create config
        run_config = ProSEXv2Config.from_dict(config_dict)
        
        run = ComparisonRun(
            name=name,
            config=run_config,
            description=description,
            modifications=config_modifications,
        )
        
        self.runs.append(run)
        logger.info(f"Added comparison run: {name}")
        
        return self
    
    def run_all(
        self,
        run_fn: Callable[[ProSEXv2Config], SharedEvaluator],
    ) -> List[ComparisonResult]:
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
            
            # Check for issues before running
            self._validate_run(run)
            
            # Execute
            evaluator = run_fn(run.config)
            result = evaluator.evaluate()
            
            comparison_result = ComparisonResult(
                run_name=run.name,
                evaluator_result=result,
                metrics={
                    "conditional_recovery": result.conditional_recovery,
                    "no_miss_rate": result.no_miss_rate,
                    "upr_attention_based": result.upr_attention_based,
                    "upr_gold_based": result.upr_gold_based,
                    "budget_utilization": result.budget_utilization,
                    "latency_mean_ms": result.latency_mean_ms,
                    "latency_p95_ms": result.latency_p95_ms,
                    "total_promoted_bytes": result.total_promoted_bytes,
                    "total_steps": result.total_steps,
                    "steps_with_miss": result.steps_with_miss,
                }
            )
            
            self.results.append(comparison_result)
        
        return self.results
    
    def generate_comparison_table(self) -> str:
        """
        Generate markdown comparison table.
        
        Returns:
            Markdown formatted table
        """
        if not self.results:
            return "No results to display."
        
        lines = [
            "# Fair Comparison Results",
            "",
            f"**Workload:** {self.workload_name}",
            f"**Seed:** {self.seed}",
            f"**Timestamp:** {datetime.now().isoformat()}",
            "",
            "## Configuration Changes",
            "",
        ]
        
        for run in self.runs:
            lines.append(f"### {run.name}")
            lines.append(f"{run.description}")
            lines.append("")
            if run.modifications:
                lines.append("Modifications:")
                for key, val in run.modifications.items():
                    lines.append(f"- `{key}`: {val}")
            else:
                lines.append("(baseline - no modifications)")
            lines.append("")
        
        lines.extend([
            "## Metrics Comparison",
            "",
            "| Run | Cond. Rec. | No-Miss Rate | UPR (Attn) | Budget Util | Latency (ms) |",
            "|-----|-----------|--------------|------------|-------------|--------------|",
        ])
        
        for result in self.results:
            m = result.metrics
            lines.append(
                f"| {result.run_name} | "
                f"{m['conditional_recovery']:.3f} | "
                f"{m['no_miss_rate']:.3f} | "
                f"{m['upr_attention_based']:.3f} | "
                f"{m['budget_utilization']:.3f} | "
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
            lines.append("```json")
            lines.append(json.dumps(result.metrics, indent=2))
            lines.append("```")
            lines.append("")
        
        return "\n".join(lines)
    
    def save_comparison(
        self,
        output_dir: str,
        name: str = "comparison",
    ) -> Dict[str, str]:
        """
        Save comparison results.
        
        Args:
            output_dir: Directory to save results
            name: Base name for files
            
        Returns:
            Dict of file paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save markdown report
        md_path = Path(output_dir) / f"{name}.md"
        with open(md_path, 'w') as f:
            f.write(self.generate_comparison_table())
        saved_files["markdown"] = str(md_path)
        
        # Save CSV table
        csv_path = Path(output_dir) / f"{name}.csv"
        self._save_csv(csv_path)
        saved_files["csv"] = str(csv_path)
        
        # Save JSON summary
        json_path = Path(output_dir) / f"{name}.json"
        summary = {
            "workload": self.workload_name,
            "seed": self.seed,
            "timestamp": datetime.now().isoformat(),
            "runs": [
                {
                    "name": r.run_name,
                    "metrics": r.metrics,
                }
                for r in self.results
            ],
        }
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files["json"] = str(json_path)
        
        logger.info(f"Comparison saved to {output_dir}")
        return saved_files
    
    def _save_csv(self, path: Path) -> None:
        """Save comparison as CSV."""
        import csv
        
        if not self.results:
            return
        
        fieldnames = [
            "run_name",
            "conditional_recovery",
            "no_miss_rate",
            "upr_attention_based",
            "upr_gold_based",
            "budget_utilization",
            "latency_mean_ms",
            "latency_p95_ms",
            "total_promoted_bytes",
            "total_steps",
            "steps_with_miss",
        ]
        
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                row = {"run_name": result.run_name}
                row.update(result.metrics)
                writer.writerow(row)
    
    def _validate_run(self, run: ComparisonRun) -> None:
        """
        Validate a run for fair comparison.
        
        Checks for potential apples-to-oranges comparisons.
        """
        # Check seed
        if run.config.seed != self.seed:
            logger.warning(
                f"[{run.name}] Seed changed from {self.seed} to {run.config.seed}. "
                "This breaks fair comparison!"
            )
        
        # Check budget if modified
        base_budget = self.base_config.eabs.budget_bytes
        run_budget = run.config.eabs.budget_bytes
        if base_budget != run_budget:
            logger.warning(
                f"[{run.name}] Budget changed from {base_budget} to {run_budget}. "
                "Ensure this is intentional for ablation."
            )
        
        # Check granularity
        base_macro = self.base_config.macro_chunk_size
        run_macro = run.config.macro_chunk_size
        if base_macro != run_macro:
            logger.warning(
                f"[{run.name}] Macro chunk size changed from {base_macro} to {run_macro}. "
                "WARNING: Denominator changed - bytes per chunk different!"
            )
    
    def _deep_update(self, d: Dict, u: Dict) -> Dict:
        """Deep update dictionary d with values from u."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
        return d


def create_standard_comparisons(
    base_config: ProSEXv2Config,
    workload_name: str = "needle_in_haystack",
    seed: int = 42,
) -> FairComparisonRunner:
    """
    Create standard set of comparisons.
    
    Comparisons:
    1. Baseline (current patched system)
    2. Burst disabled
    3. Burst radius 2
    4. TTL 0 (no sticky)
    5. TTL 8 (long sticky)
    
    Args:
        base_config: Base configuration
        workload_name: Workload name
        seed: Random seed
        
    Returns:
        Configured FairComparisonRunner
    """
    runner = FairComparisonRunner(base_config, workload_name, seed)
    
    # 1. Baseline
    runner.add_run(
        name="baseline",
        config_modifications={},
        description="Current patched baseline",
    )
    
    # 2. Burst disabled
    runner.add_run(
        name="no_burst",
        config_modifications={
            "burst": {
                "enabled": False,
                "radius": 0,
            }
        },
        description="Burst expansion disabled",
    )
    
    # 3. Burst radius 2
    runner.add_run(
        name="burst_radius_2",
        config_modifications={
            "burst": {
                "enabled": True,
                "radius": 2,
            }
        },
        description="Burst radius increased to 2",
    )
    
    # 4. TTL 0 (no sticky)
    runner.add_run(
        name="ttl_0",
        config_modifications={
            "burst": {
                "sticky_enabled": False,
                "default_ttl": 0,
            }
        },
        description="Sticky residency disabled (TTL=0)",
    )
    
    # 5. TTL 8 (long sticky)
    runner.add_run(
        name="ttl_8",
        config_modifications={
            "burst": {
                "sticky_enabled": True,
                "default_ttl": 8,
            }
        },
        description="Extended sticky residency (TTL=8)",
    )
    
    return runner
