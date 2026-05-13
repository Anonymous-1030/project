"""
Burst Ablation Study for ProSE-X 2.0

Runs burst ablation for:
1. no_burst
2. burst_radius_1
3. burst_radius_2

Reports:
- Conditional Recovery
- No-Miss Rate
- UPR (all three modes)
- promoted bytes
- active bytes
- additional bytes caused by burst
- mean latency
- P95 latency
- Burst Gain per additional byte
"""

import json
import logging
import csv
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.config import ProSEXv2Config
from src.eval.shared.evaluator_v2 import SharedEvaluatorV2
from src.eval.fairness_guard import preflight_check

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BurstAblationResult:
    """Result of a burst ablation run."""
    config_name: str
    burst_radius: int
    burst_enabled: bool
    evaluator_result: Any  # EvaluationResult
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def conditional_recovery(self) -> float:
        return self.evaluator_result.conditional_recovery
    
    @property
    def promoted_bytes(self) -> int:
        return self.evaluator_result.total_promoted_bytes
    
    @property
    def active_bytes(self) -> int:
        return self.evaluator_result.total_used_bytes
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_name": self.config_name,
            "burst_radius": self.burst_radius,
            "burst_enabled": self.burst_enabled,
            "timestamp": self.timestamp,
            "metrics": {
                "conditional_recovery": self.evaluator_result.conditional_recovery,
                "no_miss_rate": self.evaluator_result.no_miss_rate,
                "upr_attention_based": self.evaluator_result.upr_attention_based,
                "upr_gold_based": self.evaluator_result.upr_gold_based,
                "upr_recovery_based": self.evaluator_result.upr_recovery_based,
                "promoted_bytes": self.promoted_bytes,
                "active_bytes": self.active_bytes,
                "budget_utilization": self.evaluator_result.budget_utilization,
                "latency_mean_ms": self.evaluator_result.latency_mean_ms,
                "latency_p95_ms": self.evaluator_result.latency_p95_ms,
            },
        }


class BurstAblationRunner:
    """
    Runner for burst ablation study.
    
    Goal: Determine whether burst is helping because it captures useful nearby evidence,
    not just because it moves more bytes.
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
        self.results: List[BurstAblationResult] = []
        
        logger.info("BurstAblationRunner initialized")
    
    def run_all(
        self,
        run_fn: Callable[[ProSEXv2Config], SharedEvaluatorV2],
    ) -> List[BurstAblationResult]:
        """Run all ablation configurations."""
        self.results = []
        
        configurations = [
            ("no_burst", 0, False),
            ("burst_radius_1", 1, True),
            ("burst_radius_2", 2, True),
        ]
        
        for config_name, radius, enabled in configurations:
            logger.info(f"Running burst ablation: {config_name}")
            
            # Create config
            config_dict = self.base_config.to_dict()
            config_dict["burst"] = {
                **config_dict.get("burst", {}),
                "enabled": enabled,
                "radius": radius,
            }
            config_dict["seed"] = self.seed
            
            config = ProSEXv2Config.from_dict(config_dict)
            
            # Preflight check
            base_dict = self.base_config.to_dict()
            fairness = preflight_check(base_dict, config_dict, config_name)
            if not fairness.can_compare:
                logger.warning(f"Fairness check failed for {config_name}")
            
            # Run
            evaluator = run_fn(config)
            result = evaluator.evaluate()
            
            ablation_result = BurstAblationResult(
                config_name=config_name,
                burst_radius=radius,
                burst_enabled=enabled,
                evaluator_result=result,
            )
            
            self.results.append(ablation_result)
            logger.info(
                f"Completed {config_name}: CR={result.conditional_recovery:.2%}, "
                f"Bytes={result.total_promoted_bytes:,}"
            )
        
        return self.results
    
    def compute_burst_gain(self) -> Dict[str, Any]:
        """
        Compute Burst Gain for each burst configuration vs no-burst baseline.
        
        Burst Gain = (recovery_with_burst - recovery_without_burst) / 
                     (bytes_with_burst - bytes_without_burst)
        
        Unit: Recovery percentage points per additional MB
        """
        if len(self.results) < 2:
            return {}
        
        # Find no-burst baseline
        no_burst_result = None
        for r in self.results:
            if r.config_name == "no_burst":
                no_burst_result = r
                break
        
        if no_burst_result is None:
            logger.error("No no_burst result found for burst gain computation")
            return {}
        
        baseline_recovery = no_burst_result.conditional_recovery
        baseline_bytes = no_burst_result.promoted_bytes
        
        gains = {}
        
        for result in self.results:
            if result.config_name == "no_burst":
                continue
            
            recovery_delta = result.conditional_recovery - baseline_recovery
            bytes_delta_mb = (result.promoted_bytes - baseline_bytes) / 1_000_000
            
            burst_gain = (
                recovery_delta / bytes_delta_mb
                if bytes_delta_mb > 0
                else 0.0
            )
            
            gains[result.config_name] = {
                "baseline_recovery": baseline_recovery,
                "comparison_recovery": result.conditional_recovery,
                "recovery_delta": recovery_delta,
                "baseline_bytes_mb": baseline_bytes / 1_000_000,
                "comparison_bytes_mb": result.promoted_bytes / 1_000_000,
                "bytes_delta_mb": bytes_delta_mb,
                "burst_gain_per_mb": burst_gain,
                "interpretation": (
                    "positive" if burst_gain > 0 else
                    "negative" if burst_gain < 0 else
                    "zero"
                ),
            }
        
        return gains
    
    def generate_markdown_report(self) -> str:
        """Generate markdown ablation report."""
        if not self.results:
            return "No results to display."
        
        lines = [
            "# Burst Ablation Report",
            "",
            f"**Workload:** {self.workload_name}",
            f"**Seed:** {self.seed}",
            f"**Timestamp:** {datetime.now().isoformat()}",
            "",
            "## Key Question",
            "",
            "Is burst helping because it captures useful nearby evidence, ",
            "or just because it moves more bytes?",
            "",
            "## Results Table",
            "",
            "| Configuration | Cond. Rec. | No-Miss Rate | UPR (Attn) | UPR (Rec) | Promoted Bytes | Latency (ms) |",
            "|--------------|-----------|--------------|------------|-----------|---------------|--------------|",
        ]
        
        for result in self.results:
            m = result.to_dict()["metrics"]
            lines.append(
                f"| {result.config_name} | "
                f"{m['conditional_recovery']:.2%} | "
                f"{m['no_miss_rate']:.2%} | "
                f"{m['upr_attention_based']:.2%} | "
                f"{m['upr_recovery_based']:.2%} | "
                f"{m['promoted_bytes']:,} | "
                f"{m['latency_mean_ms']:.2f} |"
            )
        
        # Burst Gain section
        gains = self.compute_burst_gain()
        if gains:
            lines.extend([
                "",
                "## Burst Gain Analysis",
                "",
                "| Configuration | Recovery Δ | Bytes Δ (MB) | Burst Gain (/MB) | Interpretation |",
                "|--------------|-----------|--------------|-----------------|----------------|",
            ])
            
            for config_name, gain_data in gains.items():
                lines.append(
                    f"| {config_name} | "
                    f"{gain_data['recovery_delta']:+.2%} | "
                    f"{gain_data['bytes_delta_mb']:+.2f} | "
                    f"{gain_data['burst_gain_per_mb']:+.4f} | "
                    f"{gain_data['interpretation']} |"
                )
            
            lines.extend([
                "",
                "### Interpretation",
                "",
                "- **Positive Burst Gain**: Burst helps recovery more than it costs in bytes",
                "- **Negative Burst Gain**: Burst hurts (costs more than helps)",
                "- **Zero Burst Gain**: Burst has no effect on recovery",
                "",
            ])
        
        lines.extend([
            "",
            "## Detailed Metrics",
            "",
        ])
        
        for result in self.results:
            m = result.to_dict()["metrics"]
            lines.append(f"### {result.config_name}")
            lines.append("")
            lines.append(f"- **Conditional Recovery:** {m['conditional_recovery']:.2%}")
            lines.append(f"- **No-Miss Rate:** {m['no_miss_rate']:.2%}")
            lines.append(f"- **UPR (Attention-based):** {m['upr_attention_based']:.2%}")
            lines.append(f"- **UPR (Gold-based):** {m['upr_gold_based']:.2%}")
            lines.append(f"- **UPR (Recovery-based):** {m['upr_recovery_based']:.2%}")
            lines.append(f"- **Promoted Bytes:** {m['promoted_bytes']:,}")
            lines.append(f"- **Active Bytes:** {m['active_bytes']:,}")
            lines.append(f"- **Budget Utilization:** {m['budget_utilization']:.2%}")
            lines.append(f"- **Mean Latency:** {m['latency_mean_ms']:.2f} ms")
            lines.append(f"- **P95 Latency:** {m['latency_p95_ms']:.2f} ms")
            lines.append("")
        
        return "\n".join(lines)
    
    def save_csv(self, output_path: str) -> None:
        """Save ablation results as CSV."""
        if not self.results:
            return
        
        fieldnames = [
            "config_name",
            "burst_radius",
            "burst_enabled",
            "conditional_recovery",
            "no_miss_rate",
            "upr_attention_based",
            "upr_gold_based",
            "upr_recovery_based",
            "promoted_bytes",
            "active_bytes",
            "budget_utilization",
            "latency_mean_ms",
            "latency_p95_ms",
        ]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    "config_name": result.config_name,
                    "burst_radius": result.burst_radius,
                    "burst_enabled": result.burst_enabled,
                }
                row.update(result.to_dict()["metrics"])
                writer.writerow(row)
        
        logger.info(f"CSV saved to {output_path}")
    
    def save_json(self, output_path: str) -> None:
        """Save ablation results as JSON."""
        summary = {
            "workload": self.workload_name,
            "seed": self.seed,
            "timestamp": datetime.now().isoformat(),
            "runs": [result.to_dict() for result in self.results],
            "burst_gain_analysis": self.compute_burst_gain(),
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"JSON saved to {output_path}")
    
    def save_all(self, output_dir: str, name: str = "burst_ablation") -> Dict[str, str]:
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
        
        return saved


def create_burst_ablation_runner(
    base_config: ProSEXv2Config,
    workload_name: str = "needle_in_haystack",
    seed: int = 42,
) -> BurstAblationRunner:
    """Create standard burst ablation runner."""
    return BurstAblationRunner(base_config, workload_name, seed)


def generate_burst_ablation_report(
    results: List[BurstAblationResult],
    output_dir: str = "outputs/reports",
) -> Dict[str, Any]:
    """Generate burst ablation report from results."""
    runner = BurstAblationRunner(
        base_config=ProSEXv2Config(),
        workload_name="unknown",
        seed=42,
    )
    runner.results = results
    
    return runner.save_all(output_dir, "burst_ablation")
