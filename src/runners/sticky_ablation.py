"""
Sticky TTL Ablation Study for ProSE-X 2.0

Runs sticky TTL ablation for:
1. ttl_0 (no sticky)
2. ttl_2
3. ttl_4
4. ttl_8

Reports:
- Conditional Recovery
- No-Miss Rate
- UPR (all three modes)
- re-promotion count
- eviction count
- average residency
- promoted bytes
- mean latency
- P95 latency

Goal: Prove whether sticky reduces oscillation and redundant re-promotion,
rather than merely increasing residency.
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
class StickyAblationResult:
    """Result of a sticky TTL ablation run."""
    config_name: str
    ttl_value: int
    sticky_enabled: bool
    evaluator_result: Any  # EvaluationResult
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    re_promotion_count: int = 0
    eviction_count: int = 0
    average_residency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_name": self.config_name,
            "ttl_value": self.ttl_value,
            "sticky_enabled": self.sticky_enabled,
            "timestamp": self.timestamp,
            "metrics": {
                "conditional_recovery": self.evaluator_result.conditional_recovery,
                "no_miss_rate": self.evaluator_result.no_miss_rate,
                "upr_attention_based": self.evaluator_result.upr_attention_based,
                "upr_gold_based": self.evaluator_result.upr_gold_based,
                "upr_recovery_based": self.evaluator_result.upr_recovery_based,
                "promoted_bytes": self.evaluator_result.total_promoted_bytes,
                "budget_utilization": self.evaluator_result.budget_utilization,
                "latency_mean_ms": self.evaluator_result.latency_mean_ms,
                "latency_p95_ms": self.evaluator_result.latency_p95_ms,
            },
            "sticky_metrics": {
                "re_promotion_count": self.re_promotion_count,
                "eviction_count": self.eviction_count,
                "average_residency": self.average_residency,
            },
        }


class StickyAblationRunner:
    """
    Runner for sticky TTL ablation study.
    
    Goal: Prove whether sticky reduces oscillation and redundant re-promotion,
    rather than merely increasing residency.
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
        self.results: List[StickyAblationResult] = []
        
        logger.info("StickyAblationRunner initialized")
    
    def run_all(
        self,
        run_fn: Callable[[ProSEXv2Config], SharedEvaluatorV2],
    ) -> List[StickyAblationResult]:
        """Run all ablation configurations."""
        self.results = []
        
        configurations = [
            ("ttl_0", 0, False),
            ("ttl_2", 2, True),
            ("ttl_4", 4, True),
            ("ttl_8", 8, True),
        ]
        
        for config_name, ttl, enabled in configurations:
            logger.info(f"Running sticky ablation: {config_name}")
            
            # Create config
            config_dict = self.base_config.to_dict()
            config_dict["burst"] = {
                **config_dict.get("burst", {}),
                "sticky_enabled": enabled,
                "default_ttl": ttl,
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
            
            # Compute sticky metrics from unit traces
            re_promotions, evictions, avg_residency = self._compute_sticky_metrics(
                result.promoted_unit_traces
            )
            
            ablation_result = StickyAblationResult(
                config_name=config_name,
                ttl_value=ttl,
                sticky_enabled=enabled,
                evaluator_result=result,
                re_promotion_count=re_promotions,
                eviction_count=evictions,
                average_residency=avg_residency,
            )
            
            self.results.append(ablation_result)
            logger.info(
                f"Completed {config_name}: CR={result.conditional_recovery:.2%}, "
                f"Re-promotions={re_promotions}, Avg Residency={avg_residency:.1f}"
            )
        
        return self.results
    
    def _compute_sticky_metrics(self, unit_traces: List[Any]) -> tuple:
        """
        Compute sticky metrics from promoted unit traces.
        
        Returns:
            (re_promotion_count, eviction_count, average_residency)
        """
        if not unit_traces:
            return 0, 0, 0.0
        
        # Count unique chunks and total promotions
        chunk_promotions: Dict[str, List[int]] = {}
        
        for trace in unit_traces:
            chunk_id = trace.chunk_id
            step = trace.step_promoted
            
            if chunk_id not in chunk_promotions:
                chunk_promotions[chunk_id] = []
            chunk_promotions[chunk_id].append(step)
        
        # Re-promotions = total promotions - unique chunks
        total_promotions = len(unit_traces)
        unique_chunks = len(chunk_promotions)
        re_promotions = total_promotions - unique_chunks
        
        # Count evictions
        evictions = sum(
            1 for trace in unit_traces
            if trace.eviction_reason is not None
        )
        
        # Average residency
        total_residency = 0
        residency_count = 0
        
        for trace in unit_traces:
            if trace.eviction_step is not None:
                residency = trace.eviction_step - trace.step_promoted
                total_residency += residency
                residency_count += 1
            elif trace.ttl_original > 0:
                # Still active, use TTL as proxy
                total_residency += trace.ttl_original
                residency_count += 1
        
        avg_residency = total_residency / max(residency_count, 1)
        
        return re_promotions, evictions, avg_residency
    
    def analyze_anti_oscillation(self) -> Dict[str, Any]:
        """
        Analyze whether sticky TTL reduces oscillation.
        
        Key indicators:
        - Lower re-promotion count with higher TTL
        - More stable recovery (less variance across steps)
        - Higher average residency
        """
        if len(self.results) < 2:
            return {}
        
        # Sort by TTL
        sorted_results = sorted(self.results, key=lambda r: r.ttl_value)
        
        baseline = sorted_results[0]  # ttl_0
        
        analysis = {
            "baseline_ttl_0": {
                "re_promotions": baseline.re_promotion_count,
                "evictions": baseline.eviction_count,
                "avg_residency": baseline.average_residency,
            },
            "comparisons": {},
        }
        
        for result in sorted_results[1:]:
            analysis["comparisons"][result.config_name] = {
                "re_promotion_delta": result.re_promotion_count - baseline.re_promotion_count,
                "re_promotion_ratio": (
                    result.re_promotion_count / max(baseline.re_promotion_count, 1)
                ),
                "eviction_delta": result.eviction_count - baseline.eviction_count,
                "residency_delta": result.average_residency - baseline.average_residency,
                "anti_oscillation_indicator": (
                    "positive" if result.re_promotion_count < baseline.re_promotion_count
                    else "negative"
                ),
            }
        
        return analysis
    
    def generate_markdown_report(self) -> str:
        """Generate markdown ablation report."""
        if not self.results:
            return "No results to display."
        
        lines = [
            "# Sticky TTL Ablation Report",
            "",
            f"**Workload:** {self.workload_name}",
            f"**Seed:** {self.seed}",
            f"**Timestamp:** {datetime.now().isoformat()}",
            "",
            "## Key Question",
            "",
            "Does sticky TTL reduce oscillation and redundant re-promotion,",
            "rather than merely increasing residency?",
            "",
            "## Results Table",
            "",
            "| Configuration | Cond. Rec. | No-Miss Rate | UPR (Attn) | UPR (Rec) | Re-Promotions | Avg Residency |",
            "|--------------|-----------|--------------|------------|-----------|---------------|---------------|",
        ]
        
        for result in self.results:
            m = result.to_dict()["metrics"]
            s = result.to_dict()["sticky_metrics"]
            lines.append(
                f"| {result.config_name} | "
                f"{m['conditional_recovery']:.2%} | "
                f"{m['no_miss_rate']:.2%} | "
                f"{m['upr_attention_based']:.2%} | "
                f"{m['upr_recovery_based']:.2%} | "
                f"{s['re_promotion_count']} | "
                f"{s['average_residency']:.1f} |"
            )
        
        # Anti-oscillation analysis
        analysis = self.analyze_anti_oscillation()
        if analysis:
            lines.extend([
                "",
                "## Anti-Oscillation Analysis",
                "",
                f"**Baseline (TTL=0):** Re-promotions={analysis['baseline_ttl_0']['re_promotions']}, "
                f"Avg Residency={analysis['baseline_ttl_0']['avg_residency']:.1f}",
                "",
                "| Configuration | Re-Promo Δ | Re-Promo Ratio | Residency Δ | Indicator |",
                "|--------------|-----------|----------------|-------------|-----------|",
            ])
            
            for config_name, comp in analysis["comparisons"].items():
                indicator = "✓ Anti-oscillation" if comp["anti_oscillation_indicator"] == "positive" else "✗ More oscillation"
                lines.append(
                    f"| {config_name} | "
                    f"{comp['re_promotion_delta']:+d} | "
                    f"{comp['re_promotion_ratio']:.2f}x | "
                    f"{comp['residency_delta']:+.1f} | "
                    f"{indicator} |"
                )
            
            lines.extend([
                "",
                "### Interpretation",
                "",
                "- **Re-Promo Δ**: Negative is better (fewer re-promotions)",
                "- **Re-Promo Ratio**: < 1.0 means fewer re-promotions than baseline",
                "- **Residency Δ**: Higher means longer average stay in promoted set",
                "- **Anti-oscillation**: Positive if TTL reduces re-promotions",
                "",
            ])
        
        lines.extend([
            "",
            "## Detailed Metrics",
            "",
        ])
        
        for result in self.results:
            m = result.to_dict()["metrics"]
            s = result.to_dict()["sticky_metrics"]
            
            lines.append(f"### {result.config_name}")
            lines.append("")
            lines.append(f"- **Conditional Recovery:** {m['conditional_recovery']:.2%}")
            lines.append(f"- **No-Miss Rate:** {m['no_miss_rate']:.2%}")
            lines.append(f"- **UPR (Attention-based):** {m['upr_attention_based']:.2%}")
            lines.append(f"- **UPR (Gold-based):** {m['upr_gold_based']:.2%}")
            lines.append(f"- **UPR (Recovery-based):** {m['upr_recovery_based']:.2%}")
            lines.append(f"- **Promoted Bytes:** {m['promoted_bytes']:,}")
            lines.append(f"- **Mean Latency:** {m['latency_mean_ms']:.2f} ms")
            lines.append(f"- **P95 Latency:** {m['latency_p95_ms']:.2f} ms")
            lines.append(f"- **Re-Promotion Count:** {s['re_promotion_count']}")
            lines.append(f"- **Eviction Count:** {s['eviction_count']}")
            lines.append(f"- **Average Residency:** {s['average_residency']:.1f} steps")
            lines.append("")
        
        return "\n".join(lines)
    
    def save_csv(self, output_path: str) -> None:
        """Save ablation results as CSV."""
        if not self.results:
            return
        
        fieldnames = [
            "config_name",
            "ttl_value",
            "sticky_enabled",
            "conditional_recovery",
            "no_miss_rate",
            "upr_attention_based",
            "upr_gold_based",
            "upr_recovery_based",
            "promoted_bytes",
            "budget_utilization",
            "latency_mean_ms",
            "latency_p95_ms",
            "re_promotion_count",
            "eviction_count",
            "average_residency",
        ]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    "config_name": result.config_name,
                    "ttl_value": result.ttl_value,
                    "sticky_enabled": result.sticky_enabled,
                }
                row.update(result.to_dict()["metrics"])
                row.update(result.to_dict()["sticky_metrics"])
                writer.writerow(row)
        
        logger.info(f"CSV saved to {output_path}")
    
    def save_json(self, output_path: str) -> None:
        """Save ablation results as JSON."""
        summary = {
            "workload": self.workload_name,
            "seed": self.seed,
            "timestamp": datetime.now().isoformat(),
            "runs": [result.to_dict() for result in self.results],
            "anti_oscillation_analysis": self.analyze_anti_oscillation(),
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"JSON saved to {output_path}")
    
    def save_all(self, output_dir: str, name: str = "sticky_ablation") -> Dict[str, str]:
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


def create_sticky_ablation_runner(
    base_config: ProSEXv2Config,
    workload_name: str = "needle_in_haystack",
    seed: int = 42,
) -> StickyAblationRunner:
    """Create standard sticky ablation runner."""
    return StickyAblationRunner(base_config, workload_name, seed)


def generate_sticky_ablation_report(
    results: List[StickyAblationResult],
    output_dir: str = "outputs/reports",
) -> Dict[str, Any]:
    """Generate sticky ablation report from results."""
    runner = StickyAblationRunner(
        base_config=ProSEXv2Config(),
        workload_name="unknown",
        seed=42,
    )
    runner.results = results
    
    return runner.save_all(output_dir, "sticky_ablation")
