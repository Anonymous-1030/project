"""
Fairness Guard for ProSE-X 2.0

Prevents unfair comparisons by checking for config mismatches.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FairnessWarning:
    """A single fairness warning."""
    check_name: str
    severity: str  # "ERROR", "WARNING", "INFO"
    message: str
    baseline_value: Any = None
    comparison_value: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "severity": self.severity,
            "message": self.message,
            "baseline_value": self.baseline_value,
            "comparison_value": self.comparison_value,
        }


@dataclass
class FairnessReport:
    """Complete fairness report."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    comparison_name: str = ""
    can_compare: bool = True
    warnings: List[FairnessWarning] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "comparison_name": self.comparison_name,
            "can_compare": self.can_compare,
            "summary": {
                "errors": sum(1 for w in self.warnings if w.severity == "ERROR"),
                "warnings": sum(1 for w in self.warnings if w.severity == "WARNING"),
                "infos": sum(1 for w in self.warnings if w.severity == "INFO"),
            },
            "warnings": [w.to_dict() for w in self.warnings],
        }
    
    def save(self, output_path: str) -> None:
        """Save report to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Fairness report saved to {output_path}")


class FairnessGuard:
    """
    Guards against unfair comparisons.
    
    Checks:
    1. Same workload?
    2. Same seed?
    3. Same budget?
    4. Same retention settings?
    5. Same granularity?
    6. Same accounting mode?
    """
    
    def __init__(
        self,
        baseline_config: Dict[str, Any],
        comparison_config: Dict[str, Any],
        comparison_name: str = "comparison",
    ):
        self.baseline = baseline_config
        self.comparison = comparison_config
        self.comparison_name = comparison_name
        self.report = FairnessReport(comparison_name=comparison_name)
        
        logger.info(f"FairnessGuard initialized for: {comparison_name}")
    
    def check_same_workload(self) -> bool:
        """Check if workload is the same."""
        baseline_workload = self.baseline.get("workload") or self.baseline.get("experiment_name")
        comparison_workload = self.comparison.get("workload") or self.comparison.get("experiment_name")
        
        if baseline_workload != comparison_workload:
            warning = FairnessWarning(
                check_name="same_workload",
                severity="ERROR",
                message=(
                    f"Workload mismatch: baseline='{baseline_workload}', "
                    f"comparison='{comparison_workload}'. "
                    "Results are not comparable!"
                ),
                baseline_value=baseline_workload,
                comparison_value=comparison_workload,
            )
            self.report.warnings.append(warning)
            self.report.can_compare = False
            logger.error(f"[Fairness] {warning.message}")
            return False
        
        return True
    
    def check_same_seed(self) -> bool:
        """Check if seed is the same."""
        baseline_seed = self.baseline.get("seed")
        comparison_seed = self.comparison.get("seed")
        
        if baseline_seed is None or comparison_seed is None:
            warning = FairnessWarning(
                check_name="same_seed",
                severity="WARNING",
                message="Seed not specified in one or both configs. Randomness may differ.",
                baseline_value=baseline_seed,
                comparison_value=comparison_seed,
            )
            self.report.warnings.append(warning)
            logger.warning(f"[Fairness] {warning.message}")
            return False
        
        if baseline_seed != comparison_seed:
            warning = FairnessWarning(
                check_name="same_seed",
                severity="ERROR",
                message=(
                    f"Seed mismatch: baseline={baseline_seed}, "
                    f"comparison={comparison_seed}. "
                    "Random variations will confound results!"
                ),
                baseline_value=baseline_seed,
                comparison_value=comparison_seed,
            )
            self.report.warnings.append(warning)
            self.report.can_compare = False
            logger.error(f"[Fairness] {warning.message}")
            return False
        
        return True
    
    def check_same_budget(self) -> bool:
        """Check if budget is the same."""
        baseline_budget = self.baseline.get("eabs", {}).get("budget_bytes")
        comparison_budget = self.comparison.get("eabs", {}).get("budget_bytes")
        
        baseline_ratio = self.baseline.get("eabs", {}).get("budget_ratio_of_tail")
        comparison_ratio = self.comparison.get("eabs", {}).get("budget_ratio_of_tail")
        
        # If both have explicit budget_bytes, check those
        if baseline_budget is not None and comparison_budget is not None:
            if baseline_budget != comparison_budget:
                warning = FairnessWarning(
                    check_name="same_budget",
                    severity="WARNING",
                    message=(
                        f"Budget mismatch: baseline={baseline_budget}, "
                        f"comparison={comparison_budget}. "
                        "Different budgets will affect promotion behavior."
                    ),
                    baseline_value=baseline_budget,
                    comparison_value=comparison_budget,
                )
                self.report.warnings.append(warning)
                logger.warning(f"[Fairness] {warning.message}")
                return False
        
        # Otherwise check budget_ratio_of_tail
        elif baseline_ratio != comparison_ratio:
            warning = FairnessWarning(
                check_name="same_budget_ratio",
                severity="WARNING",
                message=(
                    f"Budget ratio mismatch: baseline={baseline_ratio}, "
                    f"comparison={comparison_ratio}. "
                    "Different budget ratios may affect results."
                ),
                baseline_value=baseline_ratio,
                comparison_value=comparison_ratio,
            )
            self.report.warnings.append(warning)
            logger.warning(f"[Fairness] {warning.message}")
            return False
        
        return True
    
    def check_same_retention_settings(self) -> bool:
        """Check if retention settings are the same."""
        baseline_anchor = self.baseline.get("anchor_ratio")
        comparison_anchor = self.comparison.get("anchor_ratio")
        
        baseline_compression = self.baseline.get("tail_compression_ratio")
        comparison_compression = self.comparison.get("tail_compression_ratio")
        
        checks_passed = True
        
        if baseline_anchor != comparison_anchor:
            warning = FairnessWarning(
                check_name="same_anchor_ratio",
                severity="WARNING",
                message=(
                    f"Anchor ratio mismatch: baseline={baseline_anchor}, "
                    f"comparison={comparison_anchor}. "
                    "Different anchor ratios affect recall."
                ),
                baseline_value=baseline_anchor,
                comparison_value=comparison_anchor,
            )
            self.report.warnings.append(warning)
            checks_passed = False
        
        if baseline_compression != comparison_compression:
            warning = FairnessWarning(
                check_name="same_compression_ratio",
                severity="INFO",
                message=(
                    f"Compression ratio mismatch: baseline={baseline_compression}, "
                    f"comparison={comparison_compression}. "
                    "May affect tail recall."
                ),
                baseline_value=baseline_compression,
                comparison_value=comparison_compression,
            )
            self.report.warnings.append(warning)
        
        return checks_passed
    
    def check_same_granularity(self) -> bool:
        """Check if chunk granularity is the same."""
        baseline_macro = self.baseline.get("macro_chunk_size")
        comparison_macro = self.comparison.get("macro_chunk_size")
        
        baseline_micro = self.baseline.get("micro_chunk_size")
        comparison_micro = self.comparison.get("micro_chunk_size")
        
        checks_passed = True
        
        if baseline_macro != comparison_macro:
            warning = FairnessWarning(
                check_name="same_granularity",
                severity="ERROR",
                message=(
                    f"Macro chunk size mismatch: baseline={baseline_macro}, "
                    f"comparison={comparison_macro}. "
                    "WARNING: Denominator changed - bytes per chunk different!"
                ),
                baseline_value=baseline_macro,
                comparison_value=comparison_macro,
            )
            self.report.warnings.append(warning)
            self.report.can_compare = False
            logger.error(f"[Fairness] {warning.message}")
            checks_passed = False
        
        if baseline_micro != comparison_micro:
            warning = FairnessWarning(
                check_name="same_micro_granularity",
                severity="WARNING",
                message=(
                    f"Micro chunk size mismatch: baseline={baseline_micro}, "
                    f"comparison={comparison_micro}. "
                    "Indexing granularity differs."
                ),
                baseline_value=baseline_micro,
                comparison_value=comparison_micro,
            )
            self.report.warnings.append(warning)
        
        return checks_passed
    
    def check_same_accounting_mode(self) -> bool:
        """Check if evaluation accounting mode is the same."""
        # This is typically implicit in the evaluation config
        # For now, we just check that both configs have evaluation enabled
        baseline_eval = self.baseline.get("evaluation", {})
        comparison_eval = self.comparison.get("evaluation", {})
        
        baseline_enabled = baseline_eval.get("compute_end_metrics", True)
        comparison_enabled = comparison_eval.get("compute_end_metrics", True)
        
        if baseline_enabled != comparison_enabled:
            warning = FairnessWarning(
                check_name="same_accounting_mode",
                severity="WARNING",
                message=(
                    f"Evaluation mode mismatch: baseline metrics={baseline_enabled}, "
                    f"comparison metrics={comparison_enabled}."
                ),
                baseline_value=baseline_enabled,
                comparison_value=comparison_enabled,
            )
            self.report.warnings.append(warning)
            return False
        
        return True
    
    def run_all_checks(self) -> FairnessReport:
        """Run all fairness checks."""
        self.check_same_workload()
        self.check_same_seed()
        self.check_same_budget()
        self.check_same_retention_settings()
        self.check_same_granularity()
        self.check_same_accounting_mode()
        
        return self.report
    
    def validate_or_raise(self) -> None:
        """
        Run checks and raise exception if comparison is not valid.
        """
        self.run_all_checks()
        
        if not self.report.can_compare:
            errors = [w for w in self.report.warnings if w.severity == "ERROR"]
            error_msg = "\n".join([e.message for e in errors])
            raise ValueError(f"Fairness check failed. Cannot compare configs:\n{error_msg}")


def preflight_check(
    baseline_config: Dict[str, Any],
    comparison_config: Dict[str, Any],
    comparison_name: str = "comparison",
    output_path: Optional[str] = None,
    raise_on_error: bool = False,
) -> FairnessReport:
    """
    Run preflight fairness check.
    
    Args:
        baseline_config: Baseline configuration
        comparison_config: Comparison configuration
        comparison_name: Name of the comparison
        output_path: Optional path to save report
        raise_on_error: If True, raise exception on fairness errors
        
    Returns:
        FairnessReport
    """
    guard = FairnessGuard(
        baseline_config=baseline_config,
        comparison_config=comparison_config,
        comparison_name=comparison_name,
    )
    
    report = guard.run_all_checks()
    
    if output_path:
        report.save(output_path)
    
    if raise_on_error and not report.can_compare:
        guard.validate_or_raise()
    
    return report
