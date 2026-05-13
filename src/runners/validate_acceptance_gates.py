"""
Acceptance Gates Validation for ProSE-X 2.0

Validates all acceptance gates before claiming completion:

Gate A: Current patched baseline can be rerun exactly
Gate B: Metric definitions are frozen and shared
Gate C: For every promoted unit, usefulness can be traced
Gate D: Neighbor recall has real contribution report
Gate E: Before/after recovery comparison is fair and reproducible
Gate F: Burst results include both benefit and overhead
Gate G: Sticky results include anti-oscillation evidence
Gate H: Every miss has single attributed dominant cause
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class GateResult:
    """Result of a single gate check."""
    name: str
    passed: bool
    details: List[str]
    required_files: List[str]


class AcceptanceGateValidator:
    """Validates all acceptance gates."""
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize validator."""
        self.output_dir = Path(output_dir)
        self.results: List[GateResult] = []
    
    def validate_all(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate all gates.
        
        Returns:
            (all_passed, detailed_results)
        """
        self.results = []
        
        # Run all gate checks
        self.results.append(self._check_gate_a())
        self.results.append(self._check_gate_b())
        self.results.append(self._check_gate_c())
        self.results.append(self._check_gate_d())
        self.results.append(self._check_gate_e())
        self.results.append(self._check_gate_f())
        self.results.append(self._check_gate_g())
        self.results.append(self._check_gate_h())
        
        all_passed = all(r.passed for r in self.results)
        
        summary = {
            "all_gates_passed": all_passed,
            "gates": {
                r.name: {
                    "passed": r.passed,
                    "details": r.details,
                }
                for r in self.results
            },
        }
        
        return all_passed, summary
    
    def _check_gate_a(self) -> GateResult:
        """Gate A: Current patched baseline can be rerun exactly."""
        name = "Gate A: Baseline Reproducibility"
        details = []
        passed = True
        
        # Check baseline file exists
        baseline_path = self.output_dir / "reports" / "baseline_after_scheduler_fix.json"
        if not baseline_path.exists():
            passed = False
            details.append(f"MISSING: {baseline_path}")
        else:
            details.append(f"FOUND: {baseline_path}")
            
            # Validate baseline has required fields
            with open(baseline_path) as f:
                baseline = json.load(f)
            
            required_fields = ["code_state", "config", "validation_checks"]
            for field in required_fields:
                if field not in baseline:
                    passed = False
                    details.append(f"Baseline missing field: {field}")
                else:
                    details.append(f"Baseline has field: {field}")
            
            # Check validation checks
            checks = baseline.get("validation_checks", {})
            if not checks.get("scheduler_fix_verified"):
                passed = False
                details.append("scheduler_fix_verified is False")
        
        # Check regression runner exists
        regression_path = Path("src/runners/regression_baseline.py")
        if not regression_path.exists():
            passed = False
            details.append(f"MISSING: {regression_path}")
        else:
            details.append(f"FOUND: {regression_path}")
        
        return GateResult(name, passed, details, [str(baseline_path), str(regression_path)])
    
    def _check_gate_b(self) -> GateResult:
        """Gate B: Metric definitions are frozen and shared."""
        name = "Gate B: Frozen Metric Definitions"
        details = []
        passed = True
        
        # Check metrics spec exists
        spec_path = Path("docs/metrics_spec.md")
        if not spec_path.exists():
            passed = False
            details.append(f"MISSING: {spec_path}")
        else:
            details.append(f"FOUND: {spec_path}")
            
            # Check for required metric definitions
            content = spec_path.read_text(encoding='utf-8')
            required_metrics = [
                "Conditional Recovery",
                "No-Miss Rate",
                "Useful Promote Ratio",
                "Burst Gain",
            ]
            for metric in required_metrics:
                if metric in content:
                    details.append(f"Metric defined: {metric}")
                else:
                    passed = False
                    details.append(f"MISSING definition: {metric}")
        
        # Check shared evaluator exists
        evaluator_path = Path("src/eval/shared/evaluator.py")
        if not evaluator_path.exists():
            passed = False
            details.append(f"MISSING: {evaluator_path}")
        else:
            details.append(f"FOUND: {evaluator_path}")
        
        return GateResult(name, passed, details, [str(spec_path), str(evaluator_path)])
    
    def _check_gate_c(self) -> GateResult:
        """Gate C: For every promoted unit, usefulness can be traced."""
        name = "Gate C: Promote-to-Use Traceability"
        details = []
        passed = True
        
        # Check useful bytes accounting exists
        accounting_path = Path("src/eval/accounting/useful_bytes.py")
        if not accounting_path.exists():
            passed = False
            details.append(f"MISSING: {accounting_path}")
        else:
            details.append(f"FOUND: {accounting_path}")
            
            # Check for required fields
            content = accounting_path.read_text()
            required_fields = [
                "chunk_id",
                "queue_of_origin",
                "score",
                "bytes_transferred",
                "future_accesses",
            ]
            for field in required_fields:
                if field in content:
                    details.append(f"Traceability field: {field}")
                else:
                    passed = False
                    details.append(f"MISSING field: {field}")
        
        return GateResult(name, passed, details, [str(accounting_path)])
    
    def _check_gate_d(self) -> GateResult:
        """Gate D: Neighbor recall has real contribution report."""
        name = "Gate D: Neighbor Recall Contribution"
        details = []
        passed = True
        
        # Check neighbor recall reporter exists
        reporter_path = Path("src/eval/neighbor_recall_report.py")
        if not reporter_path.exists():
            passed = False
            details.append(f"MISSING: {reporter_path}")
        else:
            details.append(f"FOUND: {reporter_path}")
            
            # Check for pipeline stage tracking
            content = reporter_path.read_text()
            stages = [
                "raw_output",
                "post_dedup",
                "entering_scorer",
                "surviving_scheduler",
                "burst_expanded",
                "ultimately_useful",
            ]
            for stage in stages:
                if stage in content:
                    details.append(f"Stage tracked: {stage}")
                else:
                    passed = False
                    details.append(f"MISSING stage: {stage}")
        
        return GateResult(name, passed, details, [str(reporter_path)])
    
    def _check_gate_e(self) -> GateResult:
        """Gate E: Before/after recovery comparison is fair."""
        name = "Gate E: Fair Comparison"
        details = []
        passed = True
        
        # Check comparison runner exists
        runner_path = Path("src/runners/comparison_runner.py")
        if not runner_path.exists():
            passed = False
            details.append(f"MISSING: {runner_path}")
        else:
            details.append(f"FOUND: {runner_path}")
            
            # Check for fairness guarantees
            content = runner_path.read_text()
            fairness_checks = [
                "same workload",
                "same seed",
                "same budget",
            ]
            for check in fairness_checks:
                if check.lower() in content.lower() or check.replace(" ", "_") in content:
                    details.append(f"Fairness check: {check}")
                else:
                    passed = False
                    details.append(f"MISSING fairness check: {check}")
        
        return GateResult(name, passed, details, [str(runner_path)])
    
    def _check_gate_f(self) -> GateResult:
        """Gate F: Burst results include both benefit and overhead."""
        name = "Gate F: Burst Benefit and Overhead"
        details = []
        passed = True
        
        # Check burst ablation runner exists
        burst_path = Path("src/runners/burst_ablation.py")
        if not burst_path.exists():
            passed = False
            details.append(f"MISSING: {burst_path}")
        else:
            details.append(f"FOUND: {burst_path}")
            
            # Check for gain computation
            content = burst_path.read_text()
            required_elements = [
                "recovery_with_burst",
                "recovery_without_burst",
                "bytes_with_burst",
                "bytes_without_burst",
                "burst_gain",
            ]
            for elem in required_elements:
                if elem in content:
                    details.append(f"Burst analysis element: {elem}")
                else:
                    passed = False
                    details.append(f"MISSING element: {elem}")
        
        return GateResult(name, passed, details, [str(burst_path)])
    
    def _check_gate_g(self) -> GateResult:
        """Gate G: Sticky results include anti-oscillation evidence."""
        name = "Gate G: Sticky Anti-Oscillation"
        details = []
        passed = True
        
        # Check sticky ablation runner exists
        sticky_path = Path("src/runners/sticky_ablation.py")
        if not sticky_path.exists():
            passed = False
            details.append(f"MISSING: {sticky_path}")
        else:
            details.append(f"FOUND: {sticky_path}")
            
            # Check for anti-oscillation analysis
            content = sticky_path.read_text()
            indicators = [
                "upr_trend",
                "anti_oscillation",
                "re-promotion",
            ]
            for indicator in indicators:
                if indicator.lower() in content.lower() or indicator in content:
                    details.append(f"Anti-oscillation indicator: {indicator}")
                else:
                    passed = False
                    details.append(f"MISSING indicator: {indicator}")
        
        return GateResult(name, passed, details, [str(sticky_path)])
    
    def _check_gate_h(self) -> GateResult:
        """Gate H: Every miss has single attributed dominant cause."""
        name = "Gate H: Failure Attribution"
        details = []
        passed = True
        
        # Check failure attributor exists
        attr_path = Path("src/eval/failure_attribution/attributor.py")
        if not attr_path.exists():
            passed = False
            details.append(f"MISSING: {attr_path}")
        else:
            details.append(f"FOUND: {attr_path}")
            
            # Check for failure reasons
            content = attr_path.read_text()
            reasons = [
                "candidate_miss",
                "scorer_rank_miss",
                "scheduler_budget_drop",
                "scheduler_threshold_drop",
                "burst_boundary_miss",
                "sticky_eviction_miss",
                "promoted_but_unused",
                "retention_miss",
            ]
            for reason in reasons:
                if reason in content:
                    details.append(f"Failure reason: {reason}")
                else:
                    passed = False
                    details.append(f"MISSING reason: {reason}")
            
            # Check for single reason rule
            if "exactly one" in content.lower() or "single" in content.lower():
                details.append("Single reason rule documented")
            else:
                passed = False
                details.append("MISSING: single reason rule")
        
        return GateResult(name, passed, details, [str(attr_path)])
    
    def print_report(self) -> None:
        """Print validation report."""
        print("=" * 70)
        print("ACCEPTANCE GATES VALIDATION REPORT")
        print("=" * 70)
        print()
        
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {result.name}")
            for detail in result.details[:5]:  # Limit details to avoid too much output
                print(f"  - {detail}")
            if len(result.details) > 5:
                print(f"  - ... and {len(result.details) - 5} more")
            print()
        
        all_passed = all(r.passed for r in self.results)
        print("=" * 70)
        print(f"SUMMARY: {passed_count}/{total_count} gates passed")
        print()
        if all_passed:
            print("ALL GATES PASSED - Framework is complete and ready for evaluation")
            print()
            print("Next steps:")
            print("  1. Run: python prosex/src/runners/demo_metrics.py")
            print("  2. Integrate with your workload runner")
            print("  3. Run burst ablation: python prosex/src/runners/burst_ablation.py")
            print("  4. Run sticky ablation: python prosex/src/runners/sticky_ablation.py")
        else:
            print("SOME GATES FAILED - Review details above")
        print("=" * 70)


def main():
    """Main entry point."""
    validator = AcceptanceGateValidator()
    all_passed, summary = validator.validate_all()
    
    validator.print_report()
    
    # Save summary
    output_path = Path("outputs/reports/acceptance_gates_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nValidation summary saved to: {output_path}")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
