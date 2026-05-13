"""
Causal Verification Evaluator.

Aggregates results from all 7 causal verification layers and produces
a unified pass/fail report. Follows the SharedEvaluator pattern from
eval/shared/evaluator.py for consistency with the existing evaluation framework.

Used by the unified runner (run_causal_verification.py) to produce
the final CausalVerificationReport.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core_types import (
    CausalVerificationReport,
    InterventionResult,
    QuadrantMetrics,
    BudgetProjectionResult,
    InformationBottleneckResult,
    SpoofingResult,
    CrossArchitectureResult,
    OnlineBanditResult,
)

logger = logging.getLogger(__name__)


class CausalVerificationEvaluator:
    """
    Aggregates and evaluates results from the Seven-Layer Causal Verification Framework.

    Each layer produces its own result types. This evaluator collects them,
    computes per-layer pass/fail verdicts, and generates a unified report.

    Usage:
        evaluator = CausalVerificationEvaluator()
        evaluator.add_layer_1_results(intervention_results)
        evaluator.add_layer_2_results(quadrant_metrics, qcdr, pass_fail)
        ...
        report = evaluator.finalize("experiment_001")
    """

    def __init__(self):
        self._layer_results: Dict[int, Any] = {}
        self._pass_fail: Dict[int, bool] = {}
        self._findings: List[str] = []

    # --- Layer-specific result collectors ---

    def add_layer_1_results(
        self,
        intervention_results: List[InterventionResult],
    ):
        """Add CEI results."""
        self._layer_results[1] = intervention_results

        # Pass if at least 2 dimensions pass fix-to-mean
        fix_results = [
            r for r in intervention_results
            if r.intervention_type.value == "fix_to_mean"
        ]
        n_passed = sum(1 for r in fix_results if r.pass_fail)
        passed = n_passed >= 2

        self._pass_fail[1] = passed
        if not passed:
            self._findings.append(
                f"Layer 1 (CEI): Only {n_passed}/{len(fix_results)} dimensions show "
                f"causal effect. Temporal cues may be confounding the scoring."
            )

    def add_layer_2_results(
        self,
        quadrant_metrics: List[QuadrantMetrics],
        qcdr: float,
        pass_fail: bool,
    ):
        """Add QUDM results."""
        self._layer_results[2] = {
            "quadrant_metrics": quadrant_metrics,
            "qcdr": qcdr,
            "pass_fail": pass_fail,
        }
        self._pass_fail[2] = pass_fail

        if not pass_fail:
            locality_trap = next(
                (m for m in quadrant_metrics if "high_reuse_low_utility" in m.quadrant.value),
                None,
            )
            long_range = next(
                (m for m in quadrant_metrics if "low_reuse_high_utility" in m.quadrant.value),
                None,
            )
            loc_trap_str = f"{locality_trap.admission_rate:.3f}" if locality_trap else "N/A"
            long_range_str = f"{long_range.admission_rate:.3f}" if long_range else "N/A"
            self._findings.append(
                f"Layer 2 (QUDM): QCDR={qcdr:.3f} exceeds threshold. "
                f"Locality trap admission={loc_trap_str}, "
                f"Long-range admission={long_range_str}. "
                f"Query-independent summaries are architecturally biased toward LRU-like behavior."
            )

    def add_layer_3_results(
        self,
        budget_results: List[BudgetProjectionResult],
    ):
        """Add EB-QPT results."""
        self._layer_results[3] = budget_results

        # Pass if B_Q > 0 shows improvement relative to B_Q = 0
        base_qcdr = next(
            (r.qcdr for r in budget_results if r.budget_b_q == 0), None
        )
        best_qcdr = min(r.qcdr for r in budget_results if r.budget_b_q > 0) if len(budget_results) > 1 else base_qcdr

        passed = base_qcdr is not None and best_qcdr is not None and best_qcdr < base_qcdr * 0.85
        self._pass_fail[3] = passed

        if not passed:
            self._findings.append(
                f"Layer 3 (EB-QPT): QCDR improvement with query projection is insufficient "
                f"(base={base_qcdr:.3f}, best_with_query={best_qcdr:.3f}). "
                f"64B may be a hard information ceiling."
            )

    def add_layer_4_results(
        self,
        result: InformationBottleneckResult,
    ):
        """Add ITLBP results."""
        self._layer_results[4] = result
        self._pass_fail[4] = result.is_64b_sufficient

        if not result.is_64b_sufficient:
            self._findings.append(
                f"Layer 4 (ITLBP): Saturation entropy {result.saturation_entropy:.0f} bits "
                f"exceeds 64B capacity ({result.hard_capacity_bits} bits). "
                f"64B is a hard information-theoretic ceiling. "
                f"Achieved MI at 64B: {result.achieved_mi:.3f} nats."
            )

    def add_layer_5_results(
        self,
        spoofing_results: List[SpoofingResult],
    ):
        """Add ACS results."""
        self._layer_results[5] = spoofing_results
        passed = all(r.pass_fail for r in spoofing_results)
        self._pass_fail[5] = passed

        if not passed:
            for r in spoofing_results:
                if not r.pass_fail:
                    self._findings.append(
                        f"Layer 5 (ACS): {r.spoof_type} spoofing CVI={r.cvi:.3f}. "
                        f"Query-aware variant does not sufficiently reduce spoofing "
                        f"(base_ssr={r.base_ssr:.3f}, query_aware_ssr={r.query_aware_ssr:.3f}). "
                        f"System is causally vulnerable to adversarial manipulation."
                    )

    def add_layer_6_results(
        self,
        cross_arch_results: List[CrossArchitectureResult],
    ):
        """Add CACT results."""
        self._layer_results[6] = cross_arch_results
        # Pass if at least one architecture has CEC >= threshold
        passed = any(
            r.cec_vs_mha >= 0.7 for r in cross_arch_results
            if r.architecture != "MHA"
        )
        self._pass_fail[6] = passed

        if not passed:
            self._findings.append(
                f"Layer 6 (CACT): No non-MHA architecture achieves CEC >= 0.7. "
                f"Causal effects are architecture-specific, not generalizable."
            )

    def add_layer_7_results(
        self,
        bandit_results: List[OnlineBanditResult],
    ):
        """Add OCA results."""
        self._layer_results[7] = bandit_results

        # Hard requirement: zero SBFI violations
        sbfi_ok = all(r.sbfi_boundary_violations == 0 for r in bandit_results)
        # Pass if any algorithm passes
        passed = sbfi_ok and any(r.pass_fail for r in bandit_results)
        self._pass_fail[7] = passed

        if not sbfi_ok:
            self._findings.append(
                "Layer 7 (OCA): CRITICAL - SBFI boundary violations detected! "
                "Causal isolation constraint failed."
            )
        elif not passed:
            self._findings.append(
                "Layer 7 (OCA): No bandit algorithm maintains recovery within "
                "tolerance under distribution drift. Frozen weights are not "
                "adaptively improvable with the current evidence budget."
            )

    # --- Finalization ---

    def finalize(self, experiment_id: str = "") -> CausalVerificationReport:
        """
        Produce the final CausalVerificationReport.

        Returns:
            Aggregated report with all layer results and pass/fail summary.
        """
        n_passed = sum(1 for v in self._pass_fail.values() if v)
        n_total = len(self._pass_fail)
        self._findings.insert(0, f"Overall: {n_passed}/{n_total} layers passed.")

        return CausalVerificationReport(
            experiment_id=experiment_id,
            layer_results=self._layer_results,
            pass_fail_summary=self._pass_fail,
            critical_findings=self._findings,
        )

    # ------------------------------------------------------------------
    #  Table-drawing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _table_sep(col_widths: List[int], left="+", mid="+", right="+", fill="-"):
        """Draw a horizontal table separator, e.g. +----+-----+----+"""
        parts = [fill * (w + 2) for w in col_widths]
        return left + mid.join(parts) + right

    @staticmethod
    def _table_row(col_widths: List[int], values: List[str], align="<"):
        """Draw a single data row."""
        cells = []
        for w, v in zip(col_widths, values):
            cells.append(f" {v:{align}{w}} ")
        return "|" + "|".join(cells) + "|"

    @classmethod
    def _table_print(cls, col_widths: List[int], header: List[str],
                     rows: List[List[str]], align="<"):
        """Print a complete bordered ASCII table."""
        sep = cls._table_sep(col_widths)
        print(sep)
        print(cls._table_row(col_widths, header, align="^"))
        print(sep)
        for row in rows:
            print(cls._table_row(col_widths, row, align=align))
        print(sep)

    # ------------------------------------------------------------------
    #  Report printing
    # ------------------------------------------------------------------

    def print_report(self, report: Optional[CausalVerificationReport] = None):
        """Print formatted causal verification report with full metrics."""
        if report is None:
            report = self.finalize()

        print("\n" + "=" * 80)
        print("  SEVEN-LAYER CAUSAL VERIFICATION REPORT")
        print("=" * 80)
        print(f"  Experiment: {report.experiment_id}")
        print(f"  Overall   : {report.n_passed}/{len(report.pass_fail_summary)} layers passed")
        print("=" * 80)

        layer_names = {
            1: "CEI   - Counterfactual Evidence Intervention",
            2: "QUDM  - Query-Utility Disentanglement Matrix",
            3: "EB-QPT - Evidence Budget-Query Projection Tradeoff",
            4: "ITLBP - Information-Theoretic Lower Bound Probing",
            5: "ACS   - Adversarial Causal Spoofing",
            6: "CACT  - Cross-Architectural Causal Transfer",
            7: "OCA   - Online Causal Adaptation",
        }

        for layer_num in sorted(report.pass_fail_summary.keys()):
            status = "PASS" if report.pass_fail_summary[layer_num] else "FAIL"
            name = layer_names.get(layer_num, f"Layer {layer_num}")
            print(f"\n{'-'*80}")
            print(f"  [{status}] LAYER {layer_num}: {name}")
            print(f"{'-'*80}")
            self._print_layer_metrics(layer_num, report.layer_results.get(layer_num))

        print(f"\n{'='*80}")
        print("  CRITICAL FINDINGS:")
        for i, finding in enumerate(report.critical_findings, 1):
            print(f"  {i}. {finding}")
        print("=" * 80 + "\n")

    def _print_layer_metrics(self, layer_num: int, result):
        """Print detailed metrics for a single layer."""
        if result is None:
            print("  (no data)")
            return

        if layer_num == 1:
            self._print_cei_metrics(result)
        elif layer_num == 2:
            self._print_qudm_metrics(result)
        elif layer_num == 3:
            self._print_ebqpt_metrics(result)
        elif layer_num == 4:
            self._print_itlbp_metrics(result)
        elif layer_num == 5:
            self._print_acs_metrics(result)
        elif layer_num == 6:
            self._print_cact_metrics(result)
        elif layer_num == 7:
            self._print_oca_metrics(result)

    def _print_cei_metrics(self, intervention_results):
        """Print CEI per-dimension causal effect deltas."""
        from src.core_types import CausalInterventionType
        fix_results = [r for r in intervention_results
                       if r.intervention_type == CausalInterventionType.FIX_TO_MEAN]
        ghost_results = [r for r in intervention_results
                         if r.intervention_type == CausalInterventionType.GHOST_SYNTHESIZE]

        if fix_results:
            print()
            cols = [14, 10, 11, 10, 12, 6]
            hdr = ["Dimension", "Baseline", "Intervened", "Delta", "Consistent", "Pass"]
            rows = []
            for r in fix_results:
                rows.append([
                    r.dimension.value,
                    f"{r.baseline_admission_rate:.4f}",
                    f"{r.intervention_admission_rate:.4f}",
                    f"{r.delta_admission_rate:+.4f}",
                    "YES" if r.consistent_across_phases else "NO",
                    "PASS" if r.pass_fail else "FAIL",
                ])
            self._table_print(cols, hdr, rows)

        if ghost_results:
            print("\n  Ghost Intervention (spoofability check):")
            cols = [14, 16, 6]
            hdr = ["Dimension", "Ghost Adm Rate", "Pass"]
            rows = []
            for r in ghost_results:
                rows.append([
                    r.dimension.value,
                    f"{r.intervention_admission_rate:.4f}",
                    "PASS" if r.pass_fail else "FAIL",
                ])
            self._table_print(cols, hdr, rows)

    def _print_qudm_metrics(self, result):
        """Print QUDM quadrant-level metrics."""
        if isinstance(result, dict):
            metrics = result.get("quadrant_metrics", [])
            qcdr = result.get("qcdr", 0.0)
            pass_fail = result.get("pass_fail", False)
        else:
            return

        print(f"\n  QCDR (Query-Causal Defect Rate): {qcdr:.4f}  "
              f"[threshold: 0.45, pass: {pass_fail}]")
        if qcdr > 0.45:
            print(f"  WARNING: {qcdr*100:.1f}% of admissions are locality traps rather than")
            print(f"  true long-range dependencies. System behaves like an LRU cache.")

        if metrics:
            print()
            cols = [30, 8, 10, 12, 11]
            hdr = ["Quadrant", "#Chunks", "Adm Rate", "Avg Utility", "Avg Reuse"]
            rows = []
            label_map = {
                "high_reuse_high_utility": "1. HOT (High R + High U)",
                "high_reuse_low_utility":  "2. LOC TRAP (High R + Low U)",
                "low_reuse_high_utility":  "3. LONG RANGE (Low R + High U)",
                "low_reuse_low_utility":   "4. COLD (Low R + Low U)",
            }
            for m in metrics:
                label = label_map.get(m.quadrant.value, m.quadrant.value)
                rows.append([
                    label,
                    str(m.num_chunks),
                    f"{m.admission_rate:.4f}",
                    f"{m.avg_utility:.4f}",
                    f"{m.avg_reuse_score:.4f}",
                ])
            self._table_print(cols, hdr, rows)

    def _print_ebqpt_metrics(self, budget_results):
        """Print EB-QPT budget sweep results."""
        if not budget_results:
            return
        base = budget_results[0]
        best = min(budget_results, key=lambda r: r.qcdr)

        print(f"\n  Sweeping B_Q (query sketch budget) within 64B total evidence:")
        print()
        cols = [6, 5, 10, 9, 17, 10]
        hdr = ["B_KV", "B_Q", "Recovery", "QCDR", "Long-Range Hit", "Passkey"]
        rows = []
        for r in budget_results:
            rows.append([
                str(r.budget_b_kv),
                str(r.budget_b_q),
                f"{r.recovery:.4f}",
                f"{r.qcdr:.4f}",
                f"{r.long_range_hit_rate:.4f}",
                f"{r.passkey_recovery:.4f}",
            ])
        self._table_print(cols, hdr, rows)

        improvement = (base.qcdr - best.qcdr) / max(base.qcdr, 1e-6)
        print(f"\n  QCDR improvement with query projection: {improvement*100:.1f}% "
              f"(B_Q=0 -> {base.qcdr:.4f}, B_Q={best.budget_b_q} -> {best.qcdr:.4f})")
        if improvement < 0.15:
            print(f"  WARNING: Query projection provides negligible QCDR improvement.")
            print(f"  64B may be a hard information ceiling for query-conditional utility.")

    def _print_itlbp_metrics(self, result):
        """Print ITLBP information-theoretic results."""
        if hasattr(result, 'saturation_entropy'):
            print(f"\n  Evidence budget hard capacity  : {result.hard_capacity_bits} bits (64B * 85% efficiency)")
            print(f"  Saturation entropy H_sat       : {result.saturation_entropy:.0f} bits")
            print(f"  Achieved MI at 64B encoding    : {result.achieved_mi:.4f} nats")
            print(f"  64B sufficient?                : {'YES' if result.is_64b_sufficient else 'NO'}")

            if hasattr(result, 'budget_recovery_curve') and result.budget_recovery_curve:
                print(f"\n  Budget-Recovery curve:")
                print()
                cols = [10, 16]
                hdr = ["Budget", "Recovery (R^2)"]
                rows = []
                for budget, recovery in sorted(result.budget_recovery_curve.items()):
                    marker = "  <-- 64B" if budget == 64 else ""
                    rows.append([f"{budget}B", f"{recovery:.4f}{marker}"])
                self._table_print(cols, hdr, rows)

            if result.saturation_entropy > result.hard_capacity_bits:
                print(f"\n  WARNING: H_sat ({result.saturation_entropy:.0f} bits) exceeds")
                print(f"  capacity ({result.hard_capacity_bits} bits). 64B is a HARD ceiling.")
                print(f"  The reported 0.703 recovery is not a design knee -- it is an")
                print(f"  information-theoretic upper bound for query-independent summaries.")

    def _print_acs_metrics(self, spoofing_results):
        """Print ACS adversarial spoofing results."""
        if not spoofing_results:
            return
        print()
        cols = [14, 10, 9, 11, 8, 6]
        hdr = ["Spoof Type", "Base SSR", "QA SSR", "Oracle SSR", "CVI", "Pass"]
        rows = []
        for r in spoofing_results:
            rows.append([
                r.spoof_type,
                f"{r.base_ssr:.4f}",
                f"{r.query_aware_ssr:.4f}",
                f"{r.oracle_ssr:.4f}",
                f"{r.cvi:.4f}",
                "PASS" if r.pass_fail else "FAIL",
            ])
        self._table_print(cols, hdr, rows)

        low_cvi = [r for r in spoofing_results if r.cvi < 0.3]
        if low_cvi:
            types = ', '.join(r.spoof_type for r in low_cvi)
            print(f"\n  WARNING: {types} spoofing CVI < 0.3. Query-independent ODUS-X")
            print(f"  is causally vulnerable -- it cannot distinguish genuine utility")
            print(f"  signals from adversarially inflated statistical correlations.")

    def _print_cact_metrics(self, cross_arch_results):
        """Print CACT cross-architecture consistency results."""
        if not cross_arch_results:
            return
        all_dims = set()
        for r in cross_arch_results:
            all_dims.update(r.ace_vector.keys())
            all_dims.update(r.dimension_consistencies.keys())
        dims = sorted(all_dims)

        print(f"\n  Causal Effect Consistency (CEC) across architectures:")
        print()

        # Architecture-level table
        cols = [13] + [11] * len(dims) + [12]
        hdr = ["Architecture"] + dims + ["CEC vs MHA"]
        rows = []
        for r in cross_arch_results:
            row = [r.architecture]
            for d in dims:
                row.append(f"{r.ace_vector.get(d, 0.0):.4f}")
            row.append(f"{r.cec_vs_mha:.4f}")
            rows.append(row)
        self._table_print(cols, hdr, rows)

        # Per-dimension CEC summary
        if cross_arch_results and hasattr(cross_arch_results[0], 'dimension_consistencies'):
            dcs = cross_arch_results[0].dimension_consistencies
            if dcs:
                print(f"\n  Per-dimension CEC (1.0 = perfectly consistent):")
                print()
                cols = [14, 8]
                hdr = ["Dimension", "CEC"]
                rows = []
                for d, cec in sorted(dcs.items()):
                    flag = "  LOW" if cec < 0.7 else ""
                    rows.append([d, f"{cec:.4f}{flag}"])
                self._table_print(cols, hdr, rows)

        low_cec = [r for r in cross_arch_results
                   if r.architecture != "MHA" and r.cec_vs_mha < 0.7]
        if low_cec:
            archs = ', '.join(r.architecture for r in low_cec)
            print(f"\n  WARNING: {archs} CEC < 0.7. Causal effects do not transfer")
            print(f"  across architectures. Temporal cues likely proxy for architecture-")
            print(f"  specific attention patterns, not universal utility signals.")

    def _print_oca_metrics(self, bandit_results):
        """Print OCA online causal adaptation results."""
        if not bandit_results:
            return
        print()

        # Main results table
        cols = [20, 11, 11, 10, 7, 6]
        hdr = ["Algorithm", "Cum Reward", "Cum Regret", "SBFI Viol", "Drifts", "Pass"]
        rows = []
        for r in bandit_results:
            rows.append([
                r.algorithm.value,
                f"{r.cumulative_reward:.4f}",
                f"{r.cumulative_regret:.4f}",
                str(r.sbfi_boundary_violations),
                str(len(r.drift_events)),
                "PASS" if r.pass_fail else "FAIL",
            ])
        self._table_print(cols, hdr, rows)

        # Final weights sub-table
        print()
        dims = ["e_temp", "e_struct", "e_sem", "e_hist", "e_press"]
        w_cols = [20] + [9] * len(dims)
        w_hdr = ["Algorithm"] + dims
        w_rows = []
        for r in bandit_results:
            row = [r.algorithm.value]
            for d in dims:
                val = r.final_weights.get(d, 0.0) if hasattr(r, 'final_weights') and r.final_weights else 0.0
                row.append(f"{val:.3f}")
            w_rows.append(row)
        self._table_print(w_cols, w_hdr, w_rows)

        # Summary
        best = min(bandit_results, key=lambda r: r.cumulative_regret)
        print(f"\n  Best algorithm: {best.algorithm.value} (lowest regret: {best.cumulative_regret:.4f})")

        violations = [r for r in bandit_results if r.sbfi_boundary_violations > 0]
        if violations:
            print(f"  CRITICAL: {len(violations)} algorithms violated SBFI constraint!")
        else:
            print(f"  SBFI constraint preserved: 0 violations across all algorithms.")

        no_pass = [r for r in bandit_results if not r.pass_fail]
        if no_pass:
            algos = ', '.join(r.algorithm.value for r in no_pass)
            print(f"\n  WARNING: {algos} failed to maintain recovery within tolerance.")
            print(f"  Frozen ODUS-X weights cannot be adaptively improved under")
            print(f"  distribution drift with the current 64B evidence budget.")

    def reset(self):
        """Reset all accumulated results."""
        self._layer_results.clear()
        self._pass_fail.clear()
        self._findings.clear()
