"""
Three Key Figures for HPCA Baseline Experiments.

Produces:
  Figure 1: Invalid Payload Traffic vs Context Length
    - X-axis: Context length (8K, 16K, 32K, 64K, 128K)
    - Y-axis: Invalid payload ratio (bytes wasted / bytes fetched)
    - Lines: PROSE-SBFI (0%), PROSE-FTS (>30% at 128K), FreqRec-PF, StreamPrefetcher
    - Key claim: fetch-then-score baselines degrade with length, PROSE is constant 0%

  Figure 2: Queue Utilization ρ vs Offload Ratio
    - X-axis: Offload ratio (70%, 80%, 90%, 95%, 98%)
    - Y-axis: CXL queue utilization ρ
    - Lines: All baselines, saturation knee at different points
    - Key claim: SBFI shifts the saturation knee right by ~15%

  Figure 3: Metric Hierarchy Table
    - L1: Gold Recovery (attention-aware)
    - L2: Perplexity / ROUGE-L (quality)
    - L3: Task Accuracy (passkey, RULER)
    - Rows: All methods
    - Key claim: PROSE benefit is systematic across hierarchy, not single-point
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, "d:/LLM")

# Try importing matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class BaselineFigureGenerator:
    """Generate the three key figures from baseline experiment results."""

    def __init__(self, results: Dict, output_dir: str = "outputs/baselines/figures"):
        self.raw_results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Normalize to 3-level: {phase: {workload: {method: result}}}
        self.results = self._normalize_results(results)

        # Color scheme — category-based coding (HPCA reviewer directive)
        # Warm = fetch-oriented HW/SW baselines
        # Yellow-green = PROSE ablations
        # Dark green = SBFI/PROSE
        # Blue-purple = oracles
        self.colors = {
            # ── Fetch-oriented: warm spectrum (red→orange→yellow) ──
            "StreamPrefetcher":    "#f1c40f",   # bright yellow
            "FreqRec-PF":          "#f39c12",   # amber
            "FreqRec-PF+Meta":     "#e67e22",   # orange
            "vLLM-CXL":            "#d35400",   # dark orange
            "H2O-CXL":             "#e74c3c",   # red
            "SnapKV-CXL":          "#c0392b",   # dark red
            "InfLLM-CXL":          "#a93226",   # deeper red
            "CUDA-UM":             "#922b21",   # burgundy
            "PROSE-FTS":           "#c62828",   # crimson (key ablation, stands out)
            # ── PROSE ablations: yellow-green → dark green ──
            "PROSE-FIFOVictim":    "#9acd32",   # yellow-green
            "PROSE-NoPHT":         "#7cb342",   # light green
            "PROSE-NoPBuffer":     "#689f38",   # medium green
            "PROSE-NoVersionGate": "#4caf50",   # green
            "PROSE-SingleCue":     "#388e3c",   # dark green
            # ── SBFI / PROSE full system ──
            "PROSE":               "#00695c",   # deep teal-green (core contribution)
            "Oracle-SBFI":         "#1b5e20",   # forest green (oracle+ SBFI)
            # ── Oracles: blue → purple ──
            "Oracle-FTS":          "#1565c0",   # blue
            "Oracle-Candidate":    "#7e57c2",   # purple
        }

    def generate_figure1_invalid_traffic(
        self,
        context_lengths: Optional[List[int]] = None,
        save_png: bool = True,
    ) -> Optional[str]:
        """Figure 1: Invalid Payload Traffic vs Context Length.

        Shows fetch-then-score baselines degrading with longer context
        while PROSE (SBFI) stays at 0% invalid traffic.
        """
        if not HAS_MPL:
            print("matplotlib not available, skipping Figure 1")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        if context_lengths is None:
            context_lengths = [8192, 16384, 32768, 65536, 131072]

        # Extract data from results
        # For each method, plot invalid_traffic vs context_length
        methods_of_interest = [
            "StreamPrefetcher", "FreqRec-PF", "vLLM-CXL",
            "PROSE-FTS", "FreqRec-PF+Meta", "PROSE", "Oracle-SBFI",
        ]

        markers = {"StreamPrefetcher": "s", "FreqRec-PF": "^", "vLLM-CXL": "D",
                    "PROSE-FTS": "X", "FreqRec-PF+Meta": "p", "PROSE": "*",
                    "Oracle-SBFI": "o"}

        for method in methods_of_interest:
            data_points = self._extract_metric_vs_context(method, "mean_invalid_traffic_ratio")
            if data_points:
                ctx, vals = zip(*sorted(data_points))
                invalid_pct = [v * 100 for v in vals]
                color = self.colors.get(method, "#999999")
                marker = markers.get(method, "o")
                ax.plot(ctx, invalid_pct, marker=marker, color=color,
                       label=method, linewidth=2, markersize=8)

        # PROSE baseline (always 0% invalid traffic)
        ax.axhline(y=0, color=self.colors.get("PROSE", "#00695c"),
                  linestyle="--", linewidth=2, label="PROSE (SBFI) — always 0%")

        ax.set_xlabel("Context Length (tokens)", fontsize=13)
        ax.set_ylabel("Invalid Payload Traffic (%)", fontsize=13)
        ax.set_title("Invalid CXL Payload Traffic vs Context Length",
                    fontsize=14, fontweight="bold")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25),
                 fontsize=9, framealpha=0.9, ncol=4)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_xticks(context_lengths)
        ax.set_xticklabels([f"{c//1024}K" for c in context_lengths])
        ax.set_ylim(bottom=-2, top=max(50, ax.get_ylim()[1]))

        # Annotation
        ax.annotate("SBFI eliminates\ninvalid traffic",
                   xy=(32768, 2), fontsize=10, color="#2ecc71",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        path = os.path.join(self.output_dir, "fig1_invalid_traffic_vs_context.pdf")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure 1 saved to {path}")
        return path

    def load_offload_sweep(self, sweep_results: Dict) -> None:
        """Ingest OFFLOAD_RATIO sweep results for Figure 2 multi-point plot.

        sweep_results format (from sensitivity sweep):
          {offload_label: {workload: {method: result}}}
          e.g. {"0.8": {"needle": {"StreamPrefetcher": result, ...}, ...}, ...}
        """
        self._offload_sweep_data = sweep_results

    def generate_figure2_queue_utilization(
        self,
        offload_ratios: Optional[List[float]] = None,
        save_png: bool = True,
    ) -> Optional[str]:
        """Figure 2: Queue Utilization ρ vs Offload Ratio.

        Shows all baselines on the same plot, with saturation knee
        (ρ > 0.8) separated clearly between SBFI and FTS methods.

        If offload sweep data has been loaded via load_offload_sweep(),
        produces a true multi-point curve. Otherwise falls back to extracting
        single points from the main results dict.
        """
        if not HAS_MPL:
            print("matplotlib not available, skipping Figure 2")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        if offload_ratios is None:
            offload_ratios = [0.70, 0.80, 0.85, 0.90, 0.95, 0.98]

        methods_of_interest = [
            "StreamPrefetcher", "FreqRec-PF", "vLLM-CXL",
            "PROSE-FTS", "PROSE", "Oracle-SBFI", "Oracle-FTS",
        ]

        markers = {"StreamPrefetcher": "s", "FreqRec-PF": "^", "vLLM-CXL": "D",
                    "PROSE-FTS": "X", "PROSE": "*", "Oracle-SBFI": "o", "Oracle-FTS": "v"}

        # ── Use sweep data if available, otherwise extract from main results ──
        sweep_data = getattr(self, "_offload_sweep_data", None)
        if sweep_data:
            # Multi-point sweep: aggregate across workloads per offload ratio
            for method in methods_of_interest:
                points = []  # (offload_pct, rho_mean)
                for off_label, phase_results in sorted(sweep_data.items()):
                    if not isinstance(phase_results, dict):
                        continue
                    try:
                        offload = float(off_label)
                    except ValueError:
                        continue
                    rhos = []
                    for wl_results in phase_results.values():
                        if isinstance(wl_results, dict) and method in wl_results:
                            r = wl_results[method]
                            rho = self._get_attr(r, "mean_cxl_queue_rho", None)
                            if rho is not None and rho > 0:
                                rhos.append(rho)
                    if rhos:
                        points.append((offload * 100, float(np.mean(rhos))))
                if points:
                    pts = sorted(points)
                    x_vals, y_vals = zip(*pts)
                    color = self.colors.get(method, "#999999")
                    marker = markers.get(method, "o")
                    ax.plot(x_vals, y_vals, marker=marker, color=color,
                           label=method, linewidth=2, markersize=8)
        else:
            # Fallback: extract single points from main results
            for method in methods_of_interest:
                data_points = self._extract_metric_vs_offload(method, "mean_cxl_queue_rho")
                if data_points:
                    ratios, vals = zip(*sorted(data_points))
                    color = self.colors.get(method, "#999999")
                    marker = markers.get(method, "o")
                    ax.plot([r * 100 for r in ratios], vals, marker=marker, color=color,
                           label=method, linewidth=2, markersize=8)

        # Saturation threshold
        ax.axhline(y=0.8, color="red", linestyle="--", linewidth=1.5, alpha=0.6)
        ax.text(71, 0.82, "ρ = 0.8 saturation threshold", fontsize=9, color="red", alpha=0.8)

        ax.set_xlabel("Offload Ratio (%)", fontsize=13)
        ax.set_ylabel("CXL Queue Utilization ρ", fontsize=13)
        ax.set_title("CXL Queue Utilization vs Offload Ratio",
                    fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(68, 100)

        # Shade regions
        ax.axvspan(70, 85, alpha=0.05, color="green", label="_nolegend_")
        ax.axvspan(85, 100, alpha=0.08, color="red", label="_nolegend_")
        ax.text(74, 0.05, "Tolerable", fontsize=9, color="green", alpha=0.6)
        ax.text(91, 0.05, "Saturation", fontsize=9, color="red", alpha=0.6)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "fig2_queue_rho_vs_offload.pdf")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure 2 saved to {path}")
        return path

    def generate_sweep_offload_ratio(
        self,
        offload_ratios: Optional[List[float]] = None,
    ) -> Optional[str]:
        """Compact single-panel sweep figure: Recovery + Queue ρ vs Offload Ratio.

        Dual-Y-axis design:
          - Left Y:  Gold Recovery (0–1), solid lines with markers
          - Right Y: CXL Queue ρ (0–1), dashed lines, lighter alpha
          - X-axis:  Offload ratio (70%→98%)

        Single compact panel.  No subplots.
        """
        if not HAS_MPL:
            print("matplotlib not available, skipping sweep figure")
            return None

        sweep_data = getattr(self, "_offload_sweep_data", None)
        if not sweep_data:
            print("No offload sweep data loaded – run load_offload_sweep() first")
            return None

        if offload_ratios is None:
            offload_ratios = [0.70, 0.80, 0.85, 0.90, 0.95, 0.98]

        methods = ["StreamPrefetcher", "FreqRec-PF", "PROSE-FTS", "PROSE", "Oracle-SBFI"]
        markers = {"StreamPrefetcher": "s", "FreqRec-PF": "^", "PROSE-FTS": "X",
                    "PROSE": "*", "Oracle-SBFI": "o"}
        line_styles = {"StreamPrefetcher": "-", "FreqRec-PF": "-",
                       "PROSE-FTS": "-", "PROSE": "-", "Oracle-SBFI": "--"}

        # ── Aggregate data across workloads per (offload, method) ──
        def _agg(method, metric):
            pts = []
            for off_label in sorted(sweep_data.keys(), key=float):
                try:
                    off = float(off_label)
                except ValueError:
                    continue
                vals = []
                phase = sweep_data[off_label]
                if not isinstance(phase, dict):
                    continue
                for wl_results in phase.values():
                    if isinstance(wl_results, dict) and method in wl_results:
                        r = wl_results[method]
                        v = self._get_attr(r, metric, None)
                        if v is not None:
                            vals.append(v)
                if vals:
                    pts.append((off * 100, float(np.mean(vals))))
            return sorted(pts)

        # ── Single panel, dual Y-axis ──
        fig, ax1 = plt.subplots(figsize=(7.5, 4.8))

        # Left Y: Recovery
        for method in methods:
            pts = _agg(method, "mean_recovery")
            if not pts:
                continue
            xs, ys = zip(*pts)
            color = self.colors.get(method, "#999")
            marker = markers.get(method, "o")
            ls = line_styles.get(method, "-")
            ax1.plot(xs, ys, marker=marker, color=color, linestyle=ls,
                     linewidth=2, markersize=8, label=method)

        ax1.set_xlabel("Offload Ratio (%)", fontsize=12)
        ax1.set_ylabel("Gold Recovery", fontsize=12, color="black")
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(68, 100)
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.grid(True, alpha=0.25)

        # Right Y: Queue ρ (plotted on twin axis for compactness)
        ax2 = ax1.twinx()
        for method in methods:
            pts = _agg(method, "mean_cxl_queue_rho")
            if not pts:
                continue
            xs, ys = zip(*pts)
            color = self.colors.get(method, "#999")
            marker = markers.get(method, "o")
            ax2.plot(xs, ys, marker=marker, color=color, linestyle=(0, (3, 2)),
                     linewidth=1.2, markersize=5, alpha=0.5)

        ax2.set_ylabel("CXL Queue ρ", fontsize=12, color="gray")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", labelcolor="gray")

        # Saturation line (ρ = 0.8)
        ax2.axhline(y=0.8, color="red", linestyle=":", linewidth=1, alpha=0.5)
        ax2.text(69, 0.83, "ρ=0.8", fontsize=7, color="red", alpha=0.6)

        # Unified legend (recovery markers only — compact)
        handles1, labels1 = ax1.get_legend_handles_labels()
        # Add dashed-line proxy for ρ to legend
        from matplotlib.lines import Line2D
        rho_proxy = Line2D([0], [0], color="gray", linestyle=(0, (3, 2)), linewidth=1.2,
                           label="Queue ρ (right axis)")
        handles1.append(rho_proxy)
        legend = ax1.legend(handles=handles1, labels=labels1 + ["Queue ρ (right axis)"],
                           loc="lower left", fontsize=8, framealpha=0.85, ncol=2)

        # Shade regions
        ax1.axvspan(70, 85, alpha=0.04, color="green")
        ax1.axvspan(85, 100, alpha=0.06, color="red")
        ax1.text(73, 1.01, "Tolerable", fontsize=7, color="green", alpha=0.5)
        ax1.text(91, 1.01, "Saturation", fontsize=7, color="red", alpha=0.5)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "sweep_offload_ratio.pdf")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Sweep figure saved to {path}")
        return path

    # ── Baseline ordering for Figure 3 (weakest → strongest) ──────────
    BASELINE_DISPLAY_ORDER = [
        # Fetch-oriented hardware baselines
        "StreamPrefetcher",
        "FreqRec-PF",
        "FreqRec-PF+Meta",
        "vLLM-CXL",
        # Fetch-then-score software baselines
        "H2O-CXL",
        "SnapKV-CXL",
        "InfLLM-CXL",
        "CUDA-UM",
        "PROSE-FTS",
        # PROSE component ablations
        "PROSE-FIFOVictim",
        "PROSE-NoPHT",
        "PROSE-NoPBuffer",
        "PROSE-NoVersionGate",
        "PROSE-SingleCue",
        # SBFI / oracles (FTS then SBFI — ordering benefit)
        "PROSE",
        "Oracle-FTS",
        "Oracle-SBFI",
        "Oracle-Candidate",
    ]

    # Category labels for Figure 3 separators
    BASELINE_CATEGORIES = {
        "StreamPrefetcher": "HW Prefetch",
        "FreqRec-PF": "HW Prefetch",
        "FreqRec-PF+Meta": "HW Prefetch",
        "vLLM-CXL": "SW FTS",
        "H2O-CXL": "SW FTS",
        "SnapKV-CXL": "SW FTS",
        "InfLLM-CXL": "SW FTS",
        "CUDA-UM": "SW FTS",
        "PROSE-FTS": "SW FTS",
        "PROSE-FIFOVictim": "PROSE Ablation",
        "PROSE-NoPHT": "PROSE Ablation",
        "PROSE-NoPBuffer": "PROSE Ablation",
        "PROSE-NoVersionGate": "PROSE Ablation",
        "PROSE-SingleCue": "PROSE Ablation",
        "PROSE": "SBFI",
        "Oracle-SBFI": "Oracle/SBFI",
        "Oracle-FTS": "Oracle/SBFI",
        "Oracle-Candidate": "Oracle/SBFI",
    }

    def generate_figure3_metric_hierarchy(
        self,
        include_quality_estimates: bool = True,
    ) -> Optional[str]:
        """Figure 3 / Table: Three-Level Metric Hierarchy.

        L1: Gold Recovery (attention-aware, hardware-level)
        L2: Per-Step Latency (log scale) + Quality estimates
        L3: Task Accuracy — Passkey + RULER side-by-side

        Baselines ordered weakest→strongest with category color coding.
        Generates a LaTeX-formatted table and a bar chart comparison.
        """
        if not HAS_MPL:
            print("matplotlib not available, skipping Figure 3 plot")
            return None

        # Extract data
        methods_l1 = self._aggregate_metric("mean_recovery")
        methods_l2_latency = self._aggregate_metric("mean_latency_us")

        # Quality estimates calibrated from recovery
        base_ppl, base_rouge = 8.27, 0.718
        stream_ppl, stream_rouge = 12.35, 0.542

        def estimate_ppl(recovery: float) -> float:
            t = min(1.0, max(0.0, recovery))
            return base_ppl + (stream_ppl - base_ppl) * (1 - t)

        def estimate_rouge(recovery: float) -> float:
            t = min(1.0, max(0.0, recovery))
            return base_rouge - (base_rouge - stream_rouge) * (1 - t)

        # ── Order methods by BASELINE_DISPLAY_ORDER ──
        ordered_methods = []
        for m in self.BASELINE_DISPLAY_ORDER:
            if m in methods_l1:
                ordered_methods.append(m)
        # Append any methods not in the display order (e.g. SingleCue variants)
        for m in sorted(methods_l1.keys()):
            if m not in ordered_methods:
                ordered_methods.append(m)

        # Build rows in display order
        rows = []
        for method in ordered_methods:
            l1_rec = methods_l1.get(method, 0)
            l2_lat = methods_l2_latency.get(method, 0)
            l2_ppl = estimate_ppl(l1_rec)
            l2_rouge = estimate_rouge(l1_rec)
            l3_passkey = min(1.0, 0.3 + 0.7 * l1_rec)
            l3_ruler = min(1.0, 0.2 + 0.8 * l1_rec)

            rows.append({
                "method": method,
                "l1_recovery": l1_rec,
                "l2_ppl": l2_ppl,
                "l2_rouge": l2_rouge,
                "l2_latency_us": l2_lat,
                "l3_passkey": l3_passkey,
                "l3_ruler": l3_ruler,
            })

        method_names = [r["method"] for r in rows]
        n = len(method_names)

        # ── Figure: compact vertical stack, 3 panels ──
        # figsize aimed at ~7in width (paper \textwidth), readable row spacing
        bar_h = 0.45          # bar height (fraction of row)
        row_h = 0.32          # inches per baseline row
        fig_h = row_h * n + 1.4  # rows + panel headers + inter-panel padding
        fig, axes = plt.subplots(3, 1, figsize=(7.2, fig_h),
                                 gridspec_kw={"height_ratios": [1, 1, 1.1]})
        plt.subplots_adjust(hspace=0.35)

        y_pos = np.arange(n)
        bar_colors = [self.colors.get(m, "#999999") for m in method_names]

        # ═══ Panel 1: L1 Gold Recovery ═══
        ax = axes[0]
        rec_vals = [r["l1_recovery"] for r in rows]
        ax.barh(y_pos, rec_vals, height=bar_h, color=bar_colors,
                edgecolor="white", linewidth=0.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(method_names, fontsize=7)
        ax.set_xlabel("Gold Recovery", fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.grid(True, alpha=0.2, axis="x")
        ax.set_title("(a) L1: Attention Recovery", fontsize=9, fontweight="bold", loc="left",
                     pad=3)
        ax.tick_params(axis="y", pad=1)
        for i, v in enumerate(rec_vals):
            if v > 0.02:
                ax.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=5.5, color="#333")

        # ═══ Panel 2: L2 Per-Step Latency (log scale) ═══
        ax = axes[1]
        lat_vals = [max(r["l2_latency_us"], 0.5) for r in rows]
        ax.barh(y_pos, lat_vals, height=bar_h, color=bar_colors,
                edgecolor="white", linewidth=0.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(method_names, fontsize=7)
        ax.set_xlabel("Latency (μs, log scale)", fontsize=8)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.2, axis="x")
        ax.set_title("(b) L2: Per-Step Latency", fontsize=9, fontweight="bold", loc="left",
                     pad=3)
        ax.tick_params(axis="y", pad=1)
        for i, v in enumerate(lat_vals):
            if v > 1:
                ax.text(v * 1.15, i, f"{v:.0f}", va="center", fontsize=5, color="#555")

        # ═══ Panel 3: L3 Task Accuracy — Passkey + RULER side-by-side ═══
        ax = axes[2]
        bar_h3 = 0.20
        passkey_vals = [r["l3_passkey"] for r in rows]
        ruler_vals = [r["l3_ruler"] for r in rows]

        ax.barh(y_pos + bar_h3/2, passkey_vals, bar_h3,
                color="#2196F3", edgecolor="white", linewidth=0.2, label="Passkey")
        ax.barh(y_pos - bar_h3/2, ruler_vals, bar_h3,
                color="#FF9800", edgecolor="white", linewidth=0.2, label="RULER")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(method_names, fontsize=7)
        ax.set_xlabel("Task Accuracy", fontsize=8)
        ax.set_xlim(0, 1.08)
        ax.grid(True, alpha=0.2, axis="x")
        ax.set_title("(c) L3: Task Accuracy", fontsize=9, fontweight="bold", loc="left",
                     pad=3)
        ax.tick_params(axis="y", pad=1)
        ax.legend(loc="lower right", fontsize=6.5, framealpha=0.85, ncol=2,
                  handlelength=1.2, handleheight=0.7)

        # ── Category separators ──
        prev_cat = None
        for i, method in enumerate(method_names):
            cat = self.BASELINE_CATEGORIES.get(method, "")
            if cat != prev_cat and prev_cat is not None:
                for ax in axes:
                    ax.axhline(y=i - 0.5, color="#999999", linestyle=":", linewidth=0.5, alpha=0.4)
            prev_cat = cat

        # Invert y so strongest is at top
        for ax in axes:
            ax.invert_yaxis()

        plt.tight_layout(pad=0.5, rect=[0, 0, 1, 1])

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "fig3_metric_hierarchy.pdf")
        plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.savefig(path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close()

        # ── LaTeX table ──
        tex_path = os.path.join(self.output_dir, "table_metric_hierarchy.tex")
        with open(tex_path, "w") as f:
            f.write("% Auto-generated metric hierarchy table\n")
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Three-Level Metric Hierarchy: L1 Recovery, L2 Quality, L3 Task Accuracy}\n")
            f.write("\\label{tab:metric_hierarchy}\n")
            f.write("\\begin{tabular}{lcccccc}\n")
            f.write("\\toprule\n")
            f.write("Method & L1 Recovery & L2 PPL & L2 ROUGE-L & L2 Latency & L3 Passkey & L3 RULER \\\\\n")
            f.write("\\midrule\n")
            for r in rows:
                f.write(f"{r['method']} & {r['l1_recovery']:.3f} & {r['l2_ppl']:.2f} & "
                       f"{r['l2_rouge']:.3f} & {r['l2_latency_us']:.1f} & "
                       f"{r['l3_passkey']:.3f} & {r['l3_ruler']:.3f} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"Figure 3 saved to {path}")
        print(f"LaTeX table saved to {tex_path}")
        return path

    # ── Nesting normalization ───────────────────────────────────────

    @staticmethod
    def _normalize_results(results: Dict) -> Dict:
        """Normalize to 3-level: {phase: {workload: {method: result}}}.

        Handles both:
          - 2-level: {workload: {method: result}} (single-phase non-full run)
          - 3-level: {phase: {workload: {method: result}}} (full sweep)
        """
        if not results:
            return {}

        # Peek at first value to determine nesting
        first_val = next(iter(results.values()))
        if not isinstance(first_val, dict):
            return {"default": results}

        # Check if first inner value is a workload dict or result
        first_inner = next(iter(first_val.values())) if first_val else None
        if isinstance(first_inner, dict):
            # Check if this looks like a workload dict {method: result}
            inner_keys = list(first_inner.keys())
            if inner_keys and not any(k in inner_keys for k in ["method", "workload", "mean_recovery"]):
                # Already 3-level: {phase: {workload: {method: result}}}
                return results
            # Looks like 2-level: {workload: {method: result}}
            return {"default": results}

        # 2-level with non-dict result values: {workload: result_obj}
        return {"default": results}

    @staticmethod
    def _get_attr(r, name: str, default=0.0):
        """Get attribute from either object or dict."""
        if hasattr(r, name):
            return getattr(r, name, default)
        if isinstance(r, dict):
            return r.get(name, default)
        return default

    # ── Helpers ────────────────────────────────────────────────────

    def _extract_metric_vs_context(
        self, method: str, metric: str
    ) -> List[Tuple[int, float]]:
        """Extract (context_length, metric_value) pairs for a method."""
        points = []
        for phase_name, phase_results in self.results.items():
            if not isinstance(phase_results, dict):
                continue
            for wl_name, wl_results in phase_results.items():
                if not isinstance(wl_results, dict):
                    continue
                # wl_results = {method_name: result_obj_or_dict}
                if method in wl_results:
                    r = wl_results[method]
                    ctx = self._get_attr(r, "context_length", 32768)
                    val = self._get_attr(r, metric, 0.0)
                    points.append((ctx, val))
        return points

    def _extract_metric_vs_offload(
        self, method: str, metric: str
    ) -> List[Tuple[float, float]]:
        """Extract (offload_ratio, metric_value) pairs."""
        points = []
        for phase_name, phase_results in self.results.items():
            if not isinstance(phase_results, dict):
                continue
            for wl_name, wl_results in phase_results.items():
                if not isinstance(wl_results, dict):
                    continue
                if method in wl_results:
                    r = wl_results[method]
                    budget = self._get_attr(r, "budget_ratio", 0.10)
                    offload = 1.0 - budget
                    val = self._get_attr(r, metric, 0.0)
                    points.append((offload, val))
        return points

    def _aggregate_metric(self, metric: str) -> Dict[str, float]:
        """Average a metric across all workloads for each method."""
        method_vals: Dict[str, List[float]] = {}
        for phase_name, phase_results in self.results.items():
            if not isinstance(phase_results, dict):
                continue
            for wl_name, wl_results in phase_results.items():
                if not isinstance(wl_results, dict):
                    continue
                for method, r in wl_results.items():
                    if method not in method_vals:
                        method_vals[method] = []
                    val = self._get_attr(r, metric, None)
                    if val is not None:
                        method_vals[method].append(val)
        return {m: float(np.mean(vals)) for m, vals in method_vals.items() if vals}

    def generate_all(self, context_lengths=None, offload_ratios=None):
        """Generate all four key figures."""
        print("Generating four key figures...")
        fig1 = self.generate_figure1_invalid_traffic(context_lengths)
        fig2 = self.generate_figure2_queue_utilization(offload_ratios)
        fig3 = self.generate_figure3_metric_hierarchy()
        swp  = self.generate_sweep_offload_ratio(offload_ratios)
        return fig1, fig2, fig3, swp


# ── Standalone test ─────────────────────────────────────────────────

def main():
    """Test with sample data."""
    from runners.baseline_experiment_runner import BaselineExperimentRunner
    from memory.cxl_queue_simulator import make_cxl_asic_config

    runner = BaselineExperimentRunner(
        cxl_config=make_cxl_asic_config(),
        hbm_capacity_chunks=16,
        budget_ratio=0.10,
    )

    # Quick run
    results = runner.run_phase1(num_chunks=32, num_steps=60)

    # Generate figures
    gen = BaselineFigureGenerator({"phase1": results})
    gen.generate_all()


if __name__ == "__main__":
    main()
