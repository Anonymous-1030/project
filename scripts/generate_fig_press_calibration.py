#!/usr/bin/env python3
"""
Generate Figure: PRESS Calibration & Error Analysis.

Two-panel figure for HPCA rebuttal:
  (a) Predicted vs Observed scatter (with 1:1 line and 95% CI envelope)
  (b) Per-workload error decomposition (MAPE + bias direction)

LaTeX-ready.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INPUT_JSON = Path("outputs/hpca_press_calibration/press_calibration_report.json")
OUTPUT_DIR = Path("outputs/hpca_press_calibration")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

C_PRESS = "#4C72B0"
C_GT = "#D95319"
C_ENV = "#EDB120"


def load_report():
    with open(INPUT_JSON) as f:
        return json.load(f)


def generate_figure():
    report = load_report()
    points = report["calibration_points"]
    stats = report["aggregate_stats"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fig.suptitle(
        "PRESS Calibration: Predictive-Validity under Strict Simulation",
        fontsize=12, fontweight="bold", y=0.98,
    )

    # ── (a) Predicted vs Observed scatter ────────────────────────────
    ax = axes[0]
    gt_vals = np.array([p["gt_total_us"] for p in points])
    pred_vals = np.array([p["press_total_us"] for p in points])
    workloads = [p["workload"] for p in points]
    workload_colors = {
        "sequential": "#4C72B0",
        "needle_heavy": "#D95319",
        "high_turnover": "#2E8B57",
        "realistic_synthetic": "#7F7F7F",
    }
    colors = [workload_colors.get(w, "#7F7F7F") for w in workloads]

    ax.scatter(gt_vals, pred_vals, c=colors, s=60, edgecolors="black", linewidth=0.5, alpha=0.85, zorder=3)

    # 1:1 line
    lim = [min(gt_vals.min(), pred_vals.min()) * 0.9, max(gt_vals.max(), pred_vals.max()) * 1.1]
    ax.plot(lim, lim, "k--", linewidth=1.2, label="Perfect prediction", zorder=2)

    # Fit line
    coeffs = np.polyfit(gt_vals, pred_vals, 1)
    fit_x = np.linspace(lim[0], lim[1], 100)
    fit_y = np.polyval(coeffs, fit_x)
    ax.plot(fit_x, fit_y, "-", color=C_PRESS, linewidth=1.5, label="Linear fit", zorder=2)

    # 95% CI envelope
    residuals = pred_vals - np.polyval(coeffs, gt_vals)
    sigma = np.std(residuals)
    ax.fill_between(fit_x, fit_y - 2*sigma, fit_y + 2*sigma, alpha=0.12, color=C_ENV, label="95% CI")

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed latency (trace-driven sim, μs)")
    ax.set_ylabel("PRESS predicted latency (μs)")
    ax.set_title("(a) Predicted vs Observed")

    # R^2
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((pred_vals - pred_vals.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot
    mape = stats["total_latency_mape"]
    rmse = stats["total_latency_rmse"]
    textstr = f"$R^2$={r_squared:.3f}\nMAPE={mape:.1f}%\nRMSE={rmse:.1f}%"
    ax.text(0.97, 0.05, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CCCCCC", alpha=0.95))

    # Legend with workload colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=workload_colors["sequential"], markersize=8, label="Sequential"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=workload_colors["needle_heavy"], markersize=8, label="Needle-heavy"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=workload_colors["high_turnover"], markersize=8, label="High-turnover"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=workload_colors["realistic_synthetic"], markersize=8, label="Realistic"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9, fontsize=7.5)

    # ── (b) Per-workload error decomposition ─────────────────────────
    ax = axes[1]
    by_w = stats.get("by_workload", {})
    workloads_sorted = sorted(by_w.keys())
    mape_vals = [by_w[w]["mape"] for w in workloads_sorted]
    max_err_vals = [by_w[w]["max_error_pct"] for w in workloads_sorted]
    bias_vals = [by_w[w]["mean_bias_pct"] for w in workloads_sorted]

    x = np.arange(len(workloads_sorted))
    width = 0.22

    bars1 = ax.bar(x - width, mape_vals, width, label="MAPE", color=C_PRESS, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, max_err_vals, width, label="Max error", color=C_GT, edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, [abs(b) for b in bias_vals], width, label="|Bias|", color=C_ENV, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([w.replace("_", "\n") for w in workloads_sorted], fontsize=8)
    ax.set_ylabel("Error (%)")
    ax.set_title("(b) Error by Workload")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.set_ylim(0, max(max_err_vals) * 1.25)

    # Annotate bias direction
    for i, b in enumerate(bias_vals):
        direction = "consv." if b > 0 else "optim."
        ax.annotate(f"{b:+.1f}%\n({direction})", xy=(i + width, abs(b)),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=7, color="#333333")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in [".pdf", ".png"]:
        path = OUTPUT_DIR / f"figure_press_calibration{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {path}")
    plt.close(fig)

    # Print claim text
    print("\n" + "=" * 64)
    print("PRESS CLAIM TEXT FOR PAPER")
    print("=" * 64)
    print(f"  Aggregate MAPE:  {stats['total_latency_mape']:.2f}%")
    print(f"  Aggregate RMSE:  {stats['total_latency_rmse']:.2f}%")
    print(f"  Max error:       {stats['total_latency_max_error_pct']:.2f}%")
    print(f"  R^2 (scatter):   {r_squared:.3f}")
    print("\n  Per-workload:")
    for w, s in by_w.items():
        print(f"    {w:22s}  MAPE={s['mape']:.2f}%  Max={s['max_error_pct']:.2f}%  Bias={s['mean_bias_pct']:+.2f}%")
    print("=" * 64)


if __name__ == "__main__":
    if not INPUT_JSON.exists():
        print(f"ERROR: {INPUT_JSON} not found. Run run_press_calibration.py first.")
        sys.exit(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_figure()
