"""
Multi-Tenant Figure Generator.

Reads multi-tenant evaluation JSON results and generates publication-quality
figures for:
  - Tail latency degradation waterfall (D1)
  - Priority inversion histogram (D3)
  - CXL utilization vs tenant count (D5)
  - Fairness index heatmap
  - Token bucket isolation comparison (D4)
  - Dynamic arrival warm-up curves (D6)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Attempt matplotlib import (optional — figures are skipped if unavailable)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ═══════════════════════════════════════════════════════════════════════════
# Color palette
# ═══════════════════════════════════════════════════════════════════════════

PROSE_COLOR = "#1f77b4"
FTS_COLOR = "#d62728"
SHORT_COLOR = "#2ca02c"
LONG_COLOR = "#ff7f0e"
GRID_COLOR = "#e0e0e0"

BASELINE_COLORS = {
    "PROSE": "#1f77b4",
    "PROSE-FTS": "#d62728",
    "SnapKV": "#9467bd",
    "H2O": "#8c564b",
    "Quest": "#e377c2",
}


# ═══════════════════════════════════════════════════════════════════════════
# Plotting utilities
# ═══════════════════════════════════════════════════════════════════════════

def _setup_figure(figsize=(10, 6), dpi=150):
    if not HAS_MPL:
        return None, None
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.grid(True, alpha=0.3, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def _save_figure(fig, filepath: str) -> str:
    if fig is None:
        return ""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Figure saved: {filepath}")
    return filepath


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Tail Latency Degradation Waterfall (D1)
# ═══════════════════════════════════════════════════════════════════════════

def fig_tail_latency_waterfall(
    results: Dict[str, Any],
    output_path: str = "outputs/multitenant/fig_d1_tail_degradation.pdf",
):
    """Waterfall chart showing tail latency degradation per mixing scenario."""
    if not HAS_MPL:
        print("  [skip] matplotlib not available")
        return

    dim_data = results.get("dimensions", {}).get("D1_heterogeneous_mixing", {})
    scenarios = dim_data.get("scenarios", [])

    if not scenarios:
        print("  [skip] No D1 data")
        return

    names = [s["name"] for s in scenarios]
    degradations = [s["aggregate"]["max_tail_degradation"] for s in scenarios]
    short_p99s = [s["aggregate"].get("short_context_p99_us", 0) for s in scenarios]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Left: tail degradation ratio
    x = np.arange(len(names))
    bars = ax1.bar(x, degradations, color=PROSE_COLOR, alpha=0.85, width=0.5)
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Solo baseline")
    ax1.axhline(y=1.2, color="orange", linestyle=":", linewidth=1, alpha=0.7, label="1.2x claim bound")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Tail Degradation Ratio\n(P99 multi / P99 solo)")
    ax1.set_title("Tail Latency Degradation\nby Context-Length Mix", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, color=GRID_COLOR)

    # Annotate bars
    for bar, val in zip(bars, degradations):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Right: short context P99 latency
    colors_r = [SHORT_COLOR if n.startswith(("4K", "4K+")) else LONG_COLOR for n in names]
    bars2 = ax2.bar(x, short_p99s, color=colors_r, alpha=0.85, width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Short Request P99 Latency (us)")
    ax2.set_title("Short-Context Tail Latency\nUnder Load Mixing", fontweight="bold")
    ax2.grid(True, alpha=0.3, color=GRID_COLOR)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SHORT_COLOR, label="Short context (≤8K)"),
        Patch(facecolor=LONG_COLOR, label="Long context (≥128K)"),
    ]
    ax2.legend(handles=legend_elements, fontsize=8)

    fig.suptitle("D1: Heterogeneous Context-Length Mixing", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_figure(fig, output_path)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Priority Inversion Comparison (D3)
# ═══════════════════════════════════════════════════════════════════════════

def fig_priority_inversion(
    results: Dict[str, Any],
    output_path: str = "outputs/multitenant/fig_d3_priority_inversion.pdf",
):
    """Bar chart comparing PROSE SBFI vs FTS on priority inversion impact."""
    if not HAS_MPL:
        print("  [skip] matplotlib not available")
        return

    dim_data = results.get("dimensions", {}).get("D3_priority_inversion", {})
    comparison = dim_data.get("comparison", {})
    granularity = dim_data.get("dma_granularity_analysis", [])

    if not comparison:
        print("  [skip] No D3 data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Left: SBFI vs FTS high-priority P99 latency
    methods = ["PROSE (SBFI)", "PROSE-FTS"]
    p99_vals = [comparison.get("sbfi_high_p99_us", 0), comparison.get("fts_high_p99_us", 0)]
    inv_vals = [comparison.get("sbfi_invalid_traffic", 0), comparison.get("fts_invalid_traffic", 0)]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, p99_vals, width, color=[PROSE_COLOR, FTS_COLOR],
                    alpha=0.85, label="High-Priority P99 Latency (us)")
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width / 2, [v * 100 for v in inv_vals], width,
                         color="gray", alpha=0.4, label="Invalid Traffic %")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.set_ylabel("P99 Latency (us)")
    ax1.set_title("Priority-Inversion Impact\non High-Priority Request", fontweight="bold")
    ax1.grid(True, alpha=0.3, color=GRID_COLOR)
    ax1_twin.set_ylabel("Invalid Traffic (%)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # Annotate
    for bar, val in zip(bars1, p99_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", fontsize=10, fontweight="bold")

    # Right: DMA granularity vs inversion probability
    if granularity:
        dma_sizes = [g["dma_granularity_bytes"] for g in granularity]
        inv_probs = [g["inversion_probability"] for g in granularity]
        wait_times = [g["approximate_wait_time_us"] for g in granularity]

        ax2.plot(dma_sizes, inv_probs, "o-", color=FTS_COLOR, linewidth=2, markersize=8,
                label="Inversion probability")
        ax2.set_xscale("log")
        ax2.set_xlabel("DMA Granularity (bytes)")
        ax2.set_ylabel("Inversion Probability")
        ax2.set_title("Inversion Risk vs DMA Size\n(SBFI operates at 64B, FTS at 64KB)", fontweight="bold")
        ax2.axvline(x=64, color=PROSE_COLOR, linestyle="--", alpha=0.7, label="SBFI (64B)")
        ax2.axvline(x=65536, color=FTS_COLOR, linestyle="--", alpha=0.7, label="FTS (64KB)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, color=GRID_COLOR)

    fig.suptitle("D3: Priority Inversion Quantification", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_figure(fig, output_path)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: CXL Utilization vs Tenant Count (D5)
# ═══════════════════════════════════════════════════════════════════════════

def fig_metadata_contention(
    results: Dict[str, Any],
    output_path: str = "outputs/multitenant/fig_d5_metadata_contention.pdf",
):
    """CXL link utilization, latency, and fairness vs tenant count."""
    if not HAS_MPL:
        print("  [skip] matplotlib not available")
        return

    dim_data = results.get("dimensions", {}).get("D5_metadata_contention", {})
    data = dim_data.get("results", [])

    if not data:
        print("  [skip] No D5 data")
        return

    tenants = [d["num_tenants"] for d in data]
    rhos = [d["avg_cxl_queue_rho"] for d in data]
    p99s = [d["avg_p99_latency_us"] for d in data]
    fairness = [d["fairness_index"] for d in data]
    metadata_pressure = [d["metadata_pressure"] for d in data]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    (ax1, ax2), (ax3, ax4) = axes

    # Top-left: CXL queue utilization ρ
    ax1.plot(tenants, rhos, "o-", color=PROSE_COLOR, linewidth=2, markersize=8)
    ax1.axhline(y=0.80, color="red", linestyle="--", alpha=0.6, label="Knee (ρ=0.80)")
    ax1.axhline(y=0.95, color="darkred", linestyle=":", alpha=0.6, label="Saturation (ρ=0.95)")
    ax1.set_xlabel("Number of Tenants")
    ax1.set_ylabel("CXL Queue Utilization ρ")
    ax1.set_title("Link Utilization vs Tenants", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, color=GRID_COLOR)
    ax1.set_xticks(tenants)

    # Top-right: P99 latency
    ax2.plot(tenants, p99s, "s-", color=FTS_COLOR, linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Tenants")
    ax2.set_ylabel("Avg P99 Latency (us)")
    ax2.set_title("Tail Latency vs Tenants", fontweight="bold")
    ax2.grid(True, alpha=0.3, color=GRID_COLOR)
    ax2.set_xticks(tenants)

    # Bottom-left: Fairness index
    ax3.plot(tenants, fairness, "D-", color=SHORT_COLOR, linewidth=2, markersize=8)
    ax3.set_ylim(0.5, 1.05)
    ax3.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax3.set_xlabel("Number of Tenants")
    ax3.set_ylabel("Jain's Fairness Index")
    ax3.set_title("Fairness vs Tenants", fontweight="bold")
    ax3.grid(True, alpha=0.3, color=GRID_COLOR)
    ax3.set_xticks(tenants)

    # Bottom-right: Metadata pressure
    ax4.bar(tenants, metadata_pressure, color=LONG_COLOR, alpha=0.7, width=0.5)
    ax4.set_xlabel("Number of Tenants")
    ax4.set_ylabel("Metadata Pressure (ρ × K)")
    ax4.set_title("Cumulative Metadata Load", fontweight="bold")
    ax4.grid(True, alpha=0.3, color=GRID_COLOR, axis="y")
    ax4.set_xticks(tenants)

    fig.suptitle("D5: Metadata Contention Under Tenant Scaling", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_figure(fig, output_path)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Token Bucket Isolation (D4)
# ═══════════════════════════════════════════════════════════════════════════

def fig_token_bucket_isolation(
    results: Dict[str, Any],
    output_path: str = "outputs/multitenant/fig_d4_token_bucket.pdf",
):
    """Comparison of with/without token bucket on tenant B (short context)."""
    if not HAS_MPL:
        print("  [skip] matplotlib not available")
        return

    dim_data = results.get("dimensions", {}).get("D4_token_bucket_isolation", {})
    with_tb = dim_data.get("with_token_bucket", {})
    without_tb = dim_data.get("without_token_bucket", {})
    summary = dim_data.get("summary", {})

    if not with_tb or not without_tb:
        print("  [skip] No D4 data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Left: B's latency comparison
    metrics_list = [
        ("B P99 Latency", summary.get("B_latency_with_tb_us", 0), summary.get("B_latency_without_tb_us", 0)),
        ("B Tail Degradation", summary.get("B_degradation_with_tb", 0), summary.get("B_degradation_without_tb", 0)),
    ]

    x = np.arange(len(metrics_list))
    width = 0.3
    bars1 = ax1.bar(x - width / 2, [m[1] for m in metrics_list], width,
                    color=PROSE_COLOR, alpha=0.85, label="With Token Bucket")
    bars2 = ax1.bar(x + width / 2, [m[2] for m in metrics_list], width,
                    color=FTS_COLOR, alpha=0.85, label="Without Token Bucket")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[0] for m in metrics_list], fontsize=10)
    ax1.set_ylabel("Value")
    ax1.set_title("Short Tenant (B) Under\nBandwidth Pressure from A", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, color=GRID_COLOR, axis="y")

    # Right: Starvation comparison
    starv_labels = ["With TB", "Without TB"]
    starv_vals = [
        summary.get("starvation_with_tb", 0),
        summary.get("starvation_without_tb", 0),
    ]
    colors = [PROSE_COLOR, FTS_COLOR]
    bars3 = ax2.bar(starv_labels, starv_vals, color=colors, alpha=0.85, width=0.4)
    ax2.set_ylabel("Starvation Events")
    ax2.set_title("Starvation Events for Tenant B", fontweight="bold")
    ax2.grid(True, alpha=0.3, color=GRID_COLOR, axis="y")
    for bar, val in zip(bars3, starv_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", fontsize=11, fontweight="bold")

    isolation_ratio = summary.get("isolation_improvement", 1.0)
    fig.suptitle(
        f"D4: Token Bucket Isolation (Isolation improvement: {isolation_ratio:.1f}x)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save_figure(fig, output_path)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Dynamic Arrival Warm-up (D6)
# ═══════════════════════════════════════════════════════════════════════════

def fig_dynamic_arrival(
    results: Dict[str, Any],
    output_path: str = "outputs/multitenant/fig_d6_dynamic_arrival.pdf",
):
    """Cold-start latency and PHT warm-up curves."""
    if not HAS_MPL:
        print("  [skip] matplotlib not available")
        return

    dim_data = results.get("dimensions", {}).get("D6_dynamic_arrival", {})
    arrival_results = dim_data.get("results", [])
    pht_warmup = dim_data.get("pht_warmup_analysis", {})

    if not arrival_results:
        print("  [skip] No D6 data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Left: Cold start latency by arrival mode
    modes = [r["arrival_mode"] for r in arrival_results]
    cold_starts = [r["avg_cold_start_us"] for r in arrival_results]
    colors = [PROSE_COLOR, FTS_COLOR, SHORT_COLOR]
    ax1.bar(modes, cold_starts, color=colors[:len(modes)], alpha=0.85, width=0.4)
    ax1.set_ylabel("Avg Cold-Start Latency (us)")
    ax1.set_title("Cold-Start Latency by Arrival Pattern", fontweight="bold")
    ax1.grid(True, alpha=0.3, color=GRID_COLOR, axis="y")
    for i, (mode, val) in enumerate(zip(modes, cold_starts)):
        ax1.text(i, val + 5, f"{val:.0f}", ha="center", fontsize=10, fontweight="bold")

    # Right: PHT warm-up penalty per tenant
    if pht_warmup:
        tenants = list(pht_warmup.keys())
        warmups = [pht_warmup[t]["warmup_avg_recovery"] for t in tenants]
        steadies = [pht_warmup[t]["steady_avg_recovery"] for t in tenants]

        x = np.arange(len(tenants))
        width = 0.3
        ax2.bar(x - width / 2, warmups, width, color=FTS_COLOR, alpha=0.7,
               label="Warm-up (first 10 steps)")
        ax2.bar(x + width / 2, steadies, width, color=PROSE_COLOR, alpha=0.7,
               label="Steady state (steps 40-60)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(tenants, fontsize=10)
        ax2.set_ylabel("Avg Recovery")
        ax2.set_title("PHT Warm-Up vs Steady State", fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, color=GRID_COLOR, axis="y")

    fig.suptitle("D6: Dynamic Arrival and Departure", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_figure(fig, output_path)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: Scenario Summary Dashboard
# ═══════════════════════════════════════════════════════════════════════════

def fig_summary_dashboard(
    results: Dict[str, Any],
    output_path: str = "outputs/multitenant/fig_summary_dashboard.pdf",
):
    """Single-page dashboard with key multi-tenant metrics."""
    if not HAS_MPL:
        print("  [skip] matplotlib not available")
        return

    summary = results.get("summary", {})

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150)
    axes = axes.flatten()

    # Collect key metrics across dimensions
    dims = results.get("dimensions", {})

    # D1: max tail degradation
    d1_summary = dims.get("D1_heterogeneous_mixing", {}).get("summary", {})
    worst_degradation = d1_summary.get("worst_case_tail_degradation", 1.0)
    axes[0].bar(["Worst-case"], [worst_degradation], color=FTS_COLOR, alpha=0.85, width=0.3)
    axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[0].axhline(y=1.2, color="orange", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Degradation Ratio")
    axes[0].set_title("D1: Max Tail Degradation", fontweight="bold")

    # D2: starvation bound
    d2_summary = dims.get("D2_priority_qos", {}).get("summary", {})
    bound_held = d2_summary.get("starvation_bound_held", False)
    knee = d2_summary.get("slo_violation_knee", "N/A")
    axes[1].text(0.5, 0.5,
                f"Starvation ≤ 2 epochs: {bound_held}\nSLO knee: {knee} tenants",
                transform=axes[1].transAxes, ha="center", va="center",
                fontsize=12, bbox=dict(boxstyle="round", facecolor="lightgreen" if bound_held else "salmon"))
    axes[1].set_title("D2: Starvation Bound", fontweight="bold")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # D3: SBFI advantage
    d3_comparison = dims.get("D3_priority_inversion", {}).get("comparison", {})
    advantage = d3_comparison.get("sbfi_advantage_us", 0)
    inv_reduction = dims.get("D3_priority_inversion", {}).get("summary", {}).get("inversion_wait_reduced_by_percent", 0)
    axes[2].bar(["SBFI vs FTS"], [inv_reduction], color=PROSE_COLOR, alpha=0.85, width=0.3)
    axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("Reduction (%)")
    axes[2].set_title("D3: Inversion Reduction", fontweight="bold")

    # D4: isolation improvement
    d4_summary = dims.get("D4_token_bucket_isolation", {}).get("summary", {})
    isolation = d4_summary.get("isolation_improvement", 1.0)
    axes[3].bar(["Isolation Factor"], [isolation], color=SHORT_COLOR, alpha=0.85, width=0.3)
    axes[3].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[3].set_ylabel("Improvement Factor")
    axes[3].set_title("D4: Token Bucket Benefit", fontweight="bold")

    # D5: metadata knee
    d5_summary = dims.get("D5_metadata_contention", {}).get("summary", {})
    md_knee = d5_summary.get("metadata_knee_tenants", "N/A")
    max_rho = d5_summary.get("max_rho", 0)
    axes[4].text(0.5, 0.5,
                f"Metadata knee: {md_knee} tenants\nMax ρ: {max_rho:.3f}",
                transform=axes[4].transAxes, ha="center", va="center",
                fontsize=12, bbox=dict(boxstyle="round", facecolor="lightyellow"))
    axes[4].set_title("D5: Metadata Contention", fontweight="bold")
    axes[4].set_xticks([])
    axes[4].set_yticks([])

    # D6: warm-up
    d6_summary = dims.get("D6_dynamic_arrival", {}).get("summary", {})
    max_penalty = d6_summary.get("max_warmup_penalty", 0)
    axes[5].bar(["Max Warm-up Penalty"], [max_penalty], color=LONG_COLOR, alpha=0.85, width=0.3)
    axes[5].set_ylabel("Recovery Drop")
    axes[5].set_title("D6: Cold-Start Penalty", fontweight="bold")

    fig.suptitle("PROSE Multi-Tenant Evaluation — Summary Dashboard",
                fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_figure(fig, output_path)


# ═══════════════════════════════════════════════════════════════════════════
# Batch generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_figures(
    results_json_path: str,
    output_dir: str = "outputs/multitenant",
) -> List[str]:
    """Generate all multi-tenant figures from a results JSON file.

    Args:
        results_json_path: Path to the multi-tenant results JSON.
        output_dir: Directory for output figure files.

    Returns:
        List of generated figure file paths.
    """
    if not HAS_MPL:
        print("matplotlib not available — skipping figure generation.")
        return []

    with open(results_json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    paths: List[str] = []

    generators = [
        ("fig_d1_tail_degradation.pdf", fig_tail_latency_waterfall),
        ("fig_d3_priority_inversion.pdf", fig_priority_inversion),
        ("fig_d5_metadata_contention.pdf", fig_metadata_contention),
        ("fig_d4_token_bucket.pdf", fig_token_bucket_isolation),
        ("fig_d6_dynamic_arrival.pdf", fig_dynamic_arrival),
        ("fig_summary_dashboard.pdf", fig_summary_dashboard),
    ]

    for filename, gen_fn in generators:
        out_path = os.path.join(output_dir, filename)
        try:
            gen_fn(results, out_path)
            paths.append(out_path)
        except Exception as e:
            print(f"  [error] {filename}: {e}")

    print(f"\nGenerated {len(paths)} figures in {output_dir}")
    return paths


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Tenant Figure Generator")
    parser.add_argument("results_json", type=str,
                       help="Path to multi-tenant results JSON file")
    parser.add_argument("--output-dir", type=str, default="outputs/multitenant",
                       help="Output directory for figures")
    parser.add_argument("--figure", type=str, default="all",
                       choices=["all", "d1", "d3", "d4", "d5", "d6", "summary"],
                       help="Which figure(s) to generate")
    args = parser.parse_args()

    if not os.path.exists(args.results_json):
        print(f"Error: results file not found: {args.results_json}")
        sys.exit(1)

    if args.figure == "all":
        generate_all_figures(args.results_json, args.output_dir)
    else:
        with open(args.results_json, "r", encoding="utf-8") as f:
            results = json.load(f)

        figure_map = {
            "d1": ("fig_d1_tail_degradation.pdf", fig_tail_latency_waterfall),
            "d3": ("fig_d3_priority_inversion.pdf", fig_priority_inversion),
            "d4": ("fig_d4_token_bucket.pdf", fig_token_bucket_isolation),
            "d5": ("fig_d5_metadata_contention.pdf", fig_metadata_contention),
            "d6": ("fig_d6_dynamic_arrival.pdf", fig_dynamic_arrival),
            "summary": ("fig_summary_dashboard.pdf", fig_summary_dashboard),
        }

        filename, gen_fn = figure_map[args.figure]
        out_path = os.path.join(args.output_dir, filename)
        gen_fn(results, out_path)
