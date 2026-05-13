"""
Generate facet-grid style PRESS overlap validation figure.

Layout: 4 rows (methods) x 3 columns (target scales).
Each small panel shows natural vs PRESS-matched vs unmatched
for one method at one target, across 4 metrics as grouped bars.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

METHODS = ["prose", "prose_fts", "freqrec_prefetcher", "stream_prefetcher"]
METHOD_LABELS = {
    "prose": "PROSE",
    "prose_fts": "PROSE-FTS",
    "freqrec_prefetcher": "FreqRec-PF",
    "stream_prefetcher": "StreamPF",
}

METRICS = [
    ("latency_us", "Latency (us)"),
    ("mean_recovery", "Recovery"),
    ("throughput_tps", "Throughput (k tok/s)"),
    ("queue_utilization", "Queue Util (rho)"),
]

TARGETS = [
    ("8k", "8K Target\n(4K induced)"),
    ("16k", "16K Target\n(8K induced)"),
    ("32k", "32K Target\n(16K induced)"),
]


def load():
    with open("outputs/hpca_rebuttal/press_overlap_validation.json") as f:
        return json.load(f)


def fig_facet_grid(data):
    """4x3 facet grid: rows=methods, cols=targets."""
    results = data["results"]
    fig, axes = plt.subplots(4, 3, figsize=(10, 11))
    fig.suptitle(
        "PRESS Overlapping-Scale Validation: 4 Methods x 3 Targets x 4 Metrics",
        fontsize=11,
    )

    for row, method in enumerate(METHODS):
        for col, (target, title) in enumerate(TARGETS):
            ax = axes[row, col]
            nat_key = f"natural_{target}"
            press_key = f"press_{target}_at_{int(target[:-1])//2}k"
            um_key = f"unmatched_{target}_at_{int(target[:-1])//2}k"

            # Get 4 metrics for this method
            nat_vals = [results[nat_key][method][m] for m, _ in METRICS]
            press_vals = [results[press_key][method][m] for m, _ in METRICS]
            um_vals = [results[um_key][method][m] for m, _ in METRICS]

            # Scale throughput to k tok/s for readability
            nat_vals[2] /= 1000.0
            press_vals[2] /= 1000.0
            um_vals[2] /= 1000.0

            x = np.arange(len(METRICS))
            width = 0.22

            ax.bar(x - width, nat_vals, width, label="Natural", color="#37474F", edgecolor="white", linewidth=0.5)
            ax.bar(x, press_vals, width, label="PRESS-matched", color="#00897B", edgecolor="white", linewidth=0.5)
            ax.bar(x + width, um_vals, width, label="Unmatched", color="#E53935", edgecolor="white", linewidth=0.5)

            if row == 0:
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(METHOD_LABELS[method], fontsize=9, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([m[1] for m in METRICS], rotation=25, ha="right", fontsize=6.5)
            ax.grid(True, alpha=0.2, axis="y")
            ax.tick_params(axis="y", labelsize=6.5)

            if row == 0 and col == 2:
                ax.legend(frameon=True, loc="upper right", fontsize=6.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path("outputs/hpca_rebuttal/fig_press_facet_grid.png")
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def fig_compact_invariants(data):
    """
    3x3 compact grid showing three invariants:
    Row 1: Policy ordering (throughput rank correlation)
    Row 2: Latency ratio invariance (FTS/SBFI)
    Row 3: Queue utilization trend (PROSE only)
    Cols: 8K, 16K, 32K targets
    """
    results = data["results"]
    fig, axes = plt.subplots(3, 3, figsize=(9, 8))
    fig.suptitle("PRESS Invariant Preservation", fontsize=11)

    for col, (target, title) in enumerate(TARGETS):
        nat_key = f"natural_{target}"
        press_key = f"press_{target}_at_{int(target[:-1])//2}k"
        um_key = f"unmatched_{target}_at_{int(target[:-1])//2}k"

        # Row 1: Throughput bars (policy ordering)
        ax1 = axes[0, col]
        x = np.arange(len(METHODS))
        w = 0.22
        tps_nat = [results[nat_key][m]["throughput_tps"]/1000 for m in METHODS]
        tps_press = [results[press_key][m]["throughput_tps"]/1000 for m in METHODS]
        tps_um = [results[um_key][m]["throughput_tps"]/1000 for m in METHODS]
        ax1.bar(x-w, tps_nat, w, color="#37474F", edgecolor="white", linewidth=0.5)
        ax1.bar(x, tps_press, w, color="#00897B", edgecolor="white", linewidth=0.5)
        ax1.bar(x+w, tps_um, w, color="#E53935", edgecolor="white", linewidth=0.5)
        ax1.set_title(title, fontsize=9)
        if col == 0:
            ax1.set_ylabel("Throughput\n(k tok/s)", fontsize=8)
        ax1.set_xticks(x)
        ax1.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=20, ha="right", fontsize=7)
        ax1.grid(True, alpha=0.2, axis="y")
        if col == 2:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#37474F", label="Natural"),
                Patch(facecolor="#00897B", label="PRESS"),
                Patch(facecolor="#E53935", label="Unmatched"),
            ]
            ax1.legend(handles=legend_elements, frameon=True, fontsize=7, loc="upper right")

        # Row 2: FTS/SBFI latency ratio
        ax2 = axes[1, col]
        def ratio(k):
            return results[k]["prose_fts"]["latency_us"] / max(results[k]["prose"]["latency_us"], 1e-9)
        vals = [ratio(nat_key), ratio(press_key), ratio(um_key)]
        colors = ["#37474F", "#00897B", "#E53935"]
        bars = ax2.bar(["Natural", "PRESS", "Unmatched"], vals, color=colors, width=0.5, edgecolor="white")
        ax2.axhline(1.0, color="black", linestyle="--", alpha=0.3)
        if col == 0:
            ax2.set_ylabel("FTS / SBFI\nLatency Ratio", fontsize=8)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f"{v:.1f}x",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax2.grid(True, alpha=0.2, axis="y")

        # Row 3: Queue utilization (PROSE only)
        ax3 = axes[2, col]
        rho_vals = [
            results[nat_key]["prose"]["queue_utilization"],
            results[press_key]["prose"]["queue_utilization"],
            results[um_key]["prose"]["queue_utilization"],
        ]
        bars3 = ax3.bar(["Natural", "PRESS", "Unmatched"], rho_vals, color=colors, width=0.5, edgecolor="white")
        if col == 0:
            ax3.set_ylabel("Queue Util\n(PROSE, rho)", fontsize=8)
        for bar, v in zip(bars3, rho_vals):
            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax3.grid(True, alpha=0.2, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path("outputs/hpca_rebuttal/fig_press_compact_invariants.png")
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def fig_scaling_matrix(data):
    """
    A matrix of small line plots: 3 rows (metrics) x 4 cols (methods).
    Each panel shows natural (solid) and PRESS (dashed) scaling curves.
    """
    results = data["results"]
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    fig.suptitle("PRESS Scaling Law Matrix: Natural vs PRESS-Matched", fontsize=11)

    metrics = [
        ("latency_us", "Latency (us)"),
        ("throughput_tps", "Throughput (tok/s)"),
        ("queue_utilization", "Queue Util (rho)"),
    ]

    scales = [8, 16, 32]
    scale_labels = ["8K", "16K", "32K"]

    for row, (metric, ylabel) in enumerate(metrics):
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            nat_vals = []
            press_vals = []
            for target in ("8k", "16k", "32k"):
                nat_key = f"natural_{target}"
                press_key = f"press_{target}_at_{int(target[:-1])//2}k"
                v_nat = results[nat_key][method][metric]
                v_press = results[press_key][method][metric]
                if metric == "throughput_tps":
                    v_nat /= 1000.0
                    v_press /= 1000.0
                nat_vals.append(v_nat)
                press_vals.append(v_press)

            ax.plot(scales, nat_vals, marker="o", color="#37474F", linewidth=2, label="Natural")
            ax.plot(scales, press_vals, marker="s", linestyle="--", color="#00897B", linewidth=2, label="PRESS")

            if row == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            ax.set_xticks(scales)
            ax.set_xticklabels(scale_labels, fontsize=7)
            ax.grid(True, alpha=0.2)
            if row == 0 and col == 3:
                ax.legend(frameon=True, fontsize=6.5, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path("outputs/hpca_rebuttal/fig_press_scaling_matrix.png")
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    data = load()
    fig_facet_grid(data)
    fig_compact_invariants(data)
    fig_scaling_matrix(data)
