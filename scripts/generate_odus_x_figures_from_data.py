#!/usr/bin/env python3
"""Generate ODUS-X figures from embedded JSON data (for immediate use)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = Path("outputs/hpca_odus_v2")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Embedded 3B validation result (from A100 run)
VALIDATION_JSON = {
  "aggregate": {
    "fixed_similarity": {"4096": {"mean_recovery": 0.231}, "8192": {"mean_recovery": 0.123}, "16384": {"mean_recovery": 0.17}},
    "fixed_recency": {"4096": {"mean_recovery": 0.745}, "8192": {"mean_recovery": 0.859}, "16384": {"mean_recovery": 0.656}},
    "fixed_random": {"4096": {"mean_recovery": 0.083}, "8192": {"mean_recovery": 0.116}, "16384": {"mean_recovery": 0.089}},
    "adaptive_gating": {"4096": {"mean_recovery": 0.926}, "8192": {"mean_recovery": 0.924}, "16384": {"mean_recovery": 0.823}},
    "adaptive_no_drift": {"4096": {"mean_recovery": 0.921}, "8192": {"mean_recovery": 0.919}, "16384": {"mean_recovery": 0.811}},
    "adaptive_no_similarity": {"4096": {"mean_recovery": 0.852}, "8192": {"mean_recovery": 0.905}, "16384": {"mean_recovery": 0.796}},
    "adaptive_no_pht": {"4096": {"mean_recovery": 0.824}, "8192": {"mean_recovery": 0.875}, "16384": {"mean_recovery": 0.481}},
  }
}


def plot_recovery_comparison():
    agg = VALIDATION_JSON["aggregate"]
    lengths = ["4096", "8192", "16384"]
    x = np.arange(len(lengths))
    width = 0.12

    policies = [
        ("fixed_similarity", "Sim-only", "#d62728"),
        ("fixed_recency", "Recency-only", "#ff7f0e"),
        ("fixed_random", "Random", "#7f7f7f"),
        ("adaptive_no_pht", "No PHT", "#9467bd"),
        ("adaptive_no_similarity", "No Similarity", "#8c564b"),
        ("adaptive_no_drift", "No Drift-Gating", "#e377c2"),
        ("adaptive_gating", "ODUS-X", "#2ca02c"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (key, label, color) in enumerate(policies):
        vals = [agg[key][L]["mean_recovery"] * 100 for L in lengths]
        offset = width * (idx - len(policies) / 2 + 0.5)
        bars = ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="black", linewidth=0.3)
        if key in ("adaptive_gating", "fixed_similarity"):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7)

    ax.set_ylabel("Selection Recovery (%)", fontsize=12)
    ax.set_xlabel("Context Length", fontsize=12)
    ax.set_title("ODUS-X vs. Fixed Baselines (Qwen2.5-3B, A100)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(["4K", "8K", "16K"])
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_odus_x_recovery.pdf", dpi=300)
    fig.savefig(OUTDIR / "fig_odus_x_recovery.png", dpi=300)
    print(f"[Saved] {OUTDIR / 'fig_odus_x_recovery.pdf'}")


if __name__ == "__main__":
    plot_recovery_comparison()
