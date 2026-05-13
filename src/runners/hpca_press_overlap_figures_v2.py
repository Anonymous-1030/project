"""
Regenerate PRESS overlap figures with clearer narrative.

Key messages:
  1. Policy ordering is invariant across natural and PRESS-matched runs.
  2. Latency ratio (FTS/SBFI) is preserved -- this is the pure ordering invariant.
  3. Queue utilization trend is matched; unmatched breaks it.
  4. Unmatched controls show large prediction error; PRESS-matched shows small error.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
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


def load():
    with open("outputs/hpca_rebuttal/press_overlap_validation.json") as f:
        return json.load(f)


def fig_invariant_preservation(data):
    """
    Figure 1: Three invariants preserved by PRESS-matched runs.
    Row 1: Policy ordering (throughput rank)
    Row 2: Latency ratio invariance (FTS/SBFI ratio)
    Row 3: Queue utilization trend
    """
    results = data["results"]
    targets = [
        ("8k", "8K Target"),
        ("16k", "16K Target"),
        ("32k", "32K Target"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    fig.suptitle(
        "PRESS Overlapping-Scale Validation: Invariant Preservation",
        fontsize=12,
    )

    for col, (target, title) in enumerate(targets):
        nat_key = f"natural_{target}"
        press_key = f"press_{target}_at_{int(target[:-1])//2}k"
        um_key = f"unmatched_{target}_at_{int(target[:-1])//2}k"

        # -- Row 1: Throughput (policy ordering) --
        ax1 = axes[0, col]
        x = np.arange(len(METHODS))
        width = 0.25
        nat_tps = [results[nat_key][m]["throughput_tps"] for m in METHODS]
        press_tps = [results[press_key][m]["throughput_tps"] for m in METHODS]
        um_tps = [results[um_key][m]["throughput_tps"] for m in METHODS]

        ax1.bar(x - width, nat_tps, width, label="Natural", color="#37474F")
        ax1.bar(x, press_tps, width, label="PRESS-matched", color="#00897B")
        ax1.bar(x + width, um_tps, width, label="Delib. unmatched", color="#E53935")
        ax1.set_ylabel("Throughput (tok/s)")
        if col == 0:
            ax1.set_title(f"{title}\nThroughput (Policy Ordering)")
        else:
            ax1.set_title(f"{title}\nThroughput")
        ax1.set_xticks(x)
        ax1.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=20, ha="right")
        ax1.grid(True, alpha=0.3, axis="y")
        if col == 2:
            ax1.legend(frameon=True, loc="upper right")

        # -- Row 2: Latency ratio invariance --
        ax2 = axes[1, col]
        # For each run, compute FTS/SBFI latency ratio
        def fts_ratio(key):
            fts_lat = results[key]["prose_fts"]["latency_us"]
            prose_lat = results[key]["prose"]["latency_us"]
            return fts_lat / max(prose_lat, 1e-9)

        ratios = [fts_ratio(nat_key), fts_ratio(press_key), fts_ratio(um_key)]
        colors = ["#37474F", "#00897B", "#E53935"]
        labels = ["Natural", "PRESS-matched", "Delib. unmatched"]
        bars = ax2.bar(labels, ratios, color=colors, width=0.5)
        ax2.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, label="No penalty")
        ax2.set_ylabel("FTS / SBFI Latency Ratio")
        if col == 0:
            ax2.set_title("Latency Ratio Invariance\n(FTS Penalty Preserved)")
        else:
            ax2.set_title("Latency Ratio Invariance")
        ax2.grid(True, alpha=0.3, axis="y")
        # Add value labels
        for bar, val in zip(bars, ratios):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{val:.1f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # -- Row 3: Queue utilization trend (PROSE only) --
        ax3 = axes[2, col]
        nat_rho = results[nat_key]["prose"]["queue_utilization"]
        press_rho = results[press_key]["prose"]["queue_utilization"]
        um_rho = results[um_key]["prose"]["queue_utilization"]

        bars3 = ax3.bar(labels, [nat_rho, press_rho, um_rho], color=colors, width=0.5)
        ax3.set_ylabel("Queue Utilization (rho)")
        if col == 0:
            ax3.set_title("Queue Utilization Trend\n(PROSE only)")
        else:
            ax3.set_title("Queue Utilization Trend")
        ax3.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars3, [nat_rho, press_rho, um_rho]):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    out = Path("outputs/hpca_rebuttal/fig_press_invariant_preservation.png")
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def fig_scaling_law_continuity(data):
    """
    Figure 2: PRESS projection as a continuous scaling law.
    X-axis: effective scale (4K, 8K, 16K, 32K)
    Three curves per method:
      - Natural family (solid)
      - PRESS-induced family (dashed)
      - Show that dashed curve is a shifted but parallel version of solid curve
    """
    results = data["results"]

    # Build natural family and press family
    natural_scales = {"4k": None, "8k": "natural_8k", "16k": "natural_16k", "32k": "natural_32k"}
    press_scales = {
        "4k": None,  # no press for 4K
        "8k": "press_8k_at_4k",
        "16k": "press_16k_at_8k",
        "32k": "press_32k_at_16k",
    }
    scales = [4, 8, 16, 32]
    scale_labels = ["4K", "8K", "16K", "32K"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("PRESS as Continuous Scaling Law", fontsize=12)

    metrics = [
        ("latency_us", "Latency (us)", axes[0, 0]),
        ("mean_recovery", "Recovery", axes[0, 1]),
        ("throughput_tps", "Throughput (tok/s)", axes[1, 0]),
        ("queue_utilization", "Queue Utilization (rho)", axes[1, 1]),
    ]

    for metric, ylabel, ax in metrics:
        for method in METHODS:
            nat_vals = []
            press_vals = []
            for scale_name in ("4k", "8k", "16k", "32k"):
                nat_key = natural_scales[scale_name]
                press_key = press_scales[scale_name]
                if nat_key:
                    nat_vals.append(results[nat_key][method][metric])
                else:
                    nat_vals.append(np.nan)
                if press_key:
                    press_vals.append(results[press_key][method][metric])
                else:
                    press_vals.append(np.nan)

            ax.plot(scales, nat_vals, marker="o", label=f"{METHOD_LABELS[method]} (natural)",
                   linewidth=2, linestyle="-", alpha=0.8)
            ax.plot(scales, press_vals, marker="s", label=f"{METHOD_LABELS[method]} (PRESS)",
                   linewidth=2, linestyle="--", alpha=0.8)

        ax.set_xlabel("Context Length")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(scales)
        ax.set_xticklabels(scale_labels)
        ax.grid(True, alpha=0.3)
        if metric == "latency_us":
            ax.legend(frameon=True, fontsize=7, loc="upper left")

    out = Path("outputs/hpca_rebuttal/fig_press_scaling_law.png")
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def fig_prediction_error_summary(data):
    """
    Figure 3: Prediction error summary.
    For each target and each method, show:
      - PRESS-matched relative error (should be small)
      - Deliberately unmatched relative error (should be large, especially for FTS)
    Focus on latency and rho.
    """
    errors = data["prediction_errors"]
    targets = [("8k", "8K"), ("16k", "16K"), ("32k", "32K")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("PRESS Prediction Error: Matched vs Deliberately Unmatched", fontsize=12)

    for idx, (metric, ylabel) in enumerate([("latency_press", "Latency Error"), ("rho_press", "Rho Error")]):
        ax = axes[idx]
        um_metric = metric.replace("press", "um")
        x = np.arange(len(METHODS))
        width = 0.25

        for col, (target, title) in enumerate(targets):
            press_errs = [errors[target][m][metric] for m in METHODS]
            um_errs = [errors[target][m][um_metric] for m in METHODS]
            offset = (col - 1) * width
            ax.bar(x + offset, press_errs, width, label=f"{title} matched", alpha=0.8)
            ax.bar(x + offset, [-e for e in um_errs], width, label=f"{title} unmatched", alpha=0.5)

        ax.set_ylabel(f"Relative Error ({ylabel})")
        ax.set_title(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=20, ha="right")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")
        if idx == 1:
            ax.legend(frameon=True, fontsize=7, loc="upper left")

    out = Path("outputs/hpca_rebuttal/fig_press_error_summary_v2.png")
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    data = load()
    fig_invariant_preservation(data)
    fig_scaling_law_continuity(data)
    fig_prediction_error_summary(data)
