"""
Generate HPCA Rebuttal Figures.

Produces publication-ready figures for:
  Fig 1: SBFI Irreducibility (4 metrics x 4 methods)
  Fig 2: Candidate Recall Curves (4 workloads)
  Fig 3: Summary Sensitivity Sweep
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Paper-friendly style
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

METHOD_LABELS = {
    "prose": "PROSE (SBFI)",
    "prose_fts": "PROSE-FTS",
    "freqrec_prefetcher": "FreqRec-PF",
    "stream_prefetcher": "StreamPF",
}

METHOD_COLORS = {
    "prose": "#2E7D32",      # Green
    "prose_fts": "#C62828",   # Red
    "freqrec_prefetcher": "#1565C0",  # Blue
    "stream_prefetcher": "#F57C00",   # Orange
}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def plot_sbfi_irreducibility(data: Dict[str, Any], output_dir: Path):
    """Fig: SBFI Irreducibility -- 4 subplots."""
    results = data["results"]
    seq_lens = [8192, 16384, 32768, 65536]
    methods = ["prose", "prose_fts", "freqrec_prefetcher", "stream_prefetcher"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("SBFI Irreducibility: PROSE vs Fetch-Then-Score Baselines", fontsize=12)

    metrics = [
        ("invalid_traffic_ratio", "Invalid Traffic Ratio", axes[0, 0]),
        ("hbm_pollution_rate", "HBM Pollution Rate", axes[0, 1]),
        ("p99_latency_us", "P99 Latency (us)", axes[1, 0]),
        ("queue_utilization", "Queue Utilization (rho)", axes[1, 1]),
    ]

    for metric, ylabel, ax in metrics:
        for method in methods:
            vals = [r[metric] for r in results[method]]
            ax.plot(seq_lens, vals, marker="o", label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method], linewidth=2)
        ax.set_xlabel("Context Length")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(seq_lens)
        ax.set_xticklabels([f"{s//1024}K" for s in seq_lens])
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fontsize=8)

    out = output_dir / "fig_sbfi_irreducibility.png"
    fig.savefig(out)
    logger.info(f"Saved {out}")
    plt.close(fig)


def plot_candidate_recall(data: Dict[str, Any], output_dir: Path):
    """Fig: Candidate Recall Curves -- 4 workloads."""
    results = data["results"]
    workloads = ["passkey", "ruler", "needle", "sequential"]
    k_values = [16, 32, 64, 128]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Candidate Recall Audit: Recall@N vs Consideration Set Size", fontsize=12)

    for idx, workload in enumerate(workloads):
        ax = axes[idx // 2, idx % 2]
        recalls = [results[workload]["recall_at_k"][str(k)] for k in k_values]
        ax.plot(k_values, recalls, marker="o", linewidth=2, color="#1565C0")
        ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="80% threshold")
        ax.set_xlabel("Consideration Set Size (N)")
        ax.set_ylabel("Recall@N")
        ax.set_title(workload.capitalize())
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fontsize=8)

    out = output_dir / "fig_candidate_recall.png"
    fig.savefig(out)
    logger.info(f"Saved {out}")
    plt.close(fig)


def plot_summary_sensitivity(data: Dict[str, Any], output_dir: Path):
    """Fig: Summary Sensitivity Sweep."""
    results = data["results"]
    seq_lens = [8192, 16384, 32768]
    latencies = [88, 200, 500, 1000]

    fig, ax = plt.subplots(figsize=(7, 5))

    for seq_len in seq_lens:
        tps_vals = []
        for lat_ns in latencies:
            key = f"{lat_ns}ns"
            for row in results[key]:
                if row["seq_len"] == seq_len:
                    tps_vals.append(row["throughput_tps"])
                    break
        ax.plot(latencies, tps_vals, marker="o", label=f"{seq_len//1024}K context",
                linewidth=2)

    ax.set_xlabel("Summary Read Latency (ns)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Summary Interface Sensitivity: Throughput vs Summary Latency")
    ax.set_xscale("log")
    ax.set_xticks(latencies)
    ax.set_xticklabels([f"{l}" for l in latencies])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)

    out = output_dir / "fig_summary_sensitivity.png"
    fig.savefig(out)
    logger.info(f"Saved {out}")
    plt.close(fig)


def plot_throughput_comparison(data: Dict[str, Any], output_dir: Path):
    """Additional figure: Throughput comparison across methods."""
    results = data["results"]
    seq_lens = [8192, 16384, 32768, 65536]
    methods = ["prose", "prose_fts", "freqrec_prefetcher", "stream_prefetcher"]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(seq_lens))
    width = 0.18

    for i, method in enumerate(methods):
        tps_vals = [r["throughput_tps"] for r in results[method]]
        ax.bar(x + i * width, tps_vals, width, label=METHOD_LABELS[method],
               color=METHOD_COLORS[method])

    ax.set_xlabel("Context Length")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput Comparison: PROSE vs Baselines")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{s//1024}K" for s in seq_lens])
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3, axis="y")

    out = output_dir / "fig_throughput_comparison.png"
    fig.savefig(out)
    logger.info(f"Saved {out}")
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/hpca_rebuttal")
    parser.add_argument("--output-dir", default="outputs/hpca_rebuttal")
    args = parser.parse_args()

    indir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    expA = load_json(indir / "expA_sbfi_irreducibility.json")
    expB = load_json(indir / "expB_candidate_recall.json")
    expC = load_json(indir / "expC_summary_sensitivity.json")

    plot_sbfi_irreducibility(expA, outdir)
    plot_candidate_recall(expB, outdir)
    plot_summary_sensitivity(expC, outdir)
    plot_throughput_comparison(expA, outdir)

    logger.info(f"All figures saved to {outdir}")


if __name__ == "__main__":
    main()
