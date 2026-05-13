#!/usr/bin/env python3
"""
Generate Figure 7.X: Real CXL Path Prototype Validation.

Two-panel figure for HPCA rebuttal:
  (a) CXL Transaction Volume & HBM Pollution: Fetch-Then-Score vs Metadata-First
  (b) End-to-End Latency & Throughput across context lengths

LaTeX-ready: vector fonts, single-column native size.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

INPUT_JSON = Path("outputs/hpca_real_cxl_prototype/prototype_results.json")
OUTPUT_DIR = Path("outputs/hpca_real_cxl_prototype")

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

C_FTS = "#D95319"    # Fetch-then-score (orange-red)
C_MF = "#4C72B0"     # Metadata-first (blue)
C_POLLUTION = "#C44E52"


def load_results():
    with open(INPUT_JSON) as f:
        data = json.load(f)
    return data["results"]


def generate_figure():
    results = load_results()
    ctx_lens = sorted({r["context_length"] for r in results})
    ctx_labels = [f"{c//1024}K" for c in ctx_lens]

    fts = [next(r for r in results if r["context_length"] == c and r["method"] == "fetch_then_score") for c in ctx_lens]
    mf = [next(r for r in results if r["context_length"] == c and r["method"] == "metadata_first") for c in ctx_lens]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fig.suptitle(
        "Real CXL-Path Prototype: Metadata-First vs Fetch-Then-Score (A100 + CXL 2.0)",
        fontsize=12, fontweight="bold", y=0.98,
    )

    # ── (a) CXL Bytes & HBM Pollution ────────────────────────────────
    ax = axes[0]
    x = np.arange(len(ctx_lens))
    width = 0.28

    fts_bytes = [r["cxl_bytes_transferred"] / 1e6 for r in fts]
    mf_bytes = [r["cxl_bytes_transferred"] / 1e6 for r in mf]
    fts_poll = [r["hbm_bytes_polluted"] / 1e6 for r in fts]
    mf_poll = [r["hbm_bytes_polluted"] / 1e6 for r in mf]

    ax.bar(x - width, fts_bytes, width, label="FTS CXL traffic", color=C_FTS, edgecolor="black", linewidth=0.5)
    ax.bar(x, mf_bytes, width, label="MF CXL traffic", color=C_MF, edgecolor="black", linewidth=0.5)
    ax.bar(x + width, fts_poll, width, label="FTS HBM pollution", color=C_FTS, edgecolor="black", linewidth=0.5, alpha=0.4, hatch="//")
    ax.bar(x + width, mf_poll, width, bottom=fts_poll, label="MF HBM pollution", color=C_MF, edgecolor="black", linewidth=0.5, alpha=0.4, hatch="\\")

    ax.set_xticks(x)
    ax.set_xticklabels(ctx_labels)
    ax.set_ylabel("MB per 100 steps")
    ax.set_title("(a) CXL Traffic & HBM Pollution")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7.5, ncol=2)
    ax.set_ylim(0, max(fts_bytes) * 1.35)

    # Annotate reduction
    for i, (fb, mb) in enumerate(zip(fts_bytes, mf_bytes)):
        reduction = (1 - mb / fb) * 100
        ax.annotate(f"-{reduction:.0f}%", xy=(i, mb), xytext=(i, mb + max(fts_bytes)*0.08),
                    ha="center", fontsize=8, fontweight="bold", color=C_MF)

    # ── (b) Latency & Throughput ─────────────────────────────────────
    ax1 = axes[1]
    ax2 = ax1.twinx()

    fts_lat = [r["avg_step_latency_us"] / 1000.0 for r in fts]  # ms
    mf_lat = [r["avg_step_latency_us"] / 1000.0 for r in mf]
    fts_tps = [r["throughput_tps"] for r in fts]
    mf_tps = [r["throughput_tps"] for r in mf]

    l1 = ax1.plot(x, fts_lat, "o-", color=C_FTS, label="FTS latency", linewidth=1.8, markersize=7)
    l2 = ax1.plot(x, mf_lat, "s-", color=C_MF, label="MF latency", linewidth=1.8, markersize=7)
    l3 = ax2.plot(x, fts_tps, "o--", color=C_FTS, label="FTS throughput", linewidth=1.5, markersize=6, alpha=0.7)
    l4 = ax2.plot(x, mf_tps, "s--", color=C_MF, label="MF throughput", linewidth=1.5, markersize=6, alpha=0.7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(ctx_labels)
    ax1.set_ylabel("Step latency (ms)")
    ax2.set_ylabel("Throughput (tok/s)")
    ax1.set_title("(b) End-to-End Latency & Throughput")
    ax1.set_ylim(0, max(fts_lat) * 1.25)
    ax2.set_ylim(0, max(mf_tps) * 1.3)

    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18),
               framealpha=0.9, fontsize=7.5, ncol=2)

    # Annotate latency reduction
    for i, (fl, ml) in enumerate(zip(fts_lat, mf_lat)):
        reduction = (1 - ml / fl) * 100
        ax1.annotate(f"-{reduction:.0f}%", xy=(i, ml), xytext=(i, ml + max(fts_lat)*0.06),
                     ha="center", fontsize=8, fontweight="bold", color=C_MF)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in [".pdf", ".png"]:
        path = OUTPUT_DIR / f"figure_real_cxl_prototype_validation{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {path}")
    plt.close(fig)

    # Print claim text
    print("\n" + "=" * 64)
    print("CLAIM TEXT FOR PAPER")
    print("=" * 64)
    for c, r_fts, r_mf in zip(ctx_lens, fts, mf):
        txn_red = (1 - r_mf["cxl_bytes_transferred"] / r_fts["cxl_bytes_transferred"]) * 100
        lat_red = (1 - r_mf["avg_step_latency_us"] / r_fts["avg_step_latency_us"]) * 100
        pol_red = (1 - r_mf["pollution_ratio"] / max(r_fts["pollution_ratio"], 1e-9)) * 100 if r_fts["pollution_ratio"] > 0 else 100.0
        print(f"  {c//1024}K: CXL traffic -{txn_red:.1f}%, latency -{lat_red:.1f}%, HBM pollution -{pol_red:.1f}%")
    print("=" * 64)


if __name__ == "__main__":
    if not INPUT_JSON.exists():
        print(f"ERROR: {INPUT_JSON} not found. Run run_real_cxl_prototype_eval.py first.")
        sys.exit(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_figure()
