#!/usr/bin/env python3
"""Generate ODUS-X validation figures for HPCA submission."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

OUTDIR = Path("outputs/hpca_odus_v2")
OUTDIR.mkdir(parents=True, exist_ok=True)


def plot_recovery_comparison():
    json_path = OUTDIR / "odus_x_validation.json"
    if not json_path.exists():
        print(f"[SKIP] {json_path} not found.")
        return

    data = json.load(open(json_path))
    agg = data["aggregate"]
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
        # Add value labels on top for key bars
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


def plot_drift_validation():
    json_path = OUTDIR / "odus_x_drift_validation.json"
    if not json_path.exists():
        print(f"[SKIP] {json_path} not found.")
        return

    data = json.load(open(json_path))
    agg = data["aggregate"]
    steps = ["step0", "step1", "step2"]
    x = np.arange(len(steps))
    width = 0.2

    policies = [
        ("fixed_similarity", "Sim-only", "#d62728"),
        ("fixed_recency", "Recency-only", "#ff7f0e"),
        ("fixed_stable", "Stable-only", "#1f77b4"),
        ("fixed_reactive", "Reactive-only", "#bcbd22"),
        ("adaptive_gating", "ODUS-X", "#2ca02c"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (key, label, color) in enumerate(policies):
        vals = [agg[key][s]["mean_recovery"] * 100 for s in steps]
        offset = width * (idx - len(policies) / 2 + 0.5)
        ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="black", linewidth=0.3)

    ax.set_ylabel("Selection Recovery (%)", fontsize=12)
    ax.set_xlabel("Query Step", fontsize=12)
    ax.set_title("Drift-Heavy Workload: Multi-Hop Recovery (8K Context)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1 (stable)", "Q2 (transition)", "Q3 (needle/drift)"])
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Add drift annotation
    drift = data.get("drift_levels", [0, 0, 0])
    ax.text(0.02, 0.98, f"Mean drift: Q1={drift[0]:.2f}, Q2={drift[1]:.2f}, Q3={drift[2]:.2f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_odus_x_drift.pdf", dpi=300)
    fig.savefig(OUTDIR / "fig_odus_x_drift.png", dpi=300)
    print(f"[Saved] {OUTDIR / 'fig_odus_x_drift.pdf'}")


def plot_7b_scale():
    json_path = OUTDIR / "odus_x_validation.json"
    if not json_path.exists():
        return
    data = json.load(open(json_path))
    cfg = data.get("config", {})
    model = cfg.get("model", "")
    if "7B" not in model and "7b" not in model:
        print("[SKIP] 7B data not yet available.")
        return

    agg = data["aggregate"]
    lengths = list(agg["adaptive_gating"].keys())
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(lengths))
    width = 0.25

    for key, label, color in [
        ("fixed_similarity", "Sim-only", "#d62728"),
        ("fixed_recency", "Recency-only", "#ff7f0e"),
        ("adaptive_gating", "ODUS-X", "#2ca02c"),
    ]:
        vals = [agg[key][L]["mean_recovery"] * 100 for L in lengths]
        offset = width * ({"fixed_similarity": -1, "fixed_recency": 0, "adaptive_gating": 1}[key])
        ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="black", linewidth=0.3)

    ax.set_ylabel("Selection Recovery (%)", fontsize=12)
    ax.set_xlabel("Context Length", fontsize=12)
    ax.set_title("ODUS-X Scale Test (Qwen2.5-7B, A100)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(L)//1024}K" for L in lengths])
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_odus_x_7b_scale.pdf", dpi=300)
    fig.savefig(OUTDIR / "fig_odus_x_7b_scale.png", dpi=300)
    print(f"[Saved] {OUTDIR / 'fig_odus_x_7b_scale.pdf'}")


def main():
    plot_recovery_comparison()
    plot_drift_validation()
    plot_7b_scale()
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
