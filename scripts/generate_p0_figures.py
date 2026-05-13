"""
Generate P0 closed-loop evaluation figures from (real or simulated) results.

Reads outputs/hpca_fair_hardware/p0/p0_results_*.json and produces:
  - p0_quality_comparison.pdf/png   : 2x2 quality + latency comparison
  - p0_hbm_impact.pdf/png           : HBM cap vs compression + accuracy tradeoff

Usage:
    python -m prose_v2.scripts.generate_p0_figures
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Paths ────────────────────────────────────────────────────────────
P0_DIR = Path(r"D:\LLM\outputs\hpca_fair_hardware\p0")
OUTPUT_DIR = P0_DIR

# Find most recent results file
result_files = sorted(P0_DIR.glob("p0_results_*.json"))
if not result_files:
    print("No P0 result files found in", P0_DIR)
    sys.exit(1)

RESULT_PATH = result_files[-1]
print(f"Loading results from: {RESULT_PATH}")

with open(RESULT_PATH) as f:
    DATA = json.load(f)

RESULTS = DATA["results"]

# ── Style (LaTeX-ready) ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Color palette
COLORS = {
    "full_kv": "#7F7F7F",
    "prose": "#D95319",
    "prose_no_pht": "#EDB120",
    "stream_prefetcher": "#CCB974",
    "h2o": "#4C72B0",
    "snapkv": "#55A868",
}

MARKERS = {
    "full_kv": "s",
    "prose": "o",
    "prose_no_pht": "D",
    "stream_prefetcher": "^",
    "h2o": "v",
    "snapkv": "p",
}

LABELS = {
    "full_kv": "Full-KV (Oracle)",
    "prose": "ProSE",
    "prose_no_pht": "ProSE (no PHT)",
    "stream_prefetcher": "Stream Prefetcher",
    "h2o": "H2O",
    "snapkv": "SnapKV",
}

METHOD_ORDER = ["full_kv", "prose", "prose_no_pht", "stream_prefetcher", "h2o", "snapkv"]


def extract_metric(benchmark: str, metric_key: str) -> Dict[str, Dict[int, float]]:
    """Extract {method: {length: value}} from results."""
    out = {}
    for r in RESULTS:
        if r.get("benchmark") != benchmark:
            continue
        method = r["method"]
        length = r["context_length"]
        val = r.get(metric_key)
        if val is None:
            continue
        out.setdefault(method, {})[length] = val
    return out


def extract_budget_info() -> Dict[str, Dict[int, float]]:
    """Extract budget ratios from passkey details."""
    out = {}
    for r in RESULTS:
        if r.get("benchmark") != "passkey":
            continue
        method = r["method"]
        length = r["context_length"]
        details = r.get("details", [])
        if details:
            budget = details[0].get("budget_ratio", 1.0)
            comp = details[0].get("compression_ratio", 1.0)
            out.setdefault(method, {})[length] = {
                "budget": budget,
                "compression": comp,
                "pruned_len": details[0].get("pruned_seq_len", length),
            }
    return out


def plot_quality_comparison():
    """2x2 figure: Passkey, NIAH, LongBench, Latency."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("P0: Closed-Loop Quality Validation (Qwen2.5-3B)", fontsize=14, fontweight="bold")

    # ── (a) Passkey Accuracy ──
    ax = axes[0, 0]
    pk_acc = extract_metric("passkey", "accuracy")
    for method in METHOD_ORDER:
        if method not in pk_acc:
            continue
        lengths = sorted(pk_acc[method].keys())
        vals = [pk_acc[method][l] for l in lengths]
        ax.plot(lengths, vals, marker=MARKERS[method], color=COLORS[method],
                label=LABELS[method], linewidth=2)
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Passkey Accuracy")
    ax.set_title("(a) Passkey Retrieval")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xticks([8192, 16384, 32768])
    ax.set_xticklabels(["8K", "16K", "32K"])

    # ── (b) RULER NIAH Accuracy ──
    ax = axes[0, 1]
    ruler_acc = extract_metric("ruler", "accuracy")
    for method in METHOD_ORDER:
        if method not in ruler_acc:
            continue
        lengths = sorted(ruler_acc[method].keys())
        vals = [ruler_acc[method][l] for l in lengths]
        ax.plot(lengths, vals, marker=MARKERS[method], color=COLORS[method],
                label=LABELS[method], linewidth=2)
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("NIAH Accuracy")
    ax.set_title("(b) Needle-in-a-Haystack (RULER)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xticks([8192, 16384, 32768])
    ax.set_xticklabels(["8K", "16K", "32K"])

    # ── (c) LongBench F1 ──
    ax = axes[1, 0]
    lb_data = {}
    for r in RESULTS:
        if r.get("benchmark") != "longbench":
            continue
        method = r["method"]
        length = r["context_length"]
        lb_data.setdefault(method, {})[length] = r.get("overall", 0)

    for method in ["full_kv", "prose"]:
        if method not in lb_data:
            continue
        lengths = sorted(lb_data[method].keys())
        vals = [lb_data[method][l] for l in lengths]
        ax.plot(lengths, vals, marker=MARKERS[method], color=COLORS[method],
                label=LABELS[method], linewidth=2)
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("LongBench F1 (avg)")
    ax.set_title("(c) LongBench Retrieval-Heavy Tasks")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xticks([8192, 16384, 32768])
    ax.set_xticklabels(["8K", "16K", "32K"])

    # ── (d) Mean Token Latency ──
    ax = axes[1, 1]
    lat_data = extract_metric("passkey", "mean_latency_ms")
    for method in METHOD_ORDER:
        if method not in lat_data:
            continue
        lengths = sorted(lat_data[method].keys())
        vals = [lat_data[method][l] for l in lengths]
        ax.plot(lengths, vals, marker=MARKERS[method], color=COLORS[method],
                label=LABELS[method], linewidth=2)
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Mean Latency (ms/token)")
    ax.set_title("(d) Decode Latency")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xticks([8192, 16384, 32768])
    ax.set_xticklabels(["8K", "16K", "32K"])

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in [".pdf", ".png"]:
        path = OUTPUT_DIR / f"p0_quality_comparison{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_hbm_impact():
    """HBM cap impact: compression ratio + accuracy tradeoff."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("P0: HBM Capping Impact (Qwen2.5-3B)", fontsize=14, fontweight="bold")

    budget_info = extract_budget_info()

    # ── (a) Compression Ratio by Length ──
    ax = axes[0]
    for method in METHOD_ORDER:
        if method not in budget_info:
            continue
        lengths = sorted(budget_info[method].keys())
        comps = [budget_info[method][l]["compression"] for l in lengths]
        # Convert compression ratio to % retained
        retained = [c * 100 for c in comps]
        ax.plot(lengths, retained, marker=MARKERS[method], color=COLORS[method],
                label=LABELS[method], linewidth=2)
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("KV Retained (%)")
    ax.set_title("(a) KV Cache Retention Under HBM Cap")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xticks([8192, 16384, 32768])
    ax.set_xticklabels(["8K", "16K", "32K"])
    ax.set_ylim(0, 105)

    # ── (b) Accuracy vs Retention ──
    ax = axes[1]
    pk_acc = extract_metric("passkey", "accuracy")
    for method in METHOD_ORDER:
        if method not in budget_info or method not in pk_acc:
            continue
        lengths = sorted(budget_info[method].keys())
        x_vals = [budget_info[method][l]["compression"] * 100 for l in lengths]
        y_vals = [pk_acc[method][l] for l in lengths]
        ax.plot(x_vals, y_vals, marker=MARKERS[method], color=COLORS[method],
                label=LABELS[method], linewidth=2, markersize=8)
    ax.set_xlabel("KV Retained (%)")
    ax.set_ylabel("Passkey Accuracy")
    ax.set_title("(b) Quality vs Memory Retention")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    for ext in [".pdf", ".png"]:
        path = OUTPUT_DIR / f"p0_hbm_impact{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_method_breakdown():
    """Bar chart comparing all methods at 32K."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ctx_len = 32768
    metrics = {
        "Passkey": extract_metric("passkey", "accuracy"),
        "NIAH": extract_metric("ruler", "accuracy"),
    }

    x = np.arange(len(METHOD_ORDER))
    width = 0.35

    for i, (metric_name, data) in enumerate(metrics.items()):
        vals = [data.get(m, {}).get(ctx_len, 0) for m in METHOD_ORDER]
        offset = width * (i - 0.5)
        # Passkey = solid, NIAH = lighter/hatched
        for j, (method, val) in enumerate(zip(METHOD_ORDER, vals)):
            color = COLORS[method]
            if metric_name == "NIAH":
                # Lighter shade for NIAH
                from matplotlib.colors import to_rgb
                rgb = to_rgb(color)
                color = tuple(min(1.0, c + 0.35) for c in rgb)
            bar = ax.bar(x[j] + offset, val, width, color=color, edgecolor="black", linewidth=0.5)
        
    # Manual legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#999999", edgecolor="black", label="Passkey"),
        Patch(facecolor="#CCCCCC", edgecolor="black", label="NIAH"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_ylabel("Accuracy")
    ax.set_title(f"P0: Method Comparison at {ctx_len//1024}K Context (Qwen2.5-3B)")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METHOD_ORDER], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        path = OUTPUT_DIR / f"p0_method_breakdown_32k{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    plot_quality_comparison()
    plot_hbm_impact()
    plot_method_breakdown()

    print("\nAll P0 figures generated successfully.")


if __name__ == "__main__":
    main()
