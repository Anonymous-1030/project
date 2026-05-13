#!/usr/bin/env python3
"""
Generate 2x4 comprehensive comparison figure for HPCA rebuttal.

Layout:
  Top row (Software SOTA):    Recovery | P99 Latency | Throughput | Passkey Acc
  Bottom row (Hardware SOTA): Recovery | P99 Latency | Throughput | Latency Breakdown

Design for direct LaTeX embedding without scaling:
  - Large fonts (>= 10pt equivalent)
  - High DPI (300)
  - Legends placed BELOW each plot as compact horizontal rectangles
  - Main plots use wide rectangular aspect ratio to maximize data area
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, ".")

# ── Load data ────────────────────────────────────────────────────────
with open("outputs/hpca_fair_hardware/figure_2x4_data.json") as f:
    data = json.load(f)

sw_budget = data["sw_budget"]
hw_budget = data["hw_budget"]
sw_ctx = data["sw_ctx"]
hw_ctx = data["hw_ctx"]
passkey = data["passkey"]

# ── Style configuration (LaTeX-ready) ────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Color palette
C_PROSE = "#D95319"
C_FULLKV = "#7F7F7F"
C_STREAM = "#BCBCBC"
C_H2O = "#4C72B0"
C_SNAPKV = "#55A868"
C_QUEST = "#8172B2"
C_QUESTASIC = "#C44E52"
C_RETRIEVALASIC = "#64B5CD"
C_INFINIGENASIC = "#8C8C8C"
C_STREAMPF = "#CCB974"

markers = {
    "prose": "o",
    "full_kv": "s",
    "streaming": "^",
    "h2o": "D",
    "snapkv": "v",
    "quest": "p",
    "stream_prefetcher": "^",
    "quest_asic": "D",
    "retrieval_attention_asic": "v",
    "infinigen_asic": "<",
}

# ── Helper: extract arrays ───────────────────────────────────────────
def get_arrays(budget_dict, method, keys):
    entries = budget_dict[method]
    return tuple([e[k] for e in entries] for k in keys)


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE: 2 rows x 4 columns with legends BELOW each subplot
# ═══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20.0, 9.2))
fig.patch.set_facecolor("white")

from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 4, figure=fig, wspace=0.30, hspace=0.45,
              left=0.04, right=0.98, top=0.94, bottom=0.06)

ctx_labels = ["1K", "2K", "4K", "8K", "16K"]
ctx_ticks = [1024, 2048, 4096, 8192, 16384]

# Common legend kwargs: placed below plot, centered, 3 columns
LEGEND_BELOW = dict(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.14),
    ncol=3,
    frameon=True,
    framealpha=0.95,
    edgecolor="gray",
    handlelength=1.2,
    handletextpad=0.4,
    columnspacing=0.8,
)

# ── Row 0, Col 0: Software Recovery vs Budget ────────────────────────
ax = fig.add_subplot(gs[0, 0])
for method, color, label in [
    ("prose", C_PROSE, "ProSE (ours)"),
    ("quest", C_QUEST, "Quest"),
    ("snapkv", C_SNAPKV, "SnapKV"),
    ("h2o", C_H2O, "H2O"),
    ("streaming", C_STREAM, "StreamingLLM"),
    ("full_kv", C_FULLKV, "Full-KV"),
]:
    x, y = get_arrays(sw_budget, method, ["budget", "recovery"])
    ax.plot([v*100 for v in x], [v*100 for v in y], marker=markers[method],
            color=color, label=label, zorder=3 if method=="prose" else 2)
ax.set_xlabel("KV Budget (%)")
ax.set_ylabel("Gold-Chunk Recovery (%)")
ax.set_title("(a) Recovery vs. Budget — Software Methods")
ax.set_ylim(0, 105)
ax.legend(**LEGEND_BELOW)
ax.set_xticks([2, 5, 8, 10, 15, 20, 30])
ax.annotate("ProSE matches Quest\nwith 1.6–2.3× lower latency",
            xy=(10, sw_budget["prose"][3]["recovery"]*100),
            xytext=(18, 30), fontsize=9,
            arrowprops=dict(arrowstyle="->", color=C_PROSE, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_PROSE, alpha=0.9))

# ── Row 0, Col 1: Software P99 Latency vs Context Length ─────────────
ax = fig.add_subplot(gs[0, 1])
for method, color, label in [
    ("prose", C_PROSE, "ProSE (ours)"),
    ("quest", C_QUEST, "Quest"),
    ("snapkv", C_SNAPKV, "SnapKV"),
    ("h2o", C_H2O, "H2O"),
    ("streaming", C_STREAM, "StreamingLLM"),
]:
    x, y = get_arrays(sw_ctx, method, ["tokens", "p99_us"])
    ax.plot(x, y, marker=markers[method], color=color, label=label,
            zorder=3 if method=="prose" else 2)
ax.set_xlabel("Context Length (tokens)")
ax.set_ylabel("P99 Latency (μs)")
ax.set_title("(b) P99 Latency vs. Context — Software")
ax.set_xscale("log", base=2)
ax.set_xticks(ctx_ticks)
ax.set_xticklabels(ctx_labels)
ax.legend(**LEGEND_BELOW)
quest_p99_16k = sw_ctx["quest"][4]["p99_us"]
prose_p99_16k = sw_ctx["prose"][4]["p99_us"]
speedup = quest_p99_16k / prose_p99_16k
ax.annotate(f"ProSE P99: {prose_p99_16k:.0f} μs\n({speedup:.1f}× lower than Quest)",
            xy=(16384, prose_p99_16k), xytext=(4096, 310),
            fontsize=9, arrowprops=dict(arrowstyle="->", color=C_PROSE, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_PROSE, alpha=0.9))

# ── Row 0, Col 2: Software Throughput vs Budget ──────────────────────
ax = fig.add_subplot(gs[0, 2])
for method, color, label in [
    ("prose", C_PROSE, "ProSE (ours)"),
    ("quest", C_QUEST, "Quest"),
    ("snapkv", C_SNAPKV, "SnapKV"),
    ("h2o", C_H2O, "H2O"),
    ("streaming", C_STREAM, "StreamingLLM"),
]:
    x, y = get_arrays(sw_budget, method, ["budget", "throughput"])
    ax.plot([v*100 for v in x], y, marker=markers[method], color=color,
            label=label, zorder=3 if method=="prose" else 2)
ax.set_xlabel("KV Budget (%)")
ax.set_ylabel("Throughput (tok/s)")
ax.set_title("(c) Throughput vs. Budget — Software")
ax.legend(**LEGEND_BELOW)
ax.set_xticks([2, 5, 8, 10, 15, 20, 30])
prose_tput = sw_budget["prose"][3]["throughput"]
quest_tput = sw_budget["quest"][3]["throughput"]
ax.text(20, prose_tput + 400, f"ProSE: {prose_tput:.0f} tok/s\n({prose_tput/quest_tput:.1f}× vs Quest)",
        fontsize=9, ha="center", color=C_PROSE, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_PROSE, alpha=0.9))

# ── Row 0, Col 3: Passkey Accuracy vs Context Length ─────────────────
ax = fig.add_subplot(gs[0, 3])
for method, color, label in [
    ("prose", C_PROSE, "ProSE (ours)"),
    ("quest", C_QUEST, "Quest"),
    ("snapkv", C_SNAPKV, "SnapKV"),
    ("h2o", C_H2O, "H2O"),
    ("streaming", C_STREAM, "StreamingLLM"),
    ("full_kv", C_FULLKV, "Full-KV"),
]:
    x = [e["tokens"] for e in passkey[method]]
    y = [e["accuracy"] for e in passkey[method]]
    ax.plot(x, y, marker=markers[method], color=color, label=label,
            zorder=3 if method=="prose" else 2)
ax.set_xlabel("Context Length (tokens)")
ax.set_ylabel("Passkey Accuracy (%)")
ax.set_title("(d) Passkey Accuracy vs. Context")
ax.set_xscale("log", base=2)
ax.set_xticks(ctx_ticks)
ax.set_xticklabels(ctx_labels)
ax.set_ylim(0, 105)
ax.legend(**LEGEND_BELOW)
prose_acc_16k = passkey["prose"][4]["accuracy"]
ax.annotate(f"ProSE {prose_acc_16k:.0f}% @ 16K",
            xy=(16384, prose_acc_16k), xytext=(4096, 30),
            fontsize=9, arrowprops=dict(arrowstyle="->", color=C_PROSE, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_PROSE, alpha=0.9))

# ── Row 1, Col 0: Hardware Recovery vs Budget ────────────────────────
ax = fig.add_subplot(gs[1, 0])
for method, color, label in [
    ("prose", C_PROSE, "ProSE (ours)"),
    ("stream_prefetcher", C_STREAMPF, "Stream PF (fair HW)"),
    ("quest_asic", C_QUESTASIC, "Quest-ASIC"),
    ("retrieval_attention_asic", C_RETRIEVALASIC, "RetrAttn-ASIC"),
    ("infinigen_asic", C_INFINIGENASIC, "InfiniGen-ASIC"),
    ("full_kv", C_FULLKV, "Full-KV"),
]:
    x, y = get_arrays(hw_budget, method, ["budget", "recovery"])
    ax.plot([v*100 for v in x], [v*100 for v in y], marker=markers[method],
            color=color, label=label, zorder=3 if method=="prose" else 2)
ax.set_xlabel("KV Budget (%)")
ax.set_ylabel("Gold-Chunk Recovery (%)")
ax.set_title("(e) Recovery vs. Budget — Hardware Methods")
ax.set_ylim(0, 105)
ax.legend(**LEGEND_BELOW)
ax.set_xticks([2, 5, 8, 10, 15, 20, 30])
prose_rec = hw_budget["prose"][3]["recovery"]*100
quest_rec = hw_budget["quest_asic"][3]["recovery"]*100
ax.text(20, 25, f"ProSE +{prose_rec-quest_rec:.0f}%\nvs Quest-ASIC", fontsize=9,
        ha="center", color=C_PROSE, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_PROSE, alpha=0.9))

# ── Row 1, Col 1: Hardware P99 Latency vs Context Length ─────────────
ax = fig.add_subplot(gs[1, 1])
for method, color, label in [
    ("prose", C_PROSE, "ProSE (ours)"),
    ("stream_prefetcher", C_STREAMPF, "Stream PF (fair HW)"),
    ("quest_asic", C_QUESTASIC, "Quest-ASIC"),
    ("retrieval_attention_asic", C_RETRIEVALASIC, "RetrAttn-ASIC"),
    ("infinigen_asic", C_INFINIGENASIC, "InfiniGen-ASIC"),
]:
    x, y = get_arrays(hw_ctx, method, ["tokens", "p99_us"])
    ax.plot(x, y, marker=markers[method], color=color, label=label,
            zorder=3 if method=="prose" else 2)
ax.set_xlabel("Context Length (tokens)")
ax.set_ylabel("P99 Latency (μs)")
ax.set_title("(f) P99 Latency vs. Context — Hardware")
ax.set_xscale("log", base=2)
ax.set_xticks(ctx_ticks)
ax.set_xticklabels(ctx_labels)
ax.legend(**LEGEND_BELOW)
stream_p99 = hw_ctx["stream_prefetcher"][4]["p99_us"]
prose_p99_hw = hw_ctx["prose"][4]["p99_us"]
ax.annotate(f"ProSE P99 stable @ ~{prose_p99_hw:.0f} μs\n({stream_p99/prose_p99_hw:.1f}× lower than Stream PF)",
            xy=(16384, prose_p99_hw), xytext=(2048, 155),
            fontsize=9, arrowprops=dict(arrowstyle="->", color=C_PROSE, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_PROSE, alpha=0.9))

# ── Row 1, Col 2: Hardware Throughput vs Budget ──────────────────────
ax = fig.add_subplot(gs[1, 2])
for method, color, label in [
    ("prose", C_PROSE, "ProSE (ours)"),
    ("stream_prefetcher", C_STREAMPF, "Stream PF (fair HW)"),
    ("quest_asic", C_QUESTASIC, "Quest-ASIC"),
    ("retrieval_attention_asic", C_RETRIEVALASIC, "RetrAttn-ASIC"),
    ("infinigen_asic", C_INFINIGENASIC, "InfiniGen-ASIC"),
]:
    x, y = get_arrays(hw_budget, method, ["budget", "throughput"])
    ax.plot([v*100 for v in x], y, marker=markers[method], color=color,
            label=label, zorder=3 if method=="prose" else 2)
ax.set_xlabel("KV Budget (%)")
ax.set_ylabel("Throughput (tok/s)")
ax.set_title("(g) Throughput vs. Budget — Hardware")
ax.legend(**LEGEND_BELOW)
ax.set_xticks([2, 5, 8, 10, 15, 20, 30])
prose_tput_hw = hw_budget["prose"][3]["throughput"]
stream_tput_hw = hw_budget["stream_prefetcher"][3]["throughput"]
ax.text(5, 3100, f"ProSE: {prose_tput_hw:.0f} tok/s\n({prose_tput_hw/stream_tput_hw:.1f}× vs Fair HW)",
        fontsize=9, ha="center", color=C_PROSE, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_PROSE, alpha=0.9))

# ── Row 1, Col 3: Latency Breakdown @ Budget=10% ─────────────────────
ax = fig.add_subplot(gs[1, 3])
methods_break = ["prose", "stream_prefetcher", "quest_asic", "retrieval_attention_asic", "infinigen_asic"]
labels_break = ["ProSE", "Stream PF", "Quest-ASIC", "RetrAttn-ASIC", "InfiniGen-ASIC"]
colors_bar = [C_PROSE, C_STREAMPF, C_QUESTASIC, C_RETRIEVALASIC, C_INFINIGENASIC]

compute = [hw_budget[m][3]["compute_us"] for m in methods_break]
fetch = [hw_budget[m][3]["fetch_us"] for m in methods_break]
metadata = [hw_budget[m][3]["metadata_us"] for m in methods_break]
decompress = [hw_budget[m][3]["decompress_us"] for m in methods_break]

x = np.arange(len(methods_break))
width = 0.6

bottom1 = np.array(compute)
bottom2 = bottom1 + np.array(fetch)
bottom3 = bottom2 + np.array(metadata)

bars1 = ax.bar(x, compute, width, label="Compute", color="#4C72B0", edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x, fetch, width, bottom=bottom1, label="CXL Fetch", color="#DD8452", edgecolor="white", linewidth=0.5)
bars3 = ax.bar(x, metadata, width, bottom=bottom2, label="Metadata", color="#55A868", edgecolor="white", linewidth=0.5)
bars4 = ax.bar(x, decompress, width, bottom=bottom3, label="Decompress", color="#8172B2", edgecolor="white", linewidth=0.5)

# Total latency labels on top
totals = [hw_budget[m][3]["latency_us"] for m in methods_break]
for i, total in enumerate(totals):
    offset = 6 if i != 1 else 18
    ax.text(i, total + offset, f"{total:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Latency (μs)")
ax.set_title("(h) Latency Breakdown @ 10% Budget — Hardware")
ax.set_xticks(x)
ax.set_xticklabels(labels_break, rotation=15, ha="right")
ax.legend(**LEGEND_BELOW)
ax.set_ylim(0, max(totals) * 1.28)

# Add speedup bracket between ProSE and Stream PF
prose_lat = totals[0]
stream_lat = totals[1]
bracket_y = prose_lat + 22
ax.annotate("", xy=(0, bracket_y), xytext=(1, bracket_y),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
ax.text(0.5, bracket_y + 5, f"{stream_lat/prose_lat:.1f}× speedup", ha="center", fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.9))

# ── Finalize ─────────────────────────────────────────────────────────
output_path = Path("outputs/hpca_fair_hardware/fig_2x4_comprehensive_comparison.pdf")
fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"[Saved] {output_path}")

output_png = output_path.with_suffix(".png")
fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
print(f"[Saved] {output_png}")
