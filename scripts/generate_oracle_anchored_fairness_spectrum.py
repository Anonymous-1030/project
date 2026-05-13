#!/usr/bin/env python3
"""
Fig. X: Oracle-Anchored Fairness Spectrum and Pareto Frontier
(Revised: Y=Recovery, size=Latency, family-color-coded, monotone frontier,
 L4=Attention Oracle, no fragile numbers)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(r"D:\LLM\prose_v2\outputs\hpca_fair_hardware")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data ──
# Each: (name, level, family, recovery, latency_us, marker, label_offset)
# label_offset = (dx, dy, ha, va)
METHODS = [
    ("StreamingLLM",     0, "Retention",   0.02, 224.6, "o", ( 0.00, -0.08, "center", "top")),
    ("H2O",              0, "Retention",   0.50, 195.3, "s", ( 0.00, -0.08, "center", "top")),
    ("SnapKV",           0, "Retention",   0.60, 197.5, "D", ( 0.30,  0.05, "left",   "bottom")),
    ("StreamPrefetcher", 1, "Prefetch",    0.45, 203.4, "^", ( 0.00, -0.07, "center", "top")),
    ("FreqRec-PF",       1, "Prefetch",    0.62, 192.0, "p", ( 0.30, -0.10, "left",   "top")),
    ("Quest-ASIC",       2, "Prefetch",    0.66, 197.9, "v", (-0.30, -0.08, "right",  "top")),
    ("InfiniGen",        2, "Prefetch",    0.72, 201.8, ">", ( 0.30,  0.08, "left",   "bottom")),
    ("RetrievalAttn",    2, "Retrieval",   0.50, 216.1, "<", ( 0.30, -0.06, "left",   "top")),
    ("PROSE",            3, "PROSE",       0.82, 133.0, "*", ( 0.00,  0.05, "center", "bottom")),
    ("Oracle",           4, "Oracle",      1.00,  50.0, "*", ( 0.12, -0.06, "left",   "top")),
]

FAMILY_COLORS = {
    "Retention": "#4A7C8C",   # muted blue
    "Prefetch":  "#5A8F5E",   # muted green
    "Retrieval": "#8B6BB3",   # muted purple
    "PROSE":     "#C75B39",   # PROSE red
    "Oracle":    "#333333",   # black
}

# Build data structures
data = []
for name, level, family, rec, lat_us, marker, loff in METHODS:
    data.append({
        "name": name, "level": level, "family": family,
        "recovery": rec, "latency_us": lat_us,
        "color": FAMILY_COLORS[family], "marker": marker,
        "size": lat_us * 2.8,   # size ∝ latency (larger = slower)
        "loff": loff,
    })

# ── Ideal Assumption Frontier (monotone increasing) ──
# Represents theoretical best recovery achievable at each deployability level
frontier_x = np.array([0, 1, 2, 3, 4])
frontier_y = np.array([0.62, 0.68, 0.78, 0.90, 1.00])

# Smooth interpolation for visual curve
from scipy.interpolate import make_interp_spline
x_smooth = np.linspace(0, 4, 200)
spl = make_interp_spline(frontier_x, frontier_y, k=3)
y_smooth = spl(x_smooth)

# ── Plot ──
fig, ax = plt.subplots(figsize=(7.0, 4.8))

# Very subtle vertical bands
for i in range(5):
    ax.axvspan(i - 0.48, i + 0.48, alpha=0.04, color="gray", zorder=0)

# Ideal frontier (solid line, monotone)
ax.plot(x_smooth, y_smooth, "k-", linewidth=2.0, alpha=0.35, zorder=1)
# Frontier markers omitted for cleaner look

# Label the frontier
ax.text(0.15, 0.88, "Ideal Assumption\nFrontier", fontsize=9, ha="left", va="top",
        color="#333333", alpha=0.5, style="italic")

# Scatter points (size = latency)
for d in data:
    # Oracle gets special hollow treatment
    if d["name"] == "Oracle":
        ax.scatter(d["level"], d["recovery"], s=d["size"], facecolors="none",
                   edgecolors=d["color"], marker=d["marker"], linewidths=2.0,
                   zorder=4, alpha=0.9)
    else:
        ax.scatter(d["level"], d["recovery"], s=d["size"], c=d["color"],
                   marker=d["marker"], edgecolors="black", linewidths=1.0,
                   zorder=3, alpha=0.88)

# Annotations
for d in data:
    dx, dy, ha, va = d["loff"]
    x = d["level"] + dx
    y = max(-0.05, min(1.05, d["recovery"] + dy))
    is_prose = d["name"] == "PROSE"
    is_oracle = d["name"] == "Oracle"
    ax.annotate(
        d["name"],
        xy=(d["level"], d["recovery"]),
        xytext=(x, y),
        fontsize=12 if is_prose else (11 if is_oracle else 10),
        ha=ha, va=va,
        fontweight="bold" if is_prose else ("bold" if is_oracle else "normal"),
        color=d["color"],
        arrowprops=dict(arrowstyle="->", color=d["color"], lw=1.0,
                        connectionstyle="arc3,rad=0.05"),
        zorder=5,
    )

# PROSE knee-point callout
ax.annotate(
    "KNEE POINT\n(Deployable Optimum)",
    xy=(3, 0.82),
    xytext=(1.25, 0.95),
    fontsize=11,
    ha="center", va="bottom",
    fontweight="bold",
    color="#C75B39",
    bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFF3E0",
              edgecolor="#C75B39", linewidth=1.5, alpha=0.95),
    arrowprops=dict(arrowstyle="->", color="#C75B39", lw=1.6,
                    connectionstyle="arc3,rad=0.18"),
    zorder=6,
)

# PROSE dominance zone: shaded area above all L0-L2 points
ax.fill_between([2.5, 3.5], 0.72, 0.88, alpha=0.10, color="#C75B39", zorder=0)
ax.text(2.6, 0.54, "Pareto-dominant\nin deployable region", fontsize=9, ha="center",
        color="#C75B39", fontweight="bold", alpha=0.85, zorder=5,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor="#C75B39", linewidth=1.0, alpha=0.8))

# L4 Oracle boundary line
ax.axvline(x=4, color="#333333", linestyle="-.", linewidth=2.2, alpha=0.35, zorder=1)
ax.text(4.05, 0.85, "Unimplementable\nOracle Bound\n(perfect future\nattention knowledge)",
        fontsize=9, ha="left", va="top", color="#333333", style="italic", alpha=0.6)

# ── Axis ──
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels([
    "L0\nPure Software",
    "L1\nGeneric HW\nPrefetch",
    "L2\nASIC-Accelerated\nSOTA",
    "L3\nPROSE",
    "L4\nOracle\n(Unimpl.)",
], fontsize=10, linespacing=0.95)

# Horizontal offset note for FreqRec-PF (same level as StreamPrefetcher)
# No extra label needed; the point annotation handles it

ax.set_ylabel("Normalized Recovery Score", fontsize=13, fontweight="bold")
ax.set_xlabel("Deployability →  (Intrusiveness / Assumption Strength)",
              fontsize=13, fontweight="bold")

ax.set_ylim(-0.08, 1.08)
ax.set_xlim(-0.6, 5.3)
ax.grid(True, axis="y", linestyle="--", alpha=0.3, linewidth=0.8)
ax.tick_params(axis="both", labelsize=10, width=1.0, length=4)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# ── Custom legend: family colors + size explanation ──
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=FAMILY_COLORS["Retention"],
           markeredgecolor="black", markersize=10, label="Retention-centric"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=FAMILY_COLORS["Prefetch"],
           markeredgecolor="black", markersize=10, label="Prefetch-centric"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=FAMILY_COLORS["Retrieval"],
           markeredgecolor="black", markersize=10, label="Retrieval-centric"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor=FAMILY_COLORS["PROSE"],
           markeredgecolor="black", markersize=14, label="PROSE (this work)"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="none",
           markeredgecolor=FAMILY_COLORS["Oracle"], markersize=12, markeredgewidth=1.5,
           label="Oracle bound"),
    Line2D([0], [0], linestyle="-", color="black", linewidth=2.0, alpha=0.35,
           label="Ideal frontier"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
          framealpha=0.95, edgecolor="#aaaaaa", fancybox=False, frameon=True,
          ncol=2, columnspacing=0.8)

# Size annotation
ax.text(5.15, 0.02, "Marker size ∝ decode latency\n(larger = slower)",
        fontsize=8, ha="right", va="bottom", color="#666666", style="italic",
        transform=ax.transData)

fig.subplots_adjust(left=0.10, bottom=0.19, right=0.97, top=0.94)

fig.savefig(OUTPUT_DIR / "figX_oracle_anchored_fairness_spectrum.pdf", dpi=300)
fig.savefig(OUTPUT_DIR / "figX_oracle_anchored_fairness_spectrum.png", dpi=300)
print(f"[Plot] Saved {OUTPUT_DIR / 'figX_oracle_anchored_fairness_spectrum.pdf'}")
print(f"[Plot] Saved {OUTPUT_DIR / 'figX_oracle_anchored_fairness_spectrum.png'}")
