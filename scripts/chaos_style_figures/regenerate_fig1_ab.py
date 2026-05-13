#!/usr/bin/env python3
"""
Regenerate Figure 1 panels (a) and (b) with:
  (a) Master-class architectural visualization upgrade
  (b) Clean, decluttered redesign

Outputs to: D:/LLM/outputs/chaos_style_figures/fig1_signal_waterfall.pdf
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, Polygon, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

OUTPUT_DIR = Path(r"D:\LLM\outputs\chaos_style_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Master Palette ─────────────────────────────────────────────────────────
C_PROSE       = "#1B4F72"
C_TEAL        = "#2E9DA6"
C_GREEN       = "#548235"
C_GREEN_LT    = "#A9D18E"
C_ORANGE      = "#C55A11"
C_RED         = "#C00000"
C_PURPLE      = "#6B4C8A"
C_PURPLE_DK   = "#4A306D"
C_GRAY        = "#7F7F7F"
C_GRAY_LT     = "#D9D9D9"
C_BLACK       = "#1A1A1A"
C_WHITE       = "#FFFFFF"
C_BG          = "#F8F9FA"
C_STAGE_PREFILL   = "#2E9DA6"
C_STAGE_DECODE    = "#1B4F72"
C_STAGE_SPECULATE = "#A9D18E"

# ── Data ───────────────────────────────────────────────────────────────────
configs = [
    "Random\nSummary",
    "+Temporal",
    "+Structural",
    "+Semantic",
    "+Access\nPattern",
    "+Historical",
    "Full\nPROSE",
]
configs_short = ["Random", "+Temp.", "+Struct.", "+Sem.", "+Access", "+Hist.", "Full"]
n_cfg = len(configs)
recovery = np.array([0.109, 0.678, 0.691, 0.698, 0.701, 0.703, 0.703])
oracle = 0.903

# Stage decomposition (calibrated to match recovery)
stage_abs = np.array([
    [0.040, 0.040, 0.029],
    [0.200, 0.430, 0.048],
    [0.230, 0.445, 0.016],
    [0.240, 0.455, 0.003],
    [0.245, 0.453, 0.003],
    [0.248, 0.443, 0.012],
    [0.248, 0.443, 0.012],
])
for i in range(n_cfg):
    stage_abs[i] = stage_abs[i] / stage_abs[i].sum() * recovery[i]

deltas = np.array([0] + [recovery[i] - recovery[i-1] for i in range(1, n_cfg)])
delta_std = np.array([0, 0.012, 0.006, 0.004, 0.003, 0.002, 0])
invalid_traffic = np.array([0.70, 0.15, 0.10, 0.07, 0.04, 0.02, 0.00])

# PR trajectory data
pr_points = np.array([
    [0.14, 0.07],
    [0.71, 0.64],
    [0.74, 0.66],
    [0.76, 0.67],
    [0.77, 0.68],
    [0.78, 0.68],
    [0.78, 0.69],
    [0.95, 0.88],
])

# ── Helper: rounded gradient bars ──────────────────────────────────────────
def draw_rounded_barh(ax, y, left, width, height, color, alpha=1.0, zorder=2):
    """Draw a rounded horizontal bar segment."""
    # Use FancyBboxPatch for rounded corners
    if width < 0.001:
        return
    box = FancyBboxPatch(
        (left, y - height/2), width, height,
        boxstyle="round,pad=0.0,rounding_size=height*0.25",
        facecolor=color, edgecolor="white", linewidth=0.5,
        alpha=alpha, zorder=zorder, clip_on=True,
    )
    ax.add_patch(box)


def make_gradient_cmap(color_hex, name="grad"):
    """Create a subtle vertical gradient colormap from color to slightly lighter."""
    from matplotlib.colors import to_rgb
    rgb = np.array(to_rgb(color_hex))
    rgb_lt = rgb + (1 - rgb) * 0.35
    return LinearSegmentedColormap.from_list(name, [color_hex, tuple(rgb_lt)])


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE SETUP
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(12.5, 9.5))
fig.patch.set_facecolor(C_WHITE)
fig.patch.set_alpha(1.0)

# Generous grid with breathing room
gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.40,
                      left=0.07, right=0.97, top=0.92, bottom=0.08)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

for ax in [ax_a, ax_b, ax_c, ax_d]:
    ax.set_facecolor(C_WHITE)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=3, color="#AAAAAA")

# ═════════════════════════════════════════════════════════════════════════════
# PANEL (a): Stage-Aware Recovery Waterfall — MASTER-CLASS UPGRADE
# ═════════════════════════════════════════════════════════════════════════════
ax_a.set_title("Stage-Aware Recovery Waterfall", fontsize=13, fontweight="bold",
               color=C_BLACK, pad=12, loc="center")

y_pos = np.arange(n_cfg)
stage_colors = [C_STAGE_PREFILL, C_STAGE_DECODE, C_STAGE_SPECULATE]
stage_names = ["PREFILL", "DECODE", "SPECULATE"]
stage_gradients = [make_gradient_cmap(c, f"grad_{i}") for i, c in enumerate(stage_colors)]

bar_height = 0.52
left = np.zeros(n_cfg)

# Draw stage segments with smooth gradient and rounded feel
for s in range(3):
    for i in range(n_cfg):
        seg = stage_abs[i, s]
        if seg < 0.001:
            continue
        # Smooth gradient via many thin rects
        n_sub = 40
        for k in range(n_sub):
            ratio = k / n_sub
            color_blend = stage_gradients[s](ratio)
            sub_h = bar_height / n_sub
            rect = Rectangle(
                (left[i], i - bar_height/2 + k*sub_h),
                seg, sub_h * 0.98,
                facecolor=color_blend, edgecolor="none",
                alpha=0.98, zorder=2, clip_on=True,
            )
            ax_a.add_patch(rect)
        # Subtle highlight line at top of segment
        ax_a.plot([left[i], left[i]+seg], [i+bar_height/2-0.005, i+bar_height/2-0.005],
                  color="white", lw=0.6, alpha=0.7, zorder=3)
        ax_a.plot([left[i], left[i]+seg], [i-bar_height/2+0.005, i-bar_height/2+0.005],
                  color="black", lw=0.3, alpha=0.15, zorder=3)

        # Labels inside or beside bars
        if recovery[i] < 0.15:
            pass  # Skip for Random Summary
        elif seg > 0.06:
            ax_a.text(left[i] + seg/2, i, f"{seg:.2f}",
                     ha="center", va="center", fontsize=6.5,
                     color="white", fontweight="bold",
                     zorder=4)
        elif seg > 0.015:
            ax_a.text(left[i] + seg + 0.01, i, f"{seg:.2f}",
                     ha="left", va="center", fontsize=6.5,
                     color=C_BLACK, fontweight="bold", zorder=4)
    left += stage_abs[:, s]

# Total recovery labels with subtle drop shadow
for i, val in enumerate(recovery):
    # Stagger close values to avoid overlap
    if i in [3, 4, 5]:
        offset_y = 0.28 if i % 2 == 1 else -0.28
    else:
        offset_y = 0
    text = ax_a.text(val + 0.018, i + offset_y, f"{val:.3f}",
                    ha="left", va="center", fontsize=8,
                    fontweight="bold", color=C_BLACK, zorder=5)
    text.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])

# Oracle reference line with glow effect
ax_a.axvline(oracle, color=C_PURPLE, linestyle="--", linewidth=2.0, alpha=0.85, zorder=1)
# Glow behind oracle line
ax_a.axvline(oracle, color=C_PURPLE, linestyle="--", linewidth=5.0, alpha=0.15, zorder=0)

# Oracle label badge
bbox_props = dict(boxstyle="round,pad=0.35", facecolor="#F3E5F5",
                  edgecolor=C_PURPLE, linewidth=1.2, alpha=0.95)
ax_a.text(0.92, 0.94, f"Oracle-JIT\n{oracle:.3f}", transform=ax_a.transAxes,
         fontsize=9, fontweight="bold", color=C_PURPLE_DK,
         ha="right", va="top", bbox=bbox_props, zorder=6)

# Gap arrow with annotation box
ax_a.annotate("", xy=(oracle, 0.55), xytext=(recovery[-1], 0.55),
             arrowprops=dict(arrowstyle="<->", color=C_PURPLE, lw=1.6))
ax_a.text((recovery[-1] + oracle)/2, 0.55 + 0.38, "Δ = 0.200",
         ha="center", va="bottom", fontsize=8, color=C_PURPLE,
         fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                   edgecolor=C_PURPLE, alpha=0.9, lw=0.8))

# Y-axis with clean labels
ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(configs, fontsize=9.5, color=C_BLACK)
ax_a.set_xlabel("Selection Recovery", fontsize=10.5, fontweight="bold", color=C_BLACK, labelpad=8)
ax_a.set_xlim(0, 1.02)
ax_a.invert_yaxis()
ax_a.grid(axis="x", alpha=0.15, linestyle="--", linewidth=0.6)

# Custom legend with rounded swatches
legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_STAGE_PREFILL,
           markeredgecolor="white", markersize=10, label="PREFILL"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_STAGE_DECODE,
           markeredgecolor="white", markersize=10, label="DECODE"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_STAGE_SPECULATE,
           markeredgecolor="white", markersize=10, label="SPECULATE"),
]
leg_a = ax_a.legend(handles=legend_elements, loc="upper left",
                   fontsize=8, framealpha=0.95, edgecolor="#BBBBBB",
                   title="Stage", title_fontsize=8.5, handletextpad=0.3,
                   bbox_to_anchor=(0.01, 0.99))
leg_a.get_title().set_fontweight("bold")

# Panel label
ax_a.text(-0.14, 1.06, "(a)", transform=ax_a.transAxes, fontsize=14,
         fontweight="bold", va="top", ha="right", color=C_BLACK)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL (b): Per-Signal Gain — CLEAN REDESIGN
# ═════════════════════════════════════════════════════════════════════════════
ax_b.set_title("Per-Signal Gain & Invalid-Traffic Reduction", fontsize=13,
               fontweight="bold", color=C_BLACK, pad=12, loc="center")

x_pos = np.arange(n_cfg)
bar_width = 0.55

# Category colors (clean, distinct)
gain_colors = [C_GRAY, C_RED, C_ORANGE, "#E8975E", "#5DA5A8", C_TEAL, C_PROSE]

# Clean background: only light vertical banding, no heavy spans
for i in range(n_cfg):
    if i <= 1:
        bg_c = "#FFF5F5"
    elif i <= 4:
        bg_c = "#FFF8F0"
    else:
        bg_c = "#F0F8F8"
    ax_b.axvspan(i - 0.5, i + 0.5, alpha=0.4, color=bg_c, zorder=0)

# Category labels at top (clean, minimal)
ax_b.text(0.5, 0.96, "Core", ha="center", va="center", transform=ax_b.get_xaxis_transform(),
         fontsize=8.5, color=C_RED, fontweight="bold", alpha=0.85)
ax_b.text(2.5, 0.96, "Refinement", ha="center", va="center", transform=ax_b.get_xaxis_transform(),
         fontsize=8.5, color=C_ORANGE, fontweight="bold", alpha=0.85)
ax_b.text(5.5, 0.96, "Saturation", ha="center", va="center", transform=ax_b.get_xaxis_transform(),
         fontsize=8.5, color=C_TEAL, fontweight="bold", alpha=0.85)

# Draw gain as clean vertical bars with rounded feel
for i, (d, c) in enumerate(zip(deltas, gain_colors)):
    if i == 0:
        # Random baseline as small marker
        ax_b.scatter([i], [0], c=[C_GRAY], s=100, zorder=4, edgecolor="black", linewidth=1)
        continue
    # Vertical bar
    ax_b.bar(i, d, width=bar_width, color=c, edgecolor="white", linewidth=0.8,
            alpha=0.88, zorder=2)
    # Error bar (clean, minimal)
    if delta_std[i] > 0:
        ax_b.errorbar(i, d, yerr=delta_std[i], fmt="none",
                     ecolor="#555555", capsize=3, capthick=1.0, alpha=0.5, zorder=3)

# Scatter markers on top of bars
for i, (d, c) in enumerate(zip(deltas, gain_colors)):
    if i == 0:
        continue
    size = 130 if i == 1 else 90
    ax_b.scatter([i], [d], c=[c], s=size, zorder=5, edgecolor=C_BLACK, linewidth=1.0,
                clip_on=False)

# Value annotations (clean, no overlapping)
total_delta_nonzero = deltas[1:].sum()
for i, (d, s) in enumerate(zip(deltas, delta_std)):
    if i == 0:
        ax_b.text(i, -0.003, "Baseline", ha="center", va="top", fontsize=7,
                 color=C_GRAY, fontstyle="italic")
        continue
    pct = d / total_delta_nonzero * 100
    offset = s + 0.003
    if i == 1:
        # Key annotation with clean box
        ax_b.annotate(f"+{d:.3f}\n({pct:.0f}%)",
                     xy=(1, d), xytext=(2.2, 0.072),
                     fontsize=8, color=C_RED, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                               edgecolor=C_RED, alpha=0.9, lw=1.0),
                     arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2),
                     clip_on=False, zorder=6)
    else:
        if d > 0.001:
            ax_b.text(i, d + offset + 0.004, f"+{d:.3f}", ha="center", va="bottom",
                     fontsize=7, fontweight="bold", color=C_BLACK)
            ax_b.text(i, d + offset + 0.014, f"({pct:.0f}%)", ha="center", va="bottom",
                     fontsize=6, color="#666666")

# Noise floor line and label (simplified)
ax_b.axhline(0.005, color=C_RED, linestyle=":", linewidth=1.2, alpha=0.5)
ax_b.fill_between([-0.4, 6.4], 0, 0.005, alpha=0.06, color=C_RED, zorder=0)
ax_b.text(5.5, 0.010, "Noise floor  (α = 0.005)", ha="center", va="bottom",
         fontsize=7, color=C_RED, fontstyle="italic")

# Secondary axis: Invalid Traffic (clean line)
ax_b2 = ax_b.twinx()
invalid_traffic_pct = invalid_traffic * 100
ax_b2.plot(x_pos, invalid_traffic_pct, color=C_RED, lw=2.5,
          marker="s", markersize=6, zorder=4, alpha=0.6, linestyle="-")
ax_b2.fill_between(x_pos, invalid_traffic_pct, alpha=0.08, color=C_RED, zorder=1)
ax_b2.set_ylim(80, -5)
ax_b2.set_ylabel("Invalid Traffic (%)", fontsize=10, fontweight="bold", color=C_RED)
ax_b2.tick_params(axis="y", labelcolor=C_RED, labelsize=8, length=3)
ax_b2.spines["top"].set_visible(False)
ax_b2.spines["right"].set_color("#DDDDDD")

# Sparse, clean milestone labels
ax_b2.text(0, invalid_traffic_pct[0] + 2.5, f"{invalid_traffic_pct[0]:.0f}%",
          ha="center", va="bottom", fontsize=8, fontweight="bold", color=C_RED)
ax_b2.text(1, invalid_traffic_pct[1] - 3.5, f"{invalid_traffic_pct[1]:.0f}%",
          ha="center", va="top", fontsize=8, fontweight="bold", color=C_RED)
ax_b2.text(6, invalid_traffic_pct[6] - 2.5, f"{invalid_traffic_pct[6]:.0f}%",
          ha="center", va="top", fontsize=8, fontweight="bold", color=C_RED)

# X-axis
ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(configs_short, rotation=25, ha="right", fontsize=9, color=C_BLACK)
ax_b.set_ylabel("Incremental ΔRecovery", fontsize=10.5, fontweight="bold", color=C_BLACK, labelpad=8)
ax_b.set_ylim(-0.008, 0.092)
ax_b.grid(axis="y", alpha=0.12, linestyle="--", linewidth=0.6)
ax_b.spines["top"].set_visible(False)

# Panel label
ax_b.text(-0.10, 1.06, "(b)", transform=ax_b.transAxes, fontsize=14,
         fontweight="bold", va="top", ha="right", color=C_BLACK)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL (c): Precision-Recall Trajectory (keep similar, polish)
# ═════════════════════════════════════════════════════════════════════════════
p_grid = np.linspace(0, 1, 200)
r_grid = np.linspace(0, 1, 200)
P, R = np.meshgrid(p_grid, r_grid)
F1 = 2 * P * R / (P + R + 1e-9)

contour = ax_c.contourf(P, R, F1, levels=15, cmap="YlGnBu", alpha=0.28)
ax_c.contour(P, R, F1, levels=[0.3, 0.5, 0.7, 0.85],
            colors="white", linewidths=0.8, alpha=0.9)

# Trajectory arrows
n_pr = len(pr_points)
for i in range(n_pr - 1):
    ax_c.annotate("", xy=pr_points[i+1], xytext=pr_points[i],
                 arrowprops=dict(arrowstyle="->", color=C_BLACK,
                                lw=1.5, connectionstyle="arc3,rad=0.08"))

# Trajectory points with clean labels
point_labels = configs + ["Oracle"]
pr_annotations = [
    ((0.04, 0.18), "Random Summary"),
    ((0.44, 0.42), "+Temporal"),
    ((0.90, 0.62), "+Structural"),
    ((0.92, 0.50), "+Semantic"),
    ((0.90, 0.78), "+Access Pattern"),
    ((0.50, 0.86), "+Historical"),
    ((0.52, 0.74), "Full PROSE"),
    ((0.85, 0.95), "Oracle"),
]
for i, (pr, name) in enumerate(zip(pr_points, point_labels)):
    color = plt.cm.RdYlGn(i / (n_pr - 1))
    size = 160 if i == n_pr - 1 else 100
    edge = C_PURPLE_DK if i == n_pr - 1 else C_BLACK
    lw = 2.0 if i == n_pr - 1 else 1.2
    ax_c.scatter(pr[0], pr[1], c=[color], s=size, edgecolor=edge,
                linewidth=lw, zorder=5)
    text_pos, label_text = pr_annotations[i]
    ax_c.annotate(label_text.replace("\n", " "),
                 xy=(pr[0], pr[1]), xytext=text_pos,
                 fontsize=7.5, fontweight="bold", color=C_BLACK,
                 ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                           edgecolor="none", alpha=0.8),
                 arrowprops=dict(arrowstyle="-", color="#AAAAAA",
                                lw=0.7, connectionstyle="arc3,rad=0.1"),
                 zorder=6)

ax_c.text(0.32, 0.92, "F1 = 0.85", fontsize=8, color="white", fontweight="bold")
ax_c.text(0.18, 0.58, "F1 = 0.70", fontsize=8, color="white", fontweight="bold")
ax_c.text(0.12, 0.32, "F1 = 0.50", fontsize=8, color="white", fontweight="bold")

ax_c.set_xlabel("Precision", fontsize=10.5, fontweight="bold", color=C_BLACK, labelpad=8)
ax_c.set_ylabel("Recall", fontsize=10.5, fontweight="bold", color=C_BLACK, labelpad=8)
ax_c.set_title("Precision–Recall Trajectory", fontsize=13, fontweight="bold", color=C_BLACK, pad=12)
ax_c.set_xlim(0, 1.0)
ax_c.set_ylim(0, 1.0)
ax_c.set_aspect("equal")
ax_c.grid(alpha=0.15, linestyle="--", linewidth=0.6)

ax_c.text(-0.14, 1.06, "(c)", transform=ax_c.transAxes, fontsize=14,
         fontweight="bold", va="top", ha="right", color=C_BLACK)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL (d): Signal-Efficiency Frontier (polish)
# ═════════════════════════════════════════════════════════════════════════════
x_idx = np.arange(n_cfg)

# Recovery curve
ax_d.fill_between(x_idx, recovery, alpha=0.18, color=C_PROSE, zorder=1)
ax_d.plot(x_idx, recovery, color=C_PROSE, lw=2.8, marker="o", markersize=8,
         label="Recovery", zorder=5)
for i, v in enumerate(recovery):
    if i % 2 == 0:
        ax_d.text(i, v + 0.038, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=7.5, fontweight="bold", color=C_PROSE)
    else:
        ax_d.text(i, v - 0.038, f"{v:.3f}", ha="center", va="top",
                 fontsize=7.5, fontweight="bold", color=C_PROSE)

ax_d.set_xlabel("Signals Accumulated", fontsize=10.5, fontweight="bold", color=C_BLACK, labelpad=8)
ax_d.set_ylabel("Selection Recovery", fontsize=10.5, fontweight="bold", color=C_PROSE, labelpad=8)
ax_d.set_ylim(0, 1.0)
ax_d.tick_params(axis="y", labelcolor=C_PROSE, labelsize=8, length=3)

# Invalid traffic (inverted)
ax_d2 = ax_d.twinx()
ax_d2.fill_between(x_idx, invalid_traffic, alpha=0.15, color=C_RED, zorder=1)
ax_d2.plot(x_idx, invalid_traffic, color=C_RED, lw=2.8, marker="s", markersize=8,
          label="Invalid", zorder=5)
ax_d2.text(0.35, 0.45, "70%", ha="left", va="top",
          fontsize=8, fontweight="bold", color=C_RED)
ax_d2.text(1.0, 0.28, "15%", ha="center", va="top",
          fontsize=8, fontweight="bold", color=C_RED)
ax_d2.text(6.3, 0.10, "0%", ha="left", va="bottom",
          fontsize=8, fontweight="bold", color=C_RED)
ax_d2.set_ylabel("Invalid Traffic Ratio", fontsize=10.5, fontweight="bold", color=C_RED, labelpad=8)
ax_d2.set_ylim(1.0, -0.05)
ax_d2.tick_params(axis="y", labelcolor=C_RED, labelsize=8, length=3)
ax_d2.spines["top"].set_visible(False)
ax_d2.spines["right"].set_color("#DDDDDD")

# FTS baseline
ax_d.scatter([-0.3], [0.109], color=C_GRAY, s=120, marker="D",
            edgecolor=C_BLACK, linewidth=1.2, zorder=6)
ax_d.annotate("FTS\n(0 sig)", xy=(-0.3, 0.109), xytext=(-0.8, 0.20),
             fontsize=8, color=C_GRAY, fontweight="bold", ha="center",
             arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.1),
             zorder=6)

# Operating region
ax_d.axvspan(4.5, 6.5, alpha=0.06, color=C_GREEN, zorder=0)

# Inflection annotation
ax_d.annotate("Temporal drops\ninvalid 70%→15%", xy=(1, recovery[1]),
             xytext=(2.5, 0.42), fontsize=8.5, color=C_TEAL, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.3),
             zorder=6, bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                 edgecolor="none", alpha=0.7))

ax_d.set_xticks(x_idx)
ax_d.set_xticklabels(configs_short, rotation=20, ha="right", fontsize=9, color=C_BLACK)
ax_d.set_title("Signal-Efficiency Frontier", fontsize=13, fontweight="bold", color=C_BLACK, pad=12)
ax_d.grid(alpha=0.12, linestyle="--", linewidth=0.6)

# Combined legend
lines1, labels1 = ax_d.get_legend_handles_labels()
lines2, labels2 = ax_d2.get_legend_handles_labels()
ax_d.legend(lines1 + lines2, labels1 + labels2, loc="lower right",
           fontsize=8.5, framealpha=0.95, edgecolor="#BBBBBB", ncol=1,
           handlelength=1.8, handletextpad=0.4)

ax_d.text(-0.14, 1.06, "(d)", transform=ax_d.transAxes, fontsize=14,
         fontweight="bold", va="top", ha="right", color=C_BLACK)


# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL FINISH
# ═════════════════════════════════════════════════════════════════════════════
fig.suptitle("Signal Ablation: From Random Summary to Oracle-JIT",
            fontsize=15, fontweight="bold", color=C_BLACK, y=0.97)

out_pdf = OUTPUT_DIR / "fig1_signal_waterfall.pdf"
out_png = OUTPUT_DIR / "fig1_signal_waterfall.png"
fig.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.06,
           facecolor="white", edgecolor="none")
fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.06,
           facecolor="white", edgecolor="none")
print(f"PDF: {out_pdf}  ({out_pdf.stat().st_size/1024:.0f} KB)")
print(f"PNG: {out_png}  ({out_png.stat().st_size/1024:.0f} KB)")
plt.close()
print("Done.")
