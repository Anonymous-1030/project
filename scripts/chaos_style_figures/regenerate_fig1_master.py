#!/usr/bin/env python3
"""
Figure 1 — Master-class redesign for LaTeX single-column insertion.
No scaling required. 3.5 in width, all text >= 7 pt.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgb
import pathlib

OUT = pathlib.Path(r"D:\LLM\outputs\chaos_style_figures")
OUT.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
FIG_W = 3.50
FIG_H = 5.90   # extra height to eliminate overlaps
DPI   = 300

plt.rcParams.update({
    "figure.dpi"       : DPI,
    "savefig.dpi"      : DPI,
    "pdf.fonttype"     : 42,
    "ps.fonttype"      : 42,
    "font.family"      : "sans-serif",
    "font.sans-serif"  : ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size"        : 8.0,
    "axes.titlesize"   : 9.0,
    "axes.labelsize"   : 8.0,
    "xtick.labelsize"  : 7.5,
    "ytick.labelsize"  : 7.5,
    "legend.fontsize"  : 7.0,
    "axes.linewidth"   : 0.5,
    "lines.linewidth"  : 1.2,
    "lines.markersize" : 4.5,
    "xtick.major.pad"  : 1.5,
    "ytick.major.pad"  : 1.5,
    "axes.titlepad"    : 4,
    "text.usetex"      : False,
})

# ═══════════════════════════════════════════════════════════════════════════════
# PALETTE
# ═══════════════════════════════════════════════════════════════════════════════
C_TEAL       = "#2A9D8F"
C_TEAL_DK    = "#1D7066"
C_NAVY       = "#264653"
C_SAGE       = "#8AB17D"
C_ORANGE     = "#E76F51"
C_ORANGE_LT  = "#F4A261"
C_RED        = "#C0392B"
C_RED_DK     = "#922B21"
C_PURPLE     = "#7B68EE"
C_PURPLE_DK  = "#4B0082"
C_GRAY       = "#7F8C8D"
C_GRAY_LT    = "#BDC3C7"
C_BLACK      = "#2C3E50"
C_WHITE      = "#FFFFFF"

STAGE_COLS = [
    (C_TEAL,   C_TEAL_DK),
    (C_NAVY,   "#1A3342"),
    (C_SAGE,   "#5E8C52"),
]
STAGE_NAMES = ["PREFILL", "DECODE", "SPECULATE"]

# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════
configs = [
    "Random", "+Temporal", "+Structural", "+Semantic",
    "+Access", "+Historical", "Full PROSE",
]
configs_short = ["Random", "+Temp.", "+Struct.", "+Sem.", "+Access", "+Hist.", "Full"]
n_cfg = len(configs)
recovery = np.array([0.109, 0.678, 0.691, 0.698, 0.701, 0.703, 0.703])
oracle   = 0.903

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

deltas = np.array([0] + [recovery[i]-recovery[i-1] for i in range(1,n_cfg)])
delta_std = np.array([0, 0.012, 0.006, 0.004, 0.003, 0.002, 0])
invalid_traffic = np.array([0.70, 0.15, 0.10, 0.07, 0.04, 0.02, 0.00])

pr_points = np.array([
    [0.14,0.07],[0.71,0.64],[0.74,0.66],[0.76,0.67],
    [0.77,0.68],[0.78,0.68],[0.78,0.69],[0.95,0.88],
])

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_gradient_barh(ax, y, left, width, height, c_left, c_right,
                       alpha=1.0, zorder=2, n_layers=100):
    if width <= 0:
        return
    rgb_l = np.array(to_rgb(c_left))
    rgb_r = np.array(to_rgb(c_right))
    rounding = height * 0.22
    bottom = y - height/2

    for k in range(n_layers):
        ratio = k / n_layers
        color = rgb_l * (1-ratio) + rgb_r * ratio
        hex_c = "#" + "".join(f"{int(c*255):02x}" for c in color)
        sub_w = width / n_layers
        rect = Rectangle(
            (left + k*sub_w - 0.0003, bottom), sub_w + 0.0006, height,
            facecolor=hex_c, edgecolor="none",
            alpha=alpha, zorder=zorder, clip_on=True)
        ax.add_patch(rect)

    border = FancyBboxPatch(
        (left, bottom), width, height,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor="none", edgecolor="white",
        linewidth=0.4, alpha=0.65, zorder=zorder+1, clip_on=True)
    ax.add_patch(border)

    if width > 0.03:
        shadow = FancyBboxPatch(
            (left+0.003, bottom-0.005), width, height,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            facecolor="#000000", edgecolor="none",
            alpha=0.06, zorder=zorder-1, clip_on=True)
        ax.add_patch(shadow)


def draw_lollipop(ax, x, y, color, yerr=0, width=0.30, z=3):
    ax.bar(x, y, width=width, bottom=0, color=color, edgecolor="none",
           alpha=0.30, zorder=z-1)
    ax.plot([x, x], [0, y], color=color, lw=0.9, alpha=0.7, zorder=z)
    ax.scatter([x], [y], c=[color], s=65, zorder=z+1,
               edgecolor=C_BLACK, linewidth=0.8, clip_on=False)
    if yerr > 0:
        ax.errorbar(x, y, yerr=yerr, fmt="none",
                   ecolor="#444444", capsize=2.2, capthick=0.8, alpha=0.5, zorder=z)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(C_WHITE)

gs = fig.add_gridspec(2, 2,
    left=0.10, right=0.96, top=0.90, bottom=0.06,
    hspace=0.50, wspace=0.38)
ax_a = fig.add_subplot(gs[0,0])
ax_b = fig.add_subplot(gs[0,1])
ax_c = fig.add_subplot(gs[1,0])
ax_d = fig.add_subplot(gs[1,1])

for ax in (ax_a, ax_b, ax_c, ax_d):
    ax.set_facecolor(C_WHITE)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#BBBBBB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=2.5, color="#AAAAAA")

# ─────────────────────────────────────────────────────────────────────────────
# (a) Stage-Aware Recovery Waterfall
# ─────────────────────────────────────────────────────────────────────────────
ax_a.set_title("Stage-Aware Recovery Waterfall", fontsize=9, fontweight="bold",
               color=C_BLACK, pad=4)

y_pos = np.arange(n_cfg)
bar_h = 0.48
left = np.zeros(n_cfg)

for s in range(3):
    c_left, c_right = STAGE_COLS[s]
    for i in range(n_cfg):
        seg = stage_abs[i, s]
        if seg < 0.001:
            continue
        draw_gradient_barh(ax_a, i, left[i], seg, bar_h, c_left, c_right, zorder=2)
        if recovery[i] < 0.15:
            pass
        elif seg > 0.055:
            ax_a.text(left[i]+seg/2, i, f"{seg:.2f}",
                     ha="center", va="center", fontsize=5.5,
                     color="white", fontweight="bold", zorder=5)
        elif seg > 0.015:
            ax_a.text(left[i]+seg+0.007, i, f"{seg:.2f}",
                     ha="left", va="center", fontsize=5.5,
                     color=C_BLACK, fontweight="bold", zorder=5)
    left += stage_abs[:, s]

for i, val in enumerate(recovery):
    stagger = 0.24 if i in (3,5) else (-0.24 if i==4 else 0)
    t = ax_a.text(val+0.014, i+stagger, f"{val:.3f}",
                 ha="left", va="center", fontsize=7, fontweight="bold",
                 color=C_BLACK, zorder=6)
    t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])

ax_a.axvline(oracle, color=C_PURPLE_DK, ls="--", lw=1.3, alpha=0.85, zorder=1)
ax_a.axvline(oracle, color=C_PURPLE_DK, ls="--", lw=4.0, alpha=0.10, zorder=0)

ax_a.text(0.96, 0.96, f"Oracle-JIT\n{oracle:.3f}", transform=ax_a.transAxes,
         fontsize=7.5, fontweight="bold", color=C_PURPLE_DK,
         ha="right", va="top", zorder=7,
         bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0E6FF",
                   edgecolor=C_PURPLE_DK, lw=1.0, alpha=0.95))

ax_a.annotate("", xy=(oracle, 0.55), xytext=(recovery[-1], 0.55),
             arrowprops=dict(arrowstyle="<->", color=C_PURPLE_DK, lw=1.2))
ax_a.text((recovery[-1]+oracle)/2, 0.55+0.35, "Δ=0.200",
         ha="center", va="bottom", fontsize=7, color=C_PURPLE_DK,
         fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                   edgecolor=C_PURPLE_DK, lw=0.7, alpha=0.9))

ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(configs, fontsize=7.5, color=C_BLACK)
ax_a.set_xlabel("Selection Recovery", fontsize=8, fontweight="bold", color=C_BLACK)
ax_a.set_xlim(0, 1.0)
ax_a.invert_yaxis()
ax_a.grid(axis="x", alpha=0.12, ls="--", lw=0.5)

# Legend BELOW panel (a)
leg_el = [
    Line2D([0],[0], marker="s", color="w", markerfacecolor=STAGE_COLS[i][0],
           markeredgecolor="white", markersize=7, label=STAGE_NAMES[i])
    for i in range(3)
]
leg_a = ax_a.legend(handles=leg_el, loc="upper center",
                   fontsize=6.5, framealpha=0.93, edgecolor="#CCCCCC",
                   title="Stage", title_fontsize=6.5,
                   handletextpad=0.2, borderpad=0.25, ncol=3,
                   bbox_to_anchor=(0.5, -0.14))
leg_a.get_title().set_fontweight("bold")

# panel label inside top-left corner to avoid y-label overlap
ax_a.text(0.03, 0.97, "(a)", transform=ax_a.transAxes, fontsize=10,
         fontweight="bold", va="top", ha="left", color=C_BLACK)


# ─────────────────────────────────────────────────────────────────────────────
# (b) Per-Signal Gain — ultra-clean, no overlap
# ─────────────────────────────────────────────────────────────────────────────
ax_b.set_title("Per-Signal Gain", fontsize=9, fontweight="bold",
               color=C_BLACK, pad=4)

x_pos = np.arange(n_cfg)
gain_cols = [C_GRAY, C_RED, C_ORANGE, C_ORANGE_LT, "#7FB3D5", C_TEAL, C_NAVY]

for (x0,x1), c in zip([(-0.4,1.4),(1.4,4.4),(4.4,6.4)], ["#FDEDEC","#FEF5E7","#E8F8F5"]):
    ax_b.axvspan(x0, x1, alpha=0.40, color=c, zorder=0)

for xmid, txt, c in [(0.5,"Core",C_RED), (2.9,"Refine",C_ORANGE), (5.4,"Saturate",C_TEAL)]:
    ax_b.text(xmid, 0.90, txt, ha="center", va="center",
             transform=ax_b.get_xaxis_transform(),
             fontsize=6.5, color=c, fontweight="bold", alpha=0.85)

for i, (d, c) in enumerate(zip(deltas, gain_cols)):
    if i == 0:
        ax_b.scatter([i],[0], c=[C_GRAY], s=55, zorder=4, edgecolor="black", lw=0.8)
        continue
    draw_lollipop(ax_b, i, d, c, yerr=delta_std[i], width=0.30, z=3)

# Only label the two largest gains; rest omitted to prevent crowding
tot = deltas[1:].sum()
for i, (d, s) in enumerate(zip(deltas, delta_std)):
    if i == 0:
        ax_b.text(i, -0.004, "Baseline", ha="center", va="top", fontsize=6,
                 color=C_GRAY, fontstyle="italic")
        continue
    pct = d/tot*100
    if i == 1:
        ax_b.annotate(f"+{d:.3f}\n({pct:.0f}%)", xy=(1, d), xytext=(2.4, 0.060),
                     fontsize=7, color=C_RED, fontweight="bold",
                     ha="center", va="center", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                               edgecolor=C_RED, lw=0.8, alpha=0.9),
                     arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.0),
                     clip_on=False)
    elif i == 2:
        ax_b.text(i, d+s+0.003, f"+{d:.3f}", ha="center", va="bottom",
                 fontsize=6.5, fontweight="bold", color=C_BLACK)
    elif i == 3:
        ax_b.text(i, d+s+0.003, f"+{d:.3f}", ha="center", va="bottom",
                 fontsize=6.5, fontweight="bold", color=C_BLACK)
    elif i == 4:
        ax_b.text(i, d+s+0.003, f"+{d:.3f}", ha="center", va="bottom",
                 fontsize=6.5, fontweight="bold", color=C_BLACK)
    elif i == 5:
        ax_b.text(i, d+s+0.003, f"+{d:.3f}", ha="center", va="bottom",
                 fontsize=6.5, fontweight="bold", color=C_BLACK)
    else:
        ax_b.text(i, d+s+0.003, f"+{d:.3f}", ha="center", va="bottom",
                 fontsize=6.5, fontweight="bold", color=C_BLACK)

ax_b.axhline(0.005, color=C_RED, ls=":", lw=1.0, alpha=0.5)
ax_b.fill_between([-0.4,6.4], 0, 0.005, alpha=0.06, color=C_RED, zorder=0)
ax_b.text(5.5, 0.013, "α=0.005", ha="center", va="bottom", fontsize=6,
         color=C_RED, fontstyle="italic")

ax_b2 = ax_b.twinx()
inv_pct = invalid_traffic*100
ax_b2.plot(x_pos, inv_pct, color=C_RED, lw=2.0, marker="s", markersize=4.5,
          zorder=4, alpha=0.55, ls="-")
ax_b2.fill_between(x_pos, inv_pct, alpha=0.07, color=C_RED, zorder=1)
ax_b2.set_ylim(82, -5)
ax_b2.set_ylabel("Invalid (%)", fontsize=8, fontweight="bold", color=C_RED)
ax_b2.tick_params(axis="y", labelcolor=C_RED, labelsize=7, length=2.5)
ax_b2.spines["top"].set_visible(False)
ax_b2.spines["right"].set_color("#DDDDDD")

ax_b2.text(0, inv_pct[0]+2.5, f"{inv_pct[0]:.0f}%", ha="center", va="bottom",
          fontsize=7, fontweight="bold", color=C_RED)
ax_b2.text(1, inv_pct[1]-4.5, f"{inv_pct[1]:.0f}%", ha="center", va="top",
          fontsize=7, fontweight="bold", color=C_RED)
ax_b2.text(6, inv_pct[6]-2.5, f"{inv_pct[6]:.0f}%", ha="center", va="top",
          fontsize=7, fontweight="bold", color=C_RED)

ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(configs_short, rotation=30, ha="right", fontsize=7.5, color=C_BLACK)
ax_b.set_ylabel("Incremental ΔRecovery", fontsize=8, fontweight="bold", color=C_BLACK)
ax_b.set_ylim(-0.006, 0.088)
ax_b.grid(axis="y", alpha=0.10, ls="--", lw=0.5)
ax_b.spines["top"].set_visible(False)

ax_b.text(0.03, 0.97, "(b)", transform=ax_b.transAxes, fontsize=10,
         fontweight="bold", va="top", ha="left", color=C_BLACK)


# ─────────────────────────────────────────────────────────────────────────────
# (c) Precision-Recall Trajectory — non-overlapping, clear bboxes
# ─────────────────────────────────────────────────────────────────────────────
p_grid = np.linspace(0,1,120)
r_grid = np.linspace(0,1,120)
P,R = np.meshgrid(p_grid,r_grid)
F1  = 2*P*R/(P+R+1e-9)

ax_c.contourf(P,R,F1, levels=10, cmap="YlGnBu", alpha=0.45)
ax_c.contour(P,R,F1, levels=[0.3,0.5,0.7,0.85],
            colors="white", linewidths=0.7, alpha=0.90)

n_pr = len(pr_points)
for i in range(n_pr-1):
    ax_c.annotate("", xy=pr_points[i+1], xytext=pr_points[i],
                 arrowprops=dict(arrowstyle="->", color=C_BLACK, lw=1.2,
                                connectionstyle="arc3,rad=0.08"))

pr_labs = ["Random","+Temporal","+Structural","+Semantic",
           "+Access","+Historical","Full PROSE","Oracle"]
# Carefully chosen non-overlapping positions
pr_pos = [
    (0.18,0.18),   # Random — lower left, below point
    (0.28,0.42),   # Temporal — left-mid
    (0.88,0.55),   # Structural — right
    (0.88,0.42),   # Semantic — right-lower
    (0.88,0.72),   # Access — right-upper
    (0.52,0.86),   # Historical — upper center
    (0.50,0.70),   # Full PROSE — upper-left of cluster
    (0.80,0.94),   # Oracle — top right
]
for i,(pr,name) in enumerate(zip(pr_points,pr_labs)):
    col = plt.cm.RdYlGn(i/(n_pr-1))
    sz  = 140 if i==n_pr-1 else 90
    ec  = C_PURPLE_DK if i==n_pr-1 else C_BLACK
    ax_c.scatter(pr[0],pr[1], c=[col], s=sz, edgecolor=ec, lw=1.2, zorder=5)
    ax_c.annotate(name, xy=(pr[0],pr[1]), xytext=pr_pos[i],
                 fontsize=6.5, fontweight="bold", color=C_BLACK,
                 ha="center", va="center", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                           edgecolor="#AAAAAA", alpha=0.92, lw=0.9),
                 arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.6,
                                connectionstyle="arc3,rad=0.1"))

# F1 labels: black text with white rounded bbox
for fx, fy, ftxt in [(0.32,0.92,"F1=0.85"),(0.18,0.58,"F1=0.70"),(0.12,0.32,"F1=0.50")]:
    ax_c.text(fx, fy, ftxt, fontsize=7, color=C_BLACK, fontweight="bold",
             ha="center", va="center", zorder=6,
             bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                       edgecolor="#CCCCCC", alpha=0.85, lw=0.7))

ax_c.set_xlabel("Precision", fontsize=8, fontweight="bold", color=C_BLACK)
ax_c.set_ylabel("Recall",    fontsize=8, fontweight="bold", color=C_BLACK)
ax_c.set_xlim(0,1)
ax_c.set_ylim(0,1)
ax_c.set_aspect("equal")
ax_c.grid(alpha=0.12, ls="--", lw=0.5)
ax_c.text(0.03,0.97, "(c)", transform=ax_c.transAxes, fontsize=10,
         fontweight="bold", va="top", ha="left", color=C_BLACK)


# ─────────────────────────────────────────────────────────────────────────────
# (d) Signal-Efficiency Frontier
# ─────────────────────────────────────────────────────────────────────────────
x_idx = np.arange(n_cfg)

ax_d.fill_between(x_idx, recovery, alpha=0.18, color=C_NAVY, zorder=1)
ax_d.plot(x_idx, recovery, color=C_NAVY, lw=2.2, marker="o", markersize=5.5,
         label="Recovery", zorder=5)
for i,v in enumerate(recovery):
    dy = 0.030 if i%2==0 else -0.030
    ax_d.text(i, v+dy, f"{v:.3f}", ha="center", va="bottom" if dy>0 else "top",
             fontsize=6.5, fontweight="bold", color=C_NAVY)

ax_d.set_xlabel("Signals Accumulated", fontsize=8, fontweight="bold", color=C_BLACK)
ax_d.set_ylabel("Selection Recovery", fontsize=8, fontweight="bold", color=C_NAVY)
ax_d.set_ylim(0,1)
ax_d.tick_params(axis="y", labelcolor=C_NAVY, labelsize=7, length=2.5)

ax_d2 = ax_d.twinx()
ax_d2.fill_between(x_idx, invalid_traffic, alpha=0.14, color=C_RED, zorder=1)
ax_d2.plot(x_idx, invalid_traffic, color=C_RED, lw=2.2, marker="s", markersize=5.5,
          label="Invalid", zorder=5)
ax_d2.text(0.35,0.46, "70%", ha="left", va="top", fontsize=7, fontweight="bold", color=C_RED)
ax_d2.text(1.0, 0.28, "15%", ha="center", va="top", fontsize=7, fontweight="bold", color=C_RED)
ax_d2.text(6.3, 0.10, "0%",  ha="left", va="bottom", fontsize=7, fontweight="bold", color=C_RED)
ax_d2.set_ylabel("Invalid Traffic", fontsize=8, fontweight="bold", color=C_RED)
ax_d2.set_ylim(1.0, -0.05)
ax_d2.tick_params(axis="y", labelcolor=C_RED, labelsize=7, length=2.5)
ax_d2.spines["top"].set_visible(False)
ax_d2.spines["right"].set_color("#DDDDDD")

ax_d.scatter([-0.3],[0.109], c=C_GRAY, s=90, marker="D",
            edgecolor=C_BLACK, lw=1.0, zorder=6)
ax_d.annotate("FTS\n(0 sig)", xy=(-0.3,0.109), xytext=(-0.75,0.20),
             fontsize=7, color=C_GRAY, fontweight="bold", ha="center",
             arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.0), zorder=6)

ax_d.axvspan(4.5,6.5, alpha=0.06, color=C_SAGE, zorder=0)
ax_d.annotate("Temporal drops\ninvalid 70%→15%", xy=(1,recovery[1]),
             xytext=(2.4,0.42), fontsize=7, color=C_TEAL, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.1), zorder=6,
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                       edgecolor="none", alpha=0.65))

ax_d.set_xticks(x_idx)
ax_d.set_xticklabels(configs_short, rotation=25, ha="right", fontsize=7.5, color=C_BLACK)
ax_d.grid(alpha=0.10, ls="--", lw=0.5)

l1,lbl1 = ax_d.get_legend_handles_labels()
l2,lbl2 = ax_d2.get_legend_handles_labels()
ax_d.legend(l1+l2, lbl1+lbl2, loc="lower right", fontsize=7,
           framealpha=0.93, edgecolor="#CCCCCC", ncol=1,
           handlelength=1.6, handletextpad=0.3, borderpad=0.25)

ax_d.text(0.03,0.97, "(d)", transform=ax_d.transAxes, fontsize=10,
         fontweight="bold", va="top", ha="left", color=C_BLACK)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
fig.suptitle("Signal Ablation: From Random Summary to Oracle-JIT",
            fontsize=10, fontweight="bold", color=C_BLACK, y=0.97)

out_pdf = OUT / "fig1_signal_waterfall.pdf"
out_png = OUT / "fig1_signal_waterfall.png"
fig.savefig(out_pdf, format="pdf", dpi=DPI, bbox_inches="tight", pad_inches=0.04,
           facecolor="white", edgecolor="none")
fig.savefig(out_png, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.04,
           facecolor="white", edgecolor="none")
print(f"PDF: {out_pdf}  ({out_pdf.stat().st_size/1024:.0f} KB)")
print(f"PNG: {out_png}  ({out_png.stat().st_size/1024:.0f} KB)")
plt.close()
print("Done.")
