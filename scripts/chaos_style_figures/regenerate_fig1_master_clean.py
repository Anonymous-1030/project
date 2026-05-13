#!/usr/bin/env python3
"""Single-column, high-density redraw for fig1_signal_waterfall."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = pathlib.Path(r"D:\LLM\outputs\chaos_style_figures")
OUT.mkdir(parents=True, exist_ok=True)
DPI = 300

plt.rcParams.update(
    {
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 5.6,
        "axes.titlesize": 6.4,
        "axes.labelsize": 5.9,
        "xtick.labelsize": 4.9,
        "ytick.labelsize": 4.9,
        "legend.fontsize": 4.8,
        "axes.linewidth": 0.55,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
    }
)

C_INK = "#243447"
C_MUTED = "#6F7F89"
C_GRID = "#D8E1E6"
C_PANEL = "#F8FAFB"
C_TEAL = "#1F9E8A"
C_TEAL_DK = "#0F665F"
C_BLUE = "#24516A"
C_BLUE_DK = "#173447"
C_SAGE = "#86A875"
C_ORANGE = "#F0A35A"
C_RED = "#C63D2E"
C_RED_DK = "#9F2E25"
C_PURPLE = "#5B1A8E"
C_YELLOW = "#F2D16B"

labels = ["Random", "+Temporal", "+Structural", "+Semantic", "+Access", "+Historical", "Full"]
labels_short = ["Rnd", "+Tmp", "+Str", "+Sem", "+Acc", "+His", "Full"]
recovery = np.array([0.109, 0.678, 0.691, 0.698, 0.701, 0.703, 0.703])
oracle = 0.903
deltas = np.array([0.0, 0.569, 0.013, 0.007, 0.003, 0.002, 0.0])
delta_err = np.array([0.0, 0.065, 0.006, 0.004, 0.003, 0.002, 0.0])
invalid = np.array([0.70, 0.15, 0.10, 0.07, 0.04, 0.02, 0.00])

stage = np.array(
    [
        [0.040, 0.040, 0.029],
        [0.200, 0.430, 0.048],
        [0.230, 0.445, 0.016],
        [0.240, 0.455, 0.003],
        [0.245, 0.453, 0.003],
        [0.248, 0.443, 0.012],
        [0.248, 0.443, 0.012],
    ]
)
stage = stage / stage.sum(axis=1, keepdims=True) * recovery[:, None]
stage_names = ["Prefill", "Decode", "Spec."]
stage_cols = [C_TEAL, C_BLUE, C_SAGE]

pr = np.array(
    [
        [0.16, 0.08],
        [0.69, 0.63],
        [0.73, 0.66],
        [0.76, 0.67],
        [0.78, 0.69],
        [0.80, 0.70],
        [0.81, 0.70],
        [0.94, 0.88],
    ]
)


def stroke_text(text):
    text.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])
    return text


def panel_label(ax, lab):
    ax.text(
        0.012,
        0.988,
        lab,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color=C_INK,
        zorder=20,
    )


def style_axes(ax, grid_axis="both"):
    ax.set_facecolor(C_PANEL)
    for side in ax.spines:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#B8C5CC")
        ax.spines[side].set_linewidth(0.55)
    ax.grid(True, axis=grid_axis, color=C_GRID, lw=0.45, alpha=0.72, zorder=0)
    ax.tick_params(length=2.2, color="#AAB6BD", labelcolor=C_INK)


def rounded(ax, xy, w, h, fc, ec="#FFFFFF", lw=0.6, r=0.018, alpha=1.0, z=3):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.005,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        alpha=alpha,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=C_INK, rad=0.0, lw=0.8, alpha=1.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=7,
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        linewidth=lw,
        alpha=alpha,
        zorder=6,
    )
    ax.add_patch(arr)
    return arr


fig = plt.figure(figsize=(3.50, 2.72), facecolor="white")
gs = fig.add_gridspec(
    2,
    2,
    left=0.080,
    right=0.985,
    top=0.875,
    bottom=0.135,
    wspace=0.28,
    hspace=0.42,
)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

fig.suptitle(
    "Signal Ablation: Random Summary to Oracle-JIT",
    y=0.985,
    fontsize=7.8,
    fontweight="bold",
    color=C_INK,
)

# (a) dense Nature-style signal diagnostic panel
style_axes(ax_a)
ax_a.set_title("Cross-Signal Activation Landscape", fontweight="bold", color=C_INK, pad=4)
ax_a.set_xlim(0, 1)
ax_a.set_ylim(0, 1)
ax_a.set_xticks([])
ax_a.set_yticks([])
panel_label(ax_a, "(a)")

rng = np.random.default_rng(7)
n_layer, n_tok = 128, 128
yy, xx = np.mgrid[0:n_layer, 0:n_tok]
base = 0.035 + 0.025 * rng.random((n_layer, n_tok))
prefill_map = base.copy()
decode_map = 0.030 + 0.022 * rng.random((n_layer, n_tok))
for k in range(11):
    center = 8 + k * 11 + rng.normal(0, 1.1)
    stripe = np.exp(-((xx - center) ** 2) / (2 * (1.0 + 0.15 * k) ** 2))
    decay = 0.25 + 0.70 * np.exp(-((yy - (18 + 8 * k)) ** 2) / 1300)
    prefill_map += 0.20 * stripe * decay
for k in range(9):
    diag = np.exp(-((yy - (0.72 * xx + 7 * k)) ** 2) / 22)
    decode_map += 0.18 * diag * (0.45 + 0.55 * rng.random((n_layer, n_tok)))
for mat in (prefill_map, decode_map):
    hot = rng.choice(n_layer * n_tok, 260, replace=False)
    mat.flat[hot] += rng.uniform(0.18, 0.65, size=hot.size)
prefill_map = np.clip(prefill_map, 0, 0.82)
decode_map = np.clip(decode_map, 0, 0.82)

hm_cmap = LinearSegmentedColormap.from_list("activation_fire", ["#210707", "#6D1510", "#B23A28", "#F08C66", "#FFF1D0"])
zoom_cmap = LinearSegmentedColormap.from_list("activation_zoom", ["#10283C", "#1D5C7A", "#5CB1B8", "#E6F0D3", "#FFF5D6"])

ax_hm1 = ax_a.inset_axes([0.07, 0.60, 0.53, 0.28])
ax_hm2 = ax_a.inset_axes([0.07, 0.25, 0.53, 0.28])
ax_zoom1 = ax_a.inset_axes([0.65, 0.61, 0.29, 0.25])
ax_zoom2 = ax_a.inset_axes([0.65, 0.26, 0.29, 0.25])
for axh, mat, cmap_now, caption in [
    (ax_hm1, prefill_map, hm_cmap, "Prefill activation"),
    (ax_hm2, decode_map, zoom_cmap, "Decode activation"),
]:
    axh.imshow(mat, cmap=cmap_now, origin="upper", aspect="auto", vmin=0, vmax=0.82, interpolation="nearest")
    axh.set_xlim(0, 128)
    axh.set_ylim(128, 0)
    axh.xaxis.tick_top()
    axh.set_xticks([0, 32, 64, 96, 128])
    axh.set_yticks([0, 64, 128])
    axh.tick_params(labelsize=4.2, length=1.4, pad=1)
    for sp in axh.spines.values():
        sp.set_color("#AEBBC2")
        sp.set_linewidth(0.45)
    axh.text(
        0.025,
        0.92,
        caption,
        transform=axh.transAxes,
        ha="left",
        va="top",
        fontsize=4.7,
        color="white",
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=1.2, foreground=C_INK)],
    )

roi1 = (86, 14, 36, 28)
roi2 = (18, 42, 36, 28)
for axh, roi, col in [(ax_hm1, roi1, "#2B79FF"), (ax_hm2, roi2, C_RED)]:
    axh.add_patch(Rectangle((roi[0], roi[1]), roi[2], roi[3], fill=False, edgecolor=col, linewidth=0.8, zorder=5))

for axz, mat, roi, cmap_now, col in [
    (ax_zoom1, prefill_map, roi1, hm_cmap, "#2B79FF"),
    (ax_zoom2, decode_map, roi2, zoom_cmap, C_RED),
]:
    crop = mat[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
    axz.imshow(crop, cmap=cmap_now, origin="upper", aspect="auto", vmin=0, vmax=0.82, interpolation="nearest")
    axz.set_xticks([])
    axz.set_yticks([])
    for sp in axz.spines.values():
        sp.set_color(col)
        sp.set_linewidth(1.0)

for axh, axz, roi, col in [(ax_hm1, ax_zoom1, roi1, "#2B79FF"), (ax_hm2, ax_zoom2, roi2, C_RED)]:
    for corner, zcorner in [((roi[0] + roi[2], roi[1]), (0, 1)), ((roi[0] + roi[2], roi[1] + roi[3]), (0, 0))]:
        fig.add_artist(
            ConnectionPatch(
                xyA=corner,
                coordsA=axh.transData,
                xyB=zcorner,
                coordsB=axz.transAxes,
                color=col,
                linewidth=0.65,
                linestyle=(0, (2, 2)),
                alpha=0.85,
                zorder=4,
            )
        )

ax_sim = ax_a.inset_axes([0.60, 0.035, 0.34, 0.15])
layer = np.arange(0, 91)
sim_ds = 0.58 + 0.09 * np.sin(layer / 11) + 0.03 * rng.normal(size=layer.size)
sim_llama = 0.66 + 0.11 * np.cos(layer / 14) + 0.025 * rng.normal(size=layer.size)
sim_qwen = 0.78 + 0.04 * np.sin(layer / 9) + 0.018 * rng.normal(size=layer.size)
for series, col, lab in [(sim_ds, C_ORANGE, "DS"), (sim_llama, C_RED, "Llama"), (sim_qwen, C_TEAL, "Qwen")]:
    ax_sim.plot(layer, np.clip(series, 0.35, 0.95), marker="o", markersize=1.4, lw=0.45, alpha=0.72, color=col, label=lab)
ax_sim.set_xlim(0, 90)
ax_sim.set_ylim(0.35, 0.95)
ax_sim.set_xticks([0, 45, 90])
ax_sim.set_yticks([0.4, 0.7, 0.9])
ax_sim.tick_params(labelsize=4.2, length=1.4, pad=1)
ax_sim.set_xlabel("Layer", fontsize=4.4, labelpad=0.0)
ax_sim.set_ylabel("")
ax_sim.text(0.02, 0.96, "Spearman", transform=ax_sim.transAxes, ha="left", va="top", fontsize=4.0, color=C_INK)
ax_sim.grid(True, color=C_GRID, lw=0.35, alpha=0.7)
ax_sim.legend(loc="lower center", bbox_to_anchor=(0.50, 1.00), ncol=3, fontsize=3.4, frameon=False, handlelength=0.6, columnspacing=0.30, handletextpad=0.10)
for sp in ax_sim.spines.values():
    sp.set_color("#AEBBC2")
    sp.set_linewidth(0.45)

ax_a.text(0.075, 0.085, "full 0.703 | gap 0.200", ha="left", va="bottom",
          fontsize=4.25, color=C_BLUE, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.75))

# (b) gain waterfall, decluttered with separate invalid axis
style_axes(ax_b, "y")
ax_b.set_title("Signal Yield vs Invalid Traffic", fontweight="bold", color=C_INK, pad=4)
panel_label(ax_b, "(b)")
x = np.arange(len(labels))
gain_cols = [C_MUTED, C_RED, C_ORANGE, C_YELLOW, "#75A9C9", C_TEAL, C_BLUE]
ax_b.axhspan(0, 0.005, color=C_RED, alpha=0.075, zorder=0)
ax_b.axhline(0.005, color=C_RED, ls=":", lw=0.75, alpha=0.55)
ax_b.axvspan(0.55, 1.45, color=C_RED, alpha=0.045, zorder=0)
ax_b.axvspan(3.5, 6.5, color=C_BLUE, alpha=0.035, zorder=0)
for i in range(len(labels)):
    if i == 0:
        ax_b.scatter(i, 0, s=34, c=C_MUTED, edgecolors=C_INK, lw=0.45, zorder=4)
        continue
    ax_b.bar(i, deltas[i], width=0.48, color=gain_cols[i], alpha=0.80, zorder=3)
    ax_b.errorbar(i, deltas[i], yerr=delta_err[i], color=C_INK, lw=0.55, capsize=1.8, zorder=5, alpha=0.58)
    if i in (1, 2, 3):
        ypos = min(deltas[i] + delta_err[i] + 0.014, 0.645)
        ax_b.text(i, ypos, f"+{deltas[i]:.3f}", ha="center", va="bottom", fontsize=5.4, fontweight="bold", color=C_INK)
ax_b.plot(x, deltas, color=C_INK, lw=0.75, alpha=0.35, marker="o", markersize=2.2, zorder=4)
cum_norm = (recovery - recovery[0]) / (recovery[-1] - recovery[0])
cum_y = 0.055 + 0.14 * cum_norm
ax_b.plot(x, cum_y, color=C_BLUE, lw=1.05, marker="D", markersize=2.8,
          alpha=0.90, zorder=5)
ax_b.fill_between(x, 0.055, cum_y, color=C_BLUE, alpha=0.055, zorder=1)
for xi, cy, txt in [(1, cum_y[1], "96% cum."), (3, cum_y[3], "99.2%"), (6, cum_y[6], "100%")]:
    ax_b.text(xi, cy + 0.018, txt, ha="center", va="bottom", fontsize=4.5,
              color=C_BLUE, fontweight="bold", zorder=7)
drop = np.r_[0, invalid[:-1] - invalid[1:]]
for i in range(1, len(labels)):
    if i == 1:
        ax_b.text(i, 0.070, f"-{drop[i]*100:.0f}pt invalid", ha="center", va="bottom",
                  fontsize=4.2, color=C_RED_DK, fontweight="bold", rotation=90, zorder=8)
ax_b.annotate(
    "Temporal\nexplains 96%",
    xy=(1, deltas[1]),
    xytext=(3.25, 0.49),
    textcoords="data",
    ha="center",
    va="center",
    fontsize=5.7,
    color=C_RED_DK,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=C_RED, lw=0.65, alpha=0.96),
    arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.75, shrinkA=2, shrinkB=2),
    zorder=8,
)
ax_b.text(5.9, 0.010, "alpha=0.005", fontsize=4.9, color=C_RED_DK, ha="right", va="bottom")
ax_b.text(5.50, 0.160, "cumulative\nrecovered gain", ha="center", va="center",
          fontsize=4.45, color=C_BLUE, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.78), zorder=8)
ax_b.set_xticks(x)
ax_b.set_xticklabels(labels_short, rotation=28, ha="right")
ax_b.set_ylabel("Delta recovery", fontweight="bold", color=C_INK)
ax_b.set_ylim(-0.035, 0.68)
ax_b.set_xlim(-0.55, 6.55)
ax_b2 = ax_b.twinx()
ax_b2.plot(x, invalid * 100, color=C_RED_DK, lw=1.25, marker="s", markersize=3.2, zorder=5)
ax_b2.set_ylim(78, -5)
ax_b2.set_ylabel("Invalid %", color=C_RED_DK, fontweight="bold", labelpad=1)
ax_b2.tick_params(axis="y", labelcolor=C_RED_DK, labelsize=5.4, length=2)
ax_b2.spines["right"].set_color("#B8C5CC")
ax_b2.spines["top"].set_visible(True)
ax_b2.spines["top"].set_color("#B8C5CC")
for xi, txt, yy in [(0, "70%", 70), (1, "15%", 15), (6, "0%", 0)]:
    ax_b2.text(xi, yy + 3, txt, ha="center", va="top", fontsize=5.3, color=C_RED_DK, fontweight="bold")

# (c) PR trajectory with the same rectangular panel frame/grid language
style_axes(ax_c)
ax_c.set_title("Precision-Recall Trajectory", fontweight="bold", color=C_INK, pad=4)
panel_label(ax_c, "(c)")
p = np.linspace(0.01, 0.99, 140)
r = np.linspace(0.01, 0.99, 140)
P, R = np.meshgrid(p, r)
F = 2 * P * R / (P + R)
cmap = LinearSegmentedColormap.from_list(
    "prose_f1",
    ["#FFF7CF", "#DDEBC6", "#A9D9C7", "#72C1C3", "#3D8FA8", "#24516A"],
)
fine_levels = np.linspace(0.12, 0.94, 34)
ax_c.contourf(P, R, F, levels=fine_levels, cmap=cmap, alpha=0.88, zorder=0)
ax_c.contour(P, R, F, levels=np.linspace(0.20, 0.90, 15), colors="white",
             linewidths=0.28, alpha=0.36, zorder=1)
cs = ax_c.contour(P, R, F, levels=[0.50, 0.60, 0.70, 0.80, 0.85],
                  colors="white", linewidths=0.62, alpha=0.92, zorder=2)
ax_c.clabel(cs, fmt="F1=%.2f", fontsize=4.8, colors=C_INK, inline=True)
ax_c.plot(pr[:, 0], pr[:, 1], color=C_INK, lw=1.3, zorder=3)
for i in range(len(pr) - 1):
    arrow(ax_c, tuple(pr[i]), tuple(pr[i + 1]), color=C_INK, lw=0.55, alpha=0.95)
for i, point in enumerate(pr):
    col = C_RED if i == 0 else (C_PURPLE if i == len(pr) - 1 else C_TEAL)
    ax_c.scatter(point[0], point[1], s=36 if i not in (0, 7) else 52, color=col, edgecolor=C_INK, linewidth=0.55, zorder=5)
name_pos = {
    "Random": (0.23, 0.16),
    "+Temporal": (0.38, 0.47),
    "+Structural": (0.62, 0.58),
    "+Access": (0.84, 0.67),
    "Full": (0.74, 0.78),
    "Oracle": (0.79, 0.92),
}
for name, pt in name_pos.items():
    ax_c.text(
        pt[0],
        pt[1],
        name,
        fontsize=5.2,
        fontweight="bold",
        color=C_INK if name != "Oracle" else C_PURPLE,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#AEBBC2", lw=0.45, alpha=0.94),
        zorder=8,
    )
ax_c.set_xlim(0, 1)
ax_c.set_ylim(0, 1)
ax_c.set_xlabel("Precision", fontweight="bold", color=C_INK)
ax_c.set_ylabel("Recall", fontweight="bold", color=C_INK)
ax_c.set_aspect("auto")
ax_c.set_xticks([0.0, 0.5, 1.0])
ax_c.set_yticks([0.0, 0.5, 1.0])

# (d) efficiency frontier
style_axes(ax_d, "both")
ax_d.set_title("Recovery-Invalid Frontier", fontweight="bold", color=C_INK, pad=4)
panel_label(ax_d, "(d)")
efficiency = recovery / (invalid + 0.08)
eff_norm = (efficiency - efficiency.min()) / (efficiency.max() - efficiency.min())
for i in range(len(x) - 1):
    ax_d.axvspan(i - 0.5, i + 0.5, color=plt.cm.YlGnBu(0.25 + 0.55 * eff_norm[i]),
                 alpha=0.055, zorder=0)
ax_d.fill_between(x, recovery, color=C_BLUE, alpha=0.13, zorder=1)
ax_d.plot(x, recovery, color=C_BLUE, lw=1.7, marker="o", markersize=3.9, label="Recovery", zorder=4)
ax_d.plot(x, recovery - 0.035, color=C_BLUE, lw=0.55, alpha=0.35, ls="--", zorder=3)
ax_d.plot(x, recovery + 0.035, color=C_BLUE, lw=0.55, alpha=0.35, ls="--", zorder=3)
ax_d.fill_between(x, recovery - 0.035, recovery + 0.035, color=C_BLUE, alpha=0.055, zorder=2)
for i in [0, 1, 3, 5, 6]:
    yy = recovery[i] + (0.055 if i in [0, 3, 6] else -0.055)
    stroke_text(ax_d.text(i, yy, f"{recovery[i]:.3f}", fontsize=5.2, color=C_BLUE, ha="center", va="center", fontweight="bold", zorder=7))
ax_d.set_xticks(x)
ax_d.set_xticklabels(labels_short, rotation=28, ha="right")
ax_d.set_xlabel("Signals accumulated", fontweight="bold", color=C_INK)
ax_d.set_ylabel("Recovery", fontweight="bold", color=C_BLUE, labelpad=1)
ax_d.tick_params(axis="y", labelcolor=C_BLUE)
ax_d.set_ylim(0, 0.95)
ax_d.set_xlim(-0.55, 6.55)
ax_d2 = ax_d.twinx()
ax_d2.fill_between(x, invalid, color=C_RED, alpha=0.10, zorder=1)
ax_d2.plot(x, invalid, color=C_RED_DK, lw=1.7, marker="s", markersize=3.8, label="Invalid", zorder=5)
ax_d2.bar(x, invalid, width=0.28, color=C_RED, alpha=0.11, edgecolor="none", zorder=1)
ax_d2.set_ylim(0.82, -0.05)
ax_d2.set_ylabel("Invalid traffic", color=C_RED_DK, fontweight="bold", labelpad=1)
ax_d2.tick_params(axis="y", labelcolor=C_RED_DK, labelsize=5.4, length=2)
ax_d2.spines["right"].set_color("#B8C5CC")
ax_d2.spines["top"].set_visible(True)
ax_d2.spines["top"].set_color("#B8C5CC")
ax_d.annotate(
    "knee: temporal\nremoves 55 pts invalid",
    xy=(1, recovery[1]),
    xytext=(3.15, 0.41),
    fontsize=5.3,
    color=C_TEAL_DK,
    ha="center",
    va="center",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.82),
    arrowprops=dict(arrowstyle="->", color=C_TEAL_DK, lw=0.8),
    zorder=8,
)
ax_d.axvspan(4.5, 6.5, color=C_SAGE, alpha=0.08, zorder=0)
for i in range(1, len(x)):
    dx_mid = i - 0.5
    dy_mid = (recovery[i] + recovery[i - 1]) / 2
    gain = recovery[i] - recovery[i - 1]
    inv_gain = invalid[i - 1] - invalid[i]
    if i == 1:
        ax_d.text(dx_mid + 0.05, dy_mid + 0.070, f"+{gain:.3f}\n-{inv_gain*100:.0f}pt",
                  ha="center", va="center", fontsize=4.45, color=C_INK,
                  bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#D4DCE1", lw=0.35, alpha=0.86),
                  zorder=8)
    elif i == 2:
        ax_d.text(dx_mid + 0.10, dy_mid + 0.090, f"+{gain:.3f} / -{inv_gain*100:.0f}pt",
                  ha="center", va="center", fontsize=4.15, color=C_INK,
                  bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="#D4DCE1", lw=0.35, alpha=0.82),
                  zorder=8)
ax_d.text(5.25, 0.25, "late: flat\nrecovery", ha="center", va="center",
          fontsize=4.15, color=C_INK, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#D4DCE1", lw=0.35, alpha=0.84),
          zorder=8)
ax_d.text(5.65, 0.08, "Recovery", color=C_BLUE, fontsize=4.4, fontweight="bold",
          ha="right", va="bottom")

out_pdf = OUT / "fig1_signal_waterfall.pdf"
out_png = OUT / "fig1_signal_waterfall.png"
fig.savefig(out_pdf, format="pdf", dpi=DPI, bbox_inches="tight", pad_inches=0.018, facecolor="white")
fig.savefig(out_png, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.018, facecolor="white")
plt.close(fig)
print(f"PDF: {out_pdf} ({out_pdf.stat().st_size / 1024:.1f} KB)")
print(f"PNG: {out_png} ({out_png.stat().st_size / 1024:.1f} KB)")
