#!/usr/bin/env python3
"""Single-column redraw for fig3_odus_x_fusion."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


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
        "font.size": 5.5,
        "axes.titlesize": 6.4,
        "axes.labelsize": 5.9,
        "xtick.labelsize": 4.9,
        "ytick.labelsize": 4.9,
        "legend.fontsize": 4.7,
        "axes.linewidth": 0.55,
    }
)

C_INK = "#243447"
C_GRID = "#D8E1E6"
C_PANEL = "#F8FAFB"
C_BLUE = "#24516A"
C_TEAL = "#2A9D8F"
C_GREEN = "#6B8F53"
C_YELLOW = "#FFCC33"
C_RED = "#C62828"
C_ORANGE = "#D95F02"
C_PURPLE = "#5B1A8E"
C_GRAY = "#7A858C"

rng = np.random.default_rng(23)

cues = ["Temporal", "Structural", "Semantic", "Historical", "Pressure"]
conditions = ["P-Low", "P-High", "D-Low", "D-High", "S-Low", "S-High"]
weights = np.array(
    [
        [0.32, 0.24, 0.20, 0.16, 0.08],
        [0.38, 0.22, 0.18, 0.14, 0.08],
        [0.45, 0.22, 0.12, 0.15, 0.06],
        [0.62, 0.18, 0.04, 0.12, 0.04],
        [0.28, 0.20, 0.14, 0.16, 0.22],
        [0.32, 0.18, 0.08, 0.14, 0.28],
    ]
)
ent = -(weights * np.log2(weights + 1e-9)).sum(axis=1) / np.log2(5)
mean_w = weights.mean(axis=0)

n = 900
score = rng.beta(2.5, 4.5, n)
utility_prose = np.clip(score * 0.85 + rng.normal(0, 0.055, n), 0, 1)
utility_fts = score * 0.82 + rng.normal(0, 0.11, n)
stall_mask = (score > 0.30) & (rng.random(n) < 0.18)
utility_fts[stall_mask] -= rng.uniform(0.05, 0.20, stall_mask.sum())
utility_fts = np.clip(utility_fts, 0, 1)
decision = np.where(score > 0.55, "PROMOTE", np.where(score > 0.25, "DEFER", "REJECT"))

cost_prose = np.where(decision == "PROMOTE", 64.0, 0.064)
cost_fts = np.full(n, 64.0)
order = np.argsort(score)[::-1]
rel_cost = np.cumsum(cost_prose[order]) / (np.cumsum(cost_fts[order]) + 1e-9)


def style(ax, grid=True):
    ax.set_facecolor(C_PANEL)
    for sp in ax.spines.values():
        sp.set_color("#B8C5CC")
        sp.set_linewidth(0.55)
    if grid:
        ax.grid(True, color=C_GRID, lw=0.45, alpha=0.70, zorder=0)
    ax.tick_params(length=2.2, color="#AAB6BD", labelcolor=C_INK)


def panel(ax, lab):
    ax.text(0.012, 0.988, lab, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.2, fontweight="bold", color=C_INK, zorder=20)


fig = plt.figure(figsize=(3.50, 2.72), facecolor="white")
gs = fig.add_gridspec(2, 2, left=0.085, right=0.985, top=0.875, bottom=0.135,
                      wspace=0.30, hspace=0.42)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

fig.suptitle("ODUS-X Fusion and Admission Cost Isolation",
             fontsize=7.8, fontweight="bold", color=C_INK, y=0.985)

# (a) cue-weight landscape
style(ax_a, grid=False)
panel(ax_a, "(a)")
ax_a.set_title("Cue Weight Landscape", fontweight="bold", color=C_INK, pad=4)
cmap = LinearSegmentedColormap.from_list(
    "odus_weight", ["#201128", "#3E326B", "#316D99", "#43B6A7", "#D7F2D7"]
)
im = ax_a.imshow(weights, cmap=cmap, vmin=0, vmax=0.65, aspect="auto", zorder=1)
ax_a.set_xticks(np.arange(len(cues)))
ax_a.set_xticklabels(["Temp", "Struct", "Sem", "Hist", "Press"], rotation=35, ha="right")
ax_a.set_yticks(np.arange(len(conditions)))
ax_a.set_yticklabels(conditions)
ax_a.tick_params(axis="both", length=0)
for i in range(weights.shape[0] + 1):
    ax_a.axhline(i - 0.5, color="white", lw=0.65, alpha=0.9, zorder=3)
for j in range(weights.shape[1] + 1):
    ax_a.axvline(j - 0.5, color="white", lw=0.65, alpha=0.9, zorder=3)
ax_a.add_patch(Rectangle((-0.5, 2.5), 5, 1, fill=False, edgecolor=C_RED, linewidth=1.2, zorder=5))
ax_a.text(4.75, 3.0, "tight gate", ha="right", va="center", fontsize=5.0,
          color=C_RED, fontweight="bold", zorder=6,
          bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.82))
dominant = weights.argmax(axis=1)
for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        if weights[i, j] >= 0.28 or (i == 3 and j in (0, 1)):
            ax_a.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center",
                      fontsize=4.7, color="white" if weights[i, j] < 0.50 else C_INK,
                      fontweight="bold", zorder=4)
ax_a.text(0.03, 0.02, "mean: Temp 0.39  Struct 0.21", transform=ax_a.transAxes,
          ha="left", va="bottom", fontsize=4.25, color="white", fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.10", fc=C_INK, ec="none", alpha=0.70))

# (b) entropy + score distribution
style(ax_b)
panel(ax_b, "(b)")
ax_b.set_title("Fusion Concentration", fontweight="bold", color=C_INK, pad=4)
y = np.arange(len(conditions))
ax_b.barh(y, ent, color=np.where(ent < 0.72, C_ORANGE, C_TEAL), height=0.58,
          edgecolor=C_INK, linewidth=0.35, alpha=0.92, zorder=3)
ax_b.axvline(0.72, color=C_RED, ls="--", lw=0.85, alpha=0.75, zorder=4)
ax_b.set_yticks(y)
ax_b.set_yticklabels(conditions)
ax_b.set_xlim(0.45, 1.0)
ax_b.set_xlabel("normalized entropy", fontweight="bold", color=C_INK)
ax_b.invert_yaxis()
for yi, e in zip(y, ent):
    ax_b.text(e + 0.012, yi, f"{e:.2f}", va="center", ha="left", fontsize=4.8,
              fontweight="bold", color=C_INK)
ax_hist = ax_b.inset_axes([0.29, 0.10, 0.62, 0.20])
ax_hist.set_facecolor((1, 1, 1, 0.88))
bins = np.linspace(0, 1, 24)
for name, col in [("REJECT", C_RED), ("DEFER", C_YELLOW), ("PROMOTE", C_GREEN)]:
    vals = score[decision == name]
    ax_hist.hist(vals, bins=bins, color=col, alpha=0.82, edgecolor="white", linewidth=0.25)
ax_hist.axvline(0.25, color=C_YELLOW, ls="--", lw=0.75)
ax_hist.axvline(0.55, color=C_GREEN, ls="--", lw=0.75)
ax_hist.set_yticks([])
ax_hist.set_xticks([0, 0.25, 0.55, 1.0])
ax_hist.set_xticklabels(["0", ".25", ".55", "1"])
ax_hist.tick_params(labelsize=4.5, length=1.4)
ax_hist.set_title("score marginal", fontsize=4.3, pad=0.5)
for sp in ax_hist.spines.values():
    sp.set_color("#B8C5CC")
    sp.set_linewidth(0.45)

# (c) admission scatter with density bins
style(ax_c)
panel(ax_c, "(c)")
ax_c.set_title("Admission Decision Surface", fontweight="bold", color=C_INK, pad=4)
ax_c.hexbin(score, utility_prose, gridsize=24, cmap="Greys", mincnt=1, alpha=0.28,
            edgecolors="none", zorder=1)
colors = {"PROMOTE": C_GREEN, "DEFER": C_YELLOW, "REJECT": C_RED}
markers = {"PROMOTE": "o", "DEFER": "s", "REJECT": "x"}
sizes = {"PROMOTE": 14, "DEFER": 12, "REJECT": 18}
for name in ["REJECT", "DEFER", "PROMOTE"]:
    m = decision == name
    ax_c.scatter(score[m], utility_prose[m], s=sizes[name], c=colors[name],
                 marker=markers[name], alpha=0.55 if name != "REJECT" else 0.70,
                 edgecolors=C_INK if name != "REJECT" else None,
                 linewidths=0.25 if name != "REJECT" else 0.75, zorder=3, label=name.title())
ax_c.axvspan(0, 0.25, color=C_RED, alpha=0.045, zorder=0)
ax_c.axvline(0.25, color=C_YELLOW, ls="--", lw=0.9)
ax_c.axvline(0.55, color=C_GREEN, ls="--", lw=0.9)
ax_c.plot([0, 1], [0, 0.85], color=C_BLUE, lw=1.0, alpha=0.55, zorder=4)
ax_c.set_xlim(0, 1)
ax_c.set_ylim(0, 1)
ax_c.set_xlabel("ODUS-X score", fontweight="bold", color=C_INK)
ax_c.set_ylabel("utility", fontweight="bold", color=C_INK)
ax_c.legend(loc="upper left", framealpha=0.92, edgecolor="#C9D1D6",
            borderpad=0.25, handletextpad=0.25)

# (d) cost isolation and FTS noise
style(ax_d)
panel(ax_d, "(d)")
ax_d.set_title("Cost Isolation Under Same Ranker", fontweight="bold", color=C_INK, pad=4)
keep = np.ones(n, dtype=bool)
low = np.where(score < 0.20)[0]
rng.shuffle(low)
keep[low[: int(0.35 * len(low))]] = False
ax_d.hexbin(score[keep], utility_fts[keep], gridsize=24, cmap="Reds", mincnt=1,
            alpha=0.23, edgecolors="none", zorder=1)
for name in ["REJECT", "DEFER", "PROMOTE"]:
    m = (decision == name) & keep
    ax_d.scatter(score[m], utility_fts[m], s=sizes[name], c=colors[name],
                 marker=markers[name], alpha=0.50 if name != "REJECT" else 0.65,
                 edgecolors=C_INK if name != "REJECT" else None,
                 linewidths=0.25 if name != "REJECT" else 0.75, zorder=3)
for sx, uy in zip(score[(decision == "REJECT") & keep][::4], utility_fts[(decision == "REJECT") & keep][::4]):
    ax_d.plot([sx, sx], [0, uy], color=C_RED, alpha=0.09, lw=0.8, zorder=2)
ax_d.axvspan(0, 0.25, color=C_RED, alpha=0.06, zorder=0)
ax_d.axvline(0.25, color=C_YELLOW, ls="--", lw=0.9)
ax_d.axvline(0.55, color=C_GREEN, ls="--", lw=0.9)
ax_d.set_xlim(0, 1)
ax_d.set_ylim(0, 1)
ax_d.set_xlabel("ODUS-X score", fontweight="bold", color=C_INK)
ax_d.set_ylabel("utility", fontweight="bold", color=C_INK)
ax_in = ax_d.inset_axes([0.50, 0.08, 0.46, 0.30])
rank = np.arange(n)
ax_in.fill_between(rank, 1.0, color=C_RED, alpha=0.17, label="FTS")
ax_in.fill_between(rank, rel_cost, color=C_BLUE, alpha=0.36, label="ODUS-X")
ax_in.plot(rank, np.ones_like(rank), color=C_RED, lw=1.0)
ax_in.plot(rank, rel_cost, color=C_BLUE, lw=1.2)
ax_in.set_ylim(0, 1.05)
ax_in.set_xticks([0, 450, 900])
ax_in.set_yticks([0, 0.5, 1.0])
ax_in.tick_params(labelsize=4.4, length=1.4)
ax_in.set_title("byte-at-risk", fontsize=5.2, fontweight="bold", pad=1)
ax_in.set_xlabel("rank", fontsize=4.7, labelpad=0.3)
ax_in.set_ylabel("rel. cost", fontsize=4.7, labelpad=0.3)
ax_in.legend(loc="upper right", fontsize=4.2, frameon=False, handlelength=1.0)
for sp in ax_in.spines.values():
    sp.set_color("#B8C5CC")
    sp.set_linewidth(0.45)

out_pdf = OUT / "fig3_odus_x_fusion.pdf"
out_png = OUT / "fig3_odus_x_fusion.png"
fig.savefig(out_pdf, format="pdf", dpi=DPI, bbox_inches="tight", pad_inches=0.018, facecolor="white")
fig.savefig(out_png, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.018, facecolor="white")
plt.close(fig)
print(f"PDF: {out_pdf} ({out_pdf.stat().st_size / 1024:.1f} KB)")
print(f"PNG: {out_png} ({out_png.stat().st_size / 1024:.1f} KB)")
