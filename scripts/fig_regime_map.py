"""
Operating-regime map figure.

2D phase-diagram-style plot (3.33 in x 2.9 in) that shades the decode-slack
x spill-pressure plane according to which optimization matters most:

  - SBFI-limited      : high spill, low slack  (bandwidth-bound tail)
  - scorer-limited    : low spill, high slack  (liveness-bound)
  - balanced          : middle zone
  - free zone         : negligible gains      (high slack AND low spill)

Data basis (from results/reviewer_response_v2_results.json):

  * SW admission overhead ~ 47 us (fix1_critical_path slack=0us row)
  * SBFI payload savings ~ 63%   (PROSE 50 MB vs FTS 135 MB)
  * Scorer ceiling gap   ~ 18 pp (Perfect-SBFI vs PROSE)

Benefit model:
    B_sbfi(slack, spill)   = spill_weight(spill) * payload_savings
                           * exposed_frac(slack)
    B_scorer(slack, spill) = ceiling_gap
                           * (1 - spill_weight(spill)) * slack_utilization

with
    exposed_frac(s)   = max(0, (47 - s) / 47)          # slack-bound
    spill_weight(r)   = sigmoid((r - 1) * 4)           # spill-bound
    slack_util(s)     = 1 - exposed_frac(s)            # scorer needs headroom

The dominant regime is argmax(B_sbfi, B_scorer) where the max exceeds a
threshold; otherwise "free zone".

Six real operating points from the critical-path sweep and common LLM
deployments are overlaid to anchor the phase diagram.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D


ROOT = Path("d:/LLM/prose_v2")
RESULTS = ROOT / "results" / "reviewer_response_v2_results.json"
OUT_DIR = ROOT / "figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 7.2,
        "font.weight": "bold",
        "axes.labelsize": 7.6,
        "axes.labelweight": "bold",
        "axes.titlesize": 8.2,
        "axes.titleweight": "bold",
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.4,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_constants():
    raw = json.load(open(RESULTS))
    crit = raw["fix1_critical_path"]
    sw_ov = crit["slack=0us"]["sw_overhead_us"]  # ~46.7
    # from the same file we know the overall story:
    payload_savings = 0.63    # 135 MB → 50 MB (63%)
    scorer_gap_pp = 0.18      # 0.84 → 1.00
    return sw_ov, payload_savings, scorer_gap_pp, crit


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def regime_field(slack, spill, sw_ov, payload_savings, scorer_gap):
    """Return (regime_id, dominance_strength) at every grid point.

    regime_id: 0 = free, 1 = scorer-limited, 2 = balanced, 3 = SBFI-limited
    dominance_strength in [0, 1] is the absolute winning margin.
    """
    # 1D factors
    exposed_frac = np.clip((sw_ov - slack) / sw_ov, 0.0, 1.0)
    slack_util = 1.0 - exposed_frac
    spill_wt = sigmoid((spill - 1.0) * 4.0)  # smoothly 0..1 around spill=1

    # 2D benefit fields
    B_sbfi = spill_wt * payload_savings * exposed_frac
    B_scorer = (1 - spill_wt) * scorer_gap * slack_util

    # Combine
    total_benefit = B_sbfi + B_scorer
    diff = B_sbfi - B_scorer

    regime = np.zeros_like(slack, dtype=int)
    regime[total_benefit < 0.05] = 0                         # free zone
    regime[(total_benefit >= 0.05) & (diff > 0.03)] = 3       # SBFI
    regime[(total_benefit >= 0.05) & (diff < -0.03)] = 1      # scorer
    regime[(total_benefit >= 0.05) & (np.abs(diff) <= 0.03)] = 2  # balanced

    return regime, B_sbfi, B_scorer


def main():
    setup_style()
    sw_ov, payload_savings, scorer_gap, crit = load_constants()

    # Axis grids
    slack = np.linspace(0.5, 160, 360)
    spill = np.linspace(0.05, 2.0, 360)
    SS, PP = np.meshgrid(slack, spill)
    regime, B_sbfi, B_scorer = regime_field(SS, PP, sw_ov, payload_savings, scorer_gap)

    # Palette
    c_free     = "#EEF2EF"
    c_scorer   = "#C7A2D9"   # purple for scorer
    c_balanced = "#FFE4B5"   # warm sand for balanced
    c_sbfi     = "#E67E22"   # orange for SBFI
    cmap = ListedColormap([c_free, c_scorer, c_balanced, c_sbfi])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig = plt.figure(figsize=(3.33, 2.95))
    ax = fig.add_axes([0.16, 0.165, 0.78, 0.74])

    # Filled regime map
    ax.pcolormesh(SS, PP, regime, cmap=cmap, norm=norm, shading="auto",
                  alpha=0.95, zorder=0, rasterized=True)

    # Iso-contours of |benefit| to give visual texture
    B_max = np.maximum(B_sbfi, B_scorer)
    ctr = ax.contour(
        SS, PP, B_max,
        levels=[0.05, 0.10, 0.20, 0.35, 0.50],
        colors="#3B3B3B", linewidths=0.5, linestyles="dashed",
        alpha=0.55, zorder=2,
    )
    ax.clabel(ctr, ctr.levels, inline=True, fontsize=5.6,
              fmt="%.2f", inline_spacing=3)

    # Frontier between SBFI and scorer regions (contour diff=0)
    diff = B_sbfi - B_scorer
    ctr_f = ax.contour(
        SS, PP, diff,
        levels=[0.0], colors="#1A1A1A",
        linewidths=1.1, linestyles="solid", alpha=0.85, zorder=3,
    )

    # Regime labels placed in clearly-separated pockets of the plane.
    # The operating points below are slotted to avoid these pockets.
    ax.text(4.5, 1.80, "SBFI-limited",
            fontsize=7.4, color="#8C3000", fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec=c_sbfi, lw=0.6, alpha=0.94))
    ax.text(130, 0.12, "scorer-limited",
            fontsize=7.4, color="#5E2A7C", fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec=c_scorer, lw=0.6, alpha=0.94))
    ax.text(32, 1.05, "balanced",
            fontsize=6.6, color="#A16A12", fontweight="bold",
            style="italic",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="#B68A3B", lw=0.5, alpha=0.9))
    ax.text(150, 1.88, "free",
            fontsize=6.2, color="#6B7D6B", fontweight="bold",
            style="italic", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="#B0B0B0", lw=0.4, alpha=0.85))

    # ── Real operating points ─────────────────────────────────
    # Representative deployments placed on the slack axis from the
    # critical-path sweep; spill pressure taken from common LLM workloads.
    points = [
        # (slack_us, spill_ratio, label, marker, marker_color)
        (0.5,   1.55, "dense",    "o", "#CC2936"),
        (20.0,  1.30, "MoE",      "s", "#2C5F8D"),
        (40.0,  0.90, "spec",     "D", "#148A6F"),
        (80.0,  0.55, "tenant",   "^", "#614185"),
        (160.0, 0.25, "cached",   "v", "#6B7D3A"),
    ]
    marker_handles = []
    for x, y, lbl, mk, col in points:
        ax.plot(x, y, marker=mk, markersize=6.2, color=col,
                markeredgecolor="white", markeredgewidth=0.8, zorder=6)
        marker_handles.append(
            Line2D([0], [0], marker=mk, color="none",
                   markerfacecolor=col, markeredgecolor="white",
                   markeredgewidth=0.5, markersize=5,
                   label=lbl.replace("\n", " ")),
        )

    # Label each point with a short tag; offsets chosen to sit in empty
    # stripes between the regime labels.
    label_cfg = [
        (0.5,   1.55, "dense",    (12,  -4)),
        (20.0,  1.30, "MoE",      (10,  -4)),
        (40.0,  0.90, "spec",     (-8, -12)),
        (80.0,  0.55, "tenant",   (12,  12)),
        (160.0, 0.25, "cached",   (-8,  14)),
    ]
    for x, y, lbl, (dx, dy) in label_cfg:
        ha = "right" if dx < 0 else "left"
        va = "bottom" if dy > 0 else ("top" if dy < -2 else "center")
        ax.annotate(
            lbl, xy=(x, y), xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.8, color="#111", fontweight="bold",
            ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.16", fc="white",
                      ec="#888", lw=0.35, alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="#555",
                            lw=0.4, alpha=0.6,
                            shrinkA=1, shrinkB=2),
        )

    # Axes
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.5)
    ax.set_xticks([0, 1, 10, 20, 40, 80, 160])
    ax.set_xticklabels(["0", "1", "10", "20", "40", "80", "160"])
    ax.set_xlim(0.3, 175)
    ax.set_ylim(0.0, 2.0)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel(r"Decode slack  $s$  ($\mu$s per step)",
                  labelpad=1.5, fontweight="bold")
    ax.set_ylabel(r"Spill pressure  $\rho_{\mathrm{spill}}$"
                  r"  (fetched / capacity)",
                  labelpad=2, fontweight="bold")
    ax.set_title("Operating-regime map",
                 fontweight="bold", loc="left", pad=3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.tick_params(axis="both", colors="#222")
    # keep a clean frame around the map
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#222")

    # Compact legend for markers (below axes, single row)
    leg = ax.legend(
        handles=marker_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.23),
        frameon=False, ncol=5,
        handlelength=0.8, handletextpad=0.22,
        columnspacing=0.7, labelspacing=0.2,
        borderaxespad=0.0, borderpad=0.0,
    )
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
        txt.set_fontsize(5.8)

    out_pdf = OUT_DIR / "eval_regime_map.pdf"
    out_png = OUT_DIR / "eval_regime_map.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_png, dpi=340, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
