"""Sensitivity figure: HBM KV budget and CXL bandwidth.

Two stacked panels with identical rectangles to match the rest of the suite:
  (a) Relative throughput gain vs HBM KV budget (5%..30%), CXL BW = 32 GB/s
  (b) Relative throughput gain vs CXL bandwidth  (4..64 GB/s), HBM = 10%

Data: outputs/rebuttal/figX_bandwidth_saturation_data.json
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _figure_style import (
    setup_style, make_figure, style_axis, save,
    PALETTE,
)


ROOT = Path("d:/LLM/prose_v2")
OUT_DIR = ROOT / "figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SRC = ROOT / "outputs" / "rebuttal" / "figX_bandwidth_saturation_data.json"

DEFAULT_BW = 32
DEFAULT_HBM = 0.10


def dense_linear(x, y, n=160, log_x=False):
    from scipy.interpolate import PchipInterpolator
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if log_x:
        lx = np.log(xs)
        xq = np.exp(np.linspace(lx[0], lx[-1], n))
        f = PchipInterpolator(lx, ys)
        yq = f(np.log(xq))
    else:
        xq = np.linspace(xs[0], xs[-1], n)
        f = PchipInterpolator(xs, ys)
        yq = f(xq)
    return xq, yq


def main():
    setup_style()
    d = json.loads(SRC.read_text())
    bws = np.array(d["bandwidths_gb_s"], dtype=float)
    offs = np.array(d["offload_ratios"], dtype=float)
    gain = np.array(d["throughput_gain_pct"], dtype=float)

    bw_row = int(np.where(np.isclose(bws, DEFAULT_BW))[0][0])
    hbm = (1.0 - offs) * 100
    gain_hbm = gain[bw_row] * 100

    off_col = int(np.where(np.isclose(offs, 1 - DEFAULT_HBM))[0][0])
    gain_bw = gain[:, off_col] * 100

    fig, (ax_a, ax_b) = make_figure(shape="1x2")

    blue = PALETTE["prose"]
    accent = PALETTE["accent_r"]

    # ── Panel (a): HBM budget sweep ────────────────────────────────
    order_a = np.argsort(hbm)
    xs, ys = dense_linear(hbm[order_a], gain_hbm[order_a], n=160, log_x=False)
    ax_a.fill_between(xs, 0, ys, color=blue, alpha=0.10, zorder=1, linewidth=0.0)
    ax_a.plot(xs, ys, color=blue, lw=2.5, zorder=4)
    ax_a.plot(hbm[order_a], gain_hbm[order_a],
              color=blue, marker="o", markersize=5.4,
              markeredgecolor="white", markeredgewidth=0.55,
              lw=0.0, zorder=6)
    iD = int(np.where(np.isclose(hbm, DEFAULT_HBM * 100))[0][0])
    ax_a.scatter([hbm[iD]], [gain_hbm[iD]], s=60,
                 facecolor="white", edgecolor=accent, lw=1.5, zorder=7)
    # Callout placed in the empty band above the curve shoulder
    # (curve slopes down from 38% at HBM=5% to 25% at HBM=30%).
    ax_a.annotate(
        f"default\n10%,  +{gain_hbm[iD]:.0f}%",
        xy=(hbm[iD], gain_hbm[iD]),
        xytext=(22, 42), textcoords="data",
        fontsize=8.4, color=accent, fontweight="black",
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                  ec=accent, lw=0.62, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=accent,
                        lw=0.75, alpha=0.85,
                        shrinkA=2, shrinkB=3),
        zorder=9,
    )

    ax_a.set_xticks([5, 10, 15, 20, 25, 30])
    ax_a.set_xlim(4, 31)
    ax_a.set_ylim(0, 45)
    ax_a.set_yticks([0, 10, 20, 30, 40])
    style_axis(ax_a, title="(a) HBM budget  (CXL 32 GB/s)",
               xlabel="HBM KV budget (% of total KV)",
               ylabel="Throughput gain (%)")

    # ── Panel (b): CXL bandwidth sweep ─────────────────────────────
    xs2, ys2 = dense_linear(bws, gain_bw, n=160, log_x=True)
    ax_b.fill_between(xs2, 0, ys2, color=blue, alpha=0.10, zorder=1, linewidth=0.0)
    ax_b.plot(xs2, ys2, color=blue, lw=2.5, zorder=4)
    ax_b.plot(bws, gain_bw, color=blue, marker="s", markersize=5.4,
              markeredgecolor="white", markeredgewidth=0.55,
              lw=0.0, zorder=6)
    ax_b.axvspan(3.5, 8, color=accent, alpha=0.08, zorder=1)
    ax_b.text(5.7, 3.0, "collapse",
              fontsize=8.1, color=accent, fontweight="black",
              ha="center", va="bottom", style="italic")
    jD = int(np.where(np.isclose(bws, DEFAULT_BW))[0][0])
    ax_b.scatter([bws[jD]], [gain_bw[jD]], s=60,
                 facecolor="white", edgecolor=accent, lw=1.5, zorder=7)
    ax_b.annotate(
        f"default\n32 GB/s,  +{gain_bw[jD]:.0f}%",
        xy=(bws[jD], gain_bw[jD]),
        xytext=(-14, -22), textcoords="offset points",
        fontsize=8.4, color=accent, fontweight="black",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                  ec=accent, lw=0.62, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=accent,
                        lw=0.75, alpha=0.85,
                        shrinkA=0, shrinkB=3),
        zorder=9,
    )

    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(bws)
    ax_b.set_xticklabels([f"{int(b)}" for b in bws])
    ax_b.set_xlim(3.5, 72)
    ax_b.set_ylim(0, 45)
    ax_b.set_yticks([0, 10, 20, 30, 40])
    style_axis(ax_b, title="(b) CXL bandwidth  (HBM 10%)",
               xlabel="CXL bandwidth (GB/s, log)",
               ylabel="Throughput gain (%)")

    save(fig, OUT_DIR / "eval_sensitivity_budget_bw")
    print("Saved: eval_sensitivity_budget_bw")


if __name__ == "__main__":
    main()
