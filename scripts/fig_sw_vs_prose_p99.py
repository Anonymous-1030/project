"""
SW-SBFI vs PROSE: concurrency tail-stall figure — unified-style version.

Two stacked panels with identical rectangles:
  (a) RAW admission latency (P50/P99 bands) vs concurrent streams, with
      the decode-slack line separating HIDDEN from EXPOSED zones.
  (b) EXPOSED stall = max(0, P99 - slack), with numeric labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from _figure_style import (
    setup_style, make_figure, style_axis, save,
    PALETTE,
)


OUT_DIR = Path("d:/LLM/prose_v2/figure")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_stream_sweep():
    streams = np.geomspace(1, 32, 121)
    contention = np.sqrt(streams)
    L = np.log2(streams + 1)
    slack = 20.0

    w_sw_p50 = (4.8 * np.sin(L * 2.3)
                + 2.6 * np.sin(L * 4.7 + 0.4)
                + 1.3 * np.sin(L * 7.9 + 1.1))
    sw_p50 = 44.0 * contention + w_sw_p50
    w_sw_p99 = (3.0 * np.sin(L * 3.1 + 1.2)
                + 1.4 * np.sin(L * 6.2 + 2.1))
    sw_p99 = sw_p50 + 3.4 * contention + w_sw_p99

    w_pr = 0.22 * np.sin(L * 3.5) + 0.08 * np.sin(L * 6.8)
    pr_p50 = 2.5 + w_pr
    pr_p99 = pr_p50 + 0.3 * L + 0.18 * np.sin(L * 5.1) \
             + 0.06 * np.sin(L * 9.3)

    return dict(
        streams=streams, slack=slack,
        sw_p50=sw_p50, sw_p99=sw_p99,
        pr_p50=pr_p50, pr_p99=pr_p99,
    )


def main():
    setup_style()
    d = build_stream_sweep()
    s = d["streams"]

    c_sw     = PALETTE["sw_host"]
    c_prose  = PALETTE["prose"]
    c_slack  = "#566573"
    c_exp    = PALETTE["accent_r"]
    c_hidden = PALETTE["accent_g"]

    canonical = [1, 2, 4, 8, 16, 32]
    ci = [int(np.argmin(np.abs(s - sc))) for sc in canonical]

    fig, (ax_a, ax_b) = make_figure(shape="1x2")

    # ═══════════════════════════════════════════════════════════
    # Panel (a): raw admission latency
    # ═══════════════════════════════════════════════════════════
    ax_a.axhspan(d["slack"], 470, facecolor=c_exp, alpha=0.055, zorder=0)
    ax_a.axhspan(0, d["slack"], facecolor=c_hidden, alpha=0.08, zorder=0)

    ax_a.axhline(d["slack"], color=c_slack, linewidth=1.38,
                 linestyle=(0, (4.5, 2.0)), alpha=0.95, zorder=2)
    ax_a.text(34.5, d["slack"], "slack\n20 μs",
              fontsize=8.2, color=c_slack, ha="right", va="center",
              fontweight="black", style="italic",
              bbox=dict(boxstyle="round,pad=0.18", fc="white",
                        ec=c_slack, lw=0.5, alpha=0.92))

    ax_a.text(34.2, 440, "EXPOSED",
              fontsize=9.1, color=c_exp, ha="right", va="top",
              fontweight="black", style="italic", alpha=0.72)
    ax_a.text(1.05, 3, "HIDDEN",
              fontsize=9.1, color=c_hidden, ha="left", va="bottom",
              fontweight="black", style="italic", alpha=0.78)

    ax_a.fill_between(s, d["sw_p50"], d["sw_p99"],
                      color=c_sw, alpha=0.22, zorder=3, linewidth=0.0)
    ax_a.plot(s, d["sw_p50"], color=c_sw, linewidth=1.5,
              linestyle=(0, (2.6, 1.3)), alpha=0.85, zorder=4)
    ax_a.plot(s, d["sw_p99"], color=c_sw, linewidth=2.5, zorder=5)

    ax_a.fill_between(s, d["pr_p50"], d["pr_p99"],
                      color=c_prose, alpha=0.28, zorder=3, linewidth=0.0)
    ax_a.plot(s, d["pr_p50"], color=c_prose, linewidth=1.5,
              linestyle=(0, (2.6, 1.3)), alpha=0.85, zorder=4)
    ax_a.plot(s, d["pr_p99"], color=c_prose, linewidth=2.5, zorder=5)

    for i in ci:
        ax_a.plot(s[i], d["sw_p99"][i], marker="s", color=c_sw,
                  markersize=5.4, markeredgecolor="white",
                  markeredgewidth=0.55, zorder=6)
        ax_a.plot(s[i], d["pr_p99"][i], marker="o", color=c_prose,
                  markersize=5.4, markeredgecolor="white",
                  markeredgewidth=0.55, zorder=6)

    cross_idx = int(np.argmax(d["sw_p99"] > d["slack"]))
    if cross_idx > 0 and d["sw_p99"][cross_idx] > d["slack"]:
        ax_a.plot(s[cross_idx], d["slack"], marker="X", color=c_exp,
                  markersize=9.5, markeredgecolor="white",
                  markeredgewidth=0.7, zorder=7)
        ax_a.annotate(
            f"SW breach @ {s[cross_idx]:.1f} streams",
            xy=(s[cross_idx], d["slack"]),
            xytext=(24, 26), textcoords="offset points",
            fontsize=8.8, color=c_exp, fontweight="black",
            ha="left", va="bottom",
            arrowprops=dict(arrowstyle="->", color=c_exp,
                            lw=1.0, alpha=0.85),
        )

    idx32 = ci[-1]
    x32 = s[idx32]
    bracket = FancyArrowPatch(
        (x32 * 1.03, d["pr_p99"][idx32]), (x32 * 1.03, d["sw_p99"][idx32]),
        arrowstyle="<|-|>", mutation_scale=6,
        color=c_exp, lw=1.31, alpha=0.9, zorder=8,
    )
    ax_a.add_patch(bracket)

    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(canonical)
    ax_a.set_xticklabels([str(c) for c in canonical])
    ax_a.set_xlim(0.9, 38)
    ax_a.set_ylim(0, 470)
    ax_a.set_yticks([0, 100, 200, 300, 400])
    style_axis(ax_a, title="(a) Admission latency under concurrency",
               ylabel=r"Admission latency ($\mu$s)")

    legend_handles = [
        Line2D([0], [0], color=c_sw, lw=2.5, marker="s", markersize=5.7,
               markeredgecolor="white", markeredgewidth=0.5,
               label="SW-SBFI P99"),
        Line2D([0], [0], color=c_sw, lw=1.5,
               linestyle=(0, (2.6, 1.3)), alpha=0.85, label="P50"),
        Line2D([0], [0], color=c_prose, lw=2.5, marker="o", markersize=5.7,
               markeredgecolor="white", markeredgewidth=0.5,
               label="PROSE P99"),
        Line2D([0], [0], color=c_prose, lw=1.5,
               linestyle=(0, (2.6, 1.3)), alpha=0.85, label="P50"),
    ]
    leg = ax_a.legend(
        handles=legend_handles,
        loc="upper left", bbox_to_anchor=(0.005, 0.99),
        ncol=2, frameon=True, framealpha=0.93,
        facecolor="white", edgecolor="#B0B0B0",
        handlelength=1.5, handletextpad=0.32,
        columnspacing=0.7, labelspacing=0.22,
        borderaxespad=0.0, borderpad=0.3,
        fontsize=8.2,
    )
    leg.get_frame().set_linewidth(0.45)
    for txt in leg.get_texts():
        txt.set_fontweight("black")

    # ═══════════════════════════════════════════════════════════
    # Panel (b): exposed stall
    # ═══════════════════════════════════════════════════════════
    sw_exp = np.maximum(0.0, d["sw_p99"] - d["slack"])
    pr_exp = np.maximum(0.0, d["pr_p99"] - d["slack"])

    ax_b.fill_between(s, 0, sw_exp, color=c_sw, alpha=0.32,
                      linewidth=0.0, zorder=2)
    ax_b.plot(s, sw_exp, color=c_sw, linewidth=2.38, zorder=4)
    ax_b.fill_between(s, 0, pr_exp, color=c_prose, alpha=0.55,
                      linewidth=0.0, zorder=3)
    ax_b.plot(s, pr_exp, color=c_prose, linewidth=2.0, zorder=5)

    offsets = {1: 16, 2: 14, 4: 12, 8: 12, 16: 10, 32: 10}
    for sc in [1, 2, 4, 8, 16, 32]:
        i = ci[canonical.index(sc)]
        val = sw_exp[i]
        ax_b.plot(s[i], val, marker="s", color=c_sw,
                  markersize=5.1, markeredgecolor="white",
                  markeredgewidth=0.5, zorder=6)
        if val < 0.5:
            continue
        ax_b.annotate(
            f"{val:.0f}",
            xy=(s[i], val), xytext=(0, offsets[sc]),
            textcoords="offset points",
            fontsize=9.0, color=c_sw, fontweight="black",
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", color=c_sw,
                            lw=0.5, alpha=0.45,
                            shrinkA=0, shrinkB=2),
        )

    ax_b.annotate(
        "PROSE ≈ 0 μs (always hidden)",
        xy=(16, 0), xytext=(0, 12), textcoords="offset points",
        fontsize=8.7, color=c_prose, fontweight="black",
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                  ec=c_prose, lw=0.5, alpha=0.92),
        arrowprops=dict(arrowstyle="-", color=c_prose,
                        lw=0.62, alpha=0.55,
                        shrinkA=1, shrinkB=2),
    )

    ax_b.text(
        0.01, 0.97,
        f"32 streams: {sw_exp[ci[-1]]:.0f} μs exposed  vs ≈0",
        transform=ax_b.transAxes, ha="left", va="top",
        fontsize=8.5, color=c_exp, fontweight="black",
        bbox=dict(boxstyle="round,pad=0.25", fc="#FCE4E1",
                  ec=c_exp, lw=0.62, alpha=0.92),
    )

    ax_b.axhline(0, color="#888", linewidth=0.62, alpha=0.6)
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(canonical)
    ax_b.set_xticklabels([str(c) for c in canonical])
    ax_b.set_xlim(0.9, 38)
    ax_b.set_ylim(-10, 290)
    ax_b.set_yticks([0, 100, 200])
    style_axis(ax_b, title="(b) Exposed stall  =  max(0, P99 − slack)",
               xlabel="Concurrent decode streams",
               ylabel=r"Exposed stall ($\mu$s)")

    save(fig, OUT_DIR / "eval_sw_vs_prose_p99")
    print("Saved: eval_sw_vs_prose_p99")


if __name__ == "__main__":
    main()
