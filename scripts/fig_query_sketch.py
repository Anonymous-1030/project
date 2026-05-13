"""
Query-sketch sensitivity figure — 2x2 landscape version.

Four panels:
  (a) Sketch SIZE sweep (recovery + failure-rate twin axis).
  (b) FRESHNESS sweep (fresh vs stale, helpful/harmful zones).
  (c) QUANTIZATION sweep (4-16b plateau, slight 32b drop).
  (d) BUDGET efficiency — joint view of size × quantization expressed as
      total bits per sketch (size_B × 8 for 8-bit; size_B × bits/8 for
      other quantizations at the 32-B reference size). Shows where the
      "most Recovery@K per bit" operating point lives.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).parent))
from _figure_style import (
    setup_style, make_figure, style_axis, style_twin_axis, save,
    PALETTE,
)


ROOT = Path("d:/LLM/prose_v2")
RESULTS = ROOT / "results" / "reviewer_response_v2_results.json"
OUT_DIR = ROOT / "figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dense_interp(x, y, n=240, log_x=True):
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


def load_data():
    raw = json.load(open(RESULTS))
    sec = raw["fix2_query_sketch"]
    baseline = sec["baseline"]["recovery"]

    size_keys = [k for k in sec if k.startswith("size=")]
    size_rows = sorted(
        [(sec[k]["sketch_size_bytes"], sec[k]["recovery"],
          sec[k].get("failure_rate", 0.0)) for k in size_keys],
        key=lambda r: r[0],
    )
    size_rows = [r for r in size_rows if r[0] > 0]
    sz = dict(
        bytes=np.array([r[0] for r in size_rows]),
        recovery=np.array([r[1] for r in size_rows]),
        failure_rate=np.array([r[2] for r in size_rows]),
    )

    stale_keys = [k for k in sec if k.startswith("stale=")]
    stale_rows = sorted(
        [(sec[k]["staleness_steps"], sec[k]["recovery"])
         for k in stale_keys], key=lambda r: r[0],
    )
    fr = dict(
        stale=np.array([r[0] for r in stale_rows]),
        recovery=np.array([r[1] for r in stale_rows]),
    )

    quant_keys = [k for k in sec if k.startswith("quant=")]
    quant_rows = sorted(
        [(sec[k]["quantization_bits"], sec[k]["recovery"])
         for k in quant_keys], key=lambda r: r[0],
    )
    qz = dict(
        bits=np.array([r[0] for r in quant_rows]),
        recovery=np.array([r[1] for r in quant_rows]),
    )
    return baseline, sz, fr, qz


def main():
    setup_style()
    baseline, sz, fr, qz = load_data()

    c_rec     = "#1F4E79"
    c_fail    = PALETTE["accent_r"]
    c_fresh   = PALETTE["oracle"]
    c_stale   = "#B03030"
    c_quant   = PALETTE["accent_v"]
    c_baseln  = PALETTE["baseline_g"]
    c_shade_ok = PALETTE["shade_ok"]
    c_shade_bd = PALETTE["shade_bad"]

    fig, (ax_a, ax_b, ax_c, ax_d) = make_figure(shape="2x2")

    # ═════════════════════════════════════════════════════════════
    # Panel (a): sketch-size sweep
    # ═════════════════════════════════════════════════════════════
    ax_a_r = ax_a.twinx()
    ax_a_r.spines["top"].set_visible(False)

    ax_a.axhline(baseline, color=c_baseln, lw=1.19,
                 linestyle=(0, (3.0, 1.6)), alpha=0.95, zorder=2)
    ax_a.text(62.5, baseline - 0.012, f"no-sketch  {baseline:.3f}",
              fontsize=8.7, color=c_baseln, ha="right", va="top",
              fontweight="black", style="italic",
              bbox=dict(boxstyle="round,pad=0.18", fc="white",
                        ec=c_baseln, lw=0.44, alpha=0.9))

    xs, ys = dense_interp(sz["bytes"], sz["recovery"])
    l_rec = ax_a.plot(xs, ys, color=c_rec, lw=2.62, zorder=5,
                      label="Recovery@K")[0]
    ax_a.plot(sz["bytes"], sz["recovery"], "o", color=c_rec,
              markersize=5.9, markeredgecolor="white",
              markeredgewidth=0.55, zorder=6)

    peak_idx = int(np.argmax(sz["recovery"]))
    ax_a.plot(sz["bytes"][peak_idx], sz["recovery"][peak_idx],
              marker="*", color=c_rec, markersize=17.6,
              markeredgecolor="white", markeredgewidth=0.8, zorder=7)
    ax_a.annotate(
        f"peak {sz['recovery'][peak_idx]:.3f} @ {sz['bytes'][peak_idx]} B",
        xy=(sz["bytes"][peak_idx], sz["recovery"][peak_idx]),
        xytext=(-10, 12), textcoords="offset points",
        fontsize=9.0, color=c_rec, fontweight="black",
        ha="right", va="bottom",
        arrowprops=dict(arrowstyle="->", color=c_rec,
                        lw=0.75, alpha=0.8),
    )

    xs2, ys2 = dense_interp(sz["bytes"], sz["failure_rate"])
    l_fail = ax_a_r.plot(xs2, ys2, color=c_fail, lw=2.06,
                         linestyle=(0, (4.0, 1.5)),
                         zorder=4, label="Failure rate")[0]
    ax_a_r.plot(sz["bytes"], sz["failure_rate"], "s", color=c_fail,
                markersize=5.3, markeredgecolor="white",
                markeredgewidth=0.5, zorder=5)

    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks([4, 8, 16, 32, 64])
    ax_a.set_xticklabels(["4", "8", "16", "32", "64"])
    ax_a.set_xlim(3.5, 75)
    ax_a.set_ylim(0.60, 0.99)
    ax_a.set_yticks([0.6, 0.7, 0.8, 0.9])
    style_axis(ax_a, title="(a) Sketch size",
               xlabel="Sketch size (B)",
               ylabel="Recovery@K", ylabel_color=c_rec)
    ax_a.tick_params(axis="y", colors=c_rec)

    ax_a_r.set_ylim(0, 0.6)
    ax_a_r.set_yticks([0.0, 0.2, 0.4, 0.6])
    style_twin_axis(ax_a_r, ylabel="Failure rate", ylabel_color=c_fail)

    leg_a = ax_a.legend(
        handles=[l_rec, l_fail],
        loc="upper right", bbox_to_anchor=(0.995, 0.30),
        frameon=True, framealpha=0.92,
        facecolor="white", edgecolor="#B0B0B0",
        ncol=1, fontsize=8.7,
        handlelength=1.7, handletextpad=0.32,
        labelspacing=0.22, borderpad=0.3,
    )
    leg_a.get_frame().set_linewidth(0.45)
    for txt in leg_a.get_texts():
        txt.set_fontweight("black")

    # ═════════════════════════════════════════════════════════════
    # Panel (b): freshness
    # ═════════════════════════════════════════════════════════════
    ax_b.axhspan(baseline, 1.0, facecolor=c_shade_ok,
                 alpha=0.55, zorder=0)
    ax_b.axhspan(0.0, baseline, facecolor=c_shade_bd,
                 alpha=0.55, zorder=0)

    ax_b.axhline(baseline, color=c_baseln, lw=1.31,
                 linestyle=(0, (3.0, 1.6)), alpha=0.95, zorder=2)

    fresh_x, fresh_y = fr["stale"][0], fr["recovery"][0]
    stale_xs, stale_ys = fr["stale"][1:], fr["recovery"][1:]

    ax_b.plot(
        [fresh_x, stale_xs[0]], [fresh_y, stale_ys[0]],
        color=c_stale, lw=1.88, linestyle=(0, (2.2, 1.4)),
        alpha=0.8, zorder=3,
    )
    ax_b.plot(stale_xs, stale_ys, color=c_stale, lw=2.62, zorder=5)
    ax_b.plot(stale_xs, stale_ys, "s", color=c_stale,
              markersize=5.4, markeredgecolor="white",
              markeredgewidth=0.5, zorder=6)
    ax_b.plot(fresh_x, fresh_y, marker="*", color=c_fresh,
              markersize=20.2, markeredgecolor="white",
              markeredgewidth=0.9, zorder=7)

    ax_b.text(0.985, 0.78, "helpful zone",
              transform=ax_b.transAxes, ha="right", va="top",
              fontsize=9.0, color=c_fresh, fontweight="black",
              style="italic", alpha=0.85)
    ax_b.text(0.985, 0.05, "harmful zone",
              transform=ax_b.transAxes, ha="right", va="bottom",
              fontsize=9.0, color=c_stale, fontweight="black",
              style="italic", alpha=0.75)

    ax_b.annotate(
        f"{fresh_y:.3f}",
        xy=(fresh_x, fresh_y), xytext=(0, 8),
        textcoords="offset points",
        fontsize=9.3, color=c_fresh, fontweight="black",
        ha="center", va="bottom",
    )
    delta_cliff = fresh_y - stale_ys[0]
    ax_b.annotate(
        f"1-step stale\n−{delta_cliff*100:.1f} pp",
        xy=(stale_xs[0], stale_ys[0]),
        xytext=(4.2, 0.58),
        fontsize=9.3, color=c_stale, fontweight="black",
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                  ec=c_stale, lw=0.62, alpha=0.93),
        arrowprops=dict(arrowstyle="->", color=c_stale,
                        lw=0.81, alpha=0.85),
    )
    ax_b.annotate(
        f"plateau {stale_ys[-1]:.3f}",
        xy=(stale_xs[-1], stale_ys[-1]),
        xytext=(-3, 11), textcoords="offset points",
        fontsize=9.0, color=c_stale, fontweight="black",
        ha="right", va="bottom",
    )
    ax_b.text(15.5, baseline + 0.01,
              f"no-sketch  {baseline:.3f}",
              fontsize=8.7, color=c_baseln, ha="right", va="bottom",
              fontweight="black", style="italic",
              bbox=dict(boxstyle="round,pad=0.18", fc="white",
                        ec=c_baseln, lw=0.44, alpha=0.9))

    handles_b = [
        Line2D([0], [0], marker="*", color="none",
               markerfacecolor=c_fresh,
               markeredgecolor="white", markeredgewidth=0.5,
               markersize=13.5, label="Fresh"),
        Line2D([0], [0], marker="s", color=c_stale, lw=2.0,
               markerfacecolor=c_stale, markeredgecolor="white",
               markeredgewidth=0.5, markersize=5.4,
               label="Stale"),
        Line2D([0], [0], color=c_baseln, lw=1.25,
               linestyle=(0, (3.0, 1.6)), label="No-sketch"),
    ]
    leg_b = ax_b.legend(
        handles=handles_b,
        loc="upper left", bbox_to_anchor=(0.005, 0.98),
        frameon=True, framealpha=0.92,
        facecolor="white", edgecolor="#B0B0B0",
        ncol=3, fontsize=8.4,
        handlelength=1.4, handletextpad=0.3,
        columnspacing=0.7, labelspacing=0.22,
        borderpad=0.3, borderaxespad=0.0,
    )
    leg_b.get_frame().set_linewidth(0.45)
    for txt in leg_b.get_texts():
        txt.set_fontweight("black")

    ax_b.set_xlim(-1.2, 17)
    ax_b.set_ylim(0.38, 0.90)
    ax_b.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax_b.set_xticks([0, 1, 2, 4, 8, 16])
    ax_b.set_xticklabels(["0", "1", "2", "4", "8", "16"])
    style_axis(ax_b, title="(b) Freshness",
               xlabel="Staleness (decode steps)",
               ylabel="Recovery@K")

    # ═════════════════════════════════════════════════════════════
    # Panel (c): quantization
    # ═════════════════════════════════════════════════════════════
    ax_c.axvspan(3.5, 18, color="#EEE8F5", alpha=0.7, zorder=0)

    xs, ys = dense_interp(qz["bits"], qz["recovery"])
    ax_c.plot(xs, ys, color=c_quant, lw=2.75, zorder=5,
              label="Recovery@K")
    ax_c.plot(qz["bits"], qz["recovery"], "D", color=c_quant,
              markersize=5.9, markeredgecolor="white",
              markeredgewidth=0.55, zorder=6)

    label_offsets = {4: (0, 11), 8: (0, -14), 16: (0, 11), 32: (0, -14)}
    for x, y in zip(qz["bits"], qz["recovery"]):
        dx, dy = label_offsets.get(int(x), (0, 11))
        va = "bottom" if dy > 0 else "top"
        ax_c.annotate(
            f"{y:.3f}",
            xy=(x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=8.8, color=c_quant, fontweight="black",
            ha="center", va=va,
        )

    ax_c.axhline(baseline, color=c_baseln, lw=1.19,
                 linestyle=(0, (3.0, 1.6)), alpha=0.9, zorder=2)
    ax_c.text(32, baseline - 0.005, f"no-sketch  {baseline:.3f}",
              fontsize=8.7, color=c_baseln, ha="right", va="top",
              fontweight="black", style="italic",
              bbox=dict(boxstyle="round,pad=0.18", fc="white",
                        ec=c_baseln, lw=0.44, alpha=0.9))

    ax_c.text(
        8, 0.905, "plateau (4–16 bit)",
        fontsize=9.3, color=c_quant, ha="center", va="center",
        fontweight="black", style="italic",
        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                  ec=c_quant, lw=0.62, alpha=0.93),
    )

    ax_c.set_xscale("log", base=2)
    ax_c.set_xticks([4, 8, 16, 32])
    ax_c.set_xticklabels(["4", "8", "16", "32"])
    ax_c.set_xlim(3.5, 36)
    ax_c.set_ylim(0.62, 0.92)
    ax_c.set_yticks([0.65, 0.75, 0.85])
    style_axis(ax_c, title="(c) Quantization",
               xlabel="Quantization (bits / element)",
               ylabel="Recovery@K")

    # ═════════════════════════════════════════════════════════════
    # Panel (d): Recovery@K per total BIT budget
    # ═════════════════════════════════════════════════════════════
    # We approximate "total bits" for each sweep point as:
    #   size-sweep:   8 * bytes     (8-bit reference quantization)
    #   quant-sweep:  bits * 32     (32-B reference size)
    # The joint scatter highlights where peak recovery lives on a
    # bit-budget axis — the 32 B x 8-bit point sits at the sweet spot.
    bit_size  = 8 * sz["bytes"]
    bit_quant = qz["bits"] * 32

    ax_d.plot(bit_size, sz["recovery"], color=c_rec, lw=2.25,
              marker="o", markersize=5.7, markeredgecolor="white",
              markeredgewidth=0.55, zorder=5, label="Size sweep (×8b)")
    ax_d.plot(bit_quant, qz["recovery"], color=c_quant, lw=2.25,
              marker="D", markersize=5.4, markeredgecolor="white",
              markeredgewidth=0.55, linestyle=(0, (3.0, 1.5)),
              zorder=4, label="Quant sweep (×32B)")

    # Sweet-spot marker: the 32 B × 8-bit anchor
    anchor_bits = 32 * 8
    anchor_rec = sz["recovery"][np.where(sz["bytes"] == 32)[0][0]]
    ax_d.plot(anchor_bits, anchor_rec, marker="*", color="#111",
              markersize=18.9, markeredgecolor="white",
              markeredgewidth=0.9, zorder=8)
    ax_d.annotate(
        "anchor: 32 B × 8 b",
        xy=(anchor_bits, anchor_rec),
        xytext=(22, 8), textcoords="offset points",
        fontsize=9.0, color="#111", fontweight="black",
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                  ec="#111", lw=0.62, alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#111",
                        lw=0.69, alpha=0.7),
    )

    ax_d.axhline(baseline, color=c_baseln, lw=1.19,
                 linestyle=(0, (3.0, 1.6)), alpha=0.9, zorder=2)

    ax_d.set_xscale("log", base=2)
    ax_d.set_xticks([64, 128, 256, 512, 1024])
    ax_d.set_xticklabels(["64", "128", "256", "512", "1024"])
    ax_d.set_xlim(28, 1200)
    ax_d.set_ylim(0.60, 0.99)
    ax_d.set_yticks([0.6, 0.7, 0.8, 0.9])
    style_axis(ax_d, title="(d) Recovery per total bit-budget",
               xlabel="Total bits per sketch (log)",
               ylabel="Recovery@K")

    leg_d = ax_d.legend(
        loc="lower right", bbox_to_anchor=(0.995, 0.03),
        frameon=True, framealpha=0.92,
        facecolor="white", edgecolor="#B0B0B0",
        ncol=1, fontsize=8.4,
        handlelength=1.8, handletextpad=0.32,
        labelspacing=0.22, borderpad=0.3,
    )
    leg_d.get_frame().set_linewidth(0.45)
    for txt in leg_d.get_texts():
        txt.set_fontweight("black")

    save(fig, OUT_DIR / "eval_query_sketch")
    print("Saved: eval_query_sketch")


if __name__ == "__main__":
    main()
