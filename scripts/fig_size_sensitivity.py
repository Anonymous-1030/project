"""
Chunk-size and summary-size sensitivity figure — unified-style version.

Two stacked panels with identical rectangles:
  (a) Chunk size sweep: recovery flat vs queue load rho blows up.
  (b) Summary size sweep: fixed vs calibrated scorer, sweet-spot near 64 B.
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

sys.path.insert(0, str(Path(__file__).parent))
from _figure_style import (
    setup_style, make_figure, style_axis, style_twin_axis, save,
    PALETTE,
)


ROOT = Path("d:/LLM/prose_v2")
RESULTS = ROOT / "results" / "reviewer_response_v2_results.json"
OUT_DIR = ROOT / "figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    with open(RESULTS, "r") as f:
        raw = json.load(f)
    section = raw["fix3_summary_size"]

    chunk_keys = sorted(
        [k for k in section if k.startswith("chunk=")],
        key=lambda k: section[k]["chunk_size_bytes"],
    )
    chunk = dict(
        sizes=np.array([section[k]["chunk_size_bytes"] for k in chunk_keys]) / 1024,
        recovery=np.array([section[k]["recovery"] for k in chunk_keys]),
        rho=np.array([section[k]["queue_rho"] for k in chunk_keys]),
        rho_ratio=np.array([section[k]["rho_ratio_vs_16KB"] for k in chunk_keys]),
    )

    fixed_keys = sorted(
        [k for k in section if k.endswith("_fixed_thresh")],
        key=lambda k: section[k]["summary_size_bytes"],
    )
    fixed = dict(
        sizes=np.array([section[k]["summary_size_bytes"] for k in fixed_keys]),
        recovery=np.array([section[k]["recovery"] for k in fixed_keys]),
        noise=np.array([section[k]["noise_std"] for k in fixed_keys]),
        meta_bytes=np.array([section[k]["total_metadata_bytes"] for k in fixed_keys]),
    )
    cal_keys = sorted(
        [k for k in section if k.endswith("_calibrated")],
        key=lambda k: section[k]["summary_size_bytes"],
    )
    calibrated = dict(
        sizes=np.array([section[k]["summary_size_bytes"] for k in cal_keys]),
        recovery=np.array([section[k]["recovery"] for k in cal_keys]),
        noise=np.array([section[k]["calibrated_noise_std"] for k in cal_keys]),
    )
    return chunk, fixed, calibrated


def dense_interp(x, y, n=240, kind="pchip"):
    from scipy.interpolate import PchipInterpolator, CubicSpline
    x_sorted_idx = np.argsort(x)
    xs = np.asarray(x)[x_sorted_idx]
    ys = np.asarray(y)[x_sorted_idx]
    lx = np.log(xs)
    xq = np.exp(np.linspace(lx[0], lx[-1], n))
    if kind == "cubic":
        f = CubicSpline(lx, ys, bc_type="natural")
    else:
        f = PchipInterpolator(lx, ys)
    yq = f(np.log(xq))
    return xq, yq


def main():
    setup_style()
    chunk, fixed, cal = load_data()

    c_rec    = "#1F4E79"
    c_rho    = "#D35400"
    c_noise  = PALETTE["accent_v"]
    c_fixed  = PALETTE["fts"]
    c_cal    = PALETTE["oracle"]
    c_budget = PALETTE["baseline_g"]
    c_shade  = PALETTE["shade_bad"]

    fig, (ax_a, ax_b) = make_figure(shape="1x2")

    # ═══════════════════════════════════════════════════════════
    # Panel (a): chunk-size sweep
    # ═══════════════════════════════════════════════════════════
    ax_a_r = ax_a.twinx()
    ax_a_r.spines["top"].set_visible(False)

    xs_rec, ys_rec = dense_interp(chunk["sizes"], chunk["recovery"],
                                  n=240, kind="pchip")
    xs_rho, ys_rho = dense_interp(chunk["sizes"], chunk["rho"],
                                  n=240, kind="pchip")

    l_rec = ax_a.plot(xs_rec, ys_rec, color=c_rec, lw=2.5,
                      zorder=5, label="Recovery@K")[0]
    ax_a.plot(chunk["sizes"], chunk["recovery"], "o", color=c_rec,
              markersize=5.4, markeredgecolor="white",
              markeredgewidth=0.55, zorder=6)

    ax_a_r.fill_between(xs_rho, 0, ys_rho, color=c_rho, alpha=0.14,
                        linewidth=0.0, zorder=1)
    l_rho = ax_a_r.plot(xs_rho, ys_rho, color=c_rho, lw=2.5,
                        linestyle=(0, (5.0, 1.6)),
                        zorder=5, label=r"Queue load $\rho$")[0]
    ax_a_r.plot(chunk["sizes"], chunk["rho"], "s", color=c_rho,
                markersize=5.4, markeredgecolor="white",
                markeredgewidth=0.55, zorder=6)

    for x, y in zip(chunk["sizes"], chunk["rho"]):
        ax_a_r.annotate(
            f"{y:.2f}",
            xy=(x, y), xytext=(0, 8), textcoords="offset points",
            fontsize=8.4, color=c_rho, fontweight="black",
            ha="center", va="bottom",
        )

    baseline_rec = chunk["recovery"][chunk["sizes"] == 64][0]
    ax_a.annotate(
        f"flat at {baseline_rec:.3f}",
        xy=(64, baseline_rec), xytext=(0, 12),
        textcoords="offset points",
        fontsize=8.7, color=c_rec, fontweight="black", ha="center",
        arrowprops=dict(arrowstyle="-", color=c_rec, lw=0.62, alpha=0.7),
    )

    ax_a.axvline(64, color=c_budget, lw=1.0,
                 linestyle=(0, (2.2, 1.8)), alpha=0.85, zorder=2)
    ax_a.text(64, 1.015, "64 KB",
              fontsize=8.2, color=c_budget, ha="center", va="bottom",
              fontweight="black", style="italic",
              transform=ax_a.get_xaxis_transform(),
              bbox=dict(boxstyle="round,pad=0.16", fc="white",
                        ec=c_budget, lw=0.44, alpha=0.92))

    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks([16, 32, 64, 128, 256])
    ax_a.set_xticklabels(["16", "32", "64", "128", "256"])
    ax_a.set_xlim(13, 310)
    ax_a.set_ylim(0.70, 1.0)
    ax_a.set_yticks([0.70, 0.80, 0.90, 1.00])
    style_axis(ax_a, title="(a) Chunk size  →  queue load",
               xlabel="Chunk size (KB)",
               ylabel="Recovery@K", ylabel_color=c_rec)
    ax_a.tick_params(axis="y", colors=c_rec)

    ax_a_r.set_ylim(0, 0.75)
    ax_a_r.set_yticks([0, 0.2, 0.4, 0.6])
    style_twin_axis(ax_a_r, ylabel=r"Queue load  $\rho$",
                    ylabel_color=c_rho)

    rho_ratio = chunk["rho_ratio"][-1]
    ax_a_r.text(
        0.015, 0.96,
        f"16→256 KB:  {rho_ratio:.1f}× load",
        transform=ax_a_r.transAxes, ha="left", va="top",
        fontsize=8.5, color=c_rho, fontweight="black",
        bbox=dict(boxstyle="round,pad=0.22", fc="#FDECEC",
                  ec=c_rho, lw=0.56, alpha=0.92),
    )

    leg_a = ax_a.legend(
        handles=[l_rec, l_rho],
        loc="lower right", bbox_to_anchor=(0.995, 0.02),
        frameon=True, framealpha=0.92,
        facecolor="white", edgecolor="#B0B0B0",
        ncol=1, fontsize=8.2,
        handlelength=1.7, handletextpad=0.32,
        labelspacing=0.22, borderpad=0.3,
    )
    leg_a.get_frame().set_linewidth(0.45)
    for txt in leg_a.get_texts():
        txt.set_fontweight("black")

    # ═══════════════════════════════════════════════════════════
    # Panel (b): summary-size sweep
    # ═══════════════════════════════════════════════════════════
    ax_b_r = ax_b.twinx()
    ax_b_r.spines["top"].set_visible(False)

    xs_f, ys_f = dense_interp(fixed["sizes"], fixed["recovery"],
                              n=240, kind="cubic")
    xs_c, ys_c = dense_interp(cal["sizes"], cal["recovery"],
                              n=240, kind="cubic")
    xs_m, ys_m = dense_interp(fixed["sizes"], fixed["meta_bytes"] / 1024,
                              n=240, kind="pchip")

    ax_b_r.fill_between(xs_m, 0, ys_m, color=c_noise,
                        alpha=0.09, linewidth=0.0, zorder=0)
    l_meta = ax_b_r.plot(xs_m, ys_m, color=c_noise, lw=1.75,
                         linestyle=(0, (5.0, 1.6)),
                         zorder=4, label="Metadata (KB)")[0]

    l_fix = ax_b.plot(xs_f, ys_f, color=c_fixed, lw=2.62,
                      zorder=5, label="Fixed-threshold")[0]
    ax_b.plot(fixed["sizes"], fixed["recovery"], "o", color=c_fixed,
              markersize=5.4, markeredgecolor="white",
              markeredgewidth=0.55, zorder=7)

    l_cal = ax_b.plot(xs_c, ys_c, color=c_cal, lw=2.12,
                      zorder=5, label="Calibrated")[0]
    ax_b.plot(cal["sizes"], cal["recovery"], "D", color=c_cal,
              markersize=4.9, markeredgecolor="white",
              markeredgewidth=0.5, zorder=7)

    ax_b.axvspan(48, 96, color=c_shade, alpha=0.55, zorder=1)

    peak_idx = int(np.argmax(fixed["recovery"]))
    peak_x = fixed["sizes"][peak_idx]
    peak_y = fixed["recovery"][peak_idx]
    ax_b.plot(peak_x, peak_y, marker="*", color=c_fixed,
              markersize=14.9, markeredgecolor="white",
              markeredgewidth=0.7, zorder=8)
    ax_b.annotate(
        f"peak {peak_y:.3f} @ {peak_x} B",
        xy=(peak_x, peak_y), xytext=(14, 6),
        textcoords="offset points",
        fontsize=8.5, color=c_fixed, fontweight="black",
        ha="left", va="bottom",
        arrowprops=dict(arrowstyle="->", color=c_fixed,
                        lw=0.69, alpha=0.8),
    )
    drop_y = fixed["recovery"][-1]
    ax_b.annotate(
        f"drown {drop_y:.3f}",
        xy=(fixed["sizes"][-1], drop_y),
        xytext=(-3, 14), textcoords="offset points",
        fontsize=8.5, color=c_fixed, fontweight="black",
        ha="right", va="bottom",
        arrowprops=dict(arrowstyle="->", color=c_fixed,
                        lw=0.69, alpha=0.8),
    )

    ax_b.axvline(64, color=c_budget, lw=1.0,
                 linestyle=(0, (2.2, 1.8)), alpha=0.85, zorder=3)
    ax_b.text(64, 1.015, "64 B",
              fontsize=8.2, color=c_budget, ha="center", va="bottom",
              fontweight="black", style="italic",
              transform=ax_b.get_xaxis_transform(),
              bbox=dict(boxstyle="round,pad=0.16", fc="white",
                        ec=c_budget, lw=0.44, alpha=0.92))

    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks([16, 32, 64, 128, 256])
    ax_b.set_xticklabels(["16", "32", "64", "128", "256"])
    ax_b.set_xlim(13, 310)
    ax_b.set_ylim(0.74, 0.89)
    ax_b.set_yticks([0.75, 0.80, 0.85])
    style_axis(ax_b, title="(b) Summary size  →  scorer liveness",
               xlabel="Summary size (B)",
               ylabel="Recovery@K")

    ax_b_r.set_ylim(0, 620)
    ax_b_r.set_yticks([0, 200, 400, 600])
    style_twin_axis(ax_b_r, ylabel="Metadata (KB)",
                    ylabel_color=c_noise)

    leg_b = ax_b.legend(
        handles=[l_fix, l_cal, l_meta],
        loc="lower right", bbox_to_anchor=(0.995, 0.02),
        frameon=True, framealpha=0.92,
        facecolor="white", edgecolor="#B0B0B0",
        ncol=1, fontsize=8.2,
        handlelength=1.7, handletextpad=0.32,
        labelspacing=0.22, borderpad=0.3,
    )
    leg_b.get_frame().set_linewidth(0.45)
    for txt in leg_b.get_texts():
        txt.set_fontweight("black")

    save(fig, OUT_DIR / "eval_size_sensitivity")
    print("Saved: eval_size_sensitivity")


if __name__ == "__main__":
    main()
