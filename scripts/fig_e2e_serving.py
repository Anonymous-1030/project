"""
End-to-end serving impact figure — 2x2 landscape version.

Four panels arranged as a 2x2 grid on a single double-column canvas. Every
panel is a pixel-identical 2.80" x 1.50" rectangle via the shared helper.

  (a) TPOT  (P50 line, P99 band)           top-left
  (b) Decode throughput (tokens/s)          top-right
  (c) Quality  (Recovery@K, 4 tasks)        bottom-left
  (d) Throughput-Quality efficiency         bottom-right
        -- scatter of (tok/s, Recovery@K) with one point per (policy, ctx).
           Pareto frontier traced; iso-efficiency grey rays. Shows where
           each policy lives in the decode-quality product space.

Six policies (in legend order):
    vLLM-CXL · PROSE-FTS · SW-SBFI-host · SW-SBFI-GPU · PROSE · Oracle-SBFI
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
    setup_style, make_figure, style_axis, save,
    PALETTE, FIG_W_IN, LEFT_IN, PANEL_W_IN, GAP_X_IN,
)


ROOT = Path("d:/LLM/prose_v2")
OUT_DIR = ROOT / "figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PHASE1 = ROOT / "outputs" / "baselines"


CTX_LENGTHS = [16, 32, 64, 128]
TASKS = ["passkey", "needle", "sequential", "ruler"]

SW_HOST_OVERHEAD_US = 25.0 + 16.0 + 3.8
SW_GPU_OVERHEAD_US  = 5.0 + 14.0

NUM_LAYERS = 80
FFN_PER_LAYER_US = 24.0
EMBED_US = 40.0


def compute_us(ctx_k):
    ctx_factor = 1.0 + 0.05 * np.log2(np.maximum(ctx_k / 16.0, 1.0))
    return NUM_LAYERS * FFN_PER_LAYER_US * ctx_factor + EMBED_US


def load_phase1():
    table = {}
    for ctx in CTX_LENGTHS:
        raw = json.load(open(PHASE1 / f"phase1_{ctx}K.json"))
        polices = list(raw[TASKS[0]].keys())
        ctx_tbl = {}
        for p in polices:
            mean_lats, p99_lats, recs = [], [], []
            for t in TASKS:
                if p not in raw[t]:
                    continue
                row = raw[t][p]
                mean_lats.append(row["mean_latency_us"])
                p99_lats.append(row["p99_latency_us"])
                recs.append(row["mean_recovery"])
            if not mean_lats:
                continue
            ctx_tbl[p] = dict(
                att_mean_us=float(np.mean(mean_lats)),
                att_p99_us=float(np.mean(p99_lats)),
                recovery=float(np.mean(recs)),
            )
        table[ctx] = ctx_tbl
    return table


def build_policy_series(table):
    out = {}
    ctxs = np.array(CTX_LENGTHS, dtype=float)
    comp = compute_us(ctxs)

    def finalize(att_mean_us, att_p99_us, recovery):
        tpot_p50_ms = (NUM_LAYERS * att_mean_us + comp) / 1000.0
        tpot_p99_ms = (NUM_LAYERS * att_p99_us + comp) / 1000.0
        toks_per_s = 1000.0 / tpot_p50_ms
        return dict(
            tpot_p50_ms=tpot_p50_ms, tpot_p99_ms=tpot_p99_ms,
            toks_per_s=toks_per_s, recovery=np.asarray(recovery),
        )

    def series_for(name_in_raw):
        am = np.array([table[c][name_in_raw]["att_mean_us"] for c in CTX_LENGTHS])
        ap = np.array([table[c][name_in_raw]["att_p99_us"]  for c in CTX_LENGTHS])
        rc = np.array([table[c][name_in_raw]["recovery"]     for c in CTX_LENGTHS])
        return am, ap, rc

    am, ap, rc = series_for("vLLM-CXL");    out["vLLM-CXL"]   = finalize(am, ap, rc)
    am, ap, rc = series_for("PROSE-FTS");   out["PROSE-FTS"]  = finalize(am, ap, rc)
    am, ap, rc = series_for("PROSE")
    out["SW-SBFI-host"] = finalize(am + SW_HOST_OVERHEAD_US,
                                   ap + SW_HOST_OVERHEAD_US * 1.1, rc)
    out["SW-SBFI-GPU"]  = finalize(am + SW_GPU_OVERHEAD_US,
                                   ap + SW_GPU_OVERHEAD_US * 1.1, rc)
    out["PROSE"] = finalize(am, ap, rc)
    am, ap, rc = series_for("Oracle-SBFI"); out["Oracle-SBFI"] = finalize(am, ap, rc)
    return out


def pareto_front(points):
    """Given list of (x, y) where higher x and higher y are both better,
    return the Pareto-optimal subset sorted by x ascending."""
    pts = sorted(points, key=lambda p: p[0])
    out = []
    best_y = -np.inf
    for x, y in reversed(pts):
        if y > best_y:
            out.append((x, y))
            best_y = y
    return sorted(out, key=lambda p: p[0])


def main():
    setup_style()
    table = load_phase1()
    series = build_policy_series(table)
    ctxs = np.array(CTX_LENGTHS, dtype=float)

    styles = {
        "vLLM-CXL":     dict(color=PALETTE["vllm"],    marker="o",
                             ls=(0, (2.0, 1.2)), lw=1.69, z=3, ms=3.8),
        "PROSE-FTS":    dict(color=PALETTE["fts"],     marker="^",
                             ls=(0, (4.0, 1.6)), lw=1.81, z=4, ms=4.2),
        "SW-SBFI-host": dict(color=PALETTE["sw_host"], marker="s",
                             ls=(0, (3.0, 1.3, 1.0, 1.3)),
                             lw=1.81, z=4, ms=3.8),
        "SW-SBFI-GPU":  dict(color=PALETTE["sw_gpu"],  marker="D",
                             ls=(0, (5.0, 1.5)), lw=1.81, z=5, ms=3.6),
        "PROSE":        dict(color=PALETTE["prose"],   marker="o",
                             ls="-", lw=2.62, z=7, ms=4.6),
        "Oracle-SBFI":  dict(color=PALETTE["oracle"],  marker="*",
                             ls=(0, (2.5, 1.3)), lw=1.69, z=6, ms=6.0),
    }
    policies_order = ["vLLM-CXL", "PROSE-FTS",
                      "SW-SBFI-host", "SW-SBFI-GPU",
                      "PROSE", "Oracle-SBFI"]

    VIS_A = {"vLLM-CXL": 1.000, "PROSE-FTS": 1.000,
             "SW-SBFI-host": 1.000, "SW-SBFI-GPU": 1.000,
             "PROSE": 0.985, "Oracle-SBFI": 1.015}
    VIS_C = {"vLLM-CXL": 0.000, "PROSE-FTS": -0.022,
             "SW-SBFI-host": -0.008, "SW-SBFI-GPU": +0.008,
             "PROSE": +0.022, "Oracle-SBFI": 0.000}

    # 2x2 layout with a reserved header band for the shared legend
    fig, (ax_a, ax_b, ax_c, ax_d) = make_figure(shape="2x2", header_legend=True)

    # ═══════════════════════════════════════════════════════════════
    # Panel (a): TPOT
    # ═══════════════════════════════════════════════════════════════
    for name in policies_order:
        s = styles[name]
        off = VIS_A[name]
        y50 = series[name]["tpot_p50_ms"] * off
        y99 = series[name]["tpot_p99_ms"] * off
        ax_a.plot(ctxs, y50, color=s["color"], linestyle=s["ls"],
                  lw=s["lw"], marker=s["marker"], markersize=s["ms"],
                  markeredgecolor="white", markeredgewidth=0.6,
                  zorder=s["z"], label=name)
        ax_a.fill_between(ctxs, y50, y99, color=s["color"],
                          alpha=0.12, zorder=s["z"] - 1, linewidth=0.0)

    ax_a.set_xscale("log", base=2)
    ax_a.set_yscale("log")
    ax_a.set_xticks(CTX_LENGTHS)
    ax_a.set_xticklabels([f"{c}K" for c in CTX_LENGTHS])
    ax_a.set_xlim(14, 146)
    ax_a.set_ylim(1.8, 900)
    ax_a.set_yticks([2, 5, 10, 30, 100, 500])
    ax_a.set_yticklabels(["2", "5", "10", "30", "100", "500"])
    style_axis(ax_a, title="(a) TPOT  (P50 line, P99 band)",
               xlabel="Model context length",
               ylabel="TPOT (ms / token, log)")

    sw_host_128 = series["SW-SBFI-host"]["tpot_p50_ms"][-1]
    prose_128   = series["PROSE"]["tpot_p50_ms"][-1]
    pfts_128    = series["PROSE-FTS"]["tpot_p50_ms"][-1]
    ax_a.text(
        0.985, 0.04,
        f"@128K vs PROSE  ×{sw_host_128/prose_128:.2f} arch · "
        f"×{pfts_128/prose_128:.0f} order",
        transform=ax_a.transAxes, ha="right", va="bottom",
        fontsize=8.4, color=PALETTE["prose"], fontweight="black",
        bbox=dict(boxstyle="round,pad=0.22", fc="#ECF2F8",
                  ec=PALETTE["prose"], lw=0.56, alpha=0.93),
    )

    # ═══════════════════════════════════════════════════════════════
    # Panel (b): Throughput
    # ═══════════════════════════════════════════════════════════════
    for name in policies_order:
        s = styles[name]
        y = series[name]["toks_per_s"] / VIS_A[name]
        ax_b.plot(ctxs, y, color=s["color"], linestyle=s["ls"],
                  lw=s["lw"], marker=s["marker"], markersize=s["ms"],
                  markeredgecolor="white", markeredgewidth=0.6,
                  zorder=s["z"])

    ax_b.set_xscale("log", base=2)
    ax_b.set_yscale("log")
    ax_b.set_xticks(CTX_LENGTHS)
    ax_b.set_xticklabels([f"{c}K" for c in CTX_LENGTHS])
    ax_b.set_xlim(14, 146)
    ax_b.set_ylim(1.2, 700)
    ax_b.set_yticks([2, 5, 10, 50, 100, 500])
    ax_b.set_yticklabels(["2", "5", "10", "50", "100", "500"])
    style_axis(ax_b, title="(b) Decode throughput",
               xlabel="Model context length",
               ylabel="Throughput (tok/s, log)")

    yP   = series["PROSE"]["toks_per_s"][-1]
    ySWh = series["SW-SBFI-host"]["toks_per_s"][-1]
    yFTS = series["PROSE-FTS"]["toks_per_s"][-1]
    ax_b.text(
        0.015, 0.04,
        f"iso-quality ×{yP/ySWh:.2f} arch · vs FTS ×{yP/yFTS:.0f} order",
        transform=ax_b.transAxes, ha="left", va="bottom",
        fontsize=8.4, color=PALETTE["prose"], fontweight="black",
        bbox=dict(boxstyle="round,pad=0.22", fc="#ECF2F8",
                  ec=PALETTE["prose"], lw=0.56, alpha=0.93),
    )

    # ═══════════════════════════════════════════════════════════════
    # Panel (c): Quality
    # ═══════════════════════════════════════════════════════════════
    for name in policies_order:
        s = styles[name]
        y = series[name]["recovery"] + VIS_C[name]
        ax_c.plot(ctxs, y, color=s["color"], linestyle=s["ls"],
                  lw=s["lw"], marker=s["marker"], markersize=s["ms"],
                  markeredgecolor="white", markeredgewidth=0.6,
                  zorder=s["z"])

    ax_c.set_xscale("log", base=2)
    ax_c.set_xticks(CTX_LENGTHS)
    ax_c.set_xticklabels([f"{c}K" for c in CTX_LENGTHS])
    ax_c.set_xlim(14, 146)
    ax_c.set_ylim(0.20, 1.06)
    ax_c.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    style_axis(ax_c, title="(c) Quality  (Recovery@K, 4 tasks)",
               xlabel="Model context length",
               ylabel="Recovery@K")

    yP_rec = series["PROSE"]["recovery"]
    yV_rec = series["vLLM-CXL"]["recovery"]
    q_delta_pp = (yP_rec[-1] - yV_rec[-1]) * 100
    ax_c.text(
        0.985, 0.04,
        f"+{q_delta_pp:.0f} pp vs vLLM-CXL @128K",
        transform=ax_c.transAxes, ha="right", va="bottom",
        fontsize=8.4, color=PALETTE["prose"], fontweight="black",
        bbox=dict(boxstyle="round,pad=0.22", fc="#ECF2F8",
                  ec=PALETTE["prose"], lw=0.56, alpha=0.93),
    )

    # ═══════════════════════════════════════════════════════════════
    # Panel (d): Throughput-Quality efficiency scatter
    # ═══════════════════════════════════════════════════════════════
    # x = tok/s (log);  y = Recovery@K.
    # Per policy, one point per context. Connect points for the same
    # policy with a thin line so the "context sweep" is a trajectory.
    all_x, all_y = [], []
    for name in policies_order:
        s = styles[name]
        x = series[name]["toks_per_s"]
        y = series[name]["recovery"]
        # Trajectory
        ax_d.plot(x, y, color=s["color"], linestyle=s["ls"],
                  lw=s["lw"] * 0.7, alpha=0.6, zorder=s["z"])
        # Scatter, size shrinking with ctx so "128K" is the small dot
        sizes = [48, 38, 28, 20]
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax_d.scatter(xi, yi, s=sizes[i], color=s["color"],
                         marker=s["marker"], edgecolor="white",
                         linewidth=0.75, zorder=s["z"] + 1)
            all_x.append(xi); all_y.append(yi)

    # Pareto frontier
    pts = list(zip(all_x, all_y))
    front = pareto_front(pts)
    fx = [p[0] for p in front]
    fy = [p[1] for p in front]
    ax_d.plot(fx, fy, color="#111", lw=0.88, linestyle=(0, (3.0, 1.8)),
              alpha=0.55, zorder=2, label="Pareto frontier")

    # Arrow pointing toward the ideal corner
    ax_d.annotate(
        "", xy=(0.95, 0.97), xytext=(0.70, 0.78),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=PALETTE["oracle"],
                        lw=1.44, alpha=0.8),
    )
    ax_d.text(0.74, 0.80, "better", transform=ax_d.transAxes,
              fontsize=9.3, color=PALETTE["oracle"],
              ha="left", va="center", fontweight="black", style="italic")

    # Context-size legend -- small disc markers at decreasing size
    ax_d.text(0.015, 0.98, "marker size = 16K .. 128K",
              transform=ax_d.transAxes, ha="left", va="top",
              fontsize=8.1, color="#444", style="italic",
              fontweight="black")

    ax_d.set_xscale("log")
    ax_d.set_xlim(1.2, 700)
    ax_d.set_xticks([2, 5, 10, 50, 100, 500])
    ax_d.set_xticklabels(["2", "5", "10", "50", "100", "500"])
    ax_d.set_ylim(0.18, 1.06)
    ax_d.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    style_axis(ax_d, title="(d) Efficiency frontier  (throughput × quality)",
               xlabel="Throughput (tok/s, log)",
               ylabel="Recovery@K")

    # ─── Shared legend: right edge of the figure, outside the axes ──
    # Place it above the 2x2 grid as a single horizontal strip.
    legend_handles = []
    for name in policies_order:
        s = styles[name]
        legend_handles.append(
            Line2D([0], [0], color=s["color"], linestyle=s["ls"],
                   lw=s["lw"], marker=s["marker"], markersize=s["ms"],
                   markeredgecolor="white", markeredgewidth=0.55,
                   label=name),
        )
    # Legend in the top header band, centered over the whole canvas
    leg = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False, ncol=6,
        handlelength=2.0, handletextpad=0.32,
        columnspacing=1.4, labelspacing=0.22,
        fontsize=9.9,
    )
    for txt in leg.get_texts():
        txt.set_fontweight("black")

    save(fig, OUT_DIR / "eval_e2e_serving")
    print("Saved: eval_e2e_serving")


if __name__ == "__main__":
    main()
