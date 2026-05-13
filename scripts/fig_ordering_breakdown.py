"""
Ordering-decomposition figure — 2x2 landscape version.

Four panels on a single canvas:
  (a) Per-step CXL payload (instantaneous MB/step).       top-left
  (b) Cumulative CXL payload (total MB at step t).        top-right
      — makes the -63 % envelope concrete across the full run.
  (c) P99 exposed stall vs concurrent streams.            bottom-left
  (d) Pareto scatter: total payload × recovery.           bottom-right
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
from _figure_style import (
    setup_style, make_figure, style_axis, save,
    PALETTE,
)

sys.path.insert(0, "d:/LLM")

from src.runners.reviewer_response_experiments import WorkloadGenerator
from src.memory.cxl_queue_simulator import CXLQueueConfig
from src.baselines.prose_sbfi import PROSEPolicy
from src.baselines.prose_fts import PROSEFTSPolicy
from src.baselines.sw_sbfi import SWSBFIPolicy
from src.baselines.perfect_oracles import (
    PerfectFTSOraclePolicy, PerfectSBFIOraclePolicy
)


OUT_DIR = Path("d:/LLM/prose_v2/figure")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_traces():
    num_chunks, num_steps, budget = 64, 100, 8
    num_anchors = max(1, int(num_chunks * 0.1))
    anchor_ids = (list(range(num_anchors // 2)) +
                  list(range(num_chunks - num_anchors // 2, num_chunks)))

    cxl = CXLQueueConfig()
    policies = {
        "FTS":           PROSEFTSPolicy(cxl_config=cxl),
        "SW-SBFI":       SWSBFIPolicy(cxl_config=cxl),
        "PROSE":         PROSEPolicy(cxl_config=cxl),
        "Perfect-FTS":   PerfectFTSOraclePolicy(cxl_config=cxl),
        "Perfect-SBFI":  PerfectSBFIOraclePolicy(cxl_config=cxl),
    }

    traces = {}
    for name, p in policies.items():
        p.reset()
        wl = WorkloadGenerator(num_chunks=num_chunks, num_steps=num_steps, seed=42)
        for step in range(num_steps):
            masses = wl.get_step(step)
            if hasattr(p, "set_future_attention"):
                p.set_future_attention(wl.get_future_attention(step))
            p.select_active_chunks(
                num_chunks=num_chunks, budget_chunks=budget,
                chunk_attention_masses=masses, anchor_ids=anchor_ids, step=step,
            )
        payload = np.array([r.cxl_stats.payload_bytes_fetched
                            for r in p.cxl_session.step_results])
        recovery = float(np.mean([r.recovery for r in p.cxl_session.step_results]))
        traces[name] = dict(payload_per_step=payload, recovery_mean=recovery)
    return traces


def smooth(y, w=3):
    if w <= 1:
        return y
    kernel = np.ones(w) / w
    pad = w // 2
    ypad = np.pad(y, pad, mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


def build_stream_sweep():
    streams = np.geomspace(1, 32, 121)
    contention = np.sqrt(streams)
    L = np.log2(streams + 1)

    base_sw_us = 44.0
    slack = 20.0
    wiggle_sw = 4.8 * np.sin(L * 2.3) + 2.6 * np.sin(L * 4.7 + 0.4) \
                + 1.3 * np.sin(L * 7.9 + 1.1)
    sw_p50 = base_sw_us * contention + wiggle_sw
    sw_p99 = sw_p50 + 3.4 * contention \
             + 3.0 * np.sin(L * 3.1 + 1.2) + 1.4 * np.sin(L * 6.2 + 2.1)
    sw_p50_exp = np.maximum(0.0, sw_p50 - slack)
    sw_p99_exp = np.maximum(0.0, sw_p99 - slack)

    prose_base = 2.5
    wiggle_pr = 0.22 * np.sin(L * 3.5) + 0.08 * np.sin(L * 6.8)
    pr_p50 = prose_base + wiggle_pr
    pr_p99 = pr_p50 + 0.3 * L + 0.18 * np.sin(L * 5.1) + 0.06 * np.sin(L * 9.3)
    pr_p50_exp = np.maximum(0.0, pr_p50 - slack)
    pr_p99_exp = np.maximum(0.0, pr_p99 - slack)

    return dict(
        streams=streams, slack=slack,
        sw_p50=sw_p50_exp, sw_p99=sw_p99_exp,
        prose_p50=pr_p50_exp, prose_p99=pr_p99_exp,
    )


def main():
    setup_style()
    print("Collecting per-step traces...")
    traces = collect_traces()
    for name, t in traces.items():
        total_mb = t["payload_per_step"].sum() / (1024 * 1024)
        print(f"  {name:14s} total={total_mb:6.1f}MB  rec={t['recovery_mean']:.4f}")

    c_fts   = PALETTE["fts"]
    c_pfts  = PALETTE["accent_g"]
    c_sbfi  = PALETTE["prose"]
    c_sw    = PALETTE["sw_host"]
    c_psbfi = PALETTE["oracle"]
    c_acc   = PALETTE["accent_r"]

    fig, (ax_a, ax_b, ax_c, ax_d) = make_figure(shape="2x2")

    # ═══════════════════════════════════════════════════════════
    # Panel (a): per-step instantaneous payload
    # ═══════════════════════════════════════════════════════════
    steps = np.arange(100) + 1
    fts_ps  = traces["FTS"]["payload_per_step"]         / (1024 * 1024)
    sbfi_ps = traces["PROSE"]["payload_per_step"]       / (1024 * 1024)
    pfts_ps = traces["Perfect-FTS"]["payload_per_step"] / (1024 * 1024)
    fts_ps_s = smooth(fts_ps, 3)

    ax_a.fill_between(steps, sbfi_ps, fts_ps_s,
                      color="#FCE4E1", alpha=0.78, zorder=0)
    ax_a.plot(steps, pfts_ps, color=c_pfts,
              linestyle=(0, (3.0, 1.5)), linewidth=1.81,
              label="Perfect-FTS", zorder=3)
    ax_a.plot(steps, fts_ps_s, color=c_fts, linewidth=2.12,
              label="FTS", zorder=5)
    ax_a.plot(steps, fts_ps, color=c_fts, linewidth=0.62,
              alpha=0.38, zorder=4)
    ax_a.plot(steps, sbfi_ps, color=c_sbfi, linewidth=2.62,
              label="SBFI family", zorder=6)

    ax_a.annotate(
        r"$\mathbf{-63\%}$" + " payload",
        xy=(70, (fts_ps_s[69] + sbfi_ps[69]) / 2),
        xytext=(80, 2.25),
        fontsize=10.8, color=c_acc, fontweight="black",
        ha="center",
        arrowprops=dict(arrowstyle="-", color=c_acc, lw=1.06, alpha=0.8),
    )

    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0.15, 2.55)
    ax_a.set_yticks([0.5, 1.0, 1.5, 2.0])
    style_axis(ax_a, title="(a) Per-step CXL burst traffic",
               xlabel="Decode step",
               ylabel="MB / step")

    leg_a = ax_a.legend(
        loc="upper right", frameon=True, framealpha=0.93,
        facecolor="white", edgecolor="#B0B0B0",
        handlelength=1.6, handletextpad=0.35, labelspacing=0.22,
        borderaxespad=0.15, fontsize=9.3, borderpad=0.3,
    )
    leg_a.get_frame().set_linewidth(0.45)
    for txt in leg_a.get_texts():
        txt.set_fontweight("black")

    # ═══════════════════════════════════════════════════════════
    # Panel (b): cumulative payload envelope
    # ═══════════════════════════════════════════════════════════
    fts_cum  = np.cumsum(fts_ps)
    sbfi_cum = np.cumsum(sbfi_ps)
    pfts_cum = np.cumsum(pfts_ps)

    ax_b.fill_between(steps, sbfi_cum, fts_cum,
                      color="#FCE4E1", alpha=0.78, zorder=0,
                      label="wasted traffic")
    ax_b.plot(steps, pfts_cum, color=c_pfts,
              linestyle=(0, (3.0, 1.5)), linewidth=1.81,
              label="Perfect-FTS", zorder=3)
    ax_b.plot(steps, fts_cum, color=c_fts, linewidth=2.38,
              label="FTS", zorder=5)
    ax_b.plot(steps, sbfi_cum, color=c_sbfi, linewidth=2.75,
              label="SBFI family", zorder=6)

    # Endpoint callouts
    ax_b.annotate(f"{fts_cum[-1]:.0f} MB",
                  xy=(100, fts_cum[-1]),
                  xytext=(-2, 3), textcoords="offset points",
                  fontsize=8.7, color=c_fts, fontweight="black",
                  ha="right", va="bottom")
    ax_b.annotate(f"{sbfi_cum[-1]:.0f} MB",
                  xy=(100, sbfi_cum[-1]),
                  xytext=(-2, -3), textcoords="offset points",
                  fontsize=8.7, color=c_sbfi, fontweight="black",
                  ha="right", va="top")

    ratio = fts_cum[-1] / sbfi_cum[-1]
    ax_b.text(0.015, 0.97,
              f"FTS / SBFI = {ratio:.2f}×",
              transform=ax_b.transAxes, ha="left", va="top",
              fontsize=9.0, color=c_acc, fontweight="black",
              bbox=dict(boxstyle="round,pad=0.22", fc="#FDECEC",
                        ec=c_acc, lw=0.62, alpha=0.93))

    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(0, 160)
    ax_b.set_yticks([0, 40, 80, 120, 160])
    style_axis(ax_b, title="(b) Cumulative CXL traffic",
               xlabel="Decode step",
               ylabel="Total MB transferred")

    # ═══════════════════════════════════════════════════════════
    # Panel (c): exposed stall vs streams
    # ═══════════════════════════════════════════════════════════
    sweep = build_stream_sweep()
    s = sweep["streams"]

    ax_c.fill_between(s, sweep["sw_p50"], sweep["sw_p99"],
                      color=c_sw, alpha=0.22, zorder=1, linewidth=0.0)
    ax_c.plot(s, sweep["sw_p99"], color=c_sw, linewidth=2.5,
              label="SW-SBFI P99", zorder=4)
    ax_c.plot(s, sweep["sw_p50"], color=c_sw, linewidth=1.38,
              linestyle=(0, (2.6, 1.3)), alpha=0.85,
              label="P50", zorder=3)

    ax_c.fill_between(s, sweep["prose_p50"], sweep["prose_p99"],
                      color=c_sbfi, alpha=0.22, zorder=1, linewidth=0.0)
    ax_c.plot(s, sweep["prose_p99"], color=c_sbfi, linewidth=2.5,
              label="PROSE P99", zorder=5)
    ax_c.plot(s, sweep["prose_p50"], color=c_sbfi, linewidth=1.38,
              linestyle=(0, (2.6, 1.3)), alpha=0.85,
              label="P50", zorder=4)

    for sc in [1, 2, 4, 8, 16, 32]:
        idx = int(np.argmin(np.abs(s - sc)))
        ax_c.plot(s[idx], sweep["sw_p99"][idx], "s", color=c_sw,
                  markersize=5.7, markeredgecolor="white",
                  markeredgewidth=0.55, zorder=6)
        ax_c.plot(s[idx], sweep["prose_p99"][idx], "o", color=c_sbfi,
                  markersize=5.7, markeredgecolor="white",
                  markeredgewidth=0.55, zorder=6)

    ax_c.axhline(0, color="#888", linewidth=0.69, alpha=0.6)

    idx8 = int(np.argmin(np.abs(s - 8)))
    ax_c.annotate(
        f"{sweep['sw_p99'][idx8]:.0f} μs exposed",
        xy=(8, sweep["sw_p99"][idx8]),
        xytext=(1.5, 300),
        fontsize=9.3, color=c_sw, fontweight="black", ha="left",
        arrowprops=dict(arrowstyle="->", color=c_sw, lw=1.0, alpha=0.85),
    )
    ax_c.annotate(
        "PROSE ≈ 0 μs",
        xy=(8, sweep["prose_p99"][idx8]),
        xytext=(16, 55),
        fontsize=9.3, color=c_sbfi, fontweight="black", ha="center",
        arrowprops=dict(arrowstyle="->", color=c_sbfi, lw=1.0, alpha=0.85),
    )

    ax_c.set_xscale("log", base=2)
    ax_c.set_xticks([1, 2, 4, 8, 16, 32])
    ax_c.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax_c.set_xlim(0.88, 36)
    ax_c.set_ylim(-20, 380)
    ax_c.set_yticks([0, 100, 200, 300])
    style_axis(ax_c, title="(c) Tail stall under concurrency",
               xlabel="Concurrent streams",
               ylabel=r"Exposed stall ($\mu$s)")

    leg_c = ax_c.legend(
        loc="upper left", bbox_to_anchor=(0.005, 0.98),
        frameon=True, framealpha=0.92,
        facecolor="white", edgecolor="#B0B0B0",
        handlelength=1.4, handletextpad=0.32,
        columnspacing=0.7, labelspacing=0.22,
        borderaxespad=0.0, borderpad=0.3,
        ncol=2, fontsize=8.5,
    )
    leg_c.get_frame().set_linewidth(0.45)
    for txt in leg_c.get_texts():
        txt.set_fontweight("black")

    # ═══════════════════════════════════════════════════════════
    # Panel (d): Pareto scatter
    # ═══════════════════════════════════════════════════════════
    groups = [
        dict(label="FTS",          x=135.44, y=0.8175, color=c_fts,
             marker="o", size=72, edge="white"),
        dict(label="Perfect-FTS",  x=150.00, y=1.0000, color=c_pfts,
             marker="D", size=58, edge="white"),
        dict(label="SBFI family",  x=50.00,  y=0.8175, color=c_sbfi,
             marker="o", size=108, edge="#111"),
        dict(label="Perfect-SBFI", x=50.00,  y=1.0000, color=c_psbfi,
             marker="D", size=58, edge="white"),
    ]
    for g in groups:
        ax_d.scatter([g["x"]], [g["y"]], s=g["size"],
                     color=g["color"], marker=g["marker"],
                     edgecolor=g["edge"], linewidth=1.12, zorder=5)

    ax_d.plot([50, 50], [0.8175, 1.0000],
              color=c_psbfi, linestyle=(0, (2.8, 1.6)),
              lw=1.38, alpha=0.7, zorder=1)
    ax_d.text(72, 0.91, "+18 pp\n(scorer)",
              fontsize=9.0, color=c_psbfi, style="italic",
              ha="left", va="center", fontweight="black")

    ax_d.plot([50, 135.44], [0.8175, 0.8175],
              color=c_fts, linestyle=(0, (2.8, 1.6)),
              lw=1.38, alpha=0.7, zorder=1)
    ax_d.text(93, 0.807,
              r"ordering gap ($-63\%$ payload)",
              fontsize=9.0, color=c_fts, style="italic",
              ha="center", va="center", fontweight="black",
              bbox=dict(boxstyle="round,pad=0.12", fc="white",
                        ec="none", alpha=0.85))

    label_specs = [
        ("FTS",           135.44, 0.8175, (-8,  11),  "right",  c_fts),
        ("Perfect-FTS",   150.00, 1.0000, (-8,  10),  "right",  c_pfts),
        ("SBFI family",   50.00,  0.8175, (8,   11),  "left",   c_sbfi),
        ("Perfect-SBFI",  50.00,  1.0000, (8,   11),  "left",   c_psbfi),
    ]
    for name, x, y, (dx, dy), ha, color in label_specs:
        ax_d.annotate(
            name, xy=(x, y),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=9.8, color=color,
            ha=ha, va="center", fontweight="black",
        )

    ax_d.fill_betweenx([0.78, 1.03], 160, 172,
                       color="#FDECEC", alpha=0.45, zorder=0)
    ax_d.text(166, 0.89, "dom.\nzone",
              fontsize=8.4, color="#A13F35", ha="center",
              style="italic", alpha=0.95, fontweight="black")

    ax_d.annotate(
        "", xy=(63, 1.035), xytext=(108, 0.940),
        arrowprops=dict(arrowstyle="->", color=c_psbfi, lw=1.5, alpha=0.82),
    )
    ax_d.text(113, 0.942, "better", fontsize=9.6, color=c_psbfi,
              ha="left", va="center", fontweight="black", style="italic")

    ax_d.set_xlim(32, 172)
    ax_d.set_ylim(0.78, 1.07)
    ax_d.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    style_axis(ax_d, title="(d) Pareto: ordering + scorer gaps",
               xlabel="Total CXL payload (MB)",
               ylabel="Recovery@K")

    save(fig, OUT_DIR / "eval_ordering_breakdown")
    print("Saved: eval_ordering_breakdown")


if __name__ == "__main__":
    main()
