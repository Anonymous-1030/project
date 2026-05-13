"""
LSSL stress-test figure — 2x2 landscape.

  (a) Mean Recovery@K vs miss rate, one line per scenario per policy.
  (b) p5 tail Recovery@K -- shows LSSL's hard-floor protection.
  (c) Steps below no-sketch baseline (fraction) -- the reviewer's core
      failure mode, made concrete.
  (d) LSSL state mix across scenarios -- stacked horizontal bars show
      how much of each run was fresh / aging / stale-fallback.

Data: outputs/lssl_stress/stress_results.json
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
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from _figure_style import (
    setup_style, make_figure, style_axis, save, PALETTE,
)


ROOT = Path("d:/LLM/prose_v2")
SRC = ROOT / "outputs" / "lssl_stress" / "stress_results.json"
OUT_DIR = ROOT / "figure"


SCENARIO_ORDER = ["speculative", "batched", "preempt", "kv_compress", "multitenant"]
SCENARIO_LABEL = {
    "speculative": "spec. decode",
    "batched":     "batched",
    "preempt":     "preempt",
    "kv_compress": "KV compress",
    "multitenant": "multi-tenant",
}

POLICY_COLORS = {
    "lssl":    PALETTE["prose"],      # navy (our contribution)
    "no-lssl": PALETTE["fts"],        # brick red (keep stale)
    "vanilla": PALETTE["sw_host"],    # warm orange (drop-on-miss)
}
POLICY_MARKER = {
    "lssl":    "o",
    "no-lssl": "^",
    "vanilla": "D",
}
POLICY_LABEL = {
    "lssl":    "PROSE+LSSL (ours)",
    "no-lssl": "keep-stale",
    "vanilla": "drop-on-miss",
}
POLICY_LS = {
    "lssl":    "-",
    "no-lssl": (0, (4.5, 1.8)),
    "vanilla": (0, (2.5, 1.3)),
}


def main():
    setup_style()
    raw = json.loads(SRC.read_text())
    baseline = raw["meta"]["no_sketch_baseline"]
    scenarios = raw["scenarios"]

    fig, (ax_a, ax_b, ax_c, ax_d) = make_figure(shape="2x2", header_legend=True)

    # ─── Panel (a): mean Recovery vs miss rate ────────────────────
    for policy in ["vanilla", "no-lssl", "lssl"]:
        xs, ys = [], []
        for scen in SCENARIO_ORDER:
            cells = scenarios[scen]["cells"]
            for intensity, cell in cells.items():
                xs.append(cell[policy]["miss_rate"])
                ys.append(cell[policy]["mean_recovery"])
        xs, ys = np.array(xs), np.array(ys)
        order = np.argsort(xs)
        ax_a.plot(xs[order], ys[order],
                  color=POLICY_COLORS[policy],
                  linestyle=POLICY_LS[policy],
                  lw=2.6, marker=POLICY_MARKER[policy], markersize=6.4,
                  markeredgecolor="white", markeredgewidth=0.8,
                  label=POLICY_LABEL[policy], zorder=5)

    ax_a.axhline(baseline, color=PALETTE["baseline_g"], lw=1.4,
                 linestyle=(0, (3.0, 1.6)), alpha=0.9, zorder=2)
    ax_a.text(0.42, baseline - 0.02, f"no-sketch baseline  {baseline:.2f}",
              fontsize=8.5, color=PALETTE["baseline_g"], ha="right", va="top",
              fontweight="black", style="italic",
              bbox=dict(boxstyle="round,pad=0.22", fc="white",
                        ec=PALETTE["baseline_g"], lw=0.6, alpha=0.9))

    ax_a.set_xlim(0, 0.45)
    ax_a.set_ylim(0.40, 0.90)
    ax_a.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax_a.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    style_axis(ax_a, title="(a) Mean Recovery@K vs miss rate",
               xlabel="Sketch-miss rate",
               ylabel="Mean Recovery@K")

    # ─── Panel (b): p5 (tail) Recovery ────────────────────────────
    for policy in ["vanilla", "no-lssl", "lssl"]:
        xs, ys = [], []
        for scen in SCENARIO_ORDER:
            cells = scenarios[scen]["cells"]
            for intensity, cell in cells.items():
                xs.append(cell[policy]["miss_rate"])
                ys.append(cell[policy]["p5_recovery"])
        xs, ys = np.array(xs), np.array(ys)
        order = np.argsort(xs)
        ax_b.plot(xs[order], ys[order],
                  color=POLICY_COLORS[policy],
                  linestyle=POLICY_LS[policy],
                  lw=2.6, marker=POLICY_MARKER[policy], markersize=6.4,
                  markeredgecolor="white", markeredgewidth=0.8, zorder=5)

    ax_b.axhline(baseline, color=PALETTE["baseline_g"], lw=1.4,
                 linestyle=(0, (3.0, 1.6)), alpha=0.9, zorder=2)

    # Annotate LSSL's guaranteed floor zone
    ax_b.axhspan(baseline - 0.05, baseline + 0.20,
                 color=PALETTE["shade_ok"], alpha=0.45, zorder=1)
    ax_b.text(0.42, 0.71, "LSSL floor-protected",
              fontsize=8.2, color=PALETTE["accent_g"], ha="right", va="top",
              fontweight="black", style="italic")

    ax_b.set_xlim(0, 0.45)
    ax_b.set_ylim(0.22, 0.80)
    ax_b.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax_b.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    style_axis(ax_b, title="(b) 5th-percentile (tail) Recovery@K",
               xlabel="Sketch-miss rate",
               ylabel="p5 Recovery@K")

    # ─── Panel (c): fraction of steps below baseline ──────────────
    # One grouped bar cluster per scenario (using the highest intensity)
    bar_width = 0.25
    scenarios_x = np.arange(len(SCENARIO_ORDER))
    for j, policy in enumerate(["vanilla", "no-lssl", "lssl"]):
        heights = []
        for scen in SCENARIO_ORDER:
            intensities = scenarios[scen]["intensities"]
            hardest = str(intensities[-1])
            cell = scenarios[scen]["cells"][hardest]
            n_below = cell[policy]["steps_below_baseline"]
            heights.append(n_below / raw["meta"]["num_steps"])
        offset = (j - 1) * bar_width
        ax_c.bar(scenarios_x + offset, heights,
                 width=bar_width * 0.92,
                 color=POLICY_COLORS[policy],
                 edgecolor="white", linewidth=0.8,
                 label=POLICY_LABEL[policy], zorder=5)
        # Number labels above bars
        for k, h in enumerate(heights):
            ax_c.text(scenarios_x[k] + offset, h + 0.005,
                      f"{h*100:.0f}%",
                      ha="center", va="bottom",
                      fontsize=6.4, color=POLICY_COLORS[policy],
                      fontweight="black")

    ax_c.set_xticks(scenarios_x)
    ax_c.set_xticklabels([SCENARIO_LABEL[s] for s in SCENARIO_ORDER],
                         rotation=0)
    ax_c.set_xlim(-0.6, len(SCENARIO_ORDER) - 0.4)
    ax_c.set_ylim(0, 0.55)
    ax_c.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax_c.set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    style_axis(ax_c, title="(c) Steps below no-sketch floor (worst-intensity)",
               xlabel="Failure scenario",
               ylabel="Fraction of steps")

    # ─── Panel (d): LSSL state mix (stacked horizontal bars) ──────
    c_fresh = PALETTE["oracle"]
    c_aging = PALETTE["sw_host"]
    c_stale = PALETTE["baseline_g"]      # the hard-floor fallback

    y_pos = np.arange(len(SCENARIO_ORDER))
    fresh_fracs = []
    aging_fracs = []
    stale_fracs = []        # this is the "nosketch" fraction in LSSL
    for scen in SCENARIO_ORDER:
        intensities = scenarios[scen]["intensities"]
        hardest = str(intensities[-1])
        cell = scenarios[scen]["cells"][hardest]["lssl"]
        fresh_fracs.append(cell["fresh_frac"])
        aging_fracs.append(cell["aging_frac"])
        # In LSSL the "stale" state is the nosketch fallback
        stale_fracs.append(cell["nosketch_frac"])

    fresh_fracs = np.array(fresh_fracs)
    aging_fracs = np.array(aging_fracs)
    stale_fracs = np.array(stale_fracs)

    ax_d.barh(y_pos, fresh_fracs, color=c_fresh,
              edgecolor="white", linewidth=0.8, label="Fresh", zorder=5)
    ax_d.barh(y_pos, aging_fracs, left=fresh_fracs, color=c_aging,
              edgecolor="white", linewidth=0.8, label="Aging (extrap.)", zorder=5)
    ax_d.barh(y_pos, stale_fracs, left=fresh_fracs + aging_fracs,
              color=c_stale, edgecolor="white", linewidth=0.8,
              label="Fallback (no-sketch)", zorder=5)

    # Fraction labels inside each segment
    for i in range(len(SCENARIO_ORDER)):
        if fresh_fracs[i] > 0.08:
            ax_d.text(fresh_fracs[i] / 2, y_pos[i],
                      f"{fresh_fracs[i]*100:.0f}%",
                      ha="center", va="center",
                      fontsize=7.5, color="white", fontweight="black")
        if aging_fracs[i] > 0.06:
            ax_d.text(fresh_fracs[i] + aging_fracs[i] / 2, y_pos[i],
                      f"{aging_fracs[i]*100:.0f}%",
                      ha="center", va="center",
                      fontsize=7.5, color="white", fontweight="black")
        if stale_fracs[i] > 0.05:
            ax_d.text(fresh_fracs[i] + aging_fracs[i] + stale_fracs[i] / 2,
                      y_pos[i],
                      f"{stale_fracs[i]*100:.0f}%",
                      ha="center", va="center",
                      fontsize=7.5, color="white", fontweight="black")

    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels([SCENARIO_LABEL[s] for s in SCENARIO_ORDER])
    ax_d.invert_yaxis()
    ax_d.set_xlim(0, 1.01)
    ax_d.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_d.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    style_axis(ax_d, title="(d) LSSL state mix (worst-intensity)",
               xlabel="Fraction of steps",
               ylabel="")

    # Legend as a boxed inset at the top-right of the panel -- guaranteed
    # inside the axes rectangle so no clipping.
    leg_d = ax_d.legend(
        loc="lower right", bbox_to_anchor=(0.995, 0.03),
        frameon=True, framealpha=0.94,
        facecolor="white", edgecolor="#B0B0B0",
        ncol=1, fontsize=7.6,
        handlelength=1.3, handletextpad=0.35,
        labelspacing=0.28, borderpad=0.35,
    )
    leg_d.get_frame().set_linewidth(0.6)
    for txt in leg_d.get_texts():
        txt.set_fontweight("black")

    # ─── Shared header legend: the three policies ─────────────────
    legend_handles = [
        Line2D([0], [0], color=POLICY_COLORS[p], linestyle=POLICY_LS[p],
               lw=2.6, marker=POLICY_MARKER[p], markersize=6.4,
               markeredgecolor="white", markeredgewidth=0.6,
               label=POLICY_LABEL[p])
        for p in ["vanilla", "no-lssl", "lssl"]
    ]
    leg = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False, ncol=3,
        handlelength=2.2, handletextpad=0.35,
        columnspacing=1.8, fontsize=9.5,
    )
    for txt in leg.get_texts():
        txt.set_fontweight("black")

    save(fig, OUT_DIR / "eval_lssl_stress")
    print("Saved: eval_lssl_stress")


if __name__ == "__main__":
    main()
