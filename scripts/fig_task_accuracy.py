"""
Task-accuracy figure — 2x2 landscape.

Answers reviewer concern #2: report final task-level quality instead of
the Recovery@K proxy.

  (a) Per-task accuracy at 128K (4 tasks × 4 policies, grouped bars).
  (b) Average task accuracy vs model context length (one line per policy).
  (c) Calibration curves: Recovery@K → accuracy, one curve per task.
  (d) Gap-to-Oracle decomposition at 128K — how much of the remaining
      quality gap is ordering vs scorer vs evidence.

Data: outputs/task_accuracy/task_accuracy.json
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
    setup_style, make_figure, style_axis, save, PALETTE,
)


ROOT = Path("d:/LLM/prose_v2")
SRC = ROOT / "outputs" / "task_accuracy" / "task_accuracy.json"
OUT_DIR = ROOT / "figure"


POLICY_ORDER = ["vLLM-CXL", "PROSE-FTS", "PROSE", "Oracle-SBFI"]
POLICY_COLOR = {
    "vLLM-CXL":    PALETTE["vllm"],
    "PROSE-FTS":   PALETTE["fts"],
    "PROSE":       PALETTE["prose"],
    "Oracle-SBFI": PALETTE["oracle"],
}
POLICY_HATCH = {
    "vLLM-CXL":    "",
    "PROSE-FTS":   "///",
    "PROSE":       "",
    "Oracle-SBFI": "",
}
POLICY_MARKER = {
    "vLLM-CXL":    "o",
    "PROSE-FTS":   "^",
    "PROSE":       "o",
    "Oracle-SBFI": "*",
}
POLICY_LS = {
    "vLLM-CXL":    (0, (2.0, 1.2)),
    "PROSE-FTS":   (0, (4.0, 1.6)),
    "PROSE":       "-",
    "Oracle-SBFI": (0, (2.5, 1.3)),
}

TASKS = ["passkey", "needle", "sequential", "ruler"]
TASK_LABEL = {
    "passkey":    "passkey (EM)",
    "needle":     "needle (F1)",
    "sequential": "sequential (F1)",
    "ruler":      "RULER (EM)",
}
TASK_COLOR = {
    "passkey":    "#1F4E79",
    "needle":     PALETTE["accent_v"],
    "sequential": PALETTE["sw_host"],
    "ruler":      PALETTE["accent_r"],
}
TASK_MARKER = {
    "passkey":    "o",
    "needle":     "s",
    "sequential": "D",
    "ruler":      "^",
}


def main():
    setup_style()
    raw = json.loads(SRC.read_text())
    data = raw["data"]
    contexts = raw["contexts"]

    fig, (ax_a, ax_b, ax_c, ax_d) = make_figure(shape="2x2", header_legend=True)

    # ═══════════════════════════════════════════════════════════════
    # Panel (a): per-task accuracy at 128K -- grouped bars
    # ═══════════════════════════════════════════════════════════════
    ctx_key = "128"
    x = np.arange(len(TASKS))
    group_w = 0.84
    bar_w = group_w / len(POLICY_ORDER)
    for i, pol in enumerate(POLICY_ORDER):
        accs = [data[pol][ctx_key]["accuracy"][t] for t in TASKS]
        offset = (i - (len(POLICY_ORDER) - 1) / 2) * bar_w
        ax_a.bar(x + offset, accs,
                 width=bar_w * 0.95,
                 color=POLICY_COLOR[pol],
                 edgecolor="white", linewidth=0.9,
                 hatch=POLICY_HATCH[pol],
                 label=pol, zorder=5)
        for k, v in enumerate(accs):
            ax_a.text(x[k] + offset, v + 0.015,
                      f"{v:.2f}",
                      ha="center", va="bottom",
                      fontsize=6.6, color=POLICY_COLOR[pol],
                      fontweight="black")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    ax_a.set_xlim(-0.55, len(TASKS) - 0.45)
    ax_a.set_ylim(0, 1.12)
    ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style_axis(ax_a, title="(a) Task accuracy @ 128K",
               xlabel="Task",
               ylabel="Accuracy (EM / F1)")

    # ═══════════════════════════════════════════════════════════════
    # Panel (b): avg accuracy vs context
    # ═══════════════════════════════════════════════════════════════
    for pol in POLICY_ORDER:
        ys = [data[pol][f"{c}"]["avg_accuracy"] for c in contexts]
        ax_b.plot(contexts, ys,
                  color=POLICY_COLOR[pol],
                  linestyle=POLICY_LS[pol],
                  marker=POLICY_MARKER[pol], markersize=8.0,
                  markeredgecolor="white", markeredgewidth=0.9,
                  lw=2.6, zorder=5)

    # Shade the acceptable-quality band (>= 0.5 avg accuracy)
    ax_b.axhspan(0.5, 1.05, color=PALETTE["shade_ok"],
                 alpha=0.4, zorder=1)

    # Endpoint callouts
    last_ctx = contexts[-1]
    for pol in POLICY_ORDER:
        y = data[pol][f"{last_ctx}"]["avg_accuracy"]
        ax_b.annotate(f"{y:.2f}",
                      xy=(last_ctx, y),
                      xytext=(7, 0), textcoords="offset points",
                      fontsize=7.5, color=POLICY_COLOR[pol],
                      fontweight="black", va="center")

    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(contexts)
    ax_b.set_xticklabels([f"{c}K" for c in contexts])
    ax_b.set_xlim(contexts[0] * 0.88, contexts[-1] * 1.40)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style_axis(ax_b, title="(b) Average task accuracy vs context",
               xlabel="Model context length",
               ylabel="Avg accuracy (4 tasks)")

    # ═══════════════════════════════════════════════════════════════
    # Panel (c): Recovery→Accuracy calibration curves
    # ═══════════════════════════════════════════════════════════════
    rs = np.linspace(0, 1, 201)
    task_model = raw["task_model"]
    for t in TASKS:
        m = task_model[t]
        acc = m["floor"] + (1 - m["floor"]) * rs ** m["sharpness"]
        ax_c.plot(rs, acc, color=TASK_COLOR[t], lw=2.4,
                  label=TASK_LABEL[t], zorder=5)
    # Overlay actual (recovery, accuracy) pairs for each policy
    for pol in POLICY_ORDER:
        for c in contexts:
            d = data[pol][f"{c}"]
            for t in TASKS:
                ax_c.scatter(d["recovery"][t], d["accuracy"][t],
                             color=TASK_COLOR[t],
                             marker=TASK_MARKER[t],
                             edgecolor="white", linewidth=0.7,
                             s=25, alpha=0.55, zorder=6)

    ax_c.set_xlim(0, 1.02)
    ax_c.set_ylim(0, 1.05)
    ax_c.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style_axis(ax_c, title="(c) Recovery@K → accuracy calibration",
               xlabel="Recovery@K (retrieval)",
               ylabel="Task accuracy")

    leg_c = ax_c.legend(
        loc="lower right", bbox_to_anchor=(0.995, 0.03),
        frameon=True, framealpha=0.92,
        facecolor="white", edgecolor="#B0B0B0",
        ncol=1, fontsize=7.8,
        handlelength=1.6, handletextpad=0.32,
        labelspacing=0.24, borderpad=0.3,
    )
    leg_c.get_frame().set_linewidth(0.6)
    for txt in leg_c.get_texts():
        txt.set_fontweight("black")

    # ═══════════════════════════════════════════════════════════════
    # Panel (d): gap-to-Oracle decomposition at 128K
    # ═══════════════════════════════════════════════════════════════
    #
    # At 128K:
    #   vLLM-CXL  ~ 0.29     (admission gap: naive CXL pull)
    #   PROSE-FTS ~ 0.44     (ordering gap: fetch-then-score)
    #   PROSE     ~ 0.44     (PROSE == PROSE-FTS at task level; scorer same)
    #   Oracle    ~ 1.00
    #
    # Decomposition bar shows, for each intermediate, how much of the
    # 0.71-pp gap Oracle vs vLLM-CXL has been closed. PROSE and PROSE-FTS
    # differ only in ordering (PROSE adds SBFI); both share the scorer,
    # so the remaining gap is the "scorer / evidence" gap.
    ctx = "128"
    vllm = data["vLLM-CXL"][ctx]["avg_accuracy"]
    fts  = data["PROSE-FTS"][ctx]["avg_accuracy"]
    prose = data["PROSE"][ctx]["avg_accuracy"]
    oracle = data["Oracle-SBFI"][ctx]["avg_accuracy"]

    # Bars (stacked from bottom of 0): each segment is the contribution
    # of that step to closing the gap.
    bars = [
        ("admission\n(vLLM-CXL)",        0.0,                     vllm,            PALETTE["vllm"]),
        ("+ordering\n(PROSE-FTS)",       vllm,                    fts - vllm,      PALETTE["fts"]),
        ("+SBFI\n(PROSE)",               fts,                     max(prose - fts, 0.001), PALETTE["prose"]),
        ("+scorer\n(Oracle-SBFI)",       prose,                   oracle - prose,  PALETTE["oracle"]),
    ]
    x_pos = np.arange(len(bars))
    for xi, (label, bottom, height, color) in enumerate(bars):
        ax_d.bar(xi, height, bottom=bottom, color=color,
                 edgecolor="white", linewidth=0.9, width=0.6, zorder=5)
        # Endpoint label at top
        top = bottom + height
        ax_d.text(xi, top + 0.02,
                  f"{top:.2f}",
                  ha="center", va="bottom",
                  fontsize=7.8, color=color, fontweight="black")
        # Delta label inside segment
        if height >= 0.07:
            ax_d.text(xi, bottom + height / 2,
                      f"+{height:.2f}",
                      ha="center", va="center",
                      fontsize=7.5, color="white", fontweight="black")
        elif height > 0.001 and xi == 2:
            ax_d.text(xi, bottom + 0.015,
                      f"(same scorer as PROSE-FTS)",
                      ha="center", va="bottom",
                      fontsize=6.0, color=color, fontweight="black",
                      style="italic")

    # Oracle horizontal reference
    ax_d.axhline(oracle, color=PALETTE["oracle"], lw=1.4,
                 linestyle=(0, (4.0, 1.8)), alpha=0.9, zorder=2)
    ax_d.text(len(bars) - 0.6, oracle - 0.025, "Oracle ceiling",
              fontsize=7.6, color=PALETTE["oracle"],
              ha="right", va="top", fontweight="black", style="italic")

    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels([b[0] for b in bars])
    ax_d.set_xlim(-0.55, len(bars) - 0.45)
    ax_d.set_ylim(0, 1.12)
    ax_d.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style_axis(ax_d, title="(d) Closing the accuracy gap @ 128K",
               xlabel="",
               ylabel="Avg task accuracy")

    # ─── Shared header legend for panels (a) & (b) ─────────────────
    legend_handles = []
    for pol in POLICY_ORDER:
        legend_handles.append(
            Line2D([0], [0],
                   color=POLICY_COLOR[pol],
                   linestyle=POLICY_LS[pol],
                   marker=POLICY_MARKER[pol], markersize=8.0,
                   markeredgecolor="white", markeredgewidth=0.7,
                   lw=2.6, label=pol)
        )
    leg = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False, ncol=4,
        handlelength=2.2, handletextpad=0.35,
        columnspacing=1.8, fontsize=9.5,
    )
    for txt in leg.get_texts():
        txt.set_fontweight("black")

    save(fig, OUT_DIR / "eval_task_accuracy")
    print("Saved: eval_task_accuracy")


if __name__ == "__main__":
    main()
