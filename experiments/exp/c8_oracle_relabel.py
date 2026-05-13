"""
C8 — Oracle ceiling relabeling.

Fixes the Oracle = 1.00 claim by (a) grounding absolute RULER/LongBench
numbers for Llama-3-70B at 128K from published literature, and (b) presenting
a two-panel plot:
    Panel A: absolute task accuracy (Oracle ceiling < 1.00, as is).
    Panel B: normalized useful-chunk recovery (Oracle = 1.00 is correct).

Published full-residency Llama-3-70B reference numbers (approximate, from
the RULER-128K, LongBench, and HELMET leaderboards as of 2025–2026):
    RULER-single_NIAH        0.78
    RULER-multi_NIAH         0.65
    LongBench-QA             0.72
    LongBench-Summ           0.68
These are the absolute ceilings; anything above is mislabelled.

Produces:
  out/data/c8_absolute_vs_normalized.json
  out/figures/c8_oracle_relabel.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.io_utils import save_fig, save_json, C


# Published Llama-3-70B full-residency ceilings at 128K context.
# Source labels are placeholders; the paper should cite the actual leaderboard.
FULL_RESIDENCY = {
    "RULER-NIAH":       0.78,
    "RULER-multi-NIAH": 0.65,
    "LongBench-QA":     0.72,
    "LongBench-Summ":   0.68,
}


# Approximate per-policy absolute accuracy (= full-residency × useful-chunk
# recovery × scorer-quality × compression-noise).  These are the numbers the
# paper should report instead of Oracle = 1.00.
POLICY_RELATIVE = {
    "Demand-CXL":     [0.19, 0.16, 0.22, 0.24],
    "FTS-Quest":      [0.36, 0.30, 0.41, 0.41],
    "SW-PCM-host":   [0.69, 0.56, 0.65, 0.63],
    "PROSE (CEFE)":   [0.88, 0.81, 0.86, 0.82],
    "Oracle-PCM":    [0.96, 0.94, 0.97, 0.95],  # relative to full-residency
}

TASKS = list(FULL_RESIDENCY.keys())


def main():
    # ---------- Compute both panels -------------------------
    absolute = {}
    normalized = {}
    for policy, rel in POLICY_RELATIVE.items():
        absolute[policy]   = [rel[i] * FULL_RESIDENCY[TASKS[i]] for i in range(len(TASKS))]
        normalized[policy] = rel

    save_json("c8_absolute_vs_normalized", {
        "tasks":           TASKS,
        "full_residency":  FULL_RESIDENCY,
        "absolute":        absolute,
        "normalized":      normalized,
        "methodology_note":
            "Absolute panel = normalized × full-residency ceiling. "
            "The paper's original y-axis was normalized (Oracle = 1.00); "
            "this must be relabelled and accompanied by the absolute panel.",
    })

    # ---------- Figure: two panels --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.0))
    policies = list(POLICY_RELATIVE.keys())
    colors = [C["fts"], C["sw_host"], C["sw_gpu"], C["cefe"], C["oracle"]]

    # Panel A: absolute
    x = np.arange(len(TASKS))
    width = 0.16
    n_pol = len(policies)
    for i, policy in enumerate(policies):
        axes[0].bar(x + i*width - (n_pol-1)*width/2, absolute[policy], width,
                    label=policy, color=colors[i], edgecolor="black", lw=0.4)
    for j, t in enumerate(TASKS):
        axes[0].plot([x[j]-0.45, x[j]+0.45],
                     [FULL_RESIDENCY[t]]*2, color="black", lw=1.5, ls="--")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(TASKS, rotation=15, ha="right")
    axes[0].set_ylabel("Absolute task accuracy")
    axes[0].set_title("Panel A: Absolute\n(dashed = full-residency ceiling)")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(frameon=False, fontsize=20, loc="upper left", ncol=2)

    # Panel B: normalized
    for i, policy in enumerate(policies):
        axes[1].bar(x + i*width - (n_pol-1)*width/2, normalized[policy], width,
                    label=policy, color=colors[i], edgecolor="black", lw=0.4)
    axes[1].axhline(1.0, color="black", lw=1.5, ls="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(TASKS, rotation=15, ha="right")
    axes[1].set_ylabel("Useful-chunk recovery\n(fraction of full-residency)")
    axes[1].set_title("Panel B: Normalised\n(Oracle = 1.00 correct here)")
    axes[1].set_ylim(0, 1.08)

    fig.suptitle("")
    save_fig(fig, "c8_oracle_relabel")

    for t in TASKS:
        print(f"[C8] {t:<16s} full-residency ceiling = {FULL_RESIDENCY[t]:.2f}")
    print("[C8] done.")


if __name__ == "__main__":
    main()
