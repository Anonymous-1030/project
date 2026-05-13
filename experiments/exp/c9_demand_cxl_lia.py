"""
C9 — Demand-CXL and LIA-style baselines (replacement for the misleading
    "vLLM-CXL" baseline in the current paper).

Baselines clarified:
  * Demand-CXL     : LRU eviction, demand-fetch on miss, no prefetch, no scorer.
                     This is what the paper's "vLLM-CXL" actually models.
  * LIA-style      : Coarse-scored prefetch with a recency+frequency filter at
                     host, 64 KB quantum, no PCM enforcement (payload fetches
                     before verdict binds).
  * PROSE (CEFE)   : This paper.

We evaluate all three on (Recovery@K, tok/s, RPE).

Produces:
  out/data/c9_demand_cxl_lia.json
  out/figures/c9_baseline_set.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.cxl_admission_sim import SimConfig, run_closed_loop
from sim.io_utils import save_fig, save_json, C


CONFIGS = [
    ("Demand-CXL",       "demand_cxl", "lru"),
    ("LIA-style",        "lia_style",  "freqrec"),
    ("FTS-Quest",        "fts_quest",  "quest"),
    ("SW-PCM-host",     "sw_host",    "odus_x"),
    ("PROSE (CEFE)",     "cefe",       "odus_x"),
]


def main():
    cfg = SimConfig(
        cxl_bw_gbs=16.0,
        n_candidates=1536,
        decode_slack_us=8000.0,
        decode_compute_us=12000.0,
        budget_per_step=64,
        top_k_useful=32,
        useful_fraction=0.035,
        semantic_strength=0.80,
    )

    rows = []
    for label, boundary, scorer in CONFIGS:
        r = run_closed_loop(boundary, scorer, cfg, n_steps=384, seed=41)
        r["label"] = label
        rows.append(r)

    save_json("c9_demand_cxl_lia", {"cfg": cfg.__dict__, "rows": rows})

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.0))
    labels = [r["label"] for r in rows]
    colors = [C["fts"], C["sw_host"], C["fts"], C["sw_gpu"], C["cefe"]]

    axes[0].bar(labels, [r["recovery_at_k_mean"] for r in rows],
                color=colors, edgecolor="black", lw=0.5)
    axes[0].set_title("Recovery@K")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=25, ha="right")

    axes[1].bar(labels, [r["tok_per_s_mean"] for r in rows],
                color=colors, edgecolor="black", lw=0.5)
    axes[1].set_title("Throughput (tok/s)")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=25, ha="right")

    axes[2].bar(labels,
                [r["rpe_bytes_mean"]*r["tok_per_s_mean"]/1e6 for r in rows],
                color=colors, edgecolor="black", lw=0.5)
    axes[2].set_title("RPE (MB/s)")
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=25, ha="right")

    # suptitle removed — was overlapping subplot titles
    save_fig(fig, "c9_baseline_set")

    for r in rows:
        rpe_mbs = r["rpe_bytes_mean"] * r["tok_per_s_mean"] / 1e6
        print(f"[C9] {r['label']:<32s} recov={r['recovery_at_k_mean']:.3f} "
              f"tok/s={r['tok_per_s_mean']:.1f}  RPE={rpe_mbs:.0f} MB/s")

    print("[C9] done.")


if __name__ == "__main__":
    main()
