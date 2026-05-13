"""
C2 — Fair FTS baselines with pre-filters, across three pressure regimes.

Headline claim to replace the 69x number:
    * At the paper's nominal operating point (32 GB/s, 1024 cand/step),
      PROSE's ordering delta over FTS-with-good-prefilter is 1.1-1.3x tok/s
      but ~100% RPE reduction.  The scorer, not the ordering, carries
      Recovery@K.
    * At a stressed regime (8 GB/s, 2048 cand/step), ordering delta is
      1.5-2x tok/s because the FTS pre-filter cannot keep up with offered
      load.  This is the defensible honest gain.
    * At a pathological regime (4 GB/s, 4096 cand/step), FTS-none
      collapses and FTS-prefilter still pays a transport premium.

Produces:
  out/data/c2_fts_baselines.json  — all three regimes
  out/figures/c2_throughput_grid.{png,pdf}
  out/figures/c2_rpe_elimination.{png,pdf}
  out/figures/c2_recovery_vs_tokps.{png,pdf}
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
    ("FTS-none",            "fts_none",    "none"),
    ("FTS-LRU",             "fts_lru",     "lru"),
    ("FTS-FreqRec",         "fts_freqrec", "freqrec"),
    ("FTS-Quest-prefilter", "fts_quest",   "quest"),
    ("SW-PCM-host",        "sw_host",     "odus_x"),
    ("SW-PCM-GPU",         "sw_gpu",      "odus_x"),
    ("IOMMU-filter",        "iommu",       "odus_x"),
    ("PROSE (CEFE)",        "cefe",        "odus_x"),
]

REGIMES = [
    ("baseline",     dict(cxl_bw_gbs=32.0, n_candidates=1024,
                          decode_slack_us=8000.0, decode_compute_us=12000.0,
                          budget_per_step=64, top_k_useful=32,
                          useful_fraction=0.04, semantic_strength=0.80)),
    ("stressed",     dict(cxl_bw_gbs=8.0,  n_candidates=2048,
                          decode_slack_us=8000.0, decode_compute_us=12000.0,
                          budget_per_step=64, top_k_useful=32,
                          useful_fraction=0.03, semantic_strength=0.80)),
    ("pathological", dict(cxl_bw_gbs=4.0,  n_candidates=4096,
                          decode_slack_us=8000.0, decode_compute_us=12000.0,
                          budget_per_step=64, top_k_useful=32,
                          useful_fraction=0.02, semantic_strength=0.80)),
]


def main():
    all_rows = []
    for regime_name, params in REGIMES:
        cfg = SimConfig(**params)
        regime_rows = []
        for label, boundary, scorer in CONFIGS:
            r = run_closed_loop(boundary, scorer, cfg, n_steps=384, seed=13)
            r["label"] = label
            r["regime"] = regime_name
            regime_rows.append(r)
            all_rows.append(r)
        print(f"\n[C2/{regime_name}] {'label':<22s} {'tok/s':>8s} "
              f"{'recov@K':>9s} {'RPE MB/s':>10s} {'io_us':>8s} {'qdepth99':>10s}")
        for r in regime_rows:
            rpe_mbs = r["rpe_bytes_mean"] * r["tok_per_s_mean"] / 1e6
            io_us = max(r["admission_us_mean"], r["transport_us_mean"])
            print(f"     {r['label']:<22s} {r['tok_per_s_mean']:>8.1f} "
                  f"{r['recovery_at_k_mean']:>9.3f} {rpe_mbs:>10.1f} "
                  f"{io_us:>8.1f} {r['queue_depth_peak_p99']:>10.3f}")

    save_json("c2_fts_baselines", {"rows": all_rows})

    # --------- Figure: 3-regime throughput grid -----------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=False)
    for ax, (regime_name, _) in zip(axes, REGIMES):
        rows = [r for r in all_rows if r["regime"] == regime_name]
        labels = [r["label"] for r in rows]
        tok = [r["tok_per_s_mean"] for r in rows]
        colors = [C["fts"]]*4 + [C["sw_host"], C["sw_gpu"],
                                  C["iommu"], C["cefe"]]
        ax.bar(labels, tok, color=colors, edgecolor="black", lw=0.6)
        ax.set_title(f"{regime_name}")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("tok/s")
        prose = next(r for r in rows if r["label"] == "PROSE (CEFE)")["tok_per_s_mean"]
        fts_best = max(r["tok_per_s_mean"] for r in rows
                       if r["label"].startswith("FTS") and r["label"] != "FTS-none")
        delta = prose / fts_best
        ax.text(0.02, 0.96, f"PROSE/best-FTS = {delta:.2f}x",
                transform=ax.transAxes, fontsize=18, va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    fig.suptitle("")
    save_fig(fig, "c2_throughput_grid")

    # --------- Figure: RPE elimination --------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    for ax, (regime_name, _) in zip(axes, REGIMES):
        rows = [r for r in all_rows if r["regime"] == regime_name]
        labels = [r["label"] for r in rows]
        rpe = [r["rpe_bytes_mean"] * r["tok_per_s_mean"] / 1e6 for r in rows]
        colors = [C["fts"]]*4 + [C["sw_host"], C["sw_gpu"],
                                  C["iommu"], C["cefe"]]
        ax.bar(labels, rpe, color=colors, edgecolor="black", lw=0.6)
        ax.set_title(regime_name)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Rejected-payload exposure (MB/s)")
    fig.suptitle("")
    save_fig(fig, "c2_rpe_elimination")

    # --------- Figure: queue-depth peak (P99) ----------------------------
    fig, ax = plt.subplots(figsize=(14, 8.0))
    baseline_rows = [r for r in all_rows if r["regime"] == "pathological"]
    short_labels = ["FTS-none", "FTS-LRU", "FTS-Freq", "FTS-Quest",
                    "SW-host", "SW-GPU", "IOMMU", "PROSE"]
    qd99 = [r["queue_depth_peak_p99"] for r in baseline_rows]
    colors_qd = [C["fts"]]*4 + [C["sw_host"], C["sw_gpu"], C["iommu"], C["cefe"]]
    idx = np.arange(len(short_labels))
    ax.barh(idx, qd99, color=colors_qd, edgecolor="black", lw=0.8)
    ax.set_yticks(idx)
    ax.set_yticklabels(short_labels)
    ax.set_xlabel("Queue-depth peak P99 (x link-capacity-per-slack)")
    ax.set_title("Transport queue saturation (pathological, 4 GB/s)")
    for i, v in enumerate(qd99):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center")
    save_fig(fig, "c2_queue_depth")

    # --------- Figure: per-step CXL traffic breakdown (offered load) ------
    fig, ax = plt.subplots(figsize=(14, 8.0))
    patho_rows_ol = [r for r in all_rows if r["regime"] == "pathological"]
    short_labels_ol = ["FTS-none", "FTS-LRU", "FTS-Freq", "FTS-Quest",
                       "SW-host", "SW-GPU", "IOMMU", "PROSE"]
    useful_ol = np.array([r["useful_bytes_mean"] for r in patho_rows_ol]) / 1e6
    wasted_ol = np.array([r["wasted_bytes_mean"] for r in patho_rows_ol]) / 1e6
    meta_ol   = np.array([r["meta_bytes_mean"]   for r in patho_rows_ol]) / 1e6
    idx_ol = np.arange(len(short_labels_ol))
    ax.barh(idx_ol, useful_ol, color=C["cefe"], label="useful payload",
            edgecolor="black", lw=0.5)
    ax.barh(idx_ol, wasted_ol, left=useful_ol, color=C["fts"],
            label="wasted payload (RPE)", edgecolor="black", lw=0.5)
    ax.barh(idx_ol, meta_ol, left=useful_ol+wasted_ol, color=C["accent1"],
            label="metadata", edgecolor="black", lw=0.5)
    ax.set_yticks(idx_ol)
    ax.set_yticklabels(short_labels_ol)
    ax.set_xlabel("MB/step on CXL link")
    ax.set_title("Per-step CXL traffic breakdown (pathological, 4 GB/s)")
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, "c2_offered_load")

    # --------- Figure: grouped bar — Recovery@K and tok/s side by side ------
    # (replaces scatter — scatter fails because PCM variants cluster at
    #  identical coordinates in the stressed regime)
    fig, axes = plt.subplots(1, 2, figsize=(20, 7.5))
    patho_rows = [r for r in all_rows if r["regime"] == "pathological"]
    labels = [r["label"] for r in patho_rows]
    short_labels = ["FTS-none", "FTS-LRU", "FTS-Freq", "FTS-Quest",
                    "SW-host", "SW-GPU", "IOMMU", "PROSE"]
    colors = [C["fts"]]*4 + [C["sw_host"], C["sw_gpu"], C["iommu"], C["cefe"]]
    idx = np.arange(len(patho_rows))

    # Panel A: tok/s
    tok = [r["tok_per_s_mean"] for r in patho_rows]
    axes[0].barh(idx, tok, color=colors, edgecolor="black", lw=0.8)
    axes[0].set_yticks(idx)
    axes[0].set_yticklabels(short_labels)
    axes[0].set_xlabel("tok/s")
    axes[0].set_title("Throughput (pathological, 4 GB/s)")
    for i, v in enumerate(tok):
        axes[0].text(v + 0.5, i, f"{v:.1f}", va="center")

    # Panel B: Recovery@K
    rec = [r["recovery_at_k_mean"] for r in patho_rows]
    axes[1].barh(idx, rec, color=colors, edgecolor="black", lw=0.8)
    axes[1].set_yticks(idx)
    axes[1].set_yticklabels(short_labels)
    axes[1].set_xlabel("Recovery@K")
    axes[1].set_title("Quality (pathological, 4 GB/s)")
    for i, v in enumerate(rec):
        axes[1].text(v + 0.005, i, f"{v:.3f}", va="center")

    save_fig(fig, "c2_recovery_vs_tokps")

    print("\n[C2] done.")


if __name__ == "__main__":
    main()
