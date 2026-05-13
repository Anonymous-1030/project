"""
C3 — Scorer × Ordering 2D ablation.

Rows:    ordering ∈ {FTS, SW-PCM-host, CEFE}
Columns: scorer   ∈ {Quest, FreqRec, ODUS-X}

Produces two heatmaps:
    * Recovery@K → varies mostly across COLUMNS (scorer work).
    * tok/s      → varies mostly across ROWS    (ordering work).

This is the cleanest answer to the reviewer's Fig-5(d) concern.  It makes
the current paper's "scorer is orthogonal to PCM" claim precise:
    * Under the PCM schema the two are *separable*.
    * Recovery@K is the scorer axis; tok/s is the ordering axis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.cxl_admission_sim import SimConfig, run_closed_loop
from sim.io_utils import save_fig, save_json, C


ROWS = [
    ("FTS",      "fts_quest"),  # FTS with the column scorer as pre-filter
    ("SW-host",  "sw_host"),
    ("CEFE",     "cefe"),
]
COLS = [
    ("Quest",    "quest"),
    ("FreqRec",  "freqrec"),
    ("ODUS-X",   "odus_x"),
]


def fts_boundary_for(col_scorer: str) -> str:
    # Map each scorer to its proper FTS boundary (same decision cost profile).
    return {
        "quest":   "fts_quest",
        "freqrec": "fts_freqrec",
        "odus_x":  "fts_quest",   # ODUS-X as FTS pre-filter costs like Quest
    }[col_scorer]


def main():
    cfg = SimConfig(
        cxl_bw_gbs=8.0,            # "stressed" regime — ordering actually matters
        n_candidates=2048,
        decode_slack_us=8000.0,
        decode_compute_us=12000.0,
        budget_per_step=64,
        top_k_useful=32,
        useful_fraction=0.03,
        semantic_strength=0.80,
    )

    recov = np.zeros((len(ROWS), len(COLS)))
    tokps = np.zeros((len(ROWS), len(COLS)))
    rows_json = []

    for i, (rlabel, rboundary_hint) in enumerate(ROWS):
        for j, (clabel, cscorer) in enumerate(COLS):
            boundary = fts_boundary_for(cscorer) if rlabel == "FTS" else rboundary_hint
            r = run_closed_loop(boundary, cscorer, cfg, n_steps=384, seed=29)
            recov[i, j] = r["recovery_at_k_mean"]
            tokps[i, j] = r["tok_per_s_mean"]
            r["row"] = rlabel
            r["col"] = clabel
            rows_json.append(r)

    save_json("c3_scorer_ordering_ablation", {
        "rows":    [x[0] for x in ROWS],
        "cols":    [x[0] for x in COLS],
        "recov":   recov.tolist(),
        "tokps":   tokps.tolist(),
        "details": rows_json,
    })

    # ---------------- Figure: two heatmaps --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.0))

    for ax, mat, title, fmt, cmap in [
        (axes[0], recov, "Recovery@K (scorer axis)",  "{:.2f}", "YlGn"),
        (axes[1], tokps, "tok/s (ordering axis)",     "{:.0f}", "YlOrRd"),
    ]:
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c[0] for c in COLS])
        ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in ROWS])
        ax.set_title(title)
        for i in range(len(ROWS)):
            for j in range(len(COLS)):
                ax.text(j, i, fmt.format(mat[i, j]),
                        ha="center", va="center",
                        color="black", fontsize=20)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row/col margin effects
    recov_row = recov.mean(axis=1)
    recov_col = recov.mean(axis=0)
    tokps_row = tokps.mean(axis=1)
    tokps_col = tokps.mean(axis=0)
    msg = (
        f"Recovery@K spread across scorers   (ΔR_scorer)   = "
        f"{recov_col.max()-recov_col.min():.2f}\n"
        f"Recovery@K spread across orderings (ΔR_ordering) = "
        f"{recov_row.max()-recov_row.min():.2f}\n"
        f"tok/s      spread across scorers   (Δt_scorer)   = "
        f"{tokps_col.max()-tokps_col.min():.1f}\n"
        f"tok/s      spread across orderings (Δt_ordering) = "
        f"{tokps_row.max()-tokps_row.min():.1f}"
    )
    print("[C3] margin decomposition:\n" + msg)
    fig.suptitle("")
    save_fig(fig, "c3_scorer_ordering_heatmaps")

    # ---------------- Summary JSON ----------------------------------
    save_json("c3_margin_decomposition", {
        "delta_recov_scorer":   float(recov_col.max()-recov_col.min()),
        "delta_recov_ordering": float(recov_row.max()-recov_row.min()),
        "delta_tokps_scorer":   float(tokps_col.max()-tokps_col.min()),
        "delta_tokps_ordering": float(tokps_row.max()-tokps_row.min()),
    })

    print("[C3] done.")


if __name__ == "__main__":
    main()
