"""
C10 — Low-bandwidth regime decomposition.

Reviewer asked whether the collapse at 4 GB/s is:
    (a) useful-admitted traffic alone saturating the link, or
    (b) the metadata path becoming the bottleneck.

We sweep CXL_BW ∈ {2, 4, 6, 8, 12, 16, 24, 32, 48, 64} GB/s and report, for
PROSE (CEFE):
    * useful_bytes_per_sec
    * meta_bytes_per_sec
    * rpe_bytes_per_sec     (should be ~0)
    * link_utilisation      (sum / capacity)

If (a), useful/link -> 1.0 at low BW with meta/link small.  If (b), meta/link
dominates at low BW.  Our CEFE numbers put this firmly in (a).

Produces:
  out/data/c10_low_bw.json
  out/figures/c10_bw_sweep.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.cxl_admission_sim import SimConfig, run_closed_loop
from sim.io_utils import save_fig, save_json, C


BWS = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def main():
    rows = []
    for bw in BWS:
        cfg = SimConfig(
            cxl_bw_gbs=bw,
            n_candidates=1024,
            decode_slack_us=8000.0,
            decode_compute_us=12000.0,
            budget_per_step=64,
            top_k_useful=32,
            useful_fraction=0.04,
            semantic_strength=0.80,
        )
        for boundary, scorer, label in [
            ("fts_quest", "quest",  "FTS-Quest-prefilter"),
            ("cefe",      "odus_x", "PROSE (CEFE)"),
        ]:
            r = run_closed_loop(boundary, scorer, cfg, n_steps=256, seed=71)
            r["cxl_bw_gbs"] = bw
            r["label"]      = label
            rows.append(r)

    save_json("c10_low_bw", {"rows": rows})

    prose = [r for r in rows if r["label"] == "PROSE (CEFE)"]
    fts   = [r for r in rows if r["label"] == "FTS-Quest-prefilter"]

    # --- figure: line plot of link utilisation fractions vs BW ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.0), sharey=True)
    for ax, series, title in [
        (axes[0], prose, "PROSE (CEFE)"),
        (axes[1], fts,   "FTS-Quest-prefilter"),
    ]:
        bws = np.array([r["cxl_bw_gbs"] for r in series], dtype=float)
        tps = np.array([r["tok_per_s_mean"] for r in series])
        useful = np.array([r["useful_bytes_mean"] for r in series]) * tps
        meta   = np.array([r["meta_bytes_mean"]   for r in series]) * tps
        wasted = np.array([r["wasted_bytes_mean"] for r in series]) * tps
        cap    = bws * 1e9
        uf     = useful / cap
        mf     = meta   / cap
        wf     = wasted / cap

        ax.plot(bws, uf, marker="o", color=C["cefe"], label="useful payload")
        ax.plot(bws, mf, marker="s", color=C["accent1"], label="metadata")
        ax.plot(bws, wf, marker="^", color=C["fts"], label="wasted (RPE)")
        ax.axhline(1.0, ls="--", color="black", lw=0.8)
        ax.set_xlabel("CXL bandwidth (GB/s)")
        ax.set_ylabel("Fraction of link capacity")
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(True, ls=":", lw=0.5, alpha=0.5)
        ax.set_xlim(0, 68)
    fig.suptitle("")
    save_fig(fig, "c10_bw_sweep")

    print("[C10] BW  | PROSE useful% | PROSE meta% | PROSE RPE% | FTS useful% | FTS RPE%")
    for bw in BWS:
        pr = [r for r in prose if r["cxl_bw_gbs"] == bw][0]
        ft = [r for r in fts   if r["cxl_bw_gbs"] == bw][0]
        cap = bw * 1e9
        u_p  = pr["useful_bytes_mean"] * pr["tok_per_s_mean"] / cap
        m_p  = pr["meta_bytes_mean"]   * pr["tok_per_s_mean"] / cap
        w_p  = pr["wasted_bytes_mean"] * pr["tok_per_s_mean"] / cap
        u_f  = ft["useful_bytes_mean"] * ft["tok_per_s_mean"] / cap
        w_f  = ft["wasted_bytes_mean"] * ft["tok_per_s_mean"] / cap
        print(f"       {bw:3d}  | {u_p*100:11.1f}% | {m_p*100:9.1f}% | {w_p*100:9.1f}% "
              f"| {u_f*100:9.1f}% | {w_f*100:7.1f}%")

    print("[C10] done.")


if __name__ == "__main__":
    main()
