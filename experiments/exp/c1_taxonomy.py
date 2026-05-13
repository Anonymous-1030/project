"""
C1 — Placement taxonomy.

Produces:
  out/data/c1_taxonomy.json      : 8-row taxonomy of admission filters
  out/figures/c1_taxonomy.{png,pdf}: visual placement chart
  out/data/c1_taxonomy_table.md  : LaTeX/Markdown drop-in for the paper

This experiment does not run the simulator.  It formalises the PCM schema
so the reviewer's "PCM-as-principle vs PCM-as-placement" concern is
quantitatively separable from performance numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.io_utils import save_fig, save_json, C


TAXONOMY = [
    # (name, verdict_input, binding_boundary, quantum_ns, quantum_bytes,
    #  reorder_payload?, commit_discipline, rpe_risk,
    #  schema_instance, note)
    ("Quest (ICML'24)",
        "key min/max pages",       "HBM page-load",  120,   4*1024,
        True, "implicit (page level)", "low",
        "PCM-hbm",
        "verdict binds inside HBM; CXL not in scope"),
    ("InfiniGen (OSDI'24)",
        "key proxy projection",    "host-runtime",   900,   64*1024,
        False, "implicit (no abort)", "high",
        "non-PCM for CXL",
        "advisory; reorderable by coalescer"),
    ("SnapKV",
        "self-attention mass",     "HBM page-load",  100,   4*1024,
        True, "implicit",           "low",
        "PCM-hbm",
        "operates on resident KV only"),
    ("H2O",
        "heavy-hitter history",    "HBM page-load",  120,   4*1024,
        True, "implicit",           "low",
        "PCM-hbm",
        "eviction-first, not a CXL filter"),
    ("TinyLFU (generic)",
        "frequency sketch",        "cache-insert",   50,    -1,
        True, "implicit",           "low",
        "PCM-generic",
        "classical admission filter"),
    ("SW-PCM-host (this)",
        "compact summary",         "host runtime",   47_000, 64*1024,
        True, "explicit",           "low",
        "PCM-host",
        "binds pre-doorbell; OS-crossing dominates"),
    ("SW-PCM-GPU (this)",
        "compact summary",         "GPU persistent kernel", 5_200, 64*1024,
        True, "explicit",           "low",
        "PCM-gpu",
        "contends with compute kernels"),
    ("PROSE (CEFE, this)",
        "compact summary",         "on-CE (pre-DMA-dispatch)", 3_900, 64*1024,
        True, "explicit (LSSL + commit)", "minimal",
        "PCM-cefe",
        "tightest binding point for CXL-resident KV"),
]


COL_NAMES = [
    "System", "Verdict input", "Binding boundary", "Quantum (ns)",
    "Payload quantum (B)", "Pre-payload reorder?", "Commit discipline",
    "RPE risk", "Schema instance", "Note",
]


def render_markdown() -> str:
    lines = ["| " + " | ".join(COL_NAMES) + " |",
             "|" + "|".join(["---"] * len(COL_NAMES)) + "|"]
    for row in TAXONOMY:
        vals = list(row)
        vals[4] = "—" if vals[4] == -1 else f"{vals[4]}"
        vals[3] = f"{vals[3]}"
        vals[5] = "yes" if vals[5] else "no"
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    return "\n".join(lines)


def render_figure():
    """
    Scatter placement: x = binding-quantum (log), y = payload quantum (B),
    colour = schema bucket.  PCM-cefe ends up in the tightest corner.
    Points that share coordinates are jittered; labels get manual offsets.
    """
    fig, ax = plt.subplots(figsize=(16.0, 10.0))

    bucket_color = {
        "PCM-hbm":       C["accent1"],
        "PCM-generic":   C["accent2"],
        "non-PCM for CXL": C["fts"],
        "PCM-host":      C["sw_host"],
        "PCM-gpu":       C["sw_gpu"],
        "PCM-cefe":      C["cefe"],
    }

    # Manual label offsets (dx, dy in points) to avoid overlap.
    label_offsets = {
        "Quest (ICML'24)":    (8, -14),
        "InfiniGen (OSDI'24)": (8, 8),
        "SnapKV":             (-60, -16),
        "H2O":                (-30, 10),
        "TinyLFU (generic)":  (8, -12),
        "SW-PCM-host (this)": (8, 8),
        "SW-PCM-GPU (this)": (8, -14),
        "PROSE (CEFE, this)": (8, 8),
    }

    # Jitter multipliers for points sharing similar coordinates
    jitter_x = {
        "Quest (ICML'24)":    1.0,
        "SnapKV":             0.65,
        "H2O":                1.55,
        "SW-PCM-GPU (this)": 0.85,
    }
    jitter_y = {
        "SnapKV":             0.75,
        "H2O":                1.35,
        "InfiniGen (OSDI'24)": 1.3,
        "SW-PCM-host (this)": 1.4,
        "SW-PCM-GPU (this)": 0.75,
    }

    for (name, _, _, qns, qb, _, _, _, bucket, _) in TAXONOMY:
        if qb == -1:
            qb = 1024
        xval = qns * jitter_x.get(name, 1.0)
        yval = qb * jitter_y.get(name, 1.0)
        ax.scatter(xval, yval, s=150, c=bucket_color[bucket],
                   edgecolor="black", linewidth=0.9, zorder=3)
        ofs = label_offsets.get(name, (8, 4))
        ax.annotate(name, (xval, yval), textcoords="offset points",
                    xytext=ofs, fontsize=18, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="gray")
                    if abs(ofs[0]) > 30 or abs(ofs[1]) > 14 else None)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Verdict-binding latency (ns)")
    ax.set_ylabel("Payload non-preemptive quantum (bytes)")
    ax.set_title("Placement taxonomy of admission filters\n"
                 "(lower-left = tighter binding)")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)

    # Separate the two regimes visually
    ax.axhline(y=16*1024, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.text(60, 20*1024, "CXL payload regime (64 KB)",
            fontsize=18, color="gray", style="italic")
    ax.text(60, 2*1024, "HBM page regime (4 KB)",
            fontsize=18, color="gray", style="italic")

    legend_handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                                 markerfacecolor=v, markeredgecolor="black",
                                 markersize=9, label=k)
                      for k, v in bucket_color.items()]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False,
              fontsize=16)

    save_fig(fig, "c1_taxonomy")


def main():
    save_json("c1_taxonomy", {
        "columns": COL_NAMES,
        "rows": TAXONOMY,
        "claim":
            "Under the PCM schema, prior work (Quest, InfiniGen, SnapKV, H2O, "
            "TinyLFU) instantiates at boundaries whose non-preemptive quantum "
            "is either sub-KB (HBM page) or advisory (host coalescer). PROSE-"
            "CEFE is the only instance that binds the verdict in hardware before "
            "a 64KB CXL DMA dispatches.",
    })
    md = render_markdown()
    (Path(__file__).resolve().parent.parent / "out" / "data" / "c1_taxonomy_table.md").write_text(md, encoding="utf-8")
    render_figure()
    print("[C1] taxonomy table + figure emitted.")


if __name__ == "__main__":
    main()
