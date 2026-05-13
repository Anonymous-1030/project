#!/usr/bin/env python3
"""
Run CXL Simulator Cross-Validation and 128K Sensitivity Analysis.

This script produces the empirical evidence required to answer:
  "Your simulator has not been validated against a physical CXL device."

Outputs:
  outputs/hpca_cxl_validation/cross_validation_report.json
  outputs/hpca_cxl_validation/sensitivity_128k_matrix.csv
  outputs/hpca_cxl_validation/fig_cxl_sensitivity.pdf
"""

import sys
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from hardware_model.cxl_cross_validator import CXLCrossValidator


def plot_sensitivity_heatmaps(matrix, output_dir: Path):
    """Generate latency/bandwidth heatmaps from the sensitivity matrix."""
    if not matrix:
        print("No sensitivity matrix entries to plot.")
        return

    # Filter to 128K context for the primary heatmap
    entries_128k = [m for m in matrix if m["context_length"] == 131072]
    if not entries_128k:
        entries_128k = [m for m in matrix if m["context_length"] == max(m2["context_length"] for m2 in matrix)]

    versions = sorted({m["cxl_version"] for m in entries_128k})
    fig, axes = plt.subplots(1, len(versions), figsize=(6 * len(versions), 5))
    if len(versions) == 1:
        axes = [axes]

    for ax, ver in zip(axes, versions):
        sub = [m for m in entries_128k if m["cxl_version"] == ver]
        # Pivot: x=protocol_overhead_pct, y=cxl_latency_ns, color=sim_avg_latency_ns
        overheads = sorted({m["protocol_overhead_pct"] for m in sub})
        latencies = sorted({m["cxl_latency_ns"] for m in sub})

        Z = np.zeros((len(latencies), len(overheads)))
        for i, lat in enumerate(latencies):
            for j, oh in enumerate(overheads):
                vals = [
                    m["sim_avg_latency_ns"]
                    for m in sub
                    if m["cxl_latency_ns"] == lat and m["protocol_overhead_pct"] == oh
                ]
                Z[i, j] = np.mean(vals) if vals else 0.0

        im = ax.imshow(Z, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(overheads)))
        ax.set_xticklabels([f"{o:.0f}%" for o in overheads])
        ax.set_yticks(range(len(latencies)))
        ax.set_yticklabels([f"{l:.0f}" for l in latencies])
        ax.set_xlabel("Protocol Overhead")
        ax.set_ylabel("CXL Latency (ns)")
        ax.set_title(f"CXL {ver} — Simulated Avg Latency @ 128K context")
        plt.colorbar(im, ax=ax, label="Latency (ns)")

    fig.tight_layout()
    pdf_path = output_dir / "fig_cxl_sensitivity.pdf"
    fig.savefig(pdf_path, dpi=300)
    print(f"[Plot] Saved {pdf_path}")


def plot_bandwidth_saturation(output_dir: Path):
    """Overlay published bandwidth saturation curves."""
    from hardware_model.cxl_published_validation import PUBLISHED_PROFILES

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"samsung_isscc_2022": "C0", "intel_sapphire_rapids_2023": "C1",
              "meta_micro_2023": "C2", "kaist_isca_2023": "C3", "cxl_3_0_spec": "C4"}

    for name, prof in PUBLISHED_PROFILES.items():
        if not prof.bandwidth_saturation_gb_s:
            continue
        offered = [p[0] for p in prof.bandwidth_saturation_gb_s]
        achieved = [p[1] for p in prof.bandwidth_saturation_gb_s]
        ax.plot(offered, achieved, marker="o", label=f"{prof.source}", color=colors.get(name, "gray"))

    ax.set_xlabel("Offered Load (GB/s)")
    ax.set_ylabel("Achieved Bandwidth (GB/s)")
    ax.set_title("Published CXL Bandwidth Saturation Curves")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    pdf_path = output_dir / "fig_bandwidth_saturation.pdf"
    fig.savefig(pdf_path, dpi=300)
    print(f"[Plot] Saved {pdf_path}")


def main():
    output_dir = Path("outputs/hpca_cxl_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    validator = CXLCrossValidator(output_dir=str(output_dir))
    summary = validator.run_full_validation(generate_matrix=True)
    validator.export_summary(summary)

    # Pretty-print validation results
    print("\n" + "=" * 80)
    print("CXL SIMULATOR CROSS-VALIDATION RESULTS")
    print("=" * 80)
    print(f"Overall pass rate: {summary.overall_pass_rate:.1%}")
    print(f"Consensus latency envelope (CXL 2.0): {summary.consensus_latency_envelope_ns} ns")
    print(f"Consensus protocol overhead (CXL 2.0): {summary.consensus_protocol_overhead_pct}%")
    print(f"Consensus row-buffer hit rate: {summary.consensus_row_buffer_hit_rate}")

    fails = [r for r in summary.validation_reports if not r["passed"]]
    if fails:
        print(f"\nFailed checks ({len(fails)}):")
        for r in fails[:10]:
            print(f"  - {r['check_name']}: sim={r['simulated_value']:.2f}, "
                  f"expected=[{r['expected_min']:.2f}, {r['expected_max']:.2f}]")
        if len(fails) > 10:
            print(f"  ... and {len(fails) - 10} more")
    else:
        print("\nAll validation checks passed.")

    # Plotting
    plot_sensitivity_heatmaps(summary.sensitivity_matrix, output_dir)
    plot_bandwidth_saturation(output_dir)

    print("\n" + "=" * 80)
    print("SUCCESS: CXL validation artifacts written to outputs/hpca_cxl_validation/")
    print("=" * 80)


if __name__ == "__main__":
    main()
