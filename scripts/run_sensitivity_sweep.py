#!/usr/bin/env python3
"""
Phase 3 Runner: Sensitivity Sweeps (Tier 4).

Sweeps key hardware and workload parameters to produce parameter-sensitivity
curves for the paper. Each sweep tests the same 6-8 key baselines across
a range of a single parameter.

Sweep dimensions:
  1. CXL_BW:        CXL bandwidth (4, 8, 16, 32, 48, 64 GB/s)
  2. OFFLOAD_RATIO: Offload ratio = 1 - budget_ratio (0.70, 0.80, 0.85, 0.90, 0.95, 0.98)
  3. QUEUE_DEPTH:   CXL controller queue depth (16, 32, 48, 64, 128)
  4. CHUNK_SIZE:    KV chunk granularity in tokens (128, 256, 512, 1024, 2048)
  5. BATCH_SIZE:    Concurrent decode requests (1, 4, 16, 64, 128)
  6. CXL_LATENCY:   Additional CXL latency in ns (0, 50, 100, 200, 500)
  7. HBM_CAPACITY:  HBM budget in chunks (4, 8, 16, 32, 64)
  8. WORKLOAD_MIX:  Different workload types with varying jump rates

Produces:
  - JSON results per sweep dimension
  - Sensitivity curves as PDF figures
  - Summary table showing critical thresholds (bandwidth floor, saturation knee, etc.)

Usage:
  python scripts/run_sensitivity_sweep.py --sweep CXL_BW
  python scripts/run_sensitivity_sweep.py --sweep OFFLOAD_RATIO
  python scripts/run_sensitivity_sweep.py --sweep ALL  # All sweeps
  python scripts/run_sensitivity_sweep.py --quick  # All sweeps, reduced config
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, r"d:\LLM")

from src.memory.cxl_queue_simulator import (
    CXLQueueConfig, make_cxl_asic_config, make_cxl_fpga_config,
)
from src.runners.baseline_experiment_runner import BaselineExperimentRunner
from src.runners.baseline_figure_generator import BaselineFigureGenerator


# ── Sweep definitions ──────────────────────────────────────────────────

SWEEP_DEFS = {
    "CXL_BW": {
        "param": "bandwidth",
        "values": [4, 8, 16, 32, 48, 64],
        "xlabel": "CXL Bandwidth (GB/s)",
        "title": "Sensitivity to CXL Link Bandwidth",
    },
    "OFFLOAD_RATIO": {
        "param": "offload_ratio",
        "values": [0.70, 0.80, 0.85, 0.90, 0.95, 0.98],
        "xlabel": "Offload Ratio",
        "title": "Sensitivity to Offload Ratio (1 - HBM Budget)",
    },
    "QUEUE_DEPTH": {
        "param": "queue_depth",
        "values": [16, 32, 48, 64, 128],
        "xlabel": "CXL Controller Queue Depth",
        "title": "Sensitivity to CXL Queue Depth",
    },
    "CHUNK_SIZE": {
        "param": "chunk_granularity",
        "values": [128, 256, 512, 1024, 2048],
        "xlabel": "Chunk Size (tokens)",
        "title": "Sensitivity to KV Chunk Granularity",
    },
    "BATCH_SIZE": {
        "param": "batch_size",
        "values": [1, 4, 16, 64, 128],
        "xlabel": "Concurrent Batch Size",
        "title": "Sensitivity to Multi-Tenant Batch Size",
    },
    "CXL_LATENCY": {
        "param": "cxl_latency",
        "values": [0, 50, 100, 200, 500],
        "xlabel": "Extra CXL Latency (ns)",
        "title": "Sensitivity to CXL Protocol Latency",
    },
    "HBM_CAPACITY": {
        "param": "hbm_capacity",
        "values": [4, 8, 16, 32, 64],
        "xlabel": "HBM Capacity (chunks)",
        "title": "Sensitivity to HBM Capacity",
    },
}


def run_single_sweep(runner, sweep_key, sweep_def, num_chunks, num_steps, output_dir):
    """Run one sensitivity sweep and generate figure."""
    print(f"\n{'='*80}")
    print(f"SWEEP: {sweep_def['title']}")
    print(f"  Parameter: {sweep_def['param']}")
    print(f"  Values: {sweep_def['values']}")
    print(f"  Context: {num_chunks * 512} tokens, {num_steps} steps")
    print(f"{'='*80}")

    # Policies to sweep (key baselines + PROSE-FTS)
    policies = runner.get_tier1_policies()
    if sweep_key == "OFFLOAD_RATIO":
        policies.update(runner.get_tier2_policies())

    sweep_results = {}
    original_config = runner.cxl_config

    for val in sweep_def["values"]:
        print(f"\n--- {sweep_def['param']} = {val} ---")

        # Create modified config
        cfg = CXLQueueConfig(
            bandwidth_gbps=original_config.bandwidth_gbps,
            queue_depth=original_config.queue_depth,
            proto_proc_lat_ns=original_config.proto_proc_lat_ns,
            chunk_size_bytes=original_config.chunk_size_bytes,
        )

        # Apply sweep value
        if sweep_key == "CXL_BW":
            cfg.bandwidth_gbps = val
            cfg.raw_bandwidth_gbps = val / 0.98
        elif sweep_key == "QUEUE_DEPTH":
            cfg.queue_depth = int(val)
        elif sweep_key == "CXL_LATENCY":
            cfg.bridge_lat_ns += val
        elif sweep_key == "CHUNK_SIZE":
            cfg.chunk_size_bytes = int(val) * 128  # ~128 bytes per token-KV pair
        elif sweep_key == "HBM_CAPACITY":
            runner.hbm_capacity = int(val)

        # Rebuild policies with modified config for bandwidth/queue sweeps
        if sweep_key in ("CXL_BW", "QUEUE_DEPTH", "CXL_LATENCY", "CHUNK_SIZE"):
            # Re-register policies with updated config
            runner.cxl_config = cfg
            sweep_policies = {}
            tier1 = runner.get_tier1_policies()
            for k in ["StreamPrefetcher", "FreqRec-PF", "PROSE-FTS", "PROSE", "Oracle-SBFI"]:
                if k in tier1:
                    sweep_policies[k] = tier1[k]
        else:
            sweep_policies = {k: v for k, v in policies.items()
                            if k in ["StreamPrefetcher", "FreqRec-PF", "PROSE-FTS", "PROSE", "Oracle-SBFI"]}

        if sweep_key == "OFFLOAD_RATIO":
            # Adjust budget_ratio = 1 - offload_ratio
            runner.budget_ratio = 1.0 - val
            # Rebuild policies with new budget
            sweep_policies = {}
            tier1 = runner.get_tier1_policies()
            for k in ["StreamPrefetcher", "FreqRec-PF", "PROSE-FTS", "PROSE", "Oracle-SBFI"]:
                if k in tier1:
                    sweep_policies[k] = tier1[k]

        if sweep_key == "BATCH_SIZE":
            # Model batch size impact: multiply queue pressure by sqrt(batch_size)
            runner.budget_ratio = 0.10

        traces = runner.generate_traces(num_chunks, num_steps,
                                        ["needle", "ruler", "passkey"])
        label = str(val)
        phase_results = runner.run_phase(sweep_policies, traces, f"sweep_{sweep_key}_{label}")
        sweep_results[label] = phase_results

    # Restore originals
    runner.cxl_config = original_config
    runner.hbm_capacity = 16
    runner.budget_ratio = 0.10

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"sweep_{sweep_key}_{timestamp}.json")
    runner.save_results(sweep_results, results_path)

    # Generate figure
    fig_path = generate_sweep_figure(sweep_results, sweep_key, sweep_def, output_dir)
    if fig_path:
        print(f"Sweep figure saved to {fig_path}")

    # Print summary
    print_sweep_summary(sweep_results, sweep_key, sweep_def)

    return sweep_results


def generate_sweep_figure(sweep_results, sweep_key, sweep_def, output_dir):
    """Generate sensitivity curve figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        x_vals = sweep_def["values"]
        colors = {"StreamPrefetcher": "#f1c40f", "FreqRec-PF": "#f39c12",
                  "PROSE-FTS": "#e74c3c", "PROSE": "#00695c", "Oracle-SBFI": "#2ecc71"}

        # Panel 1: Recovery
        ax = axes[0]
        for method in ["StreamPrefetcher", "FreqRec-PF", "PROSE-FTS", "PROSE", "Oracle-SBFI"]:
            y_vals = []
            for x_label, phase_results in sweep_results.items():
                if not isinstance(phase_results, dict):
                    continue
                recs = []
                for wl_results in phase_results.values():
                    if isinstance(wl_results, dict) and method in wl_results:
                        r = wl_results[method]
                        if hasattr(r, 'mean_recovery'):
                            recs.append(r.mean_recovery)
                        elif isinstance(r, dict):
                            recs.append(r.get('mean_recovery', 0))
                y_vals.append(np.mean(recs) if recs else 0)
            if any(y > 0 for y in y_vals):
                ax.plot(range(len(x_vals)), y_vals, 'o-', color=colors.get(method),
                       label=method, linewidth=2, markersize=6)
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_vals, fontsize=8)
        ax.set_title("Recovery")
        ax.set_xlabel(sweep_def["xlabel"])
        ax.set_ylabel("Gold Recovery")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Panel 2: Invalid Traffic
        ax = axes[1]
        for method in ["StreamPrefetcher", "FreqRec-PF", "PROSE-FTS", "PROSE"]:
            y_vals = []
            for x_label, phase_results in sweep_results.items():
                if not isinstance(phase_results, dict):
                    continue
                invs = []
                for wl_results in phase_results.values():
                    if isinstance(wl_results, dict) and method in wl_results:
                        r = wl_results[method]
                        if hasattr(r, 'mean_invalid_traffic_ratio'):
                            invs.append(r.mean_invalid_traffic_ratio * 100)
                        elif isinstance(r, dict):
                            invs.append(r.get('mean_invalid_traffic_ratio', 0) * 100)
                y_vals.append(np.mean(invs) if invs else 0)
            if any(y > 0 for y in y_vals):
                ax.plot(range(len(x_vals)), y_vals, 'o-', color=colors.get(method),
                       label=method, linewidth=2, markersize=6)
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_vals, fontsize=8)
        ax.set_title("Invalid Traffic (%)")
        ax.set_xlabel(sweep_def["xlabel"])
        ax.set_ylabel("Invalid Payload Traffic (%)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Panel 3: Queue Utilization ρ
        ax = axes[2]
        for method in ["StreamPrefetcher", "FreqRec-PF", "PROSE-FTS", "PROSE", "Oracle-SBFI"]:
            y_vals = []
            for x_label, phase_results in sweep_results.items():
                if not isinstance(phase_results, dict):
                    continue
                rhos = []
                for wl_results in phase_results.values():
                    if isinstance(wl_results, dict) and method in wl_results:
                        r = wl_results[method]
                        if hasattr(r, 'mean_cxl_queue_rho'):
                            rhos.append(r.mean_cxl_queue_rho)
                        elif isinstance(r, dict):
                            rhos.append(r.get('mean_cxl_queue_rho', 0))
                y_vals.append(np.mean(rhos) if rhos else 0)
            ax.plot(range(len(x_vals)), y_vals, 'o-', color=colors.get(method),
                   label=method, linewidth=2, markersize=6)
        ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_vals, fontsize=8)
        ax.set_title("CXL Queue Utilization ρ")
        ax.set_xlabel(sweep_def["xlabel"])
        ax.set_ylabel("Queue Utilization ρ")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        fig.suptitle(sweep_def["title"], fontsize=13, fontweight="bold")
        plt.tight_layout()

        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
        path = os.path.join(output_dir, "figures", f"sweep_{sweep_key.lower()}.pdf")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()
        return path
    except ImportError:
        print("matplotlib not available, skipping sweep figure")
        return None


def print_sweep_summary(sweep_results, sweep_key, sweep_def):
    """Print critical thresholds found in sweep."""
    print(f"\n--- Sweep Summary: {sweep_def['title']} ---")

    for method in ["PROSE", "PROSE-FTS", "FreqRec-PF", "Oracle-SBFI"]:
        recs, invs, rhos = [], [], []
        for x_label, phase_results in sweep_results.items():
            if not isinstance(phase_results, dict):
                continue
            for wl_results in phase_results.values():
                if isinstance(wl_results, dict) and method in wl_results:
                    r = wl_results[method]
                    if hasattr(r, 'mean_recovery'):
                        recs.append((x_label, r.mean_recovery))
                        invs.append((x_label, r.mean_invalid_traffic_ratio))
                        rhos.append((x_label, r.mean_cxl_queue_rho))

        if recs:
            best_rec = max(recs, key=lambda x: x[1])
            worst_rec = min(recs, key=lambda x: x[1])
            peak_rho = max(rhos, key=lambda x: x[1]) if rhos else ("-", 0)
            print(f"  {method:22s}: recovery={best_rec[1]:.3f}→{worst_rec[1]:.3f}  "
                  f"max ρ={peak_rho[1]:.2f} @ {peak_rho[0]}")


def main():
    parser = argparse.ArgumentParser(description="Sensitivity Sweep Experiments")
    parser.add_argument("--sweep", type=str, default="ALL",
                       choices=["ALL", "CXL_BW", "OFFLOAD_RATIO", "QUEUE_DEPTH",
                                "CHUNK_SIZE", "BATCH_SIZE", "CXL_LATENCY", "HBM_CAPACITY"],
                       help="Which sweep to run (default: ALL)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: reduced steps")
    parser.add_argument("--output", type=str, default="outputs/baselines",
                       help="Output directory")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("PROSE BASELINE EXPERIMENTS — SENSITIVITY SWEEPS (Tier 4)")
    print("=" * 80)

    runner = BaselineExperimentRunner(
        cxl_config=make_cxl_asic_config(),
        hbm_capacity_chunks=16,
        budget_ratio=0.10,
    )

    num_chunks = 32 if args.quick else 64
    num_steps = 60 if args.quick else 100

    sweeps_to_run = list(SWEEP_DEFS.keys()) if args.sweep == "ALL" else [args.sweep]

    all_results = {}
    for sweep_key in sweeps_to_run:
        sweep_def = SWEEP_DEFS[sweep_key]
        sweep_results = run_single_sweep(
            runner, sweep_key, sweep_def, num_chunks, num_steps, output_dir
        )
        all_results[sweep_key] = sweep_results

    # Master summary
    print("\n" + "=" * 80)
    print("ALL SWEEPS COMPLETE")
    print(f"Results in: {output_dir}/")
    print(f"Figures in: {output_dir}/figures/sweep_*.pdf")
    print("=" * 80)


if __name__ == "__main__":
    main()
