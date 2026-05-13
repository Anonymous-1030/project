#!/usr/bin/env python3
"""
Phase 2 Runner: Tier 2 Related-Work Positioning Baselines.

Runs the software/KV-reduction baselines extended to CXL memory:
  6.  FreqRec-PF+Meta: FreqRec with 64B summary access, fetch-then-score
  7.  H2O-CXL: Heavy-Hitter Oracle + blind CXL paging
  8.  SnapKV-CXL: Observation-window selection + blind CXL offloading
  9.  InfLLM-CXL: Retrieval-based KV + fetch-then-decide on CXL
  10. CUDA-UM: CUDA Unified Memory + cudaMemPrefetchAsync
  11. Oracle-Candidate: Oracle exposure + ODUS-X ranking (separates exposure vs ranking)

Also re-runs key Tier 1 baselines for side-by-side comparison:
  - PROSE-FTS: Core ordering ablation
  - FreqRec-PF: Strongest pure-hardware baseline
  - Oracle-SBFI: Upper bound

Produces:
  - JSON results: outputs/baselines/phase2_results.json
  - Console comparison table
  - Combined figures with Tier 1 + Tier 2

Usage:
  python scripts/run_tier2_baselines.py
  python scripts/run_tier2_baselines.py --quick
  python scripts/run_tier2_baselines.py --full
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, r"d:\LLM")

from src.memory.cxl_queue_simulator import make_cxl_asic_config
from src.runners.baseline_experiment_runner import BaselineExperimentRunner
from src.runners.baseline_figure_generator import BaselineFigureGenerator


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Baseline Experiments")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test: 32 chunks, 60 steps")
    parser.add_argument("--full", action="store_true",
                       help="Full sweep: multiple context lengths")
    parser.add_argument("--budget", type=float, default=0.10,
                       help="HBM budget ratio (default: 0.10)")
    parser.add_argument("--output", type=str, default="outputs/baselines",
                       help="Output directory")
    parser.add_argument("--compare-tier1", action="store_true", default=True,
                       help="Include Tier 1 baselines for comparison (default: True)")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("PROSE BASELINE EXPERIMENTS — PHASE 2 (Tier 2: Related Work)")
    print("=" * 80)

    cxl_config = make_cxl_asic_config()
    print(f"CXL Config: {cxl_config.bandwidth_gbps} GB/s, queue_depth={cxl_config.queue_depth}")
    print(f"HBM Budget: {args.budget:.0%}")

    runner = BaselineExperimentRunner(
        cxl_config=cxl_config,
        hbm_capacity_chunks=16,
        budget_ratio=args.budget,
    )

    # Select policies
    tier2_policies = runner.get_tier2_policies()

    if args.compare_tier1:
        tier1_key = runner.get_tier1_policies()
        # Keep only key comparison baselines
        for key in ["PROSE-FTS", "FreqRec-PF", "Oracle-SBFI"]:
            if key in tier1_key:
                tier2_policies[key] = tier1_key[key]

    print(f"\nPolicies to evaluate: {list(tier2_policies.keys())}")

    results = {}

    if args.full:
        print("\nRunning FULL sweep across context lengths...")
        configs = [
            (32, "16K"), (64, "32K"), (128, "64K"), (256, "128K"),
        ]
        for num_chunks, label in configs:
            print(f"\n{'='*60}")
            print(f"  Context: {label} ({num_chunks} chunks)")
            print(f"{'='*60}")
            phase_results = runner.run_phase(tier2_policies,
                runner.generate_traces(num_chunks, 200,
                    ["passkey", "needle", "sequential", "ruler"]),
                f"tier2_{label}")
            results[label] = phase_results
    else:
        num_chunks = 32 if args.quick else 64
        num_steps = 60 if args.quick else 200
        print(f"\nRunning Phase 2: {num_chunks} chunks, {num_steps} steps")
        traces = runner.generate_traces(num_chunks, num_steps,
                                        ["passkey", "needle", "sequential", "ruler"])
        results = runner.run_phase(tier2_policies, traces, "phase2")

    # Print results
    if not args.full:
        runner.print_comparison_table(results)
        runner.print_detailed_breakdown(results)

    # SBFI superiority check
    print("\n" + "=" * 80)
    print("SBFI vs FTS COMPARISON (Tier 2 Context)")
    print("=" * 80)
    for label, phase_results in (results.items() if args.full else [("phase2", results)]):
        if not isinstance(phase_results, dict):
            continue
        for wl_name, wl_results in phase_results.items():
            if not isinstance(wl_results, dict):
                continue
            # Find SBFI and FTS methods
            sbfi = {m: r for m, r in wl_results.items() if "SBFI" in m}
            fts = {m: r for m, r in wl_results.items()
                   if ("FTS" in m or "FreqRec" in m or "H2O" in m
                       or "SnapKV" in m or "InfLLM" in m or "vLLM" in m
                       or "CUDA" in m or "Stream" in m)}
            for s_name, s_r in sbfi.items():
                s_rec = s_r.mean_recovery if hasattr(s_r, 'mean_recovery') else s_r.get('mean_recovery', 0)
                s_inv = s_r.mean_invalid_traffic_ratio if hasattr(s_r, 'mean_invalid_traffic_ratio') else s_r.get('mean_invalid_traffic_ratio', 0)
                for f_name, f_r in fts.items():
                    f_rec = f_r.mean_recovery if hasattr(f_r, 'mean_recovery') else f_r.get('mean_recovery', 0)
                    f_inv = f_r.mean_invalid_traffic_ratio if hasattr(f_r, 'mean_invalid_traffic_ratio') else f_r.get('mean_invalid_traffic_ratio', 0)
                    if f_inv > 0.01:
                        print(f"  [{wl_name:12s}] {s_name:22s} vs {f_name:22s}: "
                              f"Δrec={s_rec-f_rec:+.3f}  Δinvalid={(f_inv-s_inv)*100:+.1f}%")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"phase2_results_{timestamp}.json")
    runner.save_results(results, results_path)
    latest_path = os.path.join(output_dir, "phase2_latest.json")
    runner.save_results(results, latest_path)

    # Generate figures
    print("\n" + "=" * 80)
    print("GENERATING COMBINED FIGURES (Tier 1 + Tier 2)")
    print("=" * 80)

    fig_gen = BaselineFigureGenerator(
        results,
        output_dir=os.path.join(output_dir, "figures"),
    )
    fig_gen.generate_all()

    # Figure data
    fig_data = runner.generate_figure_data(results)
    fig_data_path = os.path.join(output_dir, "figure_data_phase2.json")
    with open(fig_data_path, "w") as f:
        json.dump(fig_data, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE")
    print(f"Results: {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
