#!/usr/bin/env python3
"""
Phase 1 Runner: Tier 1 Must-Have Baselines + Tier 3 Key Ablations.

Runs the critical baseline experiments that support the paper's core claims:
  1. FreqRec-PF: Strongest fetch-then-score history baseline
  2. PROSE-FTS: Core ordering ablation (SBFI isolates from ranking)
  3. StreamPrefetcher: Simple hardware prefetcher (lower bound)
  4. Oracle-Policy: Perfect utility upper bound with CXL constraints
  5. vLLM-CXL: Industry-standard software tiering approach
  12. PROSE-NoPHT: Fast path coverage ablation
  13. PROSE-SingleCue: Multi-cue fusion necessity
  14. PROSE-NoPBuffer: Speculative-safe enactment necessity

Produces:
  - JSON results file: outputs/baselines/phase1_results.json
  - Console comparison table
  - Three key figures: outputs/baselines/figures/

Usage:
  python scripts/run_tier1_baselines.py
  python scripts/run_tier1_baselines.py --quick       # Fast test
  python scripts/run_tier1_baselines.py --full         # Full sweep
  python scripts/run_tier1_baselines.py --sweep CXL_BW # Sensitivity sweep
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, r"d:\LLM")

from src.memory.cxl_queue_simulator import (
    make_cxl_asic_config, make_cxl_cxl20_config, make_cxl_fpga_config,
)
from src.runners.baseline_experiment_runner import BaselineExperimentRunner
from src.runners.baseline_figure_generator import BaselineFigureGenerator


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Baseline Experiments")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test: 32 chunks, 60 steps")
    parser.add_argument("--full", action="store_true",
                       help="Full sweep: multiple context lengths")
    parser.add_argument("--sweep", type=str, default=None,
                       choices=["CXL_BW", "QUEUE_DEPTH", "OFFLOAD_RATIO", "CHUNK_SIZE", "BATCH_SIZE"],
                       help="Run sensitivity sweep")
    parser.add_argument("--budget", type=float, default=0.10,
                       help="HBM budget ratio (default: 0.10)")
    parser.add_argument("--output", type=str, default="outputs/baselines",
                       help="Output directory")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("PROSE BASELINE EXPERIMENTS — PHASE 1 (Tier 1 + Key Ablations)")
    print("=" * 80)

    cxl_config = make_cxl_asic_config()
    print(f"CXL Config: {cxl_config.bandwidth_gbps} GB/s, queue_depth={cxl_config.queue_depth}")
    print(f"HBM Budget: {args.budget:.0%}")

    runner = BaselineExperimentRunner(
        cxl_config=cxl_config,
        hbm_capacity_chunks=16,
        budget_ratio=args.budget,
    )

    results = {}

    if args.sweep:
        # ── Sensitivity sweep ──
        print(f"\nRunning sensitivity sweep: {args.sweep}")
        sweep_results = runner.run_sensitivity_sweep(
            sweep_param=args.sweep.lower().replace("cxl_bw", "bandwidth")
                         .replace("queue_depth", "queue_depth")
                         .replace("offload_ratio", "offload_ratio")
                         .replace("chunk_size", "chunk_granularity")
                         .replace("batch_size", "batch_size"),
            num_chunks=64 if not args.quick else 32,
            num_steps=150 if not args.quick else 60,
        )
        results["sweep"] = sweep_results

    elif args.full:
        # ── Full sweep: multiple context lengths ──
        print("\nRunning FULL sweep across context lengths...")
        configs = [
            (32, "16K"), (64, "32K"), (128, "64K"), (256, "128K"),
        ]
        for num_chunks, label in configs:
            print(f"\n{'='*60}")
            print(f"  Context: {label} ({num_chunks} chunks)")
            print(f"{'='*60}")
            phase_results = runner.run_phase1(
                num_chunks=num_chunks,
                num_steps=200,
            )
            results[label] = phase_results

    else:
        # ── Default: 64-chunk run (32K context) ──
        num_chunks = 32 if args.quick else 64
        num_steps = 60 if args.quick else 200
        print(f"\nRunning Phase 1: {num_chunks} chunks, {num_steps} steps")
        results = runner.run_phase1(
            num_chunks=num_chunks,
            num_steps=num_steps,
        )

    # ── Print results ──
    if not args.sweep:
        runner.print_comparison_table(results)
        runner.print_detailed_breakdown(results)

    # Print SBFI advantage summary
    print("\n" + "=" * 80)
    print("SBFI (SCORE-BEFORE-FETCH) ADVANTAGE SUMMARY")
    print("=" * 80)

    def _get_attr(r, name, default=0.0):
        if hasattr(r, name):
            return getattr(r, name, default)
        if isinstance(r, dict):
            return r.get(name, default)
        return default

    def iter_workload_results(res):
        """Yield (label, workload_name, method_results_dict) for both nesting levels."""
        for k1, v1 in res.items():
            if not isinstance(v1, dict):
                continue
            # Peek: is v1 {method: result} or {workload: {method: result}}?
            peek = next(iter(v1.values()), None) if v1 else None
            if isinstance(peek, dict):
                # Check if peek looks like a result (has "method" or "mean_recovery")
                if isinstance(peek, dict) and any(k in peek for k in ["method", "mean_recovery", "workload"]):
                    # 2-level: {workload: {method: result}} with dict results (from JSON)
                    for wl_name, wl_body in v1.items():
                        if isinstance(wl_body, dict):
                            yield k1, wl_name, wl_body
                else:
                    # 3-level: {phase: {workload: {method: result}}}
                    for wl_name, wl_body in v1.items():
                        if isinstance(wl_body, dict):
                            yield k1, wl_name, wl_body
            elif hasattr(peek, 'mean_recovery') or hasattr(peek, 'to_dict'):
                # 2-level: {workload: {method: BaselineResult}} — treated as single phase
                yield k1, k1, v1
                break

    for label, wl_name, wl_results in iter_workload_results(results):
        if not isinstance(wl_results, dict):
            continue
        print(f"\n--- {label} / {wl_name} ---")
        fts_methods = [m for m in wl_results if ("FTS" in m or "FreqRec" in m or "Stream" in m or "vLLM" in m)]
        sbfi_methods = [m for m in wl_results if "SBFI" in m and "FTS" not in m]

        for sbfi_m in sbfi_methods:
            sbfi_rec = _get_attr(wl_results[sbfi_m], 'mean_recovery')
            sbfi_inv = _get_attr(wl_results[sbfi_m], 'mean_invalid_traffic_ratio')
            for fts_m in fts_methods:
                fts_rec = _get_attr(wl_results[fts_m], 'mean_recovery')
                fts_inv = _get_attr(wl_results[fts_m], 'mean_invalid_traffic_ratio')
                rec_gap = sbfi_rec - fts_rec
                inv_gap = fts_inv - sbfi_inv
                if inv_gap > 0.01 or abs(rec_gap) > 0.01:  # Show all meaningful comparisons
                    print(f"  {sbfi_m:22s} vs {fts_m:22s}: "
                          f"Δrec={rec_gap:+.3f}  Δinvalid={inv_gap*100:+.1f}%")

    # ── Save results ──
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"phase1_results_{timestamp}.json")
    runner.save_results(results, results_path)

    # Also save latest
    latest_path = os.path.join(output_dir, "phase1_latest.json")
    runner.save_results(results, latest_path)

    # ── Generate figures ──
    print("\n" + "=" * 80)
    print("GENERATING THREE KEY FIGURES")
    print("=" * 80)

    fig_gen = BaselineFigureGenerator(
        results,
        output_dir=os.path.join(output_dir, "figures"),
    )
    fig_gen.generate_all()

    # Save figure data
    fig_data = runner.generate_figure_data(results)
    fig_data_path = os.path.join(output_dir, "figure_data.json")
    with open(fig_data_path, "w") as f:
        json.dump(fig_data, f, indent=2, default=str)
    print(f"Figure data saved to {fig_data_path}")

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print(f"Results: {results_path}")
    print(f"Figures: {os.path.join(output_dir, 'figures')}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
