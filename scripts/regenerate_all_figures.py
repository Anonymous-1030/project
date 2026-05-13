#!/usr/bin/env python3
"""
Regenerate all three HPCA figures after bug fixes.

Fix 1: Oracle timeline — now uses trace[step] (current step) not trace[step+1] (future)
Fix 2: ρ calculation — step-level bandwidth utilization, not inverted service/(service+queuing)
Fix 3: M/D/1 queuing — computed once per step, not per-`_submit` with artificial inter-arrival

Runs:
  1. OFFLOAD_RATIO sweep for Figure 2 (multi-point queue ρ)
  2. Multi-context Tier 1+2 run for Figure 1 (invalid traffic vs context) and Figure 3

Usage:
  python scripts/regenerate_all_figures.py
  python scripts/regenerate_all_figures.py --quick  # Reduced config
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, r"d:\LLM")

from src.memory.cxl_queue_simulator import (
    CXLQueueConfig, make_cxl_asic_config,
)
from src.runners.baseline_experiment_runner import BaselineExperimentRunner
from src.runners.baseline_figure_generator import BaselineFigureGenerator


def run_offload_sweep(runner, output_dir, quick=False):
    """Run OFFLOAD_RATIO sweep for Figure 2."""
    print("\n" + "=" * 80)
    print("RUNNING OFFLOAD_RATIO SWEEP (for Figure 2)")
    print("=" * 80)

    offload_values = [0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
    num_chunks = 32 if quick else 64
    num_steps = 40 if quick else 60

    sweep_results = {}
    original_budget = runner.budget_ratio

    for val in offload_values:
        print(f"\n--- offload_ratio = {val} ---")
        runner.budget_ratio = 1.0 - val

        policies = {}
        tier1 = runner.get_tier1_policies()
        for k in ["StreamPrefetcher", "FreqRec-PF", "PROSE-FTS", "Oracle-SBFI"]:
            if k in tier1:
                policies[k] = tier1[k]
        # Also include Oracle-FTS for Figure 2
        if "Oracle-FTS" in tier1:
            policies["Oracle-FTS"] = tier1["Oracle-FTS"]

        traces = runner.generate_traces(num_chunks, num_steps,
                                        ["needle", "ruler", "passkey"])
        label = str(val)
        phase_results = runner.run_phase(policies, traces, f"sweep_offload_{label}")
        sweep_results[label] = phase_results

    runner.budget_ratio = original_budget

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"sweep_OFFLOAD_RATIO_{timestamp}.json")
    runner.save_results(sweep_results, results_path)
    print(f"\nOffload sweep results saved to {results_path}")
    return sweep_results


def run_multi_context(runner, output_dir, quick=False):
    """Run Tier 1 + Tier 2 baselines across multiple context lengths.

    Returns results keyed by phase name for Figure 1 and Figure 3.
    """
    print("\n" + "=" * 80)
    print("RUNNING MULTI-CONTEXT TIER 1+2 (for Figures 1 & 3)")
    print("=" * 80)

    context_tokens = [8192, 16384, 32768, 65536, 131072]
    num_steps = 40 if quick else 60

    all_phase_results = {}

    for ctx_tokens in context_tokens:
        num_chunks = ctx_tokens // 512
        phase_name = f"context_{ctx_tokens}"
        print(f"\n--- Context: {ctx_tokens} tokens ({num_chunks} chunks) ---")

        # Get policies with fresh state
        tier1 = runner.get_tier1_policies()
        tier2 = runner.get_tier2_policies()
        tier3 = runner.get_tier3_policies()

        policies = {}
        policies.update(tier1)
        policies.update(tier2)
        # Keep key ablations
        for k in ["PROSE-NoPHT", "PROSE-NoPBuffer", "PROSE-NoVersionGate",
                   "PROSE-FIFOVictim", "PROSE-SingleCue(temporal)"]:
            if k in tier3:
                policies[k] = tier3[k]

        traces = runner.generate_traces(num_chunks, num_steps,
                                        ["passkey", "needle", "ruler"])

        # Run with adjusted budget_ratio for proportional budget
        original_budget = runner.budget_ratio
        phase_results = runner.run_phase(policies, traces, phase_name)
        runner.budget_ratio = original_budget
        all_phase_results[phase_name] = phase_results

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"multi_context_all_{timestamp}.json")
    runner.save_results(all_phase_results, results_path)
    print(f"\nMulti-context results saved to {results_path}")
    return all_phase_results


def main():
    parser = argparse.ArgumentParser(description="Regenerate all HPCA figures")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: reduced steps/chunks")
    parser.add_argument("--output", type=str, default="outputs/baselines",
                       help="Output directory")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    runner = BaselineExperimentRunner(
        cxl_config=make_cxl_asic_config(),
        hbm_capacity_chunks=16,
        budget_ratio=0.10,
    )

    # ── Step 1: Offload sweep for Figure 2 ──
    sweep_results = run_offload_sweep(runner, output_dir, quick=args.quick)

    # ── Step 2: Multi-context run for Figures 1 & 3 ──
    multi_results = run_multi_context(runner, output_dir, quick=args.quick)

    # ── Step 3: Generate all four figures ──
    print("\n" + "=" * 80)
    print("GENERATING ALL FOUR FIGURES")
    print("=" * 80)

    # Combine results for the figure generator
    combined = {}
    combined.update(multi_results)

    gen = BaselineFigureGenerator(combined, output_dir=os.path.join(output_dir, "figures"))
    gen.load_offload_sweep(sweep_results)

    context_lengths = [8192, 16384, 32768, 65536, 131072]
    offload_ratios = [0.70, 0.80, 0.85, 0.90, 0.95, 0.98]

    fig1_path = gen.generate_figure1_invalid_traffic(context_lengths)
    fig2_path = gen.generate_figure2_queue_utilization(offload_ratios)
    fig3_path = gen.generate_figure3_metric_hierarchy()
    swp_path = gen.generate_sweep_offload_ratio(offload_ratios)

    print(f"\nFigure 1:     {fig1_path}")
    print(f"Figure 2:     {fig2_path}")
    print(f"Figure 3:     {fig3_path}")
    print(f"Sweep:        {swp_path}")
    print("\nDone — all figures regenerated.")


if __name__ == "__main__":
    main()
