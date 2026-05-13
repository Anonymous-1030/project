"""
SimCXL + PROSE Co-Simulation Experiment Runner
==============================================

Runs the full experiment matrix comparing baseline (CXL Type-3 without PROSE)
against PROSE-enabled configurations in a unified cycle-accurate SimCXL simulator.

Experiment Matrix:
  Configs: baseline, prose_pht, prose_ppu, prose_full
  Seq lengths (num_chunks): 1024, 2048, 4096, 8192, 16384
  Batch sizes (active chunks/step): 8, 16, 32

Output: JSON result files + matplotlib figures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure prose_v2 is importable
_SCRIPT_DIR = str(Path(__file__).parent)
_PROSE_ROOT = str(Path(__file__).parent.parent)
_WORKSPACE = str(Path(__file__).parent.parent.parent)
for p in (_PROSE_ROOT, _WORKSPACE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from src.hardware_model.simcxl_cosim import (
    CoSimStats,
    SimCXLCoSimulator,
    SimCXLTiming,
    run_comparison_experiment,
)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    name: str
    num_chunks: int = 4096
    num_steps: int = 256
    chunk_size_bytes: int = 65536
    fast_tier_capacity_mb: float = 512.0
    promotion_budget: int = 5
    active_chunks_per_step: int = 16
    seed: int = 42
    description: str = ""


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single_experiment(
    config: ExperimentConfig,
    mode: str = "prose",
) -> Dict[str, Any]:
    """Run one experiment configuration, return stats as dict."""
    sim = SimCXLCoSimulator(mode=mode)
    sim.configure_workload(
        num_chunks=config.num_chunks,
        chunk_size_bytes=config.chunk_size_bytes,
        fast_tier_capacity_bytes=int(config.fast_tier_capacity_mb * 1024 * 1024),
        initial_fast_tier_chunks=min(
            32, int(config.fast_tier_capacity_mb * 1024 * 1024 // config.chunk_size_bytes)
        ),
    )

    rng = np.random.RandomState(config.seed)
    for step in range(config.num_steps):
        n_active = config.active_chunks_per_step
        chunk_ids = list(rng.zipf(1.5, size=n_active * 2))
        chunk_ids = [c % config.num_chunks for c in chunk_ids
                     if c < config.num_chunks]
        chunk_ids = list(dict.fromkeys(chunk_ids))[:n_active]
        if len(chunk_ids) < n_active:
            extras = [rng.randint(0, config.num_chunks)
                      for _ in range(n_active - len(chunk_ids))]
            chunk_ids.extend(extras)

        masses = {cid: float(rng.beta(0.5, 5.0)) for cid in chunk_ids}
        sim.run_decode_step(
            step, active_chunk_ids=chunk_ids,
            attention_masses=masses,
            promotion_budget=config.promotion_budget,
        )

    stats = sim.get_stats()
    result = stats.to_dict()
    result['config'] = {
        'mode': mode,
        'num_chunks': config.num_chunks,
        'num_steps': config.num_steps,
        'fast_tier_capacity_mb': config.fast_tier_capacity_mb,
        'promotion_budget': config.promotion_budget,
        'active_chunks_per_step': config.active_chunks_per_step,
    }

    # Compute summary metrics
    total_accesses = stats.fast_tier_hits + stats.fast_tier_misses
    result['summary'] = {
        'fast_tier_hit_rate': stats.fast_tier_hits / max(1, total_accesses),
        'pht_hit_rate': stats.pht_hits / max(1, stats.pht_queries) if stats.pht_queries > 0 else 0,
        'avg_req_queue_depth': float(np.mean(stats.req_queue_len_samples)) if stats.req_queue_len_samples else 0,
        'avg_rsp_queue_depth': float(np.mean(stats.rsp_queue_len_samples)) if stats.rsp_queue_len_samples else 0,
        'p95_req_queue_depth': float(np.percentile(stats.req_queue_len_samples, 95)) if stats.req_queue_len_samples else 0,
        'p95_rsp_queue_depth': float(np.percentile(stats.rsp_queue_len_samples, 95)) if stats.rsp_queue_len_samples else 0,
        'queue_full_events': stats.req_que_full_events + stats.rsp_que_full_events,
        'total_dma_promotions': stats.dma_promotions,
        'total_dma_bytes': stats.dma_bytes_transferred,
        'cxl_upstream_util': stats.cxl_upstream_utilization,
        'cxl_downstream_util': stats.cxl_downstream_utilization,
        'dram_row_hit_rate': stats.dram_row_hits / max(1, stats.dram_row_hits + stats.dram_row_misses),
    }

    return result


# ---------------------------------------------------------------------------
# Full experiment matrix
# ---------------------------------------------------------------------------

def run_experiment_matrix(
    output_dir: str = "outputs/simcxl_cosim",
    quick: bool = False,
) -> Dict[str, Any]:
    """Run the full experiment matrix and save results."""

    modes = ['baseline', 'prose']
    seq_lengths = [1024, 4096, 16384] if quick else [1024, 2048, 4096, 8192, 16384]
    active_chunks_list = [8, 16, 32]

    os.makedirs(output_dir, exist_ok=True)

    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'SimCXL + PROSE co-simulation experiment matrix',
        'parameters': {
            'modes': modes,
            'seq_lengths': seq_lengths,
            'active_chunks_per_step': active_chunks_list,
            'simcxl_params': {k: v for k, v in SimCXLTiming().__dict__.items()},
        },
        'experiments': [],
        'summary_table': [],
    }

    total_experiments = len(modes) * len(seq_lengths) * len(active_chunks_list)
    completed = 0

    for num_chunks in seq_lengths:
        for active_chunks in active_chunks_list:
            for mode in modes:
                config = ExperimentConfig(
                    name=f"{mode}_chunks{num_chunks}_active{active_chunks}",
                    num_chunks=num_chunks,
                    num_steps=64 if quick else 256,
                    active_chunks_per_step=active_chunks,
                    fast_tier_capacity_mb=min(2048.0, num_chunks * 64 / (1024 * 1024) * 8),
                    promotion_budget=min(active_chunks, 8),
                    description=f"mode={mode} seq={num_chunks}chunks batch={active_chunks}",
                )

                print(f"[{completed+1}/{total_experiments}] {config.description}")

                try:
                    result = run_single_experiment(config, mode=mode)
                    result['config']['seq_chunks'] = num_chunks
                    result['config']['active_chunks'] = active_chunks
                    all_results['experiments'].append(result)

                    all_results['summary_table'].append({
                        'mode': mode,
                        'num_chunks': num_chunks,
                        'active_chunks': active_chunks,
                        'fast_tier_hit_rate': result['summary']['fast_tier_hit_rate'],
                        'pht_hit_rate': result['summary']['pht_hit_rate'],
                        'p95_req_q': result['summary']['p95_req_queue_depth'],
                        'p95_rsp_q': result['summary']['p95_rsp_queue_depth'],
                        'queue_full': result['summary']['queue_full_events'],
                        'dma_promotions': result['summary']['total_dma_promotions'],
                        'cxl_up_util': result['summary']['cxl_upstream_util'],
                    })
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

                completed += 1

    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'simcxl_cosim_results_{timestamp}.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    header = (f"{'Mode':>8s} | {'Chunks':>8s} | {'Active':>6s} | "
              f"{'FastHit%':>9s} | {'PHTHit%':>8s} | "
              f"{'P95ReqQ':>7s} | {'P95RspQ':>7s} | "
              f"{'QFull':>6s} | {'DMAProm':>7s}")
    print(header)
    print("-" * len(header))
    for row in all_results['summary_table']:
        print(f"{row['mode']:>8s} | {row['num_chunks']:>8d} | {row['active_chunks']:>6d} | "
              f"{row['fast_tier_hit_rate']*100:>8.1f}% | {row['pht_hit_rate']*100:>7.1f}% | "
              f"{row['p95_req_q']:>7.1f} | {row['p95_rsp_q']:>7.1f} | "
              f"{row['queue_full']:>6d} | {row['dma_promotions']:>7d}")

    return all_results


# ---------------------------------------------------------------------------
# Comparison analysis
# ---------------------------------------------------------------------------

def analyze_comparison(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze differences between baseline and PROSE across experiments."""
    analysis = {'baseline_vs_prose': []}

    exps = results.get('experiments', [])
    for exp in exps:
        cfg = exp.get('config', {})
        if cfg.get('mode') == 'baseline':
            # Find matching PROSE experiment
            for exp2 in exps:
                cfg2 = exp2.get('config', {})
                if (cfg2.get('mode') == 'prose' and
                    cfg2.get('seq_chunks') == cfg.get('seq_chunks') and
                    cfg2.get('active_chunks') == cfg.get('active_chunks')):
                    baseline_sum = exp['summary']
                    prose_sum = exp2['summary']
                    analysis['baseline_vs_prose'].append({
                        'num_chunks': cfg.get('seq_chunks'),
                        'active_chunks': cfg.get('active_chunks'),
                        'hit_rate_delta': prose_sum['fast_tier_hit_rate'] - baseline_sum['fast_tier_hit_rate'],
                        'hit_rate_improvement_pct': (
                            (prose_sum['fast_tier_hit_rate'] - baseline_sum['fast_tier_hit_rate'])
                            / max(1e-6, baseline_sum['fast_tier_hit_rate']) * 100
                        ),
                        'req_q_p95_delta': prose_sum['p95_req_queue_depth'] - baseline_sum['p95_req_queue_depth'],
                        'rsp_q_p95_delta': prose_sum['p95_rsp_queue_depth'] - baseline_sum['p95_rsp_queue_depth'],
                        'queue_full_delta': prose_sum['queue_full_events'] - baseline_sum['queue_full_events'],
                        'baseline_cxl_util': baseline_sum['cxl_upstream_util'],
                        'prose_cxl_util': prose_sum['cxl_upstream_util'],
                    })
                    break

    return analysis


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figures(results: Dict[str, Any], output_dir: str = "outputs/simcxl_cosim"):
    """Generate comparison figures from experiment results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figure generation")
        return

    exps = results.get('experiments', [])
    if not exps:
        return

    # Extract data
    seq_lengths = sorted(set(e['config'].get('seq_chunks', 0) for e in exps))
    modes = sorted(set(e['config'].get('mode', '') for e in exps))

    # Pre-allocate data structures
    data = {mode: {sl: {} for sl in seq_lengths} for mode in modes}
    for exp in exps:
        cfg = exp['config']
        sl = cfg.get('seq_chunks', 0)
        mode = cfg.get('mode', '')
        ac = cfg.get('active_chunks', 16)
        if mode in data and sl in data[mode]:
            data[mode][sl][ac] = exp['summary']

    # Figure 1: Fast tier hit rate vs sequence length
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric_key in enumerate(['fast_tier_hit_rate', 'pht_hit_rate', 'cxl_upstream_util']):
        ax = axes[i]
        metric_label = {
            'fast_tier_hit_rate': 'Fast Tier Hit Rate',
            'pht_hit_rate': 'PHT Prediction Hit Rate',
            'cxl_upstream_util': 'CXL Upstream Utilization',
        }[metric_key]

        for mode in modes:
            values = []
            for sl in seq_lengths:
                mode_data = data[mode][sl]
                vals = [v[metric_key] for v in mode_data.values() if v]
                values.append(np.mean(vals) if vals else 0)
            ax.plot(seq_lengths, values, 'o-', label=mode, linewidth=2, markersize=8)

        ax.set_xlabel('Sequence Length (num chunks)')
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'fig1_fast_tier_hit_rate.pdf')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {fig_path}")

    # Figure 2: Queue depth comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (metric_key, label) in enumerate([
        ('p95_req_queue_depth', 'P95 Request Queue Depth'),
        ('p95_rsp_queue_depth', 'P95 Response Queue Depth'),
    ]):
        ax = axes[i]
        for mode in modes:
            values = []
            for sl in seq_lengths:
                mode_data = data[mode][sl]
                vals = [v[metric_key] for v in mode_data.values() if v]
                values.append(np.mean(vals) if vals else 0)
            ax.plot(seq_lengths, values, 'o-', label=mode, linewidth=2, markersize=8)

        # Horizontal line for queue depth limit
        ax.axhline(y=48, color='red', linestyle='--', alpha=0.5, label='Queue Limit (48)')
        ax.set_xlabel('Sequence Length (num chunks)')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'fig2_queue_depth.pdf')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {fig_path}")

    # Figure 3: PROSE advantage (hit rate improvement vs baseline)
    analysis = analyze_comparison(results)
    comparisons = analysis.get('baseline_vs_prose', [])

    fig, ax = plt.subplots(figsize=(8, 5))
    for ac in sorted(set(c['active_chunks'] for c in comparisons)):
        pts = [c for c in comparisons if c['active_chunks'] == ac]
        pts.sort(key=lambda x: x['num_chunks'])
        x = [p['num_chunks'] for p in pts]
        y = [p['hit_rate_improvement_pct'] for p in pts]
        ax.plot(x, y, 'o-', label=f'batch={ac}', linewidth=2, markersize=8)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Sequence Length (num chunks)')
    ax.set_ylabel('Hit Rate Improvement over Baseline (%)')
    ax.set_title('PROSE Advantage: Fast Tier Hit Rate Improvement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'fig3_prose_advantage.pdf')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {fig_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SimCXL + PROSE Co-Simulation Experiment Runner'
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs/simcxl_cosim',
        help='Output directory for results (default: outputs/simcxl_cosim)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick mode (fewer steps, fewer seq lengths)'
    )
    parser.add_argument(
        '--no-figures', action='store_true',
        help='Skip figure generation'
    )
    args = parser.parse_args()

    print("=" * 72)
    print("SimCXL + PROSE Cycle-Accurate Co-Simulation")
    print("=" * 72)
    print(f"\nSimCXL timing parameters:")
    timing = SimCXLTiming()
    for k, v in timing.__dict__.items():
        print(f"  {k}: {v}")
    print()

    start_time = time.time()

    # Run experiment matrix
    results = run_experiment_matrix(
        output_dir=args.output_dir,
        quick=args.quick,
    )

    # Generate figures
    if not args.no_figures:
        print("\nGenerating figures...")
        generate_figures(results, args.output_dir)

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
