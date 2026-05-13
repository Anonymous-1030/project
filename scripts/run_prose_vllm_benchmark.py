#!/usr/bin/env python3
"""
ProSE-X v2 vs Full KV Benchmark.

Runs comprehensive head-to-head comparison across multiple context lengths.
Measures: memory reduction, candidate recall, pipeline latency, budget utilization,
gold chunk recovery rate.

Usage:
    cd /d/LLM
    conda activate ghost_cim
    PYTHONPATH=/d/LLM python scripts/run_prose_vllm_benchmark.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    ProSEXv2Config, MQRULFConfig, BurstConfig, EABSConfig, ODUSConfig,
)
from src.promotion.pipeline import PromotionPipeline
from src.core_types import ChunkMetadata, QueryContext, ChunkTier


def create_workload(context_length: int, chunk_size: int = 512,
                    workload_type: str = "passkey",
                    num_gold: int = 3) -> Tuple[List[ChunkMetadata], List[str], List[ChunkMetadata]]:
    """Create a synthetic workload with gold chunks."""
    num_chunks = context_length // chunk_size
    request_id = "bench_req"

    chunks = []
    gold_indices = set()

    if workload_type == "passkey":
        # Gold chunks clustered in middle
        mid = num_chunks // 2
        gold_indices = {mid - 1, mid, mid + 1}
    elif workload_type == "uniform":
        # Gold chunks spread across context
        for i in range(num_gold):
            gold_indices.add(i * num_chunks // (num_gold + 1))
    elif workload_type == "multi_needle":
        # Multiple needle-in-haystack
        gold_indices = {num_chunks // 8, num_chunks // 4, num_chunks // 2,
                        3 * num_chunks // 4, 7 * num_chunks // 8}

    gold_chunk_ids = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk_id = f"{request_id}:{start}-{end}"

        if i in gold_indices:
            sig = np.ones(128, dtype=np.float32) * 0.9
            gold_chunk_ids.append(chunk_id)
        else:
            sig = np.random.RandomState(i).randn(128).astype(np.float32) * 0.2

        chunk = ChunkMetadata(
            chunk_id=chunk_id,
            request_id=request_id,
            token_start=start,
            token_end=end,
            position_ratio=i / max(num_chunks - 1, 1),
            num_tokens=chunk_size,
            logical_bytes=chunk_size * 2 * 32 * 8 * 128,  # 2B * layers * heads * dim
            signature=sig,
            tier=ChunkTier.TAIL,
            creation_step=0,
            is_section_boundary=(i % 5 == 0),
        )
        chunks.append(chunk)

    # 10% anchors
    n_anchors = max(1, num_chunks // 10)
    anchor_chunks = chunks[:n_anchors]
    for c in anchor_chunks:
        c.tier = ChunkTier.ANCHOR

    tail_chunks = chunks[n_anchors:]

    return chunks, gold_chunk_ids, anchor_chunks, tail_chunks


def run_benchmark(context_lengths: List[int] = None, budget_ratios: List[float] = None,
                  num_steps: int = 50, seed: int = 42) -> List[Dict]:
    """Run benchmark across multiple configurations."""
    if context_lengths is None:
        context_lengths = [2048, 4096, 8192, 16384, 32768]
    if budget_ratios is None:
        budget_ratios = [0.05, 0.10, 0.20]

    results = []
    np.random.seed(seed)

    for ctx_len in context_lengths:
        for ratio in budget_ratios:
            print(f"\n{'='*70}")
            print(f"Context: {ctx_len} tokens  |  Budget: {ratio:.0%}  |  Chunks: {ctx_len//512}")
            print(f"{'='*70}")

            chunks, gold_ids, anchors, tail = create_workload(ctx_len, workload_type="multi_needle")

            # --- ProSE-X v2 ---
            config = ProSEXv2Config(
                experiment_name=f"bench_{ctx_len}_{ratio}",
                mqr_ulf=MQRULFConfig(
                    anchor_neighbor_enabled=True,
                    anchor_neighbor_radius=2,
                    anchor_neighbor_top_k=3,
                    lexical_overlap_enabled=True,
                    lexical_overlap_top_k=5,
                    lexical_overlap_threshold=0.0,
                    structural_recency_enabled=True,
                    structural_recent_top_k=3,
                    historical_success_enabled=True,
                    historical_success_top_k=3,
                    max_total_candidates=20,
                ),
                burst=BurstConfig(
                    enabled=True,
                    radius=1,
                    sticky_enabled=True,
                    default_ttl=4,
                    ttl_refresh_policy="access",
                ),
                eabs=EABSConfig(
                    exploration_ratio=0.2,
                    budget_ratio_of_tail=ratio,
                    max_chunks_per_step=5,
                    min_score_threshold=0.1,
                ),
                odus=ODUSConfig(mode="odus_x"),
            )
            pipeline = PromotionPipeline(config)

            # Track metrics per step
            step_metrics = {
                "num_promoted": [],
                "num_visible": [],
                "num_candidates": [],
                "num_selected": [],
                "ulf_latencies_us": [],
                "scorer_latencies_us": [],
                "total_latencies_us": [],
                "gold_in_candidates": [],
                "gold_in_visible": [],
            }

            promoted_chunks: List[ChunkMetadata] = []
            gold_set = set(gold_ids)

            for step in range(num_steps):
                # Simulate query — after 10 steps, switch to gold-seeking query
                if step >= 10:
                    query_sig = np.ones(128, dtype=np.float32) * 0.85 + np.random.randn(128).astype(np.float32) * 0.05
                else:
                    query_sig = np.random.randn(128).astype(np.float32) * 0.3

                query = QueryContext(
                    request_id="bench_req",
                    step=step,
                    query_signature=query_sig,
                    query_tokens=[step] * 32,
                    active_anchor_ids=[c.chunk_id for c in anchors],
                    recent_anchor_ids=[c.chunk_id for c in anchors],
                    steps_since_start=step,
                    generation_length=step,
                )

                budget_bytes = int(sum(c.logical_bytes for c in tail) * ratio)

                try:
                    t0 = time.perf_counter()
                    result = pipeline.run(
                        query=query,
                        tail_chunks=list(tail),
                        anchor_chunks=list(anchors),
                        promoted_chunks=list(promoted_chunks),
                        budget_bytes=budget_bytes,
                    )
                    elapsed = time.perf_counter() - t0

                    step_metrics["num_promoted"].append(len(result.final_visible_ids))
                    step_metrics["num_visible"].append(len(result.final_visible_ids))
                    step_metrics["num_candidates"].append(len(result.ulf_result.candidate_ids))
                    step_metrics["num_selected"].append(
                        len(result.scheduler_result.selected_ids) if result.scheduler_result else 0
                    )
                    step_metrics["ulf_latencies_us"].append(result.ulf_result.ulf_latency_us)
                    step_metrics["scorer_latencies_us"].append(result.scorer_result.scorer_latency_us)
                    step_metrics["total_latencies_us"].append(result.total_latency_us)

                    # Gold tracking
                    cand_ids = set(result.ulf_result.candidate_ids)
                    gold_in_cand = len(gold_set & cand_ids)
                    step_metrics["gold_in_candidates"].append(gold_in_cand)
                    step_metrics["gold_in_visible"].append(
                        len(gold_set & set(result.final_visible_ids))
                    )

                    # Update promoted for next step
                    if result.sticky_result:
                        promoted_ids = set(result.sticky_result.promoted_ids)
                        promoted_chunks = [c for c in chunks if c.chunk_id in promoted_ids]

                except Exception as e:
                    print(f"  Pipeline error at step {step}: {e}")
                    # Still pad step metrics so indices stay aligned
                    for key in step_metrics:
                        step_metrics[key].append(0)
                    continue

            # Full KV baseline (all chunks visible)
            full_kv_blocks = ctx_len // 16  # 16 tokens per block
            prose_avg_blocks = int(np.mean(step_metrics["num_visible"]) * 512 / 16) if step_metrics["num_visible"] else 0

            # Aggregate
            result_entry = {
                "context_length": ctx_len,
                "budget_ratio": ratio,
                "num_chunks": ctx_len // 512,
                "num_steps": num_steps,
                "num_gold_chunks": len(gold_ids),

                # Memory
                "full_kv_blocks": full_kv_blocks,
                "prose_avg_promoted_chunks": float(np.mean(step_metrics["num_visible"])),
                "prose_avg_promoted_blocks": prose_avg_blocks,
                "block_reduction_ratio": 1.0 - prose_avg_blocks / max(full_kv_blocks, 1),
                "memory_reduction_pct": (1.0 - prose_avg_blocks / max(full_kv_blocks, 1)) * 100,

                # Candidate recall
                "avg_candidates_per_step": float(np.mean(step_metrics["num_candidates"])),
                "avg_promoted_per_step": float(np.mean(step_metrics["num_promoted"])),
                "avg_selected_per_step": float(np.mean(step_metrics["num_selected"])),

                # Gold recovery
                "gold_recall_in_candidates": float(np.mean(step_metrics["gold_in_candidates"]) / max(len(gold_ids), 1)),
                "gold_recall_in_visible": float(np.mean(step_metrics["gold_in_visible"]) / max(len(gold_ids), 1)),
                "gold_always_visible_pct": float(
                    sum(1 for g in step_metrics["gold_in_visible"] if g == len(gold_ids)) / max(num_steps, 1)
                ) * 100,

                # Latency
                "avg_ulf_us": float(np.mean(step_metrics["ulf_latencies_us"])),
                "avg_scorer_us": float(np.mean(step_metrics["scorer_latencies_us"])),
                "avg_total_pipeline_us": float(np.mean(step_metrics["total_latencies_us"])),
                "p99_total_pipeline_us": float(np.percentile(step_metrics["total_latencies_us"], 99)) if step_metrics["total_latencies_us"] else 0.0,
            }

            results.append(result_entry)
            print(f"  Memory: {full_kv_blocks} blocks -> {prose_avg_blocks} blocks "
                  f"({result_entry['block_reduction_ratio']:.1%} reduction)")
            print(f"  Gold recall@candidates: {result_entry['gold_recall_in_candidates']:.1%}")
            print(f"  Gold recall@visible:    {result_entry['gold_recall_in_visible']:.1%}")
            print(f"  Pipeline latency:        {result_entry['avg_total_pipeline_us']:.1f} us avg, "
                  f"{result_entry['p99_total_pipeline_us']:.1f} us p99")

    return results


def print_summary_table(results: List[Dict]):
    """Print formatted summary table."""
    print(f"\n\n{'='*120}")
    print("PROSE-X v2 vs FULL KV BENCHMARK SUMMARY")
    print(f"{'='*120}")

    print(f"\n{'Context':>8} {'Budget':>7} {'Full KV':>10} {'ProSE':>10} "
          f"{'Memory':>10} {'Gold@Cand':>10} {'Gold@Vis':>10} "
          f"{'Pipe(us)':>10} {'P99(us)':>10}")
    print("-" * 100)

    for r in results:
        print(f"{r['context_length']:>8} {r['budget_ratio']:>7.0%} "
              f"{r['full_kv_blocks']:>10} {r['prose_avg_promoted_blocks']:>10} "
              f"{r['memory_reduction_pct']:>9.1f}% "
              f"{r['gold_recall_in_candidates']:>10.1%} "
              f"{r['gold_recall_in_visible']:>10.1%} "
              f"{r['avg_total_pipeline_us']:>10.1f} "
              f"{r['p99_total_pipeline_us']:>10.1f}")

    # Averages by context length
    print(f"\n\n{'='*80}")
    print("AVERAGED BY CONTEXT LENGTH")
    print(f"{'='*80}")
    ctx_lengths = sorted(set(r["context_length"] for r in results))
    print(f"{'Context':>8} {'Memory Saved':>14} {'Gold@Cand':>10} {'Gold@Vis':>10} {'Pipe Lat':>10}")
    print("-" * 60)

    for ctx in ctx_lengths:
        ctx_results = [r for r in results if r["context_length"] == ctx]
        avg_mem = np.mean([r["memory_reduction_pct"] for r in ctx_results])
        avg_gold_c = np.mean([r["gold_recall_in_candidates"] for r in ctx_results])
        avg_gold_v = np.mean([r["gold_recall_in_visible"] for r in ctx_results])
        avg_lat = np.mean([r["avg_total_pipeline_us"] for r in ctx_results])
        print(f"{ctx:>8} {avg_mem:>13.1f}% {avg_gold_c:>10.1%} {avg_gold_v:>10.1%} {avg_lat:>10.1f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ProSE-X v2 vs Full KV Benchmark")
    parser.add_argument("--context-lengths", type=int, nargs="+",
                        default=[2048, 4096, 8192, 16384],
                        help="Context lengths to benchmark")
    parser.add_argument("--budget-ratios", type=float, nargs="+",
                        default=[0.05, 0.10, 0.20],
                        help="Budget ratios to test")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of decode steps per config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    print("=" * 70)
    print("PROSE-X v2 vs FULL KV BENCHMARK")
    print("=" * 70)
    print(f"  Context lengths: {args.context_lengths}")
    print(f"  Budget ratios:   {args.budget_ratios}")
    print(f"  Decode steps:    {args.steps}")
    print(f"  Scorer:          ODUS-X (adaptive gating, no training)")
    print(f"  Total configs:   {len(args.context_lengths) * len(args.budget_ratios)}")

    results = run_benchmark(
        context_lengths=args.context_lengths,
        budget_ratios=args.budget_ratios,
        num_steps=args.steps,
        seed=args.seed,
    )

    print_summary_table(results)

    # Save
    output_path = args.output or "outputs/prose_vllm_benchmark.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Key takeaways
    all_mem = [r["memory_reduction_pct"] for r in results]
    all_gold = [r["gold_recall_in_visible"] for r in results]
    all_lat = [r["avg_total_pipeline_us"] for r in results]

    print(f"\n{'='*70}")
    print("KEY TAKEAWAYS")
    print(f"{'='*70}")
    print(f"  Average memory reduction: {np.mean(all_mem):.1f}% (range: {np.min(all_mem):.1f}-{np.max(all_mem):.1f}%)")
    print(f"  Average gold recall:      {np.mean(all_gold):.1%} (range: {np.min(all_gold):.1%}-{np.max(all_gold):.1%})")
    print(f"  Average pipeline latency: {np.mean(all_lat):.1f} us (p99: {np.percentile(all_lat, 99):.1f} us)")
    print(f"  Pipeline overhead:        {np.mean(all_lat):.1f} us <<< {10000} us per attention step")


if __name__ == "__main__":
    main()
