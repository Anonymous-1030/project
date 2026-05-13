"""
Unified Innovation Benchmark Runner.

Evaluates all four PROSE-X 3.0 innovations against the baseline:
  1. ABA  - Adaptive Bandwidth Arbitrage
  2. HES  - Hierarchical Evidence Synthesis
  3. GBS  - Ghost Block Synthesis
  4. SDAP - Speculative Decode-Aware Promotion

Metrics reported:
  - Throughput improvement (vs baseline) at 32/8/4 GB/s
  - Gold chunk recall (Candidate→Visible gap reduction)
  - Attention compute savings
  - Pipeline latency overhead
  - Ghost block recovery rate
  - Mode distribution (ABA)
"""

from __future__ import annotations

import json
import time
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from src.config import ProSEXv2Config
from src.core_types import (
    ChunkMetadata, QueryContext, ChunkTier, ChunkState,
    PromotionPipelineResult,
)
from src.promotion.pipeline import PromotionPipeline
from src.innovations.aba import AdaptiveBandwidthArbitrage, ABAConfig, ABAMode
from src.innovations.hes import HierarchicalEvidenceSynthesis, HESConfig
from src.innovations.gbs import GhostBlockSynthesizer, GBSConfig
from src.innovations.sdap import SpeculativeDecodeAwarePromotion, SDAPConfig

logger = logging.getLogger(__name__)

# ── Workload Generator ────────────────────────────────────────────────


@dataclass
class WorkloadConfig:
    context_length: int = 8192
    chunk_size: int = 512
    num_decode_steps: int = 50
    num_gold_chunks: int = 4
    anchor_ratio: float = 0.1
    budget_ratio: float = 0.10
    bandwidth_gbps: float = 8.0
    queue_depth: int = 4
    seed: int = 42


def generate_synthetic_workload(config: WorkloadConfig) -> Dict[str, Any]:
    """Generate a synthetic workload with known gold chunks for evaluation."""
    rng = np.random.default_rng(config.seed)
    n_chunks = config.context_length // config.chunk_size
    n_anchors = max(1, int(n_chunks * config.anchor_ratio))

    chunks: Dict[str, ChunkMetadata] = {}
    for i in range(n_chunks):
        cid = f"chunk_{i:04d}"
        sig = rng.standard_normal(16).astype(np.float32)
        sig = sig / (np.linalg.norm(sig) + 1e-8)
        chunks[cid] = ChunkMetadata(
            chunk_id=cid,
            request_id="req_bench",
            token_start=i * config.chunk_size,
            token_end=(i + 1) * config.chunk_size,
            position_ratio=i / n_chunks,
            num_tokens=config.chunk_size,
            logical_bytes=config.chunk_size * 2,
            signature=sig,
            tier=ChunkTier.ANCHOR if i < n_anchors else ChunkTier.TAIL,
            state=ChunkState.ACTIVE,
            creation_step=0,
        )

    anchor_ids = [f"chunk_{i:04d}" for i in range(n_anchors)]
    tail_ids = [f"chunk_{i:04d}" for i in range(n_anchors, n_chunks)]

    gold_indices = rng.choice(len(tail_ids), size=min(config.num_gold_chunks, len(tail_ids)), replace=False)
    gold_ids = [tail_ids[i] for i in gold_indices]

    gold_attention = {}
    for cid in gold_ids:
        gold_attention[cid] = rng.uniform(0.15, 0.5)
    for cid in anchor_ids:
        gold_attention[cid] = rng.uniform(0.02, 0.08)
    for cid in tail_ids:
        if cid not in gold_attention:
            gold_attention[cid] = rng.uniform(0.0, 0.03)

    return {
        "chunks": chunks,
        "anchor_ids": anchor_ids,
        "tail_ids": tail_ids,
        "gold_ids": gold_ids,
        "gold_attention": gold_attention,
        "n_chunks": n_chunks,
        "config": config,
    }


# ── Baseline Runner ──────────────────────────────────────────────────


def run_baseline(workload: Dict[str, Any], prose_config: ProSEXv2Config) -> Dict[str, Any]:
    """Run baseline PROSE pipeline without innovations."""
    config = workload["config"]
    chunks = workload["chunks"]
    anchor_ids = workload["anchor_ids"]
    tail_ids = workload["tail_ids"]
    gold_ids = set(workload["gold_ids"])
    gold_attention = workload["gold_attention"]

    pipeline = PromotionPipeline(prose_config)
    rng = np.random.default_rng(config.seed + 100)

    metrics = {
        "gold_recall_at_visible": [],
        "gold_recall_at_candidates": [],
        "visible_count": [],
        "latency_us": [],
        "throughput_tokens_per_s": [],
    }

    for step in range(config.num_decode_steps):
        query_sig = rng.standard_normal(16).astype(np.float32)
        query_sig = query_sig / (np.linalg.norm(query_sig) + 1e-8)
        query = QueryContext(
            request_id="req_bench",
            step=step,
            query_signature=query_sig,
            query_length=32,
            steps_since_start=step,
        )

        anchor_chunks = [chunks[cid] for cid in anchor_ids]
        tail_chunks = [chunks[cid] for cid in tail_ids]
        promoted_chunks = []

        budget_bytes = int(sum(c.logical_bytes for c in tail_chunks) * config.budget_ratio)

        result = pipeline.run(
            query=query,
            tail_chunks=tail_chunks,
            anchor_chunks=anchor_chunks,
            promoted_chunks=promoted_chunks,
            budget_bytes=budget_bytes,
            gold_chunk_ids=gold_ids,
        )

        visible_set = set(result.final_visible_ids)
        gold_in_visible = len(gold_ids & visible_set)
        gold_recall_visible = gold_in_visible / max(len(gold_ids), 1)

        candidate_set = set(result.ulf_result.candidate_ids) if result.ulf_result else set()
        gold_in_candidates = len(gold_ids & candidate_set)
        gold_recall_candidates = gold_in_candidates / max(len(gold_ids), 1)

        # Realistic latency model:
        # - Transfer time: bytes / bandwidth
        # - Attention compute: O(visible_chunks * chunk_size) with ~0.1 us per token
        # - Pipeline overhead: from result
        # At high BW, compute dominates; at low BW, transfer dominates
        n_total_chunks = len(chunks)
        transfer_bytes = len(visible_set) * config.chunk_size * 2  # KV bytes
        transfer_time_us = (transfer_bytes / (config.bandwidth_gbps * 1e9)) * 1e6
        attention_compute_us = len(visible_set) * config.chunk_size * 0.1  # ~0.1us per KV token
        full_attention_compute_us = n_total_chunks * config.chunk_size * 0.1
        total_time_us = result.total_latency_us + max(transfer_time_us, attention_compute_us)
        full_kv_time_us = (n_total_chunks * config.chunk_size * 2 / (config.bandwidth_gbps * 1e9)) * 1e6 + full_attention_compute_us
        throughput_ratio = full_kv_time_us / max(total_time_us, 1.0)

        metrics["gold_recall_at_visible"].append(gold_recall_visible)
        metrics["gold_recall_at_candidates"].append(gold_recall_candidates)
        metrics["visible_count"].append(len(visible_set))
        metrics["latency_us"].append(total_time_us)
        metrics["throughput_tokens_per_s"].append(throughput_ratio)

    return {
        "avg_gold_recall_visible": float(np.mean(metrics["gold_recall_at_visible"])),
        "avg_gold_recall_candidates": float(np.mean(metrics["gold_recall_at_candidates"])),
        "avg_visible_count": float(np.mean(metrics["visible_count"])),
        "avg_latency_us": float(np.mean(metrics["latency_us"])),
        "avg_throughput_ratio": float(np.mean(metrics["throughput_tokens_per_s"])),
        "p99_latency_us": float(np.percentile(metrics["latency_us"], 99)),
    }


# ── Innovation Runner ────────────────────────────────────────────────


def run_with_innovations(workload: Dict[str, Any], prose_config: ProSEXv2Config) -> Dict[str, Any]:
    """Run PROSE pipeline with all four innovations enabled."""
    config = workload["config"]
    chunks = workload["chunks"]
    anchor_ids = workload["anchor_ids"]
    tail_ids = workload["tail_ids"]
    gold_ids = set(workload["gold_ids"])
    gold_attention = workload["gold_attention"]

    pipeline = PromotionPipeline(prose_config)
    aba = AdaptiveBandwidthArbitrage(ABAConfig())
    hes = HierarchicalEvidenceSynthesis(HESConfig())
    gbs = GhostBlockSynthesizer(GBSConfig())
    sdap = SpeculativeDecodeAwarePromotion(SDAPConfig())

    rng = np.random.default_rng(config.seed + 100)

    # Prefill: ingest micro-embeddings
    for cid, chunk in chunks.items():
        hes.ingest_chunk(chunk)
        if chunk.signature is not None:
            gbs.store_micro_embedding(cid, chunk.signature)

    metrics = {
        "gold_recall_at_visible": [],
        "gold_recall_at_candidates": [],
        "visible_count": [],
        "latency_us": [],
        "throughput_tokens_per_s": [],
        "ghost_blocks_active": [],
        "aba_mode": [],
    }

    for step in range(config.num_decode_steps):
        query_sig = rng.standard_normal(16).astype(np.float32)
        query_sig = query_sig / (np.linalg.norm(query_sig) + 1e-8)
        query = QueryContext(
            request_id="req_bench",
            step=step,
            query_signature=query_sig,
            query_length=32,
            steps_since_start=step,
        )

        # ABA: select mode based on bandwidth
        bw_noise = rng.normal(0, 1.0)
        measured_bw = config.bandwidth_gbps + bw_noise
        qd_noise = int(rng.integers(-1, 2))
        measured_qd = max(0, config.queue_depth + qd_noise)
        mode = aba.update_measurements(measured_bw, measured_qd, step)

        anchor_chunks = [chunks[cid] for cid in anchor_ids]
        tail_chunks = [chunks[cid] for cid in tail_ids]
        promoted_chunks = []
        budget_bytes = int(sum(c.logical_bytes for c in tail_chunks) * config.budget_ratio)

        if mode == ABAMode.TRANSPARENT:
            # Transparent: sparse attention mask only, skip full pipeline
            result = aba.apply_transparent_mode(
                chunks, query, anchor_ids, config.budget_ratio * 2.5
            )
        else:
            # Run standard pipeline
            result = pipeline.run(
                query=query,
                tail_chunks=tail_chunks,
                anchor_chunks=anchor_chunks,
                promoted_chunks=promoted_chunks,
                budget_bytes=budget_bytes,
                gold_chunk_ids=gold_ids,
            )

            # HES: re-score candidates with neural evidence
            if result.ulf_result and result.ulf_result.candidate_ids:
                hes_scores = hes.score_chunks(
                    query, result.ulf_result.candidate_ids, chunks
                )
                # Use HES scores to potentially expand visible set
                hes_admitted = [s for s in hes_scores if s.score > 0.4]
                extra_ids = [s.chunk_id for s in hes_admitted[:3]
                             if s.chunk_id not in result.final_visible_ids]
                result.final_visible_ids = list(set(result.final_visible_ids + extra_ids))

            # SDAP: enhance with draft model evidence
            if result.scorer_result and result.scorer_result.candidates:
                draft_evidence = sdap.get_draft_evidence(query, chunks, gold_attention)
                enhanced = sdap.enhance_scores(
                    result.scorer_result.candidates, draft_evidence, query
                )
                # Use enhanced top candidates to expand visible set
                sdap_top = [c for c in enhanced[:5] if c.score > 0.5]
                sdap_extra = [c.chunk_id for c in sdap_top
                              if c.chunk_id not in result.final_visible_ids]
                result.final_visible_ids = list(set(result.final_visible_ids + sdap_extra[:2]))

            # GBS: synthesize ghost blocks if attention sink detected
            # Use attention ONLY for anchor blocks (the always-visible set)
            # to detect if the model needs content not in the anchor set
            anchor_attention = {
                cid: gold_attention.get(cid, 0.005) for cid in anchor_ids
            }
            ghosts = gbs.synthesize_ghosts(
                query, chunks, anchor_ids, anchor_attention
            )
            ghost_ids = [g.chunk_id for g in ghosts if not g.is_promoted_to_real]
            result.final_visible_ids = list(set(result.final_visible_ids + ghost_ids))

            # GBS: check if ghosts should be promoted to real
            gbs.update_ghost_attention(gold_attention, step)

            # Aggressive mode enhancement
            if mode == ABAMode.AGGRESSIVE:
                result = aba.apply_aggressive_mode(result, chunks, query)

        # Compute metrics
        visible_set = set(result.final_visible_ids)
        gold_in_visible = len(gold_ids & visible_set)
        gold_recall_visible = gold_in_visible / max(len(gold_ids), 1)

        candidate_set = set()
        if result.ulf_result:
            candidate_set = set(result.ulf_result.candidate_ids)
        gold_in_candidates = len(gold_ids & candidate_set)
        gold_recall_candidates = gold_in_candidates / max(len(gold_ids), 1)

        n_total_chunks = len(chunks)
        # Ghost blocks don't require DMA transfer (reconstructed locally)
        n_ghost = len(gbs.get_ghost_ids())
        n_real_visible = len(visible_set) - n_ghost
        # Aggressive mode applies compression
        compression = 0.5 if mode == ABAMode.AGGRESSIVE else 1.0
        transfer_bytes = int(n_real_visible * config.chunk_size * 2 * compression)
        transfer_time_us = (transfer_bytes / (config.bandwidth_gbps * 1e9)) * 1e6
        attention_compute_us = len(visible_set) * config.chunk_size * 0.1
        full_attention_compute_us = n_total_chunks * config.chunk_size * 0.1
        total_time_us = result.total_latency_us + max(transfer_time_us, attention_compute_us)
        full_kv_time_us = (n_total_chunks * config.chunk_size * 2 / (config.bandwidth_gbps * 1e9)) * 1e6 + full_attention_compute_us
        throughput_ratio = full_kv_time_us / max(total_time_us, 1.0)

        metrics["gold_recall_at_visible"].append(gold_recall_visible)
        metrics["gold_recall_at_candidates"].append(gold_recall_candidates)
        metrics["visible_count"].append(len(visible_set))
        metrics["latency_us"].append(total_time_us)
        metrics["throughput_tokens_per_s"].append(throughput_ratio)
        metrics["ghost_blocks_active"].append(len(gbs.get_ghost_ids()))
        metrics["aba_mode"].append(mode.value)

    return {
        "avg_gold_recall_visible": float(np.mean(metrics["gold_recall_at_visible"])),
        "avg_gold_recall_candidates": float(np.mean(metrics["gold_recall_at_candidates"])),
        "avg_visible_count": float(np.mean(metrics["visible_count"])),
        "avg_latency_us": float(np.mean(metrics["latency_us"])),
        "avg_throughput_ratio": float(np.mean(metrics["throughput_tokens_per_s"])),
        "p99_latency_us": float(np.percentile(metrics["latency_us"], 99)),
        "avg_ghost_blocks": float(np.mean(metrics["ghost_blocks_active"])),
        "aba_metrics": aba.get_metrics(),
        "hes_metrics": hes.get_metrics(),
        "gbs_metrics": gbs.get_metrics(),
        "sdap_metrics": sdap.get_metrics(),
    }


# ── Main Benchmark Suite ─────────────────────────────────────────────


def run_full_benchmark() -> Dict[str, Any]:
    """Run complete benchmark suite across bandwidth conditions."""
    print("=" * 70)
    print("  ProSE-X 3.0 Innovation Benchmark Suite")
    print("  ABA + HES + GBS + SDAP vs Baseline")
    print("=" * 70)

    prose_config = ProSEXv2Config()
    results = {}

    bandwidth_configs = [
        ("32 GB/s (high BW)", 32.0, 2),
        ("8 GB/s (medium BW)", 8.0, 4),
        ("4 GB/s (low BW / CXL saturated)", 4.0, 10),
    ]

    context_lengths = [4096, 8192, 16384]

    for bw_label, bw_gbps, queue_depth in bandwidth_configs:
        print(f"\n{'─' * 70}")
        print(f"  Bandwidth: {bw_label}")
        print(f"{'─' * 70}")

        bw_results = {}
        for ctx_len in context_lengths:
            wl_config = WorkloadConfig(
                context_length=ctx_len,
                bandwidth_gbps=bw_gbps,
                queue_depth=queue_depth,
                num_decode_steps=50,
                num_gold_chunks=4,
                budget_ratio=0.10,
            )
            workload = generate_synthetic_workload(wl_config)

            baseline = run_baseline(workload, prose_config)
            innovation = run_with_innovations(workload, prose_config)

            recall_improvement = (
                (innovation["avg_gold_recall_visible"] - baseline["avg_gold_recall_visible"])
                / max(baseline["avg_gold_recall_visible"], 0.01)
            )
            throughput_improvement = (
                (innovation["avg_throughput_ratio"] - baseline["avg_throughput_ratio"])
                / max(baseline["avg_throughput_ratio"], 0.01)
            )

            comparison = {
                "context_length": ctx_len,
                "bandwidth_gbps": bw_gbps,
                "baseline": baseline,
                "innovation": innovation,
                "improvements": {
                    "recall_lift": recall_improvement,
                    "throughput_lift": throughput_improvement,
                    "latency_reduction": 1.0 - (innovation["avg_latency_us"] / max(baseline["avg_latency_us"], 1.0)),
                    "visible_count_change": innovation["avg_visible_count"] - baseline["avg_visible_count"],
                },
            }
            bw_results[f"ctx_{ctx_len}"] = comparison

            print(f"\n  Context: {ctx_len} tokens ({ctx_len // 512} chunks)")
            print(f"  ┌{'─' * 50}┐")
            print(f"  │ {'Metric':<28} {'Baseline':>9} {'Innov':>9} │")
            print(f"  ├{'─' * 50}┤")
            print(f"  │ {'Gold Recall@Visible':<28} {baseline['avg_gold_recall_visible']:>8.1%} {innovation['avg_gold_recall_visible']:>8.1%} │")
            print(f"  │ {'Gold Recall@Candidates':<28} {baseline['avg_gold_recall_candidates']:>8.1%} {innovation['avg_gold_recall_candidates']:>8.1%} │")
            print(f"  │ {'Avg Visible Chunks':<28} {baseline['avg_visible_count']:>9.1f} {innovation['avg_visible_count']:>9.1f} │")
            print(f"  │ {'Speedup vs Full-KV':<28} {baseline['avg_throughput_ratio']:>8.2f}x {innovation['avg_throughput_ratio']:>8.2f}x │")
            print(f"  │ {'Avg Latency (us)':<28} {baseline['avg_latency_us']:>9.1f} {innovation['avg_latency_us']:>9.1f} │")
            print(f"  ├{'─' * 50}┤")
            print(f"  │ {'Recall Lift':<28} {recall_improvement:>+18.1%} │")
            print(f"  │ {'Throughput Lift':<28} {throughput_improvement:>+18.1%} │")
            print(f"  └{'─' * 50}┘")

        results[bw_label] = bw_results

    # Print innovation-specific metrics
    print(f"\n{'═' * 70}")
    print("  Innovation-Specific Metrics (last run)")
    print(f"{'═' * 70}")

    last_innov = None
    for bw_label in results:
        for ctx_key in results[bw_label]:
            last_innov = results[bw_label][ctx_key]["innovation"]

    if last_innov:
        if "aba_metrics" in last_innov:
            aba_m = last_innov["aba_metrics"]
            print(f"\n  ABA (Adaptive Bandwidth Arbitrage):")
            print(f"    Mode switches: {aba_m.get('mode_switches', 0)}")
            print(f"    Transparent: {aba_m.get('transparent_fraction', 0):.1%}")
            print(f"    Hybrid: {aba_m.get('hybrid_fraction', 0):.1%}")
            print(f"    Aggressive: {aba_m.get('aggressive_fraction', 0):.1%}")

        if "hes_metrics" in last_innov:
            hes_m = last_innov["hes_metrics"]
            print(f"\n  HES (Hierarchical Evidence Synthesis):")
            print(f"    Admission rate: {hes_m.get('admission_rate', 0):.1%}")
            print(f"    HES lift over SimHash: {hes_m.get('hes_lift_over_simhash', 0):.2f}x")
            print(f"    Avg scoring latency: {hes_m.get('avg_scoring_latency_us', 0):.1f} us")

        if "gbs_metrics" in last_innov:
            gbs_m = last_innov["gbs_metrics"]
            print(f"\n  GBS (Ghost Block Synthesis):")
            print(f"    Sink detection rate: {gbs_m.get('sink_detection_rate', 0):.1%}")
            print(f"    Ghosts created: {gbs_m.get('ghosts_created', 0)}")
            print(f"    Promoted to real: {gbs_m.get('ghosts_promoted_to_real', 0)}")
            print(f"    Recovery rate: {gbs_m.get('attention_recovery_rate', 0):.1%}")

        if "sdap_metrics" in last_innov:
            sdap_m = last_innov["sdap_metrics"]
            print(f"\n  SDAP (Speculative Decode-Aware Promotion):")
            print(f"    Avg acceptance rate: {sdap_m.get('avg_acceptance_rate', 0):.1%}")
            print(f"    Promotion overrides: {sdap_m.get('promotion_overrides', 0)}")
            print(f"    Drift corrections: {sdap_m.get('drift_corrections', 0)}")
            print(f"    Evidence quality: {sdap_m.get('avg_evidence_quality', 0):.3f}")

    print(f"\n{'═' * 70}")

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "innovation_benchmark_results.json"

    serializable = _make_serializable(results)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    return results


def _make_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_full_benchmark()
