"""
vLLM Runner with Detailed ProSE-X 2.0 Metrics for WSL.

This script runs vLLM with ProSE-X promotion and prints detailed metrics:
- MQR-ULF candidate recall statistics
- Per-queue contribution counts
- Utility scoring distributions
- Scheduler decisions (exploit vs explore)
- Burst expansion statistics
- Sticky TTL updates
- Recovery metrics (retention coverage, conditional recovery)

Usage:
    python run_vllm_metrics.py \
        --model microsoft/Phi-3-mini-4k-instruct \
        --context-length 4096 \
        --budget-ratio 0.1 \
        --workload-type passkey \
        --use-prose-v2 \
        --detailed-metrics
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# ProSE-X v2 imports
from src.config import ProSEXv2Config, MQRULFConfig, BurstConfig, EABSConfig, ODUSConfig
from src.promotion.pipeline import PromotionPipeline
from src.core_types import (
    ChunkMetadata, QueryContext, ChunkTier, FailureReason,
    ULFResult, ScorerResult, SchedulerResult, BurstResult, StickyResult
)
from src.eval.metrics.recall_metrics import (
    CandidateMetricsCalculator, ScoringMetricsCalculator, SchedulerMetricsCalculator
)
from src.eval.failure_attribution.attributor import FailureAttributor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    """Detailed metrics for a single decode step."""
    step: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # ULF metrics
    ulf_candidates: int = 0
    ulf_tail_total: int = 0
    ulf_recall_rate: float = 0.0
    ulf_queue_contributions: Dict[str, int] = field(default_factory=dict)
    ulf_latency_us: float = 0.0
    
    # Scorer metrics
    scorer_candidates: int = 0
    scorer_top_score: float = 0.0
    scorer_score_mean: float = 0.0
    scorer_score_std: float = 0.0
    scorer_latency_us: float = 0.0
    
    # Scheduler metrics
    scheduler_selected: int = 0
    scheduler_exploit: int = 0
    scheduler_explore: int = 0
    scheduler_dropped_budget: int = 0
    scheduler_dropped_score: int = 0
    scheduler_dropped_confidence: int = 0
    scheduler_utilization: float = 0.0
    scheduler_latency_us: float = 0.0
    
    # Burst metrics
    burst_input: int = 0
    burst_total: int = 0
    burst_expansion: int = 0
    burst_latency_us: float = 0.0
    
    # Sticky metrics
    sticky_promoted: int = 0
    sticky_expired: int = 0
    sticky_refreshed: int = 0
    sticky_avg_ttl: float = 0.0
    sticky_latency_us: float = 0.0
    
    # Recovery tracking
    gold_recall_at_k: Dict[int, bool] = field(default_factory=dict)
    gold_rank: Optional[int] = None
    gold_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "ulf": {
                "candidates": self.ulf_candidates,
                "tail_total": self.ulf_tail_total,
                "recall_rate": self.ulf_recall_rate,
                "queue_contributions": self.ulf_queue_contributions,
                "latency_us": self.ulf_latency_us,
            },
            "scorer": {
                "candidates": self.scorer_candidates,
                "top_score": self.scorer_top_score,
                "score_mean": self.scorer_score_mean,
                "score_std": self.scorer_score_std,
                "latency_us": self.scorer_latency_us,
            },
            "scheduler": {
                "selected": self.scheduler_selected,
                "exploit": self.scheduler_exploit,
                "explore": self.scheduler_explore,
                "dropped_budget": self.scheduler_dropped_budget,
                "dropped_score": self.scheduler_dropped_score,
                "dropped_confidence": self.scheduler_dropped_confidence,
                "utilization": self.scheduler_utilization,
                "latency_us": self.scheduler_latency_us,
            },
            "burst": {
                "input": self.burst_input,
                "total": self.burst_total,
                "expansion": self.burst_expansion,
                "latency_us": self.burst_latency_us,
            },
            "sticky": {
                "promoted": self.sticky_promoted,
                "expired": self.sticky_expired,
                "refreshed": self.sticky_refreshed,
                "avg_ttl": self.sticky_avg_ttl,
                "latency_us": self.sticky_latency_us,
            },
            "recovery": {
                "gold_recall_at_k": self.gold_recall_at_k,
                "gold_rank": self.gold_rank,
                "gold_score": self.gold_score,
            },
        }


class DetailedMetricsCollector:
    """Collects detailed metrics from ProSE-X 2.0 pipeline."""
    
    def __init__(self, gold_chunk_ids: Optional[Set[str]] = None):
        self.gold_chunk_ids = gold_chunk_ids or set()
        self.step_metrics: List[StepMetrics] = []
        
        # Calculators
        self.candidate_calc = CandidateMetricsCalculator()
        self.scoring_calc = ScoringMetricsCalculator()
        self.scheduler_calc = SchedulerMetricsCalculator()
        self.attributor = FailureAttributor()
    
    def collect_step(
        self,
        step: int,
        pipeline_result: Any,  # PromotionPipelineResult
    ) -> StepMetrics:
        """Collect metrics from a pipeline step."""
        metrics = StepMetrics(step=step)
        
        # ULF metrics
        if pipeline_result.ulf_result:
            ulf = pipeline_result.ulf_result
            metrics.ulf_candidates = ulf.n_candidates
            metrics.ulf_tail_total = ulf.n_tail_total
            metrics.ulf_recall_rate = ulf.n_candidates / max(ulf.n_tail_total, 1)
            metrics.ulf_queue_contributions = ulf.per_queue_counts
            metrics.ulf_latency_us = ulf.ulf_latency_us
            
            # Gold recall at k
            if self.gold_chunk_ids:
                gold_list = list(self.gold_chunk_ids)
                for k in [1, 5, 10, 20]:
                    top_k = set(ulf.candidate_ids[:k])
                    metrics.gold_recall_at_k[k] = bool(
                        self.gold_chunk_ids & top_k
                    )
        
        # Scorer metrics
        if pipeline_result.scorer_result:
            scorer = pipeline_result.scorer_result
            metrics.scorer_candidates = scorer.n_scored
            metrics.scorer_latency_us = scorer.scorer_latency_us
            
            if scorer.candidates:
                scores = [c.score for c in scorer.candidates]
                metrics.scorer_top_score = scores[0]
                metrics.scorer_score_mean = float(np.mean(scores))
                metrics.scorer_score_std = float(np.std(scores))
                
                # Find gold rank
                for i, c in enumerate(scorer.candidates):
                    if c.chunk_id in self.gold_chunk_ids:
                        metrics.gold_rank = i + 1
                        metrics.gold_score = c.score
                        break
        
        # Scheduler metrics
        if pipeline_result.scheduler_result:
            sched = pipeline_result.scheduler_result
            metrics.scheduler_selected = len(sched.selected_ids)
            metrics.scheduler_exploit = sched.n_exploit
            metrics.scheduler_explore = sched.n_explore
            metrics.scheduler_dropped_budget = sched.n_dropped_budget
            metrics.scheduler_dropped_score = sched.n_dropped_low_score
            metrics.scheduler_dropped_confidence = sched.n_dropped_low_confidence
            metrics.scheduler_utilization = sched.utilization
            metrics.scheduler_latency_us = sched.scheduler_latency_us
        
        # Burst metrics
        if pipeline_result.burst_result:
            burst = pipeline_result.burst_result
            metrics.burst_input = burst.n_input
            metrics.burst_total = burst.n_burst_total
            metrics.burst_expansion = burst.n_expansion
            metrics.burst_latency_us = burst.burst_latency_us
        
        # Sticky metrics
        if pipeline_result.sticky_result:
            sticky = pipeline_result.sticky_result
            metrics.sticky_promoted = sticky.n_promoted
            metrics.sticky_expired = sticky.n_expired
            metrics.sticky_refreshed = sticky.n_refreshed
            metrics.sticky_avg_ttl = sticky.avg_ttl
            metrics.sticky_latency_us = sticky.sticky_latency_us
        
        self.step_metrics.append(metrics)
        return metrics
    
    def print_step_summary(self, metrics: StepMetrics) -> None:
        """Print a formatted summary of step metrics."""
        print(f"\n{'='*60}")
        print(f"Step {metrics.step} Metrics")
        print(f"{'='*60}")
        
        # ULF
        print(f"\n[ULF] Multi-Queue Recall")
        print(f"  Candidates: {metrics.ulf_candidates}/{metrics.ulf_tail_total} "
              f"({metrics.ulf_recall_rate:.1%} recall rate)")
        print(f"  Queue Contributions:")
        for queue, count in metrics.ulf_queue_contributions.items():
            print(f"    - {queue}: {count}")
        print(f"  Latency: {metrics.ulf_latency_us:.2f} μs")
        
        # Scorer
        print(f"\n[AS] Utility Scoring")
        print(f"  Candidates Scored: {metrics.scorer_candidates}")
        print(f"  Score Distribution: μ={metrics.scorer_score_mean:.3f}, "
              f"σ={metrics.scorer_score_std:.3f}, max={metrics.scorer_top_score:.3f}")
        print(f"  Latency: {metrics.scorer_latency_us:.2f} μs")
        
        # Scheduler
        print(f"\n[EABS] Exploration-Aware Scheduler")
        print(f"  Selected: {metrics.scheduler_selected} "
              f"(exploit: {metrics.scheduler_exploit}, explore: {metrics.scheduler_explore})")
        print(f"  Dropped: budget={metrics.scheduler_dropped_budget}, "
              f"score={metrics.scheduler_dropped_score}, "
              f"conf={metrics.scheduler_dropped_confidence}")
        print(f"  Budget Utilization: {metrics.scheduler_utilization:.1%}")
        print(f"  Latency: {metrics.scheduler_latency_us:.2f} μs")
        
        # Burst
        print(f"\n[BSP] Burst-and-Stick")
        print(f"  Burst: {metrics.burst_input} → {metrics.burst_total} "
              f"(+{metrics.burst_expansion} expansion)")
        print(f"  Sticky: {metrics.sticky_promoted} promoted, "
              f"{metrics.sticky_expired} expired, "
              f"{metrics.sticky_refreshed} refreshed")
        print(f"  Avg TTL: {metrics.sticky_avg_ttl:.1f}")
        print(f"  Latency: burst={metrics.burst_latency_us:.2f} μs, "
              f"sticky={metrics.sticky_latency_us:.2f} μs")
        
        # Recovery
        if metrics.gold_recall_at_k:
            print(f"\n[Recovery] Gold Chunk Tracking")
            recall_at_k_str = ", ".join(
                f"@{k}={'Y' if v else 'N'}"
                for k, v in sorted(metrics.gold_recall_at_k.items())
            )
            print(f"  Recall: {recall_at_k_str}")
            if metrics.gold_rank:
                print(f"  Gold Rank: #{metrics.gold_rank} (score: {metrics.gold_score:.3f})")
    
    def print_final_summary(self) -> None:
        """Print final summary across all steps."""
        if not self.step_metrics:
            return
        
        print(f"\n{'='*60}")
        print("Final Summary Across All Steps")
        print(f"{'='*60}")
        
        # Aggregate metrics
        avg_ulf_recall = np.mean([m.ulf_recall_rate for m in self.step_metrics])
        avg_scheduler_util = np.mean([m.scheduler_utilization for m in self.step_metrics])
        avg_burst_expansion = np.mean([
            m.burst_expansion / max(m.burst_input, 1) 
            for m in self.step_metrics if m.burst_input > 0
        ])
        
        print(f"\n[Aggregated Statistics]")
        print(f"  Avg ULF Recall Rate: {avg_ulf_recall:.1%}")
        print(f"  Avg Budget Utilization: {avg_scheduler_util:.1%}")
        print(f"  Avg Burst Expansion: {avg_burst_expansion:.1%}")
        
        # Gold recovery summary
        if self.gold_chunk_ids:
            gold_recalled_at_10 = sum(
                1 for m in self.step_metrics 
                if m.gold_recall_at_k.get(10, False)
            )
            gold_recalled_at_20 = sum(
                1 for m in self.step_metrics 
                if m.gold_recall_at_k.get(20, False)
            )
            
            print(f"\n[Gold Recovery]")
            print(f"  Steps with gold @10: {gold_recalled_at_10}/{len(self.step_metrics)}")
            print(f"  Steps with gold @20: {gold_recalled_at_20}/{len(self.step_metrics)}")
        
        # Latency breakdown
        total_ulf = sum(m.ulf_latency_us for m in self.step_metrics)
        total_scorer = sum(m.scorer_latency_us for m in self.step_metrics)
        total_scheduler = sum(m.scheduler_latency_us for m in self.step_metrics)
        total_burst = sum(m.burst_latency_us for m in self.step_metrics)
        total_sticky = sum(m.sticky_latency_us for m in self.step_metrics)
        total_all = total_ulf + total_scorer + total_scheduler + total_burst + total_sticky
        
        print(f"\n[Latency Breakdown]")
        print(f"  ULF:        {total_ulf/1000:.2f} ms ({total_ulf/max(total_all,1):.1%})")
        print(f"  Scorer:     {total_scorer/1000:.2f} ms ({total_scorer/max(total_all,1):.1%})")
        print(f"  Scheduler:  {total_scheduler/1000:.2f} ms ({total_scheduler/max(total_all,1):.1%})")
        print(f"  Burst:      {total_burst/1000:.2f} ms ({total_burst/max(total_all,1):.1%})")
        print(f"  Sticky:     {total_sticky/1000:.2f} ms ({total_sticky/max(total_all,1):.1%})")
        print(f"  Total:      {total_all/1000:.2f} ms")
    
    def save_results(self, output_path: Path) -> None:
        """Save all metrics to file."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "num_steps": len(self.step_metrics),
            "step_metrics": [m.to_dict() for m in self.step_metrics],
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def create_passkey_workload(context_length: int, chunk_size: int = 512) -> tuple:
    """
    Create a synthetic passkey workload.
    
    Returns:
        (chunks, gold_chunk_id)
    """
    num_chunks = context_length // chunk_size
    request_id = "passkey_test"
    
    chunks = []
    gold_chunk_id = None
    
    # Place passkey in middle
    gold_idx = num_chunks // 2
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk_id = f"{request_id}:{start}-{end}"
        
        # Create signature (higher for gold chunk)
        if i == gold_idx:
            signature = np.ones(128, dtype=np.float32) * 0.8
            gold_chunk_id = chunk_id
        else:
            signature = np.random.randn(128).astype(np.float32) * 0.3
        
        chunk = ChunkMetadata(
            chunk_id=chunk_id,
            request_id=request_id,
            token_start=start,
            token_end=end,
            position_ratio=i / num_chunks,
            num_tokens=chunk_size,
            logical_bytes=chunk_size * 128,
            signature=signature,
            tier=ChunkTier.TAIL,
            creation_step=0,
            is_section_boundary=(i % 5 == 0),
        )
        chunks.append(chunk)
    
    return chunks, gold_chunk_id


def run_vllm_simulation(
    model: str,
    context_length: int,
    budget_ratio: float,
    workload_type: str,
    output_dir: str,
    use_prosex: bool,
    detailed_metrics: bool,
    num_steps: int = 20,
) -> Dict[str, Any]:
    """
    Run vLLM simulation with ProSE-X 2.0 metrics.
    
    This is a simulation that mimics the vLLM integration.
    In real deployment, this would hook into actual vLLM decode loop.
    """
    print(f"\n{'#'*70}")
    print(f"# ProSE-X 2.0 vLLM Metrics Runner")
    print(f"{'#'*70}")
    print(f"Model: {model}")
    print(f"Context Length: {context_length}")
    print(f"Budget Ratio: {budget_ratio:.1%}")
    print(f"Workload Type: {workload_type}")
    print(f"Use ProSE-X v2: {use_prosex}")
    print(f"{'#'*70}\n")
    
    # Create workload
    chunks, gold_chunk_id = create_passkey_workload(context_length)
    gold_chunk_ids = {gold_chunk_id} if gold_chunk_id else set()
    
    # Split into tiers - use 10% as anchors, rest as tail
    n_anchors = max(1, len(chunks) // 10)
    anchor_chunks = chunks[:n_anchors]
    tail_chunks = chunks[n_anchors:]
    
    for c in anchor_chunks:
        c.tier = ChunkTier.ANCHOR
    
    # Initialize ProSE-X v2 pipeline
    if use_prosex:
        config = ProSEXv2Config(
            experiment_name="vllm_wsl_run",
            mqr_ulf=MQRULFConfig(
                anchor_neighbor_enabled=True,
                lexical_overlap_enabled=True,
                structural_recency_enabled=True,
                historical_success_enabled=True,
            ),
            burst=BurstConfig(
                enabled=True,
                radius=1,
                sticky_enabled=True,
                default_ttl=4,
            ),
            eabs=EABSConfig(
                exploration_ratio=0.2,
                budget_ratio_of_tail=budget_ratio,
                max_chunks_per_step=5,
            ),
            odus=ODUSConfig(mode="odus_x"),
        )
        
        pipeline = PromotionPipeline(config)
    else:
        pipeline = None
    
    # Metrics collector
    collector = DetailedMetricsCollector(gold_chunk_ids=gold_chunk_ids)
    
    # Simulate decode loop
    promoted_chunks = []
    
    for step in range(num_steps):
        # Create query context (gold signature at later steps to simulate need)
        if step >= num_steps // 2 and gold_chunk_id:
            # Simulate query attending to gold
            query_sig = np.ones(128, dtype=np.float32) * 0.8
        else:
            query_sig = np.random.randn(128).astype(np.float32) * 0.3
        
        query = QueryContext(
            request_id="passkey_test",
            step=step,
            query_signature=query_sig,
            active_anchor_ids=[c.chunk_id for c in anchor_chunks],
        )
        
        # Run pipeline
        if pipeline:
            result = pipeline.run(
                query=query,
                tail_chunks=tail_chunks,
                anchor_chunks=anchor_chunks,
                promoted_chunks=promoted_chunks,
            )
        else:
            # Fallback to simple simulation
            result = simulate_simple_step(query, tail_chunks, anchor_chunks, budget_ratio)
        
        # Collect metrics
        metrics = collector.collect_step(step, result)
        
        # Print detailed metrics
        if detailed_metrics:
            collector.print_step_summary(metrics)
        else:
            # Print brief progress
            if step % 5 == 0:
                print(f"Step {step}: ULF={metrics.ulf_candidates}, "
                      f"Scheduled={metrics.scheduler_selected}, "
                      f"Promoted={metrics.sticky_promoted}")
        
        # Update promoted chunks for next step
        if result.sticky_result:
            all_chunks = {c.chunk_id: c for c in chunks}
            promoted_chunks = [
                all_chunks[cid] for cid in result.sticky_result.promoted_ids
                if cid in all_chunks
            ]
            # Update tier
            for c in promoted_chunks:
                c.tier = ChunkTier.ANCHOR
    
    # Print final summary
    collector.print_final_summary()
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    collector.save_results(results_file)
    
    return {
        "output_dir": str(output_dir),
        "results_file": str(results_file),
        "num_steps": num_steps,
        "gold_chunk_id": gold_chunk_id,
    }


def simulate_simple_step(
    query: QueryContext,
    tail_chunks: List[ChunkMetadata],
    anchor_chunks: List[ChunkMetadata],
    budget_ratio: float,
) -> Any:
    """Simple simulation for baseline comparison."""
    from src.core_types import PromotionPipelineResult, ULFResult, ScorerResult, SchedulerResult, BurstResult, StickyResult
    
    # Simple ULF: random selection
    n_candidates = min(10, len(tail_chunks))
    candidate_ids = [c.chunk_id for c in tail_chunks[:n_candidates]]
    
    ulf_result = ULFResult(
        request_id=query.request_id,
        step=query.step,
        candidate_ids=candidate_ids,
        candidate_sources={cid: "random" for cid in candidate_ids},
        queue_contributions=[],
        n_tail_total=len(tail_chunks),
        n_candidates=n_candidates,
        per_queue_counts={"random": n_candidates},
        per_queue_unique={"random": n_candidates},
        ulf_latency_us=10.0,
    )
    
    # Simple scorer: random scores
    from src.core_types import ScoredCandidate
    scored = [
        ScoredCandidate(chunk_id=cid, score=0.5 + np.random.rand()*0.5, confidence=0.7)
        for cid in candidate_ids
    ]
    scored.sort(key=lambda x: x.score, reverse=True)
    
    scorer_result = ScorerResult(
        request_id=query.request_id,
        step=query.step,
        candidates=scored,
        n_input_candidates=n_candidates,
        n_scored=n_candidates,
        n_above_threshold=sum(1 for c in scored if c.score > 0.5),
        score_threshold=0.5,
        scorer_mode="random",
        scorer_latency_us=5.0,
    )
    
    # Simple scheduler: top k within budget
    budget = int(sum(c.logical_bytes for c in tail_chunks) * budget_ratio)
    selected = []
    used_bytes = 0
    
    for c in scored:
        chunk = next((x for x in tail_chunks if x.chunk_id == c.chunk_id), None)
        if chunk and used_bytes + chunk.logical_bytes <= budget:
            selected.append(c.chunk_id)
            used_bytes += chunk.logical_bytes
    
    scheduler_result = SchedulerResult(
        request_id=query.request_id,
        step=query.step,
        selected_ids=selected,
        selected_decisions=[],
        exploit_ids=selected,
        explore_ids=[],
        dropped_ids=[],
        dropped_decisions=[],
        budget_bytes=budget,
        used_bytes=used_bytes,
        utilization=used_bytes/max(budget,1),
        n_exploit=len(selected),
        n_explore=0,
        n_dropped_budget=0,
        n_dropped_low_score=0,
        n_dropped_low_confidence=0,
        scheduler_latency_us=2.0,
    )
    
    # Simple burst and sticky
    burst_result = BurstResult(
        request_id=query.request_id,
        step=query.step,
        input_ids=selected,
        burst_ids=selected,
        core_ids=selected,
        expansion_ids=[],
        burst_radius={cid: 0 for cid in selected},
        n_input=len(selected),
        n_burst_total=len(selected),
        n_expansion=0,
        burst_latency_us=1.0,
    )
    
    sticky_result = StickyResult(
        request_id=query.request_id,
        step=query.step,
        promoted_ids=selected,
        ttl_values={cid: 4 for cid in selected},
        ttl_updates=[],
        expired_ids=[],
        n_promoted=len(selected),
        n_expired=0,
        n_refreshed=0,
        avg_ttl=4.0,
        sticky_latency_us=1.0,
    )
    
    return PromotionPipelineResult(
        request_id=query.request_id,
        step=query.step,
        ulf_result=ulf_result,
        scorer_result=scorer_result,
        scheduler_result=scheduler_result,
        burst_result=burst_result,
        sticky_result=sticky_result,
        final_visible_ids=[c.chunk_id for c in anchor_chunks] + selected,
        final_active_bytes=used_bytes + sum(c.logical_bytes for c in anchor_chunks),
        final_promoted_bytes=used_bytes,
        total_latency_us=19.0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run vLLM with ProSE-X 2.0 detailed metrics in WSL"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=4096,
        help="Context length in tokens"
    )
    parser.add_argument(
        "--budget-ratio",
        type=float,
        default=0.1,
        help="Promotion budget ratio (0.0-1.0)"
    )
    parser.add_argument(
        "--workload-type",
        type=str,
        default="passkey",
        choices=["passkey", "multihop", "needle"],
        help="Type of workload"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/vllm_wsl",
        help="Output directory for results"
    )
    parser.add_argument(
        "--use-prose-v2",
        action="store_true",
        help="Use ProSE-X v2 with MQR-ULF and all features"
    )
    parser.add_argument(
        "--detailed-metrics",
        action="store_true",
        help="Print detailed metrics at each step"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of decode steps to simulate"
    )
    
    args = parser.parse_args()
    
    # Run simulation
    results = run_vllm_simulation(
        model=args.model,
        context_length=args.context_length,
        budget_ratio=args.budget_ratio,
        workload_type=args.workload_type,
        output_dir=args.output_dir,
        use_prosex=args.use_prosex,
        detailed_metrics=args.detailed_metrics,
        num_steps=args.num_steps,
    )
    
    print(f"\n{'='*70}")
    print(f"Run complete!")
    print(f"Results saved to: {results['results_file']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
