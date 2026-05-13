"""
Demo: Print metrics using the evaluation framework.

This script demonstrates how to use the evaluation framework
and prints actual metric values.
"""

import json
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.config import ProSEXv2Config
from src.eval.shared import SharedEvaluator, EvaluationResult
from src.eval.accounting import UsefulBytesAccountant, AccountingMode
from src.eval.failure_attribution import FailureAttributor
from src.eval.neighbor_recall_report import NeighborRecallReporter


def generate_demo_data(evaluator: SharedEvaluator, num_steps: int = 20):
    """Generate demo step data for testing."""
    random.seed(42)
    
    for step in range(num_steps):
        # Simulate gold chunks (evaluation only)
        gold_ids = {f"gold_{step}"} if step % 3 == 0 else set()
        
        # Simulate visible chunks
        visible_ids = {f"chunk_{step}", f"chunk_{step-1}"}
        if step % 3 == 0 and random.random() > 0.3:
            visible_ids.add(f"gold_{step}")  # Recovered
        
        # Simulate candidates
        candidate_ids = [f"cand_{i}" for i in range(10)]
        
        # Simulate selected and promoted
        selected_ids = [f"cand_{i}" for i in range(3)]
        promoted_ids = selected_ids + [f"burst_{i}" for i in range(2)]
        
        # Queue contributions
        queue_contribs = {
            "anchor_neighbor": {
                "raw_output": [f"nb_{i}" for i in range(4)],
                "post_dedup": [f"nb_{i}" for i in range(3)],
                "entering_scorer": [f"nb_{i}" for i in range(3)],
                "surviving_scheduler": [f"nb_0"],
                "burst_expanded": [f"nb_1"],
                "ultimately_useful": [f"nb_0"] if step % 3 == 0 else [],
            },
            "lexical_overlap": {
                "raw_output": [f"lo_{i}" for i in range(5)],
                "post_dedup": [f"lo_{i}" for i in range(4)],
                "entering_scorer": [f"lo_{i}" for i in range(4)],
                "surviving_scheduler": [f"lo_0", f"lo_1"],
                "burst_expanded": [],
                "ultimately_useful": [],
            },
        }
        
        evaluator.add_step_result(
            step=step,
            gold_chunk_ids=gold_ids,
            visible_chunk_ids=visible_ids,
            candidate_chunk_ids=candidate_ids,
            selected_chunk_ids=selected_ids,
            promoted_chunk_ids=promoted_ids,
            budget_bytes=10000,
            used_bytes=random.randint(3000, 8000),
            latency_us=random.uniform(50, 150),
            queue_contributions=queue_contribs,
        )
        
        # Add promotion records for useful bytes accounting
        for chunk_id in promoted_ids:
            evaluator.add_promotion_record(
                step=step,
                chunk_id=chunk_id,
                bytes_transferred=1024,
                queue_of_origin="anchor_neighbor" if "nb" in chunk_id else "lexical_overlap",
                score=random.uniform(0.5, 0.9),
                rank=1,
                selection_type="exploit",
                future_access_count=random.randint(0, 5),
                max_attention_weight=random.uniform(0, 0.1),
                gold_overlap_tokens=100 if chunk_id in gold_ids else 0,
                contributed_to_recovery=chunk_id in gold_ids,
            )


def print_metrics(result: EvaluationResult):
    """Print metrics in a formatted way."""
    print("\n" + "=" * 70)
    print("EVALUATION METRICS REPORT")
    print("=" * 70)
    
    print(f"\nExperiment ID: {result.experiment_id}")
    print(f"Timestamp: {result.timestamp}")
    
    print("\n" + "-" * 70)
    print("RECOVERY METRICS")
    print("-" * 70)
    print(f"  Conditional Recovery:     {result.conditional_recovery:.4f} ({result.conditional_recovery*100:.2f}%)")
    print(f"  No-Miss Rate:             {result.no_miss_rate:.4f} ({result.no_miss_rate*100:.2f}%)")
    print(f"  Total Steps:              {result.total_steps}")
    print(f"  Steps with Gold:          {result.steps_with_gold}")
    print(f"  Steps with Recovery:      {result.steps_with_recovery}")
    print(f"  Steps with Miss:          {result.steps_with_miss}")
    
    print("\n" + "-" * 70)
    print("USEFUL PROMOTE RATIO (UPR)")
    print("-" * 70)
    print(f"  Total Promoted Bytes:     {result.total_promoted_bytes:,}")
    print(f"  UPR (Attention-based):    {result.upr_attention_based:.4f} ({result.upr_attention_based*100:.2f}%)")
    print(f"  UPR (Gold-based):         {result.upr_gold_based:.4f} ({result.upr_gold_based*100:.2f}%)")
    print(f"  UPR (Recovery-based):     {result.upr_recovery_based:.4f} ({result.upr_recovery_based*100:.2f}%)")
    
    print("\n" + "-" * 70)
    print("BUDGET METRICS")
    print("-" * 70)
    print(f"  Budget Utilization:       {result.budget_utilization:.4f} ({result.budget_utilization*100:.2f}%)")
    print(f"  Total Budget Bytes:       {result.total_budget_bytes:,}")
    print(f"  Total Used Bytes:         {result.total_used_bytes:,}")
    
    print("\n" + "-" * 70)
    print("LATENCY METRICS")
    print("-" * 70)
    print(f"  Mean Latency:             {result.latency_mean_ms:.2f} ms")
    print(f"  P95 Latency:              {result.latency_p95_ms:.2f} ms")
    
    print("\n" + "-" * 70)
    print("QUEUE CONTRIBUTIONS")
    print("-" * 70)
    for queue_name, contrib in result.queue_contributions.items():
        print(f"\n  {queue_name}:")
        print(f"    Raw Output:             {contrib.raw_output_count}")
        print(f"    Post-Dedup:             {contrib.post_dedup_count}")
        print(f"    Entering Scorer:        {contrib.entering_scorer}")
        print(f"    Surviving Scheduler:    {contrib.surviving_scheduler}")
        print(f"    Burst Expanded:         {contrib.burst_expanded}")
        print(f"    Ultimately Useful:      {contrib.ultimately_useful}")
        if contrib.empty_reason:
            print(f"    Empty Reason:           {contrib.empty_reason}")
    
    print("\n" + "-" * 70)
    print("FAILURE ATTRIBUTION")
    print("-" * 70)
    fa = result.failure_attribution
    print(f"  Total Misses:             {fa.total_misses}")
    if fa.total_misses > 0:
        print(f"  Candidate Miss:           {fa.candidate_miss} ({fa.candidate_miss/fa.total_misses*100:.1f}%)")
        print(f"  Scorer Rank Miss:         {fa.scorer_rank_miss} ({fa.scorer_rank_miss/fa.total_misses*100:.1f}%)")
        print(f"  Scheduler Budget Drop:    {fa.scheduler_budget_drop} ({fa.scheduler_budget_drop/fa.total_misses*100:.1f}%)")
        print(f"  Scheduler Threshold Drop: {fa.scheduler_threshold_drop} ({fa.scheduler_threshold_drop/fa.total_misses*100:.1f}%)")
        print(f"  Burst Boundary Miss:      {fa.burst_boundary_miss} ({fa.burst_boundary_miss/fa.total_misses*100:.1f}%)")
        print(f"  Sticky Eviction Miss:     {fa.sticky_eviction_miss} ({fa.sticky_eviction_miss/fa.total_misses*100:.1f}%)")
        print(f"  Promoted But Unused:      {fa.promoted_but_unused} ({fa.promoted_but_unused/fa.total_misses*100:.1f}%)")
        print(f"  Retention Miss:           {fa.retention_miss} ({fa.retention_miss/fa.total_misses*100:.1f}%)")
    
    print("\n" + "=" * 70)


def print_useful_bytes_report(accountant: UsefulBytesAccountant):
    """Print useful bytes accounting report."""
    print("\n" + "=" * 70)
    print("USEFUL BYTES ACCOUNTING REPORT")
    print("=" * 70)
    
    report = accountant.get_accounting_report()
    
    print(f"\nTotal Units:        {report['total_units']}")
    print(f"Total Bytes:        {report['total_promoted_bytes']:,}")
    
    print("\n" + "-" * 70)
    print("BY ACCOUNTING MODE")
    print("-" * 70)
    
    for mode_name, mode_data in report['modes'].items():
        print(f"\n  {mode_name}:")
        print(f"    Useful Units:   {mode_data['useful_units']} / {mode_data['total_units']}")
        print(f"    Useful Bytes:   {mode_data['useful_bytes']:,} / {mode_data['total_bytes']:,}")
        print(f"    UPR:            {mode_data['upr']:.4f} ({mode_data['upr']*100:.2f}%)")


def print_neighbor_recall_report(reporter: NeighborRecallReporter):
    """Print neighbor recall report."""
    print("\n" + "=" * 70)
    print("NEIGHBOR RECALL CONTRIBUTION REPORT")
    print("=" * 70)
    
    report = reporter.generate_report()
    
    funnel = report.get('funnel', {})
    
    print("\n" + "-" * 70)
    print("PIPELINE FUNNEL")
    print("-" * 70)
    
    for stage_name, stage_data in funnel.items():
        print(f"\n  {stage_name}:")
        for key, value in stage_data.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")
    
    print("\n" + "-" * 70)
    print("EMPTY REASON ANALYSIS")
    print("-" * 70)
    empty_analysis = report.get('empty_reason_analysis', {})
    print(f"  Steps with Empty: {empty_analysis.get('total_steps_with_empty', 0)}")
    for reason, count in empty_analysis.get('breakdown', {}).items():
        print(f"    {reason}: {count}")
    
    print("\n" + "-" * 70)
    print("KEY FINDINGS")
    print("-" * 70)
    for finding in report.get('key_findings', []):
        print(f"  - {finding}")


def main():
    """Run demo and print metrics."""
    print("\n" + "=" * 70)
    print("ProSE-X 2.0 Evaluation Framework - Metrics Demo")
    print("=" * 70)
    
    # Create config
    config = ProSEXv2Config(seed=42)
    
    # Create evaluator
    evaluator = SharedEvaluator(
        config=config.to_dict(),
        experiment_id="demo_experiment"
    )
    
    # Generate demo data
    print("\nGenerating demo data...")
    generate_demo_data(evaluator, num_steps=20)
    
    # Evaluate
    print("Computing metrics...")
    result = evaluator.evaluate()
    
    # Print metrics
    print_metrics(result)
    
    # Also demonstrate other components
    print("\n\n" + "=" * 70)
    print("ADDITIONAL COMPONENT REPORTS")
    print("=" * 70)
    
    # Useful bytes accounting demo
    accountant = UsefulBytesAccountant()
    for record in evaluator.promotion_records:
        accountant.record_promotion(
            chunk_id=record.get("chunk_id", ""),
            step=record.get("step", 0),
            request_id="demo",
            bytes_transferred=record.get("bytes_transferred", 0),
            queue_of_origin=record.get("queue_of_origin"),
            score=record.get("score", 0),
            rank=record.get("rank", 0),
            selection_type=record.get("selection_type"),
        )
        # Simulate access
        if record.get("future_access_count", 0) > 0:
            accountant.record_access(
                chunk_id=record.get("chunk_id", ""),
                step=record.get("step", 0) + 1,
                attention_weight=record.get("max_attention_weight", 0),
            )
    
    print_useful_bytes_report(accountant)
    
    # Save to file
    output_path = "outputs/reports/demo_metrics_result.json"
    result.save(output_path)
    print(f"\n\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
