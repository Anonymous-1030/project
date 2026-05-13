"""
Run and print metrics for HPCA submission.

This script provides a lightweight, reproducible demonstration of:
1. CXL-aware prefetch protocol decisions
2. Bandwidth-utility Pareto frontier extraction
3. Per-step incremental promotion cycle model (speedup > 1)
4. Literature-backed hardware area/power estimates
5. Formal theory (approximation bounds, prefetch theorem)
6. Promotion Gap analysis (Figure 1 motivation)
7. MQR-ULF recall ablation

It avoids requiring a full GPU-backed model run, while still exercising the
actual analytical/runtime code paths added for the HPCA artifact.
"""

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.hardware.kv_cache_roofline import KVCacheRoofline, HARDWARE_PRESETS
from src.hardware_model import (
    KCMCHardwareModel,
    CycleAnalyticalModel,
    NearCXLDecompressor,
)
from src.memory.two_tier_manager import (
    TwoTierMemoryManager,
    PromotionPrefetchEngine,
    MemoryTier,
)
from src.theory import (
    FormalKVCacheRoofline,
    OptimalPromotionSolver,
    GreedyApproximation,
    AttentionPrefetchModel,
)
from src.theory.kv_roofline_formal import ChunkUtilityRecord
from src.theory.optimal_promotion import PromotionItem
from src.theory.attention_prefetch_theory import PrefetchObservation
from src.benchmarks.promotion_gap import PromotionGapExperiment
from src.runners.mqr_ablation import MQRULFAblation


def build_demo_memory_manager() -> TwoTierMemoryManager:
    manager = TwoTierMemoryManager(
        num_layers=24,
        num_kv_heads=8,
        head_dim=128,
        hbm_capacity_gb=40.0,
        dram_capacity_gb=256.0,
        device="cpu",
    )

    # Register some tail chunks with positions so byte estimation works without
    # needing a real KV tensor allocation.
    chunk_len = 256
    for cid in range(12):
        manager.chunk_positions[cid] = (cid * chunk_len, (cid + 1) * chunk_len)
        manager.chunk_tiers[cid] = MemoryTier.HOST_DRAM
        k = torch.randn(24, 1, 8, chunk_len, 128, dtype=torch.float16)
        v = torch.randn(24, 1, 8, chunk_len, 128, dtype=torch.float16)
        manager.tail_store.store_chunk(cid, k, v)
    return manager


def run_fix10_demo() -> dict:
    manager = build_demo_memory_manager()
    engine = PromotionPrefetchEngine(manager, lookahead_steps=2)

    recent_attention = [
        {0: 0.03, 1: 0.05, 2: 0.07, 3: 0.02},
        {0: 0.04, 1: 0.08, 2: 0.11, 3: 0.03},
        {0: 0.05, 1: 0.12, 2: 0.16, 3: 0.04},
    ]

    predicted = engine.predict_future_needs(current_step=7, recent_attention_patterns=recent_attention)
    result = engine.schedule_prefetch(predicted_chunks=predicted, current_compute_time_us=120.0)
    decision = engine.last_decision
    memory = manager.get_memory_breakdown()

    return {
        "predicted_chunks": predicted,
        "transfer_path": result.transfer_path,
        "prefetch_depth": result.prefetch_depth,
        "bytes_promoted": result.bytes_promoted,
        "overlapped": result.overlapped,
        "estimated_hidden_us": result.estimated_hidden_us,
        "expected_transfer_us": None if decision is None else decision.expected_transfer_us,
        "expected_exposed_us": None if decision is None else decision.expected_exposed_us,
        "memory_breakdown": memory,
    }


def run_fix11_demo() -> dict:
    roofline = KVCacheRoofline(HARDWARE_PRESETS["H100-PCIe"])

    def utility_fn(ratio: float) -> float:
        # Saturating utility curve: early promoted bytes are most useful.
        return 1.0 - math.exp(-9.0 * ratio)

    def throughput_fn(ratio: float) -> float:
        # Throughput gently declines as budget grows due to transfer pressure.
        return 9500.0 - 4200.0 * ratio

    analysis = roofline.sweep_promotion_budgets(
        seq_len=32768,
        utility_fn=utility_fn,
        throughput_fn=throughput_fn,
        interconnect="cxl",
        decode_time_us=120.0,
        label_prefix="prose",
    )

    summary = roofline.summarize_pareto_analysis(analysis)
    summary["pareto_frontier"] = [
        {
            "label": p.label,
            "budget_ratio": p.budget_ratio,
            "utility": p.utility,
            "utility_per_byte": p.utility_per_byte,
            "throughput": p.throughput,
            "exposed_latency_us": p.exposed_latency_us,
        }
        for p in analysis.pareto_frontier
    ]
    return summary


def run_kcmc_hardware_demo() -> dict:
    model = KCMCHardwareModel()
    return model.summary()


def run_cycle_model_demo() -> dict:
    model = CycleAnalyticalModel(
        interconnect_bandwidth_gbps=64.0,
        chunks_per_step=3,
        chunk_size_tokens=64,
        decompressor=NearCXLDecompressor(compressed_bits=4),
    )
    baseline = model.model_baseline_latency(seq_len=32768, num_layers=32, num_heads=8, head_dim=128)
    kcmc = model.model_kcmc_latency(
        seq_len=32768,
        retention_ratio=0.18,
        promotion_ratio=0.08,
        num_layers=32,
        num_heads=8,
        head_dim=128,
    )
    return model.summarize_comparison(baseline, kcmc)


def run_theory_demo() -> dict:
    formal = FormalKVCacheRoofline()
    records = [
        ChunkUtilityRecord(chunk_id=0, utility=0.92, attention_mass=0.20, bytes_transferred=4096),
        ChunkUtilityRecord(chunk_id=1, utility=0.78, attention_mass=0.16, bytes_transferred=4096),
        ChunkUtilityRecord(chunk_id=2, utility=0.61, attention_mass=0.10, bytes_transferred=2048),
    ]
    roofline_summary = formal.summarize(records, compute_ceiling=50000.0, bandwidth_ceiling=51.2)

    items = [
        PromotionItem(chunk_id=0, utility=0.92, bytes_cost=4096, hbm_cost=4096),
        PromotionItem(chunk_id=1, utility=0.78, bytes_cost=4096, hbm_cost=4096),
        PromotionItem(chunk_id=2, utility=0.61, bytes_cost=2048, hbm_cost=2048),
        PromotionItem(chunk_id=3, utility=0.45, bytes_cost=1024, hbm_cost=1024),
    ]
    optimal = OptimalPromotionSolver().solve(items, bandwidth_budget=8192, hbm_budget=8192)
    greedy_solver = GreedyApproximation()
    greedy = greedy_solver.solve(items, bandwidth_budget=8192, hbm_budget=8192)

    prefetch = AttentionPrefetchModel(delta=1, threshold=0.1)
    prefetch_eval = prefetch.evaluate([
        PrefetchObservation(promoted_chunk=3, target_chunk=4, observed_attention=0.18, was_useful=True),
        PrefetchObservation(promoted_chunk=4, target_chunk=5, observed_attention=0.15, was_useful=True),
        PrefetchObservation(promoted_chunk=5, target_chunk=7, observed_attention=0.09, was_useful=False),
        PrefetchObservation(promoted_chunk=7, target_chunk=8, observed_attention=0.12, was_useful=False),
    ])
    prefetch_eval["theorems"] = prefetch.formal_theorem()

    return {
        "formal_roofline": roofline_summary,
        "optimal_promotion": optimal,
        "greedy_promotion": greedy,
        "greedy_ratio": greedy_solver.approximation_ratio(optimal["objective"], greedy["objective"]),
        "greedy_theoretical_lower_bound": greedy_solver.theoretical_lower_bound(),
        "greedy_theorems": greedy_solver.formal_theorem(),
        "attention_prefetch": prefetch_eval,
    }


def run_promotion_gap_demo() -> dict:
    experiment = PromotionGapExperiment()
    experiment.run_analytical()
    return experiment.summarize()


def run_mqr_ablation_demo() -> dict:
    ablation = MQRULFAblation()
    result = ablation.run_analytical()
    return result.to_dict()


def main() -> None:
    fix10 = run_fix10_demo()
    fix11 = run_fix11_demo()
    kcmc = run_kcmc_hardware_demo()
    cycle = run_cycle_model_demo()
    theory = run_theory_demo()
    promotion_gap = run_promotion_gap_demo()
    mqr_ablation = run_mqr_ablation_demo()

    print("=" * 72)
    print("HPCA FULL METRICS REPORT")
    print("=" * 72)


    print("\n[Fix 10] CXL-aware Prefetch Protocol")
    print(f"  Predicted chunks:         {fix10['predicted_chunks']}")
    print(f"  Selected transfer path:   {fix10['transfer_path']}")
    print(f"  Prefetch depth:           {fix10['prefetch_depth']}")
    print(f"  Bytes promoted:           {fix10['bytes_promoted']:,}")
    print(f"  Overlapped with compute:  {fix10['overlapped']}")
    print(f"  Hidden transfer time:     {fix10['estimated_hidden_us']:.2f} us")
    print(f"  Expected transfer time:   {fix10['expected_transfer_us']:.2f} us")
    print(f"  Exposed latency:          {fix10['expected_exposed_us']:.2f} us")
    print(f"  Avg prefetch depth:       {fix10['memory_breakdown']['avg_prefetch_depth']:.2f}")
    print(f"  CXL prefetch hit rate:    {fix10['memory_breakdown']['cxl_prefetch_hit_rate']:.2%}")

    print("\n[Fix 11] Bandwidth-Utility Pareto Analysis")
    print(f"  Hardware:                 {fix11['hardware']}")
    print(f"  Interconnect:             {fix11['interconnect']}")
    print(f"  Swept points:             {fix11['num_points']}")
    print(f"  Pareto points:            {fix11['pareto_count']}")
    if fix11["recommended"] is not None:
        rec = fix11["recommended"]
        print(f"  Recommended point:        {rec['label']}")
        print(f"  Recommended budget:       {rec['budget_ratio']:.2%}")
        print(f"  Utility:                  {rec['utility']:.4f}")
        print(f"  Utility/byte:             {rec['utility_per_byte']:.8f}")
        print(f"  Throughput:               {rec['throughput']:.2f}")
        print(f"  Exposed latency:          {rec['exposed_latency_us']:.2f} us")

    print("\n  Pareto frontier:")
    for point in fix11["pareto_frontier"]:
        print(
            "    - {label}: budget={budget:.2%}, utility={utility:.4f}, "
            "throughput={throughput:.2f}, exposed={exposed:.2f} us".format(
                label=point["label"],
                budget=point["budget_ratio"],
                utility=point["utility"],
                throughput=point["throughput"],
                exposed=point["exposed_latency_us"],
            )
        )

    print("\n[KCMC] Hardware Feasibility")
    print(f"  Total area overhead:       {kcmc['total_area_mm2']:.5f} mm^2 ({kcmc['area_overhead_percent']:.5f}%)")
    print(f"  Total power overhead:      {kcmc['total_power_w']:.5f} W ({kcmc['power_overhead_percent']:.5f}%)")
    for comp in kcmc["components"]:
        print(
            f"    - {comp['name']}: area={comp['area_mm2']:.5f} mm^2, power={comp['power_mw']:.2f} mW, latency={comp['latency_ns']:.2f} ns"
        )

    print("\n[KCMC] Cycle-Analytical Performance Model")
    print(f"  Baseline latency:          {cycle['baseline_total_us']:.2f} us")
    print(f"  KCMC latency:              {cycle['kcmc_total_us']:.2f} us")
    print(f"  Speedup:                   {cycle['speedup']:.3f}x")
    print(f"  Hidden transfer:           {cycle['kcmc_overlap_hidden_us']:.2f} us")
    print(f"  Exposed transfer:          {cycle['kcmc_exposed_transfer_us']:.2f} us")

    print("\n[Theory] Formal Roofline + Admission + Prefetch")
    print(f"  Formal OI:                 {theory['formal_roofline']['operational_intensity']:.8f}")
    print(f"  Bound classification:      {theory['formal_roofline']['bound']}")
    print(f"  Optimal objective:         {theory['optimal_promotion']['objective']:.4f}")
    print(f"  Greedy objective:          {theory['greedy_promotion']['objective']:.4f}")
    print(f"  Greedy approximation:      {theory['greedy_ratio']:.4f}")
    print(f"  Greedy lower bound (Thm2): {theory['greedy_theoretical_lower_bound']:.2f}")
    print(f"  Prefetch accuracy:         {theory['attention_prefetch']['accuracy']:.2%}")
    print(f"  Prefetch coverage:         {theory['attention_prefetch']['coverage']:.2%}")

    print("\n[Promotion Gap] Figure 1 Motivation")
    if "measurements" in promotion_gap:
        print(f"  {'Budget':>8} {'RetGap':>8} {'ProGap':>8} {'Dominance':>10} {'Recovery':>10}")
        for m in promotion_gap["measurements"]:
            print(
                f"  {m['budget_ratio']:>7.0%} "
                f"{m['retention_gap']:>8.3f} "
                f"{m['promotion_gap']:>8.3f} "
                f"{m['promotion_gap_dominance']:>9.1%} "
                f"{m['prose_recovery_ratio']:>9.1%}"
            )
        print(f"  Crossover ratio:           {promotion_gap.get('crossover_budget_ratio')}")
        print(f"  Avg gap dominance:         {promotion_gap.get('avg_promotion_gap_dominance', 0):.1%}")
        print(f"  Avg ProSE recovery:        {promotion_gap.get('avg_prose_recovery_ratio', 0):.1%}")

    print("\n[MQR-ULF] Recall Ablation")
    if "results" in mqr_ablation:
        print(f"  {'Config':<25} {'Union':>6} {'R@5':>6}")
        for r in mqr_ablation["results"]:
            print(f"  {r['config_name']:<25} {r['union_recall']:>6.3f} {r['recall_at_5']:>6.3f}")
        print(f"  Most critical queue:       {mqr_ablation.get('most_critical_queue')}")
        print(f"  Max recall drop:           {mqr_ablation.get('max_recall_drop', 0):.3f}")

    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hpca_fixes_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "fix10": fix10,
            "fix11": fix11,
            "kcmc_hardware": kcmc,
            "cycle_model": cycle,
            "theory": theory,
            "promotion_gap": promotion_gap,
            "mqr_ablation": mqr_ablation,
        }, f, indent=2)

    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()