"""
Unified Reviewer-Response Experiment Suite.

Runs the 5 highest-priority experiments that directly address
likely rejection reasons:

1. SW-SBFI baseline (proves hardware CE is necessary)
2. Perfect Oracle bounds (decomposes contribution)
3. Deterministic/Correlated scorer (validates rescoring assumption)
4. Chunk/Summary size sensitivity (empirical validation of 64B/64KB)
5. Query-sketch ablation (validates Table IV jump)

Usage:
    python -m prosex.src.runners.reviewer_response_experiments

All experiments use the SAME workload traces, SAME CXL simulator,
and SAME evaluation metrics for fair comparison.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, "d:/LLM")

from src.memory.cxl_queue_simulator import CXLQueueConfig
from src.baselines.prose_sbfi import PROSEPolicy
from src.baselines.prose_fts import PROSEFTSPolicy
from src.baselines.sw_sbfi import SWSBFIPolicy
from src.baselines.perfect_oracles import (
    PerfectFTSOraclePolicy, PerfectSBFIOraclePolicy
)
from src.baselines.scorer_noise_experiments import (
    DeterministicScorerPolicy, CorrelatedErrorScorerPolicy, ScorerNoiseConfig
)
from src.baselines.size_sensitivity import ChunkSizeSensitivityPolicy
from src.baselines.query_sketch_ablation import (
    QuerySketchAblationPolicy, QuerySketchConfig
)


# ── Synthetic Workload Generator ──────────────────────────────────────

class WorkloadGenerator:
    """Generate realistic attention trace workloads for experiments.

    Produces attention patterns that model real LLM behavior:
      - Sparse attention (most chunks get near-zero mass)
      - Temporal locality (recently accessed chunks stay hot)
      - Needle patterns (rare high-attention on distant chunks)
      - Topic drift (attention focus shifts over generation)
    """

    def __init__(
        self,
        num_chunks: int = 64,
        num_steps: int = 100,
        sparsity: float = 0.85,
        locality_strength: float = 0.6,
        needle_probability: float = 0.05,
        drift_rate: float = 0.02,
        seed: int = 42,
    ):
        self.num_chunks = num_chunks
        self.num_steps = num_steps
        self.sparsity = sparsity
        self.locality_strength = locality_strength
        self.needle_probability = needle_probability
        self.drift_rate = drift_rate
        self.rng = np.random.default_rng(seed)

        # Pre-generate trace
        self._trace: List[np.ndarray] = []
        self._generate_trace()

    def _generate_trace(self):
        """Generate a full attention trace."""
        # Initialize: attention concentrated on first/last chunks
        prev_attn = np.zeros(self.num_chunks)
        prev_attn[0] = 0.3
        prev_attn[-1] = 0.3
        focus_center = self.num_chunks // 2

        for step in range(self.num_steps):
            attn = np.zeros(self.num_chunks)

            # Base: sparse random attention
            active_count = max(3, int(self.num_chunks * (1 - self.sparsity)))
            active_ids = self.rng.choice(self.num_chunks, size=active_count, replace=False)
            attn[active_ids] = self.rng.exponential(0.1, size=active_count)

            # Temporal locality: blend with previous step
            attn = (1 - self.locality_strength) * attn + self.locality_strength * prev_attn

            # Topic drift: shift focus center
            focus_center += self.rng.normal(0, self.drift_rate * self.num_chunks)
            focus_center = np.clip(focus_center, 0, self.num_chunks - 1)
            focus_idx = int(focus_center)
            # Gaussian bump around focus
            positions = np.arange(self.num_chunks)
            focus_bump = np.exp(-0.5 * ((positions - focus_idx) / max(3, self.num_chunks * 0.05)) ** 2)
            attn += 0.2 * focus_bump

            # Needle: rare high-attention on distant chunk
            if self.rng.random() < self.needle_probability:
                needle_id = self.rng.integers(0, self.num_chunks)
                attn[needle_id] += self.rng.uniform(0.5, 1.0)

            # Normalize
            attn = np.maximum(attn, 0)
            total = attn.sum()
            if total > 0:
                attn /= total

            self._trace.append(attn)
            prev_attn = attn.copy()

    def get_step(self, step: int) -> Dict[int, float]:
        """Get attention masses for a given step."""
        if step >= len(self._trace):
            step = step % len(self._trace)
        attn = self._trace[step]
        return {i: float(attn[i]) for i in range(self.num_chunks) if attn[i] > 1e-6}

    def get_future_attention(self, step: int) -> np.ndarray:
        """Get next-step attention (for oracle policies)."""
        next_step = min(step + 1, len(self._trace) - 1)
        return self._trace[next_step]


# ── Experiment Runner ─────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    policy_name: str
    mean_recovery: float
    invalid_traffic_ratio: float
    mean_latency_us: float
    p50_latency_us: float
    p95_latency_us: float
    p99_latency_us: float
    total_payload_bytes: int
    total_metadata_bytes: int
    mean_queue_rho: float
    sw_overhead_us: float = 0.0  # SW-SBFI only
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


def run_single_policy(
    policy,
    workload: WorkloadGenerator,
    num_chunks: int = 64,
    budget_chunks: int = 8,
    anchor_ratio: float = 0.1,
) -> ExperimentResult:
    """Run a single policy on a workload and collect metrics."""
    policy.reset()

    num_anchors = max(1, int(num_chunks * anchor_ratio))
    anchor_ids = list(range(num_anchors // 2)) + list(range(num_chunks - num_anchors // 2, num_chunks))

    latencies = []

    for step in range(workload.num_steps):
        masses = workload.get_step(step)

        # Provide future attention to oracle policies
        if hasattr(policy, "set_future_attention"):
            policy.set_future_attention(workload.get_future_attention(step))

        t0 = time.perf_counter_ns()
        policy.select_active_chunks(
            num_chunks=num_chunks,
            budget_chunks=budget_chunks,
            chunk_attention_masses=masses,
            anchor_ids=anchor_ids,
            step=step,
        )
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1000.0)  # μs

    # Collect metrics
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    total_payload = 0
    total_meta = 0
    queue_rhos = []

    if hasattr(policy, "cxl_session") and policy.cxl_session:
        for r in policy.cxl_session.step_results:
            total_payload += r.cxl_stats.payload_bytes_fetched
            total_meta += r.cxl_stats.summary_bytes_fetched
            queue_rhos.append(r.cxl_stats.queue_utilization_rho)

    sw_overhead = 0.0
    if hasattr(policy, "get_mean_step_overhead_us"):
        sw_overhead = policy.get_mean_step_overhead_us()

    return ExperimentResult(
        policy_name=policy.name,
        mean_recovery=policy.get_mean_recovery() if hasattr(policy, "get_mean_recovery") else 0.0,
        invalid_traffic_ratio=policy.get_invalid_traffic_ratio() if hasattr(policy, "get_invalid_traffic_ratio") else 0.0,
        mean_latency_us=float(np.mean(latencies)),
        p50_latency_us=latencies_sorted[int(n * 0.50)] if n > 0 else 0.0,
        p95_latency_us=latencies_sorted[int(min(n - 1, n * 0.95))] if n > 0 else 0.0,
        p99_latency_us=latencies_sorted[int(min(n - 1, n * 0.99))] if n > 0 else 0.0,
        total_payload_bytes=total_payload,
        total_metadata_bytes=total_meta,
        mean_queue_rho=float(np.mean(queue_rhos)) if queue_rhos else 0.0,
        sw_overhead_us=sw_overhead,
    )


# ── Experiment 1: SW-SBFI vs PROSE ────────────────────────────────────

def run_experiment_1_sw_sbfi(
    num_chunks: int = 64,
    num_steps: int = 100,
    budget_chunks: int = 8,
) -> Dict[str, ExperimentResult]:
    """Experiment 1: Software SBFI baseline vs PROSE hardware.

    Proves whether the hardware CE is necessary by comparing:
      - PROSE-FTS (fetch-then-score, no SBFI)
      - SW-SBFI (software score-before-fetch)
      - PROSE (hardware score-before-fetch)
    """
    print("=" * 70)
    print("EXPERIMENT 1: SW-SBFI vs PROSE (Hardware Necessity)")
    print("=" * 70)

    workload = WorkloadGenerator(num_chunks=num_chunks, num_steps=num_steps)
    cxl_config = CXLQueueConfig()

    policies = {
        "PROSE-FTS": PROSEFTSPolicy(cxl_config=cxl_config),
        "SW-SBFI": SWSBFIPolicy(cxl_config=cxl_config),
        "PROSE": PROSEPolicy(cxl_config=cxl_config),
    }

    results = {}
    for name, policy in policies.items():
        print(f"  Running {name}...")
        results[name] = run_single_policy(policy, workload, num_chunks, budget_chunks)
        r = results[name]
        print(f"    Recovery: {r.mean_recovery:.4f}")
        print(f"    Invalid traffic: {r.invalid_traffic_ratio:.4f}")
        print(f"    Queue ρ: {r.mean_queue_rho:.4f}")
        if r.sw_overhead_us > 0:
            print(f"    SW overhead: {r.sw_overhead_us:.2f} μs/step")

    # Analysis
    print("\n  ANALYSIS:")
    prose_recovery = results["PROSE"].mean_recovery
    sw_recovery = results["SW-SBFI"].mean_recovery
    fts_recovery = results["PROSE-FTS"].mean_recovery
    print(f"    SBFI ordering benefit (SW-SBFI vs FTS): {sw_recovery - fts_recovery:+.4f}")
    print(f"    Hardware benefit (PROSE vs SW-SBFI): {prose_recovery - sw_recovery:+.4f}")
    print(f"    Total PROSE benefit vs FTS: {prose_recovery - fts_recovery:+.4f}")

    return results


# ── Experiment 2: Perfect Oracle Bounds ───────────────────────────────

def run_experiment_2_oracle_bounds(
    num_chunks: int = 64,
    num_steps: int = 100,
    budget_chunks: int = 8,
) -> Dict[str, ExperimentResult]:
    """Experiment 2: Perfect oracle bounds decompose the contribution.

    Comparison ladder:
      PROSE-FTS < SW-SBFI < PROSE < Perfect-FTS < Perfect-SBFI
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Perfect Oracle Bounds (Contribution Decomposition)")
    print("=" * 70)

    workload = WorkloadGenerator(num_chunks=num_chunks, num_steps=num_steps)
    cxl_config = CXLQueueConfig()

    policies = {
        "PROSE-FTS": PROSEFTSPolicy(cxl_config=cxl_config),
        "SW-SBFI": SWSBFIPolicy(cxl_config=cxl_config),
        "PROSE": PROSEPolicy(cxl_config=cxl_config),
        "Perfect-FTS": PerfectFTSOraclePolicy(cxl_config=cxl_config),
        "Perfect-SBFI": PerfectSBFIOraclePolicy(cxl_config=cxl_config),
    }

    results = {}
    for name, policy in policies.items():
        print(f"  Running {name}...")
        results[name] = run_single_policy(policy, workload, num_chunks, budget_chunks)
        r = results[name]
        print(f"    Recovery: {r.mean_recovery:.4f}  Invalid: {r.invalid_traffic_ratio:.4f}")

    print("\n  CONTRIBUTION DECOMPOSITION:")
    recoveries = {k: v.mean_recovery for k, v in results.items()}
    print(f"    Ranker quality (PROSE-FTS → PROSE): {recoveries['PROSE'] - recoveries['PROSE-FTS']:+.4f}")
    print(f"    SBFI ordering (PROSE-FTS → SW-SBFI): {recoveries['SW-SBFI'] - recoveries['PROSE-FTS']:+.4f}")
    print(f"    HW acceleration (SW-SBFI → PROSE): {recoveries['PROSE'] - recoveries['SW-SBFI']:+.4f}")
    print(f"    Scorer gap (PROSE → Perfect-FTS): {recoveries['Perfect-FTS'] - recoveries['PROSE']:+.4f}")
    print(f"    Ordering gap (Perfect-FTS → Perfect-SBFI): {recoveries['Perfect-SBFI'] - recoveries['Perfect-FTS']:+.4f}")

    return results


# ── Experiment 3: Scorer Noise Validation ─────────────────────────────

def run_experiment_3_scorer_noise(
    num_chunks: int = 64,
    num_steps: int = 100,
    budget_chunks: int = 8,
) -> Dict[str, ExperimentResult]:
    """Experiment 3: Validate rescoring assumption.

    Tests whether cascade benefit comes from noise independence or real signal.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Scorer Noise Validation (Rescoring Assumption)")
    print("=" * 70)

    workload = WorkloadGenerator(num_chunks=num_chunks, num_steps=num_steps)
    cxl_config = CXLQueueConfig()

    policies = {
        "Deterministic-NoCascade": DeterministicScorerPolicy(
            cxl_config=cxl_config, enable_cascade=False
        ),
        "Deterministic-Cascade": DeterministicScorerPolicy(
            cxl_config=cxl_config, enable_cascade=True
        ),
        "Correlated-ρ=0.0": CorrelatedErrorScorerPolicy(
            cxl_config=cxl_config, correlation=0.0
        ),
        "Correlated-ρ=0.3": CorrelatedErrorScorerPolicy(
            cxl_config=cxl_config, correlation=0.3
        ),
        "Correlated-ρ=0.5": CorrelatedErrorScorerPolicy(
            cxl_config=cxl_config, correlation=0.5
        ),
        "Correlated-ρ=0.7": CorrelatedErrorScorerPolicy(
            cxl_config=cxl_config, correlation=0.7
        ),
        "Correlated-ρ=0.9": CorrelatedErrorScorerPolicy(
            cxl_config=cxl_config, correlation=0.9
        ),
        "Correlated-ρ=1.0": CorrelatedErrorScorerPolicy(
            cxl_config=cxl_config, correlation=1.0
        ),
    }

    results = {}
    for name, policy in policies.items():
        print(f"  Running {name}...")
        results[name] = run_single_policy(policy, workload, num_chunks, budget_chunks)
        print(f"    Recovery: {results[name].mean_recovery:.4f}")

    print("\n  ANALYSIS:")
    det_no = results["Deterministic-NoCascade"].mean_recovery
    det_yes = results["Deterministic-Cascade"].mean_recovery
    print(f"    Deterministic cascade benefit: {det_yes - det_no:+.4f}")
    if abs(det_yes - det_no) < 0.01:
        print("    → Cascade adds NO value with deterministic scoring (expected)")
        print("    → Cascade benefit in paper comes from noise independence")
    else:
        print("    → Cascade adds value even deterministically (additional context)")

    print(f"\n    Correlation sweep (cascade benefit vs correlation):")
    for rho in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
        name = f"Correlated-ρ={rho}"
        print(f"      ρ={rho}: recovery={results[name].mean_recovery:.4f}")

    return results


# ── Experiment 4: Size Sensitivity ────────────────────────────────────

def run_experiment_4_size_sensitivity(
    num_chunks: int = 64,
    num_steps: int = 100,
    budget_chunks: int = 8,
) -> Dict[str, ExperimentResult]:
    """Experiment 4: Chunk-size and summary-size sensitivity sweep."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Chunk/Summary Size Sensitivity")
    print("=" * 70)

    workload = WorkloadGenerator(num_chunks=num_chunks, num_steps=num_steps)

    chunk_sizes = [16384, 32768, 65536, 131072, 262144]  # 16KB to 256KB
    summary_sizes = [16, 32, 64, 128, 256]

    results = {}

    # Sweep chunk size (fixed summary = 64B)
    print("\n  Chunk size sweep (summary=64B):")
    for cs in chunk_sizes:
        name = f"chunk={cs//1024}KB"
        policy = ChunkSizeSensitivityPolicy(
            chunk_size_bytes=cs, summary_size_bytes=64
        )
        results[name] = run_single_policy(policy, workload, num_chunks, budget_chunks)
        r = results[name]
        print(f"    {name}: recovery={r.mean_recovery:.4f} invalid_ratio={r.invalid_traffic_ratio:.4f} ρ={r.mean_queue_rho:.4f}")

    # Sweep summary size (fixed chunk = 64KB)
    print("\n  Summary size sweep (chunk=64KB):")
    for ss in summary_sizes:
        name = f"summary={ss}B"
        policy = ChunkSizeSensitivityPolicy(
            chunk_size_bytes=65536, summary_size_bytes=ss
        )
        results[name] = run_single_policy(policy, workload, num_chunks, budget_chunks)
        r = results[name]
        print(f"    {name}: recovery={r.mean_recovery:.4f} invalid_ratio={r.invalid_traffic_ratio:.4f}")

    return results


# ── Experiment 5: Query-Sketch Ablation ───────────────────────────────

def run_experiment_5_query_sketch(
    num_chunks: int = 64,
    num_steps: int = 100,
    budget_chunks: int = 8,
) -> Dict[str, Any]:
    """Experiment 5: Query-sketch size and method ablation."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Query-Sketch Ablation")
    print("=" * 70)

    workload = WorkloadGenerator(num_chunks=num_chunks, num_steps=num_steps)
    cxl_config = CXLQueueConfig()

    results = {}

    # Size sweep
    print("\n  Sketch size sweep (method=random_proj):")
    for size in [0, 4, 8, 16, 32, 64]:
        cfg = QuerySketchConfig(sketch_size_bytes=size, sketch_method="random_proj")
        policy = QuerySketchAblationPolicy(sketch_config=cfg, cxl_config=cxl_config)
        result = run_single_policy(policy, workload, num_chunks, budget_chunks)
        ablation = policy.get_ablation_result()
        name = f"size={size}B"
        results[name] = {
            "experiment": result.to_dict(),
            "ablation": ablation.to_dict(),
        }
        print(f"    {name}: recovery={result.mean_recovery:.4f} "
              f"improvement={ablation.recovery_improvement_vs_no_sketch:+.4f} "
              f"failure_rate={ablation.failure_rate:.3f}")

    # Method sweep (fixed size=16B)
    print("\n  Sketch method sweep (size=16B):")
    for method in ["random_proj", "learned_proj", "per_layer", "per_head", "shared"]:
        cfg = QuerySketchConfig(sketch_size_bytes=16, sketch_method=method)
        policy = QuerySketchAblationPolicy(sketch_config=cfg, cxl_config=cxl_config)
        result = run_single_policy(policy, workload, num_chunks, budget_chunks)
        ablation = policy.get_ablation_result()
        name = f"method={method}"
        results[name] = {
            "experiment": result.to_dict(),
            "ablation": ablation.to_dict(),
        }
        print(f"    {name}: recovery={result.mean_recovery:.4f} "
              f"improvement={ablation.recovery_improvement_vs_no_sketch:+.4f}")

    # Staleness sweep (fixed size=16B, method=random_proj)
    print("\n  Sketch staleness sweep (size=16B):")
    for staleness in [0, 1, 2, 4, 8, 16]:
        cfg = QuerySketchConfig(sketch_size_bytes=16, sketch_staleness=staleness)
        policy = QuerySketchAblationPolicy(sketch_config=cfg, cxl_config=cxl_config)
        result = run_single_policy(policy, workload, num_chunks, budget_chunks)
        ablation = policy.get_ablation_result()
        name = f"stale={staleness}"
        results[name] = {
            "experiment": result.to_dict(),
            "ablation": ablation.to_dict(),
        }
        print(f"    stale={staleness} steps: recovery={result.mean_recovery:.4f} "
              f"improvement={ablation.recovery_improvement_vs_no_sketch:+.4f}")

    # Quantization sweep
    print("\n  Sketch quantization sweep (size=16B):")
    for bits in [4, 8, 16, 32]:
        cfg = QuerySketchConfig(sketch_size_bytes=16, quantization_bits=bits)
        policy = QuerySketchAblationPolicy(sketch_config=cfg, cxl_config=cxl_config)
        result = run_single_policy(policy, workload, num_chunks, budget_chunks)
        ablation = policy.get_ablation_result()
        name = f"quant={bits}bit"
        results[name] = {
            "experiment": result.to_dict(),
            "ablation": ablation.to_dict(),
        }
        print(f"    {bits}-bit: recovery={result.mean_recovery:.4f} "
              f"improvement={ablation.recovery_improvement_vs_no_sketch:+.4f}")

    return results


# ── Main ──────────────────────────────────────────────────────────────

def run_all_experiments(
    num_chunks: int = 64,
    num_steps: int = 100,
    budget_chunks: int = 8,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all 5 priority experiments."""
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  PROSE Reviewer-Response Experiment Suite                           ║")
    print("║  5 Priority Experiments for HPCA Rebuttal                          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\n  Config: {num_chunks} chunks, {num_steps} steps, budget={budget_chunks}")
    print()

    all_results = {}

    t0 = time.time()

    all_results["exp1_sw_sbfi"] = {
        k: v.to_dict() for k, v in
        run_experiment_1_sw_sbfi(num_chunks, num_steps, budget_chunks).items()
    }

    all_results["exp2_oracle_bounds"] = {
        k: v.to_dict() for k, v in
        run_experiment_2_oracle_bounds(num_chunks, num_steps, budget_chunks).items()
    }

    all_results["exp3_scorer_noise"] = {
        k: v.to_dict() for k, v in
        run_experiment_3_scorer_noise(num_chunks, num_steps, budget_chunks).items()
    }

    all_results["exp4_size_sensitivity"] = {
        k: v.to_dict() for k, v in
        run_experiment_4_size_sensitivity(num_chunks, num_steps, budget_chunks).items()
    }

    all_results["exp5_query_sketch"] = run_experiment_5_query_sketch(
        num_chunks, num_steps, budget_chunks
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  All experiments completed in {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Save results
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "reviewer_response_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to {out_path / 'reviewer_response_results.json'}")

    return all_results


if __name__ == "__main__":
    run_all_experiments(
        num_chunks=64,
        num_steps=100,
        budget_chunks=8,
        output_dir="d:/LLM/prosex/results",
    )
