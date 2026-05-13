"""
Unified Baseline Experiment Runner.

Runs ALL baselines under identical CXL hardware conditions with the same
attention traces.  Produces structured JSON results + console comparison tables.

Architecture:
  1. Generate realistic attention traces (synthetic, calibrated from real models)
  2. Run each baseline policy against the same traces
  3. Aggregate metrics: recovery, CXL queue ρ, invalid traffic, latency
  4. Output JSON for figure generation

Supports the three experiment phases:
  Phase 1 (Tier 1 + Tier 3 ablations): Core claim validation
  Phase 2 (Tier 2): Related work positioning
  Phase 3 (Tier 4): Sensitivity sweeps
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, "d:/LLM")

from src.memory.cxl_queue_simulator import CXLQueueConfig, make_cxl_asic_config
from src.baselines import (
    StreamPrefetcherPolicy, FreqRecPrefetcherPolicy, PROSEFTSPolicy,
    OraclePolicy, OracleCandidateOnlyPolicy, VLLMCXLPolicy,
    FreqRecMetaPolicy, H2OCXLPolicy, SnapKVCXLPolicy, InfLLMCXLPolicy,
    CUDAUnifiedMemoryPolicy,
    NoPHTPolicy, SingleCuePolicy, NoPBufferPolicy, NoVersionGatePolicy, FIFOVictimPolicy,
)
from src.baselines.prose_sbfi import PROSEPolicy
from src.runners.e2e_eval_runner import BaselinePolicy


# ── Trace generation ──────────────────────────────────────────────────

def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.maximum(arr, 0.0)
    s = arr.sum()
    return arr / s if s > 0 else arr


def generate_passkey_trace(num_chunks: int = 64, num_steps: int = 200,
                           rng: np.random.RandomState = None) -> List[np.ndarray]:
    """Passkey with drifting needle attention pattern."""
    if rng is None:
        rng = np.random.RandomState(42)
    base = np.full(num_chunks, 0.002)
    seq = []
    needle = rng.randint(4, num_chunks - 4)
    for _ in range(num_steps + 1):
        attn = base.copy()
        attn[needle] = 0.20
        for i in range(max(0, needle - 1), min(num_chunks, needle + 2)):
            if i != needle:
                attn[i] = 0.04
        attn += rng.exponential(0.002, num_chunks)
        seq.append(_normalize(attn))
        if rng.random() < 0.10:
            needle = rng.randint(0, num_chunks)
        elif rng.random() < 0.35:
            needle = max(0, min(num_chunks - 1, needle + rng.choice([-1, 1])))
    return seq


def generate_needle_trace(num_chunks: int = 64, num_steps: int = 200,
                          rng: np.random.RandomState = None) -> List[np.ndarray]:
    """Needle-in-haystack with frequent jumps."""
    if rng is None:
        rng = np.random.RandomState(123)
    base = np.full(num_chunks, 0.002)
    seq = []
    peak = rng.randint(0, num_chunks)
    for _ in range(num_steps + 1):
        attn = base.copy()
        if rng.random() < 0.18:
            peak = rng.randint(0, num_chunks)
        elif rng.random() < 0.25:
            peak = max(0, min(num_chunks - 1, peak + rng.choice([-1, 0, 1])))
        attn[peak] = 0.15
        for i in range(max(0, peak - 1), min(num_chunks, peak + 2)):
            if i != peak:
                attn[i] = 0.03
        attn += rng.exponential(0.002, num_chunks)
        seq.append(_normalize(attn))
    return seq


def generate_sequential_trace(num_chunks: int = 64, num_steps: int = 200,
                              rng: np.random.RandomState = None) -> List[np.ndarray]:
    """Sequential scanning with occasional jumps."""
    if rng is None:
        rng = np.random.RandomState(44)
    base = np.full(num_chunks, 0.002)
    seq = []
    cur = 0
    for _ in range(num_steps + 1):
        attn = base.copy()
        for i in range(max(0, cur - 1), min(num_chunks, cur + 2)):
            attn[i] = 0.08 + 0.03 * rng.random()
        attn += rng.exponential(0.002, num_chunks)
        seq.append(_normalize(attn))
        if rng.random() < 0.85:
            cur = min(num_chunks - 1, cur + 1)
        else:
            cur = rng.randint(0, num_chunks)
    return seq


def generate_ruler_trace(num_chunks: int = 64, num_steps: int = 200,
                         rng: np.random.RandomState = None) -> List[np.ndarray]:
    """RULER-style multi-key tracking with frequent jumps."""
    if rng is None:
        rng = np.random.RandomState(43)
    base = np.full(num_chunks, 0.002)
    seq = []
    p1 = num_chunks // 6
    p2 = num_chunks * 2 // 3
    for _ in range(num_steps + 1):
        attn = base.copy()
        if rng.random() < 0.20:
            p1 = rng.randint(0, num_chunks)
        elif rng.random() < 0.30:
            p1 = max(0, min(num_chunks - 1, p1 + rng.choice([-1, 0, 1])))
        if rng.random() < 0.20:
            p2 = rng.randint(0, num_chunks)
        elif rng.random() < 0.30:
            p2 = max(0, min(num_chunks - 1, p2 + rng.choice([-1, 0, 1])))
        attn[p1] = 0.12
        attn[p2] = 0.10
        attn += rng.exponential(0.003, num_chunks)
        seq.append(_normalize(attn))
    return seq


ALL_TRACE_GENERATORS = {
    "passkey": generate_passkey_trace,
    "needle": generate_needle_trace,
    "sequential": generate_sequential_trace,
    "ruler": generate_ruler_trace,
}


# ── Result types ──────────────────────────────────────────────────────

@dataclass
class BaselineResult:
    """Aggregated results for one baseline on one trace."""
    method_name: str
    workload: str
    context_length: int
    budget_ratio: float
    num_chunks: int
    num_steps: int

    # Recovery
    mean_recovery: float = 0.0
    min_recovery: float = 0.0
    max_recovery: float = 0.0
    recovery_std: float = 0.0

    # CXL metrics
    mean_invalid_traffic_ratio: float = 0.0
    mean_cxl_queue_rho: float = 0.0
    total_cxl_bytes: int = 0
    total_invalid_bytes: int = 0
    mean_saturation_multiplier: float = 1.0

    # Latency (modeled)
    mean_latency_us: float = 0.0
    p99_latency_us: float = 0.0

    # Per-step data
    step_recoveries: List[float] = field(default_factory=list)
    step_invalid_ratios: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method_name,
            "workload": self.workload,
            "context_length": self.context_length,
            "budget_ratio": self.budget_ratio,
            "num_chunks": self.num_chunks,
            "num_steps": self.num_steps,
            "mean_recovery": round(self.mean_recovery, 4),
            "min_recovery": round(self.min_recovery, 4),
            "max_recovery": round(self.max_recovery, 4),
            "recovery_std": round(self.recovery_std, 4),
            "mean_invalid_traffic_ratio": round(self.mean_invalid_traffic_ratio, 4),
            "mean_cxl_queue_rho": round(self.mean_cxl_queue_rho, 4),
            "total_cxl_bytes": self.total_cxl_bytes,
            "total_invalid_bytes": self.total_invalid_bytes,
            "mean_saturation_multiplier": round(self.mean_saturation_multiplier, 2),
            "mean_latency_us": round(self.mean_latency_us, 1),
            "p99_latency_us": round(self.p99_latency_us, 1),
        }


# ── Baseline Experiment Runner ───────────────────────────────────────

class BaselineExperimentRunner:
    """Unified runner for all baseline experiments.

    Usage:
        runner = BaselineExperimentRunner(cxl_config, hbm_chunks=16)
        results = runner.run_phase1()  # Runs Tier 1 + Tier 3 ablations
        runner.print_comparison_table(results)
        runner.save_results(results, "outputs/baselines/phase1.json")
    """

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
        hbm_capacity_chunks: int = 16,
        budget_ratio: float = 0.10,
        seed: int = 42,
    ):
        self.cxl_config = cxl_config or make_cxl_asic_config()
        self.hbm_capacity = hbm_capacity_chunks
        self.budget_ratio = budget_ratio
        self.rng = np.random.RandomState(seed)
        self._results: Dict[str, Dict[str, BaselineResult]] = {}

    # ── Policy registry ────────────────────────────────────────────

    def get_tier1_policies(self) -> Dict[str, BaselinePolicy]:
        """Tier 1: Must-have baselines for core claims."""
        return {
            "StreamPrefetcher": StreamPrefetcherPolicy(
                self.cxl_config, hbm_capacity_chunks=self.hbm_capacity),
            "FreqRec-PF": FreqRecPrefetcherPolicy(
                self.cxl_config, hbm_capacity_chunks=self.hbm_capacity),
            "PROSE-FTS": PROSEFTSPolicy(self.cxl_config),
            "PROSE": PROSEPolicy(self.cxl_config),  # Full SBFI — paper's core contribution
            "Oracle-SBFI": OraclePolicy(self.cxl_config, use_sbfi=True),
            "Oracle-FTS": OraclePolicy(self.cxl_config, use_sbfi=False),
            "vLLM-CXL": VLLMCXLPolicy(
                self.cxl_config, hbm_capacity_chunks=self.hbm_capacity),
        }

    def get_tier2_policies(self) -> Dict[str, BaselinePolicy]:
        """Tier 2: Should-have baselines for related work."""
        return {
            "FreqRec-PF+Meta": FreqRecMetaPolicy(
                self.cxl_config, hbm_capacity_chunks=self.hbm_capacity),
            "H2O-CXL": H2OCXLPolicy(self.cxl_config),
            "SnapKV-CXL": SnapKVCXLPolicy(self.cxl_config),
            "InfLLM-CXL": InfLLMCXLPolicy(self.cxl_config),
            "CUDA-UM": CUDAUnifiedMemoryPolicy(self.cxl_config),
            "Oracle-Candidate": OracleCandidateOnlyPolicy(self.cxl_config),
        }

    def get_tier3_policies(self) -> Dict[str, BaselinePolicy]:
        """Tier 3: PROSE component ablations."""
        policies = {
            "PROSE-NoPHT": NoPHTPolicy(self.cxl_config),
            "PROSE-NoPBuffer": NoPBufferPolicy(self.cxl_config),
            "PROSE-NoVersionGate": NoVersionGatePolicy(self.cxl_config),
            "PROSE-FIFOVictim": FIFOVictimPolicy(
                self.cxl_config, hbm_capacity=self.hbm_capacity),
        }
        # Add 5 single-cue variants
        for cue_idx, cue_name in enumerate(SingleCuePolicy.CUE_NAMES):
            policies[f"PROSE-SingleCue({cue_name})"] = SingleCuePolicy(
                self.cxl_config, active_cue=cue_idx)

        return policies

    def get_all_policies(self) -> Dict[str, BaselinePolicy]:
        """All CXL-aware baselines."""
        policies = {}
        policies.update(self.get_tier1_policies())
        policies.update(self.get_tier2_policies())
        policies.update(self.get_tier3_policies())
        return policies

    # ── Trace generation ────────────────────────────────────────────

    def generate_traces(
        self,
        num_chunks: int = 64,
        num_steps: int = 200,
        workloads: Optional[List[str]] = None,
    ) -> Dict[str, List[np.ndarray]]:
        """Generate realistic attention traces for evaluation.

        Args:
            num_chunks: Number of KV chunks (64 for 32K, 128 for 64K, 256 for 128K)
            num_steps: Decode steps to simulate
            workloads: Which trace types to generate
        """
        if workloads is None:
            workloads = list(ALL_TRACE_GENERATORS.keys())

        traces = {}
        for wl in workloads:
            gen = ALL_TRACE_GENERATORS.get(wl)
            if gen is not None:
                traces[wl] = gen(num_chunks, num_steps, self.rng)
        return traces

    # ── Run evaluation ──────────────────────────────────────────────

    def run_single(
        self,
        policy: BaselinePolicy,
        trace: List[np.ndarray],
        workload_name: str,
    ) -> BaselineResult:
        """Run a single baseline policy against one trace."""
        num_chunks = len(trace[0])
        budget_chunks = max(1, int(num_chunks * self.budget_ratio))
        anchor_count = max(1, budget_chunks // 4)
        anchor_ids = list(range(anchor_count))

        # Reset policy state
        if hasattr(policy, 'reset'):
            policy.reset()

        step_recoveries = []
        step_invalid_ratios = []
        total_invalid_bytes = 0
        total_cxl_bytes = 0
        queue_rhos = []
        sat_mults = []
        latencies = []

        for step in range(1, len(trace)):
            # Current attention distribution
            attn = trace[step]
            chunk_attn = {i: float(attn[i]) for i in range(num_chunks)}

            # Set oracle future if applicable
            # Oracle knows current-step attention (not next-step),
            # which is the distribution its selections are measured against.
            if isinstance(policy, (OraclePolicy, OracleCandidateOnlyPolicy)):
                policy.set_future_attention(trace[step])

            # Run policy selection
            selected = policy.select_active_chunks(
                num_chunks, budget_chunks, chunk_attn, anchor_ids, step
            )

            # Compute recovery: |selected ∩ gold| / |gold|
            gold = list(np.argsort(trace[step])[::-1][:budget_chunks])
            gold = [int(g) for g in gold if int(g) not in anchor_ids]
            intersection = len(set(selected) & set(gold))
            recovery = intersection / max(len(gold), 1)
            step_recoveries.append(recovery)

            # Collect CXL stats if available
            session = getattr(policy, 'cxl_session', None)
            if session is not None and session.step_results:
                last_result = session.step_results[-1]
                stats = last_result.cxl_stats
                step_invalid_ratios.append(stats.invalid_traffic_ratio)
                total_invalid_bytes += stats.invalid_payload_bytes
                total_cxl_bytes += stats.total_bytes_fetched
                queue_rhos.append(stats.queue_utilization_rho)
                sat_mults.append(stats.saturation_multiplier)
                latencies.append(stats.total_time_ns / 1000.0)

        if not step_recoveries:
            return BaselineResult(
                method_name=policy.name, workload=workload_name,
                context_length=num_chunks * 512, budget_ratio=self.budget_ratio,
                num_chunks=num_chunks, num_steps=len(trace),
            )

        return BaselineResult(
            method_name=policy.name,
            workload=workload_name,
            context_length=num_chunks * 512,
            budget_ratio=self.budget_ratio,
            num_chunks=num_chunks,
            num_steps=len(trace),
            mean_recovery=float(np.mean(step_recoveries)),
            min_recovery=float(np.min(step_recoveries)),
            max_recovery=float(np.max(step_recoveries)),
            recovery_std=float(np.std(step_recoveries)),
            mean_invalid_traffic_ratio=float(np.mean(step_invalid_ratios)) if step_invalid_ratios else 0.0,
            mean_cxl_queue_rho=float(np.mean(queue_rhos)) if queue_rhos else 0.0,
            total_cxl_bytes=total_cxl_bytes,
            total_invalid_bytes=total_invalid_bytes,
            mean_saturation_multiplier=float(np.mean(sat_mults)) if sat_mults else 1.0,
            mean_latency_us=float(np.mean(latencies)) if latencies else 0.0,
            p99_latency_us=float(np.percentile(latencies, 99)) if latencies else 0.0,
            step_recoveries=step_recoveries,
            step_invalid_ratios=step_invalid_ratios,
        )

    def run_phase(
        self,
        policies: Dict[str, BaselinePolicy],
        traces: Dict[str, List[np.ndarray]],
        phase_name: str = "phase1",
    ) -> Dict[str, Dict[str, BaselineResult]]:
        """Run a set of policies against all traces.

        Returns: {workload_name: {method_name: BaselineResult}}
        """
        results = {}
        total = len(traces) * len(policies)
        count = 0

        for wl_name, trace in traces.items():
            results[wl_name] = {}
            for pol_name, policy in policies.items():
                count += 1
                print(f"  [{count}/{total}] {pol_name:25s} on {wl_name:12s}...", end=" ")
                sys.stdout.flush()

                result = self.run_single(policy, trace, wl_name)
                results[wl_name][pol_name] = result

                print(f"rec={result.mean_recovery:.3f} "
                      f"inv={result.mean_invalid_traffic_ratio:.3f} "
                      f"ρ={result.mean_cxl_queue_rho:.2f}")
                sys.stdout.flush()

        self._results[phase_name] = results
        return results

    def run_phase1(self, num_chunks: int = 64, num_steps: int = 200) -> Dict:
        """Run Phase 1: Tier 1 must-have baselines + Tier 3 ablations."""
        print("=" * 80)
        print("PHASE 1: Tier 1 Core Baselines + Tier 3 PROSE Ablations")
        print(f"  CXL: {self.cxl_config.bandwidth_gbps} GB/s, "
              f"queue={self.cxl_config.queue_depth}, "
              f"HBM={self.hbm_capacity} chunks")
        print(f"  Context: {num_chunks * 512} tokens ({num_chunks} chunks), "
              f"Budget: {self.budget_ratio:.0%}, Steps: {num_steps}")
        print("=" * 80)

        traces = self.generate_traces(num_chunks, num_steps,
                                      ["passkey", "needle", "sequential", "ruler"])
        policies = {}
        policies.update(self.get_tier1_policies())
        # Also include key Tier 3 ablations in Phase 1
        tier3 = self.get_tier3_policies()
        # Keep only the 4 main ablations + 1 single-cue example
        for key in ["PROSE-NoPHT", "PROSE-NoPBuffer", "PROSE-NoVersionGate",
                     "PROSE-FIFOVictim", "PROSE-SingleCue(temporal)"]:
            if key in tier3:
                policies[key] = tier3[key]

        return self.run_phase(policies, traces, "phase1")

    def run_phase2(self, num_chunks: int = 64, num_steps: int = 200) -> Dict:
        """Run Phase 2: Tier 2 related-work baselines."""
        print("=" * 80)
        print("PHASE 2: Tier 2 Related-Work Positioning Baselines")
        print("=" * 80)

        traces = self.generate_traces(num_chunks, num_steps,
                                      ["passkey", "needle", "sequential", "ruler"])
        return self.run_phase(self.get_tier2_policies(), traces, "phase2")

    def run_sensitivity_sweep(
        self,
        policies: Optional[Dict[str, BaselinePolicy]] = None,
        sweep_param: str = "bandwidth",
        sweep_values: Optional[List[float]] = None,
        num_chunks: int = 64,
        num_steps: int = 100,
    ) -> Dict[str, Any]:
        """Run sensitivity sweep across a parameter.

        Args:
            policies: Policies to sweep (default: key Tier 1)
            sweep_param: Parameter to sweep
            sweep_values: Values to test
        """
        if policies is None:
            policies = {
                "StreamPrefetcher": StreamPrefetcherPolicy(self.cxl_config, hbm_capacity_chunks=self.hbm_capacity),
                "FreqRec-PF": FreqRecPrefetcherPolicy(self.cxl_config, hbm_capacity_chunks=self.hbm_capacity),
                "PROSE-FTS": PROSEFTSPolicy(self.cxl_config),
                "Oracle-SBFI": OraclePolicy(self.cxl_config, use_sbfi=True),
            }

        if sweep_values is None:
            if sweep_param == "bandwidth":
                sweep_values = [4, 8, 16, 32, 48, 64]
            elif sweep_param == "offload_ratio":
                sweep_values = [0.70, 0.80, 0.90, 0.95, 0.98]
            elif sweep_param == "queue_depth":
                sweep_values = [16, 32, 48, 64, 128]
            elif sweep_param == "chunk_granularity":
                sweep_values = [128, 256, 512, 1024, 2048]
            elif sweep_param == "batch_size":
                sweep_values = [1, 4, 16, 64, 128]
            else:
                sweep_values = [4, 8, 16, 32, 64]

        all_results = {}
        traces = self.generate_traces(num_chunks, num_steps, ["needle", "ruler"])

        for val in sweep_values:
            print(f"\n  --- {sweep_param} = {val} ---")

            # Apply sweep value to config/policies
            for pol_name, policy in policies.items():
                if hasattr(policy, 'reset'):
                    policy.reset()

                # Adjust config
                if sweep_param == "bandwidth":
                    if hasattr(policy, 'cxl_config'):
                        policy.cxl_config.bandwidth_gbps = val
                        policy.cxl_config.raw_bandwidth_gbps = val / 0.98
                elif sweep_param == "offload_ratio":
                    # Adjust effective budget_ratio
                    pass
                elif sweep_param == "queue_depth":
                    if hasattr(policy, 'cxl_config'):
                        policy.cxl_config.queue_depth = int(val)

            phase_results = self.run_phase(policies, traces, f"sweep_{sweep_param}_{val}")
            all_results[str(val)] = phase_results

        return all_results

    # ── Output ──────────────────────────────────────────────────────

    def _result_val(self, r, name, default=0.0):
        """Get value from BaselineResult or dict."""
        if hasattr(r, name):
            return getattr(r, name, default)
        if isinstance(r, dict):
            return r.get(name, default)
        return default

    def print_comparison_table(self, results: Dict[str, Dict[str, BaselineResult]]):
        """Print a formatted comparison table.

        Handles both 2-level ({workload: {method: result}}) and
        3-level ({phase: {workload: {method: result}}}) nesting.
        """
        print("\n" + "=" * 100)
        print("BASELINE COMPARISON TABLE")
        print("=" * 100)

        # Detect nesting
        first_val = next(iter(results.values()))
        # Flatten to {method: [values]}
        all_methods: Dict[str, Dict[str, list]] = {}

        for phase_or_wl, phase_body in results.items():
            if not isinstance(phase_body, dict):
                continue
            for wl_or_method, wl_body in phase_body.items():
                if not isinstance(wl_body, dict):
                    continue
                # Determine if this is {method: result} or deeper {method: result}
                inner_vals = list(wl_body.values())
                if inner_vals and (hasattr(inner_vals[0], 'mean_recovery') or
                                   (isinstance(inner_vals[0], dict) and 'mean_recovery' in inner_vals[0])):
                    # 2-level variant: wl_body = {method: result}
                    method_results = wl_body
                else:
                    # 3-level: wl_body = {method: result}
                    method_results = wl_body

                for method, r in method_results.items():
                    if method not in all_methods:
                        all_methods[method] = {"rec": [], "inv": [], "rho": [], "sat": [], "lat": []}
                    all_methods[method]["rec"].append(self._result_val(r, 'mean_recovery'))
                    all_methods[method]["inv"].append(self._result_val(r, 'mean_invalid_traffic_ratio'))
                    all_methods[method]["rho"].append(self._result_val(r, 'mean_cxl_queue_rho'))
                    all_methods[method]["sat"].append(self._result_val(r, 'mean_saturation_multiplier'))
                    all_methods[method]["lat"].append(self._result_val(r, 'mean_latency_us'))
                break  # Only process first workload to avoid double-counting

        # Header
        header = f"{'Method':<28} | {'Recovery':>8} | {'Invalid%':>8} | {'CXL ρ':>6} | {'Sat x':>6} | {'Lat(us)':>8}"
        print(header)
        print("-" * len(header))

        for method in sorted(all_methods.keys()):
            stats = all_methods[method]
            recs, invs, rhos, sats, lats = stats["rec"], stats["inv"], stats["rho"], stats["sat"], stats["lat"]
            if recs:
                print(f"{method:<28} | {np.mean(recs):8.3f} | "
                      f"{np.mean(invs)*100:7.1f}% | {np.mean(rhos):6.2f} | "
                      f"{np.mean(sats):6.1f}x | {np.mean(lats):8.1f}")

        print("=" * 100)

    def print_detailed_breakdown(self, results: Dict[str, Dict[str, BaselineResult]]):
        """Print per-workload breakdown. Handles both nesting levels."""
        # Detect and flatten
        for phase_or_wl, phase_body in results.items():
            if not isinstance(phase_body, dict):
                continue
            # If phase_body looks like {method: result}, it's 2-level
            inner_vals = list(phase_body.values())
            is_2level = inner_vals and any(
                hasattr(v, 'mean_recovery') or (isinstance(v, dict) and 'mean_recovery' in v)
                for v in inner_vals[:1]
            )
            if is_2level:
                workloads = [(phase_or_wl, phase_body)]
            else:
                workloads = phase_body.items()

            for wl_name, wl_results in workloads:
                if not isinstance(wl_results, dict):
                    continue
                print(f"\n--- {wl_name.upper()} ---")
                print(f"{'Method':<28} {'Recovery':>8} {'Invalid%':>8} {'CXL ρ':>6} {'Lat(us)':>8}")
                print("-" * 68)
                for method, r in sorted(wl_results.items()):
                    rec = self._result_val(r, 'mean_recovery')
                    inv = self._result_val(r, 'mean_invalid_traffic_ratio') * 100
                    rho = self._result_val(r, 'mean_cxl_queue_rho')
                    lat = self._result_val(r, 'mean_latency_us')
                    print(f"{method:<28} {rec:8.3f} {inv:7.1f}% {rho:6.2f} {lat:8.1f}")
                break  # Only first per phase to avoid explosion

    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        # Convert to serializable dict
        serializable = {}
        for key, val in results.items():
            if isinstance(val, dict):
                serializable[key] = {}
                for k2, v2 in val.items():
                    if isinstance(v2, BaselineResult):
                        serializable[key][k2] = v2.to_dict()
                    elif isinstance(v2, dict):
                        serializable[key][k2] = {
                            k3: v3.to_dict() if isinstance(v3, BaselineResult) else v3
                            for k3, v3 in v2.items()
                        }
                    else:
                        serializable[key][k2] = str(v2)
            elif isinstance(val, BaselineResult):
                serializable[key] = val.to_dict()
            else:
                serializable[key] = str(val)

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    def generate_figure_data(self, results: Dict) -> Dict[str, Any]:
        """Extract figure-ready data from results.

        Handles both 2-level ({workload: {method: result}}) and
        3-level ({phase: {workload: {method: result}}}) nesting.

        Returns data for three key figures:
          1. Invalid payload traffic vs context length
          2. Queue utilization ρ vs offload ratio
          3. Metric hierarchy table
        """
        figure_data = {
            "fig1_invalid_traffic": {},
            "fig2_queue_utilization": {},
            "fig3_metric_hierarchy": {},
        }

        def _val(r, name, default=0.0):
            if hasattr(r, name):
                return getattr(r, name, default)
            if isinstance(r, dict):
                return r.get(name, default)
            return default

        # Detect nesting: peek at first non-sweep value
        phases = {k: v for k, v in results.items() if not k.startswith("sweep")}
        if not phases:
            return figure_data

        first_val = next(iter(phases.values()))
        # Determine if first_val is workload->method or method->result
        is_3level = False
        if isinstance(first_val, dict):
            inner_vals = list(first_val.values())
            if inner_vals and isinstance(inner_vals[0], dict):
                # Check if inner keys are result fields (2-level) or method names (3-level)
                ik = list(inner_vals[0].keys())
                if ik and all(k not in ["method", "workload", "mean_recovery"] for k in ik):
                    is_3level = True

        # Normalize to 3-level iteration
        def iter_workload_methods():
            for phase_name, phase_body in phases.items():
                if not isinstance(phase_body, dict):
                    continue
                if is_3level:
                    for wl_name, wl_body in phase_body.items():
                        if not isinstance(wl_body, dict):
                            continue
                        yield phase_name, wl_name, wl_body
                else:
                    # 2-level: phase_body IS {method: result} — iterate once
                    yield phase_name, "all", phase_body

        # Aggregate
        method_stats = {}
        for phase_name, wl_name, method_results in iter_workload_methods():
            for method, r in method_results.items():
                rec = _val(r, 'mean_recovery', None)
                if rec is None:
                    continue
                if method not in method_stats:
                    method_stats[method] = {
                        "recovery": [], "invalid": [], "rho": [], "latency": []
                    }
                method_stats[method]["recovery"].append(_val(r, 'mean_recovery'))
                method_stats[method]["invalid"].append(_val(r, 'mean_invalid_traffic_ratio'))
                method_stats[method]["rho"].append(_val(r, 'mean_cxl_queue_rho'))
                method_stats[method]["latency"].append(_val(r, 'mean_latency_us'))

        for method, stats in method_stats.items():
            figure_data["fig1_invalid_traffic"][method] = {
                "mean_invalid_ratio": float(np.mean(stats["invalid"])),
                "worst_invalid_ratio": float(np.max(stats["invalid"])),
            }
            figure_data["fig2_queue_utilization"][method] = {
                "mean_rho": float(np.mean(stats["rho"])),
                "peak_rho": float(np.max(stats["rho"])),
                "mean_latency_us": float(np.mean(stats["latency"])),
            }
            figure_data["fig3_metric_hierarchy"][method] = {
                "l1_recovery": float(np.mean(stats["recovery"])),
                "l2_latency_us": float(np.mean(stats["latency"])),
            }

        return figure_data


# ── Quick test entry point ───────────────────────────────────────────

def main():
    """Quick test: run Phase 1 baselines."""
    runner = BaselineExperimentRunner(
        cxl_config=make_cxl_asic_config(),
        hbm_capacity_chunks=16,
        budget_ratio=0.10,
    )

    # Quick test with small config
    results = runner.run_phase1(num_chunks=32, num_steps=100)

    runner.print_comparison_table(results)
    runner.print_detailed_breakdown(results)

    fig_data = runner.generate_figure_data({"phase1": results})
    runner.save_results(results, "outputs/baselines/phase1_quick_test.json")
    runner.save_results(fig_data, "outputs/baselines/figure_data.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
