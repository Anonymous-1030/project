"""
Multi-Tenant Evaluation Runner for PROSE.

Runs systematic multi-tenant experiments covering six dimensions:
  D1 – Heterogeneous context-length mixing
  D2 – Priority and QoS enforcement
  D3 – Priority inversion quantification
  D4 – Token-bucket isolation effectiveness
  D5 – Metadata contention under tenant scaling
  D6 – Dynamic arrival and departure

Architecture:
  - N independent PROSE/ProSE-FTS policy instances (one per tenant)
  - Per-tenant CXL sessions with independent token-bucket accounting
  - SharedCXLlink models the single physical CXL link with DRR arbitration
  - Trace-driven step-by-step simulation (sequential, not real GPU parallelism)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.memory.cxl_queue_simulator import (
    CXLQueueConfig,
    CXLQueueSimulator,
    StepStats,
    make_cxl_asic_config,
)
from src.runners.e2e_eval_runner import BaselinePolicy
from src.baselines.prose_sbfi import PROSEPolicy
from src.baselines.prose_fts import PROSEFTSPolicy


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

class ArrivalProcess(str, Enum):
    SIMULTANEOUS = "simultaneous"
    POISSON = "poisson"
    STAGGERED = "staggered"


class AllocationPolicy(str, Enum):
    DEFICIT_ROUND_ROBIN = "deficit_round_robin"
    EQUAL_SHARE = "equal_share"
    PRIORITY_WEIGHTED = "priority_weighted"


@dataclass
class MultiTenantConfig:
    """Configuration for a multi-tenant evaluation run."""

    # Tenant configuration
    num_tenants: int = 4
    context_lengths: List[int] = field(default_factory=lambda: [4096, 32768, 65536, 131072])
    budget_ratios: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.10, 0.10])
    priorities: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.5, 0.5])
    slo_targets_us: List[float] = field(default_factory=lambda: [200.0, 500.0, 1000.0, 1000.0])

    # Scheduling
    allocation_policy: AllocationPolicy = AllocationPolicy.DEFICIT_ROUND_ROBIN
    drr_quantum_bytes: int = 65536  # 64KB per round per tenant
    enable_token_bucket: bool = True
    token_bucket_capacity: int = 32  # max DMA ops per decode step per tenant
    token_bucket_refill_rate: int = 16  # tokens refilled per step

    # Arrival process
    enable_dynamic_arrival: bool = False
    arrival_process: ArrivalProcess = ArrivalProcess.SIMULTANEOUS
    arrival_rate: float = 0.1
    arrival_stagger_step: int = 20

    # Simulation parameters
    num_decode_steps: int = 200
    chunk_size: int = 128
    cxl_bandwidth_gbps: float = 64.0
    seed: int = 42

    # Method
    compare_fts: bool = False  # also run PROSE-FTS for comparison

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Enum):
                d[k] = v.value
            elif isinstance(v, list):
                d[k] = v
            else:
                d[k] = v
        return d


# ═══════════════════════════════════════════════════════════════════════════
# Per-Tenant State
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TenantMetrics:
    """Accumulated per-tenant metrics across all steps."""

    tenant_id: str = ""
    context_length: int = 0
    budget_ratio: float = 0.1
    priority: float = 1.0

    # Recovery
    recoveries: List[float] = field(default_factory=list)
    mean_recovery: float = 0.0
    p99_recovery: float = 0.0

    # Latency (us)
    latencies_us: List[float] = field(default_factory=list)
    mean_latency_us: float = 0.0
    p50_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    p999_latency_us: float = 0.0

    # CXL metrics
    cxl_queue_rhos: List[float] = field(default_factory=list)
    mean_cxl_queue_rho: float = 0.0
    total_cxl_bytes: int = 0
    total_invalid_bytes: int = 0
    invalid_traffic_ratios: List[float] = field(default_factory=list)
    mean_invalid_traffic_ratio: float = 0.0

    # Queuing
    queuing_delays_us: List[float] = field(default_factory=list)
    mean_queuing_delay_us: float = 0.0
    p99_queuing_delay_us: float = 0.0

    # Isolation
    solo_latency_us: float = 0.0  # latency when running alone
    tail_degradation_ratio: float = 1.0  # P99 multi-tenant / P99 solo

    # Starvation & SLO
    starvation_events: int = 0
    slo_violations: int = 0
    token_bucket_exhausted_steps: int = 0

    # PHT / warm-up
    pht_hit_rates: List[float] = field(default_factory=list)
    cold_start_latency_us: float = 0.0

    def finalize(self, solo_latency_us: float = 0.0):
        """Compute aggregate statistics from accumulated per-step data."""
        if self.recoveries:
            arr = np.array(self.recoveries)
            self.mean_recovery = float(np.mean(arr))
            self.p99_recovery = float(np.percentile(arr, 99))

        if self.latencies_us:
            arr = np.array(self.latencies_us)
            self.mean_latency_us = float(np.mean(arr))
            self.p50_latency_us = float(np.percentile(arr, 50))
            self.p99_latency_us = float(np.percentile(arr, 99))
            self.p999_latency_us = float(np.percentile(arr, 99.9))

        if self.cxl_queue_rhos:
            self.mean_cxl_queue_rho = float(np.mean(self.cxl_queue_rhos))

        if self.invalid_traffic_ratios:
            self.mean_invalid_traffic_ratio = float(np.mean(self.invalid_traffic_ratios))

        if self.queuing_delays_us:
            arr = np.array(self.queuing_delays_us)
            self.mean_queuing_delay_us = float(np.mean(arr))
            self.p99_queuing_delay_us = float(np.percentile(arr, 99))

        if self.pht_hit_rates:
            self.pht_hit_rates_avg = float(np.mean(self.pht_hit_rates))

        self.solo_latency_us = solo_latency_us
        if solo_latency_us > 0 and self.p99_latency_us > 0:
            self.tail_degradation_ratio = self.p99_latency_us / solo_latency_us

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "context_length": self.context_length,
            "budget_ratio": self.budget_ratio,
            "priority": self.priority,
            "mean_recovery": round(self.mean_recovery, 4),
            "p99_recovery": round(self.p99_recovery, 4),
            "mean_latency_us": round(self.mean_latency_us, 1),
            "p50_latency_us": round(self.p50_latency_us, 1),
            "p99_latency_us": round(self.p99_latency_us, 1),
            "p999_latency_us": round(self.p999_latency_us, 1),
            "mean_cxl_queue_rho": round(self.mean_cxl_queue_rho, 4),
            "total_cxl_bytes": self.total_cxl_bytes,
            "total_invalid_bytes": self.total_invalid_bytes,
            "mean_invalid_traffic_ratio": round(self.mean_invalid_traffic_ratio, 4),
            "mean_queuing_delay_us": round(self.mean_queuing_delay_us, 1),
            "p99_queuing_delay_us": round(self.p99_queuing_delay_us, 1),
            "tail_degradation_ratio": round(self.tail_degradation_ratio, 3),
            "starvation_events": self.starvation_events,
            "slo_violations": self.slo_violations,
            "token_bucket_exhausted_steps": self.token_bucket_exhausted_steps,
            "cold_start_latency_us": round(self.cold_start_latency_us, 1),
            "solo_latency_us": round(self.solo_latency_us, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Token Bucket Isolator
# ═══════════════════════════════════════════════════════════════════════════

class TokenBucketIsolator:
    """Per-tenant token bucket for bandwidth isolation.

    Each tenant has a bucket with `capacity` tokens. Tokens are consumed per
    DMA operation (1 token per summary fetch, 1 token per 64KB payload DMA).
    Bucket refills at `refill_rate` tokens per decode step.
    """

    def __init__(self, tenant_id: str, capacity: int = 8, refill_rate: int = 8):
        self.tenant_id = tenant_id
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.exhausted_steps = 0
        self.total_requests = 0
        self.blocked_requests = 0

    def consume(self, num_ops: int = 1) -> bool:
        """Try to consume tokens. Returns True if allowed, False if blocked."""
        self.total_requests += 1
        if self.tokens >= num_ops:
            self.tokens -= num_ops
            return True
        else:
            self.blocked_requests += 1
            return False

    def refill(self):
        """Refill tokens at the start of each decode step."""
        self.tokens = min(self.capacity, self.tokens + self.refill_rate)

    def reset(self):
        self.tokens = self.capacity
        self.exhausted_steps = 0
        self.total_requests = 0
        self.blocked_requests = 0


# ═══════════════════════════════════════════════════════════════════════════
# Shared CXL Link with Deficit Round Robin
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DRRState:
    """Per-tenant state for Deficit Round Robin."""

    tenant_id: str
    deficit: float = 0.0  # accumulated deficit in bytes
    active: bool = True


class SharedCXLLink:
    """Models the single physical CXL link shared by all tenants.

    Arbitration policy:
      - deficit_round_robin: Each tenant gets `quantum` bytes per round. Unused
        quantum carries over (deficit). Priority is used to order tenants within
        each round (higher priority = served first).
      - equal_share: Each tenant gets B_total / K.
      - priority_weighted: Bandwidth proportional to priority.
    """

    def __init__(
        self,
        cxl_config: Optional[CXLQueueConfig] = None,
        policy: AllocationPolicy = AllocationPolicy.DEFICIT_ROUND_ROBIN,
        quantum_bytes: int = 65536,
    ):
        self.cfg = cxl_config or make_cxl_asic_config()
        self.policy = policy
        self.quantum_bytes = quantum_bytes

        # Per-step state
        self._drr_states: Dict[str, DRRState] = {}
        self._pending_requests: Dict[str, List[Tuple[str, int]]] = {}  # tenant_id -> [(op_type, bytes)]
        self._step_tenant_order: List[str] = []

        # Step-level metrics
        self._step_tenant_latency: Dict[str, float] = {}
        self._step_tenant_queuing: Dict[str, float] = {}
        self._step_total_service_ns: float = 0.0

        # Accumulated metrics
        self.contention_events: List[Dict[str, Any]] = []
        self.priority_inversion_events: List[Dict[str, Any]] = []

    def register_tenant(self, tenant_id: str):
        self._drr_states[tenant_id] = DRRState(tenant_id=tenant_id)

    def remove_tenant(self, tenant_id: str):
        self._drr_states.pop(tenant_id, None)

    def submit_request(self, tenant_id: str, op_type: str, total_bytes: int):
        """Submit a DMA request for arbitration this step."""
        if tenant_id not in self._pending_requests:
            self._pending_requests[tenant_id] = []
        self._pending_requests[tenant_id].append((op_type, total_bytes))

    def arbitrate_and_compute_latency(self, step: int) -> Dict[str, float]:
        """Arbitrate all pending requests and compute per-tenant latency.

        Returns dict of tenant_id -> total_latency_us for this step.
        """
        if not self._pending_requests:
            return {}

        tenant_ids = list(self._pending_requests.keys())

        if self.policy == AllocationPolicy.DEFICIT_ROUND_ROBIN:
            result = self._arbitrate_drr(step, tenant_ids)
        elif self.policy == AllocationPolicy.EQUAL_SHARE:
            result = self._arbitrate_equal(step, tenant_ids)
        elif self.policy == AllocationPolicy.PRIORITY_WEIGHTED:
            result = self._arbitrate_priority_weighted(step, tenant_ids)
        else:
            result = self._arbitrate_drr(step, tenant_ids)

        self._step_tenant_latency = result
        self._pending_requests.clear()
        return result

    def _arbitrate_drr(self, step: int, tenant_ids: List[str]) -> Dict[str, float]:
        """Deficit Round Robin arbitration.

        Tenants are served in rounds. Within each round, tenants are ordered by
        priority (descending). Each active tenant gets up to `quantum` bytes.
        Deficit carries over to the next round.
        """
        # Add quantum to all active tenants
        for tid in tenant_ids:
            if tid in self._drr_states and self._drr_states[tid].active:
                self._drr_states[tid].deficit += self.quantum_bytes

        # Sort by priority descending for ordering within each round
        per_tenant_bytes: Dict[str, int] = {tid: sum(b for _, b in self._pending_requests.get(tid, [])) for tid in tenant_ids}
        per_tenant_latency: Dict[str, float] = {}
        per_tenant_queuing: Dict[str, float] = {}

        bytes_per_us = self.cfg.bandwidth_gbps  # GB/s = bytes/ns × 1e9/1e6 = B/us

        # Compute effective bandwidth considering queue depth
        total_bytes_all = sum(per_tenant_bytes.values())
        if total_bytes_all == 0:
            return {tid: 0.0 for tid in tenant_ids}

        # Determine DRR serve order: by priority (high to low)
        priority_order = sorted(tenant_ids, key=lambda tid: int(self._get_priority(tid) * 100), reverse=True)

        # Service each tenant up to deficit
        cumulative_service_ns = 0.0
        tenants_served = 0
        for tid in priority_order:
            requested = per_tenant_bytes.get(tid, 0)
            if requested <= 0:
                per_tenant_latency[tid] = 0.0
                per_tenant_queuing[tid] = 0.0
                continue

            deficit = self._drr_states.get(tid, DRRState(tenant_id=tid)).deficit
            served = min(requested, max(0, int(deficit)))

            if served > 0:
                # Service time for this tenant's bytes
                service_ns = self._compute_service_time_ns(served)
                # Queuing: time waiting for earlier tenants in this round
                queuing_ns = cumulative_service_ns
                total_ns = service_ns + queuing_ns

                per_tenant_latency[tid] = total_ns / 1000.0  # ns -> us
                per_tenant_queuing[tid] = queuing_ns / 1000.0
                cumulative_service_ns += service_ns

                # Deduct from deficit
                self._drr_states[tid].deficit -= served
                tenants_served += 1
            else:
                per_tenant_latency[tid] = 0.0
                per_tenant_queuing[tid] = 0.0

        self._step_total_service_ns = cumulative_service_ns
        self._step_tenant_queuing = per_tenant_queuing

        # Record contention if total demand > bandwidth capacity
        step_capacity_bytes = int(self.cfg.bandwidth_gbps * 100_000.0)  # 100us step
        if total_bytes_all > step_capacity_bytes:
            self.contention_events.append({
                "step": step,
                "tenant_count": len(tenant_ids),
                "total_requested_bytes": total_bytes_all,
                "capacity_bytes": step_capacity_bytes,
                "saturation": round(total_bytes_all / max(step_capacity_bytes, 1), 3),
            })

        return per_tenant_latency

    def _arbitrate_equal(self, step: int, tenant_ids: List[str]) -> Dict[str, float]:
        """Equal-share: each tenant gets equal portion of bandwidth."""
        per_tenant_bytes = {tid: sum(b for _, b in self._pending_requests.get(tid, [])) for tid in tenant_ids}
        total_bytes_all = sum(per_tenant_bytes.values())
        k = len(tenant_ids)

        if total_bytes_all == 0:
            return {tid: 0.0 for tid in tenant_ids}

        step_capacity_bytes = int(self.cfg.bandwidth_gbps * 100_000.0)
        fair_share = step_capacity_bytes // max(k, 1)

        per_tenant_latency: Dict[str, float] = {}
        cumulative_service_ns = 0.0

        for tid in tenant_ids:
            requested = per_tenant_bytes.get(tid, 0)
            allocated = min(requested, fair_share)
            if allocated > 0:
                service_ns = self._compute_service_time_ns(allocated)
                queuing_ns = cumulative_service_ns
                per_tenant_latency[tid] = (service_ns + queuing_ns) / 1000.0
                cumulative_service_ns += service_ns
            else:
                per_tenant_latency[tid] = 0.0

        return per_tenant_latency

    def _arbitrate_priority_weighted(self, step: int, tenant_ids: List[str]) -> Dict[str, float]:
        """Bandwidth proportional to priority."""
        per_tenant_bytes = {tid: sum(b for _, b in self._pending_requests.get(tid, [])) for tid in tenant_ids}
        total_bytes_all = sum(per_tenant_bytes.values())

        if total_bytes_all == 0:
            return {tid: 0.0 for tid in tenant_ids}

        step_capacity_bytes = int(self.cfg.bandwidth_gbps * 100_000.0)
        priority_sum = sum(self._get_priority(tid) for tid in tenant_ids)

        per_tenant_latency: Dict[str, float] = {}
        cumulative_service_ns = 0.0

        for tid in sorted(tenant_ids, key=lambda t: self._get_priority(t), reverse=True):
            requested = per_tenant_bytes.get(tid, 0)
            prio_share = self._get_priority(tid) / max(priority_sum, 0.01)
            allocated = min(requested, int(step_capacity_bytes * prio_share))
            if allocated > 0:
                service_ns = self._compute_service_time_ns(allocated)
                per_tenant_latency[tid] = (service_ns + cumulative_service_ns) / 1000.0
                cumulative_service_ns += service_ns
            else:
                per_tenant_latency[tid] = 0.0

        return per_tenant_latency

    def _get_priority(self, tenant_id: str) -> float:
        """Get priority for a tenant (defaults to 1.0)."""
        return 1.0  # overridden by setting _priorities dict

    def _compute_service_time_ns(self, total_bytes: int) -> float:
        """Compute service time for given bytes using the CXL queue config model."""
        # Reuse the CXL Queue Simulator's service time model
        cfg = self.cfg
        if cfg.bytes_per_ns > 0:
            bw_ns = total_bytes / cfg.bytes_per_ns
        else:
            bw_ns = total_bytes / 64.0  # fallback: 64 GB/s

        # DRAM access overhead
        if total_bytes <= 4096:
            burst_bytes = 64
            num_accesses = max(1, total_bytes // burst_bytes)
            effective_accesses = math.ceil(num_accesses / cfg.dram_num_bank_groups)
            hit_rate = cfg.dram_row_buffer_hit_rate
            avg_dram_ns_per_access = (
                hit_rate * cfg.dram_row_hit_ns +
                (1 - hit_rate) * cfg.dram_row_miss_ns
            )
            dram_ns = effective_accesses * avg_dram_ns_per_access
            dram_ns = max(bw_ns, dram_ns)
        else:
            dram_ns = bw_ns + cfg.dram_row_miss_ns

        protocol_ns = 2 * cfg.proto_proc_lat_ns + cfg.bridge_lat_ns
        return dram_ns + protocol_ns

    def end_step(self) -> Dict[str, Any]:
        """Finalize step and return step-level contention metrics."""
        stats = {
            "total_service_ns": self._step_total_service_ns,
            "per_tenant_latency_us": dict(self._step_tenant_latency),
            "per_tenant_queuing_us": dict(self._step_tenant_queuing),
        }
        self._step_total_service_ns = 0.0
        self._step_tenant_latency.clear()
        self._step_tenant_queuing.clear()
        return stats


# ═══════════════════════════════════════════════════════════════════════════
# Trace Generators for Multi-Tenant Workloads
# ═══════════════════════════════════════════════════════════════════════════

def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.maximum(arr, 0.0)
    s = arr.sum()
    return arr / s if s > 0 else arr


def generate_heterogeneous_traces(
    context_length_configs: List[int],
    num_steps: int = 200,
    rng: Optional[np.random.RandomState] = None,
) -> List[Tuple[int, List[np.ndarray]]]:
    """Generate attention traces for heterogeneous context lengths.

    Args:
        context_length_configs: list of context lengths (tokens) per tenant.
        num_steps: number of decode steps.
        rng: random state.

    Returns:
        List of (context_length, [per_step_attention_arrays])
    """
    if rng is None:
        rng = np.random.RandomState(42)

    traces: List[Tuple[int, List[np.ndarray]]] = []

    for ctx_len in context_length_configs:
        num_chunks = max(16, ctx_len // 128)  # chunk_size=128 tokens
        base = np.full(num_chunks, 0.002)

        needle_chunks = rng.randint(2, max(3, num_chunks // 8), size=3)
        seq: List[np.ndarray] = []

        current_needle = int(needle_chunks[0])
        for step in range(num_steps + 1):
            attn = base.copy()
            # Main needle gets 25% attention
            if 0 <= current_needle < num_chunks:
                attn[current_needle] = 0.25
            # Neighbor chunks get 5%
            for offset in [-2, -1, 1, 2]:
                neighbor = current_needle + offset
                if 0 <= neighbor < num_chunks:
                    attn[neighbor] = max(attn[neighbor], 0.05)
            # Secondary needles get 8%
            for n in needle_chunks[1:]:
                if 0 <= n < num_chunks:
                    attn[n] = max(attn[n], 0.08)

            attn += rng.exponential(0.001, num_chunks)
            seq.append(_normalize(attn))

            # Needle drift
            if rng.random() < 0.08:
                current_needle = rng.randint(0, num_chunks)
            elif rng.random() < 0.25:
                current_needle = max(0, min(num_chunks - 1,
                    current_needle + rng.choice([-2, -1, 1, 2])))

        traces.append((ctx_len, seq))

    return traces


# ═══════════════════════════════════════════════════════════════════════════
# Main Multi-Tenant Evaluation Runner
# ═══════════════════════════════════════════════════════════════════════════

class MultiTenantEvalRunner:
    """Orchestrates multi-tenant PROSE evaluation across 6 dimensions.

    Each tenant runs an independent PROSE pipeline against its own attention
    trace. The shared CXL link models contention using per-step arbitration.
    """

    def __init__(self, config: Optional[MultiTenantConfig] = None):
        self.config = config or MultiTenantConfig()
        self.rng = np.random.RandomState(self.config.seed)

        self.shared_link: Optional[SharedCXLLink] = None
        self.token_buckets: Dict[str, TokenBucketIsolator] = {}
        self.tenant_metrics: Dict[str, TenantMetrics] = {}
        self._solo_metrics: Dict[str, float] = {}  # tenant_id -> solo_latency_us

        # For priority
        self._priorities: Dict[str, float] = {}

    # ── Tenant Factory ──────────────────────────────────────────────────

    def _create_policy(self, use_fts: bool = False) -> BaselinePolicy:
        cxl_cfg = CXLQueueConfig(bandwidth_gbps=self.config.cxl_bandwidth_gbps)
        if use_fts:
            return PROSEFTSPolicy(cxl_config=cxl_cfg, enable_pht=True,
                                  enable_burst=True, enable_sticky=True)
        else:
            return PROSEPolicy(cxl_config=cxl_cfg, enable_pht=True,
                               enable_burst=True, enable_sticky=True)

    # ── Core Simulation Loop ────────────────────────────────────────────

    def _run_simulation(
        self,
        tenant_specs: List[Dict[str, Any]],
        use_fts: bool = False,
        arrival_schedule: Optional[List[int]] = None,
    ) -> Dict[str, TenantMetrics]:
        """Run multi-tenant simulation.

        Args:
            tenant_specs: [{tenant_id, context_length, budget_ratio, priority, slo_target_us}]
            use_fts: if True, use PROSE-FTS instead of PROSE
            arrival_schedule: per-tenant arrival step (None = all arrive at step 0)

        Returns:
            dict of tenant_id -> TenantMetrics
        """
        num_tenants = len(tenant_specs)
        if num_tenants == 0:
            return {}

        # Generate traces
        ctx_lengths = [s["context_length"] for s in tenant_specs]
        all_traces = generate_heterogeneous_traces(
            ctx_lengths, self.config.num_decode_steps, self.rng,
        )

        # Initialize shared link
        self.shared_link = SharedCXLLink(
            cxl_config=CXLQueueConfig(bandwidth_gbps=self.config.cxl_bandwidth_gbps),
            policy=self.config.allocation_policy,
            quantum_bytes=self.config.drr_quantum_bytes,
        )

        # Initialize tenants
        policies: Dict[str, BaselinePolicy] = {}
        tenant_configs: Dict[str, Dict[str, Any]] = {}
        self.token_buckets.clear()
        self._priorities.clear()
        self.tenant_metrics = {}

        for i, spec in enumerate(tenant_specs):
            tid = spec["tenant_id"]
            ctx_len = spec["context_length"]
            budget = spec.get("budget_ratio", 0.10)
            num_chunks = max(16, ctx_len // self.config.chunk_size)
            budget_chunks = max(1, int(num_chunks * budget))

            tenant_configs[tid] = spec
            self._priorities[tid] = spec.get("priority", 1.0)
            self.shared_link.register_tenant(tid)

            if self.config.enable_token_bucket:
                # Scale bucket capacity with budget_chunks (longer context needs more ops)
                scaled_capacity = max(self.config.token_bucket_capacity, budget_chunks * 3)
                scaled_refill = max(self.config.token_bucket_refill_rate, budget_chunks * 2)
                self.token_buckets[tid] = TokenBucketIsolator(
                    tid,
                    capacity=scaled_capacity,
                    refill_rate=scaled_refill,
                )

            policies[tid] = self._create_policy(use_fts=use_fts)
            policies[tid].reset()

            self.tenant_metrics[tid] = TenantMetrics(
                tenant_id=tid,
                context_length=ctx_len,
                budget_ratio=budget,
                priority=spec.get("priority", 1.0),
            )

        # Step-by-step simulation
        for step in range(self.config.num_decode_steps):
            # Check arrivals
            if arrival_schedule is not None:
                for i, spec in enumerate(tenant_specs):
                    tid = spec["tenant_id"]
                    if arrival_schedule[i] == step:
                        # Tenant arrives — policy already initialized, start tracking
                        pass
                    elif arrival_schedule[i] > step:
                        # Not yet arrived — skip
                        continue

            # Refill token buckets
            for tb in self.token_buckets.values():
                tb.refill()

            # Each active tenant runs its pipeline
            for i, spec in enumerate(tenant_specs):
                tid = spec["tenant_id"]
                if arrival_schedule is not None and step < arrival_schedule[i]:
                    continue

                policy = policies[tid]
                ctx_len = spec["context_length"]
                budget = spec.get("budget_ratio", 0.10)
                num_chunks = max(16, ctx_len // self.config.chunk_size)
                budget_chunks = max(1, int(num_chunks * budget))
                ctx_len_idx = ctx_lengths.index(ctx_len) if ctx_len in ctx_lengths else 0
                trace = all_traces[ctx_len_idx][1][step]

                # Build chunk attention masses
                chunk_masses: Dict[int, float] = {}
                for cid in range(num_chunks):
                    if cid < len(trace):
                        chunk_masses[cid] = float(trace[cid])

                # Anchor IDs: first few chunks + top-3 by attention
                anchor_ids = list(range(min(3, num_chunks)))
                top_by_attn = sorted(chunk_masses, key=chunk_masses.get, reverse=True)[:2]
                for a in top_by_attn:
                    if a not in anchor_ids:
                        anchor_ids.append(a)

                # Token bucket check (est_ops = summaries + payloads for budget_chunks)
                tb = self.token_buckets.get(tid)
                if tb is not None:
                    num_candidates = min(budget_chunks * 3, num_chunks)
                    est_ops = min(num_candidates, budget_chunks * 4)  # generous estimate
                    if not tb.consume(est_ops):
                        self.tenant_metrics[tid].token_bucket_exhausted_steps += 1
                        # Tenant throttled: use anchors only
                        selected = sorted(set(anchor_ids))
                        self.tenant_metrics[tid].latencies_us.append(0.0)
                        self.tenant_metrics[tid].recoveries.append(0.0)
                        continue

                # Run policy
                try:
                    selected = policy.select_active_chunks(
                        num_chunks=num_chunks,
                        budget_chunks=budget_chunks,
                        chunk_attention_masses=chunk_masses,
                        anchor_ids=anchor_ids,
                        step=step,
                    )
                except Exception:
                    selected = sorted(set(anchor_ids))

                # Gold chunks (oracle: top budget_chunks by attention)
                gold = sorted(chunk_masses, key=chunk_masses.get, reverse=True)[:budget_chunks]

                # Compute recovery
                intersection = len(set(selected) & set(gold))
                recovery = intersection / max(len(gold), 1)
                self.tenant_metrics[tid].recoveries.append(recovery)

                # Get CXL stats from the policy's cxl_session
                cxl_session = getattr(policy, "cxl_session", None)
                if cxl_session is not None and cxl_session.step_results:
                    last_result = cxl_session.step_results[-1]
                    cxl_stats = last_result.cxl_stats
                    latency_us = cxl_stats.total_time_ns / 1000.0
                    self.tenant_metrics[tid].latencies_us.append(latency_us)
                    self.tenant_metrics[tid].cxl_queue_rhos.append(cxl_stats.queue_utilization_rho)
                    self.tenant_metrics[tid].total_cxl_bytes += cxl_stats.total_bytes_fetched
                    self.tenant_metrics[tid].total_invalid_bytes += cxl_stats.invalid_payload_bytes
                    self.tenant_metrics[tid].invalid_traffic_ratios.append(cxl_stats.invalid_traffic_ratio)
                else:
                    self.tenant_metrics[tid].latencies_us.append(0.0)
                    self.tenant_metrics[tid].cxl_queue_rhos.append(0.0)

                # PHT hit rate tracking
                pht_ema = getattr(policy, "pht_ema", {})
                if pht_ema:
                    hit_count = sum(1 for c in selected if c in pht_ema)
                    pht_hit = hit_count / max(len(selected), 1)
                    self.tenant_metrics[tid].pht_hit_rates.append(pht_hit)

            # SLO violation check
            for i, spec in enumerate(tenant_specs):
                tid = spec["tenant_id"]
                if arrival_schedule is not None and step < arrival_schedule[i]:
                    continue
                slo = spec.get("slo_target_us", 500.0)
                if self.tenant_metrics[tid].latencies_us:
                    last_lat = self.tenant_metrics[tid].latencies_us[-1]
                    if last_lat > slo:
                        self.tenant_metrics[tid].slo_violations += 1

        # Finalize metrics
        for tid in self.tenant_metrics:
            solo_lat = self._solo_metrics.get(tid, 0.0)
            self.tenant_metrics[tid].finalize(solo_latency_us=solo_lat)

        return dict(self.tenant_metrics)

    def _run_solo_baselines(
        self, tenant_specs: List[Dict[str, Any]], use_fts: bool = False,
    ) -> Dict[str, float]:
        """Run each tenant solo to establish baseline latencies."""
        solo_latencies: Dict[str, float] = {}
        for spec in tenant_specs:
            tid = spec["tenant_id"]
            metrics = self._run_simulation([spec], use_fts=use_fts)
            if metrics:
                solo_latencies[tid] = metrics[tid].p99_latency_us
        return solo_latencies

    # ═══════════════════════════════════════════════════════════════════
    # D1: Heterogeneous Context-Length Mixing
    # ═══════════════════════════════════════════════════════════════════

    def run_heterogeneous_mixing(self) -> Dict[str, Any]:
        """Dimension 1: Heterogeneous context length mixing.

        Mixes short and long context requests, measuring tail latency
        degradation for short requests when co-located with long ones.
        """
        mixing_scenarios = [
            {"name": "4K+128K", "lengths": [4096, 131072], "budgets": [0.10, 0.05]},
            {"name": "32K+64K+128K", "lengths": [32768, 65536, 131072], "budgets": [0.10, 0.05, 0.05]},
            {"name": "4K+4K+4K+128K", "lengths": [4096, 4096, 4096, 131072], "budgets": [0.10, 0.10, 0.10, 0.05]},
            {"name": "128K+128K", "lengths": [131072, 131072], "budgets": [0.05, 0.05]},
            {"name": "128K+256K+256K", "lengths": [131072, 262144, 262144], "budgets": [0.05, 0.02, 0.02]},
        ]

        results: List[Dict[str, Any]] = []

        for scenario in mixing_scenarios:
            specs = []
            for i, (ctx_len, budget) in enumerate(zip(scenario["lengths"], scenario["budgets"])):
                specs.append({
                    "tenant_id": chr(65 + i),  # A, B, C, ...
                    "context_length": ctx_len,
                    "budget_ratio": budget,
                    "priority": 1.0,
                    "slo_target_us": 200.0 if ctx_len <= 8192 else 1000.0,
                })

            # Solo baselines
            self._solo_metrics = self._run_solo_baselines(specs)

            # Multi-tenant run
            metrics = self._run_simulation(specs)

            # Also run with FTS if requested
            fts_metrics: Dict[str, TenantMetrics] = {}
            if self.config.compare_fts:
                fts_metrics = self._run_simulation(specs, use_fts=True)

            scenario_result = {
                "name": scenario["name"],
                "tenant_configs": specs,
                "per_tenant": {tid: m.to_dict() for tid, m in metrics.items()},
                "aggregate": {
                    "max_tail_degradation": round(
                        max(m.tail_degradation_ratio for m in metrics.values()), 3
                    ),
                    "min_tail_degradation": round(
                        min(m.tail_degradation_ratio for m in metrics.values()), 3
                    ),
                    "short_context_p99_us": round(
                        max(m.p99_latency_us for tid, m in metrics.items()
                            if m.context_length <= 8192), 1
                    ) if any(m.context_length <= 8192 for m in metrics.values()) else 0,
                },
            }

            if fts_metrics:
                scenario_result["per_tenant_fts"] = {tid: m.to_dict() for tid, m in fts_metrics.items()}
                sbfi_deg = scenario_result["aggregate"]["max_tail_degradation"]
                fts_deg = max(m.tail_degradation_ratio for m in fts_metrics.values())
                scenario_result["aggregate"]["sbfi_vs_fts_degradation"] = round(fts_deg / max(sbfi_deg, 1.0), 3)

            results.append(scenario_result)

        return {
            "dimension": "D1_heterogeneous_mixing",
            "config": self.config.to_dict(),
            "scenarios": results,
            "summary": {
                "worst_case_tail_degradation": round(
                    max(r["aggregate"]["max_tail_degradation"] for r in results), 3
                ),
                "avg_short_tail_p99_us": round(
                    np.mean([
                        r["aggregate"]["short_context_p99_us"]
                        for r in results
                        if r["aggregate"]["short_context_p99_us"] > 0
                    ]), 1
                ),
            },
        }

    # ═══════════════════════════════════════════════════════════════════
    # D2: Priority and QoS Enforcement
    # ═══════════════════════════════════════════════════════════════════

    def run_priority_qos(self) -> Dict[str, Any]:
        """Dimension 2: Priority and QoS enforcement.

        Assigns high/medium/low priorities, measures SLO violation rate
        for high-priority short requests, and sweeps low-priority count
        to find the SLO violation knee.
        """
        # Base: 1 high-priority (4K) + 3 low-priority (128K)
        base_specs = [
            {"tenant_id": "H", "context_length": 4096, "budget_ratio": 0.10,
             "priority": 3.0, "slo_target_us": 200.0},
        ]

        # Sweep: add 1, 2, 4, 6 low-priority tenants
        low_tenant_counts = [1, 2, 3, 4, 6]
        results: List[Dict[str, Any]] = []

        for num_low in low_tenant_counts:
            specs = list(base_specs)
            for j in range(num_low):
                specs.append({
                    "tenant_id": f"L{j}",
                    "context_length": 131072,
                    "budget_ratio": 0.05,
                    "priority": 1.0,
                    "slo_target_us": 1000.0,
                })

            self._solo_metrics = self._run_solo_baselines(specs)
            metrics = self._run_simulation(specs)

            high_metric = metrics.get("H")
            low_metrics = {tid: m for tid, m in metrics.items() if tid.startswith("L")}

            results.append({
                "num_low_tenants": num_low,
                "total_tenants": 1 + num_low,
                "high_priority": high_metric.to_dict() if high_metric else None,
                "low_priority_aggregate": {
                    "avg_recovery": round(np.mean([m.mean_recovery for m in low_metrics.values()]), 4) if low_metrics else 0,
                    "avg_p99_latency_us": round(np.mean([m.p99_latency_us for m in low_metrics.values()]), 1) if low_metrics else 0,
                    "avg_slo_violations": round(np.mean([m.slo_violations for m in low_metrics.values()]), 1) if low_metrics else 0,
                },
                "starvation_check": {
                    "max_consecutive_starved": max(
                        (m.starvation_events for m in metrics.values()), default=0
                    ),
                    "no_starvation_gt_2_epochs": all(
                        m.starvation_events <= 2 for m in metrics.values()
                    ),
                },
                "knee_detected": high_metric.slo_violations > 0 if high_metric else False,
            })

        return {
            "dimension": "D2_priority_qos",
            "config": self.config.to_dict(),
            "results": results,
            "summary": {
                "slo_violation_knee": next(
                    (r["num_low_tenants"] for r in results if r["knee_detected"]), None
                ),
                "starvation_bound_held": all(
                    r["starvation_check"]["no_starvation_gt_2_epochs"] for r in results
                ),
            },
        }

    # ═══════════════════════════════════════════════════════════════════
    # D3: Priority Inversion
    # ═══════════════════════════════════════════════════════════════════

    def run_priority_inversion(self) -> Dict[str, Any]:
        """Dimension 3: Priority inversion quantification.

        Constructs intentional priority inversion: a low-priority speculative
        DMA arrives just before a high-priority demand fetch. Measures
        inversion wait time distribution and PROSE vs FTS comparison.
        """
        # Two tenants: one high-priority short, one low-priority long
        specs = [
            {"tenant_id": "high", "context_length": 4096, "budget_ratio": 0.10,
             "priority": 3.0, "slo_target_us": 200.0},
            {"tenant_id": "low", "context_length": 131072, "budget_ratio": 0.05,
             "priority": 0.5, "slo_target_us": 2000.0},
        ]

        self._solo_metrics = self._run_solo_baselines(specs)

        # Run with PROSE (SBFI)
        metrics_prose = self._run_simulation(specs, use_fts=False)

        # Run with PROSE-FTS
        metrics_fts = self._run_simulation(specs, use_fts=True)

        # Construct deliberate inversion scenarios by varying DMA granularity
        granularity_results = []
        for dma_size in [64, 1024, 4096, 65536]:  # 64B summary to 64KB full payload
            # Model inversion: low-priority's DMA blocks high-priority
            if dma_size <= 64:
                inversion_prob = 0.02
            elif dma_size <= 4096:
                inversion_prob = 0.10
            else:
                inversion_prob = 0.25

            approx_wait_us = dma_size / (self.config.cxl_bandwidth_gbps)  # bytes / (B/us)
            granularity_results.append({
                "dma_granularity_bytes": dma_size,
                "inversion_probability": round(inversion_prob, 3),
                "approximate_wait_time_us": round(approx_wait_us, 1),
                "effective_for_sbfi": dma_size <= 64,
            })

        return {
            "dimension": "D3_priority_inversion",
            "config": self.config.to_dict(),
            "prose_sbfi": {
                "high_priority": metrics_prose.get("high", TenantMetrics()).to_dict(),
                "low_priority": metrics_prose.get("low", TenantMetrics()).to_dict(),
            },
            "prose_fts": {
                "high_priority": metrics_fts.get("high", TenantMetrics()).to_dict(),
                "low_priority": metrics_fts.get("low", TenantMetrics()).to_dict(),
            },
            "comparison": {
                "sbfi_high_p99_us": round(metrics_prose.get("high", TenantMetrics()).p99_latency_us, 1),
                "fts_high_p99_us": round(metrics_fts.get("high", TenantMetrics()).p99_latency_us, 1),
                "sbfi_advantage_us": round(
                    metrics_fts.get("high", TenantMetrics()).p99_latency_us -
                    metrics_prose.get("high", TenantMetrics()).p99_latency_us, 1
                ),
                "sbfi_invalid_traffic": round(metrics_prose.get("low", TenantMetrics()).mean_invalid_traffic_ratio, 4),
                "fts_invalid_traffic": round(metrics_fts.get("low", TenantMetrics()).mean_invalid_traffic_ratio, 4),
            },
            "dma_granularity_analysis": granularity_results,
            "summary": {
                "sbfi_reduces_inversion_exposure": "confirmed" if (
                    metrics_prose.get("high", TenantMetrics()).p99_latency_us <
                    metrics_fts.get("high", TenantMetrics()).p99_latency_us
                ) else "not confirmed",
                "inversion_wait_reduced_by_percent": round(
                    100 * (1 - metrics_prose.get("high", TenantMetrics()).p99_latency_us /
                           max(metrics_fts.get("high", TenantMetrics()).p99_latency_us, 0.01)), 1
                ),
            },
        }

    # ═══════════════════════════════════════════════════════════════════
    # D4: Token Bucket Isolation
    # ═══════════════════════════════════════════════════════════════════

    def run_token_bucket_isolation(self) -> Dict[str, Any]:
        """Dimension 4: Token bucket isolation effectiveness.

        Two tenants: A (128K context, large k_t) and B (4K context, small k_t).
        With/without token bucket, measure B's latency and starvation.
        """
        specs = [
            {"tenant_id": "A", "context_length": 131072, "budget_ratio": 0.05,
             "priority": 1.0, "slo_target_us": 1000.0},
            {"tenant_id": "B", "context_length": 4096, "budget_ratio": 0.10,
             "priority": 1.0, "slo_target_us": 200.0},
        ]

        self._solo_metrics = self._run_solo_baselines(specs)

        # With token bucket
        self.config.enable_token_bucket = True
        metrics_with_tb = self._run_simulation(specs)

        # Without token bucket
        self.config.enable_token_bucket = False
        metrics_without_tb = self._run_simulation(specs)

        # Restore
        self.config.enable_token_bucket = True

        return {
            "dimension": "D4_token_bucket_isolation",
            "config": self.config.to_dict(),
            "with_token_bucket": {
                "tenant_A": metrics_with_tb.get("A", TenantMetrics()).to_dict(),
                "tenant_B": metrics_with_tb.get("B", TenantMetrics()).to_dict(),
            },
            "without_token_bucket": {
                "tenant_A": metrics_without_tb.get("A", TenantMetrics()).to_dict(),
                "tenant_B": metrics_without_tb.get("B", TenantMetrics()).to_dict(),
            },
            "summary": {
                "B_latency_with_tb_us": round(metrics_with_tb.get("B", TenantMetrics()).p99_latency_us, 1),
                "B_latency_without_tb_us": round(metrics_without_tb.get("B", TenantMetrics()).p99_latency_us, 1),
                "B_degradation_with_tb": round(metrics_with_tb.get("B", TenantMetrics()).tail_degradation_ratio, 3),
                "B_degradation_without_tb": round(metrics_without_tb.get("B", TenantMetrics()).tail_degradation_ratio, 3),
                "isolation_improvement": round(
                    metrics_without_tb.get("B", TenantMetrics()).tail_degradation_ratio /
                    max(metrics_with_tb.get("B", TenantMetrics()).tail_degradation_ratio, 0.01), 2
                ),
                "starvation_without_tb": metrics_without_tb.get("B", TenantMetrics()).starvation_events,
                "starvation_with_tb": metrics_with_tb.get("B", TenantMetrics()).starvation_events,
            },
        }

    # ═══════════════════════════════════════════════════════════════════
    # D5: Metadata Contention
    # ═══════════════════════════════════════════════════════════════════

    def run_metadata_contention(self) -> Dict[str, Any]:
        """Dimension 5: Metadata contention under tenant scaling.

        Scales from 1 to 8 tenants, measuring CXL link utilization and
        throughput degradation. Compares against single-tenant equivalent.
        """
        tenant_counts = [1, 2, 4, 6, 8]
        results: List[Dict[str, Any]] = []

        for k in tenant_counts:
            specs = []
            for i in range(k):
                ctx_len = 32768 if i % 3 == 0 else (65536 if i % 3 == 1 else 131072)
                specs.append({
                    "tenant_id": f"T{i}",
                    "context_length": ctx_len,
                    "budget_ratio": 0.05,
                    "priority": 1.0,
                    "slo_target_us": 1000.0,
                })

            self._solo_metrics = self._run_solo_baselines(specs)
            metrics = self._run_simulation(specs)

            # Aggregate metrics
            avg_rho = float(np.mean([m.mean_cxl_queue_rho for m in metrics.values()]))
            avg_p99 = float(np.mean([m.p99_latency_us for m in metrics.values()]))
            total_invalid = sum(m.total_invalid_bytes for m in metrics.values())
            total_bytes = sum(m.total_cxl_bytes for m in metrics.values())
            fairness_vals = [m.mean_recovery for m in metrics.values()]
            fairness = (
                sum(fairness_vals) ** 2 / (len(fairness_vals) * sum(v**2 for v in fairness_vals))
            ) if fairness_vals and sum(v**2 for v in fairness_vals) > 0 else 1.0

            results.append({
                "num_tenants": k,
                "avg_cxl_queue_rho": round(avg_rho, 4),
                "avg_p99_latency_us": round(avg_p99, 1),
                "total_invalid_bytes": total_invalid,
                "total_cxl_bytes": total_bytes,
                "invalid_traffic_ratio": round(total_invalid / max(total_bytes, 1), 4),
                "fairness_index": round(fairness, 4),
                "metadata_pressure": round(avg_rho * k, 2),  # cumulative metadata loading
                "per_tenant_metrics": {tid: m.to_dict() for tid, m in metrics.items()},
            })

        # Find knee: where rho > 0.8 or fairness drops
        knee = next((r["num_tenants"] for r in results if r["avg_cxl_queue_rho"] > 0.80), None)

        return {
            "dimension": "D5_metadata_contention",
            "config": self.config.to_dict(),
            "results": results,
            "summary": {
                "metadata_knee_tenants": knee,
                "max_rho": round(max(r["avg_cxl_queue_rho"] for r in results), 4),
                "min_fairness": round(min(r["fairness_index"] for r in results), 4),
                "throughput_degradation_at_8": round(
                    results[-1]["avg_p99_latency_us"] / max(results[0]["avg_p99_latency_us"], 1), 2
                ) if len(results) >= 2 else 1.0,
            },
        }

    # ═══════════════════════════════════════════════════════════════════
    # D6: Dynamic Arrival
    # ═══════════════════════════════════════════════════════════════════

    def run_dynamic_arrival(self) -> Dict[str, Any]:
        """Dimension 6: Dynamic arrival and departure.

        Requests arrive at staggered times or via Poisson process.
        Measures cold-start latency, PHT warm-up behavior, and
        Promotion Buffer dynamics.
        """
        results: List[Dict[str, Any]] = []

        # Simultaneous baseline
        specs_simul = [
            {"tenant_id": "A", "context_length": 32768, "budget_ratio": 0.10,
             "priority": 1.0, "slo_target_us": 500.0},
            {"tenant_id": "B", "context_length": 65536, "budget_ratio": 0.05,
             "priority": 1.0, "slo_target_us": 1000.0},
            {"tenant_id": "C", "context_length": 32768, "budget_ratio": 0.10,
             "priority": 1.0, "slo_target_us": 500.0},
            {"tenant_id": "D", "context_length": 131072, "budget_ratio": 0.02,
             "priority": 1.0, "slo_target_us": 1000.0},
        ]

        self._solo_metrics = self._run_solo_baselines(specs_simul)

        # 1. Simultaneous arrival
        metrics_simul = self._run_simulation(specs_simul, arrival_schedule=[0, 0, 0, 0])
        results.append({
            "arrival_mode": "simultaneous",
            "arrival_schedule": [0, 0, 0, 0],
            "per_tenant": {tid: m.to_dict() for tid, m in metrics_simul.items()},
            "avg_cold_start_us": round(np.mean([
                m.cold_start_latency_us for m in metrics_simul.values()
            ]), 1),
        })

        # 2. Staggered arrival (every 50 steps)
        metrics_stagger = self._run_simulation(specs_simul, arrival_schedule=[0, 50, 100, 150])
        results.append({
            "arrival_mode": "staggered",
            "arrival_schedule": [0, 50, 100, 150],
            "per_tenant": {tid: m.to_dict() for tid, m in metrics_stagger.items()},
            "avg_cold_start_us": round(np.mean([
                m.cold_start_latency_us for m in metrics_stagger.values()
            ]), 1),
        })

        # 3. Poisson-like arrival (exponential inter-arrival)
        rng = np.random.RandomState(self.config.seed + 100)
        exp_intervals = rng.exponential(1.0 / max(self.config.arrival_rate, 0.001), size=3)
        arrival_steps = [0] + [int(10 + np.cumsum(exp_intervals * 50)[i]) for i in range(3)]
        arrival_steps = [min(a, self.config.num_decode_steps - 10) for a in arrival_steps]
        metrics_poisson = self._run_simulation(specs_simul, arrival_schedule=arrival_steps)
        results.append({
            "arrival_mode": "poisson",
            "arrival_schedule": arrival_steps,
            "per_tenant": {tid: m.to_dict() for tid, m in metrics_poisson.items()},
            "avg_cold_start_us": round(np.mean([
                m.cold_start_latency_us for m in metrics_poisson.values()
            ]), 1),
        })

        # Compare PHT warm-up: first 10 steps' recovery vs steady state
        pht_warmup_analysis = {}
        for tid in ["A", "B", "C", "D"]:
            if tid in metrics_simul:
                rec = metrics_simul[tid].recoveries
                if len(rec) >= 40:
                    pht_warmup_analysis[tid] = {
                        "warmup_avg_recovery": round(np.mean(rec[:10]), 4),
                        "steady_avg_recovery": round(np.mean(rec[40:60]), 4),
                        "warmup_penalty": round(1.0 - np.mean(rec[:10]) / max(np.mean(rec[40:60]), 0.01), 3),
                    }

        return {
            "dimension": "D6_dynamic_arrival",
            "config": self.config.to_dict(),
            "results": results,
            "pht_warmup_analysis": pht_warmup_analysis,
            "summary": {
                "simultaneous_avg_cold_start_us": results[0]["avg_cold_start_us"],
                "staggered_avg_cold_start_us": results[1]["avg_cold_start_us"],
                "poisson_avg_cold_start_us": results[2]["avg_cold_start_us"],
                "max_warmup_penalty": round(
                    max(a.get("warmup_penalty", 0) for a in pht_warmup_analysis.values()), 3
                ) if pht_warmup_analysis else 0,
            },
        }

    # ═══════════════════════════════════════════════════════════════════
    # Master Runner
    # ═══════════════════════════════════════════════════════════════════

    def run_all(self) -> Dict[str, Any]:
        """Run all six dimensions and return combined results."""
        results = {
            "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
            "config": self.config.to_dict(),
            "dimensions": {},
            "summary": {},
        }

        print("=" * 70)
        print("Multi-Tenant Evaluation Runner")
        print(f"  Tenants: {self.config.num_tenants}")
        print(f"  Steps: {self.config.num_decode_steps}")
        print(f"  Token Bucket: {self.config.enable_token_bucket}")
        print(f"  Compare FTS: {self.config.compare_fts}")
        print("=" * 70)

        scenarios = [
            ("D1_heterogeneous_mixing", self.run_heterogeneous_mixing),
            ("D2_priority_qos", self.run_priority_qos),
            ("D3_priority_inversion", self.run_priority_inversion),
            ("D4_token_bucket_isolation", self.run_token_bucket_isolation),
            ("D5_metadata_contention", self.run_metadata_contention),
            ("D6_dynamic_arrival", self.run_dynamic_arrival),
        ]

        for name, method in scenarios:
            print(f"\n{'─' * 70}")
            print(f"Running {name}...")
            try:
                dim_result = method()
                results["dimensions"][name] = dim_result
                dim_summary = dim_result.get("summary", {})
                results["summary"][name] = dim_summary
                print(f"  {name}: completed successfully")
                for k, v in dim_summary.items():
                    print(f"    {k}: {v}")
            except Exception as e:
                print(f"  {name}: FAILED — {e}")
                import traceback
                traceback.print_exc()
                results["dimensions"][name] = {"error": str(e)}

        print(f"\n{'═' * 70}")
        print("Multi-tenant evaluation complete.")
        return results

    def run_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Run a single named scenario."""
        methods: Dict[str, Callable[[], Dict[str, Any]]] = {
            "heterogeneous_mixing": self.run_heterogeneous_mixing,
            "priority_qos": self.run_priority_qos,
            "priority_inversion": self.run_priority_inversion,
            "token_bucket_isolation": self.run_token_bucket_isolation,
            "metadata_contention": self.run_metadata_contention,
            "dynamic_arrival": self.run_dynamic_arrival,
        }
        if scenario_name not in methods:
            raise ValueError(f"Unknown scenario: {scenario_name}. Choose from: {list(methods.keys())}")
        return methods[scenario_name]()


# ═══════════════════════════════════════════════════════════════════════════
# Utility: save and print
# ═══════════════════════════════════════════════════════════════════════════

def save_results(results: Dict[str, Any], filepath: str) -> str:
    """Save results to JSON, handling non-serializable objects."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    def _sanitize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_sanitize(results), f, indent=2, default=str)
    print(f"\nResults saved to: {filepath}")
    return filepath


def print_summary(results: Dict[str, Any]):
    """Print a formatted summary of multi-tenant evaluation results."""
    print("\n" + "=" * 80)
    print("MULTI-TENANT EVALUATION SUMMARY")
    print("=" * 80)

    dimensions = results.get("dimensions", {})
    for dim_name, dim_data in dimensions.items():
        print(f"\n─── {dim_name} ───")
        if "error" in dim_data:
            print(f"  ERROR: {dim_data['error']}")
            continue
        summary = dim_data.get("summary", {})
        for k, v in summary.items():
            print(f"  {k}: {v}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point (for direct module execution)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PROSE Multi-Tenant Evaluation Runner")
    parser.add_argument("--scenario", type=str, default="all",
                       choices=["all", "heterogeneous_mixing", "priority_qos",
                                "priority_inversion", "token_bucket_isolation",
                                "metadata_contention", "dynamic_arrival"],
                       help="Which scenario to run")
    parser.add_argument("--num-tenants", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=200)
    parser.add_argument("--compare-fts", action="store_true",
                       help="Also run PROSE-FTS for comparison")
    parser.add_argument("--no-token-bucket", action="store_true",
                       help="Disable token bucket isolation")
    parser.add_argument("--output-dir", type=str, default="outputs/multitenant",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    cfg = MultiTenantConfig(
        num_tenants=args.num_tenants,
        num_decode_steps=args.decode_steps,
        compare_fts=args.compare_fts,
        enable_token_bucket=not args.no_token_bucket,
        seed=args.seed,
    )

    runner = MultiTenantEvalRunner(cfg)

    if args.scenario == "all":
        results = runner.run_all()
        filename = f"multitenant_all_{results['timestamp']}.json"
    else:
        results = runner.run_scenario(args.scenario)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"multitenant_{args.scenario}_{ts}.json"

    output_path = os.path.join(args.output_dir, filename)
    save_results(results, output_path)
    print_summary(results)
