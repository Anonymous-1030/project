"""
Multi-Tier Memory Hierarchy Simulator — Cascaded Admission Across 4 Storage Tiers.

Extends the CXLQueueSimulator pattern from cxl_queue_simulator.py to model a
full storage hierarchy:

  Tier 0: HBM           — GPU local, committed residency, ~0.1us, 3350 GB/s
  Tier 1: CXL DRAM      — CXL.mem attached, ~200ns, 64 GB/s
  Tier 2: Local SSD / CXL Flash — NVMe or CXL-flash, ~10us, 7 GB/s
  Tier 3: Remote Memory — RDMA / disaggregated pool, ~50us, 25 GB/s

Key modeling: each tier has an independent M/D/1 queue simulator, and
promotion between tiers is gated by an evidence-based admission protocol
(implemented in evidence_hierarchy.py).

References:
  - CXL 3.0 Specification
  - Samsung ISSCC 2022 (CXL memory expander latency)
  - Meta MICRO 2023 (disaggregated memory characterization)
  - NVMe 2.0 baseline (PCIe Gen4 x4 SSD)
  - RDMA over Converged Ethernet (RoCEv2) latency model
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


# ── Tier Specification ─────────────────────────────────────────────────

class TierSpec(Enum):
    """Physical storage tier identifiers."""
    HBM = 0
    CXL_DRAM = 1
    CXL_FLASH = 2       # CXL-attached flash or local NVMe SSD
    REMOTE_RDMA = 3     # Network-attached / disaggregated memory pool

    @property
    def label(self) -> str:
        return _TIER_NAMES.get(self, str(self))

    @property
    def short_label(self) -> str:
        return _TIER_SHORT_NAMES.get(self, "???")


# Friendly names for display (internal, used by TierSpec properties)
_TIER_NAMES = {
    TierSpec.HBM: "HBM",
    TierSpec.CXL_DRAM: "CXL-DRAM",
    TierSpec.CXL_FLASH: "CXL-Flash/SSD",
    TierSpec.REMOTE_RDMA: "Remote-RDMA",
}

_TIER_SHORT_NAMES = {
    TierSpec.HBM: "HBM",
    TierSpec.CXL_DRAM: "CXL",
    TierSpec.CXL_FLASH: "SSD",
    TierSpec.REMOTE_RDMA: "RMT",
}

# Re-export for backwards compatibility
TIER_NAMES = _TIER_NAMES
TIER_SHORT_NAMES = _TIER_SHORT_NAMES


# ── Tier Configuration ─────────────────────────────────────────────────

@dataclass
class TierConfig:
    """Configuration for a single storage tier.

    Models the physical characteristics: access latency, sustained bandwidth,
    queue depth, and the evidence/payload sizes used at this tier boundary.
    """
    tier: TierSpec

    # Access latency (round-trip, including controller + media)
    access_latency_ns: float = 200.0       # Base RTT for a small read

    # Sustained bandwidth (achievable, after protocol overhead)
    sustained_bandwidth_gbps: float = 64.0  # GB/s = bytes/ns

    # Queue depth at the tier controller
    queue_depth: int = 48

    # Protocol processing latency (controller + bridge)
    protocol_processing_ns: float = 15.0

    # Payload sizes
    chunk_size_bytes: int = 65536          # Full KV chunk (512 tok × 128B)

    # Evidence size at this tier boundary (for admission gating)
    # This is the DEFAULT — may be overridden by evidence ladder strategy
    default_evidence_size_bytes: int = 64

    # Media access model parameters
    # For DRAM-like tiers (HBM, CXL-DRAM): row buffer hit/miss
    # For block-device tiers (SSD): page hit / page miss
    media_row_hit_ns: float = 40.0
    media_row_miss_ns: float = 120.0
    media_row_buffer_hit_rate: float = 0.30
    media_num_parallel_channels: int = 4

    # Credit-based flow control RTT (CXL, RDMA)
    credit_rtt_ns: float = 100.0

    @property
    def bytes_per_ns(self) -> float:
        return self.sustained_bandwidth_gbps

    @property
    def label(self) -> str:
        return TIER_NAMES.get(self.tier, str(self.tier))

    @property
    def short_label(self) -> str:
        return TIER_SHORT_NAMES.get(self.tier, "???")


# ── Pre-configured tier profiles ───────────────────────────────────────

def make_hbm_config() -> TierConfig:
    """HBM3 on H100-SXM: ~0.1us latency, 3350 GB/s."""
    return TierConfig(
        tier=TierSpec.HBM,
        access_latency_ns=100.0,
        sustained_bandwidth_gbps=3350.0,
        queue_depth=256,
        protocol_processing_ns=5.0,
        default_evidence_size_bytes=0,  # No evidence needed — local
        media_row_hit_ns=20.0,
        media_row_miss_ns=60.0,
        media_row_buffer_hit_rate=0.60,
        media_num_parallel_channels=32,
        credit_rtt_ns=10.0,
    )


def make_cxl_dram_config() -> TierConfig:
    """CXL 3.0 Type-3 memory expander with DDR5-4800 backend."""
    return TierConfig(
        tier=TierSpec.CXL_DRAM,
        access_latency_ns=200.0,
        sustained_bandwidth_gbps=64.0,
        queue_depth=48,
        protocol_processing_ns=15.0,
        default_evidence_size_bytes=16,  # 16B bloom fingerprint
        media_row_hit_ns=40.0,
        media_row_miss_ns=120.0,
        media_row_buffer_hit_rate=0.30,
        media_num_parallel_channels=4,
        credit_rtt_ns=100.0,
    )


def make_cxl_flash_config() -> TierConfig:
    """CXL-attached flash or local NVMe SSD.

    Models PCIe Gen4 x4 NVMe: ~7 GB/s sustained read, ~10us access latency.
    For CXL-flash: adds CXL.mem protocol overhead but similar media latency.
    """
    return TierConfig(
        tier=TierSpec.CXL_FLASH,
        access_latency_ns=10_000.0,       # 10us (NVMe NAND read latency)
        sustained_bandwidth_gbps=7.0,      # PCIe Gen4 x4 = ~7 GB/s
        queue_depth=256,                    # NVMe supports deep queues
        protocol_processing_ns=500.0,      # NVMe command processing
        default_evidence_size_bytes=64,     # 64B sketch
        media_row_hit_ns=8000.0,           # Page hit (NAND read)
        media_row_miss_ns=15000.0,         # Page miss (NAND + FTL)
        media_row_buffer_hit_rate=0.70,     # Higher hit rate (larger pages)
        media_num_parallel_channels=4,     # NVMe: 4 lanes
        credit_rtt_ns=2000.0,
    )


def make_remote_rdma_config() -> TierConfig:
    """RDMA over RoCEv2 to disaggregated memory pool.

    Models: 25 Gbps NIC, ~50us RTT (fabric + memory controller + response).
    """
    return TierConfig(
        tier=TierSpec.REMOTE_RDMA,
        access_latency_ns=50_000.0,        # 50us (RDMA read over fabric)
        sustained_bandwidth_gbps=25.0,      # 25 Gbps RoCE
        queue_depth=128,                    # RDMA QP depth
        protocol_processing_ns=2000.0,     # RDMA verbs + transport
        default_evidence_size_bytes=256,    # 256B compressed fragment
        media_row_hit_ns=40000.0,          # Remote DRAM hit
        media_row_miss_ns=80000.0,         # Remote DRAM miss (NUMA-like)
        media_row_buffer_hit_rate=0.40,
        media_num_parallel_channels=1,
        credit_rtt_ns=50000.0,
    )


# Registry
def get_default_tier_configs() -> Dict[TierSpec, TierConfig]:
    return {
        TierSpec.HBM: make_hbm_config(),
        TierSpec.CXL_DRAM: make_cxl_dram_config(),
        TierSpec.CXL_FLASH: make_cxl_flash_config(),
        TierSpec.REMOTE_RDMA: make_remote_rdma_config(),
    }


# ── Tier-Level Queue Simulator ─────────────────────────────────────────

@dataclass
class TierFetchResult:
    """Result of a fetch operation at a single tier."""
    tier: TierSpec
    chunk_ids: List[int] = field(default_factory=list)
    total_bytes: int = 0
    serialization_ns: float = 0.0
    media_access_ns: float = 0.0
    protocol_ns: float = 0.0
    queuing_ns: float = 0.0
    total_ns: float = 0.0
    queue_depth_at_submit: int = 0
    accepted: bool = True

    @property
    def total_us(self) -> float:
        return self.total_ns / 1000.0


@dataclass
class TierStepStats:
    """Per-step accounting for a single tier."""
    tier: TierSpec

    # Traffic breakdown
    total_bytes_fetched: int = 0
    evidence_bytes_fetched: int = 0
    payload_bytes_fetched: int = 0
    invalid_payload_bytes: int = 0       # Payload fetched but never used
    invalid_evidence_bytes: int = 0      # Evidence fetched for rejected chunks

    # Queue metrics
    queue_utilization_rho: float = 0.0
    queue_depth_peak: int = 0
    queue_depth_mean: float = 0.0
    queue_full_events: int = 0

    # Timing
    total_queuing_ns: float = 0.0
    total_serialization_ns: float = 0.0
    total_media_ns: float = 0.0
    total_protocol_ns: float = 0.0
    total_time_ns: float = 0.0

    # Chunk-level accounting
    total_chunks_requested: int = 0
    valid_chunks_promoted: int = 0
    invalid_chunks: int = 0              # Fetched but not used

    # Byte-at-risk: bytes residing at or below this tier × miss probability
    cumulative_byte_at_risk: int = 0

    @property
    def invalid_traffic_ratio(self) -> float:
        if self.payload_bytes_fetched == 0:
            return 0.0
        return self.invalid_payload_bytes / self.payload_bytes_fetched

    @property
    def false_positive_ratio(self) -> float:
        """Fraction of promoted chunks that turned out useless."""
        total = self.valid_chunks_promoted + self.invalid_chunks
        if total == 0:
            return 0.0
        return self.invalid_chunks / total

    @property
    def saturation_multiplier(self) -> float:
        if self.total_serialization_ns == 0:
            return 1.0
        return (self.total_serialization_ns + self.total_queuing_ns) / self.total_serialization_ns


class TierQueueSimulator:
    """M/D/1 queue simulator for a single storage tier.

    Models deterministic service time per request with Pollaczek-Khinchine
    waiting time. Shares the same modeling approach as CXLQueueSimulator
    but generalized for any storage medium.
    """

    def __init__(self, config: TierConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._queue_bytes: List[float] = []          # Pending request sizes
        self._cumulative_service_ns: float = 0.0
        self._step_stats = TierStepStats(tier=self.cfg.tier)
        self._step_fetched: Dict[int, bool] = {}     # chunk_id -> was_useful
        self._step_queue_depths: List[int] = []

    # ── Service time model ──────────────────────────────────────────

    def _service_time_ns(self, total_bytes: int) -> float:
        """Compute deterministic service time for a read request."""
        cfg = self.cfg

        # 1. Media access time
        if total_bytes <= 4096:
            # Small read: per-access latency dominates
            burst_bytes = 64
            num_accesses = max(1, math.ceil(total_bytes / burst_bytes))
            effective = math.ceil(num_accesses / max(1, cfg.media_num_parallel_channels))
            avg_per_access = (
                cfg.media_row_buffer_hit_rate * cfg.media_row_hit_ns
                + (1.0 - cfg.media_row_buffer_hit_rate) * cfg.media_row_miss_ns
            )
            media_ns = effective * avg_per_access
        else:
            # Large read: bandwidth-limited + initial access penalty
            bw_ns = total_bytes / max(cfg.sustained_bandwidth_gbps, 0.001)
            media_ns = bw_ns + cfg.media_row_miss_ns

        # 2. Protocol processing
        protocol_ns = 2 * cfg.protocol_processing_ns

        # 3. Link serialization
        serialization_ns = total_bytes / max(cfg.sustained_bandwidth_gbps, 0.001)

        # Total service time
        return media_ns + protocol_ns + cfg.access_latency_ns

    def _md1_waiting_time_ns(self, arrival_rate_per_ns: float, service_time_ns: float) -> float:
        """M/D/1 Pollaczek-Khinchine waiting time."""
        rho = arrival_rate_per_ns * service_time_ns
        if rho >= 0.99:
            return service_time_ns * 50.0
        if rho <= 0.0:
            return 0.0
        return (rho * service_time_ns) / (2.0 * max(1.0 - rho, 0.01))

    # ── Fetch operations ────────────────────────────────────────────

    def submit_evidence_fetch(self, chunk_ids: List[int],
                               evidence_size_bytes: int) -> TierFetchResult:
        """Fetch evidence/metadata for candidate chunks."""
        if not chunk_ids:
            return TierFetchResult(tier=self.cfg.tier)

        total_bytes = len(chunk_ids) * evidence_size_bytes
        return self._submit(chunk_ids, total_bytes, is_evidence=True,
                            evidence_size=evidence_size_bytes)

    def submit_payload_fetch(self, chunk_ids: List[int],
                              bytes_per_chunk: Optional[int] = None) -> TierFetchResult:
        """Fetch full payload chunks."""
        if not chunk_ids:
            return TierFetchResult(tier=self.cfg.tier)

        bpc = bytes_per_chunk or self.cfg.chunk_size_bytes
        total_bytes = len(chunk_ids) * bpc
        return self._submit(chunk_ids, total_bytes, is_evidence=False)

    def _submit(self, chunk_ids: List[int], total_bytes: int,
                is_evidence: bool = False,
                evidence_size: int = 0) -> TierFetchResult:
        """Internal submit."""
        result = TierFetchResult(tier=self.cfg.tier, chunk_ids=list(chunk_ids))

        if not chunk_ids:
            return result

        result.total_bytes = total_bytes

        # Service time
        service_ns = self._service_time_ns(total_bytes)
        result.total_ns = service_ns
        result.serialization_ns = total_bytes / max(self.cfg.sustained_bandwidth_gbps, 0.001)

        # Update queue state
        self._queue_bytes.append(float(total_bytes))
        result.queue_depth_at_submit = len(self._queue_bytes)

        if len(self._queue_bytes) > self.cfg.queue_depth:
            self._step_stats.queue_full_events += 1

        self._step_queue_depths.append(len(self._queue_bytes))
        self._cumulative_service_ns += service_ns

        # Update step stats
        if is_evidence:
            self._step_stats.evidence_bytes_fetched += total_bytes
        else:
            self._step_stats.payload_bytes_fetched += total_bytes
            # Track for usefulness accounting
            for cid in chunk_ids:
                self._step_fetched[cid] = False

        self._step_stats.total_bytes_fetched += total_bytes
        self._step_stats.total_chunks_requested += len(chunk_ids)
        self._step_stats.total_serialization_ns += result.serialization_ns
        self._step_stats.total_media_ns += service_ns
        self._step_stats.total_protocol_ns += 2 * self.cfg.protocol_processing_ns
        self._step_stats.total_time_ns += service_ns

        return result

    # ── Post-fetch tracking ─────────────────────────────────────────

    def mark_chunks_used(self, chunk_ids: List[int]):
        for cid in chunk_ids:
            if cid in self._step_fetched:
                self._step_fetched[cid] = True

    def mark_chunks_invalid(self, chunk_ids: List[int]):
        for cid in chunk_ids:
            if cid in self._step_fetched:
                if not self._step_fetched[cid]:
                    self._step_stats.invalid_payload_bytes += self.cfg.chunk_size_bytes
                    self._step_stats.invalid_chunks += 1
                    self._step_fetched[cid] = True  # prevent double-count

    def mark_evidence_invalid(self, num_rejected: int, evidence_size_bytes: int):
        """Track evidence bytes that were fetched for ultimately-rejected chunks."""
        self._step_stats.invalid_evidence_bytes += num_rejected * evidence_size_bytes

    # ── End-of-step ─────────────────────────────────────────────────

    def end_step(self, decode_step_ns: float = 100_000.0,
                  num_chunks_resident: int = 0,
                  miss_probability: float = 0.0) -> TierStepStats:
        """Finalize per-step stats with M/D/1 queuing and byte-at-risk."""

        # Count remaining unmarked payload chunks as invalid
        for cid, was_useful in self._step_fetched.items():
            if not was_useful:
                self._step_stats.invalid_payload_bytes += self.cfg.chunk_size_bytes
                self._step_stats.invalid_chunks += 1
            else:
                self._step_stats.valid_chunks_promoted += 1

        # Queue metrics
        if self._step_queue_depths:
            self._step_stats.queue_depth_peak = max(self._step_queue_depths)
            self._step_stats.queue_depth_mean = (
                sum(self._step_queue_depths) / len(self._step_queue_depths)
            )

        # M/D/1 queuing
        total_service = (self._step_stats.total_serialization_ns
                         + self._step_stats.total_media_ns
                         + self._step_stats.total_protocol_ns)
        if decode_step_ns > 0 and total_service > 0:
            rho = min(0.99, total_service / decode_step_ns)
            step_queuing = (rho * total_service) / (2.0 * max(1.0 - rho, 0.01))
            self._step_stats.queue_utilization_rho = rho
            self._step_stats.total_queuing_ns = step_queuing
            self._step_stats.total_time_ns = total_service + step_queuing

        # Byte-at-risk: bytes resident at/below this tier × miss probability
        if num_chunks_resident > 0 and miss_probability > 0:
            bytes_resident = num_chunks_resident * self.cfg.chunk_size_bytes
            self._step_stats.cumulative_byte_at_risk = int(bytes_resident * miss_probability)

        stats = self._step_stats

        # Reset accumulators
        self._step_stats = TierStepStats(tier=self.cfg.tier)
        self._step_fetched = {}
        self._step_queue_depths = []

        return stats


# ── Multi-Tier Simulator ───────────────────────────────────────────────

@dataclass
class MultiTierStepStats:
    """Aggregated per-step stats across all tiers."""
    step: int
    per_tier: Dict[TierSpec, TierStepStats] = field(default_factory=dict)

    # Cross-tier promotion counts
    cross_tier_promotions: Dict[Tuple[TierSpec, TierSpec], int] = field(default_factory=dict)
    cross_tier_payload_bytes: Dict[Tuple[TierSpec, TierSpec], int] = field(default_factory=dict)
    cross_tier_evidence_bytes: Dict[Tuple[TierSpec, TierSpec], int] = field(default_factory=dict)

    # Useful-KV recovery
    gold_chunks: List[int] = field(default_factory=list)
    recovered_chunks: List[int] = field(default_factory=list)

    # Aggregate latency
    total_stall_ns: float = 0.0  # Sum of queuing across all tiers

    @property
    def recovery(self) -> float:
        if not self.gold_chunks:
            return 0.0
        return len(set(self.recovered_chunks) & set(self.gold_chunks)) / len(self.gold_chunks)

    @property
    def total_cross_tier_payload_bytes(self) -> int:
        return sum(self.cross_tier_payload_bytes.values())

    @property
    def total_cross_tier_evidence_bytes(self) -> int:
        return sum(self.cross_tier_evidence_bytes.values())

    @property
    def total_bytes_at_risk(self) -> int:
        return sum(s.cumulative_byte_at_risk for s in self.per_tier.values())


@dataclass
class MultiTierResult:
    """Aggregate result from a multi-tier experiment run."""
    num_steps: int
    tier_configs: Dict[TierSpec, TierConfig]
    step_results: List[MultiTierStepStats] = field(default_factory=list)

    # Accumulated metrics
    total_evidence_bytes: Dict[TierSpec, int] = field(default_factory=dict)
    total_payload_bytes: Dict[TierSpec, int] = field(default_factory=dict)
    total_invalid_payload_bytes: Dict[TierSpec, int] = field(default_factory=dict)
    total_cross_tier_bytes: Dict[Tuple[TierSpec, TierSpec], int] = field(default_factory=dict)

    @property
    def mean_recovery(self) -> float:
        if not self.step_results:
            return 0.0
        return float(sum(r.recovery for r in self.step_results) / len(self.step_results))

    @property
    def p99_stall_us(self) -> float:
        stalls = sorted([r.total_stall_ns for r in self.step_results])
        if not stalls:
            return 0.0
        idx = int(math.ceil(0.99 * len(stalls))) - 1
        return stalls[max(0, min(idx, len(stalls) - 1))] / 1000.0

    @property
    def p999_stall_us(self) -> float:
        stalls = sorted([r.total_stall_ns for r in self.step_results])
        if not stalls:
            return 0.0
        idx = int(math.ceil(0.999 * len(stalls))) - 1
        return stalls[max(0, min(idx, len(stalls) - 1))] / 1000.0

    @property
    def mean_rho_per_tier(self) -> Dict[TierSpec, float]:
        result = {}
        for tier in TierSpec:
            if tier == TierSpec.HBM:
                continue
            rhos = [r.per_tier[tier].queue_utilization_rho
                    for r in self.step_results if tier in r.per_tier]
            result[tier] = float(np.mean(rhos)) if rhos else 0.0
        return result

    @property
    def mean_byte_at_risk(self) -> float:
        if not self.step_results:
            return 0.0
        return float(sum(r.total_bytes_at_risk for r in self.step_results) / len(self.step_results))

    @property
    def total_false_positive_bytes(self) -> int:
        return sum(
            self.total_invalid_payload_bytes.get(t, 0)
            for t in TierSpec if t != TierSpec.HBM
        )

    def add_step(self, stats: MultiTierStepStats):
        self.step_results.append(stats)
        # Accumulate
        for tier, ts in stats.per_tier.items():
            self.total_evidence_bytes[tier] = (
                self.total_evidence_bytes.get(tier, 0) + ts.evidence_bytes_fetched
            )
            self.total_payload_bytes[tier] = (
                self.total_payload_bytes.get(tier, 0) + ts.payload_bytes_fetched
            )
            self.total_invalid_payload_bytes[tier] = (
                self.total_invalid_payload_bytes.get(tier, 0) + ts.invalid_payload_bytes
            )
        for edge, b in stats.cross_tier_payload_bytes.items():
            self.total_cross_tier_bytes[edge] = (
                self.total_cross_tier_bytes.get(edge, 0) + b
            )

    def to_dict(self) -> dict:
        return {
            "num_steps": self.num_steps,
            "mean_recovery": round(self.mean_recovery, 4),
            "p99_stall_us": round(self.p99_stall_us, 2),
            "p999_stall_us": round(self.p999_stall_us, 2),
            "mean_rho_per_tier": {
                t.short_label: round(r, 4) for t, r in self.mean_rho_per_tier.items()
            },
            "mean_byte_at_risk": round(self.mean_byte_at_risk, 0),
            "total_evidence_bytes": {
                t.short_label: v for t, v in self.total_evidence_bytes.items()
            },
            "total_payload_bytes": {
                t.short_label: v for t, v in self.total_payload_bytes.items()
            },
            "total_false_positive_bytes": self.total_false_positive_bytes,
            "cross_tier_bytes": {
                f"{f.short_label}->{t.short_label}": v
                for (f, t), v in self.total_cross_tier_bytes.items()
            },
        }


# Need numpy for aggregation
import numpy as np
