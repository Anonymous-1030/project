"""KCMC (KV Cache Memory Controller) Microarchitecture — CXL Type-3 Device.

Core HPCA hardware contribution: a dedicated memory controller that manages
KV cache promotion between GPU HBM and CXL-attached DRAM.

Microarchitecture overview:
  ┌─────────────────────────────────────────────────────────────┐
  │  KCMC — CXL Type-3 Device Controller                       │
  │                                                             │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
  │  │ Promotion     │  │ Prefetch     │  │ Near-Memory  │      │
  │  │ Admission     │  │ Engine       │  │ Decompressor │      │
  │  │ Unit (PAU)    │  │ (PFE)        │  │ (NMD)        │      │
  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
  │         │                  │                  │              │
  │  ┌──────┴──────────────────┴──────────────────┴───────┐     │
  │  │           Metadata SRAM (Chunk Table)               │     │
  │  │  [chunk_id | score | TTL | tier | pos | bias_state] │     │
  │  └─────────────────────────────────────────────────────┘     │
  │                          │                                   │
  │  ┌───────────────────────┴─────────────────────────────┐     │
  │  │        CXL.mem Coherence Controller                  │     │
  │  │  - HDM-DB (Device-managed Bias)                      │     │
  │  │  - Back-invalidation for evicted chunks              │     │
  │  │  - Snoop filter (1024-entry tag array)               │     │
  │  └─────────────────────────────────────────────────────┘     │
  └─────────────────────────────────────────────────────────────┘

References:
  - CXL 3.0 Specification, Chapters 2-3 (CXL.io, CXL.mem)
  - Samsung CXL Memory Expander, ISSCC 2022
  - Pond: CXL-Based Memory Pooling, ASPLOS 2023
  - CACTI 7.0 for SRAM area/power estimates
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ======================================================================
# Enums and Configuration
# ======================================================================

class BiasState(str, Enum):
    """CXL.mem bias ownership state."""
    HOST_BIAS = "host_bias"
    DEVICE_BIAS = "device_bias"


class CoherenceAction(str, Enum):
    """CXL.mem coherence actions."""
    BIAS_FLIP_H2D = "host_to_device"
    BIAS_FLIP_D2H = "device_to_host"
    BACK_INVALIDATE = "back_invalidate"
    SNOOP_HIT = "snoop_hit"
    SNOOP_MISS = "snoop_miss"


@dataclass
class KCMCConfig:
    """KCMC hardware configuration."""
    # Clock
    frequency_ghz: float = 1.2

    # PAU (Promotion Admission Unit)
    pau_score_lut_entries: int = 256
    pau_feature_bits: int = 10
    pau_latency_cycles: int = 2

    # PFE (Prefetch Engine)
    pfe_queue_depth: int = 4
    pfe_descriptor_bytes: int = 64
    pfe_max_prefetch_depth: int = 4
    pfe_latency_cycles: int = 2

    # NMD (Near-Memory Decompressor)
    nmd_lanes: int = 64
    nmd_output_bits: int = 16
    nmd_compressed_bits: int = 4
    nmd_staging_buffer_kb: int = 32
    nmd_latency_cycles: int = 5

    # Metadata SRAM
    metadata_entries: int = 512
    metadata_bytes_per_entry: int = 16
    metadata_ports: int = 2  # Dual-port

    # Coherence Controller
    snoop_filter_entries: int = 1024
    snoop_filter_tag_bits: int = 32
    bias_flip_latency_ns: float = 200.0
    back_invalidation_latency_ns: float = 50.0

    # DMA Engine
    dma_queue_depth: int = 8
    dma_max_outstanding: int = 4

    @property
    def cycle_ns(self) -> float:
        return 1.0 / self.frequency_ghz


@dataclass
class ChunkTableEntry:
    """One entry in the metadata SRAM chunk table."""
    chunk_id: int = -1
    utility_score: float = 0.0
    ttl: int = 0
    tier: str = "tail"          # "anchor", "tail", "promoted", "evicted"
    position: int = 0           # Token position in sequence
    access_count: int = 0
    bias_state: BiasState = BiasState.DEVICE_BIAS
    valid: bool = False
    last_access_cycle: int = 0
    lru_counter: int = 0


@dataclass
class PrefetchDescriptor:
    """Descriptor in the prefetch queue."""
    chunk_id: int
    address: int               # DRAM address
    size_bytes: int
    priority: float
    predicted_step: int        # Which decode step this is for
    issued_cycle: int = 0
    completed: bool = False


@dataclass
class CoherenceEvent:
    """A coherence event in the CXL.mem protocol."""
    cycle: int
    action: CoherenceAction
    chunk_id: int
    latency_ns: float
    old_bias: BiasState
    new_bias: BiasState


# ======================================================================
# Component Models
# ======================================================================

class PromotionAdmissionUnit:
    """PAU: Score-based admission control for promotion requests.

    Hardware: 256-entry LUT + threshold comparator + budget counter.
    Area: 0.012 mm² at 7nm (CACTI: 0.003 mm² SRAM + 3× peripheral).
    Power: 15 mW.
    """

    def __init__(self, config: KCMCConfig):
        self.config = config
        self.score_lut: Dict[int, float] = {}
        self.threshold: float = 0.5
        self.hbm_budget_remaining: int = 0
        self.bw_budget_remaining: int = 0
        self._total_admitted: int = 0
        self._total_rejected: int = 0

    def lookup_and_admit(
        self, chunk_id: int, features_hash: int, chunk_bytes: int,
    ) -> Tuple[bool, float]:
        """Score lookup + threshold + budget check. 2-cycle latency."""
        score = self.score_lut.get(features_hash % self.config.pau_score_lut_entries, 0.5)
        if score < self.threshold:
            self._total_rejected += 1
            return False, score
        if chunk_bytes > self.hbm_budget_remaining or chunk_bytes > self.bw_budget_remaining:
            self._total_rejected += 1
            return False, score
        self.hbm_budget_remaining -= chunk_bytes
        self.bw_budget_remaining -= chunk_bytes
        self._total_admitted += 1
        return True, score

    def update_threshold(self, target_admission_rate: float) -> None:
        """Adaptive threshold: adjust to hit target admission rate."""
        total = self._total_admitted + self._total_rejected
        if total < 10:
            return
        actual_rate = self._total_admitted / total
        if actual_rate > target_admission_rate * 1.1:
            self.threshold = min(1.0, self.threshold + 0.02)
        elif actual_rate < target_admission_rate * 0.9:
            self.threshold = max(0.0, self.threshold - 0.02)


class PrefetchEngine:
    """PFE: Prefetch engine with stride prediction and CXL bias control.

    Hardware: 4-deep descriptor FIFO + stride predictor + EMA predictor.
    Area: 0.008 mm² at 7nm.
    Power: 12 mW.
    """

    def __init__(self, config: KCMCConfig):
        self.config = config
        self.queue: List[PrefetchDescriptor] = []
        self._stride_history: List[int] = []
        self._ema_alpha: float = 0.3
        self._ema_prediction: Dict[int, float] = {}
        self._hits: int = 0
        self._misses: int = 0

    def predict_next_chunks(
        self, current_promoted: List[int], _step: int = 0,
    ) -> List[int]:
        """Predict which chunks will be needed next.

        Uses stride detection + EMA of access patterns.
        """
        predictions = []

        # Stride prediction: if last N promotions have a stride, continue
        if len(self._stride_history) >= 2:
            stride = self._stride_history[-1] - self._stride_history[-2]
            if stride != 0:
                next_chunk = self._stride_history[-1] + stride
                predictions.append(next_chunk)

        # EMA prediction: chunks with high EMA score
        sorted_ema = sorted(
            self._ema_prediction.items(), key=lambda x: x[1], reverse=True
        )
        for cid, score in sorted_ema[:self.config.pfe_max_prefetch_depth]:
            if cid not in predictions and score > 0.3:
                predictions.append(cid)

        # Update stride history
        self._stride_history.extend(current_promoted)
        if len(self._stride_history) > 16:
            self._stride_history = self._stride_history[-16:]

        return predictions[:self.config.pfe_max_prefetch_depth]

    def enqueue(self, desc: PrefetchDescriptor) -> bool:
        """Add descriptor to prefetch queue. Returns False if full."""
        if len(self.queue) >= self.config.pfe_queue_depth:
            return False
        self.queue.append(desc)
        return True

    def check_hit(self, chunk_id: int) -> bool:
        """Check if chunk is already in prefetch queue (hit)."""
        for desc in self.queue:
            if desc.chunk_id == chunk_id and desc.completed:
                self._hits += 1
                self.queue.remove(desc)
                return True
        self._misses += 1
        return False

    def update_ema(self, chunk_id: int, accessed: bool) -> None:
        """Update EMA prediction for a chunk."""
        old = self._ema_prediction.get(chunk_id, 0.0)
        target = 1.0 if accessed else 0.0
        self._ema_prediction[chunk_id] = (
            self._ema_alpha * target + (1 - self._ema_alpha) * old
        )

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


class CoherenceController:
    """CXL.mem coherence controller with HDM-DB (Device-managed Bias).

    For CXL Type-3 devices, the device manages coherence:
    - Promoted chunks: device-bias (GPU can access without host snoop)
    - Evicted chunks: bias flip back to host-bias
    - Back-invalidation: notify host when device reclaims a chunk

    Hardware: 1024-entry snoop filter tag array.
    Area: 0.010 mm² at 7nm (CACTI: 1024 × 4B tag = 4KB SRAM).
    Power: 10 mW.
    """

    def __init__(self, config: KCMCConfig):
        self.config = config
        self.snoop_filter: Dict[int, BiasState] = {}
        self.events: List[CoherenceEvent] = []
        self._cycle: int = 0

    def promote_chunk(self, chunk_id: int, cycle: int) -> float:
        """Handle coherence for chunk promotion (host→device bias flip).

        Returns additional latency in ns from bias flip.
        """
        self._cycle = cycle
        old_bias = self.snoop_filter.get(chunk_id, BiasState.HOST_BIAS)

        if old_bias == BiasState.DEVICE_BIAS:
            # Already device-bias, no flip needed
            return 0.0

        # Bias flip: host → device
        self.snoop_filter[chunk_id] = BiasState.DEVICE_BIAS
        latency = self.config.bias_flip_latency_ns

        self.events.append(CoherenceEvent(
            cycle=cycle,
            action=CoherenceAction.BIAS_FLIP_H2D,
            chunk_id=chunk_id,
            latency_ns=latency,
            old_bias=old_bias,
            new_bias=BiasState.DEVICE_BIAS,
        ))
        return latency

    def evict_chunk(self, chunk_id: int, cycle: int) -> float:
        """Handle coherence for chunk eviction (device→host bias flip).

        Includes back-invalidation to notify host.
        Returns additional latency in ns.
        """
        self._cycle = cycle
        old_bias = self.snoop_filter.get(chunk_id, BiasState.DEVICE_BIAS)

        if old_bias == BiasState.HOST_BIAS:
            return 0.0

        # Back-invalidation + bias flip
        bi_latency = self.config.back_invalidation_latency_ns
        flip_latency = self.config.bias_flip_latency_ns
        total = bi_latency + flip_latency

        self.snoop_filter[chunk_id] = BiasState.HOST_BIAS

        self.events.append(CoherenceEvent(
            cycle=cycle,
            action=CoherenceAction.BACK_INVALIDATE,
            chunk_id=chunk_id,
            latency_ns=total,
            old_bias=old_bias,
            new_bias=BiasState.HOST_BIAS,
        ))
        return total

    def snoop_lookup(self, chunk_id: int) -> Tuple[bool, BiasState]:
        """Check snoop filter for chunk bias state."""
        if chunk_id in self.snoop_filter:
            return True, self.snoop_filter[chunk_id]
        return False, BiasState.HOST_BIAS


# ======================================================================
# Metadata SRAM
# ======================================================================

class MetadataSRAM:
    """Chunk table in SRAM: 512 entries × 16B = 8KB.

    Dual-port: one read + one write per cycle.
    LRU replacement for overflow.
    Area: 0.010 mm² at 7nm (CACTI: 8KB dual-port).
    Power: 10 mW.
    """

    def __init__(self, config: KCMCConfig):
        self.config = config
        self.entries: Dict[int, ChunkTableEntry] = {}
        self._lru_clock: int = 0

    def read(self, chunk_id: int) -> Optional[ChunkTableEntry]:
        entry = self.entries.get(chunk_id)
        if entry and entry.valid:
            self._lru_clock += 1
            entry.lru_counter = self._lru_clock
            return entry
        return None

    def write(self, chunk_id: int, entry: ChunkTableEntry) -> None:
        if len(self.entries) >= self.config.metadata_entries and chunk_id not in self.entries:
            self._evict_lru()
        entry.valid = True
        self.entries[chunk_id] = entry

    def _evict_lru(self) -> None:
        if not self.entries:
            return
        lru_id = min(self.entries, key=lambda k: self.entries[k].lru_counter)
        del self.entries[lru_id]

    def decrement_ttls(self) -> List[int]:
        """Decrement all TTLs by 1, return expired chunk IDs."""
        expired = []
        for cid, entry in list(self.entries.items()):
            if entry.ttl > 0:
                entry.ttl -= 1
                if entry.ttl == 0 and entry.tier == "promoted":
                    entry.tier = "tail"
                    expired.append(cid)
        return expired


# ======================================================================
# KCMC Top-Level
# ======================================================================

@dataclass
class PromotionRequest:
    """A single promotion request to the KCMC."""
    chunk_id: int
    chunk_bytes: int
    features_hash: int = 0
    priority: float = 0.0
    arrival_cycle: int = 0


@dataclass
class PromotionResponse:
    """Response from KCMC for a promotion request."""
    chunk_id: int
    admitted: bool
    score: float
    prefetch_hit: bool
    total_latency_ns: float
    # Latency breakdown
    pau_latency_ns: float = 0.0
    pfe_latency_ns: float = 0.0
    coherence_latency_ns: float = 0.0
    transfer_latency_ns: float = 0.0
    nmd_latency_ns: float = 0.0
    dma_latency_ns: float = 0.0


@dataclass
class KCMCSimulationResult:
    """Result of KCMC cycle-level simulation."""
    total_cycles: int
    total_requests: int
    admitted_requests: int
    rejected_requests: int
    prefetch_hits: int
    prefetch_misses: int
    prefetch_hit_rate: float
    coherence_events: int
    bias_flips: int
    back_invalidations: int
    ttl_expirations: int
    avg_latency_ns: float
    p50_latency_ns: float
    p95_latency_ns: float
    p99_latency_ns: float
    # Per-component area/power
    component_costs: List[Dict[str, float]] = field(default_factory=list)
    total_area_mm2: float = 0.0
    total_power_mw: float = 0.0
    area_overhead_pct: float = 0.0
    power_overhead_pct: float = 0.0
    # Per-request traces
    responses: List[PromotionResponse] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "total_cycles": self.total_cycles,
            "total_requests": self.total_requests,
            "admitted": self.admitted_requests,
            "rejected": self.rejected_requests,
            "prefetch_hit_rate": self.prefetch_hit_rate,
            "coherence_events": self.coherence_events,
            "bias_flips": self.bias_flips,
            "back_invalidations": self.back_invalidations,
            "ttl_expirations": self.ttl_expirations,
            "avg_latency_ns": self.avg_latency_ns,
            "p50_latency_ns": self.p50_latency_ns,
            "p95_latency_ns": self.p95_latency_ns,
            "p99_latency_ns": self.p99_latency_ns,
            "total_area_mm2": self.total_area_mm2,
            "total_power_mw": self.total_power_mw,
            "area_overhead_pct": self.area_overhead_pct,
            "power_overhead_pct": self.power_overhead_pct,
        }


class KCMCMicroarchitecture:
    """Top-level KCMC controller integrating all components.

    Processes promotion requests through the full pipeline:
      PAU → PFE check → CXL fetch + NMD → DMA write → Metadata update
    """

    def __init__(self, config: Optional[KCMCConfig] = None):
        self.config = config or KCMCConfig()
        self.pau = PromotionAdmissionUnit(self.config)
        self.pfe = PrefetchEngine(self.config)
        self.coherence = CoherenceController(self.config)
        self.metadata = MetadataSRAM(self.config)
        self._cycle = 0

    def process_request(
        self, req: PromotionRequest, cxl_transfer_ns: float = 1000.0,
    ) -> PromotionResponse:
        """Process a single promotion request through the KCMC pipeline.

        Args:
            req: Promotion request.
            cxl_transfer_ns: CXL transfer time for this chunk (from BW model).

        Returns:
            PromotionResponse with latency breakdown.
        """
        cycle_ns = self.config.cycle_ns

        # Stage 1: PAU — score lookup + admission
        pau_ns = self.config.pau_latency_cycles * cycle_ns
        admitted, score = self.pau.lookup_and_admit(
            req.chunk_id, req.features_hash, req.chunk_bytes
        )

        if not admitted:
            return PromotionResponse(
                chunk_id=req.chunk_id, admitted=False, score=score,
                prefetch_hit=False, total_latency_ns=pau_ns,
                pau_latency_ns=pau_ns,
            )

        # Stage 2: PFE — check prefetch queue
        pfe_ns = self.config.pfe_latency_cycles * cycle_ns
        prefetch_hit = self.pfe.check_hit(req.chunk_id)

        if prefetch_hit:
            # Data already in staging buffer — skip CXL fetch
            transfer_ns = 0.0
            nmd_ns = 0.0  # Already decompressed
        else:
            # Stage 3: CXL fetch + NMD decompression (pipelined)
            coherence_ns = self.coherence.promote_chunk(req.chunk_id, self._cycle)
            transfer_ns = cxl_transfer_ns + coherence_ns
            nmd_ns = self.config.nmd_latency_cycles * cycle_ns

        # Stage 4: DMA write to HBM
        dma_ns = 10.0  # ~10ns for DMA descriptor setup + initiation

        # Stage 5: Metadata update
        entry = ChunkTableEntry(
            chunk_id=req.chunk_id,
            utility_score=score,
            ttl=4,  # Default TTL
            tier="promoted",
            bias_state=BiasState.DEVICE_BIAS,
            valid=True,
            last_access_cycle=self._cycle,
        )
        self.metadata.write(req.chunk_id, entry)

        total_ns = pau_ns + pfe_ns + transfer_ns + max(nmd_ns, 0) + dma_ns
        self._cycle += int(total_ns / cycle_ns) + 1

        return PromotionResponse(
            chunk_id=req.chunk_id, admitted=True, score=score,
            prefetch_hit=prefetch_hit, total_latency_ns=total_ns,
            pau_latency_ns=pau_ns, pfe_latency_ns=pfe_ns,
            coherence_latency_ns=transfer_ns - cxl_transfer_ns if not prefetch_hit else 0.0,
            transfer_latency_ns=cxl_transfer_ns if not prefetch_hit else 0.0,
            nmd_latency_ns=nmd_ns if not prefetch_hit else 0.0,
            dma_latency_ns=dma_ns,
        )

    def simulate_promotion_sequence(
        self,
        requests: List[PromotionRequest],
        cxl_transfer_ns: float = 1000.0,
        hbm_budget_bytes: int = 100 * 1024 * 1024,
        bw_budget_bytes: int = 1024 * 1024,
    ) -> KCMCSimulationResult:
        """Simulate a sequence of promotion requests.

        Args:
            requests: Ordered list of promotion requests.
            cxl_transfer_ns: Per-chunk CXL transfer time.
            hbm_budget_bytes: Total HBM budget for promoted chunks.
            bw_budget_bytes: Per-step bandwidth budget.

        Returns:
            KCMCSimulationResult with detailed metrics.
        """
        self._cycle = 0
        self.pau.hbm_budget_remaining = hbm_budget_bytes
        self.pau.bw_budget_remaining = bw_budget_bytes

        responses: List[PromotionResponse] = []
        latencies: List[float] = []
        ttl_expirations = 0

        for req in requests:
            # Process request
            resp = self.process_request(req, cxl_transfer_ns)
            responses.append(resp)
            latencies.append(resp.total_latency_ns)

            # Periodic TTL decrement (every 100 cycles)
            if self._cycle % 100 == 0:
                expired = self.metadata.decrement_ttls()
                for cid in expired:
                    self.coherence.evict_chunk(cid, self._cycle)
                ttl_expirations += len(expired)

            # Issue prefetch predictions for admitted chunks
            if resp.admitted:
                promoted_ids = [r.chunk_id for r in responses if r.admitted][-8:]
                predictions = self.pfe.predict_next_chunks(promoted_ids)
                for pred_cid in predictions:
                    desc = PrefetchDescriptor(
                        chunk_id=pred_cid, address=pred_cid * 65536,
                        size_bytes=req.chunk_bytes, priority=0.5,
                        predicted_step=self._cycle,
                        issued_cycle=self._cycle, completed=True,
                    )
                    self.pfe.enqueue(desc)

        # Compute statistics
        admitted = sum(1 for r in responses if r.admitted)
        sorted_lat = sorted(latencies) if latencies else [0.0]

        def percentile(data: List[float], p: float) -> float:
            idx = min(int(len(data) * p / 100.0), len(data) - 1)
            return data[idx]

        # Area/power model
        costs = self._compute_area_power()

        coherence_events = len(self.coherence.events)
        bias_flips = sum(
            1 for e in self.coherence.events
            if e.action in (CoherenceAction.BIAS_FLIP_H2D, CoherenceAction.BIAS_FLIP_D2H)
        )
        back_invs = sum(
            1 for e in self.coherence.events
            if e.action == CoherenceAction.BACK_INVALIDATE
        )

        return KCMCSimulationResult(
            total_cycles=self._cycle,
            total_requests=len(requests),
            admitted_requests=admitted,
            rejected_requests=len(requests) - admitted,
            prefetch_hits=self.pfe._hits,
            prefetch_misses=self.pfe._misses,
            prefetch_hit_rate=self.pfe.hit_rate,
            coherence_events=coherence_events,
            bias_flips=bias_flips,
            back_invalidations=back_invs,
            ttl_expirations=ttl_expirations,
            avg_latency_ns=sum(latencies) / len(latencies) if latencies else 0.0,
            p50_latency_ns=percentile(sorted_lat, 50),
            p95_latency_ns=percentile(sorted_lat, 95),
            p99_latency_ns=percentile(sorted_lat, 99),
            component_costs=costs,
            total_area_mm2=sum(c["area_mm2"] for c in costs),
            total_power_mw=sum(c["power_mw"] for c in costs),
            area_overhead_pct=sum(c["area_mm2"] for c in costs) / 814.0 * 100,
            power_overhead_pct=sum(c["power_mw"] for c in costs) / 700_000 * 100,
            responses=responses,
        )

    def _compute_area_power(self) -> List[Dict[str, float]]:
        """Compute per-component area/power at 7nm."""
        cfg = self.config
        scale = 1.0  # 7nm baseline

        return [
            {
                "component": "PAU",
                "area_mm2": 0.012 * scale,
                "power_mw": 15.0,
                "latency_ns": cfg.pau_latency_cycles * cfg.cycle_ns,
                "notes": "256-entry score LUT + comparator + budget counter",
            },
            {
                "component": "PFE",
                "area_mm2": 0.008 * scale,
                "power_mw": 12.0,
                "latency_ns": cfg.pfe_latency_cycles * cfg.cycle_ns,
                "notes": "4-deep prefetch queue + stride predictor + EMA",
            },
            {
                "component": "NMD",
                "area_mm2": 0.045 * scale,
                "power_mw": 45.0,
                "latency_ns": cfg.nmd_latency_cycles * cfg.cycle_ns,
                "notes": "64-lane INT4→FP16 dequantizer + 32KB staging buffer",
            },
            {
                "component": "Coherence",
                "area_mm2": 0.010 * scale,
                "power_mw": 10.0,
                "latency_ns": cfg.bias_flip_latency_ns,
                "notes": "1024-entry snoop filter + bias tracking FSM",
            },
            {
                "component": "Metadata SRAM",
                "area_mm2": 0.010 * scale,
                "power_mw": 10.0,
                "latency_ns": cfg.cycle_ns,
                "notes": f"{cfg.metadata_entries}×{cfg.metadata_bytes_per_entry}B dual-port SRAM",
            },
            {
                "component": "DMA Engine",
                "area_mm2": 0.008 * scale,
                "power_mw": 12.0,
                "latency_ns": 10.0,
                "notes": f"{cfg.dma_queue_depth}-entry DMA descriptor queue",
            },
        ]
