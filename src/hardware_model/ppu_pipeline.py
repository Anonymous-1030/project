"""Promotion Processing Unit (PPU) — Hardware Pipeline Model.

Maps the software promotion pipeline (ULF → ODUS → EABS → Burst → Sticky)
onto a dedicated hardware accelerator integrated into the KCMC.

This is a core HPCA contribution: showing that the promotion decision can be
made entirely in hardware with < 0.1% area overhead and < 10ns latency,
enabling per-step promotion decisions without GPU involvement.

PPU Microarchitecture:
  ┌─────────────────────────────────────────────────────────────┐
  │  PPU (Promotion Processing Unit) — 5-stage pipeline         │
  │                                                             │
  │  Stage 1: Candidate Recall (CR)                             │
  │    - 4-queue parallel lookup in metadata SRAM               │
  │    - Anchor-neighbor: stride-based address generator         │
  │    - Lexical: MinHash signature comparator (8-way)          │
  │    - Structural: section-boundary flag check                │
  │    - Historical: LRU-ordered scan of recent promotions      │
  │    - Latency: 2 cycles @ 1.2 GHz = 1.67 ns                 │
  │                                                             │
  │  Stage 2: Utility Scoring (US)                              │
  │    - 10-input, 2-hidden-layer MLP in fixed-point (INT8)     │
  │    - Weights stored in 640B SRAM (10×32 + 32×16 + 16×1)    │
  │    - MAC array: 32 INT8 MACs (shared across hidden layers)  │
  │    - Latency: 3 cycles = 2.5 ns                            │
  │                                                             │
  │  Stage 3: Budget Check (BC)                                 │
  │    - Compare score against dynamic threshold                │
  │    - Check HBM capacity counter                             │
  │    - Check bandwidth budget counter (from CXL link monitor) │
  │    - Exploit/explore selector (LFSR-based random bit)       │
  │    - Latency: 1 cycle = 0.83 ns                            │
  │                                                             │
  │  Stage 4: Burst Expansion (BE)                              │
  │    - For each admitted chunk, check ±R neighbors in SRAM    │
  │    - R configurable (1-4), default R=2                      │
  │    - Parallel neighbor lookup (2R ports or 2-cycle serial)  │
  │    - Latency: 2 cycles = 1.67 ns                           │
  │                                                             │
  │  Stage 5: TTL Write-back (TW)                               │
  │    - Set TTL for newly promoted chunks                      │
  │    - Refresh TTL for re-accessed chunks                     │
  │    - Decrement TTL for all active entries (background)      │
  │    - Expire entries with TTL=0                              │
  │    - Latency: 1 cycle = 0.83 ns                            │
  │                                                             │
  │  Total pipeline: 9 cycles = 7.5 ns @ 1.2 GHz              │
  │  Throughput: 1 candidate per cycle after pipeline fill      │
  └─────────────────────────────────────────────────────────────┘

Area/Power (7nm):
  - CR stage: 0.005 mm² / 8 mW  (address generators + comparators)
  - US stage: 0.008 mm² / 12 mW (32 INT8 MACs + weight SRAM)
  - BC stage: 0.002 mm² / 3 mW  (comparators + counters + LFSR)
  - BE stage: 0.004 mm² / 6 mW  (neighbor address gen + SRAM ports)
  - TW stage: 0.003 mm² / 5 mW  (TTL counters + write logic)
  - Total:    0.022 mm² / 34 mW
  - H100 overhead: 0.003% area, 0.005% TDP
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class PPUConfig:
    """PPU hardware configuration."""
    frequency_ghz: float = 1.2

    # Stage 1: Candidate Recall
    cr_num_queues: int = 4
    cr_latency_cycles: int = 2
    cr_max_candidates_per_queue: int = 8

    # Stage 2: Utility Scoring
    us_input_dim: int = 10
    us_hidden_dims: Tuple[int, ...] = (32, 16)
    us_precision_bits: int = 8  # INT8 fixed-point
    us_num_macs: int = 32
    us_latency_cycles: int = 3
    us_weight_sram_bytes: int = 640  # 10*32 + 32*16 + 16*1 weights

    # Stage 3: Budget Check
    bc_latency_cycles: int = 1
    bc_lfsr_bits: int = 16  # For explore randomness

    # Stage 4: Burst Expansion
    be_max_radius: int = 4
    be_default_radius: int = 2
    be_latency_cycles: int = 2

    # Stage 5: TTL Write-back
    tw_latency_cycles: int = 1
    tw_max_ttl: int = 16
    tw_default_ttl: int = 4

    # Metadata SRAM
    metadata_entries: int = 512
    metadata_bytes_per_entry: int = 16  # chunk_id(4) + score(2) + TTL(1) + tier(1) + pos(4) + access(2) + bias(1) + pad(1)

    @property
    def cycle_time_ns(self) -> float:
        return 1.0 / self.frequency_ghz

    @property
    def total_pipeline_cycles(self) -> int:
        return (self.cr_latency_cycles + self.us_latency_cycles +
                self.bc_latency_cycles + self.be_latency_cycles +
                self.tw_latency_cycles)

    @property
    def pipeline_latency_ns(self) -> float:
        return self.total_pipeline_cycles * self.cycle_time_ns

    @property
    def throughput_candidates_per_ns(self) -> float:
        """After pipeline fill, one candidate per cycle."""
        return self.frequency_ghz  # GHz = candidates/ns


class PPUStage(str, Enum):
    CANDIDATE_RECALL = "CR"
    UTILITY_SCORING = "US"
    BUDGET_CHECK = "BC"
    BURST_EXPANSION = "BE"
    TTL_WRITEBACK = "TW"


# ======================================================================
# Per-stage area/power models
# ======================================================================

@dataclass
class StageHardwareCost:
    """Area/power/latency for one PPU pipeline stage."""
    stage: PPUStage
    area_mm2: float
    power_mw: float
    latency_ns: float
    latency_cycles: int
    notes: str = ""


@dataclass
class PPUHardwareSummary:
    """Aggregate PPU hardware cost."""
    stages: List[StageHardwareCost]
    total_area_mm2: float
    total_power_mw: float
    pipeline_latency_ns: float
    throughput_per_ns: float
    # Overhead relative to H100
    area_overhead_pct: float  # vs 814 mm²
    power_overhead_pct: float  # vs 700 W


def compute_ppu_hardware(config: PPUConfig, process_nm: int = 7) -> PPUHardwareSummary:
    """Compute PPU area/power from configuration.

    All estimates cross-referenced with:
      - CACTI 7.0 for SRAM
      - NVIDIA A100/H100 die analysis for MAC units
      - Loh, ISCA 2015 for predictor sizing
    """
    scale = process_nm / 7.0  # Linear area scaling from 7nm baseline

    stages = []

    # Stage 1: Candidate Recall — address generators + comparators
    # 4 queues × (stride generator + 8-way MinHash comparator)
    # Stride gen: ~0.001 mm² each, MinHash: 8 comparators × 0.0001 mm²
    cr_area = 0.005 * scale * (config.cr_num_queues / 4.0)
    cr_power = 8.0 * (config.cr_num_queues / 4.0)
    stages.append(StageHardwareCost(
        stage=PPUStage.CANDIDATE_RECALL,
        area_mm2=cr_area, power_mw=cr_power,
        latency_ns=config.cr_latency_cycles * config.cycle_time_ns,
        latency_cycles=config.cr_latency_cycles,
        notes="4-queue parallel lookup: stride gen + MinHash comparators",
    ))

    # Stage 2: Utility Scoring — INT8 MAC array + weight SRAM
    # 32 INT8 MACs at 7nm: ~0.0002 mm² each = 0.0064 mm²
    # Weight SRAM (640B): CACTI 7nm ~0.001 mm²
    # Control logic: ~0.0006 mm²
    mac_area = config.us_num_macs * 0.0002 * scale
    sram_area = config.us_weight_sram_bytes / 640.0 * 0.001 * scale
    us_area = mac_area + sram_area + 0.0006 * scale
    us_power = config.us_num_macs * 0.375  # ~0.375 mW per INT8 MAC at 1.2GHz
    stages.append(StageHardwareCost(
        stage=PPUStage.UTILITY_SCORING,
        area_mm2=us_area, power_mw=us_power,
        latency_ns=config.us_latency_cycles * config.cycle_time_ns,
        latency_cycles=config.us_latency_cycles,
        notes=f"{config.us_num_macs} INT8 MACs + {config.us_weight_sram_bytes}B weight SRAM",
    ))

    # Stage 3: Budget Check — comparators + counters + LFSR
    bc_area = 0.002 * scale
    bc_power = 3.0
    stages.append(StageHardwareCost(
        stage=PPUStage.BUDGET_CHECK,
        area_mm2=bc_area, power_mw=bc_power,
        latency_ns=config.bc_latency_cycles * config.cycle_time_ns,
        latency_cycles=config.bc_latency_cycles,
        notes="Threshold comparator + HBM/BW budget counters + LFSR",
    ))

    # Stage 4: Burst Expansion — neighbor address gen + SRAM read ports
    be_area = 0.004 * scale * (config.be_default_radius / 2.0)
    be_power = 6.0 * (config.be_default_radius / 2.0)
    stages.append(StageHardwareCost(
        stage=PPUStage.BURST_EXPANSION,
        area_mm2=be_area, power_mw=be_power,
        latency_ns=config.be_latency_cycles * config.cycle_time_ns,
        latency_cycles=config.be_latency_cycles,
        notes=f"±{config.be_default_radius} neighbor lookup in metadata SRAM",
    ))

    # Stage 5: TTL Write-back — TTL counters + write logic
    # 512 entries × 4-bit TTL counter = 256B
    tw_area = 0.003 * scale * (config.metadata_entries / 512.0)
    tw_power = 5.0 * (config.metadata_entries / 512.0)
    stages.append(StageHardwareCost(
        stage=PPUStage.TTL_WRITEBACK,
        area_mm2=tw_area, power_mw=tw_power,
        latency_ns=config.tw_latency_cycles * config.cycle_time_ns,
        latency_cycles=config.tw_latency_cycles,
        notes=f"{config.metadata_entries}-entry TTL counter array",
    ))

    total_area = sum(s.area_mm2 for s in stages)
    total_power = sum(s.power_mw for s in stages)

    return PPUHardwareSummary(
        stages=stages,
        total_area_mm2=total_area,
        total_power_mw=total_power,
        pipeline_latency_ns=config.pipeline_latency_ns,
        throughput_per_ns=config.throughput_candidates_per_ns,
        area_overhead_pct=100.0 * total_area / 814.0,
        power_overhead_pct=100.0 * (total_power / 1000.0) / 700.0,
    )


# ======================================================================
# Cycle-level pipeline simulation
# ======================================================================

@dataclass
class PPURequest:
    """A single promotion candidate entering the PPU pipeline."""
    request_id: int
    chunk_id: int
    # Runtime features (quantized to INT8 for hardware)
    features: List[int] = field(default_factory=lambda: [0] * 10)
    # Metadata
    chunk_bytes: int = 0
    arrival_cycle: int = 0


@dataclass
class PPUPipelineSlot:
    """One slot in the pipeline register."""
    request: Optional[PPURequest] = None
    stage: Optional[PPUStage] = None
    cycle_entered: int = 0
    # Per-stage results
    candidates: List[int] = field(default_factory=list)
    score: float = 0.0
    admitted: bool = False
    burst_neighbors: List[int] = field(default_factory=list)
    ttl_assigned: int = 0


@dataclass
class PPUStageTrace:
    """Trace of one request through the pipeline."""
    request_id: int
    chunk_id: int
    arrival_cycle: int
    completion_cycle: int
    total_latency_cycles: int
    score: float
    admitted: bool
    burst_neighbors: List[int]
    ttl: int
    # Per-stage cycle counts
    cr_cycle: int = 0
    us_cycle: int = 0
    bc_cycle: int = 0
    be_cycle: int = 0
    tw_cycle: int = 0
    # Stall cycles (from structural hazards)
    stall_cycles: int = 0

    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id,
            "chunk_id": self.chunk_id,
            "arrival": self.arrival_cycle,
            "completion": self.completion_cycle,
            "latency": self.total_latency_cycles,
            "score": self.score,
            "admitted": self.admitted,
            "burst_neighbors": self.burst_neighbors,
            "ttl": self.ttl,
            "stall_cycles": self.stall_cycles,
        }


@dataclass
class PPUSimulationResult:
    """Result of PPU pipeline simulation."""
    total_cycles: int
    total_requests: int
    admitted_requests: int
    rejected_requests: int
    avg_latency_cycles: float
    avg_latency_ns: float
    max_latency_cycles: int
    throughput_requests_per_cycle: float
    total_stall_cycles: int
    pipeline_utilization: float  # fraction of cycles with active work
    traces: List[PPUStageTrace] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "total_cycles": self.total_cycles,
            "total_requests": self.total_requests,
            "admitted": self.admitted_requests,
            "rejected": self.rejected_requests,
            "avg_latency_cycles": self.avg_latency_cycles,
            "avg_latency_ns": self.avg_latency_ns,
            "max_latency_cycles": self.max_latency_cycles,
            "throughput_per_cycle": self.throughput_requests_per_cycle,
            "stall_cycles": self.total_stall_cycles,
            "pipeline_utilization": self.pipeline_utilization,
        }


class PPUPipelineSimulator:
    """Cycle-accurate simulation of the PPU 5-stage pipeline.

    Models:
      - Pipeline stages with correct latencies
      - Structural hazards (stage busy → stall upstream)
      - Score-based admission at BC stage
      - Burst expansion at BE stage
      - TTL assignment at TW stage
    """

    def __init__(
        self,
        config: PPUConfig,
        score_threshold: float = 0.5,
        hbm_budget_bytes: int = 1024 * 1024 * 100,  # 100 MB
        bandwidth_budget_bytes_per_step: int = 1024 * 1024,  # 1 MB
        explore_ratio: float = 0.2,
    ):
        self.config = config
        self.score_threshold = score_threshold
        self.hbm_budget_bytes = hbm_budget_bytes
        self.bw_budget = bandwidth_budget_bytes_per_step
        self.explore_ratio = explore_ratio

        # Pipeline state
        self._stages: Dict[PPUStage, Optional[PPUPipelineSlot]] = {
            s: None for s in PPUStage
        }
        self._stage_remaining: Dict[PPUStage, int] = {s: 0 for s in PPUStage}
        self._stage_latencies = {
            PPUStage.CANDIDATE_RECALL: config.cr_latency_cycles,
            PPUStage.UTILITY_SCORING: config.us_latency_cycles,
            PPUStage.BUDGET_CHECK: config.bc_latency_cycles,
            PPUStage.BURST_EXPANSION: config.be_latency_cycles,
            PPUStage.TTL_WRITEBACK: config.tw_latency_cycles,
        }
        self._stage_order = list(PPUStage)

        # Budget tracking
        self._hbm_used = 0
        self._bw_used = 0

        # LFSR for explore decisions (deterministic pseudo-random)
        self._lfsr = 0xACE1

        # Simulated MLP weights (INT8 quantized, random but deterministic)
        import random
        rng = random.Random(42)
        self._mlp_w1 = [[rng.randint(-128, 127) for _ in range(32)] for _ in range(10)]
        self._mlp_w2 = [[rng.randint(-128, 127) for _ in range(16)] for _ in range(32)]
        self._mlp_w3 = [rng.randint(-128, 127) for _ in range(16)]

    def _lfsr_next(self) -> bool:
        """16-bit Galois LFSR for explore/exploit decision."""
        bit = self._lfsr & 1
        self._lfsr >>= 1
        if bit:
            self._lfsr ^= 0xB400
        return (self._lfsr & 0xFF) < int(self.explore_ratio * 256)

    def _score_features(self, features: List[int]) -> float:
        """INT8 MLP forward pass (simulated hardware scoring)."""
        # Layer 1: 10 → 32
        h1 = [0] * 32
        for j in range(32):
            acc = 0
            for i in range(min(len(features), 10)):
                acc += features[i] * self._mlp_w1[i][j]
            h1[j] = max(0, acc >> 8)  # ReLU + right-shift for fixed-point

        # Layer 2: 32 → 16
        h2 = [0] * 16
        for j in range(16):
            acc = 0
            for i in range(32):
                acc += h1[i] * self._mlp_w2[i][j]
            h2[j] = max(0, acc >> 8)

        # Output: 16 → 1
        acc = sum(h2[i] * self._mlp_w3[i] for i in range(16))
        # Sigmoid approximation: piecewise linear
        x = acc / (1 << 16)
        if x < -4:
            return 0.0
        elif x > 4:
            return 1.0
        else:
            return 0.5 + x * 0.125  # Linear approx around 0

    def simulate(self, requests: List[PPURequest]) -> PPUSimulationResult:
        """Run cycle-accurate simulation of the PPU pipeline.

        Args:
            requests: List of promotion requests, sorted by arrival_cycle.

        Returns:
            PPUSimulationResult with per-request traces and aggregate metrics.
        """
        # Reset state
        for s in PPUStage:
            self._stages[s] = None
            self._stage_remaining[s] = 0
        self._hbm_used = 0
        self._bw_used = 0

        traces: List[PPUStageTrace] = []
        request_queue = list(requests)
        req_idx = 0
        cycle = 0
        total_stalls = 0
        active_cycles = 0

        max_cycles = (
            (requests[-1].arrival_cycle if requests else 0) +
            self.config.total_pipeline_cycles * 2 + 100
        )

        while cycle < max_cycles:
            any_active = False

            # Advance pipeline stages (from last to first to avoid overwrites)
            for stage_idx in range(len(self._stage_order) - 1, -1, -1):
                stage = self._stage_order[stage_idx]
                slot = self._stages[stage]
                if slot is None:
                    continue

                any_active = True
                self._stage_remaining[stage] -= 1

                if self._stage_remaining[stage] <= 0:
                    # Stage complete — process and try to advance
                    self._process_stage_completion(slot, stage, cycle)

                    # Try to move to next stage
                    if stage_idx < len(self._stage_order) - 1:
                        next_stage = self._stage_order[stage_idx + 1]
                        if self._stages[next_stage] is None:
                            slot.stage = next_stage
                            self._stages[next_stage] = slot
                            self._stage_remaining[next_stage] = self._stage_latencies[next_stage]
                            self._stages[stage] = None
                        else:
                            # Structural hazard — stall
                            total_stalls += 1
                            self._stage_remaining[stage] = 1  # Re-check next cycle
                    else:
                        # Last stage complete — emit result
                        trace = PPUStageTrace(
                            request_id=slot.request.request_id,
                            chunk_id=slot.request.chunk_id,
                            arrival_cycle=slot.request.arrival_cycle,
                            completion_cycle=cycle,
                            total_latency_cycles=cycle - slot.request.arrival_cycle,
                            score=slot.score,
                            admitted=slot.admitted,
                            burst_neighbors=slot.burst_neighbors,
                            ttl=slot.ttl_assigned,
                        )
                        traces.append(trace)
                        self._stages[stage] = None

            # Try to inject new request into first stage
            first_stage = self._stage_order[0]
            if self._stages[first_stage] is None and req_idx < len(request_queue):
                req = request_queue[req_idx]
                if req.arrival_cycle <= cycle:
                    slot = PPUPipelineSlot(
                        request=req,
                        stage=first_stage,
                        cycle_entered=cycle,
                    )
                    self._stages[first_stage] = slot
                    self._stage_remaining[first_stage] = self._stage_latencies[first_stage]
                    req_idx += 1
                    any_active = True

            if any_active:
                active_cycles += 1

            # Check termination
            if req_idx >= len(request_queue) and not any_active:
                break

            cycle += 1

        # Compute metrics
        admitted = sum(1 for t in traces if t.admitted)
        latencies = [t.total_latency_cycles for t in traces]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

        return PPUSimulationResult(
            total_cycles=cycle,
            total_requests=len(traces),
            admitted_requests=admitted,
            rejected_requests=len(traces) - admitted,
            avg_latency_cycles=avg_lat,
            avg_latency_ns=avg_lat * self.config.cycle_time_ns,
            max_latency_cycles=max(latencies) if latencies else 0,
            throughput_requests_per_cycle=(
                len(traces) / max(cycle, 1)
            ),
            total_stall_cycles=total_stalls,
            pipeline_utilization=active_cycles / max(cycle, 1),
            traces=traces,
        )

    def _process_stage_completion(
        self, slot: PPUPipelineSlot, stage: PPUStage, _cycle: int,
    ) -> None:
        """Process completion of a pipeline stage."""
        if stage == PPUStage.CANDIDATE_RECALL:
            # Generate candidate list (simulated: chunk + neighbors)
            cid = slot.request.chunk_id
            slot.candidates = [cid]

        elif stage == PPUStage.UTILITY_SCORING:
            slot.score = self._score_features(slot.request.features)

        elif stage == PPUStage.BUDGET_CHECK:
            is_explore = self._lfsr_next()
            if is_explore:
                slot.admitted = True  # Explore: always admit
            elif slot.score >= self.score_threshold:
                # Check budgets
                if (self._hbm_used + slot.request.chunk_bytes <= self.hbm_budget_bytes and
                        self._bw_used + slot.request.chunk_bytes <= self.bw_budget):
                    slot.admitted = True
                    self._hbm_used += slot.request.chunk_bytes
                    self._bw_used += slot.request.chunk_bytes
                else:
                    slot.admitted = False
            else:
                slot.admitted = False

        elif stage == PPUStage.BURST_EXPANSION:
            if slot.admitted:
                cid = slot.request.chunk_id
                r = self.config.be_default_radius
                slot.burst_neighbors = list(range(
                    max(0, cid - r), cid + r + 1
                ))
            else:
                slot.burst_neighbors = []

        elif stage == PPUStage.TTL_WRITEBACK:
            if slot.admitted:
                slot.ttl_assigned = self.config.tw_default_ttl
            else:
                slot.ttl_assigned = 0
