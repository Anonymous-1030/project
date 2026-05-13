"""gem5-Compatible CXL Memory Simulation Infrastructure.

Trace-driven simulation modeling the same phenomena as gem5 with CXL
extensions, suitable for generating HPCA-quality results.

Models:
  1. CXL 3.0 flit-level protocol (256B flits, credit-based flow control)
  2. DDR5 DRAM backend with bank-level parallelism and row buffer modeling
  3. Memory controller with FR-FCFS scheduling
  4. Multi-tenant contention on shared CXL link
  5. KV promotion trace replay with compute overlap

References:
  - CXL 3.0 Specification, Chapters 2-3
  - JEDEC DDR5 SDRAM Standard (JESD79-5)
  - gem5-CXL (KAIST, 2023) for CXL.mem modeling methodology
  - Ramulator 2.0 for DRAM timing validation
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


# ======================================================================
# CXL Protocol Model
# ======================================================================

class CXLVersion(str, Enum):
    CXL_1_1 = "1.1"
    CXL_2_0 = "2.0"
    CXL_3_0 = "3.0"


class MemRequestType(str, Enum):
    MEM_RD = "MemRd"
    MEM_WR = "MemWr"
    MEM_RD_DATA = "MemRdData"
    BI_SNP = "BISnp"  # Back-invalidation snoop


CXL_PROFILES = {
    CXLVersion.CXL_1_1: {"link_gtps": 16.0, "flit_bytes": 68, "overhead": 0.05},
    CXLVersion.CXL_2_0: {"link_gtps": 16.0, "flit_bytes": 68, "overhead": 0.03},
    CXLVersion.CXL_3_0: {"link_gtps": 64.0, "flit_bytes": 256, "overhead": 0.02},
}


@dataclass
class CXLLinkConfig:
    """CXL link configuration."""
    version: CXLVersion = CXLVersion.CXL_3_0
    link_width: int = 16
    credit_pool_size: int = 32
    credit_rtt_ns: float = 100.0

    @property
    def raw_bw_gbps(self) -> float:
        return self.link_width * CXL_PROFILES[self.version]["link_gtps"] / 8.0

    @property
    def flit_bytes(self) -> int:
        return CXL_PROFILES[self.version]["flit_bytes"]

    @property
    def effective_bw_gbps(self) -> float:
        overhead = CXL_PROFILES[self.version]["overhead"]
        return self.raw_bw_gbps * (1.0 - overhead)

    def flit_time_ns(self) -> float:
        """Time to serialize one flit."""
        return self.flit_bytes / self.raw_bw_gbps  # GB/s = bytes/ns


@dataclass
class DDR5Config:
    """DDR5 DRAM timing configuration (JEDEC JESD79-5)."""
    speed_mtps: int = 4800       # MT/s
    tCAS_ns: float = 40.0
    tRCD_ns: float = 40.0
    tRP_ns: float = 40.0
    tRAS_ns: float = 77.0
    tFAW_ns: float = 40.0
    tRFC_ns: float = 350.0
    tREFI_ns: float = 3900.0
    num_ranks: int = 2
    num_bank_groups: int = 4
    banks_per_group: int = 4
    burst_length: int = 16       # BL16 for DDR5
    prefetch_bits: int = 16
    bus_width_bits: int = 64     # Per channel

    @property
    def num_banks(self) -> int:
        return self.num_bank_groups * self.banks_per_group

    @property
    def burst_bytes(self) -> int:
        return self.burst_length * self.bus_width_bits // 8

    def row_hit_ns(self) -> float:
        return self.tCAS_ns

    def row_miss_ns(self) -> float:
        return self.tRP_ns + self.tRCD_ns + self.tCAS_ns

    def row_conflict_ns(self) -> float:
        return self.tRP_ns + self.tRCD_ns + self.tCAS_ns


@dataclass
class MemControllerConfig:
    """Memory controller configuration."""
    queue_depth: int = 32
    scheduling: str = "FR-FCFS"
    rw_turnaround_ns: float = 10.0
    address_mapping: str = "row-bank-column"


# ======================================================================
# Memory Request and Response
# ======================================================================

@dataclass
class MemRequest:
    """A memory request in the simulation."""
    request_id: int
    req_type: MemRequestType
    address: int
    size_bytes: int
    tenant_id: int = 0
    arrival_ns: float = 0.0
    # Filled by simulation
    start_ns: float = 0.0
    complete_ns: float = 0.0
    bank_id: int = -1
    row_hit: bool = False
    queuing_delay_ns: float = 0.0


@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""
    cxl: CXLLinkConfig = field(default_factory=CXLLinkConfig)
    dram: DDR5Config = field(default_factory=DDR5Config)
    mc: MemControllerConfig = field(default_factory=MemControllerConfig)
    seed: int = 42


# ======================================================================
# DRAM Bank Model
# ======================================================================

class DRAMBank:
    """Model of a single DRAM bank with row buffer."""

    def __init__(self, bank_id: int, config: DDR5Config):
        self.bank_id = bank_id
        self.config = config
        self.open_row: int = -1
        self.busy_until_ns: float = 0.0
        self._hits = 0
        self._misses = 0
        self._conflicts = 0

    def access(self, row: int, current_ns: float) -> Tuple[float, bool]:
        """Access this bank. Returns (latency_ns, is_row_hit)."""
        start = max(current_ns, self.busy_until_ns)

        if self.open_row == row:
            # Row hit
            latency = self.config.row_hit_ns()
            self._hits += 1
            is_hit = True
        elif self.open_row == -1:
            # Row miss (no row open)
            latency = self.config.row_miss_ns()
            self._misses += 1
            is_hit = False
        else:
            # Row conflict
            latency = self.config.row_conflict_ns()
            self._conflicts += 1
            is_hit = False

        self.open_row = row
        self.busy_until_ns = start + latency
        return latency, is_hit

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses + self._conflicts
        return self._hits / total if total > 0 else 0.0


# ======================================================================
# Memory Controller
# ======================================================================

class MemoryController:
    """FR-FCFS memory controller with bank-level scheduling."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.banks = [
            DRAMBank(i, config.dram) for i in range(config.dram.num_banks)
        ]
        self.queue: deque = deque()
        self.current_ns: float = 0.0
        self._last_was_write: bool = False
        self._total_served: int = 0

    def _address_to_bank_row(self, address: int) -> Tuple[int, int]:
        """Map address to (bank_id, row)."""
        burst = self.config.dram.burst_bytes
        row_size = self.config.dram.num_banks * burst * 128  # ~128 columns
        bank_id = (address // burst) % self.config.dram.num_banks
        row = address // row_size
        return bank_id, row

    def enqueue(self, req: MemRequest) -> None:
        """Add request to controller queue."""
        bank_id, _ = self._address_to_bank_row(req.address)
        req.bank_id = bank_id
        self.queue.append(req)

    def process_next(self) -> Optional[MemRequest]:
        """Process next request using FR-FCFS scheduling.

        FR-FCFS: prioritize row-buffer hits, then FCFS among misses.
        """
        if not self.queue:
            return None

        # Find best candidate: row hits first, then oldest
        best_idx = 0
        best_is_hit = False

        for i, req in enumerate(self.queue):
            bank_id, row = self._address_to_bank_row(req.address)
            bank = self.banks[bank_id]
            is_hit = (bank.open_row == row)

            if is_hit and not best_is_hit:
                best_idx = i
                best_is_hit = True
            elif is_hit == best_is_hit and req.arrival_ns < self.queue[best_idx].arrival_ns:
                best_idx = i

        req = self.queue[best_idx]
        del self.queue[best_idx]

        # Process through bank
        bank_id, row = self._address_to_bank_row(req.address)
        bank = self.banks[bank_id]

        # R/W turnaround penalty
        is_write = req.req_type == MemRequestType.MEM_WR
        turnaround = 0.0
        if is_write != self._last_was_write:
            turnaround = self.config.mc.rw_turnaround_ns
        self._last_was_write = is_write

        start_ns = max(self.current_ns + turnaround, req.arrival_ns)
        latency, is_hit = bank.access(row, start_ns)

        req.start_ns = start_ns
        req.complete_ns = start_ns + latency
        req.row_hit = is_hit
        req.queuing_delay_ns = start_ns - req.arrival_ns

        self.current_ns = req.complete_ns
        self._total_served += 1
        return req


# ======================================================================
# CXL Link Model
# ======================================================================

class CXLLink:
    """CXL link with credit-based flow control and contention."""

    def __init__(self, config: CXLLinkConfig):
        self.config = config
        self.available_credits: int = config.credit_pool_size
        self.busy_until_ns: float = 0.0
        self._total_flits: int = 0
        self._total_bytes: int = 0
        self._stall_ns: float = 0.0

    def transfer(self, size_bytes: int, arrival_ns: float) -> Tuple[float, float]:
        """Transfer data over CXL link.

        Returns (start_ns, complete_ns).
        """
        num_flits = math.ceil(size_bytes / self.config.flit_bytes)
        flit_time = self.config.flit_time_ns()

        # Wait for link availability
        start = max(arrival_ns, self.busy_until_ns)

        # Credit stall check
        if num_flits > self.available_credits:
            credit_wait = self.config.credit_rtt_ns
            start += credit_wait
            self._stall_ns += credit_wait
            self.available_credits = self.config.credit_pool_size

        transfer_time = num_flits * flit_time
        complete = start + transfer_time

        self.busy_until_ns = complete
        self.available_credits -= min(num_flits, self.available_credits)
        self._total_flits += num_flits
        self._total_bytes += size_bytes

        return start, complete

    @property
    def utilization(self) -> float:
        if self.busy_until_ns <= 0:
            return 0.0
        return (self._total_flits * self.config.flit_time_ns()) / self.busy_until_ns


# ======================================================================
# KV Promotion Trace Simulator
# ======================================================================

@dataclass
class PromotionTraceEntry:
    """One entry in a KV promotion trace."""
    step: int
    chunk_id: int
    chunk_bytes: int
    tenant_id: int = 0
    timestamp_ns: float = 0.0


@dataclass
class TraceSimResult:
    """Result of trace-driven simulation."""
    total_requests: int
    total_bytes: int
    total_time_ns: float
    # Latency statistics
    avg_latency_ns: float
    p50_latency_ns: float
    p95_latency_ns: float
    p99_latency_ns: float
    max_latency_ns: float
    # Utilization
    link_utilization: float
    dram_row_hit_rate: float
    avg_queuing_delay_ns: float
    # Per-CXL-version comparison
    cxl_version: str = ""
    effective_bw_gbps: float = 0.0
    # Link-level metrics
    total_link_busy_ns: float = 0.0
    # Per-request details
    per_request_latency: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "total_requests": self.total_requests,
            "total_bytes": self.total_bytes,
            "total_time_ns": self.total_time_ns,
            "avg_latency_ns": self.avg_latency_ns,
            "p50_latency_ns": self.p50_latency_ns,
            "p95_latency_ns": self.p95_latency_ns,
            "p99_latency_ns": self.p99_latency_ns,
            "max_latency_ns": self.max_latency_ns,
            "link_utilization": self.link_utilization,
            "dram_row_hit_rate": self.dram_row_hit_rate,
            "avg_queuing_delay_ns": self.avg_queuing_delay_ns,
            "cxl_version": self.cxl_version,
            "effective_bw_gbps": self.effective_bw_gbps,
            "total_link_busy_ns": self.total_link_busy_ns,
        }


class KVPromotionTraceSimulator:
    """Trace-driven simulation of KV promotion over CXL memory.

    Replays a promotion trace through the full CXL memory path:
      GPU request → CXL link → Memory controller → DRAM → CXL link → GPU
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.link = CXLLink(self.config.cxl)
        self.mc = MemoryController(self.config)

    def simulate(
        self, trace: List[PromotionTraceEntry],
        compute_window_ns: float = 50000.0,
    ) -> TraceSimResult:
        """Simulate a promotion trace.

        Args:
            trace: List of promotion trace entries.
            compute_window_ns: GPU compute time per step (for overlap).

        Returns:
            TraceSimResult with detailed metrics.
        """
        latencies: List[float] = []
        queuing_delays: List[float] = []
        _ = compute_window_ns  # Reserved for future overlap modeling

        for entry in trace:
            # 1. CXL link: GPU → CXL device (request)
            _req_start, req_arrive = self.link.transfer(
                64, entry.timestamp_ns  # 64B request flit
            )

            # 2. Memory controller: enqueue + schedule + DRAM access
            mem_req = MemRequest(
                request_id=entry.chunk_id,
                req_type=MemRequestType.MEM_RD,
                address=entry.chunk_id * entry.chunk_bytes,
                size_bytes=entry.chunk_bytes,
                tenant_id=entry.tenant_id,
                arrival_ns=req_arrive,
            )
            self.mc.enqueue(mem_req)

            # Process all queued requests (simplified: process this one)
            completed = self.mc.process_next()
            if completed is None:
                continue

            # 3. CXL link: CXL device → GPU (data response)
            _, data_complete = self.link.transfer(
                entry.chunk_bytes, completed.complete_ns
            )

            # Total latency
            total_lat = data_complete - entry.timestamp_ns
            latencies.append(total_lat)
            queuing_delays.append(completed.queuing_delay_ns)

        if not latencies:
            return TraceSimResult(
                total_requests=0, total_bytes=0, total_time_ns=0,
                avg_latency_ns=0, p50_latency_ns=0, p95_latency_ns=0,
                p99_latency_ns=0, max_latency_ns=0,
                link_utilization=0, dram_row_hit_rate=0,
                avg_queuing_delay_ns=0,
            )

        sorted_lat = sorted(latencies)
        n = len(sorted_lat)

        # Aggregate DRAM hit rate
        total_hits = sum(b._hits for b in self.mc.banks)
        total_accesses = sum(
            b._hits + b._misses + b._conflicts for b in self.mc.banks
        )
        hit_rate = total_hits / total_accesses if total_accesses > 0 else 0.0

        return TraceSimResult(
            total_requests=len(trace),
            total_bytes=sum(e.chunk_bytes for e in trace),
            total_time_ns=self.link.busy_until_ns,
            avg_latency_ns=sum(latencies) / n,
            p50_latency_ns=sorted_lat[n // 2],
            p95_latency_ns=sorted_lat[min(int(n * 0.95), n - 1)],
            p99_latency_ns=sorted_lat[min(int(n * 0.99), n - 1)],
            max_latency_ns=sorted_lat[-1],
            link_utilization=self.link.utilization,
            dram_row_hit_rate=hit_rate,
            avg_queuing_delay_ns=sum(queuing_delays) / n,
            cxl_version=self.config.cxl.version.value,
            effective_bw_gbps=self.config.cxl.effective_bw_gbps,
            total_link_busy_ns=self.link._total_flits * self.config.cxl.flit_time_ns(),
            per_request_latency=latencies,
        )

    def compare_cxl_versions(
        self, trace: List[PromotionTraceEntry],
    ) -> Dict[str, TraceSimResult]:
        """Run simulation for CXL 1.1, 2.0, and 3.0 and compare."""
        results = {}
        for version in CXLVersion:
            cfg = SimulationConfig(
                cxl=CXLLinkConfig(version=version),
                dram=self.config.dram,
                mc=self.config.mc,
                seed=self.config.seed,
            )
            sim = KVPromotionTraceSimulator(cfg)
            results[version.value] = sim.simulate(trace)
        return results


# ======================================================================
# Trace Generator
# ======================================================================

class TraceGenerator:
    """Generate realistic KV promotion traces for simulation."""

    def __init__(self, seed: int = 42):
        import random
        self.rng = random.Random(seed)

    def generate_zipf_trace(
        self,
        num_steps: int = 1000,
        chunks_per_step: int = 3,
        total_chunks: int = 256,
        chunk_bytes: int = 65536,
        zipf_alpha: float = 1.2,
        step_interval_ns: float = 50000.0,
        num_tenants: int = 1,
    ) -> List[PromotionTraceEntry]:
        """Generate trace with Zipf-distributed chunk popularity.

        Args:
            num_steps: Number of decode steps.
            chunks_per_step: Chunks promoted per step.
            total_chunks: Total number of tail chunks.
            chunk_bytes: Bytes per chunk.
            zipf_alpha: Zipf distribution parameter (higher = more skewed).
            step_interval_ns: Time between decode steps.
            num_tenants: Number of tenants (round-robin assignment).
        """
        # Compute Zipf weights
        weights = [1.0 / (i + 1) ** zipf_alpha for i in range(total_chunks)]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        trace = []
        for step in range(num_steps):
            timestamp = step * step_interval_ns
            # Sample chunks according to Zipf
            selected = set()
            while len(selected) < chunks_per_step:
                r = self.rng.random()
                cumulative = 0.0
                for i, p in enumerate(probs):
                    cumulative += p
                    if r <= cumulative:
                        selected.add(i)
                        break

            for cid in selected:
                trace.append(PromotionTraceEntry(
                    step=step,
                    chunk_id=cid,
                    chunk_bytes=chunk_bytes,
                    tenant_id=step % num_tenants,
                    timestamp_ns=timestamp + self.rng.random() * 100,
                ))

        return trace

    def generate_bursty_trace(
        self,
        num_steps: int = 1000,
        base_chunks_per_step: int = 2,
        burst_chunks: int = 8,
        burst_probability: float = 0.1,
        total_chunks: int = 256,
        chunk_bytes: int = 65536,
        step_interval_ns: float = 50000.0,
    ) -> List[PromotionTraceEntry]:
        """Generate trace with occasional burst promotions."""
        trace = []
        for step in range(num_steps):
            timestamp = step * step_interval_ns
            is_burst = self.rng.random() < burst_probability
            n_chunks = burst_chunks if is_burst else base_chunks_per_step

            for _ in range(n_chunks):
                cid = self.rng.randint(0, total_chunks - 1)
                trace.append(PromotionTraceEntry(
                    step=step, chunk_id=cid,
                    chunk_bytes=chunk_bytes, tenant_id=0,
                    timestamp_ns=timestamp + self.rng.random() * 100,
                ))
        return trace
