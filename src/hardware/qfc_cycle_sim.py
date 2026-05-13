#!/usr/bin/env python3
"""
QFC (Query-Forwarding Compute) RTL-Level Cycle-Accurate Simulator
=================================================================

Tick-based 4-stage pipeline model for CXL-side near-data processing.

=== PHYSICAL DEPLOYMENT (§3.x) ===
QFC is located on the CXL MEMORY CONTROLLER (off-chip from GPU).
Its ~50.5us latency is an OFF-CHIP data-movement + compute delay that
traverses the CXL link.  This is the "slow path" for attention scoring.

Contrast with PHT/PTB/PPU (on-chip, inside GPU L2 cache controller):
  - PHT/PPU (on-chip):  1-5 cycles (~1-5ns)  — "should we promote?"
  - QFC (off-chip):     ~50.5us               — "what's the attention score?"
These operate on DIFFERENT paths. QFC is only invoked when on-chip
PHT/PPU prediction needs remote validation — the EXCEPTION case.

Pipeline stages:
  Stage 0: Arbitration    — CXL bus arbitration (FIFO)
  Stage 1: Transfer-Out   — Data transfer (Trad: 64KB / QFC: 1KB query)
  Stage 2: Compute        — Trad: GPU compute (modelled with SM pool) / QFC: MAC compute
  Stage 3: Transfer-Back  — QFC: 4B result return / Trad: N/A

Hardware specs:
- 8 MAC arrays (32-wide each), 50us compute per chunk
- Full-duplex CXL: separate upstream (GPU→CXL) and downstream (CXL→GPU)
  Each direction: 64 GB/s, 250ns one-way latency
- Traditional: 64KB fetch + GPU compute (limited by SM count)
- QFC: 1KB query + 50us MAC compute + 4B result

Key modelling improvements (v2):
- Full-duplex CXL bus: transfer-out and transfer-back run concurrently
- GPU SM pool: traditional path compute is limited by available SMs,
  not infinitely parallel
- Fair comparison: both paths have realistic parallelism constraints

Uses event-driven scheduling internally for efficiency while maintaining
cycle-accurate statistics.
"""

from __future__ import annotations

import heapq
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration & data-classes
# ---------------------------------------------------------------------------

@dataclass
class QFCConfig:
    """Configuration for the QFC cycle-accurate simulator."""
    cxl_bandwidth_gbps: float = 64.0
    cxl_latency_ns: float = 250.0
    num_mac_arrays: int = 8
    mac_width: int = 32
    d_model: int = 4096
    mac_compute_ns: float = 50_000.0       # 50 us
    full_chunk_size_bytes: int = 65536     # 64 KB
    gpu_compute_ns: float = 1_000_000.0    # 1 ms (per-chunk baseline)
    query_size_bytes: int = 1024           # 1 KB
    result_size_bytes: int = 4             # 4 B
    clock_freq_ghz: float = 1.0
    mac_fifo_depth: int = 8                # Increased from 2 to 8 for better batching
    # GPU SM pool for traditional path (models finite GPU parallelism)
    gpu_sm_count: int = 128                # A100 has 108 SMs, H100 has 132
    # CXL full-duplex: separate bandwidth for each direction
    cxl_full_duplex: bool = True           # CXL 3.0 is full-duplex
    # QFC miss coalescing: merge same-chunk queries within a batch window
    enable_miss_coalescing: bool = False     # Disabled by default (not applicable to per-request chunks)
    coalesce_window_ns: int = 5000         # 5us window for coalescing

    @property
    def bytes_per_ns(self) -> float:
        return self.cxl_bandwidth_gbps  # 64 GB/s = 64 B/ns


@dataclass
class QFCRequest:
    """Tracks a single request through the pipeline."""
    request_id: str
    chunk_id: int
    mode: str               # "traditional" or "qfc"
    submit_cycle: int
    start_transfer_cycle: int = -1
    compute_start_cycle: int = -1
    complete_cycle: int = -1
    data_bytes: int = 0
    # RTL extensions
    stage: str = "PENDING"
    assigned_mac: int = -1
    transfer_out_done_cycle: int = -1
    compute_done_cycle: int = -1
    transfer_back_done_cycle: int = -1

    @property
    def latency_ns(self) -> float:
        if self.complete_cycle < 0 or self.submit_cycle < 0:
            return float("inf")
        return float(self.complete_cycle - self.submit_cycle)


@dataclass
class QFCStats:
    total_cycles: int = 0
    total_requests: int = 0
    traditional_requests: int = 0
    qfc_requests: int = 0
    avg_latency_ns: float = 0.0
    p50_latency_ns: float = 0.0
    p95_latency_ns: float = 0.0
    p99_latency_ns: float = 0.0
    max_latency_ns: float = 0.0
    total_bytes_transferred: int = 0
    bandwidth_utilization: float = 0.0
    peak_queue_depth: int = 0
    avg_queue_depth: float = 0.0
    mac_utilization: float = 0.0
    mac_idle_cycles: int = 0


@dataclass
class QFCRTLStats(QFCStats):
    """Extended stats with RTL-level detail."""
    arbitration_stalls: int = 0
    mac_fifo_overflows: int = 0
    cxl_bus_utilization: float = 0.0
    mac_avg_queue_depth: float = 0.0
    per_stage_latency_ns: Dict[str, float] = field(default_factory=dict)


@dataclass
class BatchSimResult:
    mode: str = ""
    batch_size: int = 0
    chunks_per_request: int = 0
    total_latency_ns: float = 0.0
    per_request_latencies_ns: List[float] = field(default_factory=list)
    stats: QFCStats = field(default_factory=QFCStats)


@dataclass
class ComparisonResult:
    traditional: BatchSimResult = field(default_factory=BatchSimResult)
    qfc: BatchSimResult = field(default_factory=BatchSimResult)
    speedup: float = 1.0
    bandwidth_reduction: float = 1.0


# ---------------------------------------------------------------------------
# Internal hardware state
# ---------------------------------------------------------------------------

@dataclass
class MACArrayState:
    mac_id: int
    state: str = "IDLE"             # "IDLE" or "BUSY"
    busy_until_cycle: int = 0
    current_request: Optional[str] = None  # request_id
    fifo: List = field(default_factory=list)  # depth=2 local FIFO (request indices)
    total_busy_cycles: int = 0


@dataclass
class CXLBusState:
    state: str = "IDLE"             # "IDLE", "XFER_OUT", "XFER_BACK"
    current_request: Optional[str] = None
    free_at_cycle: int = 0
    total_busy_cycles: int = 0


@dataclass
class GPUSMState:
    """Tracks a single GPU SM processing a traditional chunk."""
    sm_id: int = 0
    busy_until_cycle: int = 0
    current_request: Optional[str] = None
    total_busy_cycles: int = 0


# ---------------------------------------------------------------------------
# Event types for event-driven scheduling
# ---------------------------------------------------------------------------

_EVT_XFER_OUT_DONE = 0
_EVT_COMPUTE_DONE = 1
_EVT_XFER_BACK_DONE = 2
_EVT_GPU_COMPUTE_DONE = 3
_EVT_BUS_FREE = 4  # sentinel: bus became free, re-evaluate
_EVT_GPU_SM_FREE = 5  # GPU SM became free, try scheduling


# ---------------------------------------------------------------------------
# RTL-Level Cycle-Accurate Simulator
# ---------------------------------------------------------------------------

class QFCCycleAccurateSim:
    """
    RTL-level cycle-accurate simulator for QFC engine.

    Uses event-driven scheduling for efficiency while maintaining
    cycle-accurate statistics for all pipeline stages.

    4-stage pipeline (processed in reverse order per tick):
      Stage 3: Transfer-Back  — QFC result return completes (downstream bus)
      Stage 2: Compute        — MAC/GPU compute completes
      Stage 1: Transfer-Out   — CXL data transfer completes (upstream bus)
      Stage 0: Arbitration    — Select next request from FIFO

    Key design choices:
    - Full-duplex CXL: upstream (GPU→CXL) and downstream (CXL→GPU) operate
      independently, enabling concurrent query submission and result return.
    - GPU SM pool: traditional path has finite compute parallelism;
      when all SMs are busy, new traditional chunks must wait.
    """

    def __init__(self, config: QFCConfig | None = None):
        self.cfg = config or QFCConfig()
        self._requests: List[QFCRequest] = []
        self._cycle: int = 0

        # Arbitration FIFO (indices into _requests)
        self._arb_queue: List[int] = []

        # Full-duplex CXL bus
        # Upstream: GPU → CXL (query/data transfer out)
        # Downstream: CXL → GPU (result/data transfer back)
        self._upstream_free_at: int = 0   # cycle when upstream bus becomes free
        self._downstream_free_at: int = 0  # cycle when downstream bus becomes free
        # Pending transfer-back queue
        self._xfer_back_pending: List[int] = []
        # Pending GPU compute queue (traditional chunks waiting for SM)
        self._gpu_compute_pending: List[int] = []

        # QFC miss coalescing: track pending queries per chunk_id
        # Format: {chunk_id: [request_idx, ...]}
        self._coalesce_buffer: Dict[int, List[int]] = {}
        self._coalesce_timer: int = 0  # cycle when current coalesce window expires
        self._coalesced_requests: int = 0  # stats: number of coalesced queries

        # MAC array states
        self._macs: List[MACArrayState] = [
            MACArrayState(mac_id=i) for i in range(self.cfg.num_mac_arrays)
        ]

        # GPU SM pool for traditional path
        self._gpu_sms: List[GPUSMState] = [
            GPUSMState(sm_id=i) for i in range(self.cfg.gpu_sm_count)
        ]
        self._active_gpu_sm_count: int = 0  # SMs currently busy

        # Event queue: (cycle, event_type, request_idx)
        self._event_queue: List[Tuple[int, int, int]] = []
        self._event_counter: int = 0  # tie-breaker for heapq

        # Stats accumulators
        self._queue_depth_samples: List[int] = []
        self._mac_queue_depth_samples: List[float] = []
        self._total_xfer_ns: int = 0
        self._total_mac_busy_ns: int = 0
        self._total_gpu_busy_ns: int = 0
        self._arbitration_stalls: int = 0
        self._mac_fifo_overflows: int = 0
        self._gpu_sm_stalls: int = 0
        self._peak_queue_depth: int = 0
        self._upstream_busy_cycles: int = 0
        self._downstream_busy_cycles: int = 0
        self._coalesced_requests: int = 0
        self._total_coalesce_savings_ns: int = 0

        # Stage latency accumulators
        self._stage_latencies: Dict[str, List[float]] = {
            "arbitration": [], "transfer_out": [],
            "compute": [], "transfer_back": [],
            "gpu_sm_wait": [],
        }

    def reset(self):
        self.__init__(self.cfg)

    # ------------------------------------------------------------------
    # Event scheduling helper
    # ------------------------------------------------------------------

    def _push_event(self, cycle: int, evt_type: int, req_idx: int):
        self._event_counter += 1
        heapq.heappush(self._event_queue,
                        (cycle, self._event_counter, evt_type, req_idx))

    def _pop_event(self) -> Tuple[int, int, int]:
        cycle, _cnt, evt_type, req_idx = heapq.heappop(self._event_queue)
        return cycle, evt_type, req_idx

    # ------------------------------------------------------------------
    # Submit requests
    # ------------------------------------------------------------------

    def submit_traditional_fetch(self, request_id: str, chunk_id: int,
                                  chunk_size_bytes: int = 65536) -> int:
        req = QFCRequest(
            request_id=request_id, chunk_id=chunk_id, mode="traditional",
            submit_cycle=self._cycle,
            data_bytes=chunk_size_bytes,
            stage="PENDING",
        )
        idx = len(self._requests)
        self._requests.append(req)
        self._arb_queue.append(idx)
        self._record_queue_depth()
        return idx

    def submit_qfc_query(self, request_id: str, chunk_id: int,
                          query_size_bytes: int = 1024) -> int:
        req = QFCRequest(
            request_id=request_id, chunk_id=chunk_id, mode="qfc",
            submit_cycle=self._cycle,
            data_bytes=query_size_bytes + self.cfg.result_size_bytes,
            stage="PENDING",
        )
        idx = len(self._requests)
        self._requests.append(req)
        
        # QFC miss coalescing: batch same-chunk queries within the SAME batch/request group
        # Extract batch index from request_id (format: "req_{batch}_chunk_{chunk}")
        if self.cfg.enable_miss_coalescing:
            batch_idx = None
            parts = request_id.split("_")
            if len(parts) >= 2:
                try:
                    batch_idx = int(parts[1])
                except (ValueError, IndexError):
                    pass
            
            # Coalescing key: (batch_idx, chunk_id) - only coalesce within same batch
            coalesce_key = (batch_idx, chunk_id)
            
            if coalesce_key not in self._coalesce_buffer:
                self._coalesce_buffer[coalesce_key] = []
                # Start coalesce window
                if self._coalesce_timer == 0:
                    self._coalesce_timer = self._cycle + self.cfg.coalesce_window_ns
            
            self._coalesce_buffer[coalesce_key].append(idx)
            self._record_queue_depth()
            return idx
        else:
            # No coalescing: immediately add to arbitration queue
            self._arb_queue.append(idx)
            self._record_queue_depth()
            return idx

    # ------------------------------------------------------------------
    # Tick — one clock cycle (1 ns @ 1 GHz)
    # ------------------------------------------------------------------

    def tick(self):
        """Advance one clock cycle. Processes stages in reverse order."""
        self._cycle += 1
        self._process_events_at(self._cycle)
        self._try_use_bus()

    # ------------------------------------------------------------------
    # Event-driven run — efficient for large latencies
    # ------------------------------------------------------------------

    def run_until_idle(self) -> int:
        """
        Process all pending requests using event-driven scheduling.
        Jumps to next event time rather than ticking every ns.

        Full-duplex: must track both upstream and downstream bus availability.
        """
        # Force flush coalesce buffer at start (all requests already submitted)
        self._try_flush_coalesce_buffer(force=True)
        self._try_use_bus()

        while self._event_queue or self._arb_queue or self._xfer_back_pending or self._gpu_compute_pending or self._coalesce_buffer:
            # Flush coalesce buffer if window expired or force flush when idle
            force_flush = not self._event_queue and not self._arb_queue
            self._try_flush_coalesce_buffer(force=force_flush)
            
            if self._event_queue:
                next_cycle = self._event_queue[0][0]
                # Also check if we need to flush coalesce buffer before next event
                if self._coalesce_buffer and self._coalesce_timer < next_cycle:
                    self._cycle = self._coalesce_timer
                    self._try_flush_coalesce_buffer()
                    self._try_use_bus()
                    continue
                    
                self._cycle = next_cycle
                self._process_events_at(self._cycle)
                self._try_use_bus()
            elif self._arb_queue or self._xfer_back_pending or self._gpu_compute_pending:
                # Requests waiting but no events — jump to nearest bus-free time
                next_free = min(self._upstream_free_at, self._downstream_free_at)
                # Also consider GPU SM freeing
                for sm in self._gpu_sms:
                    if sm.busy_until_cycle > self._cycle:
                        next_free = min(next_free, sm.busy_until_cycle)

                if next_free > self._cycle:
                    self._cycle = next_free
                    self._try_use_bus()
                else:
                    # Buses/SMs are free, try again
                    self._try_use_bus()
                    if not self._event_queue and not self._coalesce_buffer:
                        break  # Nothing can progress
            else:
                break

        return self._cycle

    def _try_flush_coalesce_buffer(self, force=False):
        """
        Flush coalesced queries to arbitration queue.
        
        For each chunk_id in coalesce buffer:
        - Only ONE representative query goes through the full pipeline
        - Other requests are marked as COALESCED and complete with the representative
        """
        if not self.cfg.enable_miss_coalescing or not self._coalesce_buffer:
            return
        
        # Flush if timer expired, force flag, or buffer is getting large
        should_flush = (force or 
                       self._cycle >= self._coalesce_timer or 
                       sum(len(v) for v in self._coalesce_buffer.values()) > 64)
        
        if not should_flush:
            return
        
        for chunk_id, req_indices in list(self._coalesce_buffer.items()):
            if not req_indices:
                continue
            
            # First request is the "representative" that goes through pipeline
            representative_idx = req_indices[0]
            self._arb_queue.append(representative_idx)
            
            # Mark other requests as coalesced - they will complete when representative completes
            if len(req_indices) > 1:
                for idx in req_indices[1:]:
                    req = self._requests[idx]
                    req.stage = "COALESCED"
                    req.assigned_mac = -2  # Special marker: coalesced request
                    self._coalesced_requests += 1
                
                # Track coalescing savings
                saved_queries = len(req_indices) - 1
                savings_ns = saved_queries * (self.cfg.query_size_bytes / self.cfg.bytes_per_ns + self.cfg.cxl_latency_ns)
                self._total_coalesce_savings_ns += savings_ns
        
        # Clear buffer and reset timer
        self._coalesce_buffer.clear()
        self._coalesce_timer = 0

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def _process_events_at(self, cycle: int):
        """Process all events scheduled at or before the given cycle."""
        while self._event_queue and self._event_queue[0][0] <= cycle:
            evt_cycle, evt_type, req_idx = self._pop_event()
            req = self._requests[req_idx]

            if evt_type == _EVT_XFER_OUT_DONE:
                self._handle_xfer_out_done(req, req_idx)
            elif evt_type == _EVT_COMPUTE_DONE:
                self._handle_compute_done(req, req_idx)
            elif evt_type == _EVT_XFER_BACK_DONE:
                self._handle_xfer_back_done(req, req_idx)
            elif evt_type == _EVT_GPU_COMPUTE_DONE:
                self._handle_gpu_compute_done(req, req_idx)
            elif evt_type == _EVT_GPU_SM_FREE:
                self._try_schedule_gpu_compute()

        # Also try scheduling pending GPU work after processing events
        self._try_schedule_gpu_compute()

    # ------------------------------------------------------------------
    # CXL Bus — full-duplex: independent upstream and downstream channels
    # ------------------------------------------------------------------

    def _is_upstream_free(self) -> bool:
        return self._cycle >= self._upstream_free_at

    def _is_downstream_free(self) -> bool:
        return self._cycle >= self._downstream_free_at

    def _try_use_bus(self):
        """
        Full-duplex bus scheduling:
        - Upstream (GPU→CXL): transfer-out (queries for QFC, full chunks for trad)
        - Downstream (CXL→GPU): transfer-back (QFC results)
        Both channels operate independently and concurrently.
        """
        # Downstream: process pending transfer-backs
        while self._is_downstream_free() and self._xfer_back_pending:
            req_idx = self._xfer_back_pending.pop(0)
            self._start_xfer_back(req_idx)

        # Upstream: process new requests from arbitration queue
        while self._is_upstream_free() and self._arb_queue:
            started = self._try_arbitrate_one()
            if not started:
                break  # Can't arbitrate (MAC backpressure or other)

        # Also try scheduling pending GPU compute
        self._try_schedule_gpu_compute()

    def _start_xfer_back(self, req_idx: int):
        """Start a transfer-back on the CXL downstream channel."""
        req = self._requests[req_idx]
        req.stage = "XFER_BACK"

        result_bytes = self.cfg.result_size_bytes
        xfer_ns = max(1, math.ceil(result_bytes / self.cfg.bytes_per_ns))
        total_time = xfer_ns + int(self.cfg.cxl_latency_ns)

        done_cycle = self._cycle + total_time
        self._downstream_free_at = done_cycle
        self._downstream_busy_cycles += total_time
        self._total_xfer_ns += xfer_ns

        req.transfer_back_done_cycle = done_cycle
        self._push_event(done_cycle, _EVT_XFER_BACK_DONE, req_idx)

    # ------------------------------------------------------------------
    # Stage 0: Arbitration — try to start one XFER_OUT
    # ------------------------------------------------------------------

    def _try_arbitrate_one(self) -> bool:
        """
        Try to dequeue one request from arbitration FIFO and start XFER_OUT.
        Returns True if a request was started, False if blocked.
        """
        if not self._arb_queue:
            return False

        idx = self._arb_queue[0]
        req = self._requests[idx]

        # For QFC: check MAC availability (backpressure)
        if req.mode == "qfc":
            mac_id = self._find_available_mac()
            if mac_id is None:
                self._mac_fifo_overflows += 1
                self._arbitration_stalls += 1
                return False
            req.assigned_mac = mac_id
            self._macs[mac_id].fifo.append(idx)

        # Dequeue
        self._arb_queue.pop(0)

        # Record arbitration wait
        arb_wait = self._cycle - req.submit_cycle
        self._stage_latencies["arbitration"].append(arb_wait)

        # Start Transfer-Out on upstream bus
        req.stage = "XFER_OUT"
        req.start_transfer_cycle = self._cycle

        if req.mode == "traditional":
            xfer_bytes = self.cfg.full_chunk_size_bytes
        else:
            xfer_bytes = self.cfg.query_size_bytes

        xfer_ns = math.ceil(xfer_bytes / self.cfg.bytes_per_ns)
        total_time = xfer_ns + int(self.cfg.cxl_latency_ns)

        done_cycle = self._cycle + total_time
        self._upstream_free_at = done_cycle
        self._upstream_busy_cycles += total_time
        self._total_xfer_ns += xfer_ns

        req.transfer_out_done_cycle = done_cycle
        self._push_event(done_cycle, _EVT_XFER_OUT_DONE, idx)
        self._record_queue_depth()
        return True

    def _find_available_mac(self) -> Optional[int]:
        """Find MAC with shortest FIFO. Returns None if all FIFOs full."""
        best_mac = None
        best_depth = self.cfg.mac_fifo_depth + 1

        for mac in self._macs:
            depth = len(mac.fifo)
            if depth < best_depth:
                best_depth = depth
                best_mac = mac.mac_id

        if best_mac is not None and best_depth < self.cfg.mac_fifo_depth:
            return best_mac
        return None

    # ------------------------------------------------------------------
    # Stage 1: Transfer-Out complete
    # ------------------------------------------------------------------

    def _handle_xfer_out_done(self, req: QFCRequest, req_idx: int):
        """Transfer-Out complete → enter Compute stage."""
        xfer_latency = req.transfer_out_done_cycle - req.start_transfer_cycle
        self._stage_latencies["transfer_out"].append(xfer_latency)

        if req.mode == "traditional":
            # Traditional: queue for GPU SM pool (limited parallelism)
            req.stage = "WAIT_GPU_SM"
            self._gpu_compute_pending.append(req_idx)
            self._try_schedule_gpu_compute()
        else:
            # QFC: ready for MAC compute
            req.stage = "WAIT_MAC"
            mac = self._macs[req.assigned_mac]
            self._try_start_mac(mac)

    # ------------------------------------------------------------------
    # GPU SM pool management for traditional path
    # ------------------------------------------------------------------

    def _find_free_gpu_sm(self) -> Optional[int]:
        """Find a free GPU SM. Returns sm_id or None."""
        for sm in self._gpu_sms:
            if sm.busy_until_cycle <= self._cycle:
                return sm.sm_id
        return None

    def _compute_gpu_compute_time(self, batch_size: int, chunks_per_request: int = 32) -> float:
        """
        Compute GPU compute time with SM contention modeling.
        
        In traditional path, larger batches cause MORE contention for GPU SMs.
        When batch_size * chunks_per_request > gpu_sm_count, chunks must queue
        for SM access, increasing effective per-chunk latency.
        
        Model: base_time * contention_factor
        - contention_factor = 1.0 when total_chunks <= sm_count
        - contention_factor > 1.0 when chunks queue for SMs
        """
        base_time = self.cfg.gpu_compute_ns
        total_chunks = batch_size * chunks_per_request
        
        # SM contention: when more chunks than SMs, queueing increases latency
        if total_chunks <= self.cfg.gpu_sm_count:
            contention_factor = 1.0
        else:
            # Linear scaling: each chunk waits for its turn
            contention_factor = total_chunks / self.cfg.gpu_sm_count
        
        return base_time * contention_factor

    def _try_schedule_gpu_compute(self):
        """Try to schedule pending traditional chunks onto free GPU SMs."""
        # Compute the actual GPU compute time based on current batch characteristics
        # We estimate batch_size from the number of unique request prefixes
        unique_requests = set()
        for idx in self._gpu_compute_pending:
            req = self._requests[idx]
            # Extract batch index from request_id format "req_{batch}_chunk_{chunk}"
            parts = req.request_id.split("_")
            if len(parts) >= 2:
                try:
                    unique_requests.add(int(parts[1]))
                except (ValueError, IndexError):
                    pass
        estimated_batch = max(1, len(unique_requests))
        
        # Estimate chunks_per_request from pending requests
        chunks_per_req = max(1, len(self._gpu_compute_pending) // max(1, estimated_batch))
        
        gpu_compute_time = int(self._compute_gpu_compute_time(estimated_batch, chunks_per_req))
        
        while self._gpu_compute_pending:
            sm_id = self._find_free_gpu_sm()
            if sm_id is None:
                if self._gpu_compute_pending:
                    self._gpu_sm_stalls += 1
                break

            req_idx = self._gpu_compute_pending.pop(0)
            req = self._requests[req_idx]
            sm = self._gpu_sms[sm_id]

            # Record SM wait time
            sm_wait = self._cycle - req.transfer_out_done_cycle
            self._stage_latencies["gpu_sm_wait"].append(sm_wait)

            # Start GPU compute on this SM
            req.stage = "COMPUTE"
            req.compute_start_cycle = self._cycle
            compute_time = gpu_compute_time
            compute_done = self._cycle + compute_time

            sm.busy_until_cycle = compute_done
            sm.current_request = req.request_id
            sm.total_busy_cycles += compute_time
            self._total_gpu_busy_ns += compute_time
            self._active_gpu_sm_count += 1

            req.compute_done_cycle = compute_done
            self._push_event(compute_done, _EVT_GPU_COMPUTE_DONE, req_idx)

    # ------------------------------------------------------------------
    # MAC compute management
    # ------------------------------------------------------------------

    def _try_start_mac(self, mac: MACArrayState):
        """Try to start computing on a MAC if it's idle and FIFO front is ready."""
        if mac.state == "BUSY" or not mac.fifo:
            return

        req_idx = mac.fifo[0]
        req = self._requests[req_idx]

        # Only start if transfer-out is complete
        if req.transfer_out_done_cycle < 0 or req.transfer_out_done_cycle > self._cycle:
            return

        # Start compute
        mac.state = "BUSY"
        mac.current_request = req.request_id
        req.stage = "COMPUTE"
        req.compute_start_cycle = self._cycle

        compute_time = int(self.cfg.mac_compute_ns)
        compute_done = self._cycle + compute_time
        mac.busy_until_cycle = compute_done
        mac.total_busy_cycles += compute_time
        self._total_mac_busy_ns += compute_time

        req.compute_done_cycle = compute_done
        self._push_event(compute_done, _EVT_COMPUTE_DONE, req_idx)

    # ------------------------------------------------------------------
    # Stage 2: Compute complete
    # ------------------------------------------------------------------

    def _handle_compute_done(self, req: QFCRequest, req_idx: int):
        """MAC compute done → queue Transfer-Back."""
        mac = self._macs[req.assigned_mac]
        mac.state = "IDLE"
        mac.current_request = None

        # Remove from MAC FIFO (always front)
        if mac.fifo and mac.fifo[0] == req_idx:
            mac.fifo.pop(0)

        compute_latency = req.compute_done_cycle - req.compute_start_cycle
        self._stage_latencies["compute"].append(compute_latency)

        # Sample MAC queue depths
        avg_depth = float(np.mean([len(m.fifo) for m in self._macs]))
        self._mac_queue_depth_samples.append(avg_depth)

        # Queue transfer-back (will be scheduled when bus is free)
        self._xfer_back_pending.append(req_idx)

        # Try to start next compute on this MAC
        self._try_start_mac(mac)

    def _handle_gpu_compute_done(self, req: QFCRequest, req_idx: int):
        """GPU compute done (traditional path) → request complete."""
        # Free the SM
        for sm in self._gpu_sms:
            if sm.current_request == req.request_id:
                sm.state = "IDLE"  # not used but for clarity
                sm.current_request = None
                break
        self._active_gpu_sm_count = max(0, self._active_gpu_sm_count - 1)

        compute_latency = req.compute_done_cycle - req.compute_start_cycle
        self._stage_latencies["compute"].append(compute_latency)
        req.stage = "DONE"
        req.complete_cycle = self._cycle

        # Try scheduling next pending chunk
        self._try_schedule_gpu_compute()

    # ------------------------------------------------------------------
    # Stage 3: Transfer-Back complete
    # ------------------------------------------------------------------

    def _handle_xfer_back_done(self, req: QFCRequest, req_idx: int):
        """Transfer-Back complete → request done."""
        xfer_back_latency = req.transfer_back_done_cycle - req.compute_done_cycle
        self._stage_latencies["transfer_back"].append(xfer_back_latency)
        req.stage = "DONE"
        req.complete_cycle = self._cycle
        
        # Also complete all coalesced requests for the same batch+chunk
        # Find all coalesced requests with same chunk_id (they share the batch info in coalesce_key)
        for other_req in self._requests:
            if (other_req.stage == "COALESCED" and 
                other_req.chunk_id == req.chunk_id and
                other_req.complete_cycle < 0):
                other_req.stage = "DONE"
                other_req.complete_cycle = self._cycle
                other_req.compute_done_cycle = req.compute_done_cycle
                other_req.transfer_back_done_cycle = req.transfer_back_done_cycle

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _record_queue_depth(self):
        depth = len(self._arb_queue)
        self._queue_depth_samples.append(depth)
        if depth > self._peak_queue_depth:
            self._peak_queue_depth = depth

    def get_stats(self) -> QFCRTLStats:
        completed = [r for r in self._requests if r.complete_cycle >= 0]
        if not completed:
            return QFCRTLStats()

        lats = np.array([r.latency_ns for r in completed], dtype=np.float64)
        total_cycles = max(self._cycle, 1)
        total_bytes = sum(r.data_bytes for r in self._requests)

        bw_util = min(1.0, self._total_xfer_ns / total_cycles)

        mac_total = total_cycles * self.cfg.num_mac_arrays
        mac_util = min(1.0, self._total_mac_busy_ns / max(mac_total, 1))
        mac_idle = max(0, mac_total - self._total_mac_busy_ns)

        qd = self._queue_depth_samples if self._queue_depth_samples else [0]

        per_stage = {}
        for stage_name, samples in self._stage_latencies.items():
            per_stage[stage_name] = float(np.mean(samples)) if samples else 0.0

        mac_avg_qd = (float(np.mean(self._mac_queue_depth_samples))
                      if self._mac_queue_depth_samples else 0.0)
        cxl_util = min(1.0, (self._upstream_busy_cycles + self._downstream_busy_cycles)
                       / max(2 * total_cycles, 1))

        # GPU SM utilization (traditional path)
        gpu_sm_total = total_cycles * self.cfg.gpu_sm_count
        gpu_sm_util = min(1.0, self._total_gpu_busy_ns / max(gpu_sm_total, 1))

        return QFCRTLStats(
            total_cycles=total_cycles,
            total_requests=len(self._requests),
            traditional_requests=sum(1 for r in self._requests if r.mode == "traditional"),
            qfc_requests=sum(1 for r in self._requests if r.mode == "qfc"),
            avg_latency_ns=float(np.mean(lats)),
            p50_latency_ns=float(np.percentile(lats, 50)),
            p95_latency_ns=float(np.percentile(lats, 95)),
            p99_latency_ns=float(np.percentile(lats, 99)),
            max_latency_ns=float(np.max(lats)),
            total_bytes_transferred=total_bytes,
            bandwidth_utilization=bw_util,
            peak_queue_depth=self._peak_queue_depth,
            avg_queue_depth=float(np.mean(qd)),
            mac_utilization=mac_util,
            mac_idle_cycles=int(mac_idle),
            arbitration_stalls=self._arbitration_stalls,
            mac_fifo_overflows=self._mac_fifo_overflows,
            cxl_bus_utilization=cxl_util,
            mac_avg_queue_depth=mac_avg_qd,
            per_stage_latency_ns=per_stage,
        )

    # ------------------------------------------------------------------
    # High-level batch helpers (backward compatible)
    # ------------------------------------------------------------------

    def simulate_batch(self, batch_size: int, chunks_per_request: int,
                        mode: str = "qfc") -> BatchSimResult:
        self.reset()
        for b in range(batch_size):
            for c in range(chunks_per_request):
                rid = f"req_{b}_chunk_{c}"
                if mode == "traditional":
                    self.submit_traditional_fetch(rid, c, self.cfg.full_chunk_size_bytes)
                else:
                    self.submit_qfc_query(rid, c, self.cfg.query_size_bytes)

        self.run_until_idle()
        stats = self.get_stats()

        # Per-request latency = max chunk latency within each request
        per_req: Dict[int, float] = {}
        for r in self._requests:
            b_idx = int(r.request_id.split("_")[1])
            per_req[b_idx] = max(per_req.get(b_idx, 0.0), r.latency_ns)

        per_req_list = [per_req[i] for i in sorted(per_req.keys())]

        return BatchSimResult(
            mode=mode,
            batch_size=batch_size,
            chunks_per_request=chunks_per_request,
            total_latency_ns=float(self._cycle),
            per_request_latencies_ns=per_req_list,
            stats=stats,
        )

    def compare_modes(self, batch_size: int,
                       chunks_per_request: int) -> ComparisonResult:
        trad = self.simulate_batch(batch_size, chunks_per_request, "traditional")
        qfc_ = self.simulate_batch(batch_size, chunks_per_request, "qfc")
        # Use p99 per-request latency for speedup (not makespan)
        trad_p99 = trad.stats.p99_latency_ns
        qfc_p99 = qfc_.stats.p99_latency_ns
        speedup = trad_p99 / max(qfc_p99, 1.0)
        bw_red = trad.stats.total_bytes_transferred / max(qfc_.stats.total_bytes_transferred, 1)
        return ComparisonResult(
            traditional=trad, qfc=qfc_,
            speedup=speedup, bandwidth_reduction=bw_red,
        )


# ---------------------------------------------------------------------------
# Convenience sweep (backward compatible)
# ---------------------------------------------------------------------------

def run_comparison_sweep(batch_sizes: List[int] | None = None,
                          chunks: int = 8) -> Dict:
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64, 128, 256]
    sim = QFCCycleAccurateSim()
    results = {}
    for bs in batch_sizes:
        cmp = sim.compare_modes(bs, chunks)
        results[bs] = {
            "traditional_latency_ns": cmp.traditional.total_latency_ns,
            "qfc_latency_ns": cmp.qfc.total_latency_ns,
            "speedup": cmp.speedup,
            "bandwidth_reduction": cmp.bandwidth_reduction,
            "trad_p99_ns": cmp.traditional.stats.p99_latency_ns,
            "qfc_p99_ns": cmp.qfc.stats.p99_latency_ns,
        }
    return results


if __name__ == "__main__":
    print("=" * 72)
    print("QFC RTL Cycle-Accurate Simulator — Sweep (8 chunks per request)")
    print("=" * 72)
    res = run_comparison_sweep()
    hdr = f"{'Batch':>6} | {'Trad (ms)':>10} | {'QFC (ms)':>10} | {'Speedup':>8} | {'BW Red':>7}"
    print(hdr)
    print("-" * len(hdr))
    for bs, d in res.items():
        print(f"{bs:>6} | {d['traditional_latency_ns']/1e6:>10.3f} | "
              f"{d['qfc_latency_ns']/1e6:>10.3f} | "
              f"{d['speedup']:>8.2f}x | "
              f"{d['bandwidth_reduction']:>6.1f}x")
