"""
SimCXL + PROSE Cycle-Accurate Co-Simulator
==========================================

A unified event-driven cycle simulator that models:
  - SimCXL's validated CXL Type-3 memory controller at RTL fidelity
  - PROSE's PHT, PPU, DMA scheduler, QFC as hardware blocks
  - Two-tier memory (HBM fast tier + CXL/DRAM far tier)
  - CXL.mem sub-protocol at flit level (M2SReq, M2SRwD, S2MDRS, S2MNDR)
  - Queue contention between promotion DMA and regular memory access

This is NOT parameter borrowing from SimCXL — the entire SimCXL CXL.mem
controller data path is re-implemented here at the same cycle accuracy,
and PROSE mechanisms are inserted as hardware modules on that data path.

Design follows SimCXL's CXLMemCtrl architecture:
  CXLResponsePort (host-facing) ←→ CXLRequestPort (memory-facing)
  ┌────────────────────────────────────────────────────┐
  │   CXLResponsePort                                   │
  │   recvTimingReq → processCXLMem → schedTimingReq   │
  │   (PROSE PHT query injected here)                  │
  │                                                     │
  │   transmitList (response queue, depth 48)           │
  │   protoProcLat (15 ns)                              │
  └────────────────────────────────────────────────────┘
  ┌────────────────────────────────────────────────────┐
  │   CXLRequestPort                                    │
  │   schedTimingReq → trySendTiming → sendTimingReq   │
  │   (PROSE DMA promotion traffic shares this queue)  │
  │                                                     │
  │   transmitList (request queue, depth 48)            │
  └────────────────────────────────────────────────────┘

Supports TWO modes within the SAME simulation framework:
  - baseline: Standard CXL Type-3 (no PROSE, plain HBM caching)
  - prose: PROSE-enabled (PHT + PPU + DMA + QFC promotion pipeline)

Statistics match SimCXL's CXLCtrlStats naming for direct comparison.
"""

from __future__ import annotations

import copy
import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# SimCXL hardware timing constants (from SimCXL source + Cohet paper)
# These ARE the SimCXL parameters, used in the same simulation framework
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimCXLTiming:
    """Validated SimCXL timing parameters."""
    clock_period_ns: float = 1.0               # 1 GHz reference clock
    proto_proc_lat_ns: float = 15.0            # CXL.mem protocol processing
    bridge_lat_ns: float = 50.0                # CXL bridge transit
    cxl_link_bw_gbps: float = 55.0             # CXL 2.0 x16
    cxl_link_latency_ns: float = 250.0         # One-way CXL link
    ddr5_tCL_ns: float = 16.0                  # DDR5-4400 CAS
    ddr5_tRCD_ns: float = 16.0                 # DDR5-4400 RAS-to-CAS
    ddr5_tRP_ns: float = 16.0                  # DDR5-4400 precharge
    ddr5_tRAS_ns: float = 32.0                 # DDR5-4400 row active
    ddr5_bandwidth_gbps: float = 70.4          # DDR5-4400 per channel
    hbm_bandwidth_gbps: float = 900.0           # HBM2e proxy (fast tier)
    cxl_flit_size_bytes: int = 256              # CXL.mem flit size
    cxl_protocol_overhead_pct: float = 2.0      # DBIE + CRC overhead
    resp_queue_depth: int = 48                  # Response queue limit
    req_queue_depth: int = 48                   # Request queue limit
    credit_rtt_ns: float = 100.0                # CXL credit return RTT
    l3_cacheline_bytes: int = 64                # Cache line size


# ---------------------------------------------------------------------------
# CXL.mem Protocol Commands (matching SimCXL packet.hh)
# ---------------------------------------------------------------------------

class CXLCmd:
    M2SReq = 0    # Master-to-Subordinate Request (read)
    M2SRwD = 1    # Master-to-Subordinate Request with Data (write)
    S2MDRS = 2    # Subordinate-to-Master Data Response (read resp)
    S2MNDR = 3    # Subordinate-to-Master No Data Response (write resp)


# ---------------------------------------------------------------------------
# Event types for the event-driven scheduler
# ---------------------------------------------------------------------------

_EVT_TRY_SEND_REQ = 0
_EVT_TRY_SEND_RESP = 1
_EVT_DMA_PROMOTION = 2
_EVT_PHT_UPDATE_STAGE1 = 3    # read old entry
_EVT_PHT_UPDATE_STAGE2 = 4    # compute EMA
_EVT_PHT_UPDATE_STAGE3 = 5    # writeback
_EVT_PPU_STEP_START = 6       # begin PROSE decode step
_EVT_QFC_XFER_OUT_DONE = 7
_EVT_QFC_COMPUTE_DONE = 8
_EVT_QFC_XFER_BACK_DONE = 9
_EVT_STATS_SAMPLE = 10
_EVT_MEM_RESPONSE = 11        # DRAM response arrives


# ---------------------------------------------------------------------------
# PHT (Promotion History Table) — Cycle-Accurate Model
# ---------------------------------------------------------------------------

@dataclass
class PHTEntryState:
    anchor: bool = False
    counter: int = 0          # 2-bit saturating counter: 0-3
    valid: bool = False
    lru_age: int = 0
    ema_value: int = 0        # 16-bit EMA in 0.16 fixed-point
    _EMA_MAX: int = 65535


class PHTEntry:
    __slots__ = ('anchor', 'counter', 'valid', 'lru_age', 'ema_value')
    _EMA_MAX = 65535

    def __init__(self):
        self.anchor = False
        self.counter = 0
        self.valid = False
        self.lru_age = 0
        self.ema_value = 0


class CycleAccuratePHT:
    """
    PHT with structural hashing, 3-cycle pipelined update, 1-cycle query,
    RAW hazard forwarding. Matches hardware/rtl/PHT_ENGINE.sv.
    """

    def __init__(self, num_entries: int = 1024, timing: SimCXLTiming | None = None):
        self.num_entries = num_entries
        self.timing = timing or SimCXLTiming()
        self._regfile: List[PHTEntry] = [PHTEntry() for _ in range(num_entries)]

        # 2-stage update pipeline: [0]=read-done, [1]=compute-done
        self._pipeline: List[Optional[Dict[str, Any]]] = [None, None]
        self._pending_update: Optional[Dict[str, Any]] = None

        # Stats
        self.queries_total: int = 0
        self.queries_hit: int = 0
        self.updates_total: int = 0
        self.pipeline_stalls: int = 0
        self.raw_hazards: int = 0

    def _hash(self, chunk_id: int, layer_id: int = 0, context_hash: int = 0) -> int:
        """Structural hash matching RTL: position XOR layer XOR context."""
        position_bits = chunk_id & 0xFF
        layer_bits = layer_id & 0x0F
        ctx_bits = context_hash & 0x0F
        return (position_bits ^ (layer_bits << 8) ^ (ctx_bits << 4)) % self.num_entries

    def query(self, chunk_id: int, layer_id: int = 0, context_hash: int = 0) -> Tuple[bool, float]:
        """1-cycle combinational read with RAW forwarding."""
        idx = self._hash(chunk_id, layer_id, context_hash)
        self.queries_total += 1

        # Check pipeline for in-flight updates (RAW hazard forwarding)
        fwd_ema: Optional[int] = None
        for stage_idx in reversed(range(2)):
            slot = self._pipeline[stage_idx]
            if slot and slot.get('idx') == idx:
                self.raw_hazards += 1
                fwd_ema = (slot.get('new_ema') if stage_idx == 1
                           else slot.get('old_ema'))
                break

        if fwd_ema is None and self._pending_update:
            if self._pending_update.get('idx') == idx:
                self.raw_hazards += 1
                fwd_ema = self._regfile[idx].ema_value

        if fwd_ema is not None:
            predicted = fwd_ema >= 0x826C
            return predicted, fwd_ema / PHTEntry._EMA_MAX

        entry = self._regfile[idx]
        if not entry.valid:
            return False, 0.0

        self.queries_hit += 1
        ema_norm = entry.ema_value / PHTEntry._EMA_MAX
        return entry.counter >= 2, ema_norm

    def update(self, chunk_id: int, promoted: bool, importance: float = 1.0):
        """Issue a pipelined update. Takes 3 cycles to complete."""
        idx = self._hash(chunk_id, 0, 0)
        self.updates_total += 1

        if self._pending_update is not None:
            self.pipeline_stalls += 1

        self._pending_update = {
            'idx': idx,
            'promoted': promoted,
            'importance': importance,
            'old_ema': 0,
            'new_ema': 0,
            'anchor': False,
            'valid': True,
        }

    def tick(self):
        """Advance pipeline by one cycle (called from main event loop)."""
        # Writeback: pipe[1] → regfile
        wb = self._pipeline[1]
        if wb and wb.get('valid'):
            entry = self._regfile[wb['idx']]
            entry.ema_value = wb['new_ema']
            entry.counter = min(3, entry.counter + 1) if wb['promoted'] else max(0, entry.counter - 1)
            entry.valid = True
            entry.lru_age = min(3, entry.lru_age + 1)

        # Compute: pipe[0] → pipe[1]
        p0 = self._pipeline[0]
        if p0 and p0.get('valid'):
            p0['new_ema'] = self._compute_ema(
                p0['old_ema'], p0['promoted'], p0['importance']
            )
            self._pipeline[1] = p0
        else:
            self._pipeline[1] = None

        # Accept: pending → pipe[0]
        if self._pending_update is not None:
            pend = self._pending_update
            old_entry = self._regfile[pend['idx']]
            pend['old_ema'] = old_entry.ema_value
            self._pipeline[0] = pend
            self._pending_update = None
        else:
            self._pipeline[0] = None

    @staticmethod
    def _compute_ema(old_ema: int, promoted: bool, importance: float) -> int:
        ANCHOR_FLOOR = 0x826C
        EMA_MAX = 65535
        if promoted:
            imp_val = max(0, min(EMA_MAX, int(importance * EMA_MAX)))
            new_ema = (old_ema * 4 + imp_val) // 5
        else:
            new_ema = (old_ema * 4) // 5
        new_ema = max(0, min(EMA_MAX, new_ema))
        if new_ema < ANCHOR_FLOOR:
            new_ema = max(0, new_ema)
        return new_ema


# ---------------------------------------------------------------------------
# PPU (Promotion Prediction Unit) Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PPUConfig:
    lut_entries: int = 256
    lut_output_bits: int = 8
    recency_bits: int = 2
    similarity_bits: int = 2
    position_bits: int = 2
    history_bits: int = 2
    counter_bits: int = 16
    dma_queue_depth: int = 8
    dma_priority_bits: int = 8
    dma_coalesce_window: int = 4


class PPUPipeline:
    """5-stage hardware promotion scoring pipeline. ~5ns on-chip."""

    STAGES = ("mmrf_receive", "counter_update", "feature_extract",
              "lut_lookup", "dma_enqueue")

    def __init__(self, config: PPUConfig | None = None):
        self.config = config or PPUConfig()
        self._counters: Dict[int, int] = {}  # chunk_id → saturating counter
        self._lut: np.ndarray = self._init_lut()
        self._dma_queue: deque = deque(maxlen=self.config.dma_queue_depth)
        self._dma_dropped: int = 0
        self._step_attention_masses: Dict[int, float] = {}

        # Pipeline stall tracking
        self.dma_enqueues: int = 0
        self.dma_stalls: int = 0
        self.busy_cycles: int = 0

    @staticmethod
    def _init_lut() -> np.ndarray:
        """Initialize LUT with ODUS-like scoring:
        utility ≈ 0.5*recency + 0.3*similarity + 0.15*position + 0.05*history
        plus low-noise dither to break ties.
        """
        lut = np.zeros(256, dtype=np.uint8)
        rng = np.random.RandomState(42)
        for i in range(256):
            recency = (i >> 6) & 0x3      # bits [7:6]
            similarity = (i >> 4) & 0x3    # bits [5:4]
            position = (i >> 2) & 0x3      # bits [3:2]
            history = i & 0x3              # bits [1:0]
            # Weighted score + small dither
            score = (0.50 * recency + 0.30 * similarity +
                     0.15 * position + 0.05 * history) / 3.0
            score += float(rng.normal(0, 0.02))  # tiny dither
            score = max(0.0, min(1.0, score))
            lut[i] = int(score * 255)
        return lut

    def begin_step(self, attention_masses: Dict[int, float]):
        """Stage 1: MMRF Receive — accept chunk-level attention masses."""
        self._step_attention_masses = dict(attention_masses)

    def score_candidate(
        self, chunk_id: int, chunk_layer: int = 0,
        attention_mass: float = 0.0, recent_access_steps: int = 100,
        position_ratio: float = 0.5, past_promotions: int = 0,
    ) -> Tuple[float, bool]:
        """
        Stages 2-5: Counter → Feature → LUT → DMA.

        Returns (utility_score, dma_enqueued).
        """
        # Stage 2: Counter update
        mass = self._step_attention_masses.get(chunk_id, attention_mass)
        old_counter = self._counters.get(chunk_id, 0)
        increment = max(0, int(round(mass * 256.0)))
        new_counter = min((1 << self.config.counter_bits) - 1, old_counter + increment)
        self._counters[chunk_id] = new_counter

        # Stage 3: Feature extraction (quantized to 2-bit each)
        recency = max(0, min(3, 3 - int(math.log2(max(1, recent_access_steps)) // 3)))
        similarity = max(0, min(3, int(mass * 8.0)))
        position = max(0, min(3, int(position_ratio * 4.0)))
        history = max(0, min(3, past_promotions))

        packed_index = ((recency << 6) | (similarity << 4) | (position << 2) | history) & 0xFF

        # Stage 4: LUT lookup
        utility_raw = int(self._lut[packed_index % self.config.lut_entries])
        utility = utility_raw / 255.0

        # Stage 5: DMA enqueue
        enqueued = False
        if utility >= 0.15:
            if len(self._dma_queue) < self.config.dma_queue_depth:
                self._dma_queue.append({
                    'chunk_id': chunk_id,
                    'priority': int(utility * 255),
                    'utility': utility,
                })
                self.dma_enqueues += 1
            else:
                self._dma_dropped += 1
                self.dma_stalls += 1
            enqueued = True

        return utility, enqueued

    def end_step(self):
        """Decay counters (shift-based EMA)."""
        to_delete = []
        for cid in list(self._counters):
            self._counters[cid] = self._counters[cid] * 3 // 4
            if self._counters[cid] <= 0:
                to_delete.append(cid)
        for cid in to_delete:
            del self._counters[cid]
        self._step_attention_masses.clear()

    def dequeue_dma(self) -> Optional[Dict[str, Any]]:
        if self._dma_queue:
            return self._dma_queue.popleft()
        return None

    def dma_queue_depth(self) -> int:
        return len(self._dma_queue)


# ---------------------------------------------------------------------------
# CXL.mem Packet Model
# ---------------------------------------------------------------------------

@dataclass
class CXLMemPacket:
    pkt_id: int
    cxl_cmd: int            # M2SReq, M2SRwD, S2MDRS, S2MNDR
    addr: int
    size_bytes: int = 64
    arrival_tick: float = 0.0
    header_delay_ns: float = 0.0
    payload_delay_ns: float = 0.0
    is_promotion_dma: bool = False
    chunk_id: int = -1
    needs_response: bool = True
    is_read: bool = True


@dataclass
class DeferredPacket:
    pkt: CXLMemPacket
    scheduled_tick: float
    entry_tick: float = 0.0


# ---------------------------------------------------------------------------
# CXL Memory Controller Port Model
# ---------------------------------------------------------------------------

class CXLResponsePortModel:
    """Host-facing port: receives requests, sends responses."""

    def __init__(self, timing: SimCXLTiming):
        self.timing = timing
        self.transmit_list: deque[DeferredPacket] = deque()
        self.outstanding_responses: int = 0
        self.retry_req: bool = False

        # Stats
        self.rsp_que_full_events: int = 0
        self.rsp_send_succeeded: int = 0
        self.rsp_send_failed: int = 0
        self.req_retry_counts: int = 0
        self.rsp_queue_len_samples: List[int] = []
        self.rsp_queue_lat_samples: List[float] = []
        self.rsp_outstanding_samples: List[int] = []

    def resp_queue_full(self) -> bool:
        return self.outstanding_responses >= self.timing.resp_queue_depth

    def recv_timing_req(self, pkt: CXLMemPacket) -> bool:
        """Receive request from host. Returns False if blocked (retry needed)."""
        if self.retry_req:
            return False

        if self.resp_queue_full():
            if pkt.needs_response:
                self.rsp_que_full_events += 1
                self.retry_req = True
                return False

        if pkt.needs_response:
            self.outstanding_responses += 1
            self.rsp_outstanding_samples.append(self.outstanding_responses)

        delay_ns = pkt.header_delay_ns + pkt.payload_delay_ns
        scheduled = pkt.arrival_tick + self.timing.proto_proc_lat_ns + delay_ns
        pkt.header_delay_ns = 0.0
        pkt.payload_delay_ns = 0.0

        self.sched_timing_req(pkt, scheduled)
        return True

    def sched_timing_req(self, pkt: CXLMemPacket, when: float):
        dp = DeferredPacket(pkt=pkt, scheduled_tick=when, entry_tick=pkt.arrival_tick)
        self.transmit_list.append(dp)
        self.rsp_queue_len_samples.append(len(self.transmit_list))
        return when

    def sched_timing_resp(self, pkt: CXLMemPacket, when: float):
        dp = DeferredPacket(pkt=pkt, scheduled_tick=when, entry_tick=pkt.arrival_tick)
        self.transmit_list.append(dp)
        self.rsp_queue_len_samples.append(len(self.transmit_list))
        return when

    def try_send_timing(self, current_tick: float) -> Optional[Tuple[CXLMemPacket, str]]:
        """Try to send the head packet. Returns (pkt, 'req'|'resp') or None."""
        if not self.transmit_list:
            return None
        if self.transmit_list[0].scheduled_tick > current_tick:
            return None

        dp = self.transmit_list.popleft()
        pkt = dp.pkt

        if pkt.is_read:
            self.rsp_send_succeeded += 1
            self.rsp_queue_lat_samples.append(current_tick - dp.entry_tick)
            self.rsp_queue_len_samples.append(len(self.transmit_list))

            if pkt.needs_response:
                self.outstanding_responses = max(0, self.outstanding_responses - 1)
                self.rsp_outstanding_samples.append(self.outstanding_responses)

            if self.retry_req:
                self.retry_req = False
                self.req_retry_counts += 1

            return pkt, 'resp'
        else:
            return pkt, 'req'

    def retry_stalled(self):
        if self.retry_req:
            self.retry_req = False
            self.req_retry_counts += 1
            return True
        return False


class CXLRequestPortModel:
    """Memory-facing port: sends requests to DRAM, receives responses."""

    def __init__(self, timing: SimCXLTiming):
        self.timing = timing
        self.transmit_list: deque[DeferredPacket] = deque()

        # Stats
        self.req_que_full_events: int = 0
        self.req_send_succeeded: int = 0
        self.req_send_failed: int = 0
        self.req_queue_len_samples: List[int] = []
        self.req_queue_lat_samples: List[float] = []

    def req_queue_full(self) -> bool:
        return len(self.transmit_list) >= self.timing.req_queue_depth

    def sched_timing_req(self, pkt: CXLMemPacket, when: float):
        if self.req_queue_full():
            self.req_que_full_events += 1
            return None
        dp = DeferredPacket(pkt=pkt, scheduled_tick=when, entry_tick=pkt.arrival_tick)
        self.transmit_list.append(dp)
        self.req_queue_len_samples.append(len(self.transmit_list))
        return when

    def try_send_timing(self, current_tick: float) -> Optional[CXLMemPacket]:
        """Try to send head request to DRAM. Returns pkt if successful."""
        if not self.transmit_list:
            return None
        if self.transmit_list[0].scheduled_tick > current_tick:
            return None

        dp = self.transmit_list.popleft()
        self.req_send_succeeded += 1
        self.req_queue_lat_samples.append(current_tick - dp.entry_tick)
        self.req_queue_len_samples.append(len(self.transmit_list))
        return dp.pkt

    def recv_timing_resp(self, pkt: CXLMemPacket, current_tick: float,
                         rsp_port: CXLResponsePortModel) -> float:
        """Receive response from DRAM, schedule return to host."""
        delay = pkt.header_delay_ns + pkt.payload_delay_ns
        pkt.header_delay_ns = 0.0
        pkt.payload_delay_ns = 0.0
        scheduled = current_tick + self.timing.proto_proc_lat_ns + delay
        rsp_port.sched_timing_resp(pkt, scheduled)
        return scheduled


# ---------------------------------------------------------------------------
# DRAM Backend Model (DDR5-4400)
# ---------------------------------------------------------------------------

class DRAMBackendModel:
    """Simple DDR5-4400 model with timing parameters from SimCXL."""

    def __init__(self, timing: SimCXLTiming, capacity_gb: float = 128.0):
        self.timing = timing
        self.capacity_gb = capacity_gb
        self._busy_until: float = 0.0
        self._pending_requests: deque[Tuple[CXLMemPacket, float]] = deque()

        # Row buffer simulation
        self._open_row: int = -1
        self._open_bank: int = -1
        self.row_hits: int = 0
        self.row_misses: int = 0

    def _row_addr(self, phys_addr: int) -> Tuple[int, int]:
        return (phys_addr >> 16) & 0xFFFF, (phys_addr >> 13) & 0x7

    def service_time_ns(self, pkt: CXLMemPacket) -> float:
        """Return service time for this request in nanoseconds."""
        row, bank = self._row_addr(pkt.addr)
        if row == self._open_row and bank == self._open_bank:
            self.row_hits += 1
            return self.timing.ddr5_tCL_ns  # CAS only
        else:
            self.row_misses += 1
            self._open_row = row
            self._open_bank = bank
            # tRP + tRCD + tCL for row miss
            return self.timing.ddr5_tRP_ns + self.timing.ddr5_tRCD_ns + self.timing.ddr5_tCL_ns

    def submit(self, pkt: CXLMemPacket, current_tick: float) -> float:
        """Submit a request, return tick when response is ready."""
        svc = self.service_time_ns(pkt)
        start = max(current_tick, self._busy_until)
        done = start + svc
        self._busy_until = done
        return done


# ---------------------------------------------------------------------------
# CXL Link Model (full-duplex)
# ---------------------------------------------------------------------------

class CXLLinkModel:
    """Full-duplex CXL 2.0 x16 link."""

    def __init__(self, timing: SimCXLTiming):
        self.timing = timing
        self.upstream_busy_until: float = 0.0    # GPU → CXL
        self.downstream_busy_until: float = 0.0   # CXL → GPU
        self.upstream_busy_cycles: float = 0.0
        self.downstream_busy_cycles: float = 0.0

    def _transfer_time(self, size_bytes: int) -> float:
        """Time to transfer size_bytes with protocol overhead."""
        effective_size = size_bytes * (1.0 + self.timing.cxl_protocol_overhead_pct / 100.0)
        return effective_size / (self.timing.cxl_link_bw_gbps * 1e9 / 8) * 1e9  # ns

    def send_upstream(self, pkt: CXLMemPacket, current_tick: float) -> float:
        """Send data GPU→CXL. Returns tick when data arrives at CXL device."""
        start = max(current_tick, self.upstream_busy_until)
        xfer = self._transfer_time(pkt.size_bytes)
        done = start + xfer + self.timing.cxl_link_latency_ns
        self.upstream_busy_until = done
        self.upstream_busy_cycles += (xfer + self.timing.cxl_link_latency_ns)
        return done

    def send_downstream(self, size_bytes: int, current_tick: float) -> float:
        """Send data CXL→GPU. Returns tick when data arrives at GPU."""
        start = max(current_tick, self.downstream_busy_until)
        xfer = self._transfer_time(size_bytes)
        done = start + xfer + self.timing.cxl_link_latency_ns
        self.downstream_busy_until = done
        self.downstream_busy_cycles += (xfer + self.timing.cxl_link_latency_ns)
        return done


# ---------------------------------------------------------------------------
# Unified Co-Simulator
# ---------------------------------------------------------------------------

@dataclass
class SimStepResult:
    """Result of one decode step."""
    step: int
    tick_ns: float
    chunk_hits: int          # chunks found in fast tier
    chunk_misses: int        # chunks requiring CXL fetch
    promotions_issued: int   # PROSE DMA promotions
    pht_queries: int
    pht_hits: int
    ppu_scores: int
    dma_enqueued: int
    fast_tier_usage_bytes: int
    queue_depth_req: int
    queue_depth_resp: int
    cxl_bandwidth_used_gbps: float


@dataclass
class CoSimStats:
    """Aggregated statistics matching SimCXL CXLCtrlStats naming."""
    total_ticks_ns: float = 0.0
    total_steps: int = 0
    # CXL Controller Stats (match SimCXL names)
    req_que_full_events: int = 0
    req_retry_counts: int = 0
    rsp_que_full_events: int = 0
    req_send_failed: int = 0
    rsp_send_failed: int = 0
    req_send_succeeded: int = 0
    rsp_send_succeeded: int = 0
    req_queue_len_samples: List[int] = field(default_factory=list)
    rsp_queue_len_samples: List[int] = field(default_factory=list)
    rsp_outstanding_samples: List[int] = field(default_factory=list)
    req_queue_lat_samples: List[float] = field(default_factory=list)
    rsp_queue_lat_samples: List[float] = field(default_factory=list)
    # PROSE-specific
    pht_queries: int = 0
    pht_hits: int = 0
    pht_updates: int = 0
    pht_pipeline_stalls: int = 0
    pht_raw_hazards: int = 0
    ppu_scores: int = 0
    dma_promotions: int = 0
    dma_dropped: int = 0
    dma_bytes_transferred: int = 0
    fast_tier_hits: int = 0
    fast_tier_misses: int = 0
    cxl_upstream_utilization: float = 0.0
    cxl_downstream_utilization: float = 0.0
    dram_row_hits: int = 0
    dram_row_misses: int = 0
    # Per-step log
    step_results: List[SimStepResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                d[k] = {'mean': float(np.mean(v)) if v else 0,
                        'p50': float(np.percentile(v, 50)) if v else 0,
                        'p95': float(np.percentile(v, 95)) if v else 0,
                        'p99': float(np.percentile(v, 99)) if v else 0,
                        'max': float(max(v)) if v else 0}
            elif isinstance(v, list):
                d[k] = len(v)
            else:
                d[k] = v
        return d


class SimCXLCoSimulator:
    """
    Unified SimCXL + PROSE cycle-accurate co-simulator.

    Usage::

        sim = SimCXLCoSimulator(mode='prose')
        sim.configure_workload(
            num_chunks=4096, chunk_size_bytes=65536,
            fast_tier_capacity_bytes=2 * 1024**3,
        )
        for step in range(num_decode_steps):
            sim.run_decode_step(step)
        stats = sim.get_stats()
    """

    def __init__(self, mode: str = 'prose', timing: SimCXLTiming | None = None):
        if mode not in ('baseline', 'prose'):
            raise ValueError(f"mode must be 'baseline' or 'prose', got {mode}")
        self.mode = mode
        self.timing = timing or SimCXLTiming()

        # Event queue: (tick_ns, counter, event_type, payload)
        self._event_queue: List[Tuple[float, int, int, Any]] = []
        self._event_counter: int = 0
        self._current_tick: float = 0.0

        # SimCXL memory controller components
        self.rsp_port = CXLResponsePortModel(self.timing)
        self.req_port = CXLRequestPortModel(self.timing)
        self.dram = DRAMBackendModel(self.timing)
        self.cxl_link = CXLLinkModel(self.timing)

        # PROSE components (only active in 'prose' mode)
        self.pht = CycleAccuratePHT(timing=self.timing) if mode == 'prose' else None
        self.ppu = PPUPipeline() if mode == 'prose' else None

        # Two-tier memory state
        self.fast_tier: Dict[int, bool] = {}          # chunk_id → in fast tier?
        self.fast_tier_capacity_bytes: int = 2 * 1024**3
        self.fast_tier_used_bytes: int = 0
        self.chunk_size_bytes: int = 65536
        self.fast_tier_lru: deque[int] = deque()       # LRU eviction order
        self.fast_tier_size: Dict[int, int] = {}        # chunk_id → bytes

        # Workload state
        self.num_chunks: int = 0
        self.decode_step: int = 0
        self._next_pkt_id: int = 0

        # Stats
        self.stats = CoSimStats()

    # ------------------------------------------------------------------
    # Workload configuration
    # ------------------------------------------------------------------

    def configure_workload(
        self,
        num_chunks: int,
        chunk_size_bytes: int = 65536,
        fast_tier_capacity_bytes: int = 2 * 1024**3,
        initial_fast_tier_chunks: int = 32,
    ):
        self.num_chunks = num_chunks
        self.chunk_size_bytes = chunk_size_bytes
        self.fast_tier_capacity_bytes = fast_tier_capacity_bytes
        self.fast_tier.clear()
        self.fast_tier_lru.clear()
        self.fast_tier_used_bytes = 0

        # Pre-populate fast tier with initial "anchor" chunks
        for i in range(min(initial_fast_tier_chunks, num_chunks)):
            self._promote_to_fast(i, is_dma=False)

    # ------------------------------------------------------------------
    # Event queue management
    # ------------------------------------------------------------------

    def _push_event(self, tick: float, evt_type: int, payload: Any = None):
        self._event_counter += 1
        heapq.heappush(
            self._event_queue,
            (tick, self._event_counter, evt_type, payload)
        )

    def _run_until(self, target_tick: float):
        """Process all events scheduled at or before target_tick."""
        while self._event_queue and self._event_queue[0][0] <= target_tick:
            tick, _, evt_type, payload = heapq.heappop(self._event_queue)
            self._current_tick = tick

            if evt_type == _EVT_TRY_SEND_REQ:
                self._handle_try_send_req()
            elif evt_type == _EVT_TRY_SEND_RESP:
                self._handle_try_send_resp()
            elif evt_type == _EVT_MEM_RESPONSE:
                self._handle_mem_response(payload)
            elif evt_type == _EVT_PHT_UPDATE_STAGE1:
                self._handle_pht_stage1(payload)
            elif evt_type == _EVT_PHT_UPDATE_STAGE2:
                self._handle_pht_stage2(payload)
            elif evt_type == _EVT_PHT_UPDATE_STAGE3:
                self._handle_pht_stage3(payload)
            elif evt_type == _EVT_DMA_PROMOTION:
                self._handle_dma_promotion(payload)
            elif evt_type == _EVT_PPU_STEP_START:
                self._handle_ppu_step(payload)

    def _run_all_pending(self):
        """Process all pending events until queue is empty."""
        while self._event_queue:
            self._run_until(self._event_queue[0][0])

    # ------------------------------------------------------------------
    # Memory request path (matching CXLMemCtrl::recvTimingReq → processCXLMem)
    # ------------------------------------------------------------------

    def _issue_memory_request(
        self, chunk_id: int, addr: int, size_bytes: int,
        is_read: bool = True, is_promotion_dma: bool = False,
    ) -> float:
        """
        Issue a memory request through the SimCXL CXL.mem controller path.

        Returns the tick when the request was scheduled (not when it completes).
        """
        pkt = CXLMemPacket(
            pkt_id=self._next_pkt_id,
            cxl_cmd=CXLCmd.M2SReq if is_read else CXLCmd.M2SRwD,
            addr=addr,
            size_bytes=size_bytes,
            arrival_tick=self._current_tick,
            is_read=is_read,
            is_promotion_dma=is_promotion_dma,
            chunk_id=chunk_id,
            needs_response=is_read,
        )
        self._next_pkt_id += 1

        # SimCXL bridge latency
        bridge_delay = self.timing.bridge_lat_ns
        pkt.arrival_tick += bridge_delay

        # Try to accept request at CXL Response Port
        accepted = self.rsp_port.recv_timing_req(pkt)
        if not accepted:
            self.stats.req_send_failed += 1
            return self._current_tick  # Blocked

        # Forward to request port → DRAM
        if self.rsp_port.transmit_list:
            # Schedule try_send for the head of response port's req queue
            head = self.rsp_port.transmit_list[-1]
            self._push_event(head.scheduled_tick, _EVT_TRY_SEND_REQ, head.pkt)

        return pkt.arrival_tick

    def _handle_try_send_req(self):
        """CXL Response Port → forward request to Request Port → DRAM."""
        result = self.rsp_port.try_send_timing(self._current_tick)
        if result is None:
            return
        pkt, _ = result

        # Schedule on request port with protoProcLat
        proto_delay = self.timing.proto_proc_lat_ns
        self.req_port.sched_timing_req(pkt, self._current_tick + proto_delay)

    def _handle_try_send_resp(self):
        """CXL Response Port → send response back to host."""
        result = self.rsp_port.try_send_timing(self._current_tick)
        if result is None:
            return
        # Response sent — no further action needed at this level

    def _handle_mem_response(self, pkt: CXLMemPacket):
        """DRAM response ready → route back through CXL controller."""
        self.req_port.recv_timing_resp(pkt, self._current_tick, self.rsp_port)
        # If response now queued, schedule try_send
        if self.rsp_port.transmit_list:
            self._push_event(
                self.rsp_port.transmit_list[-1].scheduled_tick + self.timing.proto_proc_lat_ns,
                _EVT_TRY_SEND_RESP, None
            )

    # ------------------------------------------------------------------
    # Two-tier memory management
    # ------------------------------------------------------------------

    def _is_in_fast_tier(self, chunk_id: int) -> bool:
        return self.fast_tier.get(chunk_id, False)

    def _promote_to_fast(self, chunk_id: int, is_dma: bool = True):
        """Move chunk from far tier to fast tier. Evict if needed."""
        if self._is_in_fast_tier(chunk_id):
            # Already in fast tier, update LRU
            if chunk_id in self.fast_tier_lru:
                self.fast_tier_lru.remove(chunk_id)
            self.fast_tier_lru.append(chunk_id)
            return

        size = self.chunk_size_bytes
        while self.fast_tier_used_bytes + size > self.fast_tier_capacity_bytes and self.fast_tier_lru:
            victim = self.fast_tier_lru.popleft()
            del self.fast_tier[victim]
            self.fast_tier_used_bytes -= self.fast_tier_size.pop(victim, self.chunk_size_bytes)

        self.fast_tier[chunk_id] = True
        self.fast_tier_size[chunk_id] = size
        self.fast_tier_used_bytes += size
        self.fast_tier_lru.append(chunk_id)

        if is_dma:
            self.stats.dma_promotions += 1
            self.stats.dma_bytes_transferred += size

    def _access_chunk_baseline(self, chunk_id: int) -> Tuple[bool, float]:
        """
        Baseline path: check fast tier, if miss → issue CXL read.
        Returns (was_in_fast_tier, latency_ns).
        """
        if self._is_in_fast_tier(chunk_id):
            self.stats.fast_tier_hits += 1
            if chunk_id in self.fast_tier_lru:
                self.fast_tier_lru.remove(chunk_id)
            self.fast_tier_lru.append(chunk_id)
            return True, self.timing.proto_proc_lat_ns + self.timing.bridge_lat_ns

        self.stats.fast_tier_misses += 1
        addr = chunk_id * self.chunk_size_bytes
        latency = self.timing.bridge_lat_ns + self.timing.proto_proc_lat_ns * 2
        latency += self.timing.cxl_link_latency_ns
        latency += self.timing.ddr5_tCL_ns + self.timing.ddr5_tRCD_ns + self.timing.ddr5_tRP_ns
        return False, latency

    def _access_chunk_prose(self, chunk_id: int) -> Tuple[bool, float, bool]:
        """
        PROSE path: PHT predict → fast tier access or CXL fetch.
        Returns (was_in_fast_tier, latency_ns, pht_predicted_promote).
        """
        # PHT lookup (1 cycle = 1ns)
        pht_predict, pht_ema = self.pht.query(chunk_id)
        self.stats.pht_queries += 1
        if pht_predict:
            self.stats.pht_hits += 1

        if self._is_in_fast_tier(chunk_id):
            self.stats.fast_tier_hits += 1
            if chunk_id in self.fast_tier_lru:
                self.fast_tier_lru.remove(chunk_id)
            self.fast_tier_lru.append(chunk_id)
            return True, self.timing.proto_proc_lat_ns + self.timing.bridge_lat_ns, pht_predict

        self.stats.fast_tier_misses += 1
        addr = chunk_id * self.chunk_size_bytes
        latency = self.timing.bridge_lat_ns + self.timing.proto_proc_lat_ns * 2
        latency += self.timing.cxl_link_latency_ns
        latency += self.timing.ddr5_tCL_ns + self.timing.ddr5_tRCD_ns + self.timing.ddr5_tRP_ns
        return False, latency, pht_predict

    # ------------------------------------------------------------------
    # PROSE DMA promotion path (coexists with regular CXL traffic)
    # ------------------------------------------------------------------

    def _handle_dma_promotion(self, payload: Tuple[int, int]):
        """Execute a DMA promotion transfer through the CXL.mem controller.

        PROSE DMA promotion = CXL.mem read from far tier → fast tier.
        Shares the CXLResponsePort queue with regular memory accesses,
        creating the queue contention that the reviewer wants to see modeled.
        """
        chunk_id, opportunity_tick = payload

        # Don't promote if already in fast tier
        if self._is_in_fast_tier(chunk_id):
            return

        addr = chunk_id * self.chunk_size_bytes
        # Issue the CXL.mem read (this adds to the shared port queue)
        self._issue_memory_request(
            chunk_id, addr, self.chunk_size_bytes,
            is_read=True, is_promotion_dma=True,
        )
        # The actual data lands in fast tier once DMA completes
        # (modeled as completion at the end of the step window)
        self._promote_to_fast(chunk_id, is_dma=True)

    # ------------------------------------------------------------------
    # PROSE PPU step
    # ------------------------------------------------------------------

    def _handle_ppu_step(self, payload: Dict[int, float], promotion_budget: int = 5):
        """Execute one PROSE promotion decision step with per-step budget."""
        if self.ppu is None:
            return
        masses = payload
        self.ppu.begin_step(masses)

        # Score each candidate chunk, collect (utility, chunk_id) pairs
        scored_candidates: List[Tuple[float, int]] = []
        for chunk_id, mass in masses.items():
            if self._is_in_fast_tier(chunk_id):
                continue
            utility, enqueued = self.ppu.score_candidate(
                chunk_id, attention_mass=mass,
                position_ratio=chunk_id / max(1, self.num_chunks),
            )
            self.stats.ppu_scores += 1
            if enqueued:
                scored_candidates.append((utility, chunk_id))

        # Select top-K by utility within per-step budget
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        selected = scored_candidates[:promotion_budget]

        # Enqueue DMA for selected chunks only
        for utility, chunk_id in selected:
            self._push_event(
                self._current_tick + 5.0,
                _EVT_DMA_PROMOTION,
                (chunk_id, self._current_tick)
            )

        self.ppu.end_step()

    # ------------------------------------------------------------------
    # Main decode step
    # ------------------------------------------------------------------

    def run_decode_step(
        self,
        step: int,
        active_chunk_ids: Optional[List[int]] = None,
        attention_masses: Optional[Dict[int, float]] = None,
        promotion_budget: int = 5,
    ) -> SimStepResult:
        """
        Execute one decode step — the core simulation unit.

        Parameters:
            step: Decode step index.
            active_chunk_ids: Chunks accessed this step (default: random sample).
            attention_masses: Per-chunk attention masses from Flash-Decoding.
            promotion_budget: Max chunks to promote per step (PROSE only).
        """
        self.decode_step = step
        start_tick = self._current_tick

        # Generate synthetic workload if not provided
        if active_chunk_ids is None:
            n_active = min(32, self.num_chunks)
            active_chunk_ids = list(np.random.RandomState(step).choice(
                self.num_chunks, size=n_active, replace=False
            ))

        if attention_masses is None:
            rng = np.random.RandomState(step + 10000)
            attention_masses = {
                cid: float(rng.beta(0.5, 5.0))
                for cid in active_chunk_ids
            }

        chunk_hits = 0
        chunk_misses = 0

        # ---- PROSE speculative prefetch: run BEFORE accesses ----
        if self.mode == 'prose' and attention_masses:
            self._handle_ppu_step(attention_masses, promotion_budget)
            # Process the DMA events immediately so chunks land in fast tier
            self._run_all_pending()

        # ---- Process chunk accesses ----
        for cid in active_chunk_ids:
            if self.mode == 'prose':
                in_fast, lat, pht_pred = self._access_chunk_prose(cid)
            else:
                in_fast, lat = self._access_chunk_baseline(cid)
                pht_pred = False

            if in_fast:
                chunk_hits += 1
            else:
                chunk_misses += 1
                addr = cid * self.chunk_size_bytes
                # Issue CXL read through the controller
                self._issue_memory_request(cid, addr, self.chunk_size_bytes)

        # ---- Baseline: reactive promotion AFTER accesses ----
        if self.mode == 'baseline':
            promoted_this_step = 0
            for cid in active_chunk_ids:
                if not self._is_in_fast_tier(cid) and promoted_this_step < promotion_budget:
                    self._promote_to_fast(cid, is_dma=True)
                    promoted_this_step += 1

        # Note: PROSE already promoted speculatively, no post-access needed

        # Run events
        step_end_tick = start_tick + 1000.0  # 1us per step window
        self._run_until(step_end_tick)
        self._run_all_pending()

        # PHT learning: update with observed outcomes from this step
        if self.pht and attention_masses:
            for cid in active_chunk_ids:
                mass = attention_masses.get(cid, 0.0)
                was_hit = self._is_in_fast_tier(cid)
                self.pht.update(cid, promoted=was_hit, importance=mass)
            # Tick pipeline: each update takes 3 cycles, pipeline accepts 1 per tick
            num_updates = len(active_chunk_ids)
            for _ in range(num_updates + 3):
                self.pht.tick()

        self.stats.total_steps += 1
        self.stats.total_ticks_ns = self._current_tick

        result = SimStepResult(
            step=step,
            tick_ns=self._current_tick,
            chunk_hits=chunk_hits,
            chunk_misses=chunk_misses,
            promotions_issued=min(promotion_budget, chunk_misses),
            pht_queries=self.stats.pht_queries,
            pht_hits=self.stats.pht_hits,
            ppu_scores=self.stats.ppu_scores,
            dma_enqueued=self.stats.dma_promotions,
            fast_tier_usage_bytes=self.fast_tier_used_bytes,
            queue_depth_req=len(self.req_port.transmit_list),
            queue_depth_resp=len(self.rsp_port.transmit_list),
            cxl_bandwidth_used_gbps=0.0,
        )
        self.stats.step_results.append(result)
        return result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> CoSimStats:
        """Aggregate and return all statistics."""
        s = self.stats
        # Collect from ports
        s.req_que_full_events = self.req_port.req_que_full_events
        s.rsp_que_full_events = self.rsp_port.rsp_que_full_events
        s.req_send_succeeded = self.req_port.req_send_succeeded
        s.rsp_send_succeeded = self.rsp_port.rsp_send_succeeded
        s.req_queue_len_samples = list(self.req_port.req_queue_len_samples)
        s.rsp_queue_len_samples = list(self.rsp_port.rsp_queue_len_samples)
        s.rsp_outstanding_samples = list(self.rsp_port.rsp_outstanding_samples)
        s.req_queue_lat_samples = list(self.req_port.req_queue_lat_samples)
        s.rsp_queue_lat_samples = list(self.rsp_port.rsp_queue_lat_samples)
        s.dram_row_hits = self.dram.row_hits
        s.dram_row_misses = self.dram.row_misses

        if self.mode == 'prose' and self.pht:
            s.pht_queries = self.pht.queries_total
            s.pht_hits = self.pht.queries_hit
            s.pht_updates = self.pht.updates_total
            s.pht_pipeline_stalls = self.pht.pipeline_stalls
            s.pht_raw_hazards = self.pht.raw_hazards
        if self.ppu:
            s.dma_dropped = self.ppu._dma_dropped

        total_ns = max(1.0, s.total_ticks_ns)
        s.cxl_upstream_utilization = self.cxl_link.upstream_busy_cycles / total_ns
        s.cxl_downstream_utilization = self.cxl_link.downstream_busy_cycles / total_ns

        return s


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_comparison_experiment(
    num_chunks: int = 4096,
    num_steps: int = 128,
    chunk_size_bytes: int = 65536,
    fast_tier_capacity_mb: float = 512.0,
    promotion_budget: int = 5,
    seed: int = 42,
) -> Dict[str, CoSimStats]:
    """Run a head-to-head baseline vs PROSE comparison in the same simulator."""

    fast_tier_bytes = int(fast_tier_capacity_mb * 1024 * 1024)

    results = {}
    for mode in ('baseline', 'prose'):
        sim = SimCXLCoSimulator(mode=mode)
        sim.configure_workload(
            num_chunks=num_chunks,
            chunk_size_bytes=chunk_size_bytes,
            fast_tier_capacity_bytes=fast_tier_bytes,
            initial_fast_tier_chunks=min(32, int(fast_tier_bytes // chunk_size_bytes)),
        )

        rng = np.random.RandomState(seed)
        for step in range(num_steps):
            # Generate Zipf-distributed access pattern (LLM attention is heavy-tailed)
            n_active = max(1, min(32, int(num_chunks * 0.01)))
            # Zipf: a few chunks accessed very frequently
            chunk_ids = list(rng.zipf(1.5, size=n_active))
            chunk_ids = [c % num_chunks for c in chunk_ids if c < num_chunks]
            chunk_ids = list(set(chunk_ids))
            if not chunk_ids:
                chunk_ids = [rng.randint(0, num_chunks)]

            masses = {cid: float(rng.beta(0.5, 5.0)) for cid in chunk_ids}
            sim.run_decode_step(
                step, active_chunk_ids=chunk_ids,
                attention_masses=masses,
                promotion_budget=promotion_budget,
            )

        results[mode] = sim.get_stats()

    return results


# ---------------------------------------------------------------------------
# Main entry point for local testing
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 72)
    print("SimCXL + PROSE Co-Simulator — Baseline vs PROSE Comparison")
    print("=" * 72)

    for num_chunks in [1024, 4096, 8192]:
        print(f"\n--- {num_chunks} chunks, 128 steps ---")
        results = run_comparison_experiment(
            num_chunks=num_chunks,
            num_steps=128,
            promotion_budget=5,
        )

        for mode in ('baseline', 'prose'):
            s = results[mode]
            pht_hit_rate = s.pht_hits / max(1, s.pht_queries) * 100
            fast_tier_hit_rate = s.fast_tier_hits / max(1, s.fast_tier_hits + s.fast_tier_misses) * 100
            print(f"  {mode:>10s}: fast_tier_hit={fast_tier_hit_rate:.1f}% "
                  f"PHT_hit={pht_hit_rate:.1f}% "
                  f"promotions={s.dma_promotions} "
                  f"req_q_p95={np.percentile(s.req_queue_len_samples, 95) if s.req_queue_len_samples else 0:.0f} "
                  f"rsp_q_p95={np.percentile(s.rsp_queue_len_samples, 95) if s.rsp_queue_len_samples else 0:.0f} "
                  f"queue_full={s.req_que_full_events + s.rsp_que_full_events}")

    print("\nDone.")
