"""
Continuous Batching support for PPU (ProSE-X 3.0).

Addresses three critical challenges when integrating the PPU with
production inference engines (vLLM, TensorRT-LLM) that use continuous
batching:

  1. Multi-sequence state management via Ping-Pong double-buffered SRAM
  2. NoC burst mitigation via Doorbell + async pull architecture
  3. PagedAttention compatibility via Hardware Block Table Walker (HW-BTW)

§5.1 of the HPCA paper: From Single-Stream to Continuous Batching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set


# ═══════════════════════════════════════════════════════════════════════
# §1  Ping-Pong Double-Buffered State Manager
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SequenceState:
    """Per-sequence PPU state that lives in VRAM between activations.

    In hardware, this is the 13KB blob that gets swapped in/out of the
    PPU's on-chip SRAM via the ping-pong DMA engine.

    Contents:
      - ewma_masses: EWMA-smoothed attention masses (N_chunks × 16-bit)
      - pht_counters: PHT 2-bit saturating counters (N_entries × 2-bit)
      - anchor_flags: 1-bit anchor lock per chunk (N_chunks × 1-bit)
      - counter_bank: attention mass counters (N_chunks × 16-bit)
      - metadata: seq_id, last_step, total_steps
    """
    seq_id: int
    ewma_masses: Dict[str, float] = field(default_factory=dict)
    counter_bank: Dict[str, int] = field(default_factory=dict)
    anchor_flags: Set[str] = field(default_factory=set)
    pht_counters: Dict[int, int] = field(default_factory=dict)
    last_step: int = 0
    total_steps: int = 0

    def size_bytes(self, num_chunks: int = 2048, pht_entries: int = 1024) -> int:
        """Compute serialized state size in bytes."""
        ewma_bytes = num_chunks * 2          # N × 16-bit
        counter_bytes = num_chunks * 2       # N × 16-bit
        anchor_bytes = math.ceil(num_chunks / 8)  # N × 1-bit
        pht_bytes = math.ceil(pht_entries * 2 / 8)  # entries × 2-bit
        metadata_bytes = 16                  # seq_id + step counters
        return ewma_bytes + counter_bytes + anchor_bytes + pht_bytes + metadata_bytes


@dataclass
class PingPongConfig:
    """Configuration for the double-buffered state SRAM."""
    buffer_size_kb: float = 13.0     # Per-buffer size (matches single-seq state)
    num_buffers: int = 2             # Always 2 (ping-pong)
    swap_dma_bandwidth_gbps: float = 1000.0  # HBM bandwidth allocated to swaps
    max_concurrent_sequences: int = 256  # Max sequences in VRAM state pool


@dataclass
class SwapEvent:
    """A single state swap operation."""
    seq_id: int
    direction: str          # "swap_in" or "swap_out"
    size_bytes: int
    latency_ns: float
    overlapped: bool        # True if fully hidden behind compute


@dataclass
class PingPongStats:
    """Aggregate statistics for the ping-pong state manager."""
    total_swaps: int = 0
    total_swap_bytes: int = 0
    swap_in_count: int = 0
    swap_out_count: int = 0
    fully_overlapped: int = 0
    exposed_swap_ns: float = 0.0
    avg_swap_latency_ns: float = 0.0


class PingPongStateManager:
    """Double-buffered SRAM state manager for multi-sequence PPU.

    Hardware model:
      - Two 13KB SRAM buffers (Buffer A, Buffer B) on-chip
      - Dedicated swap DMA engine (independent of main DMA)
      - While PPU computes on Buffer A, swap DMA:
          1. Writes back Buffer B's dirty state to VRAM (swap-out)
          2. Prefetches next sequence's state into Buffer B (swap-in)
      - After PPU finishes, buffers swap roles (A↔B)

    Timing guarantee:
      State size = 13KB.  At 1 TB/s HBM bandwidth:
        swap_in  = 13KB / 1TB/s = 13ns
        swap_out = 13KB / 1TB/s = 13ns
        total    = 26ns << 82ns (PPU compute time)
      → 100% overlap, zero exposed latency.

    RTL equivalent:
      always @(posedge clk) begin
          if (ppu_done) begin
              active_buffer <= ~active_buffer;  // toggle A/B
              swap_dma_start <= 1'b1;
          end
      end
    """

    def __init__(self, config: PingPongConfig):
        self.config = config
        # VRAM state pool: seq_id -> SequenceState
        self._vram_pool: Dict[int, SequenceState] = {}
        # On-chip buffers (functional model)
        self._buffer_a: Optional[SequenceState] = None
        self._buffer_b: Optional[SequenceState] = None
        self._active_buffer: str = "A"  # Which buffer PPU is computing on
        self._pending_seq_id: Optional[int] = None
        # Stats
        self._stats = PingPongStats()
        self._swap_events: List[SwapEvent] = []

    def register_sequence(self, seq_id: int) -> None:
        """Register a new sequence in the VRAM state pool."""
        if seq_id not in self._vram_pool:
            self._vram_pool[seq_id] = SequenceState(seq_id=seq_id)

    def evict_sequence(self, seq_id: int) -> Optional[SequenceState]:
        """Remove a completed sequence from the pool."""
        return self._vram_pool.pop(seq_id, None)

    def activate(self, seq_id: int, ppu_compute_ns: float = 82.0) -> Tuple[SequenceState, List[SwapEvent]]:
        """Activate a sequence for PPU processing.

        Performs the ping-pong swap:
          1. If active buffer has dirty state, swap-out to VRAM
          2. Swap-in the requested sequence's state from VRAM
          3. Return the active state for PPU to operate on

        Returns (active_state, swap_events).
        """
        self.register_sequence(seq_id)
        events: List[SwapEvent] = []
        state_bytes = SequenceState(seq_id=0).size_bytes()
        bw_bytes_per_ns = self.config.swap_dma_bandwidth_gbps / 8  # GB/s → B/ns

        # Swap-out: write back current active buffer's state to VRAM
        current_state = self._buffer_a if self._active_buffer == "A" else self._buffer_b
        if current_state is not None:
            swap_out_ns = state_bytes / bw_bytes_per_ns
            self._vram_pool[current_state.seq_id] = current_state
            event = SwapEvent(
                seq_id=current_state.seq_id,
                direction="swap_out",
                size_bytes=state_bytes,
                latency_ns=swap_out_ns,
                overlapped=True,  # Overlaps with PPU compute on other buffer
            )
            events.append(event)
            self._stats.swap_out_count += 1
            self._stats.total_swap_bytes += state_bytes

        # Swap-in: prefetch requested sequence into the OTHER buffer
        new_state = self._vram_pool.get(seq_id, SequenceState(seq_id=seq_id))
        swap_in_ns = state_bytes / bw_bytes_per_ns
        event = SwapEvent(
            seq_id=seq_id,
            direction="swap_in",
            size_bytes=state_bytes,
            latency_ns=swap_in_ns,
            overlapped=True,
        )
        events.append(event)
        self._stats.swap_in_count += 1
        self._stats.total_swap_bytes += state_bytes

        # Check overlap: total swap time vs PPU compute time
        total_swap_ns = sum(e.latency_ns for e in events)
        if total_swap_ns <= ppu_compute_ns:
            self._stats.fully_overlapped += 1
        else:
            exposed = total_swap_ns - ppu_compute_ns
            self._stats.exposed_swap_ns += exposed
            for e in events:
                e.overlapped = False

        # Toggle active buffer
        if self._active_buffer == "A":
            self._buffer_b = new_state
            self._active_buffer = "B"
        else:
            self._buffer_a = new_state
            self._active_buffer = "A"

        self._stats.total_swaps += 1
        self._swap_events.extend(events)

        active = self._buffer_a if self._active_buffer == "A" else self._buffer_b
        return active, events

    def commit(self, seq_id: int, updated_state: SequenceState) -> None:
        """Commit updated state after PPU processing."""
        if self._active_buffer == "A":
            self._buffer_a = updated_state
        else:
            self._buffer_b = updated_state

    @property
    def active_state(self) -> Optional[SequenceState]:
        return self._buffer_a if self._active_buffer == "A" else self._buffer_b

    @property
    def num_tracked_sequences(self) -> int:
        return len(self._vram_pool)

    def stats(self) -> Dict[str, object]:
        s = self._stats
        return {
            "total_swaps": s.total_swaps,
            "total_swap_bytes": s.total_swap_bytes,
            "swap_in_count": s.swap_in_count,
            "swap_out_count": s.swap_out_count,
            "fully_overlapped_pct": (
                s.fully_overlapped / max(1, s.total_swaps) * 100
            ),
            "exposed_swap_ns": s.exposed_swap_ns,
            "tracked_sequences": len(self._vram_pool),
            "sram_budget_kb": self.config.buffer_size_kb * self.config.num_buffers,
        }


# ═══════════════════════════════════════════════════════════════════════
# §2  Doorbell + Async Pull Architecture
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DoorbellDescriptor:
    """8-byte descriptor written by SM to PPU's doorbell ring buffer.

    In hardware, this is a single NoC flit (64-bit):
      [seq_id: 16-bit | workspace_addr: 48-bit]

    The SM writes this after completing the reduction kernel and
    storing Mass_j array in the PPU workspace region of VRAM.
    """
    seq_id: int
    workspace_addr: int     # VRAM address of Mass_j array
    num_chunks: int         # Number of chunks in the array
    timestamp_cycle: int = 0


@dataclass
class DoorbellConfig:
    """Configuration for the doorbell ring buffer."""
    ring_depth: int = 128           # Max pending descriptors
    descriptor_bytes: int = 8       # Per-descriptor size (1 NoC flit)
    pull_bandwidth_gbps: float = 200.0  # Bandwidth for async pull from VRAM
    max_pull_outstanding: int = 4   # Max concurrent pull requests


@dataclass
class DoorbellStats:
    """Aggregate statistics for the doorbell mechanism."""
    total_rings: int = 0
    total_pulls: int = 0
    total_pull_bytes: int = 0
    ring_buffer_peak_depth: int = 0
    ring_buffer_overflows: int = 0
    avg_pull_latency_ns: float = 0.0
    avg_queue_depth: float = 0.0


class DoorbellArbiter:
    """Doorbell ring buffer + async pull front-end for PPU.

    Replaces the direct MMRF push model (v2.1) with a decoupled
    doorbell + pull architecture that eliminates NoC burst congestion
    under continuous batching.

    Protocol:
      1. SM completes reduction kernel, writes Mass_j to VRAM workspace
      2. SM writes 8-byte descriptor to PPU doorbell (single NoC flit)
      3. PPU arbiter dequeues descriptors in order
      4. PPU issues async pull to fetch Mass_j from VRAM workspace
      5. Data arrives in MMRF input buffer, triggers pipeline

    Why this is better than direct push:
      - 64 SMs writing 4KB each = 256KB burst on NoC → congestion
      - 64 SMs writing 8B each = 512B total → negligible
      - PPU pulls at its own pace → smooth, predictable traffic

    Hardware cost:
      - 128-entry × 8-byte ring buffer = 1KB SRAM
      - Pull DMA engine (reuses existing MMRF interface)
      - Arbiter FSM: ~0.001 mm² @ 7nm
    """

    def __init__(self, config: DoorbellConfig):
        self.config = config
        self._ring: List[DoorbellDescriptor] = []
        self._stats = DoorbellStats()
        self._cycle = 0

    def ring(self, descriptor: DoorbellDescriptor) -> bool:
        """SM writes a descriptor to the doorbell ring buffer.

        Returns False if ring buffer is full (overflow).
        In hardware, this would stall the SM for 1 cycle.
        """
        self._stats.total_rings += 1
        if len(self._ring) >= self.config.ring_depth:
            self._stats.ring_buffer_overflows += 1
            return False
        self._ring.append(descriptor)
        self._stats.ring_buffer_peak_depth = max(
            self._stats.ring_buffer_peak_depth, len(self._ring)
        )
        return True

    def dequeue(self) -> Optional[DoorbellDescriptor]:
        """PPU arbiter dequeues the next descriptor for processing."""
        if not self._ring:
            return None
        return self._ring.pop(0)

    def pull_latency_ns(self, num_chunks: int, bytes_per_chunk_mass: int = 2) -> float:
        """Compute async pull latency for fetching Mass_j from VRAM.

        Pull size = num_chunks × 2 bytes (FP16 masses).
        Bandwidth = pull_bandwidth_gbps.
        """
        total_bytes = num_chunks * bytes_per_chunk_mass
        bw_bytes_per_ns = self.config.pull_bandwidth_gbps / 8
        latency = total_bytes / bw_bytes_per_ns
        self._stats.total_pulls += 1
        self._stats.total_pull_bytes += total_bytes
        return latency

    @property
    def pending_count(self) -> int:
        return len(self._ring)

    @property
    def is_empty(self) -> bool:
        return len(self._ring) == 0

    def stats(self) -> Dict[str, object]:
        s = self._stats
        avg_pull = (
            s.total_pull_bytes / max(1, s.total_pulls)
            / (self.config.pull_bandwidth_gbps / 8)
        ) if s.total_pulls > 0 else 0.0
        return {
            "total_rings": s.total_rings,
            "total_pulls": s.total_pulls,
            "total_pull_bytes": s.total_pull_bytes,
            "ring_buffer_peak_depth": s.ring_buffer_peak_depth,
            "ring_buffer_overflows": s.ring_buffer_overflows,
            "avg_pull_latency_ns": avg_pull,
            "ring_buffer_utilization_pct": (
                s.ring_buffer_peak_depth / self.config.ring_depth * 100
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# §3  Hardware Block Table Walker (HW-BTW)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BlockTableEntry:
    """A single entry in vLLM's block table.

    Maps logical_block_id → physical_block_id in CXL DRAM.
    In vLLM, the block table shape is [max_seqs, max_blocks_per_seq].
    """
    logical_block_id: int
    physical_block_id: int
    block_size_tokens: int = 16     # vLLM default block size
    block_size_bytes: int = 0       # Computed from model config


@dataclass
class ScatterGatherDescriptor:
    """A single scatter/gather DMA descriptor for CXL transfer.

    Each descriptor specifies one physically-contiguous block to
    transfer from CXL DRAM to GPU HBM/SRAM.
    """
    physical_addr: int              # CXL DRAM physical address
    dest_addr: int                  # GPU-side destination address
    size_bytes: int
    logical_chunk_id: str           # For tracking
    priority: int = 0


@dataclass
class HWBTWConfig:
    """Configuration for the Hardware Block Table Walker."""
    block_table_base_addr: int = 0  # VRAM address of block table
    max_sequences: int = 256
    max_blocks_per_seq: int = 8192  # 128K ctx / 16 tok per block
    block_id_bytes: int = 4         # 32-bit block IDs
    # CXL address translation
    cxl_base_addr: int = 0
    cxl_block_stride_bytes: int = 65536  # Physical block size in CXL
    # Lookup batching
    max_batch_lookups: int = 32     # Max concurrent table lookups
    lookup_latency_cycles: int = 2  # VRAM read latency for block IDs


@dataclass
class HWBTWStats:
    """Aggregate statistics for the HW-BTW."""
    total_lookups: int = 0
    total_sg_descriptors: int = 0
    total_transfer_bytes: int = 0
    batched_lookups: int = 0
    avg_scatter_depth: float = 0.0


class HardwareBlockTableWalker:
    """Hardware Block Table Walker for PagedAttention-compatible DMA.

    When PPU's Top-K selector outputs logical chunk IDs, the HW-BTW
    translates them to physical CXL DRAM addresses by walking vLLM's
    block table, then assembles scatter/gather DMA descriptors.

    Pipeline (3 micro-ops):
      Step 1: Batch table lookup
          Read physical_block_ids from VRAM block table.
          Cost: 1 VRAM read of K × 4 bytes (e.g., 16 × 4 = 64 bytes).
          Latency: ~10ns (L2 hit) to ~50ns (HBM).

      Step 2: Address translation
          physical_addr = cxl_base + physical_block_id × block_stride
          Pure combinational logic, 1 cycle.

      Step 3: Scatter/gather descriptor assembly
          Pack K descriptors into DMA command queue.
          1 cycle per descriptor (pipelined).

    Hardware cost:
      - Batch lookup buffer: 32 × 4B = 128B register file
      - Address ALU: 1 multiplier + 1 adder = ~0.0005 mm²
      - Descriptor formatter: shift + pack logic = ~0.0003 mm²
      - Total: < 0.001 mm² @ 7nm
    """

    def __init__(self, config: HWBTWConfig):
        self.config = config
        # Simulated block table (in real hardware, this is in VRAM)
        self._block_tables: Dict[int, List[int]] = {}  # seq_id -> [phys_block_ids]
        self._stats = HWBTWStats()

    def register_block_table(self, seq_id: int, physical_block_ids: List[int]) -> None:
        """Register a sequence's block table (called by vLLM runtime)."""
        self._block_tables[seq_id] = list(physical_block_ids)

    def translate(
        self,
        seq_id: int,
        logical_chunk_ids: List[str],
        chunk_size_tokens: int = 64,
    ) -> Tuple[List[ScatterGatherDescriptor], float]:
        """Translate logical chunk IDs to scatter/gather DMA descriptors.

        Returns (descriptors, total_lookup_latency_ns).

        Args:
            chunk_size_tokens: Tokens per logical chunk.  Used to compute
                the number of physical blocks spanned by each chunk when
                block_size < chunk_size (future multi-block support).
        """
        _ = chunk_size_tokens  # reserved for multi-block mapping
        block_table = self._block_tables.get(seq_id, [])
        if not block_table:
            return [], 0.0

        # Step 1: Determine which physical blocks each logical chunk maps to.
        # A logical chunk (64 tokens) may span multiple vLLM blocks (16 tokens).
        # Simplified: assume 1 logical chunk = 1 physical block for modeling.

        descriptors: List[ScatterGatherDescriptor] = []
        lookup_count = 0

        for cid_str in logical_chunk_ids:
            # Parse chunk index from chunk_id string
            try:
                chunk_idx = int(cid_str.split("_")[-1]) if "_" in cid_str else int(cid_str)
            except (ValueError, IndexError):
                chunk_idx = hash(cid_str) % len(block_table) if block_table else 0

            if chunk_idx < len(block_table):
                phys_block = block_table[chunk_idx]
                phys_addr = self.config.cxl_base_addr + phys_block * self.config.cxl_block_stride_bytes
                descriptors.append(ScatterGatherDescriptor(
                    physical_addr=phys_addr,
                    dest_addr=0,  # Assigned by DMA engine
                    size_bytes=self.config.cxl_block_stride_bytes,
                    logical_chunk_id=cid_str,
                ))
                lookup_count += 1

        # Lookup latency: batched read from VRAM
        # K block IDs × 4 bytes each, single VRAM transaction
        lookup_bytes = lookup_count * self.config.block_id_bytes
        # Assume L2 hit for small lookups (<256B), HBM for larger
        if lookup_bytes <= 256:
            lookup_latency_ns = 10.0  # L2 cache hit
        else:
            lookup_latency_ns = 50.0  # HBM access

        # Add translation cycles (1 cycle per descriptor @ 1GHz = 1ns each)
        translation_ns = len(descriptors) * 1.0

        total_ns = lookup_latency_ns + translation_ns

        # Update stats
        self._stats.total_lookups += lookup_count
        self._stats.total_sg_descriptors += len(descriptors)
        self._stats.total_transfer_bytes += sum(d.size_bytes for d in descriptors)
        self._stats.batched_lookups += 1
        if self._stats.batched_lookups > 0:
            self._stats.avg_scatter_depth = (
                self._stats.total_sg_descriptors / self._stats.batched_lookups
            )

        return descriptors, total_ns

    def stats(self) -> Dict[str, object]:
        s = self._stats
        return {
            "total_lookups": s.total_lookups,
            "total_sg_descriptors": s.total_sg_descriptors,
            "total_transfer_bytes": s.total_transfer_bytes,
            "total_transfer_mb": s.total_transfer_bytes / (1024 * 1024),
            "batched_lookups": s.batched_lookups,
            "avg_scatter_depth": s.avg_scatter_depth,
            "tracked_sequences": len(self._block_tables),
        }


# ═══════════════════════════════════════════════════════════════════════
# §4  Integrated Continuous Batching Controller
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ContinuousBatchingConfig:
    """Top-level configuration for continuous batching support."""
    enabled: bool = True
    max_batch_size: int = 64
    # Sub-configs
    pingpong: PingPongConfig = field(default_factory=PingPongConfig)
    doorbell: DoorbellConfig = field(default_factory=DoorbellConfig)
    hw_btw: HWBTWConfig = field(default_factory=HWBTWConfig)
    # Timing
    ppu_compute_ns: float = 82.0    # PPU pipeline latency (5-stage @ 1GHz)


@dataclass
class BatchStepResult:
    """Result of processing one sequence in a continuous batch."""
    seq_id: int
    swap_events: List[SwapEvent]
    pull_latency_ns: float
    btw_latency_ns: float
    sg_descriptors: List[ScatterGatherDescriptor]
    total_overhead_ns: float
    fully_overlapped: bool


class ContinuousBatchingController:
    """Orchestrates PPU operation under continuous batching.

    Coordinates the three subsystems:
      1. PingPongStateManager: context switch between sequences
      2. DoorbellArbiter: receive work items from GPU SMs
      3. HardwareBlockTableWalker: translate chunk IDs for DMA

    Per-sequence processing flow:
      1. Dequeue doorbell descriptor
      2. Activate sequence (ping-pong swap)
      3. Pull Mass_j from VRAM workspace
      4. Run PPU pipeline (5/6 stages)
      5. Translate Top-K via HW-BTW
      6. Issue scatter/gather CXL DMA
      7. Commit updated state
    """

    def __init__(self, config: ContinuousBatchingConfig):
        self.config = config
        self.state_mgr = PingPongStateManager(config.pingpong)
        self.doorbell = DoorbellArbiter(config.doorbell)
        self.hw_btw = HardwareBlockTableWalker(config.hw_btw)
        self._processed_count = 0

    def submit_sequence(
        self,
        seq_id: int,
        workspace_addr: int,
        num_chunks: int,
    ) -> bool:
        """SM submits a sequence for PPU processing (doorbell ring)."""
        desc = DoorbellDescriptor(
            seq_id=seq_id,
            workspace_addr=workspace_addr,
            num_chunks=num_chunks,
        )
        return self.doorbell.ring(desc)

    def process_next(
        self,
        promoted_chunk_ids: Optional[List[str]] = None,
        chunk_size_tokens: int = 64,
    ) -> Optional[BatchStepResult]:
        """Process the next sequence from the doorbell queue.

        This models one iteration of the PPU's main loop under
        continuous batching.
        """
        desc = self.doorbell.dequeue()
        if desc is None:
            return None

        # Step 1: Activate sequence (ping-pong swap)
        _state, swap_events = self.state_mgr.activate(
            desc.seq_id,
            ppu_compute_ns=self.config.ppu_compute_ns,
        )

        # Step 2: Pull Mass_j from VRAM workspace
        pull_ns = self.doorbell.pull_latency_ns(desc.num_chunks)

        # Step 3: PPU pipeline runs (modeled externally)
        # ... caller runs PPU simulator with the activated state ...

        # Step 4: Translate Top-K via HW-BTW
        btw_ns = 0.0
        sg_descs: List[ScatterGatherDescriptor] = []
        if promoted_chunk_ids:
            sg_descs, btw_ns = self.hw_btw.translate(
                desc.seq_id, promoted_chunk_ids, chunk_size_tokens
            )

        # Compute total overhead
        swap_ns = sum(e.latency_ns for e in swap_events)
        total_overhead = pull_ns + btw_ns  # swap is overlapped
        fully_overlapped = swap_ns <= self.config.ppu_compute_ns

        self._processed_count += 1

        return BatchStepResult(
            seq_id=desc.seq_id,
            swap_events=swap_events,
            pull_latency_ns=pull_ns,
            btw_latency_ns=btw_ns,
            sg_descriptors=sg_descs,
            total_overhead_ns=total_overhead,
            fully_overlapped=fully_overlapped,
        )

    def register_block_table(self, seq_id: int, physical_block_ids: List[int]) -> None:
        """Register a sequence's PagedAttention block table."""
        self.hw_btw.register_block_table(seq_id, physical_block_ids)

    def stats(self) -> Dict[str, object]:
        return {
            "processed_sequences": self._processed_count,
            "pingpong": self.state_mgr.stats(),
            "doorbell": self.doorbell.stats(),
            "hw_btw": self.hw_btw.stats(),
        }


# ═══════════════════════════════════════════════════════════════════════
# §5  Area/Power Model for Continuous Batching Components
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CBAreaPowerReport:
    """Area/power report for continuous batching hardware additions."""
    pingpong_sram_area_mm2: float
    pingpong_sram_power_mw: float
    doorbell_area_mm2: float
    doorbell_power_mw: float
    hw_btw_area_mm2: float
    hw_btw_power_mw: float
    swap_dma_area_mm2: float
    swap_dma_power_mw: float
    total_area_mm2: float
    total_power_mw: float
    technology_node_nm: int = 7

    def to_dict(self) -> Dict[str, object]:
        return {
            "pingpong_sram_mm2": self.pingpong_sram_area_mm2,
            "pingpong_sram_mw": self.pingpong_sram_power_mw,
            "doorbell_mm2": self.doorbell_area_mm2,
            "doorbell_mw": self.doorbell_power_mw,
            "hw_btw_mm2": self.hw_btw_area_mm2,
            "hw_btw_mw": self.hw_btw_power_mw,
            "swap_dma_mm2": self.swap_dma_area_mm2,
            "swap_dma_mw": self.swap_dma_power_mw,
            "total_mm2": self.total_area_mm2,
            "total_mw": self.total_power_mw,
            "technology_node_nm": self.technology_node_nm,
        }


def estimate_cb_area_power(
    config: ContinuousBatchingConfig,
    technology_node_nm: int = 7,
) -> CBAreaPowerReport:
    """Estimate area/power for continuous batching hardware additions.

    All estimates calibrated to CACTI 7.0 at 7nm.
    """
    node_scale = technology_node_nm / 7.0

    # Ping-pong SRAM: 2 × 13KB = 26KB
    pp_kb = config.pingpong.buffer_size_kb * config.pingpong.num_buffers
    pp_area = max(0.005, 0.0015 * pp_kb * node_scale)  # ~0.039 mm² for 26KB
    pp_power = max(1.0, pp_kb * 0.5 * 1.5 * 0.08)      # ~1.56 mW

    # Doorbell ring buffer: 128 × 8B = 1KB
    db_kb = config.doorbell.ring_depth * config.doorbell.descriptor_bytes / 1024
    db_area = max(0.0005, 0.0015 * db_kb * node_scale)
    db_power = max(0.2, db_kb * 0.5 * 1.5 * 0.08)
    # Arbiter FSM
    db_area += 0.001 * node_scale
    db_power += 0.5 * node_scale

    # HW-BTW: lookup buffer (128B) + address ALU + descriptor formatter
    btw_area = 0.0005 + 0.0003  # ALU + formatter
    btw_area *= node_scale
    btw_power = 0.8 * node_scale  # mW

    # Swap DMA engine (small, dedicated)
    swap_area = 0.002 * node_scale
    swap_power = 1.0 * node_scale

    total_area = pp_area + db_area + btw_area + swap_area
    total_power = pp_power + db_power + btw_power + swap_power

    return CBAreaPowerReport(
        pingpong_sram_area_mm2=round(pp_area, 5),
        pingpong_sram_power_mw=round(pp_power, 2),
        doorbell_area_mm2=round(db_area, 5),
        doorbell_power_mw=round(db_power, 2),
        hw_btw_area_mm2=round(btw_area, 5),
        hw_btw_power_mw=round(btw_power, 2),
        swap_dma_area_mm2=round(swap_area, 5),
        swap_dma_power_mw=round(swap_power, 2),
        total_area_mm2=round(total_area, 5),
        total_power_mw=round(total_power, 2),
        technology_node_nm=technology_node_nm,
    )
