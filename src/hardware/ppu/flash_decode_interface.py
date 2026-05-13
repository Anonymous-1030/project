"""
Flash-Decoding Epilogue Interface for PPU Attention Mass Extraction.

§4.2 of the HPCA paper: Zero-Overhead Attention Mass Extraction.

Core insight: FlashAttention's tiled computation with online softmax never
materializes the full 1×N attention vector.  Flash-Decoding's split-K
reduction already computes per-chunk LSE values (m_j, l_j) as intermediate
state.  The head-averaged chunk mass can be derived as a zero-cost byproduct
of the existing cross-head reduction.

Hardware interface: Memory-Mapped Register File (MMRF) on the GPU-PPU
interconnect, triggered by a valid-bitmap AND reduction.  Total additional
memory traffic: 2 × N_chunks × sizeof(FP16) = 8KB for 128K context.

Compatibility: Requires only ~10 lines of CUDA in the reduction kernel
epilogue and a single address decode rule in the memory controller.
No ISA extensions.  Compatible with TensorRT-LLM and vLLM Flash-Decoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Flash-Decoding LSE Pair ──────────────────────────────────────────

@dataclass
class ChunkLSEPair:
    """Per-chunk Log-Sum-Exp pair from Flash-Decoding's first kernel.

    Each SM processing a chunk produces:
      m_j: local maximum attention score (log-domain stability)
      l_j: local sum of exp(S_{j,t} - m_j) for t in chunk

    These are the *only* byproducts needed.  They are already computed
    and normally discarded after the reduction kernel.
    """
    chunk_id: int
    m_local: float   # local max score (scalar per head, averaged)
    l_local: float   # local exp-sum (scalar per head, averaged)


@dataclass
class ReductionResult:
    """Output of Flash-Decoding's second kernel (global reduction).

    The reduction kernel computes m_global and l_global, then derives
    the true normalized attention mass per chunk:

        Mass_j = l_j * exp(m_j - m_global) / l_global
    """
    m_global: float
    l_global: float
    chunk_masses: Dict[int, float]  # chunk_id -> normalized mass
    num_chunks: int
    num_heads_averaged: int


# ── MMRF: Memory-Mapped Register File ────────────────────────────────

@dataclass
class MMRFConfig:
    """Configuration for the PPU's MMRF input interface.

    The MMRF is a small on-chip buffer mapped to a fixed region of the
    GPU's physical address space.  The reduction kernel writes chunk
    masses here via normal global stores (ST.CS), and the memory
    controller routes writes to [PPU_BASE, PPU_BASE+SIZE) to the PPU
    input FIFO instead of L2/DRAM.

    This requires NO ISA extensions — identical mechanism to how GPUs
    access NVLink/PCIe MMIO registers.
    """
    max_chunks: int = 4096          # Max chunks supported (128K ctx / 32 tok)
    fifo_depth: int = 32            # Input FIFO depth (handles burst writes)
    data_width_bits: int = 16       # FP16 mass values from GPU
    output_width_bits: int = 16     # Q0.15 fixed-point to pipeline
    reorder_buffer: bool = True     # Handle out-of-order chunk arrival
    trigger_mode: str = "bitmap"    # "bitmap" (AND reduction) or "count"


@dataclass
class MMRFReceiveResult:
    """Result of MMRF receiving and format-casting chunk masses."""
    chunk_masses: Dict[str, float]  # chunk_id (str) -> normalized mass
    num_received: int
    all_valid: bool                 # True when all expected chunks arrived
    format_cast_cycles: int         # Always 1 (FP16 -> Q0.15)
    reorder_stalls: int             # Stalls due to out-of-order arrival


# ── Functional Models ─────────────────────────────────────────────────

class FlashDecodeEpilogueExporter:
    """Functional model of the Flash-Decoding reduction kernel epilogue.

    Simulates the zero-overhead extraction of chunk-level attention masses
    from Flash-Decoding's split-K architecture.

    In real hardware:
      1. First kernel (per-SM): computes local (m_j, l_j) per chunk per head
      2. Second kernel (reduction): computes m_global, l_global, then
         derives Mass_j = l_j * exp(m_j - m_global) / l_global
      3. Epilogue: writes Mass_j array to MMRF via ST.CS (streaming store)

    This class models step 2+3 functionally.
    """

    def __init__(self, num_heads: int = 1):
        self.num_heads = num_heads

    def compute_chunk_masses_from_lse(
        self,
        lse_pairs: List[ChunkLSEPair],
    ) -> ReductionResult:
        """Derive normalized chunk masses from LSE pairs.

        This is what the reduction kernel computes.  The key formula:

            Mass_j = l_j * exp(m_j - m_global) / l_global

        where:
            m_global = max(m_j for all j)
            l_global = sum(l_j * exp(m_j - m_global) for all j)

        Cost: O(N_chunks) scalar ops.  Trivial compared to the attention
        computation itself.
        """
        if not lse_pairs:
            return ReductionResult(
                m_global=0.0, l_global=0.0,
                chunk_masses={}, num_chunks=0,
                num_heads_averaged=self.num_heads,
            )

        # Step 1: Find global maximum (numerically stable reduction)
        m_global = max(p.m_local for p in lse_pairs)

        # Step 2: Compute rescaled local sums and global denominator
        rescaled = {}
        l_global = 0.0
        for p in lse_pairs:
            # l_j * exp(m_j - m_global): rescale to common base
            rescaled_l = p.l_local * math.exp(p.m_local - m_global)
            rescaled[p.chunk_id] = rescaled_l
            l_global += rescaled_l

        # Step 3: Normalize to get true attention masses
        chunk_masses = {}
        if l_global > 0:
            for chunk_id, rl in rescaled.items():
                chunk_masses[chunk_id] = rl / l_global
        else:
            for p in lse_pairs:
                chunk_masses[p.chunk_id] = 1.0 / len(lse_pairs)

        return ReductionResult(
            m_global=m_global,
            l_global=l_global,
            chunk_masses=chunk_masses,
            num_chunks=len(lse_pairs),
            num_heads_averaged=self.num_heads,
        )

    def simulate_from_attention_vector(
        self,
        attention_masses: Dict[int, float],
        chunk_size: int = 64,
    ) -> Tuple[List[ChunkLSEPair], ReductionResult]:
        """Reverse-engineer LSE pairs from known attention masses.

        For simulation/testing: given the "ground truth" chunk masses,
        construct plausible (m_j, l_j) pairs that would produce them
        via the Flash-Decoding reduction.

        This allows the PPU simulator to work with existing evaluation
        infrastructure that provides attention masses directly.
        """
        lse_pairs = []
        for chunk_id, mass in attention_masses.items():
            # Construct synthetic LSE pair:
            # If mass = l_j * exp(m_j - m_global) / l_global,
            # we can set m_j = 0 for all chunks (uniform base),
            # then l_j = mass * l_global.  With m_global = 0 and
            # l_global = sum(l_j) = sum(mass * l_global) = l_global,
            # this is consistent.  Just use l_j = mass directly.
            lse_pairs.append(ChunkLSEPair(
                chunk_id=chunk_id,
                m_local=0.0,       # uniform base (simulation only)
                l_local=max(mass, 1e-10),
            ))

        result = self.compute_chunk_masses_from_lse(lse_pairs)
        return lse_pairs, result


class MMRFReceiver:
    """Functional model of the PPU's MMRF input stage.

    Hardware behavior:
      1. Receives FP16 chunk masses from GPU via memory-mapped writes
      2. Reorder buffer: uses chunk_id as direct write address (no CAM)
      3. Valid bitmap: tracks which chunks have been written
      4. Trigger: asserts when all expected chunks are valid
      5. Format cast: FP16 -> Q0.15 fixed-point (single-cycle shift)

    RTL equivalent:
        always @(posedge clk) begin
            if (mmrf_write_en) begin
                mass_buffer[mmrf_write_addr] <= mmrf_write_data;
                valid_bits[mmrf_write_addr] <= 1'b1;
            end
            all_valid <= &valid_bits[0:N_chunks-1];
            if (all_valid) trigger_stage2 <= 1'b1;
        end
    """

    def __init__(self, config: MMRFConfig):
        self.config = config
        self._buffer: Dict[int, float] = {}
        self._valid_bits: set = set()
        self._expected_chunks: int = 0
        self._reorder_stalls: int = 0

    def reset(self, expected_chunks: int) -> None:
        """Reset for a new decode step."""
        self._buffer.clear()
        self._valid_bits.clear()
        self._expected_chunks = expected_chunks
        self._reorder_stalls = 0

    def write(self, chunk_id: int, mass: float) -> None:
        """Receive a single chunk mass from the reduction kernel.

        In hardware: a single ST.CS instruction from the GPU writes
        to address PPU_BASE + chunk_id * sizeof(FP16).  The memory
        controller routes this to the MMRF instead of L2.
        """
        self._buffer[chunk_id] = mass
        self._valid_bits.add(chunk_id)

    def write_batch(self, chunk_masses: Dict[int, float]) -> None:
        """Receive all chunk masses (models the full epilogue write burst)."""
        for cid, mass in chunk_masses.items():
            self.write(cid, mass)

    @property
    def all_valid(self) -> bool:
        """Check if all expected chunks have been received.

        Hardware: AND reduction over valid_bits[0:N_chunks-1].
        """
        return len(self._valid_bits) >= self._expected_chunks

    def read_all(self) -> MMRFReceiveResult:
        """Read all buffered masses and format-cast to fixed-point.

        The format cast (FP16 -> Q0.15) is a single-cycle operation:
        extract mantissa, shift by exponent.  Since Mass_j ∈ [0, 1],
        the exponent is always ≤ 0, so it's just a right-shift.
        """
        # Convert integer chunk_ids to string chunk_ids for pipeline compat
        str_masses = {str(cid): mass for cid, mass in self._buffer.items()}

        return MMRFReceiveResult(
            chunk_masses=str_masses,
            num_received=len(self._buffer),
            all_valid=self.all_valid,
            format_cast_cycles=1,  # Always 1 cycle
            reorder_stalls=self._reorder_stalls,
        )


# ── Timing Model ──────────────────────────────────────────────────────

@dataclass
class FlashDecodeTimingModel:
    """Models the timing overlap between Flash-Decoding and PPU.

    Timeline for a single decode step:
      T=0          : Flash-Decoding Kernel 1 starts (per-head, per-split)
      T=T_k1       : Kernel 1 completes, LSE pairs in L2/VRAM
      T=T_k1       : Reduction Kernel starts
      T=T_k1+T_red : Reduction complete, Mass_j written to MMRF
      T=T_k1+T_red : PPU triggered (all_valid asserted)
      T=T_ppu_done : PPU pipeline complete, Top-K ready
      T=T_ppu_done : DMA Engine starts CXL prefetch
      T=T_dma_done : Prefetch complete (overlaps with next step's compute)
    """
    # Kernel 1 timing (depends on model size and seq_len)
    kernel1_us: float = 400.0       # Flash-Decoding first kernel
    reduction_us: float = 10.0      # Reduction kernel (including epilogue)
    reduction_tail_us: float = 5.0  # Worst-case tail latency variance

    # PPU timing
    ppu_cycles: int = 82            # Total PPU pipeline cycles
    ppu_clock_ghz: float = 1.0      # PPU clock frequency

    # DMA timing (CXL 2.0 defaults)
    dma_base_latency_us: float = 0.8
    dma_bandwidth_gbps: float = 64.0
    chunk_bytes: int = 65536        # 64KB per chunk

    def compute_timeline(
        self,
        num_promoted: int = 6,
        max_outstanding: int = 16,
    ) -> Dict[str, float]:
        """Compute full timeline for one decode step."""
        ppu_us = self.ppu_cycles / (self.ppu_clock_ghz * 1e3)

        # DMA transfer time
        if num_promoted == 0:
            dma_us = 0.0
        else:
            bw_bytes_per_us = self.dma_bandwidth_gbps * 1e3 / 8
            first = self.dma_base_latency_us + self.chunk_bytes / bw_bytes_per_us
            if num_promoted > 1:
                remaining = (num_promoted - 1) * self.chunk_bytes / bw_bytes_per_us
                pipeline_factor = min(num_promoted, max_outstanding)
                dma_us = first + remaining / max(pipeline_factor, 1)
            else:
                dma_us = first

        t_k1_end = self.kernel1_us
        t_red_end = t_k1_end + self.reduction_us
        t_ppu_end = t_red_end + ppu_us
        t_dma_end = t_ppu_end + dma_us

        # Total compute window (next step starts here)
        compute_window = self.kernel1_us + self.reduction_us
        exposed_latency = max(0.0, t_dma_end - compute_window)

        return {
            "kernel1_end_us": t_k1_end,
            "reduction_end_us": t_red_end,
            "ppu_end_us": t_ppu_end,
            "dma_end_us": t_dma_end,
            "compute_window_us": compute_window,
            "ppu_latency_us": ppu_us,
            "dma_latency_us": dma_us,
            "exposed_latency_us": exposed_latency,
            "margin_us": compute_window - t_dma_end if t_dma_end < compute_window else 0.0,
            "fully_hidden": exposed_latency == 0.0,
        }
