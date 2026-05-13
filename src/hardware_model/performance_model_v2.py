"""Cycle-analytical performance model v2 with queuing theory and contention.

HPCA upgrade over v1: The original model used a simple max(compute, transfer)
overlap model.  This version adds:

1. **M/D/1 queuing** at the CXL memory controller — promotion requests queue
   behind each other, so effective latency depends on offered load (ρ).
2. **DRAM bank conflict modeling** — row buffer misses add ~20ns penalty.
3. **Multi-tenant contention** — K tenants sharing the CXL link see
   per-tenant bandwidth B/K and increased queuing delay.
4. **Pipeline bubble modeling** — structural hazards in the KCMC pipeline
   (NMD busy, DMA queue full) create stalls.
5. **Prefetch misprediction penalty** — wrong prefetches waste bandwidth
   and evict useful data from the staging buffer.

Key equations:
  - Offered load: ρ = λ × S, where λ = chunks_per_step / T_step, S = service time
  - M/D/1 waiting time: W = ρ × S / (2 × (1 - ρ))  [Pollaczek-Khinchine]
  - Effective per-request latency: L_eff = S + W + L_bank_conflict
  - Per-step exposed latency: L_exposed = max(0, N_chunks × L_eff - T_compute)

References:
  - Kleinrock, "Queueing Systems Vol 1", 1975 (M/D/1 formula)
  - Kim et al., "Pond: CXL-Based Memory Pooling", ASPLOS 2023
  - Li et al., "CXL-ANNS", ISCA 2023 (CXL queuing model)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .near_cxl_decompressor import NearCXLDecompressor
from .performance_model import LatencyBreakdown


@dataclass
class DRAMTimingConfig:
    """DDR5 DRAM timing parameters (JEDEC standard).

    Default values correspond to DDR5-4800 (common in CXL expanders).
    All times in nanoseconds.
    """
    tCAS: float = 40.0      # Column access strobe latency
    tRCD: float = 40.0      # Row-to-column delay
    tRP: float = 40.0       # Row precharge time
    tRAS: float = 77.0      # Row active time
    tRC: float = 117.0      # Row cycle time (tRAS + tRP)
    tFAW: float = 40.0      # Four-activate window
    tRFC: float = 350.0     # Refresh cycle time
    tREFI: float = 3900.0   # Refresh interval

    num_banks: int = 16          # Banks per rank
    num_bank_groups: int = 4     # Bank groups
    row_buffer_size_bytes: int = 8192  # 8KB row buffer

    def row_hit_latency_ns(self) -> float:
        """Latency for a row buffer hit (column access only)."""
        return self.tCAS

    def row_miss_latency_ns(self) -> float:
        """Latency for a row buffer miss (precharge + activate + column)."""
        return self.tRP + self.tRCD + self.tCAS

    def row_conflict_latency_ns(self) -> float:
        """Latency for a row conflict (same bank, different row open)."""
        return self.tRP + self.tRCD + self.tCAS

    def avg_latency_ns(self, hit_rate: float = 0.3) -> float:
        """Average DRAM access latency given row buffer hit rate."""
        return (hit_rate * self.row_hit_latency_ns() +
                (1 - hit_rate) * self.row_miss_latency_ns())

    def refresh_overhead_fraction(self) -> float:
        """Fraction of time lost to refresh."""
        return self.tRFC / self.tREFI


@dataclass
class CXLProtocolConfig:
    """CXL 3.0 protocol parameters.

    References:
      - CXL 3.0 Specification, Chapter 3 (CXL.mem)
      - Samsung CXL Memory Expander, ISSCC 2022
    """
    version: str = "3.0"
    flit_size_bytes: int = 256       # CXL 3.0 uses 256B flits
    link_width: int = 16             # x16 link
    link_rate_gtps: float = 64.0     # GT/s per lane (CXL 3.0 = PCIe 6.0)
    protocol_overhead: float = 0.02  # 2% for CXL.mem TLP encapsulation
    credit_rtt_ns: float = 100.0     # Credit round-trip time

    # CXL.mem specific
    bias_flip_latency_ns: float = 200.0  # Host-bias ↔ Device-bias transition
    snoop_latency_ns: float = 50.0       # Back-invalidation snoop latency
    hdm_type: str = "HDM-DB"             # Device-managed bias (Type-3)

    def raw_bandwidth_gbps(self) -> float:
        """Raw link bandwidth before protocol overhead."""
        return self.link_width * self.link_rate_gtps / 8.0  # GT/s → GB/s

    def effective_bandwidth_gbps(self) -> float:
        """Effective bandwidth after protocol overhead."""
        return self.raw_bandwidth_gbps() * (1.0 - self.protocol_overhead)

    def flit_serialization_ns(self) -> float:
        """Time to serialize one flit onto the link."""
        bw_bytes_per_ns = self.raw_bandwidth_gbps()  # GB/s = bytes/ns
        return self.flit_size_bytes / bw_bytes_per_ns


@dataclass
class QueuingModelConfig:
    """Configuration for the M/D/1 queuing model."""
    queue_depth: int = 32            # Memory controller queue depth
    scheduling_policy: str = "FR-FCFS"  # First-Ready First-Come-First-Served
    rw_switch_penalty_ns: float = 10.0  # Read/write turnaround penalty


@dataclass
class LatencyBreakdownV2:
    """Extended latency breakdown with queuing and contention details."""
    # Core latencies (same as v1)
    total_us: float = 0.0
    compute_us: float = 0.0
    hbm_attention_us: float = 0.0
    promotion_us: float = 0.0
    decompression_us: float = 0.0
    exposed_transfer_us: float = 0.0
    overlap_hidden_us: float = 0.0

    # New: queuing and contention
    queuing_delay_us: float = 0.0        # M/D/1 waiting time
    dram_access_us: float = 0.0          # DRAM backend latency
    bank_conflict_penalty_us: float = 0.0  # Extra from row conflicts
    protocol_overhead_us: float = 0.0    # CXL protocol overhead
    refresh_penalty_us: float = 0.0      # DRAM refresh interference
    pipeline_bubble_us: float = 0.0      # KCMC pipeline stalls
    metadata_write_us: float = 0.0       # Async metadata write traffic
    exposed_metadata_write_us: float = 0.0  # Exposed metadata write latency

    # Multi-tenant
    contention_delay_us: float = 0.0     # From sharing CXL link
    num_tenants: int = 1

    # Utilization metrics
    link_utilization: float = 0.0        # ρ: offered load / capacity
    dram_utilization: float = 0.0
    prefetch_hit_rate: float = 0.0
    row_buffer_hit_rate: float = 0.0

    def to_v1(self) -> LatencyBreakdown:
        """Convert to v1 format for backward compatibility."""
        return LatencyBreakdown(
            total_us=self.total_us,
            compute_us=self.compute_us,
            hbm_attention_us=self.hbm_attention_us,
            promotion_us=self.promotion_us,
            decompression_us=self.decompression_us,
            exposed_transfer_us=self.exposed_transfer_us,
            overlap_hidden_us=self.overlap_hidden_us,
        )


class CycleAnalyticalModelV2:
    """Analytical decode-latency model with queuing theory and contention.

    Upgrades over v1:
      1. M/D/1 queuing at CXL memory controller
      2. DRAM bank conflict and refresh modeling
      3. Multi-tenant bandwidth sharing and contention
      4. KCMC pipeline bubble modeling
      5. Prefetch misprediction penalty

    The model computes per-step decode latency for autoregressive generation.
    """

    def __init__(
        self,
        hbm_bandwidth_gbps: float = 3350.0,
        cxl_config: Optional[CXLProtocolConfig] = None,
        dram_config: Optional[DRAMTimingConfig] = None,
        queuing_config: Optional[QueuingModelConfig] = None,
        sparse_speedup: float = 2.2,
        base_compute_us: float = 120.0,
        prefetch_accuracy: float = 0.85,
        chunks_per_step: int = 3,
        chunk_size_tokens: int = 64,
        num_tenants: int = 1,
        decompressor: Optional[NearCXLDecompressor] = None,
    ) -> None:
        self.hbm_bandwidth_gbps = hbm_bandwidth_gbps
        self.cxl = cxl_config or CXLProtocolConfig()
        self.dram = dram_config or DRAMTimingConfig()
        self.queuing = queuing_config or QueuingModelConfig()
        self.sparse_speedup = sparse_speedup
        self.base_compute_us = base_compute_us
        self.prefetch_accuracy = prefetch_accuracy
        self.chunks_per_step = chunks_per_step
        self.chunk_size_tokens = chunk_size_tokens
        self.num_tenants = num_tenants
        self.decompressor = decompressor or NearCXLDecompressor()

    def kv_bytes_per_token(
        self, num_layers: int, num_heads: int, head_dim: int,
        bytes_per_elem: int = 2,
    ) -> int:
        """Bytes for one token's KV across all layers (K + V)."""
        return 2 * num_layers * num_heads * head_dim * bytes_per_elem

    # ------------------------------------------------------------------
    # M/D/1 queuing model
    # ------------------------------------------------------------------
    def _md1_waiting_time_us(
        self, arrival_rate_per_us: float, service_time_us: float,
    ) -> float:
        """M/D/1 queue waiting time (Pollaczek-Khinchine for deterministic service).

        W = ρ·S / (2·(1-ρ))  where ρ = λ·S

        Reference: Kleinrock, "Queueing Systems Vol 1", 1975, Eq. 5.72
        """
        rho = arrival_rate_per_us * service_time_us
        if rho >= 0.99:
            # Near saturation — cap at large but finite value
            return service_time_us * 50.0
        if rho <= 0.0:
            return 0.0
        return (rho * service_time_us) / (2.0 * (1.0 - rho))

    # ------------------------------------------------------------------
    # DRAM access modeling
    # ------------------------------------------------------------------
    def _dram_access_latency_us(
        self, chunk_bytes: int, row_buffer_hit_rate: float = 0.3,
    ) -> Tuple[float, float]:
        """Model DRAM backend latency for a chunk transfer.

        Returns (access_latency_us, bank_conflict_penalty_us).
        For large sequential reads, the latency is bandwidth-bound rather
        than purely latency-bound, so we floor it by sustained DRAM BW.
        """
        avg_ns = self.dram.avg_latency_ns(row_buffer_hit_rate)

        # Number of DRAM accesses: chunk_bytes / burst_length (64B for DDR5)
        burst_bytes = 64
        num_accesses = max(1, chunk_bytes // burst_bytes)

        # Bank-level parallelism: accesses to different banks overlap
        effective_accesses = math.ceil(num_accesses / self.dram.num_bank_groups)
        access_latency_ns = effective_accesses * avg_ns

        # Bank conflict penalty: probability of hitting same bank
        conflict_prob = 1.0 / self.dram.num_banks
        conflict_penalty_ns = (
            num_accesses * conflict_prob *
            (self.dram.row_conflict_latency_ns() - self.dram.row_hit_latency_ns())
        )

        # Refresh interference
        refresh_frac = self.dram.refresh_overhead_fraction()
        refresh_penalty_ns = access_latency_ns * refresh_frac

        total_ns = access_latency_ns + conflict_penalty_ns + refresh_penalty_ns
        latency_us = total_ns / 1000.0

        # Bandwidth floor: DDR5-4800 ~38.4 GB/s per channel; assume 2 channels
        sustained_dram_bw_gbps = 76.8
        bw_floor_us = (chunk_bytes / (sustained_dram_bw_gbps * 1024**3)) * 1e6

        # For large sequential reads the controller pipelines bursts,
        # so the effective time is bandwidth-bound rather than the
        # sum of individual access latencies.
        return max(bw_floor_us, avg_ns / 1000.0), conflict_penalty_ns / 1000.0

    # ------------------------------------------------------------------
    # CXL protocol overhead
    # ------------------------------------------------------------------
    def _cxl_protocol_overhead_us(self, chunk_bytes: int) -> float:
        """CXL.mem protocol overhead for one chunk transfer.

        Includes: flit serialization, credit flow control, bias check.
        """
        num_flits = math.ceil(chunk_bytes / self.cxl.flit_size_bytes)
        serialization_ns = num_flits * self.cxl.flit_serialization_ns()

        # Credit stall: credits return pipelined during long transfers;
        # only one RTT stall if the burst exceeds the initial credit pool.
        credit_ns = (
            self.cxl.credit_rtt_ns if num_flits > self.queuing.queue_depth else 0.0
        )

        # Bias check (device-managed, no host snoop needed for Type-3 HDM-DB)
        bias_ns = 0.0  # No bias flip for normal reads in HDM-DB mode

        return (serialization_ns + credit_ns + bias_ns) / 1000.0

    # ------------------------------------------------------------------
    # KCMC pipeline bubble modeling
    # ------------------------------------------------------------------
    def _pipeline_bubble_us(
        self, chunks_this_step: int, nmd_throughput_gbps: float,
        chunk_bytes: int,
    ) -> float:
        """Model pipeline bubbles from structural hazards.

        Bubbles occur when:
        1. NMD is busy decompressing previous chunk
        2. DMA write queue is full
        3. Metadata SRAM port conflict
        """
        # NMD throughput limit: can decompress one chunk every T_nmd
        nmd_time_per_chunk_us = (
            chunk_bytes / (nmd_throughput_gbps * 1e9 / 1e6)
        ) if nmd_throughput_gbps > 0 else 0.0

        # If chunks arrive faster than NMD can process, bubbles form
        # Arrival interval = CXL transfer time per chunk
        bw = self.cxl.effective_bandwidth_gbps() / max(self.num_tenants, 1)
        arrival_interval_us = (chunk_bytes / (bw * 1e9 / 1e6)) if bw > 0 else 0.0

        if arrival_interval_us > 0 and nmd_time_per_chunk_us > arrival_interval_us:
            bubble_per_chunk = nmd_time_per_chunk_us - arrival_interval_us
            return bubble_per_chunk * chunks_this_step
        return 0.0

    # ------------------------------------------------------------------
    # Baseline model (same as v1)
    # ------------------------------------------------------------------
    def model_baseline_latency(
        self,
        seq_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
    ) -> LatencyBreakdownV2:
        """Full-KV dense attention baseline (no promotion needed)."""
        bytes_per_tok = self.kv_bytes_per_token(num_layers, num_heads, head_dim)
        kv_bytes = seq_len * bytes_per_tok
        hbm_us = (kv_bytes / (self.hbm_bandwidth_gbps * 1024**3)) * 1e6
        total = self.base_compute_us + hbm_us
        return LatencyBreakdownV2(
            total_us=total,
            compute_us=self.base_compute_us,
            hbm_attention_us=hbm_us,
            num_tenants=self.num_tenants,
        )

    # ------------------------------------------------------------------
    # KCMC model with queuing + contention
    # ------------------------------------------------------------------
    def model_kcmc_latency(
        self,
        seq_len: int,
        retention_ratio: float,
        promotion_ratio: float,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        compression_bits: int = 4,
        row_buffer_hit_rate: float = 0.3,
    ) -> LatencyBreakdownV2:
        """KCMC latency with queuing, DRAM timing, and contention.

        This is the main model for HPCA evaluation.
        """
        retention_ratio = min(max(retention_ratio, 1e-4), 1.0)
        promotion_ratio = min(max(promotion_ratio, 0.0), 1.0)
        bytes_per_tok = self.kv_bytes_per_token(num_layers, num_heads, head_dim)
        kv_bytes = seq_len * bytes_per_tok

        # 1. Sparse HBM attention
        active_ratio = retention_ratio + promotion_ratio
        sparse_hbm_bytes = kv_bytes * active_ratio
        hbm_attention_us = (
            sparse_hbm_bytes / (self.hbm_bandwidth_gbps * 1024**3)
        ) * 1e6

        # 2. Compute with sparse speedup
        compute_us = self.base_compute_us / max(self.sparse_speedup, 1.0)

        # 3. Per-step promotion: chunks_per_step chunks
        chunk_bytes = self.chunk_size_tokens * bytes_per_tok
        compressed_chunk_bytes = int(chunk_bytes * compression_bits / 16)

        # 3a. CXL transfer time (per chunk, with bandwidth sharing)
        per_tenant_bw = self.cxl.effective_bandwidth_gbps() / max(self.num_tenants, 1)
        transfer_per_chunk_us = (
            compressed_chunk_bytes / (per_tenant_bw * 1024**3)
        ) * 1e6

        # 3b. DRAM access latency
        dram_us, bank_conflict_us = self._dram_access_latency_us(
            compressed_chunk_bytes, row_buffer_hit_rate
        )

        # 3c. CXL protocol overhead
        protocol_us = self._cxl_protocol_overhead_us(compressed_chunk_bytes)

        # 3d. Decompression
        self.decompressor.compressed_bits = compression_bits
        fetch = self.decompressor.model_fetch(
            output_bytes=chunk_bytes,
            link_bandwidth_gbps=per_tenant_bw,
        )
        decompress_us = fetch["decompression_time_us"]

        # 3e. Service time per chunk (DRAM + transfer + protocol + decompress)
        service_time_us = max(
            dram_us + transfer_per_chunk_us + protocol_us,
            decompress_us,
        )

        # 4. Queuing delay — batch arrival model
        # All promotion chunks for a step arrive together at step start.
        # If the batch service time fits within the compute window, no queuing.
        compute_window_us = compute_us + hbm_attention_us
        batch_service_time_us = service_time_us * self.chunks_per_step
        if batch_service_time_us <= compute_window_us:
            queuing_us = 0.0
            link_utilization = batch_service_time_us / max(compute_window_us, 1e-9)
        else:
            # Batch exceeds compute window: use M/D/1 for the excess portion
            excess_fraction = 1.0 - compute_window_us / batch_service_time_us
            arrival_rate = (self.chunks_per_step * excess_fraction) / max(batch_service_time_us, 1e-9)
            queuing_us = self._md1_waiting_time_us(arrival_rate, service_time_us)
            link_utilization = 1.0

        # 5. Multi-tenant contention delay
        contention_us = 0.0
        if self.num_tenants > 1:
            # Each tenant's requests see other tenants' requests in the queue
            # Approximate: multiply queuing delay by contention factor
            contention_factor = 1.0 + 0.3 * math.log2(self.num_tenants)
            contention_us = queuing_us * (contention_factor - 1.0)

        # 6. Pipeline bubbles
        nmd_throughput = self.decompressor.decompression_throughput_gbps()
        bubble_us = self._pipeline_bubble_us(
            self.chunks_per_step, nmd_throughput, chunk_bytes
        )

        # 7. Refresh penalty
        refresh_us = dram_us * self.dram.refresh_overhead_fraction()

        # 8. Total per-chunk effective latency
        effective_per_chunk_us = (
            service_time_us + queuing_us + contention_us
        )

        # 9. Total promotion latency for all chunks this step
        total_promotion_us = effective_per_chunk_us * self.chunks_per_step + bubble_us

        # 10. Overlap with compute
        overlap_hidden_us = (
            min(total_promotion_us, compute_window_us) * self.prefetch_accuracy
        )
        exposed_transfer_us = max(0.0, total_promotion_us - overlap_hidden_us)

        # 11. Prefetch misprediction penalty
        mispredict_penalty_us = (
            (1.0 - self.prefetch_accuracy) * service_time_us
        )
        exposed_transfer_us += mispredict_penalty_us

        # 12. Metadata write overhead (reviewer concern: sync/consistency cost)
        # Only chunk metadata (tier, TTL, access counters) is written back;
        # KV data itself is WORM during decode.
        metadata_write_bytes = self.chunks_per_step * 200  # ~200 B per chunk
        metadata_write_us = (
            metadata_write_bytes / (per_tenant_bw * 1024**3)
        ) * 1e6
        # Batched async DMA; overlaps with compute window
        exposed_metadata_write_us = max(0.0, metadata_write_us - compute_window_us)
        total_us = compute_window_us + exposed_transfer_us + exposed_metadata_write_us

        return LatencyBreakdownV2(
            total_us=total_us,
            compute_us=compute_us,
            hbm_attention_us=hbm_attention_us,
            promotion_us=transfer_per_chunk_us * self.chunks_per_step,
            decompression_us=decompress_us * self.chunks_per_step,
            exposed_transfer_us=exposed_transfer_us,
            overlap_hidden_us=overlap_hidden_us,
            queuing_delay_us=queuing_us * self.chunks_per_step,
            dram_access_us=dram_us * self.chunks_per_step,
            bank_conflict_penalty_us=bank_conflict_us * self.chunks_per_step,
            protocol_overhead_us=protocol_us * self.chunks_per_step,
            refresh_penalty_us=refresh_us * self.chunks_per_step,
            pipeline_bubble_us=bubble_us,
            contention_delay_us=contention_us * self.chunks_per_step,
            metadata_write_us=metadata_write_us,
            exposed_metadata_write_us=exposed_metadata_write_us,
            num_tenants=self.num_tenants,
            link_utilization=link_utilization,
            dram_utilization=min(
                dram_us * self.chunks_per_step / max(compute_window_us, 1e-9), 1.0
            ),
            prefetch_hit_rate=self.prefetch_accuracy,
            row_buffer_hit_rate=row_buffer_hit_rate,
        )

    # ------------------------------------------------------------------
    # Comparison utilities
    # ------------------------------------------------------------------
    def compute_speedup(
        self, baseline: LatencyBreakdownV2, kcmc: LatencyBreakdownV2,
    ) -> float:
        return baseline.total_us / max(kcmc.total_us, 1e-9)

    def sweep_promotion_ratio(
        self,
        seq_len: int,
        retention_ratio: float,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        ratios: Optional[List[float]] = None,
        compression_bits: int = 4,
    ) -> List[Dict[str, float]]:
        """Sweep promotion ratio and return speedup + latency breakdown."""
        if ratios is None:
            ratios = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

        baseline = self.model_baseline_latency(
            seq_len, num_layers, num_heads, head_dim
        )
        results = []
        for ratio in ratios:
            kcmc = self.model_kcmc_latency(
                seq_len, retention_ratio, ratio,
                num_layers, num_heads, head_dim, compression_bits,
            )
            results.append({
                "promotion_ratio": ratio,
                "speedup": self.compute_speedup(baseline, kcmc),
                "total_us": kcmc.total_us,
                "exposed_us": kcmc.exposed_transfer_us,
                "queuing_us": kcmc.queuing_delay_us,
                "link_util": kcmc.link_utilization,
                "contention_us": kcmc.contention_delay_us,
            })
        return results

    def sweep_tenants(
        self,
        seq_len: int,
        retention_ratio: float,
        promotion_ratio: float,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        tenant_counts: Optional[List[int]] = None,
    ) -> List[Dict[str, float]]:
        """Sweep number of tenants and show contention impact."""
        if tenant_counts is None:
            tenant_counts = [1, 2, 4, 8, 16]

        original_tenants = self.num_tenants
        results = []
        for k in tenant_counts:
            self.num_tenants = k
            baseline = self.model_baseline_latency(
                seq_len, num_layers, num_heads, head_dim
            )
            kcmc = self.model_kcmc_latency(
                seq_len, retention_ratio, promotion_ratio,
                num_layers, num_heads, head_dim,
            )
            results.append({
                "num_tenants": k,
                "speedup": self.compute_speedup(baseline, kcmc),
                "total_us": kcmc.total_us,
                "per_tenant_bw_gbps": (
                    self.cxl.effective_bandwidth_gbps() / k
                ),
                "queuing_us": kcmc.queuing_delay_us,
                "contention_us": kcmc.contention_delay_us,
                "link_util": kcmc.link_utilization,
            })
        self.num_tenants = original_tenants
        return results

    def summarize_comparison(
        self,
        baseline: LatencyBreakdownV2,
        kcmc: LatencyBreakdownV2,
    ) -> Dict[str, float]:
        """Detailed comparison summary for paper tables."""
        return {
            "baseline_total_us": baseline.total_us,
            "kcmc_total_us": kcmc.total_us,
            "speedup": self.compute_speedup(baseline, kcmc),
            "baseline_hbm_attention_us": baseline.hbm_attention_us,
            "kcmc_hbm_attention_us": kcmc.hbm_attention_us,
            "kcmc_compute_us": kcmc.compute_us,
            "kcmc_exposed_transfer_us": kcmc.exposed_transfer_us,
            "kcmc_overlap_hidden_us": kcmc.overlap_hidden_us,
            "kcmc_queuing_delay_us": kcmc.queuing_delay_us,
            "kcmc_dram_access_us": kcmc.dram_access_us,
            "kcmc_bank_conflict_us": kcmc.bank_conflict_penalty_us,
            "kcmc_protocol_overhead_us": kcmc.protocol_overhead_us,
            "kcmc_refresh_penalty_us": kcmc.refresh_penalty_us,
            "kcmc_pipeline_bubble_us": kcmc.pipeline_bubble_us,
            "kcmc_contention_delay_us": kcmc.contention_delay_us,
            "kcmc_decompression_us": kcmc.decompression_us,
            "link_utilization": kcmc.link_utilization,
            "dram_utilization": kcmc.dram_utilization,
            "num_tenants": kcmc.num_tenants,
        }


# ======================================================================
# Pre-configured hardware profiles for paper experiments
# ======================================================================

def make_h100_sxm_cxl30() -> CycleAnalyticalModelV2:
    """H100-SXM with CXL 3.0 memory expander."""
    return CycleAnalyticalModelV2(
        hbm_bandwidth_gbps=3350.0,
        cxl_config=CXLProtocolConfig(
            version="3.0", link_rate_gtps=64.0, link_width=16,
        ),
        dram_config=DRAMTimingConfig(),  # DDR5-4800
        sparse_speedup=2.5,
        base_compute_us=5000.0,  # ~5 ms decode step for 3B model
    )


def make_h100_pcie_cxl20() -> CycleAnalyticalModelV2:
    """H100-PCIe with CXL 2.0 (PCIe Gen5 link)."""
    return CycleAnalyticalModelV2(
        hbm_bandwidth_gbps=2000.0,
        cxl_config=CXLProtocolConfig(
            version="2.0", link_rate_gtps=32.0, link_width=16,
            bias_flip_latency_ns=300.0,  # CXL 2.0 slower bias flip
        ),
        sparse_speedup=2.2,
        base_compute_us=120.0,
    )


def make_a100_pcie() -> CycleAnalyticalModelV2:
    """A100-80G with PCIe Gen4 (no native CXL, use PCIe for transfers)."""
    return CycleAnalyticalModelV2(
        hbm_bandwidth_gbps=2039.0,
        cxl_config=CXLProtocolConfig(
            version="1.1", link_rate_gtps=16.0, link_width=16,
            protocol_overhead=0.05,  # Higher overhead for PCIe-only
        ),
        sparse_speedup=2.0,
        base_compute_us=10000.0,  # ~10 ms decode step for 3B model on A100
    )


HARDWARE_PROFILES = {
    "H100-SXM-CXL3.0": make_h100_sxm_cxl30,
    "H100-PCIe-CXL2.0": make_h100_pcie_cxl20,
    "A100-PCIe": make_a100_pcie,
}
