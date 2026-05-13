"""Cycle-analytical performance model for KCMC-enabled KV cache inference.

HPCA revision: The previous model computed promotion latency for ALL promoted
bytes in a single monolithic transfer, yielding speedup < 1.  The corrected
model uses **per-step incremental promotion**: during each decode step the
prefetch engine transfers only the chunks needed for the *next* step (typically
2-4 chunks) while the current step's attention runs concurrently on the GPU.

Key modelling equations:
  - Per-step promotable bytes  B_step = BW_eff × T_compute
  - Per-step promotion is fully hidden when B_step >= bytes_per_chunk × N_chunks
  - Total decode latency ≈ N_steps × max(T_compute + T_hbm_attn, T_step_promo)
  - Amortized promotion overhead = max(0, T_step_promo - T_compute) per step
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .near_cxl_decompressor import NearCXLDecompressor


@dataclass
class LatencyBreakdown:
    total_us: float
    compute_us: float
    hbm_attention_us: float
    promotion_us: float
    decompression_us: float
    exposed_transfer_us: float
    overlap_hidden_us: float


class CycleAnalyticalModel:
    """Analytical decode-latency model for comparing baseline vs KCMC.

    The model targets the *per-step* latency during autoregressive decoding.
    Promotion is pipelined: while step t runs attention on HBM-resident KV,
    chunks predicted for step t+1 are fetched from DRAM/CXL in the background.
    """

    def __init__(
        self,
        hbm_bandwidth_gbps: float = 3350.0,
        interconnect_bandwidth_gbps: float = 64.0,
        sparse_speedup: float = 2.2,
        base_compute_us: float = 120.0,
        prefetch_accuracy: float = 0.85,
        chunks_per_step: int = 3,
        chunk_size_tokens: int = 64,
        decompressor: NearCXLDecompressor | None = None,
    ) -> None:
        self.hbm_bandwidth_gbps = hbm_bandwidth_gbps
        self.interconnect_bandwidth_gbps = interconnect_bandwidth_gbps
        self.sparse_speedup = sparse_speedup
        self.base_compute_us = base_compute_us
        self.prefetch_accuracy = prefetch_accuracy
        self.chunks_per_step = chunks_per_step
        self.chunk_size_tokens = chunk_size_tokens
        self.decompressor = decompressor or NearCXLDecompressor()

    def kv_bytes_per_token(self, num_layers: int, num_heads: int, head_dim: int, bytes_per_elem: int = 2) -> int:
        return 2 * num_layers * num_heads * head_dim * bytes_per_elem

    # ------------------------------------------------------------------
    # Baseline: full-KV dense attention in HBM
    # ------------------------------------------------------------------
    def model_baseline_latency(
        self,
        seq_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
    ) -> LatencyBreakdown:
        kv_bytes = seq_len * self.kv_bytes_per_token(num_layers, num_heads, head_dim)
        hbm_attention_us = (kv_bytes / (self.hbm_bandwidth_gbps * (1024**3))) * 1e6
        total = self.base_compute_us + hbm_attention_us
        return LatencyBreakdown(
            total_us=total,
            compute_us=self.base_compute_us,
            hbm_attention_us=hbm_attention_us,
            promotion_us=0.0,
            decompression_us=0.0,
            exposed_transfer_us=0.0,
            overlap_hidden_us=0.0,
        )

    # ------------------------------------------------------------------
    # KCMC: sparse attention + per-step incremental promotion
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
    ) -> LatencyBreakdown:
        retention_ratio = min(max(retention_ratio, 1e-4), 1.0)
        promotion_ratio = min(max(promotion_ratio, 0.0), 1.0)
        bytes_per_tok = self.kv_bytes_per_token(num_layers, num_heads, head_dim)
        kv_bytes = seq_len * bytes_per_tok

        # 1. Sparse HBM attention: only attend to retained (anchor) tokens
        active_ratio = retention_ratio + promotion_ratio
        sparse_hbm_bytes = kv_bytes * active_ratio
        hbm_attention_us = (sparse_hbm_bytes / (self.hbm_bandwidth_gbps * (1024**3))) * 1e6

        # 2. Compute time with sparse attention speedup
        compute_us = self.base_compute_us / max(self.sparse_speedup, 1.0)

        # 3. Per-step incremental promotion (the key fix)
        #    Each decode step promotes only `chunks_per_step` chunks, not all.
        per_step_promoted_tokens = self.chunks_per_step * self.chunk_size_tokens
        per_step_output_bytes = per_step_promoted_tokens * bytes_per_tok

        decompressor = self.decompressor
        decompressor.compressed_bits = compression_bits
        fetch = decompressor.model_fetch(
            output_bytes=per_step_output_bytes,
            link_bandwidth_gbps=self.interconnect_bandwidth_gbps,
        )
        step_transfer_us = fetch["transfer_time_us"]
        step_decompress_us = fetch["decompression_time_us"]
        step_fetch_us = fetch["effective_fetch_time_us"]

        # 4. Overlap: promotion runs concurrently with compute + HBM attention
        compute_window_us = compute_us + hbm_attention_us
        overlap_hidden_us = min(step_fetch_us, compute_window_us) * self.prefetch_accuracy
        exposed_transfer_us = max(0.0, step_fetch_us - overlap_hidden_us)

        # 5. Total per-step latency
        total = compute_window_us + exposed_transfer_us

        # Report the per-step promotion cost (not the full-sequence aggregate)
        return LatencyBreakdown(
            total_us=total,
            compute_us=compute_us,
            hbm_attention_us=hbm_attention_us,
            promotion_us=step_transfer_us,
            decompression_us=step_decompress_us,
            exposed_transfer_us=exposed_transfer_us,
            overlap_hidden_us=overlap_hidden_us,
        )

    def compute_speedup(self, baseline: LatencyBreakdown, kcmc: LatencyBreakdown) -> float:
        return baseline.total_us / max(kcmc.total_us, 1e-9)

    def summarize_comparison(self, baseline: LatencyBreakdown, kcmc: LatencyBreakdown) -> Dict[str, float]:
        return {
            "baseline_total_us": baseline.total_us,
            "kcmc_total_us": kcmc.total_us,
            "speedup": self.compute_speedup(baseline, kcmc),
            "baseline_hbm_attention_us": baseline.hbm_attention_us,
            "kcmc_hbm_attention_us": kcmc.hbm_attention_us,
            "kcmc_exposed_transfer_us": kcmc.exposed_transfer_us,
            "kcmc_overlap_hidden_us": kcmc.overlap_hidden_us,
            "kcmc_decompression_us": kcmc.decompression_us,
        }