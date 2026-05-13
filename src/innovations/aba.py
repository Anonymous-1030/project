"""
Adaptive Bandwidth Arbitrage (ABA).

Solves the "1.0x at high bandwidth" problem by switching PROSE's operational
mode based on measured link bandwidth and queue pressure.

Key insight: At 32 GB/s, full-KV fetch latency is already hidden by compute.
PROSE's value at high bandwidth comes from reducing attention COMPUTE (sparse
block mask), not memory transfer. At low bandwidth, the value comes from
reducing DMA volume via selective promotion.

Three modes:
  - Transparent (BW > 16 GB/s, low pressure): sparse attention mask only
  - Hybrid (4-16 GB/s): standard PROSE pipeline with prefetch overlap
  - Aggressive (BW < 4 GB/s or high queue pressure): full pipeline + compression
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from src.core_types import (
    ChunkMetadata, QueryContext, PromotionPipelineResult,
    ScoredCandidate, SchedulerResult, SchedulerDecision,
    PromoteDecision, FailureReason,
)

logger = logging.getLogger(__name__)


class ABAMode(str, Enum):
    TRANSPARENT = "transparent"
    HYBRID = "hybrid"
    AGGRESSIVE = "aggressive"


@dataclass
class ABAConfig:
    high_bw_threshold_gbps: float = 16.0
    low_bw_threshold_gbps: float = 4.0
    high_pressure_queue_depth: int = 8
    low_pressure_queue_depth: int = 4
    mode_switch_hysteresis_steps: int = 3
    sparse_attention_budget_ratio: float = 0.25
    aggressive_compression_ratio: float = 0.5
    aggressive_prefetch_lookahead: int = 3
    enable_mode_logging: bool = True


@dataclass
class ABAMetrics:
    mode_history: List[str] = field(default_factory=list)
    mode_switches: int = 0
    transparent_steps: int = 0
    hybrid_steps: int = 0
    aggressive_steps: int = 0
    compute_savings_flops: float = 0.0
    bandwidth_savings_bytes: float = 0.0
    total_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        total = max(self.total_steps, 1)
        return {
            "mode_switches": self.mode_switches,
            "transparent_fraction": self.transparent_steps / total,
            "hybrid_fraction": self.hybrid_steps / total,
            "aggressive_fraction": self.aggressive_steps / total,
            "compute_savings_flops": self.compute_savings_flops,
            "bandwidth_savings_bytes": self.bandwidth_savings_bytes,
            "total_steps": self.total_steps,
        }


class AdaptiveBandwidthArbitrage:
    """
    Runtime bandwidth-aware mode selector for PROSE pipeline.

    Monitors measured link bandwidth and queue depth to select the optimal
    operating mode each decode step.
    """

    def __init__(self, config: Optional[ABAConfig] = None):
        self.config = config or ABAConfig()
        self.current_mode = ABAMode.HYBRID
        self._mode_hold_counter = 0
        self._measured_bw_history: List[float] = []
        self._queue_depth_history: List[int] = []
        self.metrics = ABAMetrics()

    def update_measurements(
        self,
        measured_bw_gbps: float,
        queue_depth: int,
        step: int,
    ) -> ABAMode:
        """Update bandwidth/pressure measurements and select mode."""
        self._measured_bw_history.append(measured_bw_gbps)
        self._queue_depth_history.append(queue_depth)
        if len(self._measured_bw_history) > 16:
            self._measured_bw_history.pop(0)
            self._queue_depth_history.pop(0)

        smoothed_bw = np.mean(self._measured_bw_history[-4:])
        smoothed_qd = np.mean(self._queue_depth_history[-4:])

        new_mode = self._select_mode(smoothed_bw, smoothed_qd)

        if new_mode != self.current_mode:
            self._mode_hold_counter += 1
            if self._mode_hold_counter >= self.config.mode_switch_hysteresis_steps:
                old_mode = self.current_mode
                self.current_mode = new_mode
                self._mode_hold_counter = 0
                self.metrics.mode_switches += 1
                if self.config.enable_mode_logging:
                    logger.info(
                        f"ABA mode switch: {old_mode.value} -> {new_mode.value} "
                        f"(bw={smoothed_bw:.1f} GB/s, qd={smoothed_qd:.1f})"
                    )
        else:
            self._mode_hold_counter = 0

        self.metrics.total_steps += 1
        self.metrics.mode_history.append(self.current_mode.value)
        if self.current_mode == ABAMode.TRANSPARENT:
            self.metrics.transparent_steps += 1
        elif self.current_mode == ABAMode.HYBRID:
            self.metrics.hybrid_steps += 1
        else:
            self.metrics.aggressive_steps += 1

        return self.current_mode

    def _select_mode(self, bw_gbps: float, queue_depth: float) -> ABAMode:
        if bw_gbps > self.config.high_bw_threshold_gbps and queue_depth < self.config.low_pressure_queue_depth:
            return ABAMode.TRANSPARENT
        elif bw_gbps < self.config.low_bw_threshold_gbps or queue_depth > self.config.high_pressure_queue_depth:
            return ABAMode.AGGRESSIVE
        else:
            return ABAMode.HYBRID

    def apply_transparent_mode(
        self,
        all_chunks: Dict[str, ChunkMetadata],
        query: QueryContext,
        anchor_ids: List[str],
        budget_ratio: float = 0.25,
    ) -> PromotionPipelineResult:
        """
        Transparent mode: no promotion pipeline, just sparse attention mask.

        Selects top chunks by lightweight scoring (recency + position) to
        build a sparse block mask. Advantage comes from compute reduction.
        """
        start = time.time()
        n_total = len(all_chunks)
        n_visible = max(1, int(n_total * budget_ratio))

        scored = []
        for cid, chunk in all_chunks.items():
            if cid in anchor_ids:
                continue
            recency = 1.0 / (1.0 + max(0, query.step - chunk.last_access_step))
            position = 1.0 - chunk.position_ratio
            score = 0.6 * recency + 0.4 * position
            scored.append((cid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        selected_ids = [cid for cid, _ in scored[:n_visible]]
        final_visible = list(anchor_ids) + selected_ids

        compute_saved = (n_total - len(final_visible)) * 512 * 128
        self.metrics.compute_savings_flops += compute_saved

        latency_us = (time.time() - start) * 1e6
        return PromotionPipelineResult(
            request_id=query.request_id,
            step=query.step,
            final_visible_ids=final_visible,
            final_active_bytes=len(final_visible) * 512 * 2,
            total_latency_us=latency_us,
        )

    def apply_aggressive_mode(
        self,
        pipeline_result: PromotionPipelineResult,
        all_chunks: Dict[str, ChunkMetadata],
        query: QueryContext,
    ) -> PromotionPipelineResult:
        """
        Aggressive mode: enhance pipeline result with compression + prefetch.

        Adds speculative prefetch candidates and applies compression hints.
        """
        start = time.time()

        prefetch_ids = self._predict_next_step_needs(all_chunks, query)
        enhanced_visible = list(set(pipeline_result.final_visible_ids + prefetch_ids))

        bw_saved = len(prefetch_ids) * 512 * 2 * self.config.aggressive_compression_ratio
        self.metrics.bandwidth_savings_bytes += bw_saved

        latency_us = (time.time() - start) * 1e6
        return PromotionPipelineResult(
            request_id=pipeline_result.request_id,
            step=pipeline_result.step,
            ulf_result=pipeline_result.ulf_result,
            scorer_result=pipeline_result.scorer_result,
            scheduler_result=pipeline_result.scheduler_result,
            burst_result=pipeline_result.burst_result,
            sticky_result=pipeline_result.sticky_result,
            final_visible_ids=enhanced_visible,
            final_active_bytes=len(enhanced_visible) * 512 * 2,
            final_promoted_bytes=pipeline_result.final_promoted_bytes,
            total_latency_us=pipeline_result.total_latency_us + latency_us,
        )

    def _predict_next_step_needs(
        self,
        all_chunks: Dict[str, ChunkMetadata],
        query: QueryContext,
    ) -> List[str]:
        """Predict chunks likely needed in next N steps based on access patterns."""
        candidates = []
        for cid, chunk in all_chunks.items():
            if chunk.promoted_count > 0 and chunk.sticky_ttl == 0:
                recency = 1.0 / (1.0 + max(0, query.step - chunk.last_access_step))
                if recency > 0.3:
                    candidates.append((cid, recency))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in candidates[:self.config.aggressive_prefetch_lookahead]]

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.to_dict()

    def reset(self):
        self.current_mode = ABAMode.HYBRID
        self._mode_hold_counter = 0
        self._measured_bw_history.clear()
        self._queue_depth_history.clear()
        self.metrics = ABAMetrics()
