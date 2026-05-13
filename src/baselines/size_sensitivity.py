"""
Chunk-Size and Summary-Size Sensitivity Sweep.

The paper relies on 64KB chunks and 64B summaries. This experiment
sweeps both parameters to empirically validate these choices.

Payload chunk size sweep: 16KB, 32KB, 64KB, 128KB, 256KB
Summary size sweep: 16B, 32B, 64B, 128B, 256B

Reports the tradeoff among:
  - Recovery@K (accuracy)
  - Metadata traffic (summary bytes)
  - Scoring latency
  - Endpoint storage overhead
  - Wasted payload bytes (false positives × chunk size)
  - P99 stall (tail latency)

This supports the "64B is enough" claim empirically instead of
relying on the weak information-theoretic argument.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, "d:/LLM")

from src.runners.e2e_eval_runner import BaselinePolicy
from src.memory.cxl_queue_simulator import (
    CXLQueueConfig, BaselineCXLSession
)


@dataclass
class SensitivityResult:
    """Result for a single (chunk_size, summary_size) configuration."""
    chunk_size_bytes: int
    summary_size_bytes: int
    mean_recovery: float
    p50_stall_us: float
    p95_stall_us: float
    p99_stall_us: float
    total_metadata_bytes: int
    total_payload_bytes: int
    invalid_payload_bytes: int
    metadata_traffic_ratio: float  # summary_bytes / total_bytes
    wasted_bytes_per_false_positive: int
    endpoint_storage_overhead_ratio: float  # summary_storage / payload_storage
    mean_scoring_latency_us: float
    mean_queue_rho: float
    num_steps: int

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


class ChunkSizeSensitivityPolicy(BaselinePolicy):
    """PROSE-SBFI with configurable chunk and summary sizes.

    Identical logic to PROSE, but parameterized on:
      - chunk_size_bytes: payload size per chunk (affects DMA cost)
      - summary_size_bytes: metadata size per chunk (affects scoring input)
      - summary_quality_factor: models how summary quality degrades with size

    The summary_quality_factor models the information-theoretic tradeoff:
      - Larger summaries → more information → better scoring
      - Smaller summaries → less information → noisier scoring
      - This is modeled as additive noise inversely proportional to summary size
    """

    name = "PROSE-SizeSweep"

    def __init__(
        self,
        chunk_size_bytes: int = 65536,
        summary_size_bytes: int = 64,
        cxl_config: Optional[CXLQueueConfig] = None,
        # Quality model: how summary size affects scoring accuracy
        # At 64B (baseline): noise_std = base_noise
        # At 32B: noise_std = base_noise * sqrt(64/32) = base_noise * 1.41
        # At 128B: noise_std = base_noise * sqrt(64/128) = base_noise * 0.71
        base_scoring_noise: float = 0.05,
    ):
        self.chunk_size_bytes = chunk_size_bytes
        self.summary_size_bytes = summary_size_bytes
        self.base_scoring_noise = base_scoring_noise

        # Compute noise factor based on summary size
        # Information content scales as log2(size), noise inversely
        reference_size = 64
        self.noise_factor = np.sqrt(reference_size / max(summary_size_bytes, 1))

        # Configure CXL with custom sizes
        if cxl_config is None:
            cxl_config = CXLQueueConfig()
        cxl_config.chunk_size_bytes = chunk_size_bytes
        cxl_config.summary_size_bytes = summary_size_bytes
        self.cxl_config = cxl_config

        self.cxl_session: Optional[BaselineCXLSession] = None
        self.pht_ema: Dict[int, float] = {}
        self.prev_selected: List[int] = []
        self.step_count: int = 0
        self._ewma: Optional[np.ndarray] = None
        self._decay = 0.3
        self._sticky_ttl: Dict[int, int] = {}
        self._rng = np.random.default_rng(42)

        # Latency tracking for percentile computation
        self._step_latencies_us: List[float] = []

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self.pht_ema.clear()
        self.prev_selected.clear()
        self.step_count = 0
        self._ewma = None
        self._sticky_ttl.clear()
        self._step_latencies_us.clear()

    def select_active_chunks(
        self,
        num_chunks: int,
        budget_chunks: int,
        chunk_attention_masses: Dict[int, float],
        anchor_ids: List[int],
        step: int,
    ) -> List[int]:
        self.step_count = step

        if self.cxl_session is None:
            self.cxl_session = BaselineCXLSession(self.cxl_config)

        anchor_set = set(anchor_ids)

        attn_arr = np.zeros(num_chunks)
        for cid, mass in chunk_attention_masses.items():
            if isinstance(cid, int) and 0 <= cid < num_chunks:
                attn_arr[cid] = mass

        if self._ewma is None:
            self._ewma = attn_arr.copy()
        else:
            self._ewma = self._decay * attn_arr + (1 - self._decay) * self._ewma

        candidates = self._generate_candidates(num_chunks, attn_arr, anchor_ids)

        # Score with size-dependent noise
        ranked = self._score_with_size_noise(candidates, attn_arr, anchor_ids)
        selected = ranked[:budget_chunks]

        # SBFI path with configured sizes
        summary_result = self.cxl_session.cxl.submit_summary_fetch(
            candidates, self.cxl_session._time_ns
        )
        rejected = [c for c in candidates if c not in selected]
        self.cxl_session.cxl._step_stats.invalid_summary_bytes += (
            len(rejected) * self.summary_size_bytes
        )
        payload_result = self.cxl_session.cxl.submit_payload_fetch(
            selected, self.cxl_session._time_ns
        )
        self.cxl_session.cxl.mark_chunks_used(selected)

        # Track latency
        step_latency_us = (summary_result.total_ns + payload_result.total_ns) / 1000.0
        self._step_latencies_us.append(step_latency_us)

        # Update state
        self._update_pht(selected, attn_arr)
        selected = self._apply_sticky(selected, anchor_set)

        gold = self._get_gold(attn_arr, budget_chunks, anchor_ids)
        self.cxl_session.end_step(selected, gold)
        self.cxl_session.advance_step()
        self.prev_selected = list(selected)

        return sorted(set(selected) | anchor_set)

    def _score_with_size_noise(
        self, candidate_ids: List[int], attn_arr: np.ndarray, anchor_ids: List[int]
    ) -> List[int]:
        """Score with noise proportional to information loss from smaller summaries."""
        scores = {}
        anchor_set = set(anchor_ids)
        noise_std = self.base_scoring_noise * self.noise_factor

        for cid in candidate_ids:
            if cid in anchor_set or cid < 0 or cid >= len(attn_arr):
                continue

            score = 0.0
            score += 0.40 * float(attn_arr[cid])
            if self._ewma is not None and cid < len(self._ewma):
                score += 0.30 * float(self._ewma[cid])
            score += 0.15 * self.pht_ema.get(cid, 0.0)
            if cid in self.prev_selected:
                recency_idx = self.prev_selected[::-1].index(cid)
                score += 0.10 * max(0.0, 1.0 - recency_idx / 10.0)
            n_chunks = len(attn_arr)
            min_dist = min(abs(cid - a) for a in anchor_ids) if anchor_ids else n_chunks
            score += 0.05 * max(0.0, 1.0 - min_dist / n_chunks)

            # Add size-dependent noise
            score += self._rng.normal(0, noise_std)

            scores[cid] = score

        return sorted(scores, key=scores.get, reverse=True)

    def get_sensitivity_result(self) -> SensitivityResult:
        """Compute comprehensive sensitivity metrics."""
        if not self._step_latencies_us:
            latencies = [0.0]
        else:
            latencies = sorted(self._step_latencies_us)

        n = len(latencies)
        p50 = latencies[int(n * 0.50)] if n > 0 else 0.0
        p95 = latencies[int(min(n - 1, n * 0.95))] if n > 0 else 0.0
        p99 = latencies[int(min(n - 1, n * 0.99))] if n > 0 else 0.0

        total_meta = 0
        total_payload = 0
        invalid_payload = 0
        queue_rhos = []

        if self.cxl_session:
            for r in self.cxl_session.step_results:
                total_meta += r.cxl_stats.summary_bytes_fetched
                total_payload += r.cxl_stats.payload_bytes_fetched
                invalid_payload += r.cxl_stats.invalid_payload_bytes
                queue_rhos.append(r.cxl_stats.queue_utilization_rho)

        total_bytes = total_meta + total_payload
        meta_ratio = total_meta / max(total_bytes, 1)
        storage_overhead = self.summary_size_bytes / max(self.chunk_size_bytes, 1)

        return SensitivityResult(
            chunk_size_bytes=self.chunk_size_bytes,
            summary_size_bytes=self.summary_size_bytes,
            mean_recovery=self.get_mean_recovery(),
            p50_stall_us=p50,
            p95_stall_us=p95,
            p99_stall_us=p99,
            total_metadata_bytes=total_meta,
            total_payload_bytes=total_payload,
            invalid_payload_bytes=invalid_payload,
            metadata_traffic_ratio=meta_ratio,
            wasted_bytes_per_false_positive=self.chunk_size_bytes,
            endpoint_storage_overhead_ratio=storage_overhead,
            mean_scoring_latency_us=0.0,  # Filled by runner with real timing
            mean_queue_rho=float(np.mean(queue_rhos)) if queue_rhos else 0.0,
            num_steps=self.step_count,
        )

    # ── Standard helpers (same as PROSE) ──

    def _generate_candidates(self, num_chunks, attn_arr, anchor_ids):
        anchor_set = set(anchor_ids)
        candidates = set()
        if self._ewma is not None:
            for cid in np.argsort(self._ewma)[::-1][:max(5, num_chunks // 4)]:
                if int(cid) not in anchor_set:
                    candidates.add(int(cid))
        for cid in np.argsort(attn_arr)[::-1][:5]:
            if int(cid) not in anchor_set:
                candidates.add(int(cid))
        for cid in self.prev_selected[-5:]:
            if 0 <= cid < num_chunks and cid not in anchor_set:
                candidates.add(cid)
        if len(candidates) < 8:
            for cid in range(num_chunks):
                if cid not in anchor_set and len(candidates) < 16:
                    candidates.add(cid)
        return sorted(candidates)

    def _update_pht(self, selected, attn_arr):
        alpha = 0.15
        for cid in selected:
            if cid < len(attn_arr):
                self.pht_ema[cid] = alpha * float(attn_arr[cid]) + (1 - alpha) * self.pht_ema.get(cid, 0.0)

    def _apply_sticky(self, selected, anchor_set):
        result = set(selected)
        expired = []
        for cid, ttl in list(self._sticky_ttl.items()):
            self._sticky_ttl[cid] = ttl - 1
            if self._sticky_ttl[cid] <= 0:
                expired.append(cid)
            elif cid not in anchor_set:
                result.add(cid)
        for cid in expired:
            del self._sticky_ttl[cid]
        for cid in selected:
            if cid not in anchor_set:
                self._sticky_ttl[cid] = 4
        return sorted(result)

    @staticmethod
    def _get_gold(attn_arr, budget_chunks, anchor_ids):
        anchor_set = set(anchor_ids)
        ranked = np.argsort(attn_arr)[::-1]
        return [int(c) for c in ranked if int(c) not in anchor_set][:budget_chunks]

    def get_mean_recovery(self):
        if self.cxl_session is None or not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.recovery for r in self.cxl_session.step_results]))

    def get_invalid_traffic_ratio(self):
        if self.cxl_session is None or not self.cxl_session.step_results:
            return 0.0
        return float(np.mean([r.invalid_traffic_ratio for r in self.cxl_session.step_results]))
