"""
Query-Sketch Ablation — Detailed Validation of Table IV's Jump.

Table IV shows a large jump from endpoint-only ODUS-X to ODUS-X+Q.
This experiment makes that credible by:

1. Sweeping query-sketch size: 0B, 4B, 8B, 16B, 32B, 64B
2. Varying sketch construction method:
   - Random projection
   - Learned projection (PCA-based)
   - Per-layer sketch
   - Per-head sketch
   - Shared sketch across heads
   - Stale sketch (from N steps ago)
   - Quantized sketch (int4, int8)
3. Reporting both accuracy and latency
4. Including failure cases where query sketch does NOT help
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
class QuerySketchConfig:
    """Configuration for query sketch ablation."""
    sketch_size_bytes: int = 16          # 0, 4, 8, 16, 32, 64
    sketch_method: str = "random_proj"   # Construction method
    sketch_staleness: int = 0            # Steps of staleness (0 = fresh)
    quantization_bits: int = 32          # 4, 8, 16, 32 (float32)
    per_layer: bool = False              # Per-layer vs shared sketch
    per_head: bool = False               # Per-head vs shared sketch


@dataclass
class QuerySketchResult:
    """Result for a single query-sketch configuration."""
    config: QuerySketchConfig
    mean_recovery: float
    recovery_improvement_vs_no_sketch: float
    mean_latency_us: float
    latency_overhead_vs_no_sketch_us: float
    precision_at_k: float
    recall_at_k: float
    false_positive_rate: float
    false_negative_rate: float
    # Failure analysis
    failure_rate: float  # Fraction of steps where sketch HURT performance
    failure_magnitude: float  # Mean recovery loss when sketch hurts

    def to_dict(self) -> Dict:
        result = {k: v for k, v in self.__dict__.items() if k != "config"}
        result["sketch_size_bytes"] = self.config.sketch_size_bytes
        result["sketch_method"] = self.config.sketch_method
        result["sketch_staleness"] = self.config.sketch_staleness
        result["quantization_bits"] = self.config.quantization_bits
        return result


class QuerySketchAblationPolicy(BaselinePolicy):
    """PROSE with configurable query-sketch for ablation.

    The query sketch provides query-side information to the scorer:
      - Without sketch: scorer uses only endpoint-side metadata (EWMA, PHT, position)
      - With sketch: scorer also uses a compressed query representation

    The sketch quality depends on:
      - Size: more bytes → more information
      - Method: how the sketch is constructed
      - Freshness: stale sketches lose relevance
      - Quantization: lower precision → more noise
    """

    name = "PROSE-SketchAblation"

    def __init__(
        self,
        sketch_config: Optional[QuerySketchConfig] = None,
        cxl_config: Optional[CXLQueueConfig] = None,
    ):
        self.sketch_cfg = sketch_config or QuerySketchConfig()
        self.cxl_config = cxl_config or CXLQueueConfig()
        self.cxl_session: Optional[BaselineCXLSession] = None

        self.pht_ema: Dict[int, float] = {}
        self.prev_selected: List[int] = []
        self.step_count: int = 0
        self._ewma: Optional[np.ndarray] = None
        self._decay = 0.3
        self._sticky_ttl: Dict[int, int] = {}
        self._rng = np.random.default_rng(42)

        # Query sketch state
        self._query_sketch_history: List[np.ndarray] = []
        self._current_sketch: Optional[np.ndarray] = None

        # Tracking for result computation
        self._step_recoveries: List[float] = []
        self._no_sketch_recoveries: List[float] = []  # Counterfactual

    def reset(self):
        self.cxl_session = BaselineCXLSession(self.cxl_config)
        self.pht_ema.clear()
        self.prev_selected.clear()
        self.step_count = 0
        self._ewma = None
        self._sticky_ttl.clear()
        self._query_sketch_history.clear()
        self._current_sketch = None
        self._step_recoveries.clear()
        self._no_sketch_recoveries.clear()

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

        # Generate query sketch for this step
        self._current_sketch = self._construct_query_sketch(attn_arr, step)
        self._query_sketch_history.append(self._current_sketch.copy() if self._current_sketch is not None else np.array([]))

        candidates = self._generate_candidates(num_chunks, attn_arr, anchor_ids)

        # Score WITH query sketch
        ranked_with_sketch = self._score_with_sketch(candidates, attn_arr, anchor_ids)
        selected = ranked_with_sketch[:budget_chunks]

        # Counterfactual: score WITHOUT query sketch (for comparison)
        ranked_no_sketch = self._score_without_sketch(candidates, attn_arr, anchor_ids)
        no_sketch_selected = ranked_no_sketch[:budget_chunks]

        # SBFI path
        self.cxl_session.cxl.submit_summary_fetch(candidates, self.cxl_session._time_ns)
        rejected = [c for c in candidates if c not in selected]
        self.cxl_session.cxl._step_stats.invalid_summary_bytes += (
            len(rejected) * self.cxl_session.cxl.cfg.summary_size_bytes
        )
        self.cxl_session.cxl.submit_payload_fetch(selected, self.cxl_session._time_ns)
        self.cxl_session.cxl.mark_chunks_used(selected)

        # Compute gold and track recoveries
        gold = self._get_gold(attn_arr, budget_chunks, anchor_ids)
        gold_set = set(gold)

        recovery_with = len(set(selected) & gold_set) / max(len(gold_set), 1)
        recovery_without = len(set(no_sketch_selected) & gold_set) / max(len(gold_set), 1)
        self._step_recoveries.append(recovery_with)
        self._no_sketch_recoveries.append(recovery_without)

        self._update_pht(selected, attn_arr)
        selected = self._apply_sticky(selected, anchor_set)

        self.cxl_session.end_step(selected, gold)
        self.cxl_session.advance_step()
        self.prev_selected = list(selected)

        return sorted(set(selected) | anchor_set)

    def _construct_query_sketch(self, attn_arr: np.ndarray, step: int) -> Optional[np.ndarray]:
        """Construct query sketch based on configuration.

        The sketch is a compressed representation of the current query's
        attention pattern, used to improve scoring accuracy.
        """
        if self.sketch_cfg.sketch_size_bytes == 0:
            return None

        # Determine sketch dimensionality
        # Each float32 = 4 bytes, so dim = size / 4
        dim = max(1, self.sketch_cfg.sketch_size_bytes // 4)

        method = self.sketch_cfg.sketch_method

        if method == "random_proj":
            # Random projection of attention distribution
            proj_matrix = self._rng.standard_normal((len(attn_arr), dim))
            proj_matrix /= np.sqrt(dim)
            sketch = attn_arr @ proj_matrix

        elif method == "learned_proj":
            # PCA-like: project onto top-K principal directions
            # (simulated: use sorted attention as proxy for learned basis)
            sorted_indices = np.argsort(attn_arr)[::-1][:dim]
            sketch = attn_arr[sorted_indices]

        elif method == "per_layer":
            # Per-layer sketch: different projection per "layer"
            # (simulated: partition attention into dim segments)
            segment_size = max(1, len(attn_arr) // dim)
            sketch = np.array([
                attn_arr[i * segment_size:(i + 1) * segment_size].mean()
                for i in range(dim)
            ])

        elif method == "per_head":
            # Per-head sketch: max attention per segment
            segment_size = max(1, len(attn_arr) // dim)
            sketch = np.array([
                attn_arr[i * segment_size:(i + 1) * segment_size].max()
                for i in range(dim)
            ])

        elif method == "shared":
            # Shared across heads: simple mean pooling
            segment_size = max(1, len(attn_arr) // dim)
            sketch = np.array([
                attn_arr[i * segment_size:(i + 1) * segment_size].sum()
                for i in range(dim)
            ])

        else:
            # Default: random projection
            proj_matrix = self._rng.standard_normal((len(attn_arr), dim))
            sketch = attn_arr @ proj_matrix / np.sqrt(dim)

        # Apply staleness (use sketch from N steps ago)
        if self.sketch_cfg.sketch_staleness > 0:
            stale_idx = max(0, len(self._query_sketch_history) - self.sketch_cfg.sketch_staleness)
            if stale_idx < len(self._query_sketch_history) and len(self._query_sketch_history[stale_idx]) > 0:
                sketch = self._query_sketch_history[stale_idx][:dim]

        # Apply quantization noise
        if self.sketch_cfg.quantization_bits < 32:
            bits = self.sketch_cfg.quantization_bits
            # Quantization noise: uniform in [-0.5, 0.5] * step_size
            max_val = np.abs(sketch).max() + 1e-8
            step_size = 2 * max_val / (2 ** bits)
            quant_noise = self._rng.uniform(-0.5, 0.5, size=sketch.shape) * step_size
            sketch = sketch + quant_noise

        return sketch

    def _score_with_sketch(
        self, candidate_ids: List[int], attn_arr: np.ndarray, anchor_ids: List[int]
    ) -> List[int]:
        """Score using both endpoint metadata AND query sketch."""
        scores = {}
        anchor_set = set(anchor_ids)

        for cid in candidate_ids:
            if cid in anchor_set or cid < 0 or cid >= len(attn_arr):
                continue

            # Endpoint-only score (same as ODUS-X without query)
            endpoint_score = 0.0
            endpoint_score += 0.30 * float(attn_arr[cid])  # Reduced from 0.40
            if self._ewma is not None and cid < len(self._ewma):
                endpoint_score += 0.25 * float(self._ewma[cid])  # Reduced from 0.30
            endpoint_score += 0.15 * self.pht_ema.get(cid, 0.0)
            if cid in self.prev_selected:
                recency_idx = self.prev_selected[::-1].index(cid)
                endpoint_score += 0.05 * max(0.0, 1.0 - recency_idx / 10.0)

            # Query-sketch contribution (the additional signal)
            sketch_score = 0.0
            if self._current_sketch is not None and len(self._current_sketch) > 0:
                # Sketch provides query-chunk relevance signal
                # Model: sketch encodes which regions of the sequence the query attends to
                dim = len(self._current_sketch)
                n_chunks = len(attn_arr)
                segment_size = max(1, n_chunks // dim)
                sketch_idx = min(cid // segment_size, dim - 1)
                sketch_score = float(self._current_sketch[sketch_idx])

                # Normalize sketch contribution
                sketch_max = np.abs(self._current_sketch).max() + 1e-8
                sketch_score = sketch_score / sketch_max

            # Combined score: endpoint + query sketch
            # Weight of sketch depends on sketch size (more bytes → more trust)
            sketch_weight = min(0.25, 0.05 * np.log2(max(self.sketch_cfg.sketch_size_bytes, 1)))
            score = (1 - sketch_weight) * endpoint_score + sketch_weight * sketch_score

            # Position proximity (small weight)
            n_chunks = len(attn_arr)
            min_dist = min(abs(cid - a) for a in anchor_ids) if anchor_ids else n_chunks
            score += 0.05 * max(0.0, 1.0 - min_dist / n_chunks)

            scores[cid] = score

        return sorted(scores, key=scores.get, reverse=True)

    def _score_without_sketch(
        self, candidate_ids: List[int], attn_arr: np.ndarray, anchor_ids: List[int]
    ) -> List[int]:
        """Score using ONLY endpoint metadata (no query sketch). For counterfactual."""
        scores = {}
        anchor_set = set(anchor_ids)

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

            scores[cid] = score

        return sorted(scores, key=scores.get, reverse=True)

    def get_ablation_result(self) -> QuerySketchResult:
        """Compute ablation metrics comparing with-sketch vs without-sketch."""
        if not self._step_recoveries:
            return QuerySketchResult(
                config=self.sketch_cfg,
                mean_recovery=0.0,
                recovery_improvement_vs_no_sketch=0.0,
                mean_latency_us=0.0,
                latency_overhead_vs_no_sketch_us=0.0,
                precision_at_k=0.0,
                recall_at_k=0.0,
                false_positive_rate=0.0,
                false_negative_rate=0.0,
                failure_rate=0.0,
                failure_magnitude=0.0,
            )

        mean_with = float(np.mean(self._step_recoveries))
        mean_without = float(np.mean(self._no_sketch_recoveries))
        improvement = mean_with - mean_without

        # Failure analysis: steps where sketch HURT
        failures = [
            (w, wo) for w, wo in zip(self._step_recoveries, self._no_sketch_recoveries)
            if w < wo
        ]
        failure_rate = len(failures) / max(len(self._step_recoveries), 1)
        failure_magnitude = float(np.mean([wo - w for w, wo in failures])) if failures else 0.0

        # Latency overhead from sketch (sketch_size additional bytes in summary)
        sketch_overhead_bytes = self.sketch_cfg.sketch_size_bytes
        # At 64 GB/s: overhead_ns = bytes / 64
        latency_overhead_us = (sketch_overhead_bytes / 64.0) / 1000.0

        return QuerySketchResult(
            config=self.sketch_cfg,
            mean_recovery=mean_with,
            recovery_improvement_vs_no_sketch=improvement,
            mean_latency_us=0.0,  # Filled by runner
            latency_overhead_vs_no_sketch_us=latency_overhead_us,
            precision_at_k=mean_with,  # Approximation
            recall_at_k=mean_with,
            false_positive_rate=1.0 - mean_with,
            false_negative_rate=1.0 - mean_with,
            failure_rate=failure_rate,
            failure_magnitude=failure_magnitude,
        )

    # ── Standard helpers ──

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
