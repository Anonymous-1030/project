"""
Core PPU microarchitecture model (v2.1 — Flash-Decoding compatible).

5-stage pipeline with MMRF-based attention mass ingestion:

  Stage 1: MMRF Receive & Format Cast
      - Receives pre-computed chunk-level attention masses from
        Flash-Decoding's reduction kernel epilogue via memory-mapped
        register file (MMRF).  FP16 → Q0.15 single-cycle cast.
      - Zero overhead: no token-level aggregation, no FlashAttention
        inner-loop intrusion.  See flash_decode_interface.py §4.2.

  Stage 2: Attention Mass Counter Update
      - Saturating counters with shift-based EMA decay.

  Stage 3: 4-signal Feature Extraction
      - Recency, similarity, position, history → quantized LUT index.

  Stage 4: Quantized LUT Lookup
      - Single-cycle SRAM read → utility + confidence.

  Stage 5: DMA Request Enqueue
      - Priority queue with adjacent-chunk coalescing.

=== ON-CHIP LATENCY NOTE (§3.x Physical Deployment) ===
ALL 5 stages execute ON-CHIP, inside the GPU L2 cache controller.
Total pipeline latency: 5 cycles (~5ns @ 1GHz).
This is the "fast path" decision delay — it does NOT traverse the CXL link.

The "slow path" (QFC, ~50.5us) runs on the CXL memory controller and is
only used for remote attention scoring when on-chip information is
insufficient. In the common case, PHT prediction + PPU LUT scoring
complete in <5ns without any CXL traffic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any
from collections import deque
import math

import numpy as np

from src.config import PPUConfig
from src.core_types import ChunkMetadata, QueryContext
from src.hardware.ppu.flash_decode_interface import (
    FlashDecodeEpilogueExporter,
    MMRFConfig,
    MMRFReceiver,
    MMRFReceiveResult,
)


@dataclass
class QuantizedFeatures:
    """Quantized 4-feature representation used to index the LUT."""

    recency: int
    similarity: int
    position: int
    history: int
    packed_index: int
    analog: Dict[str, float] = field(default_factory=dict)


@dataclass
class DMARequest:
    """A single promotion DMA request issued by the PPU."""

    chunk_id: str
    request_id: str
    logical_bytes: int
    priority: int
    utility: float
    source_tier: str = "TAIL"
    coalesced_chunk_ids: List[str] = field(default_factory=list)


@dataclass
class PPUResult:
    """Single-chunk PPU inference result."""

    chunk_id: str
    utility: float
    confidence: float
    lut_index: int
    quantized_features: QuantizedFeatures
    dma_enqueued: bool
    dma_priority: int
    counter_value: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class AttentionMassCounterBank:
    """Saturating per-chunk attention mass counters with shift-based decay."""

    def __init__(self, config: PPUConfig):
        self.config = config
        self._counters: Dict[str, int] = {}
        self._max_value = (1 << config.counter_bits) - 1

    def update(self, chunk_id: str, attention_mass: float) -> int:
        """Accumulate scaled attention mass into a saturating counter."""
        old = self._counters.get(chunk_id, 0)
        increment = max(0, int(round(attention_mass * 256.0)))
        new = min(self._max_value, old + increment)
        self._counters[chunk_id] = new
        self._evict_if_needed()
        return new

    def decay_all(self) -> None:
        """Approximate EMA-like decay with a right shift.

        Each step multiplies the counter by (1 - 2^{-shift}).
        For shift=1 this halves the counter, for shift=2 it keeps 3/4, etc.
        Crucially, when value >> shift == 0 we subtract 1 so that small
        counters always drain to zero and never become "zombie" entries.
        """
        shift = max(0, int(self.config.counter_decay_shift))
        if shift == 0:
            return
        for key, value in list(self._counters.items()):
            drop = value >> shift
            # Guarantee progress: even the smallest non-zero counter decays.
            decayed = value - max(1, drop)
            if decayed <= 0:
                self._counters.pop(key, None)
            else:
                self._counters[key] = decayed

    def get(self, chunk_id: str) -> int:
        return self._counters.get(chunk_id, 0)

    def snapshot(self) -> Dict[str, int]:
        return dict(self._counters)

    def _evict_if_needed(self) -> None:
        limit = max(1, self.config.num_counter_entries)
        if len(self._counters) <= limit:
            return
        # Hardware-faithful enough: retain hottest counters.
        hottest = sorted(self._counters.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        self._counters = dict(hottest)


class FeatureExtractor:
    """Extracts the reduced 4D feature vector used by the hardware LUT."""

    def __init__(self, config: PPUConfig):
        self.config = config

    def extract(
        self,
        chunk: ChunkMetadata,
        query: QueryContext,
        all_chunks: Dict[str, ChunkMetadata],
        attention_counter_value: int = 0,
    ) -> QuantizedFeatures:
        recency_val = self._compute_recency(chunk, query, attention_counter_value)
        similarity_val = self._compute_similarity(chunk, query)
        position_val = float(chunk.position_ratio)
        history_val = self._compute_history(chunk)

        q_recency = self._quantize(recency_val, self.config.recency_bits)
        q_similarity = self._quantize(similarity_val, self.config.similarity_bits)
        q_position = self._quantize(position_val, self.config.position_bits)
        q_history = self._quantize(history_val, self.config.history_bits)
        packed = self._pack(q_recency, q_similarity, q_position, q_history)

        return QuantizedFeatures(
            recency=q_recency,
            similarity=q_similarity,
            position=q_position,
            history=q_history,
            packed_index=packed,
            analog={
                "recency": recency_val,
                "similarity": similarity_val,
                "position": position_val,
                "history": history_val,
                "attention_counter_value": float(attention_counter_value),
                "active_chunks": float(len(all_chunks)),
            },
        )

    def _compute_recency(
        self,
        chunk: ChunkMetadata,
        query: QueryContext,
        attention_counter_value: int,
    ) -> float:
        if chunk.last_access_step >= 0:
            steps_since = max(0, query.step - chunk.last_access_step)
            base = 1.0 / (1.0 + math.log2(steps_since + 1.0))
        else:
            base = 0.0
        counter_bonus = min(1.0, attention_counter_value / float((1 << self.config.counter_bits) - 1 or 1))
        return min(1.0, 0.75 * base + 0.25 * counter_bonus)

    def _compute_similarity(self, chunk: ChunkMetadata, query: QueryContext) -> float:
        """Cosine similarity mapped to [0, 1], consistent with ODUS.

        ODUS uses raw cosine similarity in [-1, 1] as a feature, but the
        PPU LUT requires inputs in [0, 1].  We apply the same affine
        mapping here and in the LUT distiller so the LUT faithfully
        reproduces ODUS behaviour under quantization.
        """
        if query.query_signature is None or chunk.signature is None:
            return 0.0
        a = query.query_signature
        b = chunk.signature
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        raw_cosine = float(np.dot(a_norm, b_norm))  # [-1, 1]
        # Affine map [-1, 1] → [0, 1]: same transform used in lut_distill.py
        return max(0.0, min(1.0, (raw_cosine + 1.0) * 0.5))

    def _compute_history(self, chunk: ChunkMetadata) -> float:
        if chunk.promoted_count <= 0:
            return 0.0
        success = chunk.access_count / max(1, chunk.promoted_count)
        return max(0.0, min(1.0, success))

    def _quantize(self, value: float, bits: int) -> int:
        levels = max(1, (1 << bits) - 1)
        value = max(0.0, min(1.0, float(value)))
        return int(round(value * levels))

    def _pack(self, recency: int, similarity: int, position: int, history: int) -> int:
        per_feature_bits = max(1, self.config.lut_index_bits // 4)
        mask = (1 << per_feature_bits) - 1
        packed = (
            ((recency & mask) << (3 * per_feature_bits))
            | ((similarity & mask) << (2 * per_feature_bits))
            | ((position & mask) << per_feature_bits)
            | (history & mask)
        )
        return packed & ((1 << self.config.lut_index_bits) - 1)


class UtilityLUT:
    """Quantized LUT that approximates ODUS utility predictions."""

    def __init__(self, config: PPUConfig, initial_table: Optional[Sequence[int]] = None):
        self.config = config
        self.num_entries = 1 << config.lut_index_bits
        self.max_output = (1 << config.lut_output_bits) - 1
        if initial_table is None:
            self._table = np.zeros(self.num_entries, dtype=np.uint8)
        else:
            arr = np.asarray(initial_table, dtype=np.int32)
            if arr.size != self.num_entries:
                raise ValueError(f"Expected LUT with {self.num_entries} entries, got {arr.size}")
            self._table = np.clip(arr, 0, self.max_output).astype(np.uint8)

    def lookup(self, index: int) -> Tuple[float, float]:
        raw = int(self._table[index % self.num_entries])
        utility = raw / float(self.max_output or 1)
        confidence = 0.5 + 0.5 * abs(utility - 0.5) * 2.0
        return utility, min(1.0, confidence)

    def program(self, values: Sequence[int]) -> None:
        arr = np.asarray(values, dtype=np.int32)
        if arr.size != self.num_entries:
            raise ValueError(f"Expected {self.num_entries} entries, got {arr.size}")
        self._table = np.clip(arr, 0, self.max_output).astype(np.uint8)

    def dump(self) -> np.ndarray:
        return self._table.copy()


class DMARequestQueue:
    """Finite hardware queue for promotion DMA requests with simple coalescing."""

    def __init__(self, config: PPUConfig):
        self.config = config
        self._queue: deque[DMARequest] = deque(maxlen=config.dma_queue_depth)
        self._dropped = 0

    def enqueue(self, request: DMARequest) -> bool:
        if self._queue and self._can_coalesce(self._queue[-1], request):
            tail = self._queue[-1]
            tail.logical_bytes += request.logical_bytes
            tail.priority = max(tail.priority, request.priority)
            tail.utility = max(tail.utility, request.utility)
            tail.coalesced_chunk_ids.append(request.chunk_id)
            return True
        if len(self._queue) >= self.config.dma_queue_depth:
            self._dropped += 1
            return False
        if not request.coalesced_chunk_ids:
            request.coalesced_chunk_ids = [request.chunk_id]
        self._queue.append(request)
        return True

    def dequeue(self) -> Optional[DMARequest]:
        if not self._queue:
            return None
        return self._queue.popleft()

    def snapshot(self) -> List[DMARequest]:
        return list(self._queue)

    def stats(self) -> Dict[str, int]:
        return {"depth": len(self._queue), "dropped": self._dropped}

    def _can_coalesce(self, a: DMARequest, b: DMARequest) -> bool:
        if a.request_id != b.request_id:
            return False
        return len(a.coalesced_chunk_ids) < max(1, self.config.dma_coalesce_window)


class PromotionPredictionUnit:
    """Top-level hardware model wrapping all PPU subcomponents.

    v2.1 pipeline (5 stages, Flash-Decoding compatible):

      Stage 1: MMRF Receive & Format Cast  (1 cycle)
          Chunk-level masses arrive from Flash-Decoding reduction
          kernel epilogue via memory-mapped register file.  FP16→Q0.15.

      Stage 2: Counter Update              (1 cycle)
          Saturating attention mass counters with shift-based decay.

      Stage 3: Feature Extraction           (1 cycle)
          4D quantized features → packed LUT index.

      Stage 4: LUT Lookup                   (1 cycle)
          Single-cycle SRAM read → utility + confidence.

      Stage 5: DMA Enqueue                  (1 cycle)
          Priority queue with adjacent-chunk coalescing.

    The MMRF stage replaces the old assumption of token-level attention
    aggregation, which was incompatible with FlashAttention's tiled
    online-softmax (attention vector is never materialized).
    """

    # 5-stage pipeline names (used by simulator and CACTI model)
    STAGE_NAMES = (
        "mmrf_receive",
        "counter_update",
        "feature_extract",
        "lut_lookup",
        "dma_enqueue",
    )

    def __init__(self, config: PPUConfig, lut_values: Optional[Sequence[int]] = None):
        self.config = config

        # Stage 1: MMRF receiver (Flash-Decoding interface)
        mmrf_config = MMRFConfig(
            max_chunks=config.num_counter_entries,
            fifo_depth=32,
            data_width_bits=16,
            output_width_bits=16,
            reorder_buffer=True,
        )
        self.mmrf = MMRFReceiver(mmrf_config)
        self.flash_decode = FlashDecodeEpilogueExporter(num_heads=1)

        # Stage 2: Attention mass counters
        self.counters = AttentionMassCounterBank(config)

        # Stage 3: Feature extractor
        self.extractor = FeatureExtractor(config)

        # Stage 4: Utility LUT
        self.lut = UtilityLUT(config, initial_table=lut_values)

        # Stage 5: DMA request queue
        self.dma_queue = DMARequestQueue(config)

        # Per-step MMRF state
        self._step_masses: Dict[str, float] = {}

    def begin_step(self, attention_masses: Dict[str, float]) -> MMRFReceiveResult:
        """Stage 1: Receive chunk masses via MMRF at the start of each step.

        In real hardware, the Flash-Decoding reduction kernel writes
        chunk masses to the MMRF via streaming stores.  The MMRF
        triggers the pipeline when all chunks are valid.

        In simulation, we accept pre-computed masses (from the eval
        framework) and route them through the MMRF functional model
        to maintain cycle-accurate accounting.
        """
        # Convert string chunk_ids to int for MMRF (hardware uses int addresses)
        int_masses = {}
        str_to_int = {}
        for i, (cid, mass) in enumerate(attention_masses.items()):
            int_masses[i] = mass
            str_to_int[i] = cid

        self.mmrf.reset(expected_chunks=len(int_masses))
        self.mmrf.write_batch(int_masses)
        result = self.mmrf.read_all()

        # Store masses keyed by original string chunk_ids for pipeline use
        self._step_masses = dict(attention_masses)
        return result

    def process_candidate(
        self,
        chunk: ChunkMetadata,
        query: QueryContext,
        all_chunks: Dict[str, ChunkMetadata],
        attention_mass: float = 0.0,
        enqueue_threshold: float = 0.0,
    ) -> PPUResult:
        """Stages 2-5: Process a single candidate through the pipeline.

        Stage 1 (MMRF receive) is handled by begin_step() at the start
        of each decode step.  The per-chunk mass is looked up from the
        MMRF buffer rather than passed as a raw float.
        """
        # Use MMRF-buffered mass if available (Flash-Decoding path)
        mass = self._step_masses.get(chunk.chunk_id, attention_mass)

        # Stage 2: Counter update
        counter_value = self.counters.update(chunk.chunk_id, mass)

        # Stage 3: Feature extraction
        q = self.extractor.extract(chunk, query, all_chunks, counter_value)

        # Stage 4: LUT lookup
        utility, confidence = self.lut.lookup(q.packed_index)

        # Stage 5: DMA enqueue
        priority = int(round(utility * ((1 << self.config.dma_priority_bits) - 1)))
        enqueued = False
        if utility >= enqueue_threshold:
            enqueued = self.dma_queue.enqueue(
                DMARequest(
                    chunk_id=chunk.chunk_id,
                    request_id=chunk.request_id,
                    logical_bytes=chunk.logical_bytes,
                    priority=priority,
                    utility=utility,
                )
            )
        return PPUResult(
            chunk_id=chunk.chunk_id,
            utility=utility,
            confidence=confidence,
            lut_index=q.packed_index,
            quantized_features=q,
            dma_enqueued=enqueued,
            dma_priority=priority,
            counter_value=counter_value,
            metadata={
                "attention_mass": mass,
                "mmrf_buffered": chunk.chunk_id in self._step_masses,
            },
        )

    def end_step(self) -> None:
        """End-of-step: decay counters and clear MMRF buffer."""
        self.counters.decay_all()
        self._step_masses.clear()


# ────────────────────────────────────────────────────────────────
# Adaptive PPU (v4.0) — HAAS-integrated promotion prediction
# ────────────────────────────────────────────────────────────────


class AdaptivePromotionPredictionUnit(PromotionPredictionUnit):
    """
    PPU v4.0 with Hardware-Adaptive Admission Scoring (HAAS).

    Extends the v2.1 5-stage pipeline with online learning:

      Stage 4 (replaced):  LUT Lookup  →  Systolic DPE dot-product
      Stage 6 (new):       SGD Weight Update (end-of-step, off critical path)

    The HAAS sub-components (DPE, SGD adapter, PID controller, quantile
    sketch) operate entirely on-chip and never consume SM cycles.

    Pipeline (HAAS mode):
      ┌──────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐
      │  MMRF    │→│ Attention  │→│ Feature    │→│ Systolic     │
      │ Receiver │  │ Mass Ctr   │  │ Extractor  │  │ DPE (1×11)  │
      │(FP16→Q15)│  │(per chunk) │  │(11 signals)│  │ score=Σw·f  │
      └──────────┘ └────────────┘ └────────────┘ └──────┬───────┘
           ↑                                              ↓
      Flash-Decoding              ┌───────────────────────┐
      Reduction Kernel            │ DMA Request Queue      │
                                  │ (PID-adaptive θ)      │
                                  └───────────────────────┘
                                              │
      ┌───────────────────────────────────────┘
      │  Stage 6 (end-of-step, off critical path):
      │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │  │ SGD Weight   │  │ PID Update   │  │ Quantile     │
      │  │ Adapter      │  │ (pressure)   │  │ Sketch       │
      │  └──────────────┘  └──────────────┘  └──────────────┘
    """

    # Extended stage names for HAAS mode
    STAGE_NAMES_HAAS = (
        "mmrf_receive",
        "counter_update",
        "feature_extract",
        "systolic_dpe",
        "dma_enqueue",
        "sgd_update",        # off critical path
    )

    def __init__(
        self,
        config: PPUConfig,
        lut_values: Optional[Sequence[int]] = None,
    ):
        super().__init__(config, lut_values)
        self._haas = None

        if config.haas_enabled:
            self._init_haas()

    def _init_haas(self) -> None:
        """Initialize HAAS sub-components from PPU config."""
        from src.hardware.ppu.adaptive_scorer import (
            HAASConfig,
            HardwareAdaptiveAdmissionScorer,
        )
        from src.hardware.ppu.pid_controller import PIDConfig
        from src.hardware.ppu.quantile_sketch import QuantileSketchConfig

        haas_cfg = HAASConfig(
            num_features=self.config.haas_num_features,
            simd_width=self.config.haas_simd_width,
            sgd_enabled=self.config.haas_sgd_enabled,
            sgd_learning_rate=self.config.haas_sgd_learning_rate,
            sgd_learning_rate_decay=self.config.haas_sgd_lr_decay,
            sgd_min_lr=self.config.haas_sgd_min_lr,
            pid_config=PIDConfig(
                kp=self.config.haas_pid_kp,
                ki=self.config.haas_pid_ki,
                kd=self.config.haas_pid_kd,
                target_pressure=self.config.haas_pid_target_pressure,
            ),
            sketch_config=QuantileSketchConfig(
                num_bins=self.config.haas_sketch_bins,
            ),
            sketch_target_accept_rate=self.config.haas_sketch_target_accept_rate,
            threshold_mode=self.config.haas_threshold_mode,
            blend_alpha=self.config.haas_blend_alpha,
        )
        self._haas = HardwareAdaptiveAdmissionScorer(haas_cfg)
        self._haas_feature_matrix: List[np.ndarray] = []
        self._haas_chunk_ids: List[str] = []

    @property
    def haas(self):
        """Access the HAAS scorer for introspection."""
        return self._haas

    @property
    def is_adaptive(self) -> bool:
        return self._haas is not None

    def process_candidate(
        self,
        chunk: ChunkMetadata,
        query: QueryContext,
        all_chunks: Dict[str, ChunkMetadata],
        attention_mass: float = 0.0,
        enqueue_threshold: float = 0.0,
    ) -> PPUResult:
        """Stages 2-5: Process candidate, using HAAS DPE if enabled.

        When HAAS is enabled, Stage 4 uses the systolic dot-product engine
        instead of the static LUT.  The enqueue_threshold is adaptively
        controlled by the PID + quantile composition.
        """
        mass = self._step_masses.get(chunk.chunk_id, attention_mass)

        # Stage 2: Counter update
        counter_value = self.counters.update(chunk.chunk_id, mass)

        # Stage 3: Feature extraction
        q = self.extractor.extract(chunk, query, all_chunks, counter_value)

        if self._haas is not None:
            # Stage 4 (HAAS): Systolic DPE dot-product with 11-dim features
            features_11d = self._extract_haas_features(
                q, chunk, query, all_chunks, counter_value
            )
            score = self._haas.dpe.compute(features_11d)
            confidence = 0.5 + 0.5 * abs(score - 0.5) * 2.0

            # Adaptive threshold from PID + quantile
            queue_pressure = len(self.dma_queue.snapshot()) / max(1, self.config.dma_queue_depth)
            pid_th = self._haas.pid.step(queue_pressure)
            quantile_th = self._haas.sketch.adaptive_threshold(
                target_accept_rate=self._haas.config.sketch_target_accept_rate,
            )
            adaptive_th = self._haas._compose_threshold(pid_th, quantile_th)
            effective_threshold = max(enqueue_threshold, adaptive_th)

            # Store for end-of-step SGD update
            self._haas_feature_matrix.append(features_11d)
            self._haas_chunk_ids.append(chunk.chunk_id)

            # Update quantile sketch
            self._haas.sketch.update(score)
        else:
            # Stage 4 (standard): LUT lookup
            score, confidence = self.lut.lookup(q.packed_index)
            effective_threshold = enqueue_threshold

        # Stage 5: DMA enqueue
        priority = int(round(score * ((1 << self.config.dma_priority_bits) - 1)))
        enqueued = False
        if score >= effective_threshold:
            enqueued = self.dma_queue.enqueue(
                DMARequest(
                    chunk_id=chunk.chunk_id,
                    request_id=chunk.request_id,
                    logical_bytes=chunk.logical_bytes,
                    priority=priority,
                    utility=score,
                )
            )
        return PPUResult(
            chunk_id=chunk.chunk_id,
            utility=score,
            confidence=confidence,
            lut_index=q.packed_index,
            quantized_features=q,
            dma_enqueued=enqueued,
            dma_priority=priority,
            counter_value=counter_value,
            metadata={
                "attention_mass": mass,
                "mmrf_buffered": chunk.chunk_id in self._step_masses,
                "haas_active": self._haas is not None,
                "adaptive_threshold": effective_threshold if self._haas is not None else enqueue_threshold,
            },
        )

    def _extract_haas_features(
        self,
        q: QuantizedFeatures,
        chunk: ChunkMetadata,
        query: QueryContext,
        all_chunks: Dict[str, ChunkMetadata],
        counter_value: int,
    ) -> np.ndarray:
        """Extract the full 11-dim feature vector for HAAS.

        Mirrors the ODUS-X feature set: recency, position, similarity,
        lexical_overlap, anchor_dist, promoted_dist, history (promo_count
        + success_rate), ewma, window_avg, pht_score, anchor_bonus.
        """
        import math

        # Base features from quantized extractor (analog values)
        analog = q.analog
        recency = analog.get("recency", 0.0)
        similarity = analog.get("similarity", 0.0)
        position = analog.get("position", 0.0)
        history = analog.get("history", 0.0)

        # Lexical overlap: approximate from signature similarity
        lexical = similarity * 0.5  # Proxy: lexical ≈ 0.5 * semantic similarity

        # Anchor distance: normalized
        if hasattr(chunk, 'distance_to_nearest_anchor') and chunk.distance_to_nearest_anchor >= 0:
            anchor_dist = 1.0 / (1.0 + chunk.distance_to_nearest_anchor / 5.0)
        else:
            anchor_dist = 0.1

        # Promoted distance
        if hasattr(chunk, 'distance_to_promoted') and chunk.distance_to_promoted >= 0:
            promoted_dist = 1.0 / (1.0 + chunk.distance_to_promoted / 5.0)
        else:
            promoted_dist = 0.1

        # Promotion history (combined)
        if chunk.promoted_count > 0:
            success_rate = min(chunk.access_count / max(chunk.promoted_count, 1), 1.0)
            promo_history = min(chunk.promoted_count / 5.0, 1.0) * success_rate
        else:
            promo_history = 0.0

        # EWMA (from chunk metadata or counter)
        ewma_val = counter_value / max(1, (1 << self.config.counter_bits) - 1)

        # Window average: crude FIR approximation
        window_val = ewma_val  # Simplified; real impl would use a shift register

        # PHT score (from chunk metadata)
        if chunk.promoted_count >= 3:
            pht_val = 0.8
        elif chunk.promoted_count >= 1:
            pht_val = 0.4
        else:
            pht_val = 0.0

        # Anchor bonus
        anchor_bonus = 1.0 if chunk.promoted_count >= 3 else 0.0

        return np.array([
            recency,       # 0
            position,      # 1
            similarity,    # 2
            lexical,       # 3
            anchor_dist,   # 4
            promoted_dist, # 5
            promo_history, # 6
            ewma_val,      # 7
            window_val,    # 8
            pht_val,       # 9
            anchor_bonus,  # 10
        ], dtype=np.float64)

    def end_step(
        self,
        promoted_chunk_ids: Optional[List[str]] = None,
        accessed_chunk_ids: Optional[set] = None,
    ) -> Optional[dict]:
        """End-of-step: decay counters, run SGD update, return HAAS stats."""
        self.counters.decay_all()

        haas_stats = None
        if self._haas is not None and promoted_chunk_ids is not None:
            # Build feature snapshot from this step's accumulated features
            feature_snapshots = {}
            for cid, feats in zip(self._haas_chunk_ids, self._haas_feature_matrix):
                feature_snapshots[cid] = feats

            # Run SGD weight update
            n_updates = self._haas.record_step_outcomes(
                promoted_ids=promoted_chunk_ids,
                accessed_chunk_ids=accessed_chunk_ids or set(),
                feature_snapshots=feature_snapshots,
            )

            # Decay quantile sketch
            self._haas.sketch.decay()

            haas_stats = {
                "sgd_updates_applied": n_updates,
                **self._haas.get_stats(),
            }

        # Clear step buffers
        self._step_masses.clear()
        if self._haas is not None:
            self._haas_feature_matrix.clear()
            self._haas_chunk_ids.clear()

        return haas_stats

    def get_haas_report(self) -> dict:
        """Return comprehensive HAAS status report."""
        if self._haas is None:
            return {"haas_enabled": False}
        return {
            "haas_enabled": True,
            **self._haas.get_stats(),
            "dpe_cycles_per_dot": self._haas.dpe.estimate_cycles(),
            "dma_queue_depth": len(self.dma_queue.snapshot()),
            "dma_queue_stats": self.dma_queue.stats(),
        }
