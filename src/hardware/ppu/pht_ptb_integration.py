"""
PHT/PTB integration with the existing PPU pipeline (v2.1).

Provides:
- PHTAugmentedFeatureExtractor: 5D features (adds PHT prediction)
- PTBAugmentedPrefetchEngine: speculative prefetch from PTB
- PHTAugmentedPPU: 6-stage pipeline with MMRF + PHT/PTB
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np

from src.config import PPUConfig
from src.core_types import ChunkMetadata, QueryContext
from src.hardware.ppu.ppu_core import (
    AttentionMassCounterBank,
    DMARequest,
    DMARequestQueue,
    FeatureExtractor,
    PPUResult,
    PromotionPredictionUnit,
    QuantizedFeatures,
    UtilityLUT,
)
from src.hardware.ppu.flash_decode_interface import (
    FlashDecodeEpilogueExporter,
    MMRFConfig,
    MMRFReceiver,
    MMRFReceiveResult,
)
from src.hardware.ppu.pht import (
    PHTConfig,
    PHTResult,
    PromotionHistoryTable,
)
from src.hardware.ppu.ptb import (
    PTBConfig,
    PTBEntry,
    PTBLookupResult,
    PromotionTargetBuffer,
)


# ---------------------------------------------------------------------------
# 5D Feature Extractor (PHT as 5th dimension)
# ---------------------------------------------------------------------------

class PHTAugmentedFeatureExtractor(FeatureExtractor):
    """Extended FeatureExtractor that adds PHT prediction as 5th feature.

    Original 4D: [recency, similarity, position, history]
    Augmented 5D: [recency, similarity, position, history, pht_prediction]

    With 5 features × 2 bits each = 10-bit LUT index → 1024 entries.
    """

    def __init__(self, config: PPUConfig, pht: PromotionHistoryTable):
        super().__init__(config)
        self.pht = pht
        # PHT prediction uses 2 bits like other features
        self._pht_bits = 2

    def extract(
        self,
        chunk: ChunkMetadata,
        query: QueryContext,
        all_chunks: Dict[str, ChunkMetadata],
        attention_counter_value: int = 0,
    ) -> QuantizedFeatures:
        """Extract 5D features including PHT prediction."""
        # Compute base 4 features (same as parent)
        recency_val = self._compute_recency(chunk, query, attention_counter_value)
        similarity_val = self._compute_similarity(chunk, query)
        position_val = float(chunk.position_ratio)
        history_val = self._compute_history(chunk)

        # 5th feature: PHT prediction
        pht_val = self.pht.get_prediction_feature(chunk, query)

        # Quantize all 5
        q_recency = self._quantize(recency_val, self.config.recency_bits)
        q_similarity = self._quantize(similarity_val, self.config.similarity_bits)
        q_position = self._quantize(position_val, self.config.position_bits)
        q_history = self._quantize(history_val, self.config.history_bits)
        q_pht = self._quantize(pht_val, self._pht_bits)

        # Pack 5 features into index
        packed = self._pack_5d(
            q_recency, q_similarity, q_position, q_history, q_pht
        )

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
                "pht_prediction": pht_val,
                "attention_counter_value": float(attention_counter_value),
                "active_chunks": float(len(all_chunks)),
            },
        )

    def _pack_5d(
        self,
        recency: int,
        similarity: int,
        position: int,
        history: int,
        pht: int,
    ) -> int:
        """Pack 5 features into a single LUT index.

        Layout (MSB to LSB): recency | similarity | position | history | pht
        Each feature uses 2 bits → 10-bit total → 1024 entries.
        """
        bits = 2  # bits per feature
        mask = (1 << bits) - 1
        packed = (
            ((recency & mask) << (4 * bits))
            | ((similarity & mask) << (3 * bits))
            | ((position & mask) << (2 * bits))
            | ((history & mask) << bits)
            | (pht & mask)
        )
        # Mask to 10 bits
        return packed & 0x3FF


# ---------------------------------------------------------------------------
# PTB-Augmented Prefetch Engine
# ---------------------------------------------------------------------------

@dataclass
class SpeculativePrefetchResult:
    """Result of PTB-driven speculative prefetch."""

    issued_chunk_ids: List[str]
    ptb_hits: int
    pht_predicted_promote: int
    total_bytes: int
    # v2.2: Probation tracking
    probation_chunk_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Probation State (v2.2 — speculative prefetch misprediction mitigation)
# ---------------------------------------------------------------------------

@dataclass
class ProbationEntry:
    """Tracks a speculatively prefetched block in its trial period.

    If the PPU's slow-path LUT scoring disagrees with the PHT prediction,
    or if the block is not accessed within probation_steps, it becomes
    a low-priority eviction candidate.  This prevents HBM pollution from
    wasted speculative prefetches.
    """

    chunk_id: str
    request_id: str
    issued_step: int           # Step when the prefetch was issued
    pht_confidence: float      # PHT confidence at time of prediction
    ppu_utility: float = 0.0   # PPU LUT utility score (filled later)
    confirmed: bool = False     # Whether PPU confirmed the prediction
    evicted: bool = False       # Whether evicted from probation

    @property
    def is_wasted(self) -> bool:
        """Whether this prefetch was confirmed as useless."""
        return self.confirmed and self.ppu_utility < 0.3


class ProbationManager:
    """Manages the probation (trial period) for speculatively prefetched blocks.

    Design rationale (§3.x, Fast Path vs Slow Path consistency):
      - PHT predicts "promote" and PTB hits → speculative DMA prefetch issued
      - If PPU's LUT scoring later shows low utility (< threshold), the block
        is tagged as "Probation" and will be the FIRST candidate for eviction
      - If not accessed within N steps, auto-downgrade to eviction candidate

    This mechanism bounds the worst-case HBM pollution from PHT mispredictions,
    which is critical because the fast path is speculative by design.
    """

    def __init__(
        self,
        probation_steps: int = 8,
        utility_threshold: float = 0.3,
    ):
        self.probation_steps = probation_steps
        self.utility_threshold = utility_threshold

        # Active probation entries: chunk_id → ProbationEntry
        self._entries: Dict[str, ProbationEntry] = {}

        # Stats
        self._total_probation = 0
        self._confirmed_useful = 0
        self._confirmed_wasted = 0
        self._auto_expired = 0
        self._bytes_wasted = 0

    def add_to_probation(
        self,
        chunk_id: str,
        request_id: str,
        step: int,
        pht_confidence: float,
    ) -> None:
        """Add a speculatively prefetched block to probation."""
        self._entries[chunk_id] = ProbationEntry(
            chunk_id=chunk_id,
            request_id=request_id,
            issued_step=step,
            pht_confidence=pht_confidence,
        )
        self._total_probation += 1

    def confirm_with_utility(
        self,
        chunk_id: str,
        ppu_utility: float,
        chunk_bytes: int = 0,
    ) -> Optional[str]:
        """Confirm probation entry with PPU utility score.

        Returns the chunk_id if it should be marked as low-priority eviction
        candidate (wasted prefetch), None if it's useful.
        """
        entry = self._entries.get(chunk_id)
        if entry is None:
            return None

        entry.ppu_utility = ppu_utility
        entry.confirmed = True

        if ppu_utility < self.utility_threshold:
            # Wasted prefetch: mark for eviction
            entry.evicted = True
            self._confirmed_wasted += 1
            self._bytes_wasted += chunk_bytes
            self._entries.pop(chunk_id, None)
            return chunk_id
        else:
            self._confirmed_useful += 1
            self._entries.pop(chunk_id, None)
            return None

    def step_check(self, current_step: int, accessed_ids: Set[str]) -> List[str]:
        """Check probation entries at each step.

        Returns list of chunk_ids that should be downgraded to eviction
        candidates because they were not accessed within probation_steps.
        """
        expired = []
        for cid, entry in list(self._entries.items()):
            if entry.evicted:
                continue
            # Check if accessed
            if cid in accessed_ids:
                self._confirmed_useful += 1
                self._entries.pop(cid, None)
                continue
            # Check if probation period expired
            if current_step - entry.issued_step >= self.probation_steps:
                entry.evicted = True
                expired.append(cid)
                self._auto_expired += 1

        for cid in expired:
            self._entries.pop(cid, None)

        return expired

    def stats(self) -> Dict[str, Any]:
        total = max(1, self._total_probation)
        return {
            "total_probation": self._total_probation,
            "active_probation": len(self._entries),
            "confirmed_useful": self._confirmed_useful,
            "confirmed_wasted": self._confirmed_wasted,
            "auto_expired": self._auto_expired,
            "bytes_wasted": self._bytes_wasted,
            "prefetch_accuracy": self._confirmed_useful / total,
            "useless_prefetch_rate": self._confirmed_wasted / total,
        }


class PTBAugmentedPrefetchEngine:
    """Wraps prefetch logic to add PTB-driven speculative prefetch.

    Before the scorer runs, checks PTB for cached targets whose PHT
    predicts promote. Issues speculative prefetch for these targets.
    If the scorer later confirms, data is already in staging buffer.

    v2.2: Prefetched blocks are placed in Probation (trial period).
    If PPU scoring disagrees or the block goes unaccessed for N steps,
    it is immediately marked as a low-priority eviction candidate.
    """

    def __init__(
        self,
        pht: PromotionHistoryTable,
        ptb: PromotionTargetBuffer,
        max_speculative_prefetches: int = 4,
        probation_steps: int = 8,
    ):
        self.pht = pht
        self.ptb = ptb
        self.max_speculative = max_speculative_prefetches

        # v2.2: Probation manager
        self.probation = ProbationManager(
            probation_steps=probation_steps,
            utility_threshold=0.3,
        )

        # Stats
        self._total_speculative = 0
        self._confirmed = 0
        self._wasted = 0

    def speculative_prefetch(
        self,
        query: QueryContext,
        candidate_chunks: List[ChunkMetadata],
    ) -> SpeculativePrefetchResult:
        """Issue speculative prefetches for PHT-predicted + PTB-cached targets.

        Algorithm:
        1. For each candidate, check PHT prediction
        2. If PHT predicts promote, check PTB for cached target
        3. If PTB hit, issue speculative prefetch (up to max_speculative)
        4. v2.2: Add prefetched blocks to probation for tracking
        """
        issued = []
        probation_ids = []
        ptb_hits = 0
        pht_promotes = 0
        total_bytes = 0

        for chunk in candidate_chunks:
            if len(issued) >= self.max_speculative:
                break

            pht_result = self.pht.predict(chunk, query)
            if not pht_result.predict_promote:
                continue
            pht_promotes += 1

            sig = pht_result.signature
            ptb_result = self.ptb.lookup(sig)
            if ptb_result.hit and ptb_result.entry is not None:
                ptb_hits += 1
                issued.append(chunk.chunk_id)
                total_bytes += chunk.logical_bytes
                self._total_speculative += 1

                # v2.2: Add to probation
                self.probation.add_to_probation(
                    chunk_id=chunk.chunk_id,
                    request_id=chunk.request_id,
                    step=query.step,
                    pht_confidence=pht_result.confidence,
                )
                probation_ids.append(chunk.chunk_id)

        return SpeculativePrefetchResult(
            issued_chunk_ids=issued,
            ptb_hits=ptb_hits,
            pht_predicted_promote=pht_promotes,
            total_bytes=total_bytes,
            probation_chunk_ids=probation_ids,
        )

    def confirm_prefetch(self, chunk_ids: List[str]) -> None:
        """Mark speculative prefetches as confirmed (scorer agreed)."""
        self._confirmed += len(chunk_ids)

    def confirm_with_utility(
        self,
        chunk_id: str,
        ppu_utility: float,
        chunk_bytes: int = 0,
    ) -> Optional[str]:
        """Confirm a probation entry with PPU utility score.

        Returns chunk_id if it's a wasted prefetch (should be evicted).
        This is the v2.2 "slow path validates fast path" mechanism.
        """
        return self.probation.confirm_with_utility(
            chunk_id, ppu_utility, chunk_bytes
        )

    def mark_wasted(self, chunk_ids: List[str]) -> None:
        """Mark speculative prefetches as wasted (scorer disagreed)."""
        self._wasted += len(chunk_ids)

    def step_check(self, current_step: int, accessed_ids: Set[str]) -> List[str]:
        """Check probation entries at each step for auto-expiry.

        Returns list of chunk_ids that should be marked as eviction candidates.
        """
        return self.probation.step_check(current_step, accessed_ids)

    def stats(self) -> Dict[str, Any]:
        total = max(1, self._total_speculative)
        result = {
            "total_speculative": self._total_speculative,
            "confirmed": self._confirmed,
            "wasted": self._wasted,
            "accuracy": self._confirmed / total,
        }
        # v2.2: Include probation stats
        result["probation"] = self.probation.stats()
        return result


# ---------------------------------------------------------------------------
# PHT-Augmented PPU (5-stage pipeline)
# ---------------------------------------------------------------------------

class PHTAugmentedPPU:
    """Extended PPU with PHT and PTB integration (v2.1).

    Pipeline: 6 stages (vs standard 5):
      1. mmrf_receive        (1 cycle, FP16→Q0.15 format cast)
      2. counter_update      (1 cycle)
      3. feature_extract+PHT (1 cycle, parallel — no added latency)
      4. lut_lookup           (1 cycle, now 10-bit index → 1024 entries)
      5. dma_enqueue          (1 cycle)
      6. ptb_update           (1 cycle)

    The MMRF receive stage (v2.1) replaces the old token-level attention
    aggregation assumption with Flash-Decoding-compatible chunk-level
    mass ingestion.  The PHT lookup runs in parallel with feature
    extraction.  The PTB update is appended as a 6th stage.
    """

    # 6-stage pipeline names
    STAGE_NAMES = (
        "mmrf_receive",
        "counter_update",
        "feature_extract_pht",
        "lut_lookup",
        "dma_enqueue",
        "ptb_update",
    )

    def __init__(
        self,
        config: PPUConfig,
        pht_config: PHTConfig,
        ptb_config: PTBConfig,
        lut_values: Optional[Sequence[int]] = None,
    ):
        self.config = config
        self.pht_config = pht_config
        self.ptb_config = ptb_config

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

        # Stage 2: Core PPU components
        self.counters = AttentionMassCounterBank(config)
        self.pht = PromotionHistoryTable(pht_config)
        self.ptb = PromotionTargetBuffer(ptb_config)

        # Stage 3: 5D feature extractor
        self.extractor = PHTAugmentedFeatureExtractor(config, self.pht)

        # Stage 4: LUT (10-bit index with PHT as 5th feature → 1024 entries)
        lut_config = PPUConfig(**{
            **config.__dict__,
            "lut_index_bits": 10,
        })
        self.lut = UtilityLUT(lut_config, initial_table=lut_values)

        # Stage 5: DMA queue
        self.dma_queue = DMARequestQueue(config)

        # Speculative prefetch engine
        self.prefetch_engine = PTBAugmentedPrefetchEngine(
            self.pht, self.ptb
        )

        # Per-step tracking
        self._step_promoted: List[Tuple[ChunkMetadata, QueryContext, int]] = []
        self._step_masses: Dict[str, float] = {}

    def begin_step(self, attention_masses: Dict[str, float]) -> MMRFReceiveResult:
        """Stage 1: Receive chunk masses via MMRF at the start of each step.

        In real hardware, the Flash-Decoding reduction kernel writes
        chunk masses to the MMRF via streaming stores.  The MMRF
        triggers the pipeline when all chunks are valid.
        """
        int_masses = {}
        for i, (cid, mass) in enumerate(attention_masses.items()):
            int_masses[i] = mass

        self.mmrf.reset(expected_chunks=len(int_masses))
        self.mmrf.write_batch(int_masses)
        result = self.mmrf.read_all()

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
        """Process with PHT/PTB augmentation (v2.1).

        Stage 1: MMRF receive (handled by begin_step)
        Stage 2: Counter update
        Stage 3: Feature extraction + PHT lookup (parallel)
        Stage 4: LUT lookup with 5D features
        Stage 5: DMA enqueue
        Stage 6: PTB bookkeeping (deferred to end_step)
        """
        # Use MMRF-buffered mass if available (Flash-Decoding path)
        mass = self._step_masses.get(chunk.chunk_id, attention_mass)

        # Stage 2: Counter update
        counter_value = self.counters.update(chunk.chunk_id, mass)

        # Stage 2: Feature extraction (includes PHT as 5th feature)
        q = self.extractor.extract(chunk, query, all_chunks, counter_value)

        # Also get explicit PHT prediction for metadata
        pht_result = self.pht.predict(chunk, query)

        # PTB lookup (parallel with feature extraction)
        ptb_result = self.ptb.lookup(pht_result.signature)

        # Stage 3: LUT lookup
        utility, confidence = self.lut.lookup(q.packed_index)

        # Boost confidence if PTB hit (cached target = higher confidence)
        if ptb_result.hit:
            confidence = min(1.0, confidence + 0.15)

        # Stage 4: DMA enqueue
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

        # v2.2: Confirm probation entry with PPU utility score.
        # If this chunk was speculatively prefetched (in probation), the
        # slow-path LUT scoring now validates or refutes the fast-path
        # PHT prediction. Wasted prefetches are marked for eviction.
        eviction_candidate = self.prefetch_engine.confirm_with_utility(
            chunk_id=chunk.chunk_id,
            ppu_utility=utility,
            chunk_bytes=chunk.logical_bytes,
        )

        # Track for Stage 5 (deferred PTB update)
        if enqueued:
            self._step_promoted.append(
                (chunk, query, pht_result.signature)
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
                "pht_prediction": pht_result.predict_promote,
                "pht_confidence": pht_result.confidence,
                "pht_counter": pht_result.counter_value,
                "ptb_hit": ptb_result.hit,
                "probation_eviction": eviction_candidate is not None,
            },
        )

    def end_step(
        self,
        promoted_chunk_ids: List[str],
        accessed_chunk_ids: List[str],
        query: QueryContext,
        all_chunks: Dict[str, ChunkMetadata],
    ) -> Dict[str, Any]:
        """End-of-step: update PHT counters, PTB, and probation.

        Stage 5 (deferred): For each promoted chunk:
          - If accessed → PHT.update(was_useful=True), PTB.insert
          - If not accessed → PHT.update(was_useful=False), PTB.invalidate

        v2.2 Probation: Check probation entries for auto-expiry.
        Blocks not accessed within probation_steps are marked as
        low-priority eviction candidates.
        """
        accessed_set = set(accessed_chunk_ids)
        promoted_set = set(promoted_chunk_ids)

        pht_updates = 0
        ptb_inserts = 0
        ptb_invalidations = 0

        for chunk_meta, q_ctx, signature in self._step_promoted:
            was_useful = chunk_meta.chunk_id in accessed_set
            self.pht.update(chunk_meta, q_ctx, was_useful)
            pht_updates += 1

            if was_useful:
                # Insert successful target into PTB
                self.ptb.insert(
                    signature=signature,
                    chunk_id=chunk_meta.chunk_id,
                    chunk_address=chunk_meta.token_start,
                    utility_score=1.0,  # confirmed useful
                    step=q_ctx.step,
                )
                ptb_inserts += 1
            else:
                # Invalidate failed prediction
                self.ptb.invalidate(signature)
                ptb_invalidations += 1

        # v2.2: Probation step check — auto-expire unaccessed blocks
        probation_expired = self.prefetch_engine.step_check(
            current_step=query.step,
            accessed_ids=accessed_set,
        )

        # Age PTB entries
        aged = self.ptb.age_entries(query.step)

        # Decay counters
        self.counters.decay_all()
        self.pht.step()

        # Clear per-step tracking
        self._step_promoted.clear()
        self._step_masses.clear()

        return {
            "pht_updates": pht_updates,
            "ptb_inserts": ptb_inserts,
            "ptb_invalidations": ptb_invalidations,
            "ptb_aged": len(aged),
            "probation_expired": len(probation_expired),
            "probation_expired_ids": probation_expired,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "pht": self.pht.stats(),
            "ptb": self.ptb.stats(),
            "prefetch": self.prefetch_engine.stats(),
            "dma_queue": self.dma_queue.stats(),
        }
