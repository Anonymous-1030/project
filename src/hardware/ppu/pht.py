"""
Promotion History Table (PHT) — branch-predictor-inspired hardware mechanism.

Analogous to a Branch History Table (BHT) in CPU branch prediction:
- Indexed by chunk structural signature (position_hash XOR layer_hash XOR context_hash)
- 2-bit saturating counters tracking promotion success/failure
- Predicts whether a tail chunk should be promoted before the scorer runs

=== PHYSICAL LOCATION & LATENCY ===
PHT is integrated INSIDE the GPU's L2 cache controller (on-chip).
Its 1-cycle query latency is an ON-CHIP decision delay — it does NOT
cross the CXL link.  This is the "fast path" for promotion prediction.

Contrast with QFC (Query-Forwarding Compute) on the CXL memory controller:
  - PHT (on-chip):   1 cycle (~1ns)   — "should we promote?"
  - QFC (off-chip):  ~50.5us          — "what's the attention score?"
These two latencies operate on DIFFERENT paths:
  - Fast path:  PHT → PTB → speculative DMA prefetch (all on-chip)
  - Slow path:  QFC → remote attention compute (cross-CXL, rare)

Area at 7nm: 1024 entries × 2 bits = 256B SRAM + hash unit ≈ 0.004 mm², 3 mW
Latency: 1 cycle (parallel with feature extraction) — ON-CHIP ONLY
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from src.core_types import ChunkMetadata, QueryContext


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PHTConfig:
    """PHT hardware configuration."""

    num_entries: int = 1024
    counter_bits: int = 2            # 2-bit saturating counter
    position_hash_bits: int = 8      # from chunk position_ratio
    layer_hash_bits: int = 4         # from active anchor context
    context_hash_bits: int = 4       # from query signature
    prediction_threshold: int = 2    # >= threshold → predict promote
    enable_periodic_decay: bool = False
    decay_interval_steps: int = 100


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PHTEntry:
    """Single PHT entry: k-bit saturating counter.

    For 2-bit counters:
      0 = strongly_not_promote
      1 = weakly_not_promote
      2 = weakly_promote
      3 = strongly_promote
    """

    counter: int = 1  # start weakly_not (conservative)

    def predict_promote(self, threshold: int = 2) -> bool:
        return self.counter >= threshold

    def update(self, was_useful: bool, max_val: int = 3) -> None:
        if was_useful:
            self.counter = min(max_val, self.counter + 1)
        else:
            self.counter = max(0, self.counter - 1)

    @property
    def confidence(self) -> float:
        """Confidence in [0, 1]. Higher when counter is near extremes."""
        # For 2-bit: 0→1.0, 1→0.33, 2→0.33, 3→1.0
        mid = 1.5
        return min(1.0, abs(self.counter - mid) / mid)


@dataclass
class PHTResult:
    """Result of a single PHT prediction."""

    chunk_id: str
    signature: int
    predict_promote: bool
    confidence: float
    counter_value: int
    feature_value: float  # [0, 1] for PPU feature integration


# ---------------------------------------------------------------------------
# Promotion History Table
# ---------------------------------------------------------------------------

class PromotionHistoryTable:
    """Hardware model for the Promotion History Table.

    Structural analogy to branch prediction:
      Branch Prediction          KV Promotion Prediction
      ─────────────────          ───────────────────────
      PC-indexed BHT             Signature-indexed PHT
      2-bit saturating counter   2-bit saturating counter
      Taken / Not-taken          Promote / Skip
      Misprediction → flush      Misprediction → wasted BW
    """

    def __init__(self, config: PHTConfig):
        self.config = config
        self._max_counter = (1 << config.counter_bits) - 1
        self._table: List[PHTEntry] = [
            PHTEntry(counter=1) for _ in range(config.num_entries)
        ]
        # Statistics
        self._total_predictions = 0
        self._correct_predictions = 0
        self._promote_predictions = 0
        self._skip_predictions = 0
        self._steps_since_decay = 0

    # ── Signature computation ──────────────────────────────────────────

    def compute_signature(self, chunk: ChunkMetadata, query: QueryContext) -> int:
        """Compute structural signature for PHT indexing.

        signature = (position_hash << 8) XOR (layer_hash << 4) XOR context_hash
        Then modulo num_entries.

        position_hash: 8-bit quantization of position_ratio
        layer_hash:    4-bit hash of active anchor pattern
        context_hash:  4-bit hash of query signature
        """
        # Position hash: quantize position_ratio to 8 bits
        pos_q = int(chunk.position_ratio * 255.0) & 0xFF

        # Layer hash: hash of active anchor IDs (proxy for attention pattern)
        layer_h = 0
        for aid in query.active_anchor_ids[:4]:
            layer_h ^= hash(aid) & 0xF
        layer_h &= 0xF

        # Context hash: hash of query signature
        ctx_h = 0
        if query.query_signature is not None:
            # Use first few bytes of signature
            sig_bytes = query.query_signature.tobytes()[:4]
            for b in sig_bytes:
                ctx_h ^= b
        ctx_h &= 0xF

        # Use prime-multiplied hashing to avoid modulo-induced collision
        raw = (pos_q * 73856093) ^ (layer_h * 19349663) ^ (ctx_h * 83492791)
        return raw % self.config.num_entries

    # ── Prediction ─────────────────────────────────────────────────────

    def predict(self, chunk: ChunkMetadata, query: QueryContext) -> PHTResult:
        """Predict whether chunk should be promoted."""
        sig = self.compute_signature(chunk, query)
        entry = self._table[sig]
        should_promote = entry.predict_promote(self.config.prediction_threshold)
        conf = entry.confidence
        feat = entry.counter / float(self._max_counter)  # [0, 1]

        self._total_predictions += 1
        if should_promote:
            self._promote_predictions += 1
        else:
            self._skip_predictions += 1

        return PHTResult(
            chunk_id=chunk.chunk_id,
            signature=sig,
            predict_promote=should_promote,
            confidence=conf,
            counter_value=entry.counter,
            feature_value=feat,
        )

    def batch_predict(
        self,
        chunks: List[ChunkMetadata],
        query: QueryContext,
    ) -> List[PHTResult]:
        """Batch prediction for pipeline efficiency."""
        return [self.predict(c, query) for c in chunks]

    def get_prediction_feature(
        self, chunk: ChunkMetadata, query: QueryContext
    ) -> float:
        """Return PHT prediction as [0, 1] feature for the 5th PPU dimension.

        Maps counter value linearly: 0→0.0, 1→0.33, 2→0.67, 3→1.0
        """
        sig = self.compute_signature(chunk, query)
        entry = self._table[sig]
        return entry.counter / float(self._max_counter)

    # ── Update ─────────────────────────────────────────────────────────

    def update(
        self,
        chunk: ChunkMetadata,
        query: QueryContext,
        was_useful: bool,
    ) -> None:
        """Update PHT after observing promotion outcome.

        was_useful=True  → promoted chunk was actually accessed (hit)
        was_useful=False → promoted chunk was NOT accessed (miss / wasted BW)
        """
        sig = self.compute_signature(chunk, query)
        entry = self._table[sig]

        # Track accuracy before update
        predicted = entry.predict_promote(self.config.prediction_threshold)
        if predicted == was_useful:
            self._correct_predictions += 1

        entry.update(was_useful, max_val=self._max_counter)

    def batch_update(
        self,
        chunks: List[ChunkMetadata],
        query: QueryContext,
        outcomes: List[bool],
    ) -> None:
        """Batch update for multiple promotion outcomes."""
        for chunk, useful in zip(chunks, outcomes):
            self.update(chunk, query, useful)

    # ── Periodic decay ─────────────────────────────────────────────────

    def step(self) -> None:
        """Called once per decode step. Handles periodic decay."""
        self._steps_since_decay += 1
        if (
            self.config.enable_periodic_decay
            and self._steps_since_decay >= self.config.decay_interval_steps
        ):
            self.decay_all()
            self._steps_since_decay = 0

    def decay_all(self) -> None:
        """Shift all counters toward neutral (1 for 2-bit).

        Strongly promote (3) → weakly promote (2)
        Weakly promote (2) → stays (2)
        Weakly not (1) → stays (1)
        Strongly not (0) → weakly not (1)
        """
        neutral = self._max_counter // 2  # 1 for 2-bit
        for entry in self._table:
            if entry.counter > neutral + 1:
                entry.counter -= 1
            elif entry.counter < neutral:
                entry.counter += 1

    # ── Statistics ─────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return prediction accuracy, coverage, counter distribution."""
        counter_dist = [0] * (self._max_counter + 1)
        for entry in self._table:
            counter_dist[entry.counter] += 1

        accuracy = (
            self._correct_predictions / max(1, self._total_predictions)
        )
        promote_rate = (
            self._promote_predictions / max(1, self._total_predictions)
        )

        return {
            "total_predictions": self._total_predictions,
            "correct_predictions": self._correct_predictions,
            "accuracy": accuracy,
            "promote_rate": promote_rate,
            "counter_distribution": counter_dist,
            "num_entries": self.config.num_entries,
            "counter_bits": self.config.counter_bits,
        }

    def reset_stats(self) -> None:
        self._total_predictions = 0
        self._correct_predictions = 0
        self._promote_predictions = 0
        self._skip_predictions = 0
