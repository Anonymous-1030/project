"""
ODUS-X: Cross-Context Adaptive Retention Scorer.

Replaces the failed "oracle-distilled attention proxy" with a
workload-adaptive, latency-minimizing ensemble ranker.

Key insight: no lightweight single cue can reliably mirror full attention
mass (validated by A100 experiments: input_emb recall@5 < 15%,
bottom-layer hidden-state recall@5 < 15%). ODUS-X therefore treats
promotion as a multi-objective ranking problem under partial
observability, where a dynamic gating mechanism switches cue weights
based on query-drift and temporal stability.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from src.core_types import ChunkMetadata, QueryContext
from src.promotion.scorer.odus import RuntimeFeatures


class AdaptiveGatingScorer:
    """
    ODUS-X scorer. No offline training required. No attention oracle.
    
    Maintains per-chunk EWMA, PHT, and window buffers internally.
    Gating switches between three operational modes:
      - stable   : temporal locality dominates (sequential / streaming)
      - mixed    : balance between temporal and reactive cues
      - reactive : semantic/structural cues dominate (needle / drift)
    """

    def __init__(
        self,
        force_mode: Optional[str] = None,
        zero_similarity: bool = False,
        zero_pht: bool = False,
    ):
        # Per-chunk persistent state
        self.ewma: Dict[str, float] = {}
        self.window_buffer: Dict[str, List[float]] = {}
        self.pht_score: Dict[str, float] = {}
        self.pht_anchor: Dict[str, bool] = {}

        # Drift detector state
        self.prev_query_sig: Optional[np.ndarray] = None
        self.drift_level: float = 0.0
        self.step_count: int = 0

        # Budget-pressure awareness (injected from scheduler)
        self.budget_pressure: float = 0.0

        # Ablation controls
        self.force_mode = force_mode
        self.zero_similarity = zero_similarity
        self.zero_pht = zero_pht

    def set_budget_pressure(self, pressure: float):
        """Called by scheduler/scorer wrapper before each step."""
        self.budget_pressure = float(pressure)

    def update_drift(self, query: QueryContext):
        """Update drift detector from query signature evolution."""
        if query.query_signature is not None and self.prev_query_sig is not None:
            sim = _cosine_sim(query.query_signature, self.prev_query_sig)
            # sim = 1.0 → identical → drift = 0.0
            # sim = 0.0 → orthogonal → drift = 1.0
            self.drift_level = max(0.0, min(1.0, 1.0 - sim))
        else:
            self.drift_level = 0.0

        if query.query_signature is not None:
            self.prev_query_sig = query.query_signature.copy()
        self.step_count = query.step

    def update_chunk_state(self, chunk: ChunkMetadata, query: QueryContext):
        """Update EWMA, window, and PHT for a single chunk."""
        cid = chunk.chunk_id

        # EWMA: boost if accessed in previous step, otherwise decay
        old_ewma = self.ewma.get(cid, 0.0)
        if chunk.last_access_step >= 0 and chunk.last_access_step == query.step - 1:
            new_ewma = 0.7 * 1.0 + 0.3 * old_ewma
        else:
            new_ewma = old_ewma * 0.85
        self.ewma[cid] = new_ewma

        # Window buffer for FIR-style smoothing
        buf = self.window_buffer.get(cid, [])
        buf.append(new_ewma)
        if len(buf) > 5:
            buf.pop(0)
        self.window_buffer[cid] = buf

        # PHT anchor latch: hardware 1-bit flag if chunk was heavily promoted
        if chunk.promoted_count >= 3:
            self.pht_anchor[cid] = True

        # PHT score: EWMA of promotion success rate
        old_pht = self.pht_score.get(cid, 0.0)
        if chunk.promoted_count > 0:
            success_rate = min(chunk.access_count / max(chunk.promoted_count, 1), 1.0)
        else:
            success_rate = 0.0

        if chunk.last_access_step >= 0 and chunk.last_access_step == query.step - 1:
            new_pht = 0.6 * success_rate + 0.4 * old_pht
        else:
            new_pht = old_pht * 0.85
        self.pht_score[cid] = new_pht

    def score(
        self,
        features: RuntimeFeatures,
        chunk: ChunkMetadata,
        query: QueryContext,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Compute promotion score for a chunk.
        
        Returns:
            (score, confidence, component_dict)
        """
        self.update_drift(query)
        self.update_chunk_state(chunk, query)

        # Determine operational mode from drift level (unless forced for ablation)
        if self.force_mode is not None:
            mode = self.force_mode
        elif self.drift_level > 0.35:
            mode = "reactive"
        elif self.drift_level > 0.15:
            mode = "mixed"
        else:
            mode = "stable"

        # Budget pressure can force more aggressive pruning → higher confidence threshold
        # We reflect this by boosting structural/historical cues under pressure
        pressure_boost = 0.0
        if self.budget_pressure > 0.9:
            pressure_boost = 0.1

        weights = self._get_weights(mode, pressure_boost)

        # Feature normalization
        recency = features.chunk_recency
        position = 1.0 - features.chunk_position
        similarity = max(0.0, features.query_chunk_similarity)
        lexical = features.lexical_overlap
        anchor_dist = 1.0 / (1.0 + features.distance_to_nearest_anchor / 5.0)
        promoted_dist = 1.0 / (1.0 + features.distance_to_promoted / 5.0)
        history = min(features.past_promotion_count / 5.0, 1.0) * features.past_promotion_success_rate

        ewma_val = self.ewma.get(chunk.chunk_id, 0.0)
        buf = self.window_buffer.get(chunk.chunk_id, [ewma_val])
        window_val = float(np.mean(buf)) if buf else ewma_val
        pht_val = self.pht_score.get(chunk.chunk_id, 0.0)
        anchor_bonus = 1.0 if self.pht_anchor.get(chunk.chunk_id, False) else 0.0

        # Apply ablation masks
        if self.zero_similarity:
            similarity = 0.0
        if self.zero_pht:
            pht_val = 0.0
            ewma_val = 0.0
            window_val = 0.0
            anchor_bonus = 0.0

        score = (
            weights["recency"] * recency +
            weights["position"] * position +
            weights["similarity"] * similarity +
            weights["lexical"] * lexical +
            weights["anchor"] * anchor_dist +
            weights["promoted"] * promoted_dist +
            weights["history"] * history +
            weights["ewma"] * ewma_val +
            weights["window"] * window_val +
            weights["pht"] * pht_val +
            weights["anchor_bonus"] * anchor_bonus
        )

        # Confidence: higher at extremes, but budget pressure raises threshold
        confidence = 0.5 + 0.5 * abs(score - 0.5) * 2.0
        if self.budget_pressure > 0.9:
            confidence = min(1.0, confidence * 1.2)

        components = {
            "mode": mode,
            "drift": round(self.drift_level, 3),
            "budget_pressure": round(self.budget_pressure, 3),
            "recency": round(recency, 3),
            "similarity": round(similarity, 3),
            "ewma": round(ewma_val, 3),
            "window": round(window_val, 3),
            "pht": round(pht_val, 3),
        }
        return float(score), float(confidence), components

    def _get_weights(self, mode: str, pressure_boost: float = 0.0) -> Dict[str, float]:
        """Return cue weights for a given operational mode."""
        if mode == "stable":
            w = {
                "recency": 0.30,
                "position": 0.10,
                "similarity": 0.05,
                "lexical": 0.05,
                "anchor": 0.05,
                "promoted": 0.10,
                "history": 0.10,
                "ewma": 0.15,
                "window": 0.05,
                "pht": 0.05,
                "anchor_bonus": 0.00,
            }
        elif mode == "mixed":
            w = {
                "recency": 0.15,
                "position": 0.05,
                "similarity": 0.15,
                "lexical": 0.10,
                "anchor": 0.10,
                "promoted": 0.10,
                "history": 0.10,
                "ewma": 0.10,
                "window": 0.05,
                "pht": 0.05,
                "anchor_bonus": 0.05,
            }
        else:  # reactive
            w = {
                "recency": 0.05,
                "position": 0.00,
                "similarity": 0.25,
                "lexical": 0.15,
                "anchor": 0.20,
                "promoted": 0.05,
                "history": 0.05,
                "ewma": 0.05,
                "window": 0.00,
                "pht": 0.05,
                "anchor_bonus": 0.05,
            }

        # Under budget pressure, boost structural/historical signals
        if pressure_boost > 0.0:
            w["pht"] += pressure_boost
            w["history"] += pressure_boost
            w["anchor_bonus"] += pressure_boost
            # Renormalize roughly by clipping
            for k in w:
                w[k] = min(1.0, w[k])
        return w


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)
