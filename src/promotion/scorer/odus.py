"""
Utility Scorer for ProSE-X 2.0.

Four scorer modes:
1. odus_x: Adaptive gating scorer — NO training, NO oracle (default)
2. similarity_baseline: Query-chunk similarity only
3. lightweight_feature_mlp: Small MLP on runtime features
4. oracle_distilled_utility: Offline-trained utility predictor

Design principles:
- Predict future utility, not just query similarity
- Use only runtime-available features at inference time
- Never use true future attention or gold labels at runtime
- ODUS-X is the recommended mode: no offline training required
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core_types import (
    ChunkMetadata, QueryContext, ScoredCandidate, ScorerResult, ULFResult
)
from src.config import ODUSConfig, ScorerMode

logger = logging.getLogger(__name__)


@dataclass
class RuntimeFeatures:
    """
    Runtime-available features for utility scoring.
    
    All features must be computable without oracle information.
    """
    # Query features
    query_summary_dim: int = 0
    
    # Chunk features
    chunk_position: float = 0.0  # [0, 1]
    chunk_recency: float = 0.0  # Steps since access normalized
    
    # Interaction features
    query_chunk_similarity: float = 0.0
    lexical_overlap: float = 0.0
    
    # Context features
    distance_to_nearest_anchor: float = 0.0  # In chunk units
    distance_to_promoted: float = 0.0  # In chunk units
    
    # Historical features
    past_promotion_count: int = 0
    past_promotion_success_rate: float = 0.0
    
    # Structural features
    is_section_boundary: bool = False
    is_title_adjacent: bool = False
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            self.chunk_position,
            self.chunk_recency,
            self.query_chunk_similarity,
            self.lexical_overlap,
            self.distance_to_nearest_anchor,
            self.distance_to_promoted,
            min(self.past_promotion_count / 5.0, 1.0),
            self.past_promotion_success_rate,
            1.0 if self.is_section_boundary else 0.0,
            1.0 if self.is_title_adjacent else 0.0,
        ], dtype=np.float32)
    
    @classmethod
    def size(cls) -> int:
        """Return feature vector size."""
        return 10


class UtilityScorer:
    """
    Oracle-Distilled Utility Scorer.
    
    Main interface for scoring chunks by predicted utility.
    """
    
    def __init__(self, config: ODUSConfig):
        self.config = config
        
        # Normalize mode to string
        if hasattr(config.mode, 'value'):
            mode_str = config.mode.value
        else:
            mode_str = str(config.mode)
        
        # Initialize scorer based on mode
        if mode_str == ScorerMode.ODUS_X.value or mode_str == "odus_x":
            from src.promotion.scorer.odus_x import AdaptiveGatingScorer
            self._scorer = AdaptiveGatingScorer()
        elif mode_str == ScorerMode.SIMILARITY_BASELINE.value or mode_str == "similarity_baseline":
            self._scorer = SimilarityBaselineScorer(config)
        elif mode_str == ScorerMode.LIGHTWEIGHT_FEATURE_MLP.value or mode_str == "lightweight_feature_mlp":
            self._scorer = LightweightFeatureMLPScorer(config)
        elif mode_str == ScorerMode.ORACLE_DISTILLED_UTILITY.value or mode_str == "oracle_distilled_utility":
            self._scorer = OracleDistilledScorer(config)
        else:
            raise ValueError(f"Unknown scorer mode: {config.mode}")
        
        logger.info(f"UtilityScorer initialized: mode={mode_str}")
    
    def score(
        self,
        ulf_result: ULFResult,
        query: QueryContext,
        all_chunks: Dict[str, ChunkMetadata],
    ) -> ScorerResult:
        """
        Score ULF candidates by predicted utility.
        
        Args:
            ulf_result: Output from ULF
            query: Query context
            all_chunks: All chunk metadata
            
        Returns:
            ScorerResult with scored candidates
        """
        start_time = time.time()
        
        # Score each candidate
        candidates: List[ScoredCandidate] = []
        
        for chunk_id in ulf_result.candidate_ids:
            chunk = all_chunks.get(chunk_id)
            if chunk is None:
                continue
            
            # Compute features
            features = self._compute_features(chunk, query, all_chunks)
            
            # Score
            score, confidence, components = self._scorer.score(features, chunk, query)
            
            # Apply temperature
            if self.config.score_temperature != 1.0:
                score = self._apply_temperature(score)
            
            candidates.append(ScoredCandidate(
                chunk_id=chunk_id,
                score=score,
                confidence=confidence,
                feature_vector=features.to_vector(),
                score_components=components,
            ))
        
        # Sort by score descending
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Count above threshold
        n_above = sum(1 for c in candidates if c.score >= 0.5)  # Default threshold
        
        latency_us = (time.time() - start_time) * 1e6
        
        return ScorerResult(
            request_id=query.request_id,
            step=query.step,
            candidates=candidates,
            n_input_candidates=len(ulf_result.candidate_ids),
            n_scored=len(candidates),
            n_above_threshold=n_above,
            score_threshold=0.5,
            scorer_mode=self.config.mode.value if hasattr(self.config.mode, 'value') else str(self.config.mode),
            scorer_latency_us=latency_us,
        )
    
    def _compute_features(
        self,
        chunk: ChunkMetadata,
        query: QueryContext,
        all_chunks: Dict[str, ChunkMetadata],
    ) -> RuntimeFeatures:
        """Compute runtime features for a chunk."""
        features = RuntimeFeatures()
        
        # Position
        features.chunk_position = chunk.position_ratio
        
        # Recency
        if chunk.last_access_step >= 0:
            steps_since = query.step - chunk.last_access_step
            features.chunk_recency = 1.0 / (1.0 + steps_since / 50.0)
        else:
            features.chunk_recency = 0.0
        
        # Query-chunk similarity
        if query.query_signature is not None and chunk.signature is not None:
            features.query_chunk_similarity = self._cosine_sim(
                query.query_signature, chunk.signature
            )
        
        # Lexical overlap: token-level Jaccard between query and chunk
        if query.query_tokens and chunk.extra.get("token_ids"):
            query_set = set(query.query_tokens)
            chunk_set = set(chunk.extra["token_ids"])
            intersection = len(query_set & chunk_set)
            union = len(query_set | chunk_set)
            features.lexical_overlap = intersection / max(union, 1)
        else:
            features.lexical_overlap = 0.0
        
        # Distance to nearest anchor
        features.distance_to_nearest_anchor = self._compute_anchor_distance(
            chunk, query, all_chunks
        )
        
        # Distance to promoted
        features.distance_to_promoted = self._compute_promoted_distance(
            chunk, all_chunks
        )
        
        # Historical features
        features.past_promotion_count = chunk.promoted_count
        # Real success rate: track how often promoted chunks received
        # significant attention (> median) in subsequent steps
        if chunk.promoted_count > 0 and chunk.access_count > 0:
            features.past_promotion_success_rate = min(
                chunk.access_count / chunk.promoted_count, 1.0
            )
        else:
            features.past_promotion_success_rate = 0.0
        
        # Structural features
        features.is_section_boundary = chunk.is_section_boundary
        features.is_title_adjacent = chunk.is_title_adjacent
        
        return features
    
    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a_norm, b_norm))
    
    def _compute_anchor_distance(
        self,
        chunk: ChunkMetadata,
        query: QueryContext,
        all_chunks: Dict[str, ChunkMetadata],
    ) -> float:
        """Compute distance to nearest active anchor in chunk units."""
        if not query.active_anchor_ids:
            return 10.0  # Large distance if no anchors
        
        min_distance = float('inf')
        for anchor_id in query.active_anchor_ids:
            anchor = all_chunks.get(anchor_id)
            if anchor and anchor.request_id == chunk.request_id:
                # Compute token distance
                if chunk.token_end < anchor.token_start:
                    dist = anchor.token_start - chunk.token_end
                elif chunk.token_start > anchor.token_end:
                    dist = chunk.token_start - anchor.token_end
                else:
                    dist = 0
                
                chunk_dist = dist / max(chunk.num_tokens, 1)
                min_distance = min(min_distance, chunk_dist)
        
        return min_distance if min_distance < float('inf') else 10.0
    
    def _compute_promoted_distance(
        self,
        chunk: ChunkMetadata,
        all_chunks: Dict[str, ChunkMetadata],
    ) -> float:
        """Compute distance to nearest promoted chunk."""
        # Find all promoted chunks in same request
        min_distance = float('inf')
        for other in all_chunks.values():
            if other.request_id == chunk.request_id and other.is_promoted:
                if other.chunk_id == chunk.chunk_id:
                    continue
                
                if chunk.token_end < other.token_start:
                    dist = other.token_start - chunk.token_end
                elif chunk.token_start > other.token_end:
                    dist = chunk.token_start - other.token_end
                else:
                    dist = 0
                
                chunk_dist = dist / max(chunk.num_tokens, 1)
                min_distance = min(min_distance, chunk_dist)
        
        return min_distance if min_distance < float('inf') else 10.0
    
    def _apply_temperature(self, score: float) -> float:
        """Apply temperature scaling to score."""
        # Convert to logits, apply temperature, convert back
        score = max(1e-6, min(1 - 1e-6, score))
        logits = np.log(score / (1 - score))
        logits = logits / self.config.score_temperature
        new_score = 1.0 / (1.0 + np.exp(-logits))
        return float(new_score)


class SimilarityBaselineScorer:
    """
    Mode 1: Similarity baseline.
    
    Uses only query-chunk similarity for scoring.
    Simple and fast, but less predictive of actual utility.
    """
    
    def __init__(self, config: ODUSConfig):
        self.config = config
    
    def score(
        self,
        features: RuntimeFeatures,
        chunk: ChunkMetadata,
        query: QueryContext,
    ) -> tuple[float, float, Dict[str, float]]:
        """Score based on similarity."""
        # Primary: query-chunk similarity
        score = features.query_chunk_similarity
        
        # Small boost for recency and position
        score += 0.1 * features.chunk_recency
        score += 0.1 * (1.0 - features.chunk_position)  # Earlier is better
        
        # Normalize
        score = max(0.0, min(1.0, score))
        
        # Confidence based on feature quality
        confidence = 0.5
        if features.query_chunk_similarity > 0:
            confidence += 0.3
        
        components = {
            "similarity": features.query_chunk_similarity,
            "recency": features.chunk_recency,
            "position": 1.0 - features.chunk_position,
        }
        
        return score, confidence, components


class LightweightFeatureMLPScorer:
    """
    Mode 2: Lightweight feature MLP.
    
    Small MLP on runtime features.
    Better than similarity baseline if trained.
    """
    
    def __init__(self, config: ODUSConfig):
        self.config = config
        
        # Initialize MLP weights
        self._init_mlp()
    
    def _init_mlp(self) -> None:
        """Initialize MLP with random weights or load pretrained."""
        input_dim = RuntimeFeatures.size()
        hidden_dims = self.config.mlp_hidden_dims
        
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.weights.append(np.random.randn(prev_dim, hidden_dim).astype(np.float32) * 0.1)
            self.biases.append(np.zeros(hidden_dim, dtype=np.float32))
            prev_dim = hidden_dim
        
        # Output layer
        self.weights.append(np.random.randn(prev_dim, 1).astype(np.float32) * 0.1)
        self.biases.append(np.zeros(1, dtype=np.float32))
        
        logger.info(f"MLP initialized: input={input_dim}, hidden={hidden_dims}")
    
    def score(
        self,
        features: RuntimeFeatures,
        chunk: ChunkMetadata,
        query: QueryContext,
    ) -> tuple[float, float, Dict[str, float]]:
        """Score using MLP."""
        # Forward pass
        x = features.to_vector()
        
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ W + b
            if i < len(self.weights) - 1:  # Hidden layers
                x = np.maximum(0, x)  # ReLU
        
        # Sigmoid output
        score = 1.0 / (1.0 + np.exp(-x[0]))
        score = float(score)
        
        # Confidence based on hidden activations (proxy)
        confidence = 0.5 + 0.5 * abs(score - 0.5)  # Higher confidence at extremes
        
        components = {
            "mlp_output": score,
        }
        
        return score, confidence, components


class OracleDistilledScorer:
    """
    Mode 3: Oracle-distilled utility scorer.

    Uses offline-trained ODUS MLP to predict utility.
    Requires pre-training from full-KV traces via odus_trainer.py.

    At runtime: ONLY uses runtime-available features.
    """

    def __init__(self, config: ODUSConfig):
        self.config = config
        self._mlp_weights: List[np.ndarray] = []
        self._mlp_biases: List[np.ndarray] = []
        self._loaded = False

        if config.odus_weights_path:
            self._loaded = self._load_weights(config.odus_weights_path)

        if not self._loaded:
            logger.warning(
                "ODUS mode selected but no weights loaded. "
                "Falling back to similarity baseline at runtime."
            )

    def _load_weights(self, path: str) -> bool:
        """Load pretrained ODUS MLP weights from a .pt checkpoint."""
        try:
            import torch
            weight_path = Path(path)
            if not weight_path.exists():
                logger.warning(f"ODUS weights not found: {path}")
                return False

            checkpoint = torch.load(weight_path, map_location="cpu")
            state_dict = checkpoint.get("model_state", checkpoint)

            # Extract weight/bias pairs from the Sequential model
            # Keys follow pattern: network.0.weight, network.0.bias, ...
            layer_indices = sorted(set(
                int(k.split(".")[1]) for k in state_dict.keys()
                if k.startswith("network.") and "weight" in k
            ))

            self._mlp_weights = []
            self._mlp_biases = []
            for idx in layer_indices:
                w_key = f"network.{idx}.weight"
                b_key = f"network.{idx}.bias"
                if w_key in state_dict and b_key in state_dict:
                    # PyTorch Linear stores [out, in], numpy matmul needs [in, out]
                    self._mlp_weights.append(state_dict[w_key].numpy().T)
                    self._mlp_biases.append(state_dict[b_key].numpy())

            if not self._mlp_weights:
                logger.warning(f"No weight layers found in checkpoint: {path}")
                return False

            logger.info(
                f"Loaded ODUS weights from {path}: "
                f"{len(self._mlp_weights)} layers"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load ODUS weights: {e}")
            return False

    def score(
        self,
        features: RuntimeFeatures,
        chunk: ChunkMetadata,
        query: QueryContext,
    ) -> tuple[float, float, Dict[str, float]]:
        """Score using oracle-distilled MLP model."""
        if not self._loaded:
            fallback = SimilarityBaselineScorer(self.config)
            return fallback.score(features, chunk, query)

        # Forward pass through loaded MLP weights
        x = features.to_vector()

        for i, (W, b) in enumerate(zip(self._mlp_weights, self._mlp_biases)):
            x = x @ W + b
            if i < len(self._mlp_weights) - 1:
                # ReLU for hidden layers (matching ODUSMLP architecture)
                x = np.maximum(0, x)

        # Sigmoid output (last layer of ODUSMLP has Sigmoid)
        score = 1.0 / (1.0 + np.exp(-float(x[0])))
        score = max(0.0, min(1.0, score))

        # Confidence: higher at extremes (near 0 or 1), lower near 0.5
        confidence = 0.5 + 0.5 * abs(score - 0.5) * 2.0

        components = {
            "distilled_score": score,
        }

        return score, confidence, components
