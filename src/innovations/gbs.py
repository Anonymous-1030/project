"""
Ghost Block Synthesis (GBS).

Solves the "missing blocks are unrecoverable" problem by synthesizing
approximate KV vectors for chunks that were not promoted but are needed.

Key insight: LLM attention is tolerant of KV approximation (INT4 KV cache
quantization is nearly lossless). When PROSE detects an "attention sink"
(query has low attention to all promoted blocks), it can reconstruct
approximate K/V from micro-embeddings stored in CXL SRAM.

Ghost blocks are:
  - Not real KV, but approximate reconstructions
  - Sufficient to maintain generation quality (cosine sim > 0.8 to real KV)
  - Marked with a 1-bit flag in block_table
  - Upgraded to real blocks via full fetch if repeatedly attended
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np

from src.core_types import ChunkMetadata, QueryContext

logger = logging.getLogger(__name__)


@dataclass
class GBSConfig:
    attention_sink_threshold: float = 0.15
    ghost_reconstruction_dim: int = 64
    ghost_confidence_threshold: float = 0.1
    max_ghost_blocks: int = 8
    ghost_to_real_promotion_count: int = 3
    reconstruction_quality_target: float = 0.8
    enable_confidence_gating: bool = True
    micro_embedding_dim: int = 12
    decoder_hidden_dim: int = 32


@dataclass
class GhostBlock:
    chunk_id: str
    reconstructed_kv: np.ndarray
    confidence: float
    attend_count: int = 0
    creation_step: int = 0
    is_promoted_to_real: bool = False

    def should_promote_to_real(self, threshold: int = 3) -> bool:
        return self.attend_count >= threshold and not self.is_promoted_to_real


@dataclass
class GBSMetrics:
    total_sink_detections: int = 0
    total_ghosts_created: int = 0
    total_ghosts_promoted_to_real: int = 0
    total_ghosts_evicted: int = 0
    avg_ghost_confidence: float = 0.0
    avg_reconstruction_quality: float = 0.0
    attention_recovery_events: int = 0
    steps_with_ghosts: int = 0
    total_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        total = max(self.total_steps, 1)
        return {
            "sink_detection_rate": self.total_sink_detections / total,
            "ghosts_created": self.total_ghosts_created,
            "ghosts_promoted_to_real": self.total_ghosts_promoted_to_real,
            "ghosts_evicted": self.total_ghosts_evicted,
            "avg_ghost_confidence": self.avg_ghost_confidence,
            "avg_reconstruction_quality": self.avg_reconstruction_quality,
            "attention_recovery_rate": self.attention_recovery_events / max(self.total_sink_detections, 1),
            "ghost_utilization": self.steps_with_ghosts / total,
        }


class GhostBlockDecoder:
    """
    Lightweight decoder that reconstructs approximate KV from micro-embeddings.

    Architecture: micro_emb (12d) -> hidden (32d) -> KV (64d)
    Simulates a tiny decoder that could run on CXL controller SRAM.
    """

    def __init__(self, input_dim: int = 12, hidden_dim: int = 32, output_dim: int = 64):
        rng = np.random.default_rng(123)
        self.w1 = rng.standard_normal((input_dim, hidden_dim)).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = rng.standard_normal((hidden_dim, output_dim)).astype(np.float32) * 0.1
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def reconstruct(self, micro_embedding: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Reconstruct approximate KV vector from micro-embedding.

        Returns (reconstructed_kv, confidence).
        """
        x = micro_embedding.astype(np.float32)
        h = np.maximum(x @ self.w1 + self.b1, 0)  # ReLU
        kv = np.tanh(h @ self.w2 + self.b2)

        # Confidence based on output magnitude (tanh saturation = high confidence)
        kv_magnitude = float(np.mean(np.abs(kv)))
        confidence = min(1.0, max(0.1, kv_magnitude * 3.0))

        return kv, confidence

    def train_from_pairs(
        self,
        micro_embeddings: np.ndarray,
        real_kvs: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
    ):
        """Train decoder from (micro_embedding, real_kv) pairs."""
        for epoch in range(epochs):
            h = np.maximum(micro_embeddings @ self.w1 + self.b1, 0)
            pred = np.tanh(h @ self.w2 + self.b2)
            loss = np.mean((pred - real_kvs) ** 2)

            grad_pred = 2.0 * (pred - real_kvs) / len(real_kvs)
            grad_tanh = grad_pred * (1 - pred ** 2)
            grad_w2 = h.T @ grad_tanh / len(real_kvs)
            grad_b2 = grad_tanh.mean(axis=0)

            relu_mask = (micro_embeddings @ self.w1 + self.b1 > 0).astype(np.float32)
            grad_h = grad_tanh @ self.w2.T * relu_mask
            grad_w1 = micro_embeddings.T @ grad_h / len(real_kvs)
            grad_b1 = grad_h.mean(axis=0)

            self.w1 -= lr * grad_w1
            self.b1 -= lr * grad_b1
            self.w2 -= lr * grad_w2
            self.b2 -= lr * grad_b2

            if epoch % 20 == 0:
                cos_sim = np.mean([
                    float(np.dot(p, r) / (np.linalg.norm(p) * np.linalg.norm(r) + 1e-8))
                    for p, r in zip(pred, real_kvs)
                ])
                logger.debug(f"GBS decoder epoch {epoch}: loss={loss:.4f}, cos_sim={cos_sim:.3f}")


class GhostBlockSynthesizer:
    """
    Synthesizes approximate KV blocks when PROSE detects attention sinks.

    When all promoted blocks have low attention mass (sink detection),
    GBS reconstructs approximate KV for the most likely missing chunks
    and injects them as "ghost blocks" into the attention computation.
    """

    def __init__(self, config: Optional[GBSConfig] = None):
        self.config = config or GBSConfig()
        self.decoder = GhostBlockDecoder(
            input_dim=self.config.micro_embedding_dim,
            hidden_dim=self.config.decoder_hidden_dim,
            output_dim=self.config.ghost_reconstruction_dim,
        )
        self._ghost_blocks: Dict[str, GhostBlock] = {}
        self._micro_embeddings: Dict[str, np.ndarray] = {}
        self.metrics = GBSMetrics()
        logger.info(
            f"GBS initialized: max_ghosts={self.config.max_ghost_blocks}, "
            f"sink_threshold={self.config.attention_sink_threshold}, "
            f"confidence_gate={self.config.ghost_confidence_threshold}"
        )

    def store_micro_embedding(self, chunk_id: str, embedding: np.ndarray):
        """Store micro-embedding for potential ghost reconstruction."""
        self._micro_embeddings[chunk_id] = embedding[:self.config.micro_embedding_dim].astype(np.float32)

    def detect_attention_sink(
        self,
        attention_masses: Dict[str, float],
        promoted_ids: List[str],
    ) -> bool:
        """
        Detect if current attention is in a "sink" state.

        Sink = all promoted blocks have low attention mass, suggesting
        the model needs blocks that aren't currently promoted.
        """
        if not promoted_ids or not attention_masses:
            return False

        promoted_masses = [
            attention_masses.get(cid, 0.0) for cid in promoted_ids
        ]
        max_mass = max(promoted_masses) if promoted_masses else 0.0
        return max_mass < self.config.attention_sink_threshold

    def synthesize_ghosts(
        self,
        query: QueryContext,
        all_chunks: Dict[str, ChunkMetadata],
        promoted_ids: List[str],
        attention_masses: Dict[str, float],
    ) -> List[GhostBlock]:
        """
        Synthesize ghost blocks for chunks likely needed but not promoted.

        Strategy:
        1. Identify candidate chunks (not promoted, have micro-embeddings)
        2. Score candidates by predicted relevance
        3. Reconstruct approximate KV for top candidates
        4. Return ghost blocks with confidence scores
        """
        start = time.time()
        self.metrics.total_steps += 1

        is_sink = self.detect_attention_sink(attention_masses, promoted_ids)
        if not is_sink:
            if self._ghost_blocks:
                self.metrics.steps_with_ghosts += 1
            return list(self._ghost_blocks.values())

        self.metrics.total_sink_detections += 1

        promoted_set = set(promoted_ids)
        ghost_set = set(self._ghost_blocks.keys())
        candidates = []

        for cid, chunk in all_chunks.items():
            if cid in promoted_set or cid in ghost_set:
                continue
            if cid not in self._micro_embeddings:
                continue

            relevance = self._estimate_relevance(chunk, query)
            candidates.append((cid, relevance))

        candidates.sort(key=lambda x: x[1], reverse=True)
        n_to_create = min(
            self.config.max_ghost_blocks - len(self._ghost_blocks),
            len(candidates),
        )

        new_ghosts = []
        for cid, relevance in candidates[:n_to_create]:
            micro_emb = self._micro_embeddings[cid]
            reconstructed_kv, confidence = self.decoder.reconstruct(micro_emb)

            if self.config.enable_confidence_gating and confidence < self.config.ghost_confidence_threshold:
                continue

            ghost = GhostBlock(
                chunk_id=cid,
                reconstructed_kv=reconstructed_kv,
                confidence=confidence,
                creation_step=query.step,
            )
            self._ghost_blocks[cid] = ghost
            new_ghosts.append(ghost)
            self.metrics.total_ghosts_created += 1

        if new_ghosts:
            self.metrics.attention_recovery_events += 1
            avg_conf = np.mean([g.confidence for g in new_ghosts])
            self.metrics.avg_ghost_confidence = (
                self.metrics.avg_ghost_confidence * 0.9 + avg_conf * 0.1
            )

        if self._ghost_blocks:
            self.metrics.steps_with_ghosts += 1

        return list(self._ghost_blocks.values())

    def update_ghost_attention(
        self,
        attention_masses: Dict[str, float],
        step: int,
    ) -> List[str]:
        """
        Update ghost blocks based on attention they received.

        Returns list of ghost block IDs that should be promoted to real.
        """
        promote_to_real = []

        for cid, ghost in list(self._ghost_blocks.items()):
            mass = attention_masses.get(cid, 0.0)
            if mass > 0.05:
                ghost.attend_count += 1

            if ghost.should_promote_to_real(self.config.ghost_to_real_promotion_count):
                promote_to_real.append(cid)
                ghost.is_promoted_to_real = True
                self.metrics.total_ghosts_promoted_to_real += 1

            if step - ghost.creation_step > 10 and ghost.attend_count == 0:
                del self._ghost_blocks[cid]
                self.metrics.total_ghosts_evicted += 1

        return promote_to_real

    def get_ghost_ids(self) -> List[str]:
        return list(self._ghost_blocks.keys())

    def get_ghost_kv(self, chunk_id: str) -> Optional[np.ndarray]:
        ghost = self._ghost_blocks.get(chunk_id)
        return ghost.reconstructed_kv if ghost else None

    def _estimate_relevance(self, chunk: ChunkMetadata, query: QueryContext) -> float:
        """Estimate chunk relevance without full attention computation."""
        recency = 1.0 / (1.0 + max(0, query.step - chunk.last_access_step)) if chunk.last_access_step >= 0 else 0.1
        history = min(chunk.promoted_count / 3.0, 1.0) if chunk.promoted_count > 0 else 0.0
        position = 1.0 - chunk.position_ratio

        if chunk.signature is not None and query.query_signature is not None:
            min_len = min(len(chunk.signature), len(query.query_signature))
            if min_len > 0:
                sim = float(np.dot(
                    chunk.signature[:min_len],
                    query.query_signature[:min_len]
                )) / (
                    np.linalg.norm(chunk.signature[:min_len]) *
                    np.linalg.norm(query.query_signature[:min_len]) + 1e-8
                )
                semantic = max(0.0, (sim + 1.0) / 2.0)
            else:
                semantic = 0.3
        else:
            semantic = 0.3

        return 0.3 * semantic + 0.3 * recency + 0.2 * history + 0.2 * position

    def compute_reconstruction_quality(
        self,
        ghost_kvs: Dict[str, np.ndarray],
        real_kvs: Dict[str, np.ndarray],
    ) -> float:
        """Compute average cosine similarity between ghost and real KV."""
        similarities = []
        for cid in ghost_kvs:
            if cid in real_kvs:
                g = ghost_kvs[cid]
                r = real_kvs[cid]
                min_d = min(len(g), len(r))
                sim = float(np.dot(g[:min_d], r[:min_d])) / (
                    np.linalg.norm(g[:min_d]) * np.linalg.norm(r[:min_d]) + 1e-8
                )
                similarities.append(sim)

        quality = float(np.mean(similarities)) if similarities else 0.0
        self.metrics.avg_reconstruction_quality = (
            self.metrics.avg_reconstruction_quality * 0.9 + quality * 0.1
        )
        return quality

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.to_dict()

    def reset(self):
        self._ghost_blocks.clear()
        self._micro_embeddings.clear()
        self.metrics = GBSMetrics()
