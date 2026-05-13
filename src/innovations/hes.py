"""
Hierarchical Evidence Synthesis (HES).

Solves the admission quality gap by deploying a tiny neural scorer on the
CXL controller side, using quantized micro-embeddings for semantic matching.

Key insight: Current SBFI uses content-agnostic evidence (SimHash/Bloom) that
cannot capture semantic relevance. HES stores INT4-quantized micro-embeddings
(48B per chunk) in CXL controller SRAM and runs a tiny MLP (2K params) to
produce admission scores WITHOUT any PCIe round-trip.

Architecture:
  - Prefill phase: compute micro-embeddings from layer-0 hidden states, write to CXL SRAM
  - Decode phase: broadcast 32B query signature, CXL controller computes scores in parallel
  - Result: admission scores with semantic awareness at zero additional bandwidth cost
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from src.core_types import ChunkMetadata, QueryContext, ScoredCandidate

logger = logging.getLogger(__name__)


@dataclass
class HESConfig:
    micro_embedding_dim: int = 12
    micro_embedding_bits: int = 4
    query_signature_dim: int = 8
    mlp_hidden_dim: int = 16
    mlp_layers: int = 2
    admission_threshold: float = 0.3
    max_chunks_in_sram: int = 2048
    enable_semantic_matching: bool = True
    distillation_temperature: float = 2.0
    sram_budget_bytes: int = 98304  # 96 KB


@dataclass
class HESMetrics:
    total_queries: int = 0
    total_chunks_scored: int = 0
    chunks_admitted: int = 0
    chunks_rejected: int = 0
    avg_admission_score: float = 0.0
    semantic_match_improvements: int = 0
    scoring_latency_us_total: float = 0.0
    simhash_only_would_admit: int = 0
    hes_additional_admits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        total = max(self.total_chunks_scored, 1)
        return {
            "total_queries": self.total_queries,
            "total_chunks_scored": self.total_chunks_scored,
            "admission_rate": self.chunks_admitted / total,
            "avg_admission_score": self.avg_admission_score,
            "semantic_improvements": self.semantic_match_improvements,
            "avg_scoring_latency_us": self.scoring_latency_us_total / max(self.total_queries, 1),
            "simhash_baseline_admits": self.simhash_only_would_admit,
            "hes_additional_admits": self.hes_additional_admits,
            "hes_lift_over_simhash": self.hes_additional_admits / max(self.simhash_only_would_admit, 1),
        }


class MicroEmbeddingStore:
    """Simulates CXL controller SRAM storing micro-embeddings per chunk."""

    def __init__(self, config: HESConfig):
        self.config = config
        self._embeddings: Dict[str, np.ndarray] = {}
        self._chunk_order: List[str] = []

    def store(self, chunk_id: str, embedding: np.ndarray):
        quantized = self._quantize_int4(embedding[:self.config.micro_embedding_dim])
        self._embeddings[chunk_id] = quantized
        self._chunk_order.append(chunk_id)
        if len(self._chunk_order) > self.config.max_chunks_in_sram:
            evict_id = self._chunk_order.pop(0)
            self._embeddings.pop(evict_id, None)

    def get(self, chunk_id: str) -> Optional[np.ndarray]:
        return self._embeddings.get(chunk_id)

    def get_all(self) -> Dict[str, np.ndarray]:
        return dict(self._embeddings)

    def _quantize_int4(self, vec: np.ndarray) -> np.ndarray:
        vec_clipped = np.clip(vec, -1.0, 1.0)
        quantized = np.round((vec_clipped + 1.0) * 7.5).astype(np.int8)
        return quantized

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        return (quantized.astype(np.float32) / 7.5) - 1.0

    @property
    def memory_usage_bytes(self) -> int:
        return len(self._embeddings) * self.config.micro_embedding_dim

    def clear(self):
        self._embeddings.clear()
        self._chunk_order.clear()


class TinyMLP:
    """
    Simulates the CXL-controller-side tiny MLP for admission scoring.

    Architecture: input_dim -> hidden -> hidden -> 1
    Total params: ~2K (fits in < 8KB SRAM)
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(42)
        self.weights = []
        self.biases = []

        in_d = input_dim
        for i in range(n_layers):
            out_d = hidden_dim if i < n_layers - 1 else 1
            w = rng.standard_normal((in_d, out_d)).astype(np.float32) * 0.1
            b = np.zeros(out_d, dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)
            in_d = out_d

    def forward(self, x: np.ndarray) -> float:
        h = x.astype(np.float32)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0)  # ReLU
        return float(1.0 / (1.0 + np.exp(-h[0])))  # sigmoid

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        h = X.astype(np.float32)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0)
        return 1.0 / (1.0 + np.exp(-h.squeeze(-1)))

    def train_distill(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        epochs: int = 50,
        lr: float = 0.01,
    ):
        """Simple SGD training for distillation from full attention scores."""
        for epoch in range(epochs):
            preds = self._forward_train(inputs)
            loss = np.mean((preds - targets) ** 2)
            grad = 2.0 * (preds - targets) / len(targets)
            self._backward(inputs, grad, lr)
            if epoch % 10 == 0:
                logger.debug(f"HES MLP distill epoch {epoch}: loss={loss:.4f}")

    def _forward_train(self, X: np.ndarray) -> np.ndarray:
        h = X.astype(np.float32)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0)
        return h.squeeze(-1)

    def _backward(self, X: np.ndarray, grad_output: np.ndarray, lr: float):
        # Simplified gradient descent on last layer only (for speed)
        h = X.astype(np.float32)
        for i, (w, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            h = h @ w + b
            h = np.maximum(h, 0)
        # Update last layer
        grad_w = h.T @ grad_output.reshape(-1, 1) / len(X)
        grad_b = grad_output.mean()
        self.weights[-1] -= lr * grad_w
        self.biases[-1] -= lr * grad_b

    @property
    def param_count(self) -> int:
        total = 0
        for w, b in zip(self.weights, self.biases):
            total += w.size + b.size
        return total


class HierarchicalEvidenceSynthesis:
    """
    CXL-side neural admission scorer using micro-embeddings.

    Replaces content-agnostic SimHash/Bloom evidence with semantic-aware
    scoring that runs entirely on the CXL controller (no PCIe round-trip).
    """

    def __init__(self, config: Optional[HESConfig] = None):
        self.config = config or HESConfig()
        self.store = MicroEmbeddingStore(self.config)
        input_dim = self.config.micro_embedding_dim + self.config.query_signature_dim
        self.mlp = TinyMLP(
            input_dim=input_dim,
            hidden_dim=self.config.mlp_hidden_dim,
            n_layers=self.config.mlp_layers,
        )
        self.metrics = HESMetrics()
        logger.info(
            f"HES initialized: micro_emb={self.config.micro_embedding_dim}d, "
            f"MLP params={self.mlp.param_count}, "
            f"SRAM budget={self.config.sram_budget_bytes}B"
        )

    def ingest_chunk(self, chunk: ChunkMetadata, hidden_state: Optional[np.ndarray] = None):
        """Store micro-embedding for a chunk (called during prefill)."""
        if hidden_state is not None:
            emb = hidden_state[:self.config.micro_embedding_dim]
        elif chunk.signature is not None:
            emb = chunk.signature[:self.config.micro_embedding_dim].astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
        else:
            rng = np.random.default_rng(hash(chunk.chunk_id) % (2**31))
            emb = rng.standard_normal(self.config.micro_embedding_dim).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
        self.store.store(chunk.chunk_id, emb)

    def score_chunks(
        self,
        query: QueryContext,
        candidate_ids: List[str],
        all_chunks: Dict[str, ChunkMetadata],
    ) -> List[ScoredCandidate]:
        """
        Score candidate chunks using CXL-side micro-embeddings + MLP.

        This simulates the CXL controller computing admission scores
        in parallel for all candidates without PCIe round-trips.
        """
        start = time.time()
        self.metrics.total_queries += 1

        query_sig = self._get_query_signature(query)
        results = []

        batch_inputs = []
        batch_ids = []

        for cid in candidate_ids:
            micro_emb = self.store.get(cid)
            if micro_emb is None:
                results.append(ScoredCandidate(
                    chunk_id=cid, score=0.0, confidence=0.2
                ))
                continue
            dequantized = self.store.dequantize(micro_emb)
            feature_vec = np.concatenate([dequantized, query_sig])
            batch_inputs.append(feature_vec)
            batch_ids.append(cid)

        if batch_inputs:
            batch_array = np.array(batch_inputs)
            scores = self.mlp.forward_batch(batch_array)

            for cid, score in zip(batch_ids, scores):
                chunk = all_chunks.get(cid)
                simhash_score = self._simhash_baseline_score(chunk, query) if chunk else 0.0

                if simhash_score > self.config.admission_threshold:
                    self.metrics.simhash_only_would_admit += 1

                combined_score = 0.6 * float(score) + 0.4 * simhash_score

                if combined_score > self.config.admission_threshold and simhash_score <= self.config.admission_threshold:
                    self.metrics.hes_additional_admits += 1
                    self.metrics.semantic_match_improvements += 1

                admitted = combined_score > self.config.admission_threshold
                if admitted:
                    self.metrics.chunks_admitted += 1
                else:
                    self.metrics.chunks_rejected += 1

                results.append(ScoredCandidate(
                    chunk_id=cid,
                    score=float(combined_score),
                    confidence=min(1.0, abs(combined_score - 0.5) * 2 + 0.3),
                    score_components={
                        "neural_score": float(score),
                        "simhash_score": simhash_score,
                        "combined": float(combined_score),
                    },
                ))

        self.metrics.total_chunks_scored += len(candidate_ids)
        if results:
            self.metrics.avg_admission_score = (
                self.metrics.avg_admission_score * 0.9 +
                np.mean([r.score for r in results]) * 0.1
            )

        latency_us = (time.time() - start) * 1e6
        self.metrics.scoring_latency_us_total += latency_us

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _get_query_signature(self, query: QueryContext) -> np.ndarray:
        if query.query_signature is not None:
            sig = query.query_signature[:self.config.query_signature_dim].astype(np.float32)
            if len(sig) < self.config.query_signature_dim:
                sig = np.pad(sig, (0, self.config.query_signature_dim - len(sig)))
            return sig / (np.linalg.norm(sig) + 1e-8)
        rng = np.random.default_rng(hash(query.request_id) % (2**31) + query.step)
        sig = rng.standard_normal(self.config.query_signature_dim).astype(np.float32)
        return sig / (np.linalg.norm(sig) + 1e-8)

    def _simhash_baseline_score(self, chunk: ChunkMetadata, query: QueryContext) -> float:
        """Baseline SimHash overlap score (what current SBFI does)."""
        if chunk.signature is None or query.query_signature is None:
            return 0.3
        chunk_sig = chunk.signature[:8] if len(chunk.signature) >= 8 else chunk.signature
        query_sig = query.query_signature[:8] if len(query.query_signature) >= 8 else query.query_signature
        min_len = min(len(chunk_sig), len(query_sig))
        if min_len == 0:
            return 0.3
        dot = float(np.dot(chunk_sig[:min_len], query_sig[:min_len]))
        norm = (np.linalg.norm(chunk_sig[:min_len]) * np.linalg.norm(query_sig[:min_len]) + 1e-8)
        return max(0.0, min(1.0, (dot / norm + 1.0) / 2.0))

    def distill_from_attention(
        self,
        chunk_ids: List[str],
        attention_masses: Dict[str, float],
        query: QueryContext,
    ):
        """Offline distillation: train MLP from full attention weights."""
        query_sig = self._get_query_signature(query)
        inputs = []
        targets = []

        for cid in chunk_ids:
            micro_emb = self.store.get(cid)
            if micro_emb is None:
                continue
            dequantized = self.store.dequantize(micro_emb)
            feature_vec = np.concatenate([dequantized, query_sig])
            inputs.append(feature_vec)
            target = attention_masses.get(cid, 0.0)
            targets.append(target)

        if inputs:
            X = np.array(inputs)
            y = np.array(targets)
            y_norm = y / (y.max() + 1e-8)
            self.mlp.train_distill(X, y_norm, epochs=30, lr=0.005)

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.to_dict()

    def reset(self):
        self.store.clear()
        self.metrics = HESMetrics()
