"""
Speculative Decode-Aware Promotion (SDAP).

Couples PROSE promotion decisions with speculative decoding's verify/reject
cycle to extract "free" attention evidence from the draft model.

Key insight: In speculative decoding, the draft model's attention patterns
are computed anyway (as part of generating speculative tokens). These patterns
provide query-aware evidence about which KV blocks are relevant — far stronger
than ODUS-X's static cues — at zero additional cost.

Architecture:
  1. Draft model generates K speculative tokens
  2. During draft, capture per-chunk attention masses (free byproduct)
  3. Use draft attention as evidence for PROSE promotion decisions
  4. On verify: accepted tokens confirm evidence; rejected tokens update drift
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from src.core_types import (
    ChunkMetadata, QueryContext, ScoredCandidate,
    PromotionPipelineResult,
)

logger = logging.getLogger(__name__)


@dataclass
class SDAPConfig:
    draft_speculation_length: int = 5
    draft_attention_weight: float = 0.7
    odus_weight: float = 0.3
    acceptance_boost_factor: float = 1.3
    rejection_drift_penalty: float = 0.8
    min_draft_confidence: float = 0.2
    enable_rejection_feedback: bool = True
    enable_acceptance_boost: bool = True
    draft_model_layers_to_use: int = 1
    attention_aggregation: str = "mean"  # "mean", "max", "last_layer"


@dataclass
class DraftAttentionEvidence:
    """Evidence extracted from draft model's attention patterns."""
    chunk_attention_masses: Dict[str, float]
    draft_tokens_generated: int
    draft_tokens_accepted: int
    acceptance_rate: float
    query_drift_signal: float
    step: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_chunks_with_evidence": len(self.chunk_attention_masses),
            "draft_tokens": self.draft_tokens_generated,
            "accepted": self.draft_tokens_accepted,
            "acceptance_rate": self.acceptance_rate,
            "drift_signal": self.query_drift_signal,
        }


@dataclass
class SDAPMetrics:
    total_steps: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    avg_acceptance_rate: float = 0.0
    evidence_quality_scores: List[float] = field(default_factory=list)
    drift_corrections: int = 0
    promotion_overrides: int = 0
    compute_savings_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "avg_acceptance_rate": self.avg_acceptance_rate,
            "avg_evidence_quality": float(np.mean(self.evidence_quality_scores[-100:])) if self.evidence_quality_scores else 0.0,
            "drift_corrections": self.drift_corrections,
            "promotion_overrides": self.promotion_overrides,
            "compute_savings_ratio": self.compute_savings_ratio,
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
        }


class DraftModelSimulator:
    """
    Simulates a draft model's attention patterns for SDAP.

    In production, this would hook into the actual draft model's attention
    layers. Here we simulate realistic draft attention based on chunk
    properties and query context.
    """

    def __init__(self, speculation_length: int = 5, n_layers: int = 1):
        self.speculation_length = speculation_length
        self.n_layers = n_layers
        self._rng = np.random.default_rng(42)

    def generate_draft_attention(
        self,
        query: QueryContext,
        all_chunks: Dict[str, ChunkMetadata],
        real_attention_hint: Optional[Dict[str, float]] = None,
    ) -> DraftAttentionEvidence:
        """
        Simulate draft model attention over chunks.

        If real_attention_hint is provided (from oracle/eval), use it as
        ground truth with noise. Otherwise, generate synthetic attention
        based on chunk properties.
        """
        chunk_masses: Dict[str, float] = {}

        for cid, chunk in all_chunks.items():
            if real_attention_hint and cid in real_attention_hint:
                base_mass = real_attention_hint[cid]
                noise = self._rng.normal(0, 0.1)
                mass = max(0.0, base_mass + noise)
            else:
                mass = self._compute_synthetic_attention(chunk, query)
            chunk_masses[cid] = mass

        total_mass = sum(chunk_masses.values()) + 1e-8
        chunk_masses = {k: v / total_mass for k, v in chunk_masses.items()}

        n_generated = self.speculation_length
        acceptance_rate = 0.6 + self._rng.uniform(-0.2, 0.2)
        n_accepted = max(1, int(n_generated * acceptance_rate))

        drift_signal = 0.0
        if query.query_signature is not None:
            drift_signal = float(self._rng.uniform(0, 0.3))

        return DraftAttentionEvidence(
            chunk_attention_masses=chunk_masses,
            draft_tokens_generated=n_generated,
            draft_tokens_accepted=n_accepted,
            acceptance_rate=acceptance_rate,
            query_drift_signal=drift_signal,
            step=query.step,
        )

    def _compute_synthetic_attention(
        self, chunk: ChunkMetadata, query: QueryContext
    ) -> float:
        """Compute synthetic attention mass based on chunk properties."""
        recency = 1.0 / (1.0 + max(0, query.step - chunk.last_access_step)) if chunk.last_access_step >= 0 else 0.05
        position_bias = np.exp(-chunk.position_ratio * 2.0)
        history_signal = min(chunk.access_count / 5.0, 1.0) if chunk.access_count > 0 else 0.0

        if chunk.signature is not None and query.query_signature is not None:
            min_len = min(len(chunk.signature), len(query.query_signature))
            if min_len > 0:
                sim = float(np.dot(
                    chunk.signature[:min_len], query.query_signature[:min_len]
                )) / (
                    np.linalg.norm(chunk.signature[:min_len]) *
                    np.linalg.norm(query.query_signature[:min_len]) + 1e-8
                )
                semantic = max(0.0, (sim + 1.0) / 2.0)
            else:
                semantic = 0.2
        else:
            semantic = 0.2

        mass = 0.35 * semantic + 0.30 * recency + 0.20 * position_bias + 0.15 * history_signal
        noise = self._rng.normal(0, 0.05)
        return max(0.0, mass + noise)


class SpeculativeDecodeAwarePromotion:
    """
    Integrates draft model attention evidence into PROSE promotion decisions.

    Replaces or augments ODUS-X scoring with draft-model attention masses,
    providing query-aware evidence at zero additional compute cost.
    """

    def __init__(self, config: Optional[SDAPConfig] = None):
        self.config = config or SDAPConfig()
        self.draft_model = DraftModelSimulator(
            speculation_length=self.config.draft_speculation_length,
            n_layers=self.config.draft_model_layers_to_use,
        )
        self._prev_evidence: Optional[DraftAttentionEvidence] = None
        self._acceptance_history: List[float] = []
        self.metrics = SDAPMetrics()
        logger.info(
            f"SDAP initialized: spec_length={self.config.draft_speculation_length}, "
            f"draft_weight={self.config.draft_attention_weight}"
        )

    def get_draft_evidence(
        self,
        query: QueryContext,
        all_chunks: Dict[str, ChunkMetadata],
        real_attention_hint: Optional[Dict[str, float]] = None,
    ) -> DraftAttentionEvidence:
        """Get attention evidence from draft model speculation."""
        evidence = self.draft_model.generate_draft_attention(
            query, all_chunks, real_attention_hint
        )
        self.metrics.total_steps += 1
        self.metrics.total_draft_tokens += evidence.draft_tokens_generated
        self.metrics.total_accepted_tokens += evidence.draft_tokens_accepted
        self._acceptance_history.append(evidence.acceptance_rate)
        if len(self._acceptance_history) > 50:
            self._acceptance_history.pop(0)
        self.metrics.avg_acceptance_rate = float(np.mean(self._acceptance_history))
        self._prev_evidence = evidence
        return evidence

    def enhance_scores(
        self,
        odus_scores: List[ScoredCandidate],
        draft_evidence: DraftAttentionEvidence,
        query: QueryContext,
    ) -> List[ScoredCandidate]:
        """
        Enhance ODUS-X scores with draft model attention evidence.

        Combines ODUS static scoring with draft-model query-aware attention
        using configurable weights. Applies acceptance/rejection feedback.
        """
        enhanced = []
        draft_masses = draft_evidence.chunk_attention_masses

        acceptance_factor = 1.0
        if self.config.enable_acceptance_boost and draft_evidence.acceptance_rate > 0.7:
            acceptance_factor = self.config.acceptance_boost_factor
        elif self.config.enable_rejection_feedback and draft_evidence.acceptance_rate < 0.3:
            acceptance_factor = self.config.rejection_drift_penalty
            self.metrics.drift_corrections += 1

        for candidate in odus_scores:
            draft_mass = draft_masses.get(candidate.chunk_id, 0.0)

            if draft_mass < self.config.min_draft_confidence:
                enhanced_score = candidate.score
                enhanced_confidence = candidate.confidence * 0.8
            else:
                enhanced_score = (
                    self.config.odus_weight * candidate.score +
                    self.config.draft_attention_weight * draft_mass * acceptance_factor
                )
                enhanced_confidence = min(1.0, candidate.confidence * 1.2)

            if abs(enhanced_score - candidate.score) > 0.1:
                self.metrics.promotion_overrides += 1

            enhanced.append(ScoredCandidate(
                chunk_id=candidate.chunk_id,
                score=float(enhanced_score),
                confidence=float(enhanced_confidence),
                feature_vector=candidate.feature_vector,
                score_components={
                    **(candidate.score_components or {}),
                    "draft_mass": float(draft_mass),
                    "acceptance_factor": float(acceptance_factor),
                    "odus_original": candidate.score,
                },
            ))

        enhanced.sort(key=lambda x: x.score, reverse=True)

        if enhanced and odus_scores:
            quality = self._compute_evidence_quality(odus_scores, enhanced, draft_evidence)
            self.metrics.evidence_quality_scores.append(quality)

        return enhanced

    def compute_savings(
        self,
        n_total_chunks: int,
        n_visible_chunks: int,
    ) -> float:
        """Compute attention compute savings from sparse block selection."""
        if n_total_chunks == 0:
            return 0.0
        savings = 1.0 - (n_visible_chunks / n_total_chunks)
        self.metrics.compute_savings_ratio = (
            self.metrics.compute_savings_ratio * 0.95 + savings * 0.05
        )
        return savings

    def _compute_evidence_quality(
        self,
        original: List[ScoredCandidate],
        enhanced: List[ScoredCandidate],
        evidence: DraftAttentionEvidence,
    ) -> float:
        """Measure how much draft evidence changed the ranking."""
        orig_order = [c.chunk_id for c in original]
        new_order = [c.chunk_id for c in enhanced]

        if not orig_order:
            return 0.0

        kendall_distance = 0
        n = min(len(orig_order), len(new_order), 10)
        for i in range(n):
            for j in range(i + 1, n):
                if orig_order[i] in new_order and orig_order[j] in new_order:
                    ni = new_order.index(orig_order[i])
                    nj = new_order.index(orig_order[j])
                    if (ni - nj) * (i - j) < 0:
                        kendall_distance += 1

        max_pairs = n * (n - 1) / 2
        quality = kendall_distance / max_pairs if max_pairs > 0 else 0.0
        return quality

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.to_dict()

    def reset(self):
        self._prev_evidence = None
        self._acceptance_history.clear()
        self.metrics = SDAPMetrics()
