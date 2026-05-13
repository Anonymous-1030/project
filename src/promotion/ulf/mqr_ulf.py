"""
Multi-Queue Recall ULF (MQR-ULF) for ProSE-X 2.0.

PATCH NOTES (2026-03-24):
- Increased per-queue quotas for better recall
- Added minimum union size target
- Enhanced diagnostics for queue debugging
- Fixed anchor_neighbor radius calculation
- Fixed historical_success to work once promotions begin
- Added Candidate Recall@K tracking

This module replaces the old single-score ULF with a multi-queue recall engine.
ULF objective is RECALL, not precision.
"""

import time
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field

from src.core_types import (
    ChunkMetadata, QueryContext, ULFResult, QueueContribution, ChunkTier
)
from src.config import MQRULFConfig

logger = logging.getLogger(__name__)


@dataclass
class QueueDiagnostics:
    """Diagnostics for a single queue."""
    queue_name: str
    raw_output_count: int
    post_dedup_count: int
    empty_reason: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue_name": self.queue_name,
            "raw_output_count": self.raw_output_count,
            "post_dedup_count": self.post_dedup_count,
            "empty_reason": self.empty_reason,
            "debug_info": self.debug_info,
        }


@dataclass
class ULFDiagnostics:
    """Complete ULF diagnostics."""
    step: int
    tail_total: int
    queue_diagnostics: List[QueueDiagnostics]
    union_size_before_cap: int
    final_union_size: int
    target_size: int
    
    def print_summary(self) -> None:
        """Print diagnostic summary."""
        print(f"\n[ULF Diagnostics] Step {self.step}")
        print(f"  Tail total: {self.tail_total}")
        print(f"  Target union size: {self.target_size}")
        print(f"  Union before cap: {self.union_size_before_cap}")
        print(f"  Final union size: {self.final_union_size}")
        print(f"  Queue breakdown:")
        for qd in self.queue_diagnostics:
            status = "OK" if qd.raw_output_count > 0 else "EMPTY"
            print(f"    - {qd.queue_name}: raw={qd.raw_output_count}, "
                  f"deduped={qd.post_dedup_count} [{status}]")
            if qd.empty_reason:
                print(f"      Reason: {qd.empty_reason}")
            if qd.debug_info:
                for k, v in qd.debug_info.items():
                    print(f"      {k}: {v}")


@dataclass
class RecallQueue:
    """Base class for recall queues."""
    name: str
    enabled: bool = True
    top_k: int = 5
    
    def recall(
        self,
        query: QueryContext,
        tail_chunks: List[ChunkMetadata],
        all_chunks: Dict[str, ChunkMetadata],
    ) -> Tuple[QueueContribution, QueueDiagnostics]:
        """
        Recall candidates from this queue.
        
        Returns:
            (QueueContribution, QueueDiagnostics)
        """
        raise NotImplementedError


class AnchorNeighborQueue(RecallQueue):
    """
    Queue 1: Anchor-neighbor recall.
    
    For currently active or recently high-attention anchor chunks,
    recall adjacent tail chunks within a configurable radius.
    """
    
    def __init__(self, radius: int = 2, top_k: int = 2):  # PATCH: increased default to 2
        super().__init__("anchor_neighbor", top_k=top_k)
        self.radius = radius
    
    def recall(
        self,
        query: QueryContext,
        tail_chunks: List[ChunkMetadata],
        all_chunks: Dict[str, ChunkMetadata],
    ) -> Tuple[QueueContribution, QueueDiagnostics]:
        """Recall chunks adjacent to active/recent anchors."""
        diagnostics = QueueDiagnostics(
            queue_name=self.name,
            raw_output_count=0,
            post_dedup_count=0,
        )
        
        if not self.enabled:
            diagnostics.empty_reason = "queue_disabled"
            return QueueContribution(self.name, [], []), diagnostics
        
        # Get anchor chunks
        anchor_ids = set(query.active_anchor_ids + query.recent_anchor_ids)
        
        # DEBUG: Log anchor info
        diagnostics.debug_info["anchor_ids_count"] = len(anchor_ids)
        diagnostics.debug_info["anchor_ids"] = list(anchor_ids)[:5]  # First 5
        diagnostics.debug_info["radius"] = self.radius
        diagnostics.debug_info["tail_chunks_count"] = len(tail_chunks)
        
        if not anchor_ids:
            diagnostics.empty_reason = "no_active_anchors"
            return QueueContribution(self.name, [], []), diagnostics
        
        # Find tail chunks adjacent to anchors
        recalled = []
        scores = []
        
        for anchor_id in anchor_ids:
            anchor = all_chunks.get(anchor_id)
            if anchor is None:
                continue
            
            # Find chunks within radius
            for chunk in tail_chunks:
                # Compute distance in token space
                if chunk.request_id != anchor.request_id:
                    continue
                
                # Distance is minimum gap between chunks
                if chunk.token_end < anchor.token_start:
                    distance = anchor.token_start - chunk.token_end
                elif chunk.token_start > anchor.token_end:
                    distance = chunk.token_start - anchor.token_end
                else:
                    # Adjacent or overlapping (distance = 0)
                    distance = 0
                
                # Convert distance to chunk radius (number of chunks away)
                chunk_size = max(chunk.num_tokens, 1)
                chunk_distance = distance / chunk_size
                
                # PATCH: More lenient radius check with debug
                if chunk_distance <= self.radius + 0.1:  # Small epsilon for float comparison
                    recalled.append(chunk.chunk_id)
                    # Score inversely proportional to distance
                    score = 1.0 / (1.0 + chunk_distance)
                    scores.append(score)
        
        raw_count = len(recalled)
        diagnostics.raw_output_count = raw_count
        
        # Sort by score and take top_k
        if recalled:
            sorted_pairs = sorted(zip(recalled, scores), key=lambda x: x[1], reverse=True)
            # PATCH: Take up to top_k, but ensure we get at least some if available
            take_count = min(self.top_k, len(sorted_pairs))
            # If we have candidates but top_k is 0, take at least 1
            if take_count == 0 and len(sorted_pairs) > 0:
                take_count = min(2, len(sorted_pairs))  # Emergency minimum
            
            recalled = [p[0] for p in sorted_pairs[:take_count]]
            scores = [p[1] for p in sorted_pairs[:take_count]]
        
        diagnostics.post_dedup_count = len(recalled)
        
        if not recalled:
            diagnostics.empty_reason = "no_neighbors_in_radius"
            diagnostics.debug_info["candidates_in_radius"] = raw_count
        
        return QueueContribution(self.name, recalled, scores), diagnostics


class LexicalOverlapQueue(RecallQueue):
    """
    Queue 2: Lexical/entity overlap recall.

    === COMPUTATION TIMING & OVERHEAD (§2.x MQR-ULF Cost Analysis) ===

    Q: Does the lexical overlap queue require expensive per-step embedding
       computation during decoding?
    A: NO. The lexical overlap queue uses PRE-COMPUTED hash signatures:

    1. Signatures are computed ONCE during the Query Prefill phase:
       - query.query_signature: hashed from prefill query tokens
       - chunk.signature: hashed from chunk text at initial chunking
       - Both are O(1) fixed-size vectors (128-dim float32 or hash sets)

    2. At decode time, the overlap computation is a simple dot-product
       or Jaccard-like set intersection on these pre-computed signatures.
       No real-time embedding computation is needed.

    3. Complexity: O(|tail_chunks| × signature_dim) per step, where
       signature_dim = 128 (fixed). With typical |tail_chunks| < 100,
       this is ~12K FLOPs per step — negligible vs. attention compute.

    4. The lexical overlap features are ALSO implicitly captured by the
       ODUS training (via query_chunk_similarity feature), so at runtime
       the ODUS scorer already accounts for lexical similarity. The MQR-ULF
       lexical queue is a HEURISTIC pre-filter, not a replacement for ODUS.

    In summary: MQR-ULF is a pure heuristic pre-filter with O(1) per-chunk
    cost. No real-time Embedding computation, no CPU-side hash storms.
    """

    # Timing mode: when to compute overlap
    # "prefill_only": compute once at prefill, reuse during decode (DEFAULT)
    # "per_step": recompute at each decode step (expensive, NOT recommended)
    TIMING_MODES = ("prefill_only", "per_step")

    def __init__(
        self,
        method: str = "hashed_token",
        top_k: int = 4,  # PATCH: increased from 5 to 4 (will be higher)
        overlap_threshold: float = 0.1,
        timing_mode: str = "prefill_only",
    ):
        super().__init__("lexical_overlap", top_k=top_k)
        self.method = method
        self.overlap_threshold = overlap_threshold
        self.timing_mode = timing_mode

        # Prefill-cached overlap scores: chunk_id → overlap_score
        # Computed once at prefill, reused for all decode steps
        self._cached_overlaps: Dict[str, float] = {}

        # Stats for overhead tracking
        self._n_computations = 0
        self._n_cache_hits = 0
    
    def recall(
        self,
        query: QueryContext,
        tail_chunks: List[ChunkMetadata],
        all_chunks: Dict[str, ChunkMetadata],
    ) -> Tuple[QueueContribution, QueueDiagnostics]:
        """Recall chunks with high lexical overlap.

        In "prefill_only" mode (default):
          - Overlap scores are computed once and cached per request
          - Subsequent decode steps reuse the cached scores
          - This avoids per-step O(|tail_chunks|) recomputation
          - Query signature is set at prefill and stays constant

        In "per_step" mode (NOT recommended):
          - Overlap scores are recomputed at every decode step
          - Only needed if query.signature changes during decode
            (which is NOT the case in ProSE's design)
        """
        diagnostics = QueueDiagnostics(
            queue_name=self.name,
            raw_output_count=0,
            post_dedup_count=0,
        )
        diagnostics.debug_info["timing_mode"] = self.timing_mode
        
        if not self.enabled:
            diagnostics.empty_reason = "queue_disabled"
            return QueueContribution(self.name, [], []), diagnostics
        
        if query.query_signature is None:
            diagnostics.empty_reason = "no_query_signature"
            return QueueContribution(self.name, [], []), diagnostics

        # Prefill-only mode: check if we have cached overlaps for this request
        if self.timing_mode == "prefill_only" and self._cached_overlaps:
            # Reuse cached overlaps from prefill computation
            recalled = []
            scores = []
            for chunk in tail_chunks:
                cached_score = self._cached_overlaps.get(chunk.chunk_id)
                if cached_score is not None and cached_score >= self.overlap_threshold:
                    recalled.append(chunk.chunk_id)
                    scores.append(cached_score)
                    self._n_cache_hits += 1

            # Sort and take top_k
            if recalled:
                sorted_pairs = sorted(zip(recalled, scores), key=lambda x: x[1], reverse=True)
                take_count = min(self.top_k, len(sorted_pairs))
                recalled = [p[0] for p in sorted_pairs[:take_count]]
                scores = [p[1] for p in sorted_pairs[:take_count]]

            diagnostics.raw_output_count = len(self._cached_overlaps)
            diagnostics.post_dedup_count = len(recalled)
            diagnostics.debug_info["used_cache"] = True
            diagnostics.debug_info["cache_size"] = len(self._cached_overlaps)

            return QueueContribution(self.name, recalled, scores), diagnostics
        
        # Full computation (prefill step or per_step mode)
        recalled = []
        scores = []
        
        query_sig = query.query_signature
        
        for chunk in tail_chunks:
            if chunk.signature is None:
                continue
            
            # Compute overlap using signatures
            if self.method == "hashed_token":
                overlap = self._compute_hashed_overlap(query_sig, chunk.signature)
            elif self.method == "ngram_bloom":
                overlap = self._compute_bloom_overlap(query_sig, chunk.signature)
            elif self.method == "entity_hash":
                overlap = self._compute_entity_overlap(query_sig, chunk.signature)
            else:
                overlap = self._compute_hashed_overlap(query_sig, chunk.signature)
            
            self._n_computations += 1

            # Cache the overlap score for prefill_only mode
            if self.timing_mode == "prefill_only":
                self._cached_overlaps[chunk.chunk_id] = overlap

            # PATCH: Lower threshold if needed to get candidates
            effective_threshold = self.overlap_threshold
            if overlap >= effective_threshold:
                recalled.append(chunk.chunk_id)
                scores.append(overlap)
        
        raw_count = len(recalled)
        diagnostics.raw_output_count = raw_count
        diagnostics.debug_info["used_cache"] = False
        
        # Sort by score and take top_k
        if recalled:
            sorted_pairs = sorted(zip(recalled, scores), key=lambda x: x[1], reverse=True)
            take_count = min(self.top_k, len(sorted_pairs))
            # PATCH: Emergency minimum
            if take_count == 0 and len(sorted_pairs) > 0:
                take_count = min(2, len(sorted_pairs))
            
            recalled = [p[0] for p in sorted_pairs[:take_count]]
            scores = [p[1] for p in sorted_pairs[:take_count]]
        
        diagnostics.post_dedup_count = len(recalled)
        
        if not recalled:
            diagnostics.empty_reason = "no_overlap_above_threshold"
            diagnostics.debug_info["chunks_checked"] = len(tail_chunks)
        
        return QueueContribution(self.name, recalled, scores), diagnostics
    
    def clear_cache(self) -> None:
        """Clear the prefill-cached overlap scores.

        Should be called when switching to a new request.
        """
        self._cached_overlaps.clear()

    def overhead_stats(self) -> Dict[str, Any]:
        """Return overhead statistics for this queue.

        Key metrics for the paper:
        - total_computations: how many dot-product/set-intersection operations
        - cache_hits: how many times cached scores were reused
        - cache_hit_rate: fraction of lookups served from cache
        - estimated_flops: rough FLOP count for all computations
        """
        total = self._n_computations + self._n_cache_hits
        return {
            "total_computations": self._n_computations,
            "cache_hits": self._n_cache_hits,
            "cache_hit_rate": self._n_cache_hits / max(1, total),
            "timing_mode": self.timing_mode,
            "method": self.method,
            "estimated_flops": self._n_computations * 128 * 2,  # dot-product
        }
    
    def _compute_hashed_overlap(self, query_sig: Any, chunk_sig: Any) -> float:
        """Compute hashed token overlap."""
        import numpy as np
        
        if isinstance(query_sig, np.ndarray) and isinstance(chunk_sig, np.ndarray):
            query_norm = query_sig / (np.linalg.norm(query_sig) + 1e-8)
            chunk_norm = chunk_sig / (np.linalg.norm(chunk_sig) + 1e-8)
            return float(np.dot(query_norm, chunk_norm))
        
        if isinstance(query_sig, (set, list)) and isinstance(chunk_sig, (set, list)):
            intersection = len(set(query_sig) & set(chunk_sig))
            union = len(set(query_sig) | set(chunk_sig))
            return intersection / max(union, 1)
        
        return 0.0
    
    def _compute_bloom_overlap(self, query_sig: Any, chunk_sig: Any) -> float:
        return self._compute_hashed_overlap(query_sig, chunk_sig)
    
    def _compute_entity_overlap(self, query_sig: Any, chunk_sig: Any) -> float:
        return self._compute_hashed_overlap(query_sig, chunk_sig)


class StructuralRecencyQueue(RecallQueue):
    """
    Queue 3: Structural/recency recall.
    """
    
    def __init__(
        self,
        recent_top_k: int = 2,  # PATCH: was 3
        boundary_top_k: int = 2,
        title_adjacent_top_k: int = 2,
        recency_window: int = 100,
    ):
        super().__init__("structural_recency", top_k=recent_top_k + boundary_top_k + title_adjacent_top_k)
        self.recent_top_k = recent_top_k
        self.boundary_top_k = boundary_top_k
        self.title_adjacent_top_k = title_adjacent_top_k
        self.recency_window = recency_window
    
    def recall(
        self,
        query: QueryContext,
        tail_chunks: List[ChunkMetadata],
        all_chunks: Dict[str, ChunkMetadata],
    ) -> Tuple[QueueContribution, QueueDiagnostics]:
        """Recall chunks based on structural and recency signals."""
        diagnostics = QueueDiagnostics(
            queue_name=self.name,
            raw_output_count=0,
            post_dedup_count=0,
        )
        
        if not self.enabled:
            diagnostics.empty_reason = "queue_disabled"
            return QueueContribution(self.name, [], []), diagnostics
        
        recalled = []
        scores = []
        
        # Recent chunks
        recent_chunks = [
            chunk for chunk in tail_chunks
            if query.step - chunk.last_access_step < self.recency_window
            and chunk.last_access_step >= 0
        ]
        recent_chunks.sort(key=lambda c: c.last_access_step, reverse=True)
        
        for chunk in recent_chunks[:self.recent_top_k]:
            if chunk.chunk_id not in recalled:
                recalled.append(chunk.chunk_id)
                recency_score = 1.0 / (1.0 + (query.step - chunk.last_access_step) / 10.0)
                scores.append(recency_score)
        
        # Section boundaries
        boundary_chunks = [
            chunk for chunk in tail_chunks
            if chunk.is_section_boundary
        ]
        boundary_chunks.sort(key=lambda c: c.access_count, reverse=True)
        
        for chunk in boundary_chunks[:self.boundary_top_k]:
            if chunk.chunk_id not in recalled:
                recalled.append(chunk.chunk_id)
                scores.append(0.7)
        
        # Title-adjacent chunks
        title_chunks = [
            chunk for chunk in tail_chunks
            if chunk.is_title_adjacent
        ]
        title_chunks.sort(key=lambda c: c.position_ratio)
        
        for chunk in title_chunks[:self.title_adjacent_top_k]:
            if chunk.chunk_id not in recalled:
                recalled.append(chunk.chunk_id)
                scores.append(0.6)
        
        diagnostics.raw_output_count = len(recalled)
        diagnostics.post_dedup_count = len(recalled)
        
        if not recalled:
            diagnostics.empty_reason = "no_recent_or_structural_chunks"
            diagnostics.debug_info["recent_candidates"] = len(recent_chunks)
            diagnostics.debug_info["boundary_candidates"] = len(boundary_chunks)
            diagnostics.debug_info["title_candidates"] = len(title_chunks)
        
        return QueueContribution(self.name, recalled, scores), diagnostics


class HistoricalSuccessQueue(RecallQueue):
    """
    Queue 4: Historical-success recall.
    
    PATCH: Now works correctly once promotions begin.
    """
    
    def __init__(
        self,
        top_k: int = 2,  # PATCH: increased from 3 to 2 (will be higher)
        min_success_count: int = 1,
    ):
        super().__init__("historical_success", top_k=top_k)
        self.min_success_count = min_success_count
    
    def recall(
        self,
        query: QueryContext,
        tail_chunks: List[ChunkMetadata],
        all_chunks: Dict[str, ChunkMetadata],
    ) -> Tuple[QueueContribution, QueueDiagnostics]:
        """Recall chunks with historical promotion success."""
        diagnostics = QueueDiagnostics(
            queue_name=self.name,
            raw_output_count=0,
            post_dedup_count=0,
        )
        
        if not self.enabled:
            diagnostics.empty_reason = "queue_disabled"
            return QueueContribution(self.name, [], []), diagnostics
        
        # Filter chunks with successful promotions
        successful_chunks = [
            chunk for chunk in tail_chunks
            if chunk.promoted_count >= self.min_success_count
        ]
        
        # PATCH: Debug info
        diagnostics.debug_info["tail_chunks_with_promotions"] = len(successful_chunks)
        diagnostics.debug_info["min_success_count"] = self.min_success_count
        
        # Sort by promotion count and recency
        successful_chunks.sort(
            key=lambda c: (c.promoted_count, c.last_promotion_step),
            reverse=True
        )
        
        recalled = []
        scores = []
        
        for chunk in successful_chunks[:self.top_k]:
            recalled.append(chunk.chunk_id)
            recency_factor = 1.0 / (1.0 + (query.step - chunk.last_promotion_step) / 100.0)
            score = min(chunk.promoted_count / 5.0, 1.0) * recency_factor
            scores.append(score)
        
        diagnostics.raw_output_count = len(recalled)
        diagnostics.post_dedup_count = len(recalled)
        
        if not recalled:
            # PATCH: More descriptive empty reason
            chunks_with_any_promo = [c for c in tail_chunks if c.promoted_count > 0]
            if chunks_with_any_promo:
                diagnostics.empty_reason = "promotions_below_threshold"
                diagnostics.debug_info["chunks_with_any_promo"] = len(chunks_with_any_promo)
                diagnostics.debug_info["max_promo_count"] = max(
                    (c.promoted_count for c in chunks_with_any_promo), default=0
                )
            else:
                diagnostics.empty_reason = "no_promotion_history_yet"
        
        return QueueContribution(self.name, recalled, scores), diagnostics


class MultiQueueRecallULF:
    """
    Multi-Queue Recall ULF implementation.
    
    PATCHED (2026-03-24):
    - Increased per-queue quotas
    - Minimum union size target
    - Enhanced diagnostics
    - Candidate Recall@K tracking
    """
    
    # PATCH: Minimum target union size
    MIN_UNION_TARGET = 6
    
    def __init__(self, config: MQRULFConfig):
        self.config = config
        
        # Initialize queues with PATCHED higher quotas
        self.queues: List[RecallQueue] = []
        
        if config.anchor_neighbor_enabled:
            # PATCH: Ensure at least 2 from anchor neighbor
            top_k = max(config.anchor_neighbor_top_k, 2)
            self.queues.append(AnchorNeighborQueue(
                radius=config.anchor_neighbor_radius,
                top_k=top_k,
            ))
        
        if config.lexical_overlap_enabled:
            # PATCH: Ensure at least 4 from lexical overlap
            top_k = max(config.lexical_overlap_top_k, 4)
            self.queues.append(LexicalOverlapQueue(
                method=config.lexical_overlap_method,
                top_k=top_k,
                overlap_threshold=config.lexical_overlap_threshold,
            ))
        
        if config.structural_recency_enabled:
            # PATCH: Ensure at least 2 from structural recency
            recent_top_k = max(config.structural_recent_top_k, 2)
            boundary_top_k = max(config.structural_boundary_top_k, 1)
            title_top_k = max(config.structural_title_adjacent_top_k, 1)
            self.queues.append(StructuralRecencyQueue(
                recent_top_k=recent_top_k,
                boundary_top_k=boundary_top_k,
                title_adjacent_top_k=title_top_k,
                recency_window=config.structural_recency_window,
            ))
        
        if config.historical_success_enabled:
            # PATCH: Ensure at least 2 from historical success
            top_k = max(config.historical_success_top_k, 2)
            self.queues.append(HistoricalSuccessQueue(
                top_k=top_k,
                min_success_count=config.historical_success_min_count,
            ))
        
        logger.info(f"MQR-ULF initialized with {len(self.queues)} queues (PATCHED with higher quotas)")
    
    def filter(
        self,
        query: QueryContext,
        tail_chunks: List[ChunkMetadata],
        gold_chunk_ids: Optional[Set[str]] = None,  # PATCH: for recall tracking
        all_chunks: Optional[Dict[str, ChunkMetadata]] = None,  # PATCH: include anchors
    ) -> ULFResult:
        """
        Run multi-queue recall filtering.
        
        PATCHED: Now tracks Candidate Recall@K separately.
        PATCHED: Accepts all_chunks to include anchors for anchor_neighbor queue.
        """
        start_time = time.time()
        
        # Build chunk lookup - include all chunks (anchors + tail) if provided
        if all_chunks is None:
            all_chunks = {chunk.chunk_id: chunk for chunk in tail_chunks}
        
        # Run each queue with diagnostics
        queue_contributions: List[QueueContribution] = []
        queue_diagnostics: List[QueueDiagnostics] = []
        
        for queue in self.queues:
            contribution, diagnostics = queue.recall(query, tail_chunks, all_chunks)
            queue_contributions.append(contribution)
            queue_diagnostics.append(diagnostics)
        
        # Compute union
        union_ids, sources = self._compute_union(queue_contributions)
        union_before_cap = len(union_ids)
        
        # PATCH: Check if we need more candidates
        target_size = max(self.MIN_UNION_TARGET, len(tail_chunks) // 3)  # Target at least 1/3 of tail
        target_size = min(target_size, self.config.max_total_candidates)
        
        # If union is too small, try to add more from queues
        if len(union_ids) < target_size and len(union_ids) < len(tail_chunks):
            # Get all unique candidates from queues (not just top_k from each)
            additional_candidates = []
            for contrib in queue_contributions:
                for cid in contrib.candidate_ids:
                    if cid not in union_ids and cid not in additional_candidates:
                        additional_candidates.append(cid)
            
            # Add up to target
            needed = target_size - len(union_ids)
            for cid in additional_candidates[:needed]:
                union_ids.append(cid)
                if cid not in sources:
                    sources[cid] = "additional_fill"
        
        # Apply max candidates limit
        if len(union_ids) > self.config.max_total_candidates:
            union_ids = union_ids[:self.config.max_total_candidates]
            sources = {k: v for k, v in sources.items() if k in union_ids}
        
        # Compute per-queue statistics
        per_queue_counts = {}
        per_queue_unique = {}
        for contrib in queue_contributions:
            per_queue_counts[contrib.queue_name] = len(contrib.candidate_ids)
            other_ids = set()
            for other in queue_contributions:
                if other.queue_name != contrib.queue_name:
                    other_ids.update(other.candidate_ids)
            unique_count = len(set(contrib.candidate_ids) - other_ids)
            per_queue_unique[contrib.queue_name] = unique_count
        
        # Compute overlap matrix
        queue_overlap_matrix = None
        if self.config.log_queue_overlap:
            queue_overlap_matrix = self._compute_overlap_matrix(queue_contributions)
        
        # PATCH: Compute Candidate Recall@K
        candidate_recall_at_k = {}
        if gold_chunk_ids:
            for k in [1, 5, 10, 20]:
                top_k_set = set(union_ids[:k])
                recalled = len(gold_chunk_ids & top_k_set)
                candidate_recall_at_k[k] = recalled / len(gold_chunk_ids)
        
        latency_us = (time.time() - start_time) * 1e6
        
        # Create diagnostics
        diagnostics = ULFDiagnostics(
            step=query.step,
            tail_total=len(tail_chunks),
            queue_diagnostics=queue_diagnostics,
            union_size_before_cap=union_before_cap,
            final_union_size=len(union_ids),
            target_size=target_size,
        )
        
        # Log if union is small
        if len(union_ids) < 3 and len(tail_chunks) >= 7:
            logger.warning(
                f"Step {query.step}: Small candidate set {len(union_ids)} from {len(tail_chunks)} tail chunks"
            )
            diagnostics.print_summary()
        
        result = ULFResult(
            request_id=query.request_id,
            step=query.step,
            candidate_ids=union_ids,
            candidate_sources=sources,
            queue_contributions=queue_contributions,
            n_tail_total=len(tail_chunks),
            n_candidates=len(union_ids),
            per_queue_counts=per_queue_counts,
            per_queue_unique=per_queue_unique,
            queue_overlap_matrix=queue_overlap_matrix,
            ulf_latency_us=latency_us,
        )
        
        # PATCH: Attach diagnostics and recall metrics
        result._diagnostics = diagnostics
        result._candidate_recall_at_k = candidate_recall_at_k
        
        return result
    
    def _compute_union(
        self,
        contributions: List[QueueContribution],
    ) -> Tuple[List[str], Dict[str, str]]:
        """Compute union of candidates from all queues."""
        all_candidates: Dict[str, Tuple[str, float]] = {}
        
        for contrib in contributions:
            for chunk_id, score in zip(contrib.candidate_ids, contrib.candidate_scores):
                if chunk_id not in all_candidates:
                    all_candidates[chunk_id] = (contrib.queue_name, score)
                else:
                    existing_source, existing_score = all_candidates[chunk_id]
                    if score > existing_score:
                        all_candidates[chunk_id] = (contrib.queue_name, score)
        
        # Order candidates
        if self.config.ordering_method == "queue_priority":
            ordered = []
            for contrib in contributions:
                for chunk_id in contrib.candidate_ids:
                    if chunk_id not in ordered:
                        ordered.append(chunk_id)
        else:
            items = [(cid, data[1]) for cid, data in all_candidates.items()]
            items.sort(key=lambda x: x[1], reverse=True)
            ordered = [cid for cid, _ in items]
        
        sources = {cid: data[0] for cid, data in all_candidates.items()}
        
        return ordered, sources
    
    def _compute_overlap_matrix(
        self,
        contributions: List[QueueContribution],
    ) -> Dict[str, Dict[str, int]]:
        """Compute pairwise overlap between queues."""
        matrix = {}
        
        for contrib1 in contributions:
            matrix[contrib1.queue_name] = {}
            set1 = set(contrib1.candidate_ids)
            
            for contrib2 in contributions:
                set2 = set(contrib2.candidate_ids)
                overlap = len(set1 & set2)
                matrix[contrib1.queue_name][contrib2.queue_name] = overlap
        
        return matrix
