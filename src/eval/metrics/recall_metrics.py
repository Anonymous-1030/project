"""
Recall metrics for ProSE-X 2.0.

Candidate-stage metrics:
- Candidate Recall@K
- Union candidate set size
- Per-queue contribution counts
- Per-queue overlap and unique contribution
- Missed-gold-before-scoring flag
"""

import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass

from src.core_types import ULFResult, FailureReason

logger = logging.getLogger(__name__)


@dataclass
class CandidateMetrics:
    """Metrics for the candidate generation stage."""
    
    # Recall metrics
    candidate_recall_at_k: Dict[int, float]  # K -> recall
    
    # Set statistics
    n_tail_total: int
    n_candidates: int
    candidate_set_ratio: float
    
    # Per-queue statistics
    per_queue_counts: Dict[str, int]
    per_queue_unique: Dict[str, int]
    per_queue_recall: Dict[str, float]  # If gold known
    
    # Overlap statistics
    queue_overlap_matrix: Optional[Dict[str, Dict[str, int]]]
    avg_queue_overlap: float
    
    # Miss tracking
    missed_gold_before_scoring: bool
    missed_gold_ids: List[str]
    
    # Latency
    ulf_latency_us: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "candidate_recall_at_k": self.candidate_recall_at_k,
            "n_tail_total": self.n_tail_total,
            "n_candidates": self.n_candidates,
            "candidate_set_ratio": self.candidate_set_ratio,
            "per_queue_counts": self.per_queue_counts,
            "per_queue_unique": self.per_queue_unique,
            "per_queue_recall": self.per_queue_recall,
            "avg_queue_overlap": self.avg_queue_overlap,
            "missed_gold_before_scoring": self.missed_gold_before_scoring,
            "missed_gold_ids": self.missed_gold_ids,
            "ulf_latency_us": self.ulf_latency_us,
        }


class CandidateMetricsCalculator:
    """Calculator for candidate-stage metrics."""
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 5, 10, 20]
    
    def compute(
        self,
        ulf_result: ULFResult,
        gold_chunk_ids: Optional[Set[str]] = None,
    ) -> CandidateMetrics:
        """
        Compute candidate-stage metrics.
        
        Args:
            ulf_result: Output from ULF
            gold_chunk_ids: Optional set of gold chunk IDs for recall computation
            
        Returns:
            CandidateMetrics
        """
        # Basic counts
        n_tail = ulf_result.n_tail_total
        n_candidates = ulf_result.n_candidates
        
        # Candidate set ratio
        ratio = n_candidates / max(n_tail, 1)
        
        # Compute recall@K
        recall_at_k = {}
        if gold_chunk_ids:
            for k in self.k_values:
                top_k = set(ulf_result.candidate_ids[:k])
                recalled = len(top_k & gold_chunk_ids)
                recall_at_k[k] = recalled / max(len(gold_chunk_ids), 1)
        else:
            recall_at_k = {k: 0.0 for k in self.k_values}
        
        # Per-queue recall (if gold known)
        per_queue_recall = {}
        if gold_chunk_ids:
            for contrib in ulf_result.queue_contributions:
                queue_set = set(contrib.candidate_ids)
                recalled = len(queue_set & gold_chunk_ids)
                per_queue_recall[contrib.queue_name] = recalled / max(len(gold_chunk_ids), 1)
        
        # Compute average overlap
        avg_overlap = 0.0
        if ulf_result.queue_overlap_matrix:
            overlaps = []
            for q1, row in ulf_result.queue_overlap_matrix.items():
                for q2, val in row.items():
                    if q1 != q2:
                        overlaps.append(val)
            if overlaps:
                avg_overlap = sum(overlaps) / len(overlaps)
        
        # Miss detection
        missed_gold = []
        if gold_chunk_ids:
            candidate_set = set(ulf_result.candidate_ids)
            missed_gold = [gid for gid in gold_chunk_ids if gid not in candidate_set]
        
        return CandidateMetrics(
            candidate_recall_at_k=recall_at_k,
            n_tail_total=n_tail,
            n_candidates=n_candidates,
            candidate_set_ratio=ratio,
            per_queue_counts=ulf_result.per_queue_counts,
            per_queue_unique=ulf_result.per_queue_unique,
            per_queue_recall=per_queue_recall,
            queue_overlap_matrix=ulf_result.queue_overlap_matrix,
            avg_queue_overlap=avg_overlap,
            missed_gold_before_scoring=len(missed_gold) > 0,
            missed_gold_ids=missed_gold,
            ulf_latency_us=ulf_result.ulf_latency_us,
        )


@dataclass
class ScoringMetrics:
    """Metrics for the scoring stage."""
    
    # Ranking metrics
    gold_rank: Optional[int]  # Rank of gold chunk (if present)
    gold_score: Optional[float]  # Score of gold chunk
    top_score: float  # Score of top candidate
    score_margin: Optional[float]  # top_score - gold_score
    
    # Threshold metrics
    n_above_threshold: int
    gold_above_threshold: bool
    
    # Distribution metrics
    score_mean: float
    score_std: float
    score_min: float
    score_max: float
    
    # Miss tracking
    missed_gold_after_scoring: bool
    
    # Latency
    scorer_latency_us: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "gold_rank": self.gold_rank,
            "gold_score": self.gold_score,
            "top_score": self.top_score,
            "score_margin": self.score_margin,
            "n_above_threshold": self.n_above_threshold,
            "gold_above_threshold": self.gold_above_threshold,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "missed_gold_after_scoring": self.missed_gold_after_scoring,
            "scorer_latency_us": self.scorer_latency_us,
        }


class ScoringMetricsCalculator:
    """Calculator for scoring-stage metrics."""
    
    def compute(
        self,
        scorer_result: Any,  # ScorerResult
        gold_chunk_id: Optional[str] = None,
    ) -> ScoringMetrics:
        """Compute scoring-stage metrics."""
        candidates = scorer_result.candidates
        
        if not candidates:
            return ScoringMetrics(
                gold_rank=None,
                gold_score=None,
                top_score=0.0,
                score_margin=None,
                n_above_threshold=0,
                gold_above_threshold=False,
                score_mean=0.0,
                score_std=0.0,
                score_min=0.0,
                score_max=0.0,
                missed_gold_after_scoring=gold_chunk_id is not None,
                scorer_latency_us=scorer_result.scorer_latency_us,
            )
        
        scores = [c.score for c in candidates]
        
        # Find gold rank
        gold_rank = None
        gold_score = None
        if gold_chunk_id:
            for i, c in enumerate(candidates):
                if c.chunk_id == gold_chunk_id:
                    gold_rank = i + 1  # 1-indexed
                    gold_score = c.score
                    break
        
        # Score margin
        top_score = scores[0]
        score_margin = None
        if gold_score is not None:
            score_margin = top_score - gold_score
        
        # Distribution
        import numpy as np
        score_mean = float(np.mean(scores))
        score_std = float(np.std(scores))
        score_min = float(np.min(scores))
        score_max = float(np.max(scores))
        
        # Miss detection
        missed_after_scoring = gold_chunk_id is not None and gold_rank is None
        
        return ScoringMetrics(
            gold_rank=gold_rank,
            gold_score=gold_score,
            top_score=top_score,
            score_margin=score_margin,
            n_above_threshold=scorer_result.n_above_threshold,
            gold_above_threshold=gold_score is not None and gold_score >= scorer_result.score_threshold,
            score_mean=score_mean,
            score_std=score_std,
            score_min=score_min,
            score_max=score_max,
            missed_gold_after_scoring=missed_after_scoring,
            scorer_latency_us=scorer_result.scorer_latency_us,
        )


@dataclass
class SchedulerMetrics:
    """Metrics for the scheduler stage."""
    
    # Selection counts
    n_exploit: int
    n_explore: int
    n_dropped: int
    
    # Drop reasons
    n_dropped_budget: int
    n_dropped_low_score: int
    n_dropped_low_confidence: int
    
    # Budget utilization
    budget_bytes: int
    used_bytes: int
    utilization: float
    
    # Miss tracking
    missed_gold_due_to_budget: bool
    missed_gold_due_to_score: bool
    
    # Latency
    scheduler_latency_us: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "n_exploit": self.n_exploit,
            "n_explore": self.n_explore,
            "n_dropped": self.n_dropped,
            "n_dropped_budget": self.n_dropped_budget,
            "n_dropped_low_score": self.n_dropped_low_score,
            "n_dropped_low_confidence": self.n_dropped_low_confidence,
            "budget_bytes": self.budget_bytes,
            "used_bytes": self.used_bytes,
            "utilization": self.utilization,
            "missed_gold_due_to_budget": self.missed_gold_due_to_budget,
            "missed_gold_due_to_score": self.missed_gold_due_to_score,
            "scheduler_latency_us": self.scheduler_latency_us,
        }


class SchedulerMetricsCalculator:
    """Calculator for scheduler-stage metrics."""
    
    def compute(
        self,
        scheduler_result: Any,  # SchedulerResult
        gold_chunk_id: Optional[str] = None,
    ) -> SchedulerMetrics:
        """Compute scheduler-stage metrics."""
        # Determine miss reasons
        missed_due_to_budget = False
        missed_due_to_score = False
        
        if gold_chunk_id and gold_chunk_id in scheduler_result.dropped_ids:
            # Find the decision
            for decision in scheduler_result.dropped_decisions:
                if decision.chunk_id == gold_chunk_id:
                    if decision.rejection_reason == FailureReason.SCHEDULER_BUDGET_CUT:
                        missed_due_to_budget = True
                    elif decision.rejection_reason in [FailureReason.LOW_SCORE, FailureReason.SCORE_THRESHOLD_MISS]:
                        missed_due_to_score = True
                    break
        
        return SchedulerMetrics(
            n_exploit=scheduler_result.n_exploit,
            n_explore=scheduler_result.n_explore,
            n_dropped=len(scheduler_result.dropped_ids),
            n_dropped_budget=scheduler_result.n_dropped_budget,
            n_dropped_low_score=scheduler_result.n_dropped_low_score,
            n_dropped_low_confidence=scheduler_result.n_dropped_low_confidence,
            budget_bytes=scheduler_result.budget_bytes,
            used_bytes=scheduler_result.used_bytes,
            utilization=scheduler_result.utilization,
            missed_gold_due_to_budget=missed_due_to_budget,
            missed_gold_due_to_score=missed_due_to_score,
            scheduler_latency_us=scheduler_result.scheduler_latency_us,
        )
