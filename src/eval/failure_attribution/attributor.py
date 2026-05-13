"""
Fine-Grained Failure Attribution for ProSE-X 2.0

Assigns exactly ONE dominant failure reason per miss.
No "misc" bucket unless absolutely necessary.
"""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FailureReason(Enum):
    """
    Allowed failure reasons.
    
    Exactly one dominant reason per miss.
    Assignment must be deterministic.
    """
    # ULF stage failures
    CANDIDATE_MISS = "candidate_miss"
    QUEUE_MISS = "queue_miss"
    
    # Scoring stage failures
    SCORER_RANK_MISS = "scorer_rank_miss"
    LOW_SCORE = "low_score"
    
    # Scheduler stage failures
    SCHEDULER_BUDGET_DROP = "scheduler_budget_drop"
    SCHEDULER_THRESHOLD_DROP = "scheduler_threshold_drop"
    LOW_CONFIDENCE = "low_confidence"
    
    # Burst stage failures
    BURST_BOUNDARY_MISS = "burst_boundary_miss"
    
    # Sticky stage failures
    STICKY_EVICTION_MISS = "sticky_eviction_miss"
    TTL_EXPIRED = "ttl_expired"
    
    # Usage failures
    PROMOTED_BUT_UNUSED = "promoted_but_unused"
    
    # Retention failures
    RETENTION_MISS = "retention_miss"
    
    # Success (not a failure)
    RECOVERED = "recovered"
    UNKNOWN = "unknown"


@dataclass
class FailureEvent:
    """Record of a single failure/miss event."""
    step: int
    chunk_id: str
    reason: FailureReason
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Context at failure time
    was_candidate: bool = False
    candidate_rank: Optional[int] = None
    candidate_score: Optional[float] = None
    was_selected: bool = False
    was_burst_expanded: bool = False
    was_sticky: bool = False
    ttl_at_fail: Optional[int] = None


@dataclass
class FailureAttributionReport:
    """Complete failure attribution report."""
    total_steps: int = 0
    total_misses: int = 0
    
    # Histogram of reasons
    candidate_miss: int = 0
    scorer_rank_miss: int = 0
    scheduler_budget_drop: int = 0
    scheduler_threshold_drop: int = 0
    burst_boundary_miss: int = 0
    sticky_eviction_miss: int = 0
    promoted_but_unused: int = 0
    retention_miss: int = 0
    
    # Per-step breakdown
    events: List[FailureEvent] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_steps": self.total_steps,
            "total_misses": self.total_misses,
            "histogram": {
                "candidate_miss": self.candidate_miss,
                "scorer_rank_miss": self.scorer_rank_miss,
                "scheduler_budget_drop": self.scheduler_budget_drop,
                "scheduler_threshold_drop": self.scheduler_threshold_drop,
                "burst_boundary_miss": self.burst_boundary_miss,
                "sticky_eviction_miss": self.sticky_eviction_miss,
                "promoted_but_unused": self.promoted_but_unused,
                "retention_miss": self.retention_miss,
            },
            "events": [
                {
                    "step": e.step,
                    "chunk_id": e.chunk_id,
                    "reason": e.reason.value,
                    "details": e.details,
                }
                for e in self.events
            ],
        }


class FailureAttributor:
    """
    Assigns dominant failure reasons for misses.
    
    RULES:
    1. Exactly one dominant reason per miss
    2. Assign the EARLIEST stage where failure could have been prevented
    3. Deterministic assignment
    4. No "misc" bucket unless absolutely necessary
    """
    
    def __init__(self):
        """Initialize attributor."""
        self.events: List[FailureEvent] = []
        logger.info("FailureAttributor initialized")
    
    def attribute_failure(
        self,
        step: int,
        gold_chunk_id: str,
        gold_exists_in_system: bool,
        candidate_ids: List[str],
        scored_candidates: List[Dict[str, Any]],
        selected_ids: List[str],
        burst_expanded_ids: List[str],
        sticky_promoted_ids: List[str],
        was_previously_promoted: bool = False,
        previously_evicted: bool = False,
        accessed_after_promotion: bool = False,
        **context
    ) -> FailureReason:
        """
        Attribute the dominant failure reason for a miss.
        
        Args:
            step: Current step
            gold_chunk_id: The gold chunk that was missed
            gold_exists_in_system: Whether gold exists in tail/promoted universe
            candidate_ids: ULF output candidates
            scored_candidates: Scored candidates (sorted by score)
            selected_ids: Scheduler selected chunks
            burst_expanded_ids: Burst expansion output
            sticky_promoted_ids: Final sticky promoted set
            was_previously_promoted: Whether gold was promoted before
            previously_evicted: Whether gold was evicted
            accessed_after_promotion: Whether gold was accessed when promoted
            **context: Additional context
            
        Returns:
            The dominant FailureReason
        """
        # Rule: Check earliest stage first
        
        # Stage 0: Retention check
        if not gold_exists_in_system:
            reason = FailureReason.RETENTION_MISS
            details = {"issue": "gold_evicted_from_system"}
        
        # Stage 1: ULF - Did gold make it to candidates?
        elif gold_chunk_id not in candidate_ids:
            reason = FailureReason.CANDIDATE_MISS
            details = {
                "candidates": len(candidate_ids),
                "queues_contributing": context.get("queue_contributions", {}),
            }
        
        # Stage 2: Scorer - Was gold scored high enough?
        else:
            # Find gold in scored candidates
            gold_rank = None
            gold_score = None
            for i, cand in enumerate(scored_candidates):
                if cand.get("chunk_id") == gold_chunk_id:
                    gold_rank = i + 1
                    gold_score = cand.get("score", 0.0)
                    break
            
            if gold_rank is None:
                # Should not happen if in candidates
                reason = FailureReason.SCORER_RANK_MISS
                details = {"issue": "gold_not_in_scored"}
            
            elif gold_rank > len(selected_ids):
                # Gold was not in top selections
                # Check if it was a budget or threshold issue
                threshold = context.get("score_threshold", 0.0)
                
                if gold_score is not None and gold_score < threshold:
                    reason = FailureReason.SCHEDULER_THRESHOLD_DROP
                    details = {
                        "gold_rank": gold_rank,
                        "gold_score": gold_score,
                        "threshold": threshold,
                    }
                else:
                    reason = FailureReason.SCHEDULER_BUDGET_DROP
                    details = {
                        "gold_rank": gold_rank,
                        "gold_score": gold_score,
                        "selected_count": len(selected_ids),
                    }
            
            # Stage 3: Burst - Was gold near selected but outside burst?
            elif gold_chunk_id not in burst_expanded_ids:
                reason = FailureReason.BURST_BOUNDARY_MISS
                details = {
                    "selected_neighbors": selected_ids,
                    "burst_radius": context.get("burst_radius", 0),
                }
            
            # Stage 4: Sticky - Was gold promoted but evicted?
            elif was_previously_promoted and previously_evicted:
                reason = FailureReason.STICKY_EVICTION_MISS
                details = {"ttl_expired": True}
            
            # Stage 5: Usage - Was gold promoted but never used?
            elif was_previously_promoted and not accessed_after_promotion:
                reason = FailureReason.PROMOTED_BUT_UNUSED
                details = {"promotion_without_access": True}
            
            else:
                reason = FailureReason.UNKNOWN
                details = {"edge_case": True}
        
        # Record the event
        event = FailureEvent(
            step=step,
            chunk_id=gold_chunk_id,
            reason=reason,
            details=details,
            was_candidate=gold_chunk_id in candidate_ids,
            candidate_rank=gold_rank if 'gold_rank' in locals() else None,
            candidate_score=gold_score if 'gold_score' in locals() else None,
            was_selected=gold_chunk_id in selected_ids,
            was_burst_expanded=gold_chunk_id in burst_expanded_ids,
            was_sticky=gold_chunk_id in sticky_promoted_ids,
        )
        self.events.append(event)
        
        logger.debug(
            f"Step {step}: Gold {gold_chunk_id} missed - {reason.value}"
        )
        
        return reason
    
    def generate_report(self, total_steps: int) -> FailureAttributionReport:
        """
        Generate complete failure attribution report.
        
        Args:
            total_steps: Total number of steps in experiment
            
        Returns:
            FailureAttributionReport
        """
        report = FailureAttributionReport(
            total_steps=total_steps,
            total_misses=len(self.events),
            events=self.events,
        )
        
        # Count histogram
        for event in self.events:
            if event.reason == FailureReason.CANDIDATE_MISS:
                report.candidate_miss += 1
            elif event.reason == FailureReason.SCORER_RANK_MISS:
                report.scorer_rank_miss += 1
            elif event.reason == FailureReason.SCHEDULER_BUDGET_DROP:
                report.scheduler_budget_drop += 1
            elif event.reason == FailureReason.SCHEDULER_THRESHOLD_DROP:
                report.scheduler_threshold_drop += 1
            elif event.reason == FailureReason.BURST_BOUNDARY_MISS:
                report.burst_boundary_miss += 1
            elif event.reason == FailureReason.STICKY_EVICTION_MISS:
                report.sticky_eviction_miss += 1
            elif event.reason == FailureReason.PROMOTED_BUT_UNUSED:
                report.promoted_but_unused += 1
            elif event.reason == FailureReason.RETENTION_MISS:
                report.retention_miss += 1
        
        return report
    
    def get_stage_breakdown(self) -> Dict[str, Any]:
        """
        Get breakdown by pipeline stage.
        
        Returns:
            Dict with counts per stage
        """
        return {
            "ulf_stage": {
                "candidate_miss": sum(1 for e in self.events if e.reason == FailureReason.CANDIDATE_MISS),
            },
            "scorer_stage": {
                "scorer_rank_miss": sum(1 for e in self.events if e.reason == FailureReason.SCORER_RANK_MISS),
            },
            "scheduler_stage": {
                "budget_drop": sum(1 for e in self.events if e.reason == FailureReason.SCHEDULER_BUDGET_DROP),
                "threshold_drop": sum(1 for e in self.events if e.reason == FailureReason.SCHEDULER_THRESHOLD_DROP),
            },
            "burst_stage": {
                "boundary_miss": sum(1 for e in self.events if e.reason == FailureReason.BURST_BOUNDARY_MISS),
            },
            "sticky_stage": {
                "eviction_miss": sum(1 for e in self.events if e.reason == FailureReason.STICKY_EVICTION_MISS),
            },
            "usage_stage": {
                "promoted_but_unused": sum(1 for e in self.events if e.reason == FailureReason.PROMOTED_BUT_UNUSED),
            },
            "retention_stage": {
                "retention_miss": sum(1 for e in self.events if e.reason == FailureReason.RETENTION_MISS),
            },
        }
    
    def clear(self) -> None:
        """Clear all events."""
        self.events.clear()
        logger.info("FailureAttributor cleared")
