"""
Useful Bytes Accounting for ProSE-X 2.0

Strict, explicit accounting for whether promoted bytes are useful.

NON-NEGOTIABLE: A promoted byte is NEVER automatically counted as useful.
It must satisfy at least one explicit usefulness criterion.
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AccountingMode(Enum):
    """Accounting mode for usefulness determination."""
    ATTENTION_ACCESS_BASED = "attention_access_based"
    GOLD_OVERLAP_BASED = "gold_overlap_based"  # Evaluation only
    RECOVERY_EVENT_BASED = "recovery_event_based"


@dataclass
class PromotionUnit:
    """
    Complete record of a promotion event.
    
    Captures the full promote-to-use chain for traceability.
    """
    # Identity
    chunk_id: str
    step_promoted: int
    request_id: str
    
    # Source information
    center_unit_id: Optional[str] = None  # If burst-expanded, the core unit
    burst_expanded_ids: List[str] = field(default_factory=list)
    queue_of_origin: Optional[str] = None  # Which ULF queue produced this
    
    # Scoring info
    score: float = 0.0
    rank: int = 0
    selection_type: Optional[str] = None  # "exploit" or "explore"
    
    # Transfer info
    bytes_transferred: int = 0
    first_visible_step: Optional[int] = None
    
    # Usage tracking
    future_accesses: List[Dict[str, Any]] = field(default_factory=list)
    cumulative_attention: float = 0.0
    
    # Evaluation-only annotations (NEVER flow to online decisions)
    gold_overlap_tokens: int = 0
    overlaps_gold_region: bool = False
    contributed_to_recovery: bool = False
    
    # TTL lifecycle
    ttl_original: int = 0
    ttl_at_promotion: int = 0
    eviction_step: Optional[int] = None
    eviction_reason: Optional[str] = None
    
    # timestamps
    promotion_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "step_promoted": self.step_promoted,
            "request_id": self.request_id,
            "center_unit_id": self.center_unit_id,
            "burst_expanded_ids": self.burst_expanded_ids,
            "queue_of_origin": self.queue_of_origin,
            "score": self.score,
            "rank": self.rank,
            "selection_type": self.selection_type,
            "bytes_transferred": self.bytes_transferred,
            "first_visible_step": self.first_visible_step,
            "future_access_count": len(self.future_accesses),
            "cumulative_attention": self.cumulative_attention,
            "gold_overlap_tokens": self.gold_overlap_tokens,
            "overlaps_gold_region": self.overlaps_gold_region,
            "contributed_to_recovery": self.contributed_to_recovery,
            "ttl_original": self.ttl_original,
            "eviction_step": self.eviction_step,
            "eviction_reason": self.eviction_reason,
        }


@dataclass
class UsefulnessVerdict:
    """Verdict on whether a promotion unit is useful."""
    chunk_id: str
    is_useful: bool
    mode: AccountingMode
    reasons: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


class UsefulBytesAccountant:
    """
    Strict accountant for useful promoted bytes.
    
    Implements the rule: Promoted bytes are NOT automatically useful.
    They must satisfy explicit criteria.
    """
    
    def __init__(
        self,
        attention_threshold: float = 0.01,
        attention_window: int = 10,
    ):
        """
        Initialize accountant.
        
        Args:
            attention_threshold: Minimum attention weight to count as useful
            attention_window: Steps to look ahead for attention
        """
        self.attention_threshold = attention_threshold
        self.attention_window = attention_window
        
        # Storage
        self.units: Dict[str, PromotionUnit] = {}
        self.step_log: List[Dict[str, Any]] = []
        
        logger.info(
            f"UsefulBytesAccountant initialized: "
            f"attention_threshold={attention_threshold}, "
            f"attention_window={attention_window}"
        )
    
    def record_promotion(
        self,
        chunk_id: str,
        step: int,
        request_id: str,
        bytes_transferred: int,
        **kwargs
    ) -> PromotionUnit:
        """
        Record a new promotion event.
        
        Args:
            chunk_id: Unique chunk identifier
            step: Step when promoted
            request_id: Request identifier
            bytes_transferred: Bytes transferred for this promotion
            **kwargs: Additional metadata
            
        Returns:
            PromotionUnit record
        """
        unit = PromotionUnit(
            chunk_id=chunk_id,
            step_promoted=step,
            request_id=request_id,
            bytes_transferred=bytes_transferred,
            center_unit_id=kwargs.get("center_unit_id"),
            burst_expanded_ids=kwargs.get("burst_expanded_ids", []),
            queue_of_origin=kwargs.get("queue_of_origin"),
            score=kwargs.get("score", 0.0),
            rank=kwargs.get("rank", 0),
            selection_type=kwargs.get("selection_type"),
            first_visible_step=kwargs.get("first_visible_step"),
            ttl_original=kwargs.get("ttl_original", 0),
            ttl_at_promotion=kwargs.get("ttl_at_promotion", 0),
        )
        
        self.units[chunk_id] = unit
        
        logger.debug(f"Recorded promotion: {chunk_id} at step {step}")
        
        return unit
    
    def record_access(
        self,
        chunk_id: str,
        step: int,
        attention_weight: float,
    ) -> None:
        """
        Record that a promoted chunk was accessed.
        
        Args:
            chunk_id: Chunk that was accessed
            step: Access step
            attention_weight: Attention weight received
        """
        if chunk_id not in self.units:
            logger.warning(f"Access recorded for unknown chunk: {chunk_id}")
            return
        
        unit = self.units[chunk_id]
        
        # Record the access
        access_record = {
            "step": step,
            "attention_weight": attention_weight,
            "steps_after_promotion": step - unit.step_promoted,
        }
        unit.future_accesses.append(access_record)
        unit.cumulative_attention += attention_weight
        
        logger.debug(
            f"Recorded access: {chunk_id} at step {step}, "
            f"attention={attention_weight:.4f}"
        )
    
    def record_eviction(
        self,
        chunk_id: str,
        step: int,
        reason: str,
    ) -> None:
        """
        Record that a chunk was evicted.
        
        Args:
            chunk_id: Chunk that was evicted
            step: Eviction step
            reason: Why it was evicted
        """
        if chunk_id not in self.units:
            return
        
        unit = self.units[chunk_id]
        unit.eviction_step = step
        unit.eviction_reason = reason
        
        logger.debug(f"Recorded eviction: {chunk_id} at step {step}, reason={reason}")
    
    def evaluate_usefulness(
        self,
        mode: AccountingMode,
        chunk_id: Optional[str] = None,
    ) -> List[UsefulnessVerdict]:
        """
        Evaluate usefulness of promotions.
        
        Args:
            mode: Accounting mode to use
            chunk_id: If specified, evaluate only this chunk
            
        Returns:
            List of usefulness verdicts
        """
        if chunk_id:
            units = [self.units.get(chunk_id)] if chunk_id in self.units else []
        else:
            units = list(self.units.values())
        
        verdicts = []
        
        for unit in units:
            if unit is None:
                continue
                
            verdict = self._evaluate_single_unit(unit, mode)
            verdicts.append(verdict)
        
        return verdicts
    
    def _evaluate_single_unit(
        self,
        unit: PromotionUnit,
        mode: AccountingMode,
    ) -> UsefulnessVerdict:
        """Evaluate usefulness of a single unit."""
        reasons = []
        evidence = {}
        
        if mode == AccountingMode.ATTENTION_ACCESS_BASED:
            # Criterion: Was accessed with sufficient attention
            valid_accesses = [
                a for a in unit.future_accesses
                if a["attention_weight"] >= self.attention_threshold
                and a["steps_after_promotion"] <= self.attention_window
            ]
            
            is_useful = len(valid_accesses) > 0
            
            if is_useful:
                reasons.append(f"accessed_with_threshold_{self.attention_threshold}")
                evidence["access_count"] = len(valid_accesses)
                evidence["max_attention"] = max(
                    (a["attention_weight"] for a in valid_accesses), default=0.0
                )
            
        elif mode == AccountingMode.GOLD_OVERLAP_BASED:
            # Criterion: Overlaps with gold region (evaluation only)
            is_useful = unit.overlaps_gold_region or unit.gold_overlap_tokens > 0
            
            if is_useful:
                reasons.append(f"gold_overlap_{unit.gold_overlap_tokens}_tokens")
                evidence["gold_overlap_tokens"] = unit.gold_overlap_tokens
            
        elif mode == AccountingMode.RECOVERY_EVENT_BASED:
            # Criterion: Contributed to a recovery event
            is_useful = unit.contributed_to_recovery
            
            if is_useful:
                reasons.append("contributed_to_recovery_event")
                evidence["recovery_event"] = True
        
        else:
            is_useful = False
            reasons.append("unknown_accounting_mode")
        
        return UsefulnessVerdict(
            chunk_id=unit.chunk_id,
            is_useful=is_useful,
            mode=mode,
            reasons=reasons,
            evidence=evidence,
        )
    
    def get_accounting_report(
        self,
        modes: Optional[List[AccountingMode]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive accounting report.
        
        Args:
            modes: List of modes to report (defaults to all)
            
        Returns:
            Dictionary with accounting results for each mode
        """
        if modes is None:
            modes = list(AccountingMode)
        
        total_bytes = sum(u.bytes_transferred for u in self.units.values())
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_units": len(self.units),
            "total_promoted_bytes": total_bytes,
            "modes": {},
        }
        
        for mode in modes:
            verdicts = self.evaluate_usefulness(mode)
            
            useful_count = sum(1 for v in verdicts if v.is_useful)
            useful_bytes = sum(
                self.units[v.chunk_id].bytes_transferred
                for v in verdicts if v.is_useful
            )
            
            upr = useful_bytes / total_bytes if total_bytes > 0 else 0.0
            
            report["modes"][mode.value] = {
                "useful_units": useful_count,
                "total_units": len(verdicts),
                "useful_bytes": useful_bytes,
                "total_bytes": total_bytes,
                "upr": upr,
                "verdicts": [
                    {
                        "chunk_id": v.chunk_id,
                        "is_useful": v.is_useful,
                        "reasons": v.reasons,
                    }
                    for v in verdicts
                ],
            }
        
        return report
    
    def get_queue_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """
        Get usefulness breakdown by queue of origin.
        
        Returns:
            Dict mapping queue_name -> stats
        """
        breakdown = {}
        
        for unit in self.units.values():
            queue = unit.queue_of_origin or "unknown"
            
            if queue not in breakdown:
                breakdown[queue] = {
                    "total_units": 0,
                    "total_bytes": 0,
                    "units_by_mode": {mode.value: 0 for mode in AccountingMode},
                    "bytes_by_mode": {mode.value: 0 for mode in AccountingMode},
                }
            
            breakdown[queue]["total_units"] += 1
            breakdown[queue]["total_bytes"] += unit.bytes_transferred
            
            # Evaluate for each mode
            for mode in AccountingMode:
                verdict = self._evaluate_single_unit(unit, mode)
                if verdict.is_useful:
                    breakdown[queue]["units_by_mode"][mode.value] += 1
                    breakdown[queue]["bytes_by_mode"][mode.value] += unit.bytes_transferred
        
        # Compute ratios
        for queue, stats in breakdown.items():
            total_bytes = stats["total_bytes"]
            for mode in AccountingMode:
                useful_bytes = stats["bytes_by_mode"][mode.value]
                stats[f"upr_{mode.value}"] = (
                    useful_bytes / total_bytes if total_bytes > 0 else 0.0
                )
        
        return breakdown
    
    def clear(self) -> None:
        """Clear all records."""
        self.units.clear()
        self.step_log.clear()
        logger.info("UsefulBytesAccountant cleared")
