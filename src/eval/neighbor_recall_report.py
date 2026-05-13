"""
Neighbor Recall Contribution Report for ProSE-X 2.0

Generates detailed report on anchor_neighbor queue contributions.
Goal: Prove neighbor recall produces not just candidates, but useful promotions.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QueueStageMetrics:
    """Metrics for a queue at a specific pipeline stage."""
    count: int = 0
    bytes: int = 0
    chunk_ids: List[str] = field(default_factory=list)


@dataclass
class NeighborQueueReport:
    """Detailed report for anchor_neighbor queue."""
    
    # Pipeline stage counts
    raw_output: QueueStageMetrics = field(default_factory=QueueStageMetrics)
    post_dedup: QueueStageMetrics = field(default_factory=QueueStageMetrics)
    entering_scorer: QueueStageMetrics = field(default_factory=QueueStageMetrics)
    surviving_scheduler: QueueStageMetrics = field(default_factory=QueueStageMetrics)
    burst_expanded: QueueStageMetrics = field(default_factory=QueueStageMetrics)
    ultimately_useful: QueueStageMetrics = field(default_factory=QueueStageMetrics)
    
    # Empty reason tracking (why anchor_neighbor is empty)
    empty_reasons: Dict[str, int] = field(default_factory=dict)
    
    # Detailed records
    promoted_units: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "raw_output": {
                "count": self.raw_output.count,
                "bytes": self.raw_output.bytes,
            },
            "post_dedup": {
                "count": self.post_dedup.count,
                "bytes": self.post_dedup.bytes,
            },
            "entering_scorer": {
                "count": self.entering_scorer.count,
                "bytes": self.entering_scorer.bytes,
            },
            "surviving_scheduler": {
                "count": self.surviving_scheduler.count,
                "bytes": self.surviving_scheduler.bytes,
            },
            "burst_expanded": {
                "count": self.burst_expanded.count,
                "bytes": self.burst_expanded.bytes,
            },
            "ultimately_useful": {
                "count": self.ultimately_useful.count,
                "bytes": self.ultimately_useful.bytes,
            },
            "empty_reasons": self.empty_reasons,
            "promoted_units": self.promoted_units,
        }


class NeighborRecallReporter:
    """
    Generates detailed neighbor recall contribution report.
    
    This answers Q1: Does neighbor recall produce not just candidates,
    but useful promotions?
    """
    
    # Valid empty reasons (from metrics_spec.md)
    VALID_EMPTY_REASONS = [
        "no_active_anchors",
        "zero_radius",
        "dedup_removed_all",
        "queue_cap_zero",
        "filtering_removed_all",
        "no_neighbors_in_radius",
        "queue_disabled",
    ]
    
    def __init__(self):
        """Initialize reporter."""
        self.report = NeighborQueueReport()
        self.step_data: List[Dict[str, Any]] = []
        logger.info("NeighborRecallReporter initialized")
    
    def record_step(
        self,
        step: int,
        raw_output: List[str],
        post_dedup: List[str],
        entering_scorer: List[str],
        surviving_scheduler: List[str],
        burst_expanded: List[str],
        ultimately_useful: List[str],
        empty_reason: Optional[str] = None,
        chunk_bytes: Optional[Dict[str, int]] = None,
        debug_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record data for one step.
        
        Args:
            step: Step number
            raw_output: Raw output from anchor_neighbor queue
            post_dedup: After deduplication
            entering_scorer: Candidates entering scorer
            surviving_scheduler: Chunks selected by scheduler
            burst_expanded: Additional chunks from burst
            ultimately_useful: Chunks that proved useful
            empty_reason: If empty, why
            chunk_bytes: Map of chunk_id -> bytes
            debug_info: Additional debug info
        """
        chunk_bytes = chunk_bytes or {}
        
        # Update aggregate counts
        def add_to_stage(stage: QueueStageMetrics, chunk_ids: List[str]):
            for cid in chunk_ids:
                stage.count += 1
                stage.bytes += chunk_bytes.get(cid, 0)
                if cid not in stage.chunk_ids:
                    stage.chunk_ids.append(cid)
        
        add_to_stage(self.report.raw_output, raw_output)
        add_to_stage(self.report.post_dedup, post_dedup)
        add_to_stage(self.report.entering_scorer, entering_scorer)
        add_to_stage(self.report.surviving_scheduler, surviving_scheduler)
        add_to_stage(self.report.burst_expanded, burst_expanded)
        add_to_stage(self.report.ultimately_useful, ultimately_useful)
        
        # Track empty reasons
        if empty_reason:
            if empty_reason not in self.report.empty_reasons:
                self.report.empty_reasons[empty_reason] = 0
            self.report.empty_reasons[empty_reason] += 1
        
        # Store step data
        self.step_data.append({
            "step": step,
            "raw_count": len(raw_output),
            "post_dedup_count": len(post_dedup),
            "entering_scorer_count": len(entering_scorer),
            "surviving_scheduler_count": len(surviving_scheduler),
            "burst_expanded_count": len(burst_expanded),
            "ultimately_useful_count": len(ultimately_useful),
            "empty_reason": empty_reason,
            "debug_info": debug_info or {},
        })
    
    def record_promoted_unit(
        self,
        chunk_id: str,
        step: int,
        bytes_transferred: int,
        score: float,
        rank: int,
        selection_type: str,
        from_anchor_neighbor: bool,
        became_useful: bool,
        usefulness_reason: Optional[str] = None,
    ) -> None:
        """
        Record a promoted unit from anchor_neighbor path.
        
        Args:
            chunk_id: Chunk ID
            step: Promotion step
            bytes_transferred: Bytes
            score: Score from scorer
            rank: Rank in candidates
            selection_type: "exploit" or "explore"
            from_anchor_neighbor: Whether from anchor_neighbor queue
            became_useful: Whether it proved useful
            usefulness_reason: Why it was useful
        """
        if not from_anchor_neighbor:
            return
        
        unit = {
            "chunk_id": chunk_id,
            "step": step,
            "bytes_transferred": bytes_transferred,
            "score": score,
            "rank": rank,
            "selection_type": selection_type,
            "became_useful": became_useful,
            "usefulness_reason": usefulness_reason,
        }
        
        self.report.promoted_units.append(unit)
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate complete neighbor recall report.
        
        Returns:
            Comprehensive report dictionary
        """
        report = self.report.to_dict()
        
        # Add derived metrics
        raw_count = self.report.raw_output.count
        useful_count = self.report.ultimately_useful.count
        
        report["funnel"] = {
            "raw_to_dedup": {
                "input": raw_count,
                "output": self.report.post_dedup.count,
                "retention": self.report.post_dedup.count / max(raw_count, 1),
            },
            "dedup_to_scorer": {
                "input": self.report.post_dedup.count,
                "output": self.report.entering_scorer.count,
                "retention": self.report.entering_scorer.count / max(self.report.post_dedup.count, 1),
            },
            "scorer_to_scheduler": {
                "input": self.report.entering_scorer.count,
                "output": self.report.surviving_scheduler.count,
                "retention": self.report.surviving_scheduler.count / max(self.report.entering_scorer.count, 1),
            },
            "scheduler_to_burst": {
                "input": self.report.surviving_scheduler.count,
                "output": self.report.burst_expanded.count + self.report.surviving_scheduler.count,
                "expansion_factor": (self.report.burst_expanded.count + self.report.surviving_scheduler.count) / max(self.report.surviving_scheduler.count, 1),
            },
            "promotion_to_useful": {
                "promoted": self.report.burst_expanded.count + self.report.surviving_scheduler.count,
                "useful": useful_count,
                "yield": useful_count / max(self.report.burst_expanded.count + self.report.surviving_scheduler.count, 1),
            },
        }
        
        # Empty reason breakdown
        report["empty_reason_analysis"] = {
            "total_steps_with_empty": sum(self.report.empty_reasons.values()),
            "breakdown": self.report.empty_reasons,
        }
        
        # Per-step summary
        report["step_summary"] = {
            "total_steps": len(self.step_data),
            "steps_with_any_neighbor_output": sum(
                1 for s in self.step_data if s["raw_count"] > 0
            ),
            "steps_with_useful_neighbor": sum(
                1 for s in self.step_data if s["ultimately_useful_count"] > 0
            ),
        }
        
        # Key findings
        report["key_findings"] = self._generate_key_findings()
        
        report["timestamp"] = datetime.now().isoformat()
        
        return report
    
    def _generate_key_findings(self) -> List[str]:
        """Generate key findings from the data."""
        findings = []
        
        raw = self.report.raw_output.count
        useful = self.report.ultimately_useful.count
        
        if raw == 0:
            findings.append("CRITICAL: anchor_neighbor produced zero raw candidates")
        elif useful == 0:
            findings.append("WARNING: anchor_neighbor produced candidates but none were useful")
        else:
            yield_rate = useful / raw
            findings.append(
                f"anchor_neighbor yield: {useful}/{raw} = {yield_rate:.2%} "
                "(raw to useful)"
            )
        
        # Check for empty reason patterns
        if self.report.empty_reasons:
            most_common = max(self.report.empty_reasons.items(), key=lambda x: x[1])
            findings.append(
                f"Most common empty reason: '{most_common[0]}' ({most_common[1]} steps)"
            )
        
        return findings
    
    def save(self, output_path: str) -> None:
        """Save report to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        report = self.generate_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Neighbor recall report saved to {output_path}")
    
    def clear(self) -> None:
        """Clear all data."""
        self.report = NeighborQueueReport()
        self.step_data.clear()
        logger.info("NeighborRecallReporter cleared")


def generate_neighbor_recall_report(
    step_results: List[Dict[str, Any]],
    output_path: str,
) -> Dict[str, Any]:
    """
    Convenience function to generate report from step results.
    
    Args:
        step_results: List of step results with queue contribution data
        output_path: Where to save the report
        
    Returns:
        The report dictionary
    """
    reporter = NeighborRecallReporter()
    
    for step_result in step_results:
        queue_contribs = step_result.get("queue_contributions", {})
        anchor_data = queue_contribs.get("anchor_neighbor", {})
        
        reporter.record_step(
            step=step_result.get("step", 0),
            raw_output=anchor_data.get("raw_output", []),
            post_dedup=anchor_data.get("post_dedup", []),
            entering_scorer=anchor_data.get("entering_scorer", []),
            surviving_scheduler=anchor_data.get("surviving_scheduler", []),
            burst_expanded=anchor_data.get("burst_expanded", []),
            ultimately_useful=anchor_data.get("ultimately_useful", []),
            empty_reason=anchor_data.get("empty_reason"),
            chunk_bytes=step_result.get("chunk_bytes", {}),
        )
    
    reporter.save(output_path)
    return reporter.generate_report()
