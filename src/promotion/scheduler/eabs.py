"""
Exploration-Aware Budget Scheduler (EABS) for ProSE-X 2.0.

PATCH NOTES (2026-03-24):
- Fixed scheduler veto behavior: mandatory exploit fallback ensures at least top-1 selection
- Score threshold converted from hard veto to soft advisory
- Added invariant checks with explicit failure reasons
- Full decision logging for debugging

Split budget into exploit and explore portions:
- exploit: top scored candidates
- explore: controlled sampling from uncertain / under-validated candidates
"""

import time
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

from src.core_types import (
    ChunkMetadata, QueryContext, ScorerResult, SchedulerResult,
    SchedulerDecision, PromoteDecision, FailureReason
)
from src.config import EABSConfig

logger = logging.getLogger(__name__)


class SchedulerDecisionLog:
    """Detailed log of scheduler decision inputs and outputs."""
    
    def __init__(self):
        self.exploit_slots: int = 0
        self.explore_slots: int = 0
        self.score_threshold: float = 0.0
        self.confidence_threshold: float = 0.0
        self.budget_bytes: int = 0
        self.candidate_scores: List[Tuple[str, float, float]] = []  # (id, score, confidence)
        self.selected_ids: List[str] = []
        self.rejected: List[Tuple[str, float, float, str]] = []  # (id, score, confidence, reason)
        self.fallback_triggered: bool = False
        self.invariant_violations: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "exploit_slots": self.exploit_slots,
            "explore_slots": self.explore_slots,
            "score_threshold": self.score_threshold,
            "confidence_threshold": self.confidence_threshold,
            "budget_bytes": self.budget_bytes,
            "candidate_scores": self.candidate_scores,
            "selected_ids": self.selected_ids,
            "rejected": self.rejected,
            "fallback_triggered": self.fallback_triggered,
            "invariant_violations": self.invariant_violations,
        }
    
    def print_summary(self) -> None:
        """Print detailed decision summary."""
        print(f"\n[Scheduler Decision Log]")
        print(f"  Config: exploit_slots={self.exploit_slots}, explore_slots={self.explore_slots}")
        print(f"  Thresholds: score={self.score_threshold:.3f}, confidence={self.confidence_threshold:.3f}")
        print(f"  Budget: {self.budget_bytes} bytes")
        print(f"  Candidates ({len(self.candidate_scores)}):")
        for cid, score, conf in self.candidate_scores[:10]:  # Print first 10
            marker = " <-- SELECTED" if cid in self.selected_ids else ""
            print(f"    - {cid}: score={score:.3f}, conf={conf:.3f}{marker}")
        if len(self.candidate_scores) > 10:
            print(f"    ... and {len(self.candidate_scores) - 10} more")
        print(f"  Selected: {len(self.selected_ids)} chunks")
        print(f"  Rejected: {len(self.rejected)} chunks")
        for cid, score, conf, reason in self.rejected[:5]:
            print(f"    - {cid}: score={score:.3f}, conf={conf:.3f}, reason={reason}")
        if self.fallback_triggered:
            print(f"  *** FALLBACK TRIGGERED: forced selection of top candidate ***")
        if self.invariant_violations:
            print(f"  *** INVARIANT VIOLATIONS: ***")
            for v in self.invariant_violations:
                print(f"    - {v}")


class ExplorationAwareBudgetScheduler:
    """
    Exploration-Aware Budget Scheduler.
    
    Makes final promotion decisions with explicit exploit/explore split.
    
    PATCHED BEHAVIOR:
    - Mandatory exploit fallback: always selects at least top-1 if candidates exist
    - Score threshold is advisory, not veto
    - Full decision logging
    - Invariant checks
    """
    
    def __init__(self, config: EABSConfig):
        self.config = config
        
        # Validation
        assert 0.0 <= config.exploration_ratio <= 1.0, \
            f"Exploration ratio must be in [0, 1], got {config.exploration_ratio}"
        
        logger.info(
            f"EABS initialized: exploration_ratio={config.exploration_ratio}, "
            f"max_chunks={config.max_chunks_per_step}, "
            f"score_threshold={config.min_score_threshold}, "
            f"conf_threshold={config.confidence_threshold}"
        )
    
    def schedule(
        self,
        scorer_result: ScorerResult,
        query: QueryContext,
        all_chunks: Dict[str, ChunkMetadata],
        budget_bytes: int,
    ) -> SchedulerResult:
        """
        Schedule promotions with exploit/explore split.
        
        PATCHED: Always selects at least top-1 exploit candidate if candidates exist.
        """
        start_time = time.time()
        decision_log = SchedulerDecisionLog()
        
        candidates = scorer_result.candidates
        n_candidates = len(candidates)
        
        # Populate decision log
        decision_log.score_threshold = self.config.min_score_threshold
        decision_log.confidence_threshold = self.config.confidence_threshold
        decision_log.budget_bytes = budget_bytes
        decision_log.candidate_scores = [
            (c.chunk_id, c.score, c.confidence) for c in candidates
        ]
        
        if n_candidates == 0:
            return SchedulerResult(
                request_id=query.request_id,
                step=query.step,
                selected_ids=[],
                selected_decisions=[],
                exploit_ids=[],
                explore_ids=[],
                dropped_ids=[],
                dropped_decisions=[],
                budget_bytes=budget_bytes,
                used_bytes=0,
                utilization=0.0,
                n_exploit=0,
                n_explore=0,
                n_dropped_budget=0,
                n_dropped_low_score=0,
                n_dropped_low_confidence=0,
                scheduler_latency_us=(time.time() - start_time) * 1e6,
            )
        
        # Compute budget split
        total_slots = min(self.config.max_chunks_per_step, n_candidates)
        
        # PATCH: Ensure at least 1 exploit slot if candidates exist
        # This prevents the case where all slots go to exploration
        explore_slots = max(
            self.config.exploration_min_budget,
            int(total_slots * self.config.exploration_ratio)
        )
        exploit_slots = max(1, total_slots - explore_slots)  # At least 1 exploit slot
        
        decision_log.exploit_slots = exploit_slots
        decision_log.explore_slots = explore_slots
        
        # Split candidates into exploit and explore pools
        exploit_candidates, explore_pool = self._split_candidates(
            candidates, exploit_slots
        )
        
        # Select exploit candidates (with threshold as advisory)
        selected_exploit, exploit_rejected = self._select_exploit_advisory(
            exploit_candidates, all_chunks, budget_bytes, decision_log
        )
        
        # MANDATORY FALLBACK: If no exploit selected but candidates exist, force top-1
        if not selected_exploit and exploit_candidates and budget_bytes > 0:
            top_candidate = exploit_candidates[0]
            chunk = all_chunks.get(top_candidate.chunk_id)
            if chunk and chunk.logical_bytes <= budget_bytes:
                selected_exploit = [top_candidate]
                decision_log.fallback_triggered = True
                logger.warning(
                    f"Step {query.step}: MANDATORY FALLBACK triggered - "
                    f"forcing selection of top candidate {top_candidate.chunk_id} "
                    f"(score={top_candidate.score:.3f}, "
                    f"threshold={self.config.min_score_threshold:.3f})"
                )
        
        # Select explore candidates
        exploit_bytes = sum(
            all_chunks[c.chunk_id].logical_bytes 
            for c in selected_exploit 
            if c.chunk_id in all_chunks
        )
        remaining_budget = budget_bytes - exploit_bytes
        
        selected_explore = self._select_explore(
            explore_pool, explore_slots, all_chunks, remaining_budget
        )
        
        # Combine selections
        selected_ids = [c.chunk_id for c in selected_exploit] + [c.chunk_id for c in selected_explore]
        exploit_ids = [c.chunk_id for c in selected_exploit]
        explore_ids = [c.chunk_id for c in selected_explore]
        
        decision_log.selected_ids = selected_ids
        
        # Track decisions and dropped
        selected_decisions: List[SchedulerDecision] = []
        dropped_decisions: List[SchedulerDecision] = []
        dropped_ids: List[str] = []
        
        used_bytes = 0
        
        # Process exploit selections
        for candidate in selected_exploit:
            chunk = all_chunks.get(candidate.chunk_id)
            if chunk:
                selected_decisions.append(SchedulerDecision(
                    chunk_id=candidate.chunk_id,
                    decision=PromoteDecision.PROMOTE,
                    score=candidate.score,
                    confidence=candidate.confidence,
                    selection_type="exploit" + ("_fallback" if decision_log.fallback_triggered else ""),
                ))
                used_bytes += chunk.logical_bytes
        
        # Process explore selections
        for candidate in selected_explore:
            chunk = all_chunks.get(candidate.chunk_id)
            if chunk:
                selected_decisions.append(SchedulerDecision(
                    chunk_id=candidate.chunk_id,
                    decision=PromoteDecision.PROMOTE,
                    score=candidate.score,
                    confidence=candidate.confidence,
                    selection_type="explore",
                ))
                used_bytes += chunk.logical_bytes
        
        # Track dropped candidates
        n_dropped_budget = 0
        n_dropped_low_score = 0
        n_dropped_low_confidence = 0
        
        for candidate in candidates:
            if candidate.chunk_id in selected_ids:
                continue
            
            chunk = all_chunks.get(candidate.chunk_id)
            if chunk is None:
                continue
            
            # Determine rejection reason with chunk info for budget check
            reason = self._determine_rejection_reason(candidate, used_bytes, budget_bytes, chunk)
            
            if reason == FailureReason.SCHEDULER_BUDGET_CUT:
                n_dropped_budget += 1
            elif reason == FailureReason.LOW_SCORE:
                n_dropped_low_score += 1
            elif reason == FailureReason.LOW_CONFIDENCE:
                n_dropped_low_confidence += 1
            
            dropped_decisions.append(SchedulerDecision(
                chunk_id=candidate.chunk_id,
                decision=PromoteDecision.SKIP,
                score=candidate.score,
                confidence=candidate.confidence,
                rejection_reason=reason,
            ))
            dropped_ids.append(candidate.chunk_id)
            decision_log.rejected.append((
                candidate.chunk_id, candidate.score, candidate.confidence, reason.value
            ))
        
        # INVARIANT CHECKS
        # Check 1: If gold_rank == 1 and selected == 0 and dropped_budget == 0
        gold_rank = None
        for i, c in enumerate(candidates):
            if c.chunk_id in getattr(scorer_result, 'gold_chunk_ids', set()):
                gold_rank = i + 1
                break
        
        if gold_rank == 1 and len(selected_ids) == 0 and n_dropped_budget == 0:
            violation = "scheduler_veto_on_gold_rank1"
            decision_log.invariant_violations.append(violation)
            logger.error(f"INVARIANT VIOLATION: {violation}")
        
        # Check 2: If candidates > 0 and selected == 0 and utilization == 0 and dropped_budget == 0
        if n_candidates > 0 and len(selected_ids) == 0 and used_bytes == 0 and n_dropped_budget == 0:
            violation = "illegal_zero_selection"
            decision_log.invariant_violations.append(violation)
            logger.error(f"INVARIANT VIOLATION: {violation}")
        
        latency_us = (time.time() - start_time) * 1e6
        
        result = SchedulerResult(
            request_id=query.request_id,
            step=query.step,
            selected_ids=selected_ids,
            selected_decisions=selected_decisions,
            exploit_ids=exploit_ids,
            explore_ids=explore_ids,
            dropped_ids=dropped_ids,
            dropped_decisions=dropped_decisions,
            budget_bytes=budget_bytes,
            used_bytes=used_bytes,
            utilization=used_bytes / max(budget_bytes, 1),
            n_exploit=len(exploit_ids),
            n_explore=len(explore_ids),
            n_dropped_budget=n_dropped_budget,
            n_dropped_low_score=n_dropped_low_score,
            n_dropped_low_confidence=n_dropped_low_confidence,
            scheduler_latency_us=latency_us,
        )
        
        # Attach decision log to result for debugging
        result._decision_log = decision_log
        
        # Print detailed log if invariants violated or fallback triggered
        if decision_log.invariant_violations or decision_log.fallback_triggered:
            decision_log.print_summary()
        
        if self.config.log_exploit_explore_split:
            logger.debug(
                f"EABS step {query.step}: exploit={result.n_exploit}, "
                f"explore={result.n_explore}, budget_util={result.utilization:.2%}, "
                f"selected={len(selected_ids)}"
            )
        
        return result
    
    def _split_candidates(
        self,
        candidates: List[Any],
        exploit_slots: int,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Split candidates into exploit and explore pools.
        
        Exploit pool: top exploit_slots candidates
        Explore pool: remaining candidates
        """
        if exploit_slots >= len(candidates):
            return candidates, []
        
        exploit_candidates = candidates[:exploit_slots]
        explore_pool = candidates[exploit_slots:]
        
        return exploit_candidates, explore_pool
    
    def _select_exploit_advisory(
        self,
        candidates: List[Any],
        all_chunks: Dict[str, ChunkMetadata],
        budget_bytes: int,
        decision_log: SchedulerDecisionLog,
    ) -> Tuple[List[Any], List[Tuple[Any, str]]]:
        """
        Select exploit candidates with advisory thresholds.
        
        PATCHED: Thresholds are advisory - candidates are selected even if below threshold,
        but the threshold violations are logged.
        
        Returns:
            (selected_list, rejected_list_with_reason)
        """
        selected = []
        rejected = []
        used_bytes = 0
        
        for candidate in candidates:
            chunk = all_chunks.get(candidate.chunk_id)
            if chunk is None:
                rejected.append((candidate, "chunk_not_found"))
                continue
            
            # Check budget first (hard constraint)
            if used_bytes + chunk.logical_bytes > budget_bytes:
                rejected.append((candidate, "budget_exceeded"))
                continue
            
            # Apply advisory thresholds - log but don't veto
            below_score_threshold = candidate.score < self.config.min_score_threshold
            below_conf_threshold = (
                self.config.skip_if_low_confidence and 
                candidate.confidence < self.config.confidence_threshold
            )
            
            if below_score_threshold or below_conf_threshold:
                # Log advisory rejection but still select if top candidate
                reason = []
                if below_score_threshold:
                    reason.append(f"score_{candidate.score:.3f}_below_{self.config.min_score_threshold:.3f}")
                if below_conf_threshold:
                    reason.append(f"conf_{candidate.confidence:.3f}_below_{self.config.confidence_threshold:.3f}")
                
                # For advisory mode, we still select but log the threshold violation
                logger.debug(
                    f"Advisory threshold violation for {candidate.chunk_id}: "
                    f"{', '.join(reason)} - selecting anyway"
                )
            
            # Select this candidate
            selected.append(candidate)
            used_bytes += chunk.logical_bytes
        
        return selected, rejected
    
    def _select_explore(
        self,
        pool: List[Any],
        explore_slots: int,
        all_chunks: Dict[str, ChunkMetadata],
        remaining_budget: int,
    ) -> List[Any]:
        """
        Select explore candidates.
        
        Strategy: uncertainty-weighted sampling
        """
        if explore_slots == 0 or not pool:
            return []
        
        # Compute uncertainty for each candidate
        uncertainties = []
        for candidate in pool:
            # Uncertainty = 1 - confidence (higher = more uncertain)
            uncertainty = max(0.01, 1.0 - candidate.confidence)  # Min uncertainty to avoid 0
            uncertainties.append(uncertainty)
        
        # Apply temperature for sampling
        if self.config.exploration_temperature > 0:
            # Convert to probabilities
            total_uncertainty = sum(uncertainties)
            if total_uncertainty < 1e-8:
                # Fallback to uniform if all uncertainties are 0
                probs = np.ones(len(pool)) / len(pool)
            else:
                probs = np.array(uncertainties) / total_uncertainty
            
            # Apply temperature
            if self.config.exploration_temperature != 1.0:
                probs = np.power(probs, 1.0 / self.config.exploration_temperature)
                probs = probs / (probs.sum() + 1e-8)
            
            # Sample without replacement
            n_samples = min(explore_slots, len(pool))
            indices = np.random.choice(
                len(pool), size=n_samples, replace=False, p=probs
            )
            selected = [pool[i] for i in indices]
        else:
            # Deterministic: pick most uncertain
            sorted_indices = np.argsort(uncertainties)[::-1]
            selected = [pool[i] for i in sorted_indices[:explore_slots]]
        
        # Apply budget filter
        filtered = []
        used_bytes = 0
        for candidate in selected:
            chunk = all_chunks.get(candidate.chunk_id)
            if chunk is None:
                continue
            if used_bytes + chunk.logical_bytes <= remaining_budget:
                filtered.append(candidate)
                used_bytes += chunk.logical_bytes
        
        return filtered
    
    def _determine_rejection_reason(
        self,
        candidate: Any,
        used_bytes: int,
        budget_bytes: int,
        chunk: Optional[ChunkMetadata] = None,
    ) -> FailureReason:
        """Determine why a candidate was rejected."""
        # Check if adding this chunk would exceed budget
        if chunk is not None:
            if used_bytes + chunk.logical_bytes > budget_bytes:
                return FailureReason.SCHEDULER_BUDGET_CUT
        elif used_bytes >= budget_bytes:
            return FailureReason.SCHEDULER_BUDGET_CUT
        
        if candidate.score < self.config.min_score_threshold:
            return FailureReason.LOW_SCORE
        
        if self.config.skip_if_low_confidence and candidate.confidence < self.config.confidence_threshold:
            return FailureReason.LOW_CONFIDENCE
        
        return FailureReason.UNKNOWN


class DeterministicScheduler(ExplorationAwareBudgetScheduler):
    """
    Deterministic scheduler (no exploration).
    
    Baseline for ablation studies.
    Equivalent to EABS with exploration_ratio=0.
    """
    
    def __init__(self, config: EABSConfig):
        # Force exploration ratio to 0
        config = EABSConfig(**{**config.__dict__, "exploration_ratio": 0.0})
        super().__init__(config)
