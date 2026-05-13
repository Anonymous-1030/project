"""
Bandwidth-Aware EABS (Explore-Exploit Budget Scheduler).

Phase 3.4: Hardware-aware scheduler that accounts for promotion bandwidth.

Unlike the basic EABS that only considers HBM capacity, this version:
1. Models PCIe/CXL promotion bandwidth as a constraint
2. Schedules promotions to overlap with compute
3. Adapts budget based on measured promotion latency
4. Uses prefetch engine for proactive promotion

Key insight: The effective promotion budget is not just HBM capacity,
but min(HBM capacity, bandwidth * available_time).
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from src.core_types import (
    SchedulerDecision, SchedulerResult, ScoredCandidate,
    PromoteDecision, FailureReason, ChunkMetadata
)
from src.memory.two_tier_manager import (
    TwoTierMemoryManager, PromotionPrefetchEngine, BandwidthSpec
)

logger = logging.getLogger(__name__)


class PromotionConstraint(str, Enum):
    """What constrains promotion at this step."""
    HBM_CAPACITY = "hbm_capacity"      # HBM is full
    BANDWIDTH = "bandwidth"            # Promotion bandwidth limited
    COMPUTE_OVERLAP = "compute_overlap"  # Can overlap with compute
    NONE = "none"                      # No constraint


@dataclass
class BandwidthAwareBudget:
    """Budget computed with bandwidth awareness."""
    # Base budget from HBM capacity
    hbm_budget_bytes: int = 0
    
    # Bandwidth-limited budget
    bandwidth_budget_bytes: int = 0
    
    # Effective budget (min of above)
    effective_budget_bytes: int = 0
    
    # Time available for promotion this step
    available_time_us: float = 0.0
    
    # Constraint type
    constraint: PromotionConstraint = PromotionConstraint.NONE
    
    # Bandwidth spec used
    bandwidth_spec: Optional[BandwidthSpec] = None


@dataclass
class PromotionPlan:
    """Planned promotion with timing estimates."""
    chunk_id: str
    bytes: int
    estimated_time_us: float
    can_overlap: bool
    priority: float


class BandwidthAwareEABS:
    """
    Explore-Exploit Budget Scheduler with bandwidth awareness.
    
    This is the Phase 3.4 upgrade that integrates with TwoTierMemoryManager
to model real hardware constraints (PCIe/CXL bandwidth).
    
    The scheduler:
    1. Computes HBM capacity budget (how much fits)
    2. Computes bandwidth budget (how much can be promoted in time)
    3. Uses the more constraining of the two
    4. Plans promotions to overlap with compute
    5. Prefetches predicted future needs
    """
    
    def __init__(
        self,
        memory_manager: TwoTierMemoryManager,
        base_explore_ratio: float = 0.2,
        score_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        enable_prefetch: bool = True,
        enable_bandwidth_modeling: bool = True,
    ):
        """
        Initialize bandwidth-aware EABS.
        
        Args:
            memory_manager: TwoTierMemoryManager for HBM/DRAM management
            base_explore_ratio: Base ratio for exploration (0-1)
            score_threshold: Minimum score for promotion
            confidence_threshold: Minimum confidence for promotion
            enable_prefetch: Whether to use prefetch engine
            enable_bandwidth_modeling: Whether to model bandwidth constraints
        """
        self.memory_manager = memory_manager
        self.base_explore_ratio = base_explore_ratio
        self.score_threshold = score_threshold
        self.confidence_threshold = confidence_threshold
        self.enable_bandwidth_modeling = enable_bandwidth_modeling
        
        # Prefetch engine
        self.prefetch_engine = None
        if enable_prefetch:
            self.prefetch_engine = PromotionPrefetchEngine(memory_manager)
        
        # Statistics
        self.stats = {
            "steps": 0,
            "total_budget_bytes": 0,
            "used_budget_bytes": 0,
            "bandwidth_limited_steps": 0,
            "hbm_limited_steps": 0,
            "overlapped_promotions": 0,
            "prefetch_hits": 0,
        }
        
        # Recent attention patterns for prediction
        self.recent_attention: List[Dict[int, float]] = []
        self.max_history = 4
        
        logger.info(
            f"BandwidthAwareEABS initialized:\n"
            f"  explore_ratio={base_explore_ratio}\n"
            f"  bandwidth_modeling={enable_bandwidth_modeling}\n"
            f"  prefetch={enable_prefetch}"
        )
    
    def compute_bandwidth_aware_budget(
        self,
        current_step: int,
        estimated_compute_time_us: float = 100.0,
    ) -> BandwidthAwareBudget:
        """
        Compute effective budget considering both HBM capacity and bandwidth.
        
        Args:
            current_step: Current generation step
            estimated_compute_time_us: Estimated compute time for overlap
            
        Returns:
            BandwidthAwareBudget
        """
        # HBM capacity budget
        hbm_used = self.memory_manager.stats["hbm_bytes_used"]
        hbm_total = self.memory_manager.hbm_capacity_bytes
        hbm_budget = max(0, hbm_total - hbm_used)
        
        if not self.enable_bandwidth_modeling:
            return BandwidthAwareBudget(
                hbm_budget_bytes=hbm_budget,
                bandwidth_budget_bytes=float('inf'),
                effective_budget_bytes=hbm_budget,
                available_time_us=estimated_compute_time_us,
                constraint=PromotionConstraint.HBM_CAPACITY if hbm_budget == 0 else PromotionConstraint.NONE,
            )
        
        # Bandwidth budget: how much can we promote in available time?
        # Assume PCIe for now (could be CXL)
        pcie_bw = self.memory_manager.bandwidth.pcie_bw_gbps  # GB/s
        available_time_s = estimated_compute_time_us / 1e6
        
        # Effective bandwidth considering overhead
        effective_bw = pcie_bw * 0.8  # 80% efficiency
        bandwidth_budget = int(effective_bw * available_time_s * 1024 ** 3)
        
        # Determine constraint
        if hbm_budget == 0:
            constraint = PromotionConstraint.HBM_CAPACITY
        elif bandwidth_budget < hbm_budget * 0.5:
            constraint = PromotionConstraint.BANDWIDTH
        else:
            constraint = PromotionConstraint.COMPUTE_OVERLAP
        
        # Effective budget is the more constraining
        effective_budget = min(hbm_budget, bandwidth_budget)
        
        return BandwidthAwareBudget(
            hbm_budget_bytes=hbm_budget,
            bandwidth_budget_bytes=bandwidth_budget,
            effective_budget_bytes=effective_budget,
            available_time_us=estimated_compute_time_us,
            constraint=constraint,
        )
    
    def schedule(
        self,
        request_id: str,
        step: int,
        candidates: List[ScoredCandidate],
        all_chunks: Dict[str, ChunkMetadata],
        current_budget_bytes: Optional[int] = None,
        estimated_compute_us: float = 100.0,
    ) -> SchedulerResult:
        """
        Schedule promotions with bandwidth awareness.
        
        Args:
            request_id: Request ID
            step: Current step
            candidates: Scored candidates from ODUS
            all_chunks: All chunk metadata
            current_budget_bytes: Optional override for budget
            estimated_compute_us: Estimated compute time for overlap planning
            
        Returns:
            SchedulerResult with promotion decisions
        """
        start_time = time.time()
        
        # Compute bandwidth-aware budget
        budget_info = self.compute_bandwidth_aware_budget(step, estimated_compute_us)
        
        if current_budget_bytes is not None:
            budget_bytes = current_budget_bytes
        else:
            budget_bytes = budget_info.effective_budget_bytes
        
        # Filter candidates above threshold
        qualified = [
            c for c in candidates
            if c.score >= self.score_threshold 
            and c.confidence >= self.confidence_threshold
        ]
        
        # Split into exploit vs explore
        n_exploit = int(len(qualified) * (1 - self.base_explore_ratio))
        exploit_candidates = qualified[:n_exploit]
        explore_candidates = qualified[n_exploit:]
        
        # Create promotion plans with size estimates
        def estimate_chunk_bytes(chunk_id: str) -> int:
            chunk = all_chunks.get(chunk_id)
            if chunk is None:
                return 0
            # Rough estimate: bytes per token * num tokens
            return chunk.num_tokens * self.memory_manager.bytes_per_token
        
        plans: List[PromotionPlan] = []
        for c in exploit_candidates:
            bytes_needed = estimate_chunk_bytes(c.chunk_id)
            # Estimate promotion time
            promo_time = self.memory_manager.bandwidth.promotion_time_us(
                bytes_needed, via="pcie"
            )
            can_overlap = promo_time < estimated_compute_us
            
            plans.append(PromotionPlan(
                chunk_id=c.chunk_id,
                bytes=bytes_needed,
                estimated_time_us=promo_time,
                can_overlap=can_overlap,
                priority=c.score * c.confidence,
            ))
        
        # Sort by priority
        plans.sort(key=lambda p: p.priority, reverse=True)
        
        # Greedy selection respecting budget
        selected: List[SchedulerDecision] = []
        dropped: List[SchedulerDecision] = []
        used_bytes = 0
        
        for plan in plans:
            if used_bytes + plan.bytes <= budget_bytes:
                # Can promote
                selected.append(SchedulerDecision(
                    chunk_id=plan.chunk_id,
                    decision=PromoteDecision.PROMOTE,
                    score=plan.priority,
                    confidence=1.0,  # From ODUS
                    selection_type="exploit",
                ))
                used_bytes += plan.bytes
            else:
                # Budget exceeded
                reason = FailureReason.SCHEDULER_BUDGET_CUT
                if budget_info.constraint == PromotionConstraint.BANDWIDTH:
                    reason = FailureReason.SCHEDULER_BUDGET_CUT  # Bandwidth limited
                
                dropped.append(SchedulerDecision(
                    chunk_id=plan.chunk_id,
                    decision=PromoteDecision.DEFER,
                    score=plan.priority,
                    confidence=1.0,
                    rejection_reason=reason,
                ))
        
        # Handle explore candidates (random subset)
        import random
        n_explore = min(len(explore_candidates), max(1, len(explore_candidates) // 3))
        explore_selected = random.sample(explore_candidates, n_explore) if explore_candidates else []
        
        for c in explore_selected:
            bytes_needed = estimate_chunk_bytes(c.chunk_id)
            if used_bytes + bytes_needed <= budget_bytes:
                selected.append(SchedulerDecision(
                    chunk_id=c.chunk_id,
                    decision=PromoteDecision.PROMOTE,
                    score=c.score,
                    confidence=c.confidence,
                    selection_type="explore",
                ))
                used_bytes += bytes_needed
        
        # Update statistics
        self.stats["steps"] += 1
        self.stats["total_budget_bytes"] += budget_bytes
        self.stats["used_budget_bytes"] += used_bytes
        
        if budget_info.constraint == PromotionConstraint.BANDWIDTH:
            self.stats["bandwidth_limited_steps"] += 1
        elif budget_info.constraint == PromotionConstraint.HBM_CAPACITY:
            self.stats["hbm_limited_steps"] += 1
        
        # Count overlapped promotions
        n_overlapped = sum(1 for s in selected if s.selection_type == "exploit")
        self.stats["overlapped_promotions"] += n_overlapped
        
        # Execute promotions (this would be async in real impl)
        selected_ids = [s.chunk_id for s in selected if s.decision == PromoteDecision.PROMOTE]
        
        if selected_ids:
            # Convert string chunk_ids to int (if they're integers as strings)
            int_chunk_ids = []
            for cid in selected_ids:
                try:
                    int_chunk_ids.append(int(cid))
                except ValueError:
                    continue
            
            if int_chunk_ids:
                promo_result = self.memory_manager.promote_chunks(
                    chunk_ids=int_chunk_ids,
                    async_prefetch=True,
                    estimated_compute_us=estimated_compute_us,
                )
                
                # Update prefetch stats
                if promo_result.overlapped:
                    self.stats["overlapped_promotions"] += len(promo_result.promoted_chunks)
        
        # Trigger prefetch for predicted future needs
        if self.prefetch_engine and self.recent_attention:
            predicted = self.prefetch_engine.predict_future_needs(
                step, self.recent_attention
            )
            if predicted:
                self.prefetch_engine.schedule_prefetch(predicted, estimated_compute_us)
        
        # Update attention history
        attention_pattern = {
            int(c.chunk_id) if c.chunk_id.isdigit() else hash(c.chunk_id): c.score
            for c in candidates[:10]
        }
        self.recent_attention.append(attention_pattern)
        if len(self.recent_attention) > self.max_history:
            self.recent_attention.pop(0)
        
        latency_us = (time.time() - start_time) * 1e6
        
        # Build result
        exploit_ids = [s.chunk_id for s in selected if s.selection_type == "exploit"]
        explore_ids = [s.chunk_id for s in selected if s.selection_type == "explore"]
        
        return SchedulerResult(
            request_id=request_id,
            step=step,
            selected_ids=[s.chunk_id for s in selected],
            selected_decisions=selected,
            exploit_ids=exploit_ids,
            explore_ids=explore_ids,
            dropped_ids=[d.chunk_id for d in dropped],
            dropped_decisions=dropped,
            budget_bytes=budget_bytes,
            used_bytes=used_bytes,
            utilization=used_bytes / max(budget_bytes, 1),
            n_exploit=len(exploit_ids),
            n_explore=len(explore_ids),
            n_dropped_budget=len(dropped),
            n_dropped_low_score=0,
            n_dropped_low_confidence=0,
            scheduler_latency_us=latency_us,
        )
    
    def get_analytics(self) -> Dict:
        """Get scheduler analytics."""
        total_steps = max(self.stats["steps"], 1)
        
        return {
            "steps": self.stats["steps"],
            "avg_budget_utilization": self.stats["used_budget_bytes"] / max(self.stats["total_budget_bytes"], 1),
            "bandwidth_limited_fraction": self.stats["bandwidth_limited_steps"] / total_steps,
            "hbm_limited_fraction": self.stats["hbm_limited_steps"] / total_steps,
            "overlapped_promotions": self.stats["overlapped_promotions"],
            "memory_breakdown": self.memory_manager.get_memory_breakdown(),
        }
    
    def compute_optimal_explore_ratio(
        self,
        measured_bandwidth_utilization: float,
    ) -> float:
        """
        Adapt explore ratio based on measured bandwidth.
        
        If bandwidth is underutilized, increase exploration.
        If bandwidth is saturated, decrease exploration.
        
        Args:
            measured_bandwidth_utilization: Measured PCIe/CXL utilization
            
        Returns:
            New explore ratio
        """
        if measured_bandwidth_utilization < 0.5:
            # Bandwidth underutilized, can explore more
            return min(0.4, self.base_explore_ratio * 1.2)
        elif measured_bandwidth_utilization > 0.9:
            # Bandwidth saturated, reduce exploration
            return max(0.05, self.base_explore_ratio * 0.8)
        else:
            return self.base_explore_ratio


class AdaptiveEABS(BandwidthAwareEABS):
    """
    EABS with online adaptation of explore ratio.
    
    Uses multi-armed bandit formulation to adapt exploration
    based on measured promotion success rates.
    """
    
    def __init__(
        self,
        memory_manager: TwoTierMemoryManager,
        **kwargs
    ):
        super().__init__(memory_manager, **kwargs)
        
        # UCB statistics for each "arm" (chunk category)
        self.arm_rewards: Dict[str, List[float]] = {
            "anchor_neighbor": [],
            "high_similarity": [],
            "exploration": [],
        }
        self.arm_counts: Dict[str, int] = {k: 0 for k in self.arm_rewards}
        
        self.ucb_c = 1.0  # Exploration constant
    
    def ucb_score(self, arm: str, total_steps: int) -> float:
        """Compute UCB score for an arm."""
        if self.arm_counts[arm] == 0:
            return float('inf')
        
        avg_reward = np.mean(self.arm_rewards[arm]) if self.arm_rewards[arm] else 0
        exploration_bonus = self.ucb_c * np.sqrt(np.log(total_steps) / self.arm_counts[arm])
        
        return avg_reward + exploration_bonus
    
    def update_reward(self, arm: str, reward: float):
        """Update reward for an arm."""
        self.arm_rewards[arm].append(reward)
        self.arm_counts[arm] += 1
        
        # Keep history bounded
        if len(self.arm_rewards[arm]) > 100:
            self.arm_rewards[arm] = self.arm_rewards[arm][-100:]
