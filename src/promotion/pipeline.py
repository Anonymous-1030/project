"""
Promotion Pipeline for ProSE-X 2.0.

Integrates all promotion components:
1. Multi-Queue Recall ULF
2. Oracle-Distilled Utility Scorer
3. Exploration-Aware Budget Scheduler
4. Burst-and-Stick Expansion
5. Sticky TTL Management

This is the main entry point for promotion decisions.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Set

from src.core_types import (
    ChunkMetadata, QueryContext, PromotionPipelineResult,
    ChunkTier
)
from src.config import ProSEXv2Config
from src.promotion.ulf.mqr_ulf import MultiQueueRecallULF
from src.promotion.scorer.odus import UtilityScorer
from src.promotion.scheduler.eabs import ExplorationAwareBudgetScheduler
from src.promotion.burst.burst_expand import BurstExpander
from src.promotion.sticky.ttl_manager import StickyTTLManager

logger = logging.getLogger(__name__)


class PromotionPipeline:
    """
    Complete promotion pipeline for ProSE-X 2.0.
    
    Executes the full cascade:
    ULF -> Scorer -> Scheduler -> Burst -> Sticky
    """
    
    def __init__(self, config: ProSEXv2Config):
        self.config = config
        
        # Initialize components
        self.ulf = MultiQueueRecallULF(config.mqr_ulf)
        self.scorer = UtilityScorer(config.odus)
        self.scheduler = ExplorationAwareBudgetScheduler(config.eabs)
        self.burst = BurstExpander(config.burst)
        self.sticky = StickyTTLManager(config.burst)
        
        # Chunk storage
        self._chunks: Dict[str, ChunkMetadata] = {}
        
        logger.info("PromotionPipeline initialized")
    
    def run(
        self,
        query: QueryContext,
        tail_chunks: List[ChunkMetadata],
        anchor_chunks: List[ChunkMetadata],
        promoted_chunks: List[ChunkMetadata],
        budget_bytes: Optional[int] = None,
        gold_chunk_ids: Optional[Set[str]] = None,  # PATCH: for Candidate Recall@K
    ) -> PromotionPipelineResult:
        """
        Run the complete promotion pipeline.
        
        Args:
            query: Query context
            tail_chunks: Current tail chunks
            anchor_chunks: Current anchor chunks
            promoted_chunks: Currently promoted chunks
            budget_bytes: Promotion budget (optional, uses config default)
            
        Returns:
            PromotionPipelineResult with all stage outputs
        """
        total_start = time.time()
        
        # Update chunk registry
        all_chunks = {}
        for chunk in tail_chunks + anchor_chunks + promoted_chunks:
            all_chunks[chunk.chunk_id] = chunk
        self._chunks = all_chunks
        
        # Update query with current anchor info
        query.active_anchor_ids = [c.chunk_id for c in anchor_chunks]
        
        # Stage 1: ULF (with gold chunk tracking for Candidate Recall@K)
        # PATCH: Pass all_chunks to include anchors for anchor_neighbor queue
        ulf_result = self.ulf.filter(query, tail_chunks, gold_chunk_ids=gold_chunk_ids, all_chunks=all_chunks)
        
        # Stage 2: Scorer
        scorer_result = self.scorer.score(ulf_result, query, all_chunks)
        
        # Stage 3: Scheduler
        if budget_bytes is None:
            budget_bytes = self._compute_budget(tail_chunks)
        
        scheduler_result = self.scheduler.schedule(
            scorer_result, query, all_chunks, budget_bytes
        )
        
        # Stage 4: Burst expansion
        burst_result = self.burst.expand(
            scheduler_result.selected_ids,
            all_chunks,
            query.step,
        )
        
        # Stage 5: Sticky TTL management
        # Track accessed chunks (promoted in previous steps)
        accessed_ids = [c.chunk_id for c in promoted_chunks]
        
        sticky_result = self.sticky.update(
            burst_result.burst_ids,
            accessed_ids,
            all_chunks,
            query.step,
        )
        sticky_result.request_id = query.request_id
        
        # Compute final visible set
        final_visible = self._compute_final_visible(
            anchor_chunks, promoted_chunks, sticky_result.promoted_ids
        )
        
        # Compute bytes
        final_active_bytes = sum(
            all_chunks[cid].logical_bytes
            for cid in final_visible if cid in all_chunks
        )
        final_promoted_bytes = sum(
            all_chunks[cid].logical_bytes
            for cid in sticky_result.promoted_ids if cid in all_chunks
        )
        
        total_latency = (time.time() - total_start) * 1e6
        
        return PromotionPipelineResult(
            request_id=query.request_id,
            step=query.step,
            ulf_result=ulf_result,
            scorer_result=scorer_result,
            scheduler_result=scheduler_result,
            burst_result=burst_result,
            sticky_result=sticky_result,
            final_visible_ids=final_visible,
            final_active_bytes=final_active_bytes,
            final_promoted_bytes=final_promoted_bytes,
            total_latency_us=total_latency,
        )
    
    def _compute_budget(
        self,
        tail_chunks: List[ChunkMetadata],
    ) -> int:
        """Compute promotion budget."""
        config = self.config.eabs
        
        if config.budget_bytes is not None:
            return config.budget_bytes
        
        tail_bytes = sum(c.logical_bytes for c in tail_chunks)
        return int(tail_bytes * config.budget_ratio_of_tail)
    
    def _compute_final_visible(
        self,
        anchor_chunks: List[ChunkMetadata],
        promoted_chunks: List[ChunkMetadata],
        new_promoted_ids: List[str],
    ) -> List[str]:
        """
        Compute final set of visible (full-fidelity) chunks.
        
        Visible set = anchors + previously promoted (not expired) + newly promoted
        """
        visible = set()
        
        # Anchors
        for chunk in anchor_chunks:
            visible.add(chunk.chunk_id)
        
        # Previously promoted
        for chunk in promoted_chunks:
            visible.add(chunk.chunk_id)
        
        # Newly promoted
        visible.update(new_promoted_ids)
        
        return list(visible)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all components."""
        return {
            "sticky": self.sticky.get_stats(),
        }
    
    def reset(self) -> None:
        """Reset pipeline state."""
        self.sticky.clear()
        self._chunks.clear()


class AblationPipeline:
    """
    Pipeline with ablation support.
    
    Allows swapping individual components for ablation studies.
    """
    
    def __init__(self, config: ProSEXv2Config):
        self.config = config
        self.pipeline = PromotionPipeline(config)
    
    def with_ulf(self, ulf) -> "AblationPipeline":
        """Swap ULF component."""
        self.pipeline.ulf = ulf
        return self
    
    def with_scorer(self, scorer) -> "AblationPipeline":
        """Swap scorer component."""
        self.pipeline.scorer = scorer
        return self
    
    def with_scheduler(self, scheduler) -> "AblationPipeline":
        """Swap scheduler component."""
        self.pipeline.scheduler = scheduler
        return self
    
    def with_burst(self, burst) -> "AblationPipeline":
        """Swap burst component."""
        self.pipeline.burst = burst
        return self
    
    def with_sticky(self, sticky) -> "AblationPipeline":
        """Swap sticky component."""
        self.pipeline.sticky = sticky
        return self
    
    def run(self, *args, **kwargs) -> PromotionPipelineResult:
        """Run the pipeline."""
        return self.pipeline.run(*args, **kwargs)
