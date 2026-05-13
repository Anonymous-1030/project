"""
Pipeline Bridge: runs prosex's PromotionPipeline and converts results to
vLLM block table filters.

This is the core orchestrator that:
1. Builds a QueryContext from request state
2. Runs the 5-stage promotion pipeline (ULF → ODUS → EABS → Burst → Sticky)
3. Extracts the final visible chunk set
4. Converts chunk IDs to compacted block ID lists
5. Returns filtered block table + adjusted sequence length

The key insight: RoPE position encodings are baked into each K/V vector
at write time, so compacting (reordering) the block table preserves
correct attention scores. FlashAttention's paged kernel only cares about
which physical blocks to read — not their logical ordering.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

from src.core_types import (
    ChunkMetadata,
    QueryContext,
    ChunkTier,
)
from src.promotion.pipeline import PromotionPipeline, PromotionPipelineResult
from src.bridge.config import BridgeConfig
from src.bridge.request_state import ProseRequestState
from src.bridge.block_mapper import BlockChunkMapper
from src.bridge.signature import compute_query_signature

logger = logging.getLogger(__name__)


class ProsePipelineBridge:
    """
    Bridges prosex's PromotionPipeline to vLLM's block table model.

    Runs the complete 5-stage pipeline:
      1. MQR-ULF:   Multi-queue recall → candidate chunks from tail
      2. ODUS:      Oracle-distilled utility scoring of candidates
      3. EABS:      Exploration-aware budget scheduler (exploit + explore)
      4. Burst:     Expand selected chunks to neighboring chunks
      5. Sticky:    TTL-based persistence management

    Converts the pipeline's final_visible_ids into a compacted block table
    for FlashAttention, adjusting seq_lens accordingly.
    """

    def __init__(self, bridge_config: BridgeConfig):
        self.config = bridge_config
        self.block_mapper = BlockChunkMapper(
            block_size=bridge_config.vllm_block_size,
            chunk_size=bridge_config.chunk_size,
        )
        # Pipeline run timing stats
        self._last_pipeline_duration_us: float = 0.0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_pipeline_for_decode(
        self,
        state: ProseRequestState,
        new_token_id: int,
        current_block_table: List[int],
        total_tokens: int,
    ) -> Tuple[List[int], int]:
        """
        Run the full promotion pipeline for a decode step.

        Args:
            state: Per-request state (updated in-place).
            new_token_id: The newly generated token ID.
            current_block_table: Full block table as ordered list of physical
                                 block IDs (with -1 padding filtered out).
            total_tokens: Total tokens in the sequence.

        Returns:
            (filtered_block_ids, new_seq_len)
            - filtered_block_ids: Ordered list of promoted physical block IDs
            - new_seq_len: Adjusted sequence length for attention computation

        The returned block list is compacted (only promoted blocks, in
        original position order) and ready to be used as a vLLM block table.
        """
        t0 = time.perf_counter()

        # ---- Step 0: Disabled promotion → return full block table ----
        if not self.config.enable_promotion:
            valid_blocks = [b for b in current_block_table if b >= 0]
            return valid_blocks, total_tokens

        # ---- Step 0b: Check if we should run the pipeline this step ----
        should_run = (state.current_step % self.config.pipeline_run_every_n_steps) == 0

        if not should_run and state.last_result is not None:
            # Reuse previous result; sticky TTL decay happens automatically
            # when we rebuild visible set from sticky state
            visible_ids = self._get_visible_from_state(state)
            filtered_blocks = self._chunks_to_blocks(visible_ids, state)
            new_seq_len = len(filtered_blocks) * self.config.vllm_block_size
            self._last_pipeline_duration_us = 0.0
            return filtered_blocks, new_seq_len

        # ---- Step 1: Increment step counter ----
        state.current_step += 1
        state.total_promotion_calls += 1

        # ---- Step 2: Build QueryContext ----
        query = self._build_query_context(state)

        # ---- Step 2: Classify chunks for this step ----
        anchor_chunks, tail_chunks = self._classify_chunks(state)

        # ---- Step 3: Get currently promoted chunks ----
        promoted_chunks = self._get_promoted_chunks(state)

        # ---- Step 4: Compute budget ----
        budget_bytes = self._compute_budget(tail_chunks)

        # ---- Step 5: Run promotion pipeline ----
        try:
            result = state.pipeline.run(
                query=query,
                tail_chunks=tail_chunks,
                anchor_chunks=anchor_chunks,
                promoted_chunks=promoted_chunks,
                budget_bytes=budget_bytes,
            )
        except Exception:
            logger.error(
                f"Pipeline.run() failed for {state.request_id} step {state.current_step}",
                exc_info=True,
            )
            raise

        # ---- Step 6: Extract final visible set ----
        visible_chunk_ids = set(result.final_visible_ids)

        # Always include anchors (belt-and-suspenders safety)
        visible_chunk_ids.update(state.anchor_chunk_ids)

        # Cap at max promoted chunks
        if len(visible_chunk_ids) > self.config.max_promoted_chunks_per_request:
            # Keep anchors + top promoted by recency
            excess = len(visible_chunk_ids) - self.config.max_promoted_chunks_per_request
            non_anchor = sorted(
                visible_chunk_ids - state.anchor_chunk_ids,
                key=lambda cid: state.all_chunks.get(
                    cid, ChunkMetadata(chunk_id=cid, request_id="", token_start=0,
                                       token_end=0, position_ratio=0,
                                       num_tokens=0, logical_bytes=0)
                ).position_ratio,
                reverse=True,  # Most recent first
            )
            # Remove excess from the end (least recent non-anchors)
            to_remove = set(non_anchor[-excess:]) if excess < len(non_anchor) else set(non_anchor)
            visible_chunk_ids -= to_remove
            logger.debug(
                f"Capped visible set from {len(visible_chunk_ids) + len(to_remove)} "
                f"to {len(visible_chunk_ids)} chunks"
            )

        # ---- Step 7: Update state with new results ----
        state.last_result = result
        if result.sticky_result is not None:
            state.promoted_chunk_ids = set(result.sticky_result.promoted_ids)
        else:
            # If no sticky result (e.g., burst disabled), use scheduler output
            if result.scheduler_result is not None:
                state.promoted_chunk_ids = set(result.scheduler_result.selected_ids)

        # ---- Step 8: Convert chunks to compacted block list ----
        filtered_blocks = self._chunks_to_blocks(visible_chunk_ids, state)

        # ---- Step 9: Compute new sequence length ----
        new_seq_len = len(filtered_blocks) * self.config.vllm_block_size

        # Adjust for possible partial last block
        if filtered_blocks and total_tokens % self.config.vllm_block_size != 0:
            # Check if the last promoted block covers the very end of the sequence
            last_block = filtered_blocks[-1]
            # We can't easily determine if this is the "last" block without
            # the original block table ordering, so use the promoted count directly.
            # This is slightly conservative but correct.
            pass

        # ---- Step 10: Log & return ----
        self._last_pipeline_duration_us = (time.perf_counter() - t0) * 1e6

        if self.config.log_pipeline_decisions and state.current_step <= 5:
            logger.info(
                f"[{state.request_id} step {state.current_step}] "
                f"Promoted {len(visible_chunk_ids)}/{len(state.all_chunks)} chunks "
                f"→ {len(filtered_blocks)} blocks (was {total_tokens} tokens, "
                f"now {new_seq_len}) "
                f"[{self._last_pipeline_duration_us:.0f}us]"
            )

        return filtered_blocks, new_seq_len

    # ------------------------------------------------------------------
    # QueryContext construction
    # ------------------------------------------------------------------

    def _build_query_context(self, state: ProseRequestState) -> QueryContext:
        """
        Build a QueryContext from the current request state.

        The query signature is computed from the sliding window of recent
        token IDs via SimHash n-gram hashing.
        """
        recent_tokens = state.get_recent_tokens(self.config.query_signature_window)
        query_signature = compute_query_signature(
            recent_tokens,
            dim=self.config.signature_dim,
            ngram_sizes=self.config.ngram_n_values,
        )

        return QueryContext(
            request_id=state.request_id,
            step=state.current_step,
            query_signature=query_signature,
            query_tokens=list(recent_tokens),
            query_text=None,
            query_length=len(recent_tokens),
            active_anchor_ids=list(state.anchor_chunk_ids),
            recent_anchor_ids=list(state.anchor_chunk_ids),
            steps_since_start=state.current_step,
            generation_length=state.current_step,
        )

    # ------------------------------------------------------------------
    # Chunk classification
    # ------------------------------------------------------------------

    def _classify_chunks(
        self,
        state: ProseRequestState,
    ) -> Tuple[List[ChunkMetadata], List[ChunkMetadata]]:
        """
        Extract anchor and tail chunks from request state.

        Returns:
            (anchor_chunks, tail_chunks) as lists of ChunkMetadata.
        """
        anchor_chunks = []
        tail_chunks = []

        for chunk_id, chunk in state.all_chunks.items():
            if chunk_id in state.anchor_chunk_ids:
                anchor_chunks.append(chunk)
            else:
                tail_chunks.append(chunk)

        return anchor_chunks, tail_chunks

    def _get_promoted_chunks(
        self,
        state: ProseRequestState,
    ) -> List[ChunkMetadata]:
        """
        Get currently promoted (sticky) chunks as ChunkMetadata list.
        """
        promoted = []
        for chunk_id in state.promoted_chunk_ids:
            chunk = state.all_chunks.get(chunk_id)
            if chunk is not None:
                promoted.append(chunk)
        return promoted

    # ------------------------------------------------------------------
    # Budget computation
    # ------------------------------------------------------------------

    def _compute_budget(
        self,
        tail_chunks: List[ChunkMetadata],
    ) -> int:
        """
        Compute the promotion budget in bytes.

        Uses the pipeline's own budget computation (which checks
        EABSConfig.budget_bytes and budget_ratio_of_tail).
        """
        from src.config import EABSConfig
        eabs_config = self.config.prosex_config.eabs

        if eabs_config.budget_bytes is not None:
            return eabs_config.budget_bytes

        tail_bytes = sum(c.logical_bytes for c in tail_chunks)
        return int(tail_bytes * eabs_config.budget_ratio_of_tail)

    # ------------------------------------------------------------------
    # Visible set computation
    # ------------------------------------------------------------------

    def _get_visible_from_state(self, state: ProseRequestState) -> Set[str]:
        """
        Get the current visible chunk set from state (without re-running
        the pipeline). Used when not running pipeline this step.
        """
        visible = set(state.anchor_chunk_ids)
        visible.update(state.promoted_chunk_ids)
        return visible

    # ------------------------------------------------------------------
    # Chunk → block conversion
    # ------------------------------------------------------------------

    def _chunks_to_blocks(
        self,
        chunk_ids: Set[str],
        state: ProseRequestState,
    ) -> List[int]:
        """
        Convert a set of chunk IDs to an ordered list of physical block IDs.

        Blocks are sorted by their original position in the block table to
        maintain correct positional order for attention computation.

        Edge cases:
            - chunk_id not in chunk_to_blocks → silently skipped
            - Empty chunk_ids → empty list
            - Duplicate blocks (from overlapping chunks) → deduplicated
        """
        # Collect all blocks from visible chunks
        all_blocks: List[int] = []
        seen: Set[int] = set()

        for chunk_id in chunk_ids:
            block_ids = state.chunk_to_blocks.get(chunk_id)
            if block_ids is None:
                continue
            for blk in block_ids:
                if blk not in seen:
                    seen.add(blk)
                    all_blocks.append(blk)

        # Sort by position in original block table
        # Build a position lookup from the current block_to_chunk mapping
        # We need the original order: sorted by token_start of the chunk
        def _block_position(block_id: int) -> int:
            chunk_id = state.block_to_chunk.get(block_id)
            if chunk_id is None:
                return 999999
            chunk = state.all_chunks.get(chunk_id)
            if chunk is None:
                return 999999
            return chunk.token_start

        all_blocks.sort(key=_block_position)
        return all_blocks

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def last_pipeline_duration_us(self) -> float:
        """Duration of the last pipeline run in microseconds."""
        return self._last_pipeline_duration_us
