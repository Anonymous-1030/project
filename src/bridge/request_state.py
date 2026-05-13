"""
Per-request state management for the prosex ↔ vLLM bridge.

Manages the lifecycle of ProseRequestState through the generative process:
- Request addition (prefill): build chunks, classify, initialize pipeline
- Decode step: update tokens, run pipeline, produce filtered block list
- Request removal: cleanup state

Thread safety: uses a re-entrant lock for concurrent access from vLLM's
async engine and scheduler threads.
"""

import logging
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from src.core_types import ChunkMetadata, QueryContext
from src.promotion.pipeline import PromotionPipeline, PromotionPipelineResult
from src.bridge.config import BridgeConfig
from src.bridge.chunk_builder import (
    build_chunks_from_request,
    classify_anchor_tail,
)
from src.bridge.block_mapper import BlockChunkMapper
from src.bridge.signature import compute_query_signature

logger = logging.getLogger(__name__)


@dataclass
class ProseRequestState:
    """
    Per-request state for prosex promotion pipeline integration.

    Tracks everything needed to run the promotion pipeline for a single
    vLLM request across its decode lifecycle.
    """

    request_id: str

    # Pipeline instance (one per request for independent sticky/TTL state)
    pipeline: PromotionPipeline

    # Chunk data
    all_chunks: Dict[str, ChunkMetadata] = field(default_factory=dict)
    anchor_chunk_ids: set = field(default_factory=set)
    tail_chunk_ids: set = field(default_factory=set)

    # Block mapping (populated after first block table is known)
    chunk_to_blocks: Dict[str, List[int]] = field(default_factory=dict)
    block_to_chunk: Dict[int, str] = field(default_factory=dict)

    # Dynamic state
    current_step: int = 0
    total_tokens: int = 0
    promoted_chunk_ids: set = field(default_factory=set)
    last_result: Optional[PromotionPipelineResult] = None

    # Token tracking for query signature (sliding window)
    recent_token_ids: List[int] = field(default_factory=list)
    max_recent_tokens: int = 64  # More than query_signature_window for safety

    # Prefill state
    prefill_complete: bool = False
    num_prefill_tokens: int = 0

    # Stats
    total_promotion_calls: int = 0
    total_pipeline_errors: int = 0
    total_blocks_filtered: int = 0
    total_blocks_total: int = 0

    def record_step(self, new_token_id: int):
        """Update state for a new decode step."""
        self.current_step += 1
        self.recent_token_ids.append(new_token_id)
        # Maintain sliding window
        if len(self.recent_token_ids) > self.max_recent_tokens:
            self.recent_token_ids = self.recent_token_ids[-self.max_recent_tokens:]

    def get_recent_tokens(self, window: int) -> List[int]:
        """Get the most recent N token IDs for query signature."""
        return self.recent_token_ids[-window:] if self.recent_token_ids else []


class BridgeStateManager:
    """
    Thread-safe registry of per-request prosex state.

    Manages the lifecycle:
      on_request_added → on_decode_step* → on_request_removed

    Thread safety is critical because vLLM's async engine may call these
    from different threads (scheduler, engine core, API handler).
    """

    def __init__(self, bridge_config: BridgeConfig):
        self.config = bridge_config
        self._states: Dict[str, ProseRequestState] = {}
        self._lock = threading.RLock()
        self._block_mapper = BlockChunkMapper(
            block_size=bridge_config.vllm_block_size,
            chunk_size=bridge_config.chunk_size,
        )
        # Model dimensions (set on first request, validated thereafter)
        self._num_layers: Optional[int] = None
        self._num_kv_heads: Optional[int] = None
        self._head_dim: Optional[int] = None
        self._dtype_size: int = 2  # float16

    # ------------------------------------------------------------------
    # Lifecycle: request added
    # ------------------------------------------------------------------

    def on_request_added(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        dtype_size: int = 2,
    ) -> ProseRequestState:
        """
        Called when a new request enters vLLM's scheduler.

        Builds chunks, computes signatures, classifies anchors/tail,
        and initializes the promotion pipeline.

        Args:
            request_id: vLLM request ID.
            prompt_token_ids: Full prompt token IDs.
            num_layers: Number of attention layers.
            num_kv_heads: KV heads per layer.
            head_dim: Head dimension.
            dtype_size: Bytes per dtype element.

        Returns:
            The initialized ProseRequestState.

        Edge cases:
            - Duplicate request_id → logs warning, returns existing state
            - Empty prompt → creates state with no chunks (degenerate case)
            - Very long prompt → creates many chunks, all handled
        """
        with self._lock:
            # Check for duplicate
            if request_id in self._states:
                logger.warning(
                    f"Request {request_id} already exists in state manager, "
                    f"returning existing state"
                )
                return self._states[request_id]

            # Cache model dimensions
            self._num_layers = num_layers
            self._num_kv_heads = num_kv_heads
            self._head_dim = head_dim
            self._dtype_size = dtype_size

            # Build chunks
            chunks = build_chunks_from_request(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                chunk_size=self.config.chunk_size,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype_size=dtype_size,
            )

            # Classify anchor / tail
            anchor_ids, tail_ids = classify_anchor_tail(
                chunks=chunks,
                anchor_ratio=self.config.anchor_ratio,
            )

            # Create pipeline
            pipeline = PromotionPipeline(self.config.prosex_config)

            state = ProseRequestState(
                request_id=request_id,
                pipeline=pipeline,
                all_chunks=chunks,
                anchor_chunk_ids=set(anchor_ids),
                tail_chunk_ids=set(tail_ids),
                num_prefill_tokens=len(prompt_token_ids),
                total_tokens=len(prompt_token_ids),
            )

            self._states[request_id] = state

            logger.info(
                f"Request {request_id} added: {len(chunks)} chunks "
                f"({len(anchor_ids)} anchors, {len(tail_ids)} tail), "
                f"{len(prompt_token_ids)} tokens"
            )
            return state

    # ------------------------------------------------------------------
    # Lifecycle: prefill complete
    # ------------------------------------------------------------------

    def on_prefill_complete(
        self,
        request_id: str,
        block_table: List[int],
    ):
        """
        Called after prefill completes for a request.

        Builds the block→chunk mapping now that the physical block table
        is known.

        Args:
            request_id: vLLM request ID.
            block_table: Full block table (ordered list of physical block IDs).
        """
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                logger.warning(
                    f"on_prefill_complete: request {request_id} not found"
                )
                return

            state.prefill_complete = True

            # Build block-chunk maps
            chunk_to_blocks, block_to_chunk = self._block_mapper.build_block_chunk_maps(
                block_table=block_table,
                num_tokens=state.num_prefill_tokens,
            )

            # Update chunk IDs to match block mapper's format
            # Block mapper uses "req:..." prefix but chunks use "{request_id}:..."
            # We need to reconcile: re-key chunk_to_blocks to use request_id prefix
            reconciled_ctb: Dict[str, List[int]] = {}
            reconciled_btc: Dict[int, str] = {}

            for chunk_key, blk_list in chunk_to_blocks.items():
                # Convert "req:{start}-{end}" → "{request_id}:{start}-{end}"
                new_key = chunk_key.replace("req:", f"{request_id}:", 1)
                reconciled_ctb[new_key] = blk_list
                for blk_id in blk_list:
                    reconciled_btc[blk_id] = new_key

            state.chunk_to_blocks = reconciled_ctb
            state.block_to_chunk = reconciled_btc

            logger.debug(
                f"Prefill complete for {request_id}: "
                f"{len(reconciled_ctb)} chunks mapped to blocks"
            )

    # ------------------------------------------------------------------
    # Lifecycle: decode step
    # ------------------------------------------------------------------

    def on_decode_step(
        self,
        request_id: str,
        new_token_id: int,
        current_block_table: List[int],
        total_tokens: int,
    ) -> Tuple[List[int], int, bool]:
        """
        Called at each decode step for promotion pipeline execution.

        Args:
            request_id: vLLM request ID.
            new_token_id: The newly generated token ID.
            current_block_table: Full block table for this request (ordered).
            total_tokens: Total tokens in the sequence so far.

        Returns:
            Tuple of (filtered_block_ids, new_seq_len, was_filtered).

            filtered_block_ids: Ordered list of promoted block IDs (compacted).
            new_seq_len: Adjusted sequence length for attention.
            was_filtered: True if promotion was active (vs all blocks passed through).

        Edge cases:
            - State not found → returns full block table unfiltered
            - Pipeline disabled → returns full block table unfiltered
            - Pipeline error → falls back to full if fallback_on_pipeline_error
            - Empty promoted set → returns anchor blocks only
        """
        state = self._states.get(request_id)
        if state is None:
            logger.warning(f"on_decode_step: request {request_id} not found")
            valid_blocks = [b for b in current_block_table if b >= 0]
            return valid_blocks, len(valid_blocks) * self.config.vllm_block_size, False

        # Update token tracking (step counter is handled in run_pipeline_for_decode)
        state.recent_token_ids.append(new_token_id)
        if len(state.recent_token_ids) > state.max_recent_tokens:
            state.recent_token_ids = state.recent_token_ids[-state.max_recent_tokens:]
        state.total_tokens = total_tokens

        # Update block maps (new blocks may have been allocated)
        self._update_block_maps(state, current_block_table, total_tokens)

        if not self.config.enable_promotion:
            valid_blocks = [b for b in current_block_table if b >= 0]
            return valid_blocks, total_tokens, False

        # Run pipeline (delegate to pipeline bridge for full logic)
        from src.bridge.pipeline_bridge import ProsePipelineBridge

        try:
            bridge = ProsePipelineBridge(self.config)
            filtered_blocks, new_seq_len = bridge.run_pipeline_for_decode(
                state=state,
                new_token_id=new_token_id,
                current_block_table=current_block_table,
                total_tokens=total_tokens,
            )

            state.total_promotion_calls += 1
            state.total_blocks_filtered += len(filtered_blocks)
            state.total_blocks_total += len([b for b in current_block_table if b >= 0])

            was_filtered = len(filtered_blocks) < len(
                [b for b in current_block_table if b >= 0]
            )
            return filtered_blocks, new_seq_len, was_filtered

        except Exception as e:
            logger.error(
                f"Pipeline error for request {request_id} step {state.current_step}: {e}",
                exc_info=True,
            )
            state.total_pipeline_errors += 1

            if self.config.fallback_on_pipeline_error:
                logger.warning(
                    f"Falling back to full attention for request {request_id}"
                )
                valid_blocks = [b for b in current_block_table if b >= 0]
                return valid_blocks, total_tokens, False
            else:
                raise

    # ------------------------------------------------------------------
    # Lifecycle: request removed
    # ------------------------------------------------------------------

    def on_request_removed(self, request_id: str):
        """
        Called when a request completes and is removed from vLLM.

        Frees all per-request state and pipeline resources.
        """
        with self._lock:
            if request_id in self._states:
                state = self._states.pop(request_id)
                # Reset pipeline (clears sticky TTL state and chunk registry)
                try:
                    state.pipeline.reset()
                except Exception:
                    pass  # Best-effort cleanup
                logger.debug(
                    f"Request {request_id} removed: "
                    f"{state.total_promotion_calls} promotion calls, "
                    f"{state.total_pipeline_errors} errors"
                )
            else:
                logger.debug(
                    f"on_request_removed: request {request_id} not found "
                    f"(may have been already cleaned up)"
                )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_state(self, request_id: str) -> Optional[ProseRequestState]:
        """Get the state for a request (thread-safe)."""
        with self._lock:
            return self._states.get(request_id)

    def get_promoted_block_ids(self, request_id: str) -> List[int]:
        """
        Get the currently promoted block IDs for a request.

        Returns empty list if request not found or no promotions active.
        """
        state = self.get_state(request_id)
        if state is None:
            return []

        blocks = []
        for chunk_id in state.promoted_chunk_ids:
            if chunk_id in state.chunk_to_blocks:
                blocks.extend(state.chunk_to_blocks[chunk_id])
        return blocks

    def has_state(self, request_id: str) -> bool:
        """Check if a request has active state."""
        with self._lock:
            return request_id in self._states

    def get_stats(self) -> dict:
        """Get aggregate statistics across all requests."""
        with self._lock:
            total_requests = len(self._states)
            total_calls = sum(s.total_promotion_calls for s in self._states.values())
            total_errors = sum(s.total_pipeline_errors for s in self._states.values())
            total_blocks_filtered = sum(
                s.total_blocks_filtered for s in self._states.values()
            )
            total_blocks = sum(s.total_blocks_total for s in self._states.values())
            avg_filter_ratio = (
                total_blocks_filtered / max(total_blocks, 1)
                if total_blocks > 0
                else 0.0
            )

            return {
                "total_requests": total_requests,
                "total_promotion_calls": total_calls,
                "total_pipeline_errors": total_errors,
                "total_blocks_filtered": total_blocks_filtered,
                "total_blocks_total": total_blocks,
                "avg_filter_ratio": avg_filter_ratio,
            }

    def cleanup_orphaned(self, active_request_ids: set):
        """
        Remove states for requests that are no longer active in vLLM.

        Call periodically to prevent memory leaks from requests that
        were removed without on_request_removed being called.

        Args:
            active_request_ids: Set of currently active vLLM request IDs.
        """
        with self._lock:
            orphaned = set(self._states.keys()) - active_request_ids
            for req_id in orphaned:
                self.on_request_removed(req_id)
            if orphaned:
                logger.info(f"Cleaned up {len(orphaned)} orphaned request states")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_block_maps(
        self,
        state: ProseRequestState,
        current_block_table: List[int],
        total_tokens: int,
    ):
        """
        Update block-chunk maps when new blocks are allocated (decode phase).

        During decode, each new token might fill a new block. We need to keep
        the mapping up to date.
        """
        # Rebuild if block table changed significantly
        old_block_count = len(state.block_to_chunk)
        new_valid_blocks = [b for b in current_block_table if b >= 0]
        new_block_count = len(new_valid_blocks)

        if new_block_count > old_block_count:
            ctb, btc = self._block_mapper.build_block_chunk_maps(
                block_table=new_valid_blocks,
                num_tokens=total_tokens,
            )
            # Reconcile chunk key prefixes
            reconciled_ctb: Dict[str, List[int]] = {}
            reconciled_btc: Dict[int, str] = {}
            for chunk_key, blk_list in ctb.items():
                new_key = chunk_key.replace("req:", f"{state.request_id}:", 1)
                reconciled_ctb[new_key] = blk_list
                for blk_id in blk_list:
                    reconciled_btc[blk_id] = new_key

            state.chunk_to_blocks = reconciled_ctb
            state.block_to_chunk = reconciled_btc
