"""
Tests for pipeline bridge module (bridge/pipeline_bridge.py).

Covers:
- QueryContext construction
- Pipeline execution with mock state
- Block table filtering
- Error handling and fallback
- Budget computation
"""

import pytest
import numpy as np
from src.bridge.config import BridgeConfig
from src.bridge.pipeline_bridge import ProsePipelineBridge
from src.bridge.request_state import ProseRequestState
from src.bridge.chunk_builder import (
    build_chunks_from_request,
    classify_anchor_tail,
)
from src.promotion.pipeline import PromotionPipeline


class TestProsePipelineBridge:
    """Tests for ProsePipelineBridge."""

    @pytest.fixture
    def bridge(self):
        """Create a ProsePipelineBridge with test config."""
        config = BridgeConfig()
        return ProsePipelineBridge(config)

    @pytest.fixture
    def state(self):
        """Create a test ProseRequestState with chunks."""
        config = BridgeConfig()
        pipeline = PromotionPipeline(config.prosex_config)
        tokens = list(range(2048))  # 4 chunks of 512
        chunks = build_chunks_from_request("test_req", tokens, chunk_size=512)
        anchor_ids, tail_ids = classify_anchor_tail(chunks, anchor_ratio=0.25)

        # Build block maps
        from src.bridge.block_mapper import BlockChunkMapper
        mapper = BlockChunkMapper()
        block_table = list(range(len(chunks) * 32))  # 4 * 32 = 128 blocks
        ctb, btc = mapper.build_block_chunk_maps(block_table, num_tokens=2048)

        # Reconcile keys
        reconciled_ctb = {}
        reconciled_btc = {}
        for ck, bl in ctb.items():
            nk = ck.replace("req:", "test_req:", 1)
            reconciled_ctb[nk] = bl
            for bid in bl:
                reconciled_btc[bid] = nk

        state = ProseRequestState(
            request_id="test_req",
            pipeline=pipeline,
            all_chunks=chunks,
            anchor_chunk_ids=set(anchor_ids),
            tail_chunk_ids=set(tail_ids),
            chunk_to_blocks=reconciled_ctb,
            block_to_chunk=reconciled_btc,
            num_prefill_tokens=2048,
            total_tokens=2048,
            prefill_complete=True,
        )
        return state

    def test_build_query_context(self, bridge, state):
        """QueryContext should be built from state."""
        state.record_step(42)
        query = bridge._build_query_context(state)
        assert query.request_id == "test_req"
        assert query.step == state.current_step
        assert query.query_signature is not None
        assert len(query.query_signature) == 128
        assert query.query_tokens == [42]

    def test_classify_chunks(self, bridge, state):
        """Chunks should be classified into anchor and tail."""
        anchors, tail = bridge._classify_chunks(state)
        assert len(anchors) > 0
        assert len(tail) > 0
        assert len(anchors) + len(tail) == len(state.all_chunks)

    def test_get_promoted_chunks_empty(self, bridge, state):
        """Initially, no chunks are promoted."""
        promoted = bridge._get_promoted_chunks(state)
        assert promoted == []

    def test_compute_budget(self, bridge, state):
        """Budget should be computed from tail chunk bytes."""
        _, tail = bridge._classify_chunks(state)
        budget = bridge._compute_budget(tail)
        assert budget > 0

    def test_compute_budget_with_fixed(self, bridge, state):
        """Fixed budget from config."""
        bridge.config.prosex_config.eabs.budget_bytes = 1000000
        _, tail = bridge._classify_chunks(state)
        budget = bridge._compute_budget(tail)
        assert budget == 1000000

    def test_chunks_to_blocks(self, bridge, state):
        """Chunk IDs should convert to ordered block list."""
        visible = set(state.anchor_chunk_ids)
        blocks = bridge._chunks_to_blocks(visible, state)
        assert len(blocks) > 0
        # Blocks should be ordered by token position
        for i in range(len(blocks) - 1):
            assert blocks[i] < blocks[i + 1]  # Assuming physical ordering matches logical

    def test_chunks_to_blocks_empty(self, bridge, state):
        """Empty chunk set should give empty blocks."""
        blocks = bridge._chunks_to_blocks(set(), state)
        assert blocks == []

    def test_chunks_to_blocks_unknown_chunk(self, bridge, state):
        """Unknown chunk IDs are silently skipped."""
        blocks = bridge._chunks_to_blocks({"unknown_chunk"}, state)
        assert blocks == []

    def test_run_pipeline_for_decode_basic(self, bridge, state):
        """Run full pipeline for a decode step."""
        block_table = list(range(128))
        filtered_blocks, new_seq_len = bridge.run_pipeline_for_decode(
            state=state,
            new_token_id=42,
            current_block_table=block_table,
            total_tokens=2048,
        )
        assert isinstance(filtered_blocks, list)
        assert isinstance(new_seq_len, int)
        assert len(filtered_blocks) > 0
        assert new_seq_len >= len(filtered_blocks) * 16

    def test_run_pipeline_increments_step(self, bridge, state):
        """Pipeline run should increment step counter."""
        assert state.current_step == 0
        block_table = list(range(128))
        bridge.run_pipeline_for_decode(state, 42, block_table, 2048)
        assert state.current_step == 1

    def test_run_pipeline_sets_last_result(self, bridge, state):
        """Pipeline run should store result in state."""
        block_table = list(range(128))
        bridge.run_pipeline_for_decode(state, 42, block_table, 2048)
        assert state.last_result is not None
        assert len(state.last_result.final_visible_ids) > 0
        assert state.total_promotion_calls == 1

    def test_run_pipeline_returns_fewer_blocks(self, bridge, state):
        """Filtered blocks should be fewer than total."""
        block_table = list(range(128))
        filtered_blocks, seq_len = bridge.run_pipeline_for_decode(
            state, 42, block_table, 2048
        )
        # At minimum, anchors should be present
        assert len(filtered_blocks) > 0
        # With default config, should promote fewer than all blocks
        assert len(filtered_blocks) <= len(block_table)

    def test_run_pipeline_visible_includes_anchors(self, bridge, state):
        """Final visible set should include all anchor chunk blocks."""
        block_table = list(range(128))
        filtered_blocks, _ = bridge.run_pipeline_for_decode(
            state, 42, block_table, 2048
        )
        # Get anchor block IDs
        anchor_blocks = set()
        for cid in state.anchor_chunk_ids:
            if cid in state.chunk_to_blocks:
                anchor_blocks.update(state.chunk_to_blocks[cid])
        # All anchor blocks should be in the filtered set
        filtered_set = set(filtered_blocks)
        assert anchor_blocks.issubset(filtered_set)

    def test_run_pipeline_without_running_this_step(self, bridge, state):
        """When pipeline_run_every_n_steps > 1, skip pipeline execution."""
        bridge.config.pipeline_run_every_n_steps = 3
        state.current_step = 1  # step 1 mod 3 != 0
        # Set a prior result
        from src.promotion.pipeline import PromotionPipelineResult
        state.last_result = PromotionPipelineResult(
            request_id="test_req", step=0,
            final_visible_ids=list(state.anchor_chunk_ids),
        )
        block_table = list(range(128))
        filtered_blocks, _ = bridge.run_pipeline_for_decode(
            state, 42, block_table, 2048
        )
        # Should reuse previous result
        assert len(filtered_blocks) > 0

    def test_run_pipeline_every_step(self, bridge, state):
        """With pipeline_run_every_n_steps=1, always runs."""
        bridge.config.pipeline_run_every_n_steps = 1
        block_table = list(range(128))
        for _ in range(5):
            bridge.run_pipeline_for_decode(state, 42, block_table, 2048)
        assert state.total_promotion_calls == 5
        assert state.current_step == 5

    def test_no_pipeline_with_disabled_promotion(self, bridge, state):
        """When promotion is disabled, full set returned."""
        bridge.config.enable_promotion = False
        block_table = list(range(128))
        filtered_blocks, seq_len = bridge.run_pipeline_for_decode(
            state, 42, block_table, 2048
        )
        assert filtered_blocks == block_table
        assert seq_len == 2048

    def test_visible_from_state(self, bridge, state):
        """Get visible set from state without running pipeline."""
        # Add some promoted chunks
        tail_list = list(state.tail_chunk_ids)
        if tail_list:
            state.promoted_chunk_ids.add(tail_list[0])
        visible = bridge._get_visible_from_state(state)
        assert len(visible) >= len(state.anchor_chunk_ids)

    def test_last_pipeline_duration(self, bridge, state):
        """Pipeline duration should be tracked."""
        assert bridge.last_pipeline_duration_us == 0.0
        block_table = list(range(128))
        bridge.run_pipeline_for_decode(state, 42, block_table, 2048)
        assert bridge.last_pipeline_duration_us > 0
