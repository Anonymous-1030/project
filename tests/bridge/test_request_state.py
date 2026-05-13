"""
Tests for request state manager (bridge/request_state.py).

Covers:
- State lifecycle: add → update → remove
- Thread safety
- Block map updates
- Statistics
"""

import pytest
import threading
from src.bridge.config import BridgeConfig
from src.bridge.request_state import (
    BridgeStateManager,
    ProseRequestState,
)
from src.config import EABSConfig


class TestProseRequestState:
    """Tests for ProseRequestState."""

    def test_initialization(self):
        """Basic initialization."""
        from src.promotion.pipeline import PromotionPipeline
        from src.config import ProSEXv2Config
        pipeline = PromotionPipeline(ProSEXv2Config())
        state = ProseRequestState(
            request_id="test_req",
            pipeline=pipeline,
        )
        assert state.request_id == "test_req"
        assert state.current_step == 0
        assert state.total_tokens == 0
        assert state.prefill_complete is False
        assert state.promoted_chunk_ids == set()

    def test_record_step(self):
        """Step recording should increment counter and track token."""
        pipeline = type("MockPipeline", (), {"reset": lambda self: None, "run": lambda *a, **kw: None})()
        state = ProseRequestState(request_id="test", pipeline=pipeline)
        state.record_step(42)
        assert state.current_step == 1
        assert state.recent_token_ids == [42]

    def test_record_multiple_steps(self):
        """Multiple step recordings."""
        pipeline = type("MockPipeline", (), {"reset": lambda self: None, "run": lambda *a, **kw: None})()
        state = ProseRequestState(request_id="test", pipeline=pipeline)
        for tid in range(10):
            state.record_step(tid)
        assert state.current_step == 10
        assert len(state.recent_token_ids) == 10

    def test_sliding_window(self):
        """Recent tokens should respect max window size."""
        pipeline = type("MockPipeline", (), {"reset": lambda self: None, "run": lambda *a, **kw: None})()
        state = ProseRequestState(request_id="test", pipeline=pipeline, max_recent_tokens=5)
        for tid in range(20):
            state.record_step(tid)
        assert len(state.recent_token_ids) == 5
        assert state.recent_token_ids == [15, 16, 17, 18, 19]

    def test_get_recent_tokens(self):
        """Get recent tokens with custom window."""
        pipeline = type("MockPipeline", (), {"reset": lambda self: None, "run": lambda *a, **kw: None})()
        state = ProseRequestState(request_id="test", pipeline=pipeline)
        for tid in range(100):
            state.record_step(tid)
        recent = state.get_recent_tokens(10)
        assert recent == [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]


class TestBridgeStateManager:
    """Tests for BridgeStateManager."""

    @pytest.fixture
    def manager(self):
        """Create a BridgeStateManager with test config."""
        config = BridgeConfig()
        return BridgeStateManager(config)

    @pytest.fixture
    def manager_with_eabs_config(self):
        """Manager with specific EABS config."""
        config = BridgeConfig()
        return BridgeStateManager(config)

    def test_add_request(self, manager):
        """Adding a request should create state."""
        state = manager.on_request_added("req1", list(range(1024)))
        assert state.request_id == "req1"
        assert len(state.all_chunks) == 2  # 1024 tokens / 512 = 2 chunks
        assert manager.has_state("req1")

    def test_add_duplicate_request(self, manager):
        """Adding the same request twice returns existing state."""
        s1 = manager.on_request_added("req1", list(range(512)))
        s2 = manager.on_request_added("req1", list(range(1024)))
        assert s1 is s2  # Same object returned

    def test_add_empty_prompt(self, manager):
        """Empty prompt should create state with no chunks."""
        state = manager.on_request_added("req1", [])
        assert state.request_id == "req1"
        assert len(state.all_chunks) == 0

    def test_remove_request(self, manager):
        """Removing a request should clean up state."""
        manager.on_request_added("req1", list(range(512)))
        assert manager.has_state("req1")
        manager.on_request_removed("req1")
        assert not manager.has_state("req1")

    def test_remove_nonexistent(self, manager):
        """Removing non-existent request should not raise."""
        manager.on_request_removed("nonexistent")  # No exception

    def test_get_state(self, manager):
        """Get state for a request."""
        manager.on_request_added("req1", list(range(512)))
        state = manager.get_state("req1")
        assert state is not None
        assert state.request_id == "req1"

    def test_get_state_nonexistent(self, manager):
        """Get state for non-existent request."""
        assert manager.get_state("nonexistent") is None

    def test_anchor_tail_classification(self, manager):
        """Request should have both anchors and tail chunks."""
        manager.on_request_added("req1", list(range(5120)))  # 10 chunks
        state = manager.get_state("req1")
        assert len(state.anchor_chunk_ids) > 0
        assert len(state.tail_chunk_ids) > 0
        assert len(state.anchor_chunk_ids) + len(state.tail_chunk_ids) == 10

    def test_prefill_complete(self, manager):
        """Prefill complete should build block maps."""
        manager.on_request_added("req1", list(range(1024)))  # 2 chunks = 64 blocks
        block_table = list(range(64))
        manager.on_prefill_complete("req1", block_table)
        state = manager.get_state("req1")
        assert state.prefill_complete is True
        assert len(state.chunk_to_blocks) == 2
        assert len(state.block_to_chunk) == 64

    def test_on_decode_step_no_state(self, manager):
        """Decode step on unknown request returns full blocks."""
        blocks, seq_len, filtered = manager.on_decode_step(
            "unknown", 42, list(range(64)), 1024
        )
        assert len(blocks) == 64
        assert filtered is False  # No filtering because no state

    def test_on_decode_step_promotion_disabled(self, manager):
        """When promotion is disabled, full blocks returned."""
        manager.config.enable_promotion = False
        manager.on_request_added("req1", list(range(1024)))
        manager.on_prefill_complete("req1", list(range(64)))
        blocks, seq_len, filtered = manager.on_decode_step(
            "req1", 42, list(range(65)), 1025  # +1 block for new token
        )
        assert filtered is False

    def test_stats(self, manager):
        """Stats should reflect current state."""
        manager.on_request_added("req1", list(range(512)))
        manager.on_request_added("req2", list(range(1024)))
        stats = manager.get_stats()
        assert stats["total_requests"] == 2

    def test_cleanup_orphaned(self, manager):
        """Orphaned requests should be cleaned up."""
        manager.on_request_added("req1", list(range(512)))
        manager.on_request_added("req2", list(range(512)))
        assert manager.get_stats()["total_requests"] == 2
        manager.cleanup_orphaned({"req1"})  # req2 is orphaned
        assert manager.get_stats()["total_requests"] == 1
        assert manager.has_state("req1")
        assert not manager.has_state("req2")

    def test_get_promoted_block_ids(self, manager):
        """Get promoted block IDs."""
        manager.on_request_added("req1", list(range(1024)))
        manager.on_prefill_complete("req1", list(range(64)))
        # Initially no promoted blocks
        blocks = manager.get_promoted_block_ids("req1")
        assert blocks == []

    def test_get_promoted_block_ids_unknown(self, manager):
        """Unknown request returns empty."""
        assert manager.get_promoted_block_ids("unknown") == []

    def test_thread_safety_add(self, manager):
        """Concurrent adds from multiple threads."""
        errors = []

        def add_request(idx):
            try:
                manager.on_request_added(f"req_{idx}", list(range(512)))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_request, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert manager.get_stats()["total_requests"] == 50

    def test_thread_safety_add_remove(self, manager):
        """Concurrent adds and removes."""
        errors = []

        def worker(action, idx):
            try:
                if action == "add":
                    manager.on_request_added(f"req_{idx}", list(range(512)))
                elif action == "remove":
                    manager.on_request_removed(f"req_{idx}")
            except Exception as e:
                errors.append((action, idx, e))

        # Add all first
        for i in range(30):
            manager.on_request_added(f"req_{i}", list(range(512)))

        # Then remove/add concurrently
        threads = []
        for i in range(30):
            if i % 2 == 0:
                threads.append(threading.Thread(target=worker, args=("remove", i)))
            else:
                threads.append(threading.Thread(target=worker, args=("add", i + 30)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
