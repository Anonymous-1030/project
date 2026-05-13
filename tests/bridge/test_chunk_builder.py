"""
Tests for chunk builder module (bridge/chunk_builder.py).

Covers:
- Basic chunk construction
- Edge cases: empty, single token, exact division, remainder
- Anchor/tail classification
- Byte estimation
"""

import pytest
from src.bridge.chunk_builder import (
    build_chunks_from_request,
    classify_anchor_tail,
    estimate_chunk_bytes,
)
from src.core_types import ChunkTier


class TestBuildChunksFromRequest:
    """Tests for build_chunks_from_request."""

    def test_empty_prompt(self):
        """Empty prompt should return empty dict."""
        chunks = build_chunks_from_request("req1", [])
        assert chunks == {}

    def test_single_chunk(self):
        """Prompt smaller than chunk_size should produce 1 chunk."""
        tokens = list(range(256))
        chunks = build_chunks_from_request("req1", tokens, chunk_size=512)
        assert len(chunks) == 1
        chunk = list(chunks.values())[0]
        assert chunk.num_tokens == 256
        assert chunk.token_start == 0
        assert chunk.token_end == 256
        assert chunk.request_id == "req1"
        assert chunk.tier == ChunkTier.TAIL  # Not yet classified

    def test_exact_division(self):
        """Prompt exactly divisible by chunk_size."""
        tokens = list(range(1024))
        chunks = build_chunks_from_request("req1", tokens, chunk_size=512)
        assert len(chunks) == 2
        c0 = chunks["req1:0-512"]
        c1 = chunks["req1:512-1024"]
        assert c0.num_tokens == 512
        assert c1.num_tokens == 512

    def test_remainder(self):
        """Prompt not divisible by chunk_size."""
        tokens = list(range(600))
        chunks = build_chunks_from_request("req1", tokens, chunk_size=512)
        assert len(chunks) == 2
        c0 = chunks["req1:0-512"]
        c1 = chunks["req1:512-600"]
        assert c0.num_tokens == 512
        assert c1.num_tokens == 88

    def test_many_chunks(self):
        """Many chunks from a long prompt."""
        tokens = list(range(5000))
        chunks = build_chunks_from_request("req1", tokens, chunk_size=512)
        assert len(chunks) == 10  # ceil(5000/512) = 10

    def test_signatures_are_computed(self):
        """Each chunk should have a non-None signature."""
        tokens = list(range(1024))
        chunks = build_chunks_from_request("req1", tokens, chunk_size=512)
        for chunk in chunks.values():
            assert chunk.signature is not None
            assert len(chunk.signature) == 128

    def test_position_ratio(self):
        """Position ratio should be correct."""
        tokens = list(range(1536))
        chunks = build_chunks_from_request("req1", tokens, chunk_size=512)
        c0 = chunks["req1:0-512"]
        c1 = chunks["req1:512-1024"]
        c2 = chunks["req1:1024-1536"]
        assert c0.position_ratio == 0.0
        assert c1.position_ratio == pytest.approx(512 / 1536)
        assert c2.position_ratio == pytest.approx(1024 / 1536)

    def test_logical_bytes(self):
        """Logical bytes should scale with num_tokens."""
        tokens = list(range(512))
        chunks = build_chunks_from_request(
            "req1", tokens, chunk_size=512,
            num_layers=32, num_kv_heads=8, head_dim=128, dtype_size=2,
        )
        chunk = list(chunks.values())[0]
        expected_bytes = 2 * 32 * 8 * 128 * 512 * 2  # K+V × layers × heads × dim × tokens × fp16
        assert chunk.logical_bytes == expected_bytes

    def test_different_request_ids(self):
        """Chunks from different requests should have different IDs."""
        chunks_a = build_chunks_from_request("reqA", list(range(512)))
        chunks_b = build_chunks_from_request("reqB", list(range(512)))
        ids_a = set(chunks_a.keys())
        ids_b = set(chunks_b.keys())
        assert ids_a.isdisjoint(ids_b)


class TestClassifyAnchorTail:
    """Tests for classify_anchor_tail."""

    def test_empty_chunks(self):
        """Empty chunk dict."""
        anchors, tail = classify_anchor_tail({}, 0.1)
        assert anchors == []
        assert tail == []

    def test_single_chunk(self):
        """Single chunk should be anchor."""
        tokens = list(range(512))
        chunks = build_chunks_from_request("req1", tokens)
        anchors, tail = classify_anchor_tail(chunks, 0.1)
        assert len(anchors) == 1
        assert len(tail) == 0
        assert chunks[anchors[0]].tier == ChunkTier.ANCHOR

    def test_ten_percent_anchor(self):
        """10% anchor ratio on 20 chunks → 2 anchors."""
        tokens = list(range(20 * 512))
        chunks = build_chunks_from_request("req1", tokens)
        anchors, tail = classify_anchor_tail(chunks, 0.1)
        expected_anchors = max(1, int(20 * 0.1))  # 2
        assert len(anchors) == expected_anchors + 1  # +1 for first chunk always anchor
        assert len(tail) == len(chunks) - len(anchors)

    def test_anchor_ratio_zero(self):
        """anchor_ratio=0 should still have at least 1 anchor."""
        tokens = list(range(5 * 512))
        chunks = build_chunks_from_request("req1", tokens)
        anchors, tail = classify_anchor_tail(chunks, 0.0)
        assert len(anchors) >= 1

    def test_anchor_ratio_one(self):
        """anchor_ratio=1 should make all chunks anchors."""
        tokens = list(range(3 * 512))
        chunks = build_chunks_from_request("req1", tokens)
        anchors, tail = classify_anchor_tail(chunks, 1.0)
        assert len(anchors) == len(chunks)
        assert len(tail) == 0

    def test_first_chunk_is_anchor(self):
        """The first chunk should always be in the anchor set."""
        tokens = list(range(10 * 512))
        chunks = build_chunks_from_request("req1", tokens)
        anchors, _ = classify_anchor_tail(chunks, 0.05)
        first_chunk_id = chunks["req1:0-512"].chunk_id
        assert first_chunk_id in anchors

    def test_tier_updated(self):
        """Chunk tiers should be updated after classification."""
        tokens = list(range(5 * 512))
        chunks = build_chunks_from_request("req1", tokens)
        anchors, tail = classify_anchor_tail(chunks, 0.2)
        for cid in anchors:
            assert chunks[cid].tier == ChunkTier.ANCHOR
        for cid in tail:
            assert chunks[cid].tier == ChunkTier.TAIL


class TestEstimateChunkBytes:
    """Tests for byte estimation."""

    def test_basic(self):
        """Basic byte estimation."""
        b = estimate_chunk_bytes(512, num_layers=32, num_kv_heads=8, head_dim=128, dtype_size=2)
        expected = 2 * 32 * 8 * 128 * 512 * 2  # K+V × L × H × D × T × fp16
        assert b == expected

    def test_zero_tokens(self):
        """Zero tokens should give zero bytes."""
        assert estimate_chunk_bytes(0) == 0

    def test_fp8(self):
        """FP8 dtype (1 byte per element)."""
        b = estimate_chunk_bytes(512, dtype_size=1)
        expected = 2 * 32 * 8 * 128 * 512 * 1
        assert b == expected
