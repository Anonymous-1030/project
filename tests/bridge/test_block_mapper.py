"""
Tests for block mapper module (bridge/block_mapper.py).

Covers:
- Building block-chunk maps
- Forward mapping (chunks → blocks)
- Reverse mapping (blocks → chunks)
- Block table filtering
- Edge cases: empty, partial last chunk, non-contiguous blocks
"""

import pytest
from src.bridge.block_mapper import BlockChunkMapper


class TestBlockChunkMapperInit:
    """Tests for initialization."""

    def test_default_sizes(self):
        """Default block_size=16, chunk_size=512."""
        m = BlockChunkMapper()
        assert m.block_size == 16
        assert m.chunk_size == 512
        assert m.blocks_per_chunk == 32

    def test_custom_sizes(self):
        """Custom sizes."""
        m = BlockChunkMapper(block_size=32, chunk_size=1024)
        assert m.blocks_per_chunk == 32

    def test_invalid_sizes(self):
        """chunk_size must be divisible by block_size."""
        with pytest.raises(ValueError):
            BlockChunkMapper(block_size=16, chunk_size=513)


class TestBuildBlockChunkMaps:
    """Tests for build_block_chunk_maps."""

    def test_empty_block_table(self):
        """Empty block table."""
        m = BlockChunkMapper()
        ctb, btc = m.build_block_chunk_maps([], 0)
        assert ctb == {}
        assert btc == {}

    def test_single_block(self):
        """Single block (< 512 tokens)."""
        m = BlockChunkMapper()
        block_table = [5]  # single physical block
        ctb, btc = m.build_block_chunk_maps(block_table, num_tokens=16)
        assert len(ctb) == 1
        assert len(btc) == 1
        assert btc[5] is not None

    def test_exact_chunk(self):
        """Exactly 32 blocks = 1 chunk."""
        m = BlockChunkMapper()
        block_table = list(range(32))
        ctb, btc = m.build_block_chunk_maps(block_table, num_tokens=512)
        assert len(ctb) == 1  # 1 chunk
        assert len(btc) == 32  # 32 blocks
        # All blocks should map to the same chunk
        chunk_ids = set(btc.values())
        assert len(chunk_ids) == 1

    def test_multiple_chunks(self):
        """2 chunks (64 blocks)."""
        m = BlockChunkMapper()
        block_table = list(range(64))
        ctb, btc = m.build_block_chunk_maps(block_table, num_tokens=1024)
        assert len(ctb) == 2
        assert len(btc) == 64
        # First 32 blocks → chunk 0, next 32 → chunk 1
        chunk_ids = set(btc.values())
        assert len(chunk_ids) == 2

    def test_partial_last_chunk(self):
        """Last chunk has fewer than chunk_size tokens."""
        m = BlockChunkMapper()
        # 48 blocks = 768 tokens = 1 full chunk (512) + 1 partial (256)
        block_table = list(range(48))
        ctb, btc = m.build_block_chunk_maps(block_table, num_tokens=768)
        assert len(ctb) == 2
        # First chunk: 32 blocks, second chunk: 16 blocks
        for chunk_id, blocks in ctb.items():
            if "0-512" in chunk_id:
                assert len(blocks) == 32
            elif "512-768" in chunk_id:
                assert len(blocks) == 16

    def test_negative_blocks_filtered(self):
        """Block IDs with -1 should be filtered out."""
        m = BlockChunkMapper()
        block_table = [0, 1, 2, -1, -1]
        ctb, btc = m.build_block_chunk_maps(block_table, num_tokens=48)
        assert len(btc) == 3  # Only 3 valid blocks


class TestMapChunksToBlocks:
    """Tests for map_chunks_to_blocks (forward mapping)."""

    def test_basic(self):
        """Convert chunk IDs to block IDs."""
        m = BlockChunkMapper()
        ctb = {
            "req:0-512": [0, 1, 2, 3],
            "req:512-1024": [4, 5, 6, 7],
        }
        blocks = m.map_chunks_to_blocks(["req:0-512"], ctb)
        assert blocks == [0, 1, 2, 3]

    def test_multiple_chunks(self):
        """Multiple chunks, blocks ordered by position."""
        m = BlockChunkMapper()
        ctb = {
            "req:0-512": [10, 11],
            "req:512-1024": [20, 21],
        }
        blocks = m.map_chunks_to_blocks(["req:512-1024", "req:0-512"], ctb)
        # Should be sorted by token_start, so chunk 0-512 first
        assert blocks == [10, 11, 20, 21]

    def test_missing_chunk(self):
        """Chunk ID not in map is skipped."""
        m = BlockChunkMapper()
        ctb = {"req:0-512": [0, 1, 2]}
        blocks = m.map_chunks_to_blocks(["req:0-512", "req:999-999"], ctb)
        assert blocks == [0, 1, 2]

    def test_empty_chunks(self):
        """Empty chunk list."""
        m = BlockChunkMapper()
        blocks = m.map_chunks_to_blocks([], {"req:0-512": [0, 1]})
        assert blocks == []

    def test_duplicate_blocks_deduplicated(self):
        """Overlapping chunks with shared blocks are deduplicated."""
        m = BlockChunkMapper()
        ctb = {
            "req:0-512": [0, 1, 2],
            "req:256-768": [1, 2, 3],  # Overlap with first
        }
        blocks = m.map_chunks_to_blocks(["req:0-512", "req:256-768"], ctb)
        assert blocks == [0, 1, 2, 3]  # No duplicates, position-ordered


class TestReverseMapping:
    """Tests for block → chunk mapping."""

    def test_basic(self):
        """Find chunk for a block."""
        m = BlockChunkMapper()
        btc = {5: "req:0-512", 6: "req:512-1024"}
        assert m.map_block_to_chunk(5, btc) == "req:0-512"
        assert m.map_block_to_chunk(6, btc) == "req:512-1024"

    def test_not_found(self):
        """Block not in map."""
        m = BlockChunkMapper()
        assert m.map_block_to_chunk(999, {}) is None

    def test_map_blocks_to_chunks(self):
        """Group blocks by chunk."""
        m = BlockChunkMapper()
        btc = {0: "req:0-512", 1: "req:0-512", 2: "req:512-1024"}
        result = m.map_blocks_to_chunks([0, 1, 2], btc)
        assert result == {
            "req:0-512": [0, 1],
            "req:512-1024": [2],
        }

    def test_unmapped_blocks(self):
        """Blocks without chunk mapping are omitted."""
        m = BlockChunkMapper()
        btc = {0: "req:0-512"}
        result = m.map_blocks_to_chunks([0, 999], btc)
        assert result == {"req:0-512": [0]}


class TestTokenPositionHelpers:
    """Tests for token/block position helpers."""

    def test_token_to_logical_block(self):
        m = BlockChunkMapper()
        assert m.token_to_logical_block(0) == 0
        assert m.token_to_logical_block(15) == 0
        assert m.token_to_logical_block(16) == 1
        assert m.token_to_logical_block(100) == 6

    def test_logical_block_to_token_start(self):
        m = BlockChunkMapper()
        assert m.logical_block_to_token_start(0) == 0
        assert m.logical_block_to_token_start(1) == 16
        assert m.logical_block_to_token_start(10) == 160

    def test_token_to_chunk_index(self):
        m = BlockChunkMapper()
        assert m.token_to_chunk_index(0) == 0
        assert m.token_to_chunk_index(511) == 0
        assert m.token_to_chunk_index(512) == 1
        assert m.token_to_chunk_index(1024) == 2


class TestComputeFilteredSeqLen:
    """Tests for sequence length computation on filtered block tables."""

    def test_no_blocks(self):
        m = BlockChunkMapper()
        assert m.compute_filtered_seq_len([], {}, 1024) == 0

    def test_full_blocks(self):
        """All blocks promoted."""
        m = BlockChunkMapper()
        blocks = list(range(64))  # 64 blocks = 1024 tokens
        seq_len = m.compute_filtered_seq_len(blocks, {}, 1024)
        assert seq_len == 64 * 16  # 1024

    def test_partial_blocks(self):
        """Only some blocks promoted."""
        m = BlockChunkMapper()
        blocks = list(range(32))  # 32 blocks = 512 tokens
        seq_len = m.compute_filtered_seq_len(blocks, {}, 1024)
        assert seq_len == 512
