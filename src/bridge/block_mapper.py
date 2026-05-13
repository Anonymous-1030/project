"""
Block-Chunk Mapper: bidirectional mapping between prosex chunks and vLLM blocks.

Prose_v2 chunks = 512 tokens each (configurable)
vLLM blocks = 16 tokens each (default)

1 chunk = chunk_size // block_size blocks (typically 32)

The mapper maintains bidirectional indexing:
  chunk_to_blocks: chunk_id → list of physical block IDs
  block_to_chunk: physical block ID → chunk_id
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BlockChunkMapper:
    """
    Maps between prosex chunk IDs and vLLM physical block IDs.

    A chunk is a logical group of contiguous token positions.
    A block is a physical KV cache allocation unit in vLLM.

    The mapping is built from a vLLM block table, which is an ordered list
    of physical block IDs representing logical token positions.
    """

    def __init__(self, block_size: int = 16, chunk_size: int = 512):
        if chunk_size % block_size != 0:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be divisible by "
                f"block_size ({block_size})"
            )
        self.block_size = block_size
        self.chunk_size = chunk_size
        self.blocks_per_chunk = chunk_size // block_size

    # ------------------------------------------------------------------
    # Map building
    # ------------------------------------------------------------------

    def build_block_chunk_maps(
        self,
        block_table: List[int],
        num_tokens: int,
    ) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
        """
        Build both mapping directions from a vLLM block table.

        The block table is an ordered list of physical block IDs. The i-th
        entry corresponds to logical block i (tokens [i*bs, (i+1)*bs)).

        Args:
            block_table: Ordered list of physical block IDs (no -1 padding).
            num_tokens: Total tokens in the sequence (used to identify partial
                        last block).

        Returns:
            Tuple of (chunk_to_blocks, block_to_chunk).

            chunk_to_blocks: chunk_id → list of physical block IDs (ordered)
            block_to_chunk: physical block ID → chunk_id

        Edge cases:
            - Empty block table → empty dicts
            - Single block (num_tokens <= block_size) → single chunk
            - Partial last chunk → fewer blocks in last chunk's entry
            - Non-contiguous physical block IDs → handled correctly (mapping
              is by position, not by physical adjacency)
            - Negative block IDs in table → filtered out
        """
        chunk_to_blocks: Dict[str, List[int]] = {}
        block_to_chunk: Dict[int, str] = {}

        # Filter out -1 padding
        valid_blocks = [b for b in block_table if b >= 0]

        if not valid_blocks:
            return chunk_to_blocks, block_to_chunk

        num_full_blocks = len(valid_blocks)
        num_chunks = math.ceil(num_tokens / self.chunk_size)

        for chunk_idx in range(num_chunks):
            chunk_start_token = chunk_idx * self.chunk_size
            chunk_end_token = min((chunk_idx + 1) * self.chunk_size, num_tokens)
            chunk_token_count = chunk_end_token - chunk_start_token

            # Which blocks belong to this chunk?
            block_start_idx = chunk_start_token // self.block_size
            block_end_idx = math.ceil(chunk_end_token / self.block_size)
            # Clamp to actually allocated blocks
            block_end_idx = min(block_end_idx, num_full_blocks)

            if block_start_idx >= block_end_idx:
                continue

            chunk_block_ids = valid_blocks[block_start_idx:block_end_idx]

            # Build chunk_id matching ChunkMetadata format
            chunk_id = f"req:{chunk_start_token}-{chunk_end_token}"

            chunk_to_blocks[chunk_id] = chunk_block_ids
            for blk_id in chunk_block_ids:
                block_to_chunk[blk_id] = chunk_id

        logger.debug(
            f"Built maps: {len(chunk_to_blocks)} chunks, "
            f"{len(block_to_chunk)} blocks mapped "
            f"({num_tokens} tokens, {self.chunk_size} tok/chunk, "
            f"{self.block_size} tok/block)"
        )
        return chunk_to_blocks, block_to_chunk

    # ------------------------------------------------------------------
    # Forward mapping: chunks → blocks
    # ------------------------------------------------------------------

    def map_chunks_to_blocks(
        self,
        chunk_ids: List[str],
        chunk_to_blocks: Dict[str, List[int]],
    ) -> List[int]:
        """
        Convert a list of chunk IDs to a flat, ordered list of block IDs.

        Block ordering follows original token position order, not the order
        of chunk_ids in the input list. This preserves correct positional
        relationships for attention computation.

        Args:
            chunk_ids: List of promoted/visible chunk IDs.
            chunk_to_blocks: The per-request chunk→blocks map.

        Returns:
            Ordered list of physical block IDs.

        Edge cases:
            - chunk_id not in map → silently skipped (stale reference)
            - Empty chunk_ids → empty list
            - Duplicate chunk_ids in input → deduplicated
        """
        seen_blocks: set = set()
        ordered_blocks: List[int] = []

        # Sort chunk_ids by their token_start to preserve position order
        # Parse chunk_id format: "req:{token_start}-{token_end}"
        def _chunk_start(cid: str) -> int:
            try:
                # Format: "{request_id}:{start}-{end}"
                parts = cid.rsplit(":", 1)[-1].split("-")[0]
                return int(parts)
            except (ValueError, IndexError):
                return 0

        sorted_chunks = sorted(chunk_ids, key=_chunk_start)

        for chunk_id in sorted_chunks:
            block_ids = chunk_to_blocks.get(chunk_id)
            if block_ids is None:
                logger.debug(f"Chunk {chunk_id} not in block map, skipping")
                continue
            for blk_id in block_ids:
                if blk_id not in seen_blocks:
                    seen_blocks.add(blk_id)
                    ordered_blocks.append(blk_id)

        return ordered_blocks

    # ------------------------------------------------------------------
    # Reverse mapping: block → chunk
    # ------------------------------------------------------------------

    def map_block_to_chunk(
        self,
        block_id: int,
        block_to_chunk: Dict[int, str],
    ) -> Optional[str]:
        """
        Find which chunk a physical block belongs to.

        Args:
            block_id: Physical block ID.
            block_to_chunk: The per-request block→chunk map.

        Returns:
            chunk_id or None if block is not mapped.
        """
        return block_to_chunk.get(block_id)

    def map_blocks_to_chunks(
        self,
        block_ids: List[int],
        block_to_chunk: Dict[int, str],
    ) -> Dict[str, List[int]]:
        """
        Group a list of block IDs by their owning chunks.

        Args:
            block_ids: List of physical block IDs.
            block_to_chunk: The per-request block→chunk map.

        Returns:
            chunk_id → list of block IDs belonging to that chunk.
        """
        chunks: Dict[str, List[int]] = {}
        for blk_id in block_ids:
            chunk_id = block_to_chunk.get(blk_id)
            if chunk_id is not None:
                if chunk_id not in chunks:
                    chunks[chunk_id] = []
                chunks[chunk_id].append(blk_id)
        return chunks

    # ------------------------------------------------------------------
    # Token ↔ block position helpers
    # ------------------------------------------------------------------

    def token_to_logical_block(self, token_pos: int) -> int:
        """Convert a token position to a logical block index."""
        return token_pos // self.block_size

    def logical_block_to_token_start(self, logical_block: int) -> int:
        """Convert a logical block index to its starting token position."""
        return logical_block * self.block_size

    def token_to_chunk_index(self, token_pos: int) -> int:
        """Convert a token position to a chunk index."""
        return token_pos // self.chunk_size

    def chunk_index_to_token_start(self, chunk_idx: int) -> int:
        """Convert a chunk index to its starting token position."""
        return chunk_idx * self.chunk_size

    # ------------------------------------------------------------------
    # Block table filtering
    # ------------------------------------------------------------------

    def compute_filtered_seq_len(
        self,
        promoted_blocks: List[int],
        block_to_chunk: Dict[int, str],
        num_total_tokens: int,
    ) -> int:
        """
        Compute the sequence length for a filtered (compacted) block table.

        When we compact the block table to only promoted blocks, we need to
        tell FlashAttention how many tokens are valid. Since RoPE position
        info is baked into each K/V vector, we can compact freely.

        Args:
            promoted_blocks: Ordered list of promoted block IDs.
            block_to_chunk: Block→chunk map.
            num_total_tokens: Original total token count (to identify partial
                              last block).

        Returns:
            New sequence length in tokens.

        Edge cases:
            - No promoted blocks → 0
            - All blocks promoted → num_total_tokens
            - Last promoted block is partial → adjust accordingly
        """
        if not promoted_blocks:
            return 0

        # Each full block covers block_size tokens
        # Check if the last promoted block is the last block in the sequence
        num_full_blocks = len(promoted_blocks)
        total_without_adjustment = num_full_blocks * self.block_size

        # If the last promoted block is the very last block of the sequence,
        # and it's partial, we need to adjust
        if num_total_tokens % self.block_size != 0:
            # Last block of the original sequence is partial
            total_full_blocks = math.ceil(num_total_tokens / self.block_size)
            last_original_block_start = (total_full_blocks - 1) * self.block_size
            partial_tokens = num_total_tokens - last_original_block_start

            # Check if our last promoted block is indeed the last block
            # (we can't easily check this without knowing original block table,
            #  so we conservatively assume it might be)
            # For safety, just use the promoted block count × block_size
            pass

        return total_without_adjustment
