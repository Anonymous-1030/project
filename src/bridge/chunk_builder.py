"""
Chunk builder: converts vLLM request token sequences into prosex ChunkMetadata.

Splits a prompt's token IDs into fixed-size chunks and constructs
ChunkMetadata objects with signatures, positions, and tier information.
"""

import logging
import math
from typing import Dict, List, Tuple, Optional

from src.core_types import ChunkMetadata, ChunkTier, ChunkState
from src.bridge.signature import compute_chunk_signature

logger = logging.getLogger(__name__)


def build_chunks_from_request(
    request_id: str,
    prompt_token_ids: List[int],
    chunk_size: int = 512,
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype_size: int = 2,  # float16 = 2 bytes
) -> Dict[str, ChunkMetadata]:
    """
    Build ChunkMetadata objects from a request's prompt token IDs.

    Splits the prompt into fixed-size chunks and populates all runtime-available
    fields. Signatures are computed via SimHash n-gram hashing.

    Args:
        request_id: vLLM request identifier.
        prompt_token_ids: Full prompt token IDs.
        chunk_size: Tokens per chunk (default 512).
        num_layers: Number of transformer layers (for byte estimation).
        num_kv_heads: Number of KV heads (for byte estimation).
        head_dim: Head dimension (for byte estimation).
        dtype_size: Bytes per dtype element (2 for float16, 1 for fp8).

    Returns:
        Dict mapping chunk_id → ChunkMetadata.

    Edge cases:
        - Empty prompt → empty dict
        - Prompt < chunk_size → single chunk covering all tokens
        - Prompt exactly divisible by chunk_size → all full chunks
        - Very long prompts → many chunks, all handled

    Chunk ID format: "{request_id}:{token_start}-{token_end}"
    """
    if not prompt_token_ids:
        logger.debug(f"Empty prompt for request {request_id}, returning no chunks")
        return {}

    total_tokens = len(prompt_token_ids)
    num_chunks = math.ceil(total_tokens / chunk_size)

    # Estimate bytes per token in KV cache:
    # 2 (K+V) × num_layers × num_kv_heads × head_dim × dtype_size
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_size

    chunks: Dict[str, ChunkMetadata] = {}

    for i in range(num_chunks):
        token_start = i * chunk_size
        token_end = min((i + 1) * chunk_size, total_tokens)
        num_tokens = token_end - token_start

        chunk_id = f"{request_id}:{token_start}-{token_end}"
        chunk_token_ids = prompt_token_ids[token_start:token_end]

        # Compute signature from token IDs
        signature = compute_chunk_signature(chunk_token_ids)

        # Structural markers — set based on position heuristics
        is_section_boundary = (i > 0) and _is_likely_section_boundary(
            prompt_token_ids, token_start
        )
        is_title_adjacent = _is_near_start(i, num_chunks)

        # Start as TAIL — classification into anchor/tail happens later
        chunk = ChunkMetadata(
            chunk_id=chunk_id,
            request_id=request_id,
            token_start=token_start,
            token_end=token_end,
            position_ratio=token_start / max(total_tokens, 1),
            num_tokens=num_tokens,
            logical_bytes=num_tokens * bytes_per_token,
            signature=signature,
            is_section_boundary=is_section_boundary,
            is_title_adjacent=is_title_adjacent,
            is_code_block=False,
            tier=ChunkTier.TAIL,
            state=ChunkState.ACTIVE,
            creation_step=0,
        )
        chunks[chunk_id] = chunk

    logger.debug(
        f"Built {len(chunks)} chunks for request {request_id} "
        f"({total_tokens} tokens, {chunk_size} tokens/chunk)"
    )
    return chunks


def classify_anchor_tail(
    chunks: Dict[str, ChunkMetadata],
    anchor_ratio: float = 0.1,
) -> Tuple[List[str], List[str]]:
    """
    Classify chunks into anchor (always visible) and tail (promotion candidates).

    Strategy: Most recent chunks become anchors. The rationale is that in
    autoregressive generation, tokens closest to the current generation
    position tend to be most relevant (recency bias).

    Additionally, the very first chunk is always an anchor (beginning of
    context often contains important instructions/system prompts).

    Args:
        chunks: All chunks for a request.
        anchor_ratio: Fraction of chunks to mark as anchors (0.0 to 1.0).

    Returns:
        Tuple of (anchor_chunk_ids, tail_chunk_ids).

    Edge cases:
        - 1 chunk total → it becomes the sole anchor, tail is empty
        - 0 chunks → empty anchors and tail
        - anchor_ratio=0 → first chunk is still anchor (minimum safety)
        - anchor_ratio=1 → all chunks are anchors, tail is empty
    """
    if not chunks:
        return [], []

    sorted_chunks = sorted(chunks.values(), key=lambda c: c.token_start)
    num_chunks = len(sorted_chunks)
    num_anchors = max(1, int(num_chunks * anchor_ratio))

    # Ensure at least the first chunk is an anchor (safety)
    anchor_ids: set = {sorted_chunks[0].chunk_id}

    # Add the most recent chunks as anchors
    for chunk in sorted_chunks[-num_anchors:]:
        anchor_ids.add(chunk.chunk_id)

    tail_ids = [c.chunk_id for c in sorted_chunks if c.chunk_id not in anchor_ids]

    # Update tier metadata
    for chunk_id in anchor_ids:
        chunks[chunk_id].tier = ChunkTier.ANCHOR
    for chunk_id in tail_ids:
        chunks[chunk_id].tier = ChunkTier.TAIL

    logger.debug(
        f"Classified {len(anchor_ids)} anchors, {len(tail_ids)} tail "
        f"out of {num_chunks} total chunks (ratio={anchor_ratio})"
    )
    return list(anchor_ids), tail_ids


def _is_likely_section_boundary(
    token_ids: List[int],
    position: int,
    lookback: int = 16,
) -> bool:
    """
    Heuristic for detecting section boundaries based on token patterns.

    This is a placeholder heuristic. In practice, section boundaries would
    be detected from document structure (markdown headers, double newlines,
    etc.) which requires text access.

    Currently returns False. Future enhancement: detect from tokenizer
    newline patterns or document metadata.
    """
    # Placeholder: can be enhanced with tokenizer-specific logic
    # e.g., detecting double-newline token patterns for Llama tokenizer
    return False


def _is_near_start(chunk_index: int, total_chunks: int, threshold: int = 2) -> bool:
    """Check if a chunk is near the beginning of the document."""
    return chunk_index < threshold


def estimate_chunk_bytes(
    num_tokens: int,
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype_size: int = 2,
) -> int:
    """
    Estimate the byte size of a chunk's KV cache entries.

    Formula: 2 (K+V) × layers × kv_heads × head_dim × tokens × dtype_size

    Args:
        num_tokens: Number of tokens in the chunk.
        num_layers: Number of attention layers.
        num_kv_heads: Number of KV heads (GQA: may differ from Q heads).
        head_dim: Dimension of each head.
        dtype_size: Bytes per element (2=FP16, 1=FP8).

    Returns:
        Estimated bytes.
    """
    return 2 * num_layers * num_kv_heads * head_dim * num_tokens * dtype_size
