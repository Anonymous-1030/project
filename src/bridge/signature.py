"""
SimHash-based chunk and query signature computation.

Since FlashAttention does not produce attention weights, we use n-gram
hashing to create lightweight 128-dim signatures for lexical overlap
computation in the ULF stage.

Algorithm (SimHash):
1. Extract token IDs from the sequence
2. Generate n-grams (n=1,2,3) from the token IDs
3. Hash each n-gram to positions in a 128-dim vector via a fixed hash
4. Accumulate +1/-1 weighted contributions (SimHash technique)
5. Take sign of accumulated vector → binary sketch
6. Return as float32 numpy array for prosex compatibility

The resulting signatures can be compared via cosine similarity (dot product
after normalization) to approximate lexical overlap between chunks.
"""

import hashlib
import struct
import logging
import numpy as np
from typing import List, Tuple, Sequence, Optional

logger = logging.getLogger(__name__)

# Fixed seed material for deterministic hashing across runs
_HASH_SEED = b"prosex_bridge_signature_v1"

# Cache for the random projection vectors used in n-gram hashing
_PROJECTION_CACHE: dict = {}  # (n, dim) -> np.ndarray of shape [dim]


def _get_projection_vectors(n: int, dim: int) -> np.ndarray:
    """
    Get or create deterministic random projection vectors for n-gram hashing.

    Returns a (dim,) shaped array of {-1, +1} values derived from a
    deterministic hash, ensuring reproducibility across runs.
    """
    cache_key = (n, dim)
    if cache_key in _PROJECTION_CACHE:
        return _PROJECTION_CACHE[cache_key]

    # Generate deterministic {-1, +1} values from SHA-256
    vectors = np.zeros(dim, dtype=np.float32)
    seed = _HASH_SEED + struct.pack("<i", n)

    for i in range(dim):
        h = hashlib.sha256(seed + struct.pack("<i", i)).digest()
        # Use first byte: even → +1, odd → -1
        vectors[i] = 1.0 if (h[0] & 1) == 0 else -1.0

    _PROJECTION_CACHE[cache_key] = vectors
    return vectors


def _token_ngrams(
    token_ids: Sequence[int],
    n: int,
) -> List[Tuple[int, ...]]:
    """
    Extract n-grams from a sequence of token IDs.

    Args:
        token_ids: Sequence of token IDs.
        n: N-gram size (1, 2, 3, ...).

    Returns:
        List of n-gram tuples. For n=1, each tuple has one element.

    Edge cases:
        - Empty sequence → empty list
        - Sequence shorter than n → empty list
        - Single-token sequence with n=1 → [(token,)]
    """
    if len(token_ids) < n:
        return []
    return [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]


def _hash_ngram(ngram: Tuple[int, ...]) -> int:
    """
    Deterministically hash an n-gram tuple to a 64-bit integer.

    Uses SHA-256 truncated to 64 bits for collision resistance.
    The seed ensures reproducibility across Python processes.
    """
    data = _HASH_SEED + struct.pack(f"<{len(ngram)}q", *ngram)
    digest = hashlib.sha256(data).digest()
    return struct.unpack("<q", digest[:8])[0]


def _hash_to_index(hash_val: int, dim: int) -> int:
    """
    Map a hash value to an index in [0, dim-1].

    Uses modulo with a final mix step to reduce bias from the modulo operation
    when dim is not a power of 2.
    """
    # Use upper bits (better distribution than lower bits for many hash functions)
    mixed = (hash_val >> 32) ^ (hash_val & 0xFFFFFFFF)
    return abs(mixed) % dim


def compute_chunk_signature(
    token_ids: List[int],
    dim: int = 128,
    ngram_sizes: Tuple[int, ...] = (1, 2, 3),
) -> np.ndarray:
    """
    Compute a SimHash signature for a chunk of tokens.

    Args:
        token_ids: Token IDs in the chunk.
        dim: Signature dimension (default 128).
        ngram_sizes: N-gram sizes to use.

    Returns:
        np.ndarray of shape (dim,) with float32 values in range [-1, 1].

    Edge cases:
        - Empty token list → zero vector (all zeros)
        - Single token → only 1-grams apply
        - Very long chunks → uses all tokens (caller should bound if needed)
        - Duplicate n-grams → they contribute multiple hash votes (by design)

    The signature is normalized to unit length for cosine similarity comparison.
    If all votes cancel out (zero norm), returns a zero vector.
    """
    if not token_ids:
        logger.debug("Empty token list, returning zero signature")
        return np.zeros(dim, dtype=np.float32)

    # Accumulate SimHash votes
    accumulator = np.zeros(dim, dtype=np.float32)

    for n in ngram_sizes:
        if n > len(token_ids):
            continue

        proj = _get_projection_vectors(n, dim)
        ngrams = _token_ngrams(token_ids, n)

        for ngram in ngrams:
            h = _hash_ngram(ngram)
            idx = _hash_to_index(h, dim)
            # SimHash: each n-gram votes +1 or -1 at its projected position
            accumulator += proj * (1.0 if (h & 1) else -1.0)

    # Normalize to unit length
    norm = np.linalg.norm(accumulator)
    if norm > 1e-10:
        accumulator /= norm

    return accumulator.astype(np.float32)


def compute_query_signature(
    recent_token_ids: List[int],
    dim: int = 128,
    ngram_sizes: Tuple[int, ...] = (1, 2, 3),
) -> np.ndarray:
    """
    Compute a signature for the query context (recent decode tokens).

    This uses the same SimHash algorithm as compute_chunk_signature but on
    the sliding window of recent token IDs.

    Args:
        recent_token_ids: Recent token IDs (sliding window).
        dim: Signature dimension.
        ngram_sizes: N-gram sizes.

    Returns:
        np.ndarray of shape (dim,) float32.
    """
    return compute_chunk_signature(recent_token_ids, dim=dim, ngram_sizes=ngram_sizes)


def signature_similarity(
    sig1: np.ndarray,
    sig2: np.ndarray,
) -> float:
    """
    Compute cosine similarity between two signatures.

    Args:
        sig1: First signature vector.
        sig2: Second signature vector.

    Returns:
        Cosine similarity in range [-1, 1].

    Edge cases:
        - Zero vectors → 0.0 (no similarity)
        - Identical vectors → 1.0
        - Opposite vectors → -1.0
    """
    norm1 = np.linalg.norm(sig1)
    norm2 = np.linalg.norm(sig2)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    return float(np.dot(sig1, sig2))


def signature_hamming_similarity(
    sig1: np.ndarray,
    sig2: np.ndarray,
) -> float:
    """
    Compute approximate Jaccard similarity via Hamming distance on sign bits.

    For SimHash, the Hamming distance between sign bits approximates the
    angular distance, which approximates Jaccard similarity.

    Args:
        sig1: First signature vector.
        sig2: Second signature vector.

    Returns:
        Estimated Jaccard similarity in range [0, 1].
    """
    bits1 = (sig1 >= 0).astype(np.int32)
    bits2 = (sig2 >= 0).astype(np.int32)
    matches = np.sum(bits1 == bits2)
    return float(matches) / len(sig1)
