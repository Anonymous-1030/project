"""
Tests for signature computation module (bridge/signature.py).

Covers:
- Basic signature computation
- Empty and edge case inputs
- Determinism (same input → same output)
- Similarity between related sequences
- Query signature computation
"""

import pytest
import numpy as np
from src.bridge.signature import (
    compute_chunk_signature,
    compute_query_signature,
    signature_similarity,
    signature_hamming_similarity,
)


class TestComputeChunkSignature:
    """Tests for compute_chunk_signature."""

    def test_basic_signature(self):
        """A normal chunk should produce a 128-dim float32 vector."""
        tokens = list(range(512))  # 512 token IDs
        sig = compute_chunk_signature(tokens)
        assert isinstance(sig, np.ndarray)
        assert sig.dtype == np.float32
        assert sig.shape == (128,)

    def test_custom_dim(self):
        """Custom dimension should work."""
        tokens = list(range(100))
        sig = compute_chunk_signature(tokens, dim=64)
        assert sig.shape == (64,)

    def test_empty_tokens(self):
        """Empty token list should return zero vector."""
        sig = compute_chunk_signature([])
        assert np.allclose(sig, 0.0)
        assert sig.shape == (128,)

    def test_single_token(self):
        """Single token (only 1-grams apply)."""
        sig = compute_chunk_signature([42])
        assert sig.shape == (128,)
        # Should be normalized to unit length
        assert 0.9 < np.linalg.norm(sig) < 1.1

    def test_two_tokens(self):
        """Two tokens (1-grams and 2-grams)."""
        sig = compute_chunk_signature([1, 2])
        assert sig.shape == (128,)
        norm = np.linalg.norm(sig)
        if norm > 1e-10:
            assert 0.9 < norm < 1.1

    def test_determinism(self):
        """Same input should always produce the same signature."""
        tokens = list(range(512))
        sig1 = compute_chunk_signature(tokens)
        sig2 = compute_chunk_signature(tokens)
        assert np.array_equal(sig1, sig2)

    def test_different_inputs_different_signatures(self):
        """Different token sequences should produce different signatures."""
        sig_a = compute_chunk_signature(list(range(512)))
        sig_b = compute_chunk_signature(list(range(512, 1024)))
        # They should NOT be identical
        assert not np.array_equal(sig_a, sig_b)

    def test_similar_inputs_similar_signatures(self):
        """Similar token sequences should have similar signatures."""
        base = list(range(500))
        # Same start, different end
        sig_a = compute_chunk_signature(base + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        sig_b = compute_chunk_signature(base + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13])
        sim = signature_similarity(sig_a, sig_b)
        # Should be very similar (only 1 token differs out of 512)
        assert sim > 0.9

    def test_dissimilar_inputs(self):
        """Very different sequences should have low similarity."""
        sig_a = compute_chunk_signature(list(range(512)))
        sig_b = compute_chunk_signature(list(range(1000, 1512)))
        sim = signature_similarity(sig_a, sig_b)
        # Should be reasonably different
        assert sim < 0.8

    def test_ngram_1_only(self):
        """Using only 1-grams should work."""
        sig = compute_chunk_signature(list(range(100)), ngram_sizes=(1,))
        assert sig.shape == (128,)

    def test_ngram_1_2(self):
        """Using 1-grams and 2-grams."""
        sig = compute_chunk_signature(list(range(100)), ngram_sizes=(1, 2))
        assert sig.shape == (128,)


class TestComputeQuerySignature:
    """Tests for compute_query_signature."""

    def test_basic(self):
        """Should work on recent token window."""
        tokens = [1, 2, 3, 4, 5]
        sig = compute_query_signature(tokens)
        assert sig.shape == (128,)
        assert sig.dtype == np.float32

    def test_empty(self):
        """Empty recent tokens."""
        sig = compute_query_signature([])
        assert np.allclose(sig, 0.0)

    def test_same_as_chunk(self):
        """Query signature should match chunk signature for same input."""
        tokens = list(range(32))
        sig_chunk = compute_chunk_signature(tokens)
        sig_query = compute_query_signature(tokens)
        assert np.array_equal(sig_chunk, sig_query)


class TestSignatureSimilarity:
    """Tests for similarity functions."""

    def test_identical(self):
        """Identical vectors should have similarity 1.0."""
        sig = compute_chunk_signature(list(range(100)))
        assert signature_similarity(sig, sig) == pytest.approx(1.0, abs=1e-6)

    def test_zero_vector(self):
        """Zero vector should have 0 similarity."""
        zero = np.zeros(128, dtype=np.float32)
        sig = compute_chunk_signature(list(range(100)))
        assert signature_similarity(zero, sig) == 0.0
        assert signature_similarity(sig, zero) == 0.0

    def test_both_zero(self):
        """Two zero vectors."""
        zero = np.zeros(128, dtype=np.float32)
        assert signature_similarity(zero, zero) == 0.0

    def test_hamming_identical(self):
        """Hamming similarity of identical vectors."""
        sig = compute_chunk_signature(list(range(100)))
        assert signature_hamming_similarity(sig, sig) == pytest.approx(1.0)

    def test_hamming_zero(self):
        """Hamming similarity with zero."""
        zero = np.zeros(128, dtype=np.float32)
        sig = compute_chunk_signature(list(range(100)))
        sim = signature_hamming_similarity(zero, sig)
        assert 0.0 <= sim <= 1.0


class TestEdgeCases:
    """Edge case tests for signature computation."""

    def test_very_long_sequence(self):
        """Very long token sequence (10K tokens)."""
        tokens = list(range(10000))
        sig = compute_chunk_signature(tokens)
        assert sig.shape == (128,)
        # Should still be normalized
        norm = np.linalg.norm(sig)
        if norm > 1e-10:
            assert 0.9 < norm < 1.1

    def test_duplicate_tokens(self):
        """Duplicate token IDs should be handled correctly."""
        sig = compute_chunk_signature([1] * 100)
        assert sig.shape == (128,)

    def test_negative_token_ids(self):
        """Negative token IDs (unusual but should be handled)."""
        sig = compute_chunk_signature([-1, -2, 0, 1, 2])
        assert sig.shape == (128,)

    def test_large_token_ids(self):
        """Very large token IDs (up to vocab size)."""
        sig = compute_chunk_signature([32000, 50000, 100000])
        assert sig.shape == (128,)
