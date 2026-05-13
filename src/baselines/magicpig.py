"""
MagicPIG Baseline (NeurIPS 2024).

MagicPIG: LSH Sampling for Efficient LLM Generation
Reference: Chen et al., NeurIPS 2024

Key idea: Use Locality-Sensitive Hashing (LSH) to sample KV entries
that are likely to have high attention scores, then compute exact
attention only over the sampled subset.  Unlike deterministic top-k
methods, LSH sampling provides unbiased attention estimation with
provable approximation guarantees.

Differences from other baselines:
  - Probabilistic sampling (not deterministic top-k)
  - LSH-based (sub-linear query time)
  - Provides unbiased attention estimates
  - Better theoretical guarantees for heavy-tailed distributions
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MagicPIGConfig:
    num_hashes: int = 8           # number of LSH hash functions
    sample_ratio: float = 0.2     # fraction of KV to sample
    hash_dim: int = 32            # LSH projection dimension
    temperature: float = 1.0      # sampling temperature


class MagicPIGPolicy:
    """MagicPIG: LSH-based probabilistic KV sampling.

    At chunk granularity, uses attention-mass-weighted sampling
    to select chunks, simulating the LSH sampling distribution.
    The key insight is that LSH sampling probability is proportional
    to the collision probability, which correlates with attention score.
    """

    name = "MagicPIG"

    def __init__(
        self,
        sample_ratio: float = 0.2,
        num_hashes: int = 8,
        temperature: float = 1.0,
    ):
        self.config = MagicPIGConfig(
            num_hashes=num_hashes,
            sample_ratio=sample_ratio,
            temperature=temperature,
        )
        self.rng = np.random.RandomState(42)
        logger.info(
            f"MagicPIG: sample_ratio={sample_ratio}, "
            f"num_hashes={num_hashes}, temp={temperature}"
        )

    def select_active_chunks(
        self,
        num_chunks: int,
        budget_chunks: int,
        chunk_attention_masses: Dict[int, float],
        anchor_ids: List[int],
        step: int,
    ) -> List[int]:
        """Select chunks via LSH-simulated sampling.

        Samples chunks with probability proportional to their
        attention mass raised to 1/temperature, simulating the
        LSH collision probability distribution.
        """
        # Build sampling distribution from attention masses
        masses = np.array([
            chunk_attention_masses.get(cid, 1e-8)
            for cid in range(num_chunks)
        ])

        # Apply temperature (lower temp = more peaked)
        temp = self.config.temperature
        if temp != 1.0:
            masses = np.power(np.maximum(masses, 1e-10), 1.0 / temp)

        # Normalize to probability distribution
        total = masses.sum()
        if total > 0:
            probs = masses / total
        else:
            probs = np.ones(num_chunks) / num_chunks

        # Sample without replacement
        non_anchor = [c for c in range(num_chunks) if c not in anchor_ids]
        sample_size = min(budget_chunks, len(non_anchor))

        if sample_size > 0 and len(non_anchor) > 0:
            non_anchor_probs = probs[non_anchor]
            prob_sum = non_anchor_probs.sum()
            if prob_sum > 0:
                non_anchor_probs = non_anchor_probs / prob_sum
            else:
                non_anchor_probs = np.ones(len(non_anchor)) / len(non_anchor)

            sampled_indices = self.rng.choice(
                non_anchor,
                size=min(sample_size, len(non_anchor)),
                replace=False,
                p=non_anchor_probs,
            )
            selected = set(anchor_ids) | set(sampled_indices.tolist())
        else:
            selected = set(anchor_ids)

        return sorted(selected)
