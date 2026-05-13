"""
InfiniGen Baseline (OSDI 2024).

InfiniGen: Efficient Generative Inference of Large Language Models
with Dynamic KV Cache Management
Reference: Lee et al., OSDI 2024

Key idea: Use a lightweight "prefill predictor" (the previous layer's
attention pattern) to predict which KV entries will be needed in the
current layer, then prefetch only those entries from CPU memory.
This creates a layer-by-layer speculative prefetch pipeline.

Differences from Quest/RetrievalAttention:
  - Layer-wise prediction (not global)
  - Uses previous layer's attention as predictor for next layer
  - Speculative prefetch overlaps with compute
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InfiniGenConfig:
    prefetch_ratio: float = 0.3    # fraction of KV to prefetch per layer
    predictor_layers: int = 1      # how many previous layers to use
    speculative_depth: int = 2     # prefetch this many layers ahead


class InfiniGenPolicy:
    """InfiniGen: layer-wise speculative KV prefetch.

    At chunk granularity, uses a two-pass strategy:
    1. First pass: compute attention with anchor-only KV
    2. Use attention pattern to predict which tail chunks to fetch
    3. Second pass: compute with fetched chunks

    For the promotion-level simulation, we model this as:
    attention-weighted selection with a recency bias (since InfiniGen
    favors recently-accessed entries due to its layer-pipeline design).
    """

    name = "InfiniGen"

    def __init__(
        self,
        prefetch_ratio: float = 0.3,
        speculative_depth: int = 2,
    ):
        self.config = InfiniGenConfig(
            prefetch_ratio=prefetch_ratio,
            speculative_depth=speculative_depth,
        )
        self.prev_step_attention: Dict[int, float] = {}
        logger.info(
            f"InfiniGen: prefetch_ratio={prefetch_ratio}, "
            f"spec_depth={speculative_depth}"
        )

    def select_active_chunks(
        self,
        num_chunks: int,
        budget_chunks: int,
        chunk_attention_masses: Dict[int, float],
        anchor_ids: List[int],
        step: int,
    ) -> List[int]:
        """Select chunks via layer-wise speculative prediction.

        Combines current attention with previous step's pattern
        (simulating cross-layer prediction).
        """
        # Blend current and previous attention (InfiniGen's cross-layer signal)
        alpha = 0.7  # weight for current step
        blended = {}
        for cid in range(num_chunks):
            curr = chunk_attention_masses.get(cid, 0.0)
            prev = self.prev_step_attention.get(cid, 0.0)
            blended[cid] = alpha * curr + (1 - alpha) * prev

        # Store for next step
        self.prev_step_attention = dict(chunk_attention_masses)

        # Select top chunks by blended score
        candidates = sorted(blended.items(), key=lambda x: x[1], reverse=True)
        selected = set(anchor_ids)
        for cid, _ in candidates:
            if len(selected) >= budget_chunks + len(anchor_ids):
                break
            selected.add(cid)
        return sorted(selected)
