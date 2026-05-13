"""
RetrievalAttention Baseline (NeurIPS 2024).

RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval
Reference: Liu et al., NeurIPS 2024

Key idea: Build an approximate nearest-neighbor (ANN) index over KV cache
keys.  At each decode step, use the query vector to retrieve the top-k
most relevant KV entries via ANN search, then compute exact attention
only over the retrieved subset.

Differences from Quest:
  - Token-level retrieval (not page-level)
  - Uses ANN index (HNSW/IVF) for sub-linear search
  - Maintains a persistent index across decode steps
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalAttentionConfig:
    top_k_tokens: int = 256       # tokens to retrieve per step
    index_type: str = "flat"      # "flat", "ivf", "hnsw" (flat for simulation)
    nprobe: int = 8               # IVF nprobe
    recent_window: int = 64       # always keep recent tokens


class RetrievalAttentionPolicy:
    """RetrievalAttention: ANN-based token-level KV retrieval.

    At chunk granularity (for ProSE integration), selects chunks whose
    aggregate key similarity to the current query is highest.
    """

    name = "RetrievalAttention"

    def __init__(
        self,
        top_k_tokens: int = 256,
        recent_window: int = 64,
    ):
        self.config = RetrievalAttentionConfig(
            top_k_tokens=top_k_tokens,
            recent_window=recent_window,
        )
        logger.info(
            f"RetrievalAttention: top_k={top_k_tokens}, recent={recent_window}"
        )

    def select_active_chunks(
        self,
        num_chunks: int,
        budget_chunks: int,
        chunk_attention_masses: Dict[int, float],
        anchor_ids: List[int],
        step: int,
    ) -> List[int]:
        """Select chunks via simulated ANN retrieval.

        Uses current-step attention mass as proxy for query-key similarity.
        Always includes recent chunks (simulating the recent window).
        """
        # Recent window: last few chunks always active
        recent_chunks = max(1, self.config.recent_window // 64)
        recent_ids = list(range(max(0, num_chunks - recent_chunks), num_chunks))

        # Rank remaining by attention mass (proxy for ANN similarity)
        candidates = [
            (cid, chunk_attention_masses.get(cid, 0.0))
            for cid in range(num_chunks)
            if cid not in recent_ids
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        selected = set(anchor_ids) | set(recent_ids)
        remaining_budget = budget_chunks + len(anchor_ids) - len(selected)
        for cid, _ in candidates:
            if remaining_budget <= 0:
                break
            if cid not in selected:
                selected.add(cid)
                remaining_budget -= 1
        return sorted(selected)
