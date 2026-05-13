"""
Quest Baseline (ICML 2024).

Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
Reference: Tang et al., ICML 2024

Key idea: At each decode step, use the current query token's key projection
to estimate which KV pages are relevant, then only load those pages for
full attention.  Quest partitions the KV cache into fixed-size pages and
keeps a lightweight per-page "key centroid" in GPU memory.  At query time
it computes inner products between the query and all centroids, selects
the top-k pages, and fetches only those from CPU/CXL memory.

Differences from H2O/SnapKV:
  - Dynamic per-step selection (not static one-shot)
  - Page-granularity retrieval (not token-level)
  - No cumulative history; purely query-driven
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuestConfig:
    page_size: int = 16          # tokens per page
    top_k_pages: int = 8         # pages to retrieve per step
    centroid_method: str = "mean" # "mean" or "max"


class QuestPolicy:
    """Quest: query-aware page-level KV retrieval.

    Maintains per-page key centroids and selects pages dynamically
    based on query-centroid similarity at each decode step.
    """

    name = "Quest"

    def __init__(
        self,
        page_size: int = 16,
        top_k_pages: int = 8,
        centroid_method: str = "mean",
    ):
        self.config = QuestConfig(page_size, top_k_pages, centroid_method)
        # page_centroids[page_id] = centroid vector (np.ndarray)
        self.page_centroids: Dict[int, np.ndarray] = {}
        self.num_pages: int = 0
        logger.info(f"Quest: page_size={page_size}, top_k={top_k_pages}")

    def build_page_index(
        self,
        chunk_attention_masses: Dict[int, float],
        num_chunks: int,
    ) -> None:
        """Build page centroids from chunk-level attention masses.

        In the real Quest, centroids are computed from key projections.
        Here we use attention masses as a proxy for the centroid quality
        signal, which is sufficient for the promotion-level simulation.
        """
        self.num_pages = num_chunks
        for cid in range(num_chunks):
            mass = chunk_attention_masses.get(cid, 0.0)
            self.page_centroids[cid] = np.array([mass])

    def select_active_chunks(
        self,
        num_chunks: int,
        budget_chunks: int,
        chunk_attention_masses: Dict[int, float],
        anchor_ids: List[int],
        step: int,
    ) -> List[int]:
        """Select chunks via query-centroid similarity."""
        # Rebuild index every step (Quest is fully dynamic)
        self.build_page_index(chunk_attention_masses, num_chunks)

        # Rank by current-step attention mass (proxy for query-centroid sim)
        candidates = [
            (cid, chunk_attention_masses.get(cid, 0.0))
            for cid in range(num_chunks)
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        selected = set(anchor_ids)
        for cid, _ in candidates:
            if len(selected) >= budget_chunks + len(anchor_ids):
                break
            selected.add(cid)
        return sorted(selected)


class QuestRunner:
    """End-to-end Quest runner for benchmark evaluation."""

    def __init__(
        self,
        model_wrapper,
        page_size: int = 16,
        top_k_pages: int = 8,
    ):
        self.model_wrapper = model_wrapper
        self.policy = QuestPolicy(page_size, top_k_pages)
        logger.info(f"QuestRunner: page_size={page_size}, top_k={top_k_pages}")

    def run(self, context_input_ids, query_input_ids, max_new_tokens=50):
        import torch
        full_input_ids = torch.cat([context_input_ids, query_input_ids], dim=1)
        seq_len = full_input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model_wrapper.model(
                input_ids=full_input_ids,
                output_attentions=True,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            if outputs.attentions:
                last_attn = outputs.attentions[-1]
                avg_attn = last_attn[0, :, -1, :].mean(dim=0).cpu()
            else:
                avg_attn = torch.ones(seq_len) / seq_len

        # Build page-level attention
        page_size = self.policy.config.page_size
        num_pages = math.ceil(seq_len / page_size)
        page_attn = {}
        for pid in range(num_pages):
            start = pid * page_size
            end = min(start + page_size, seq_len)
            page_attn[pid] = float(avg_attn[start:end].sum())

        # Select top-k pages
        budget = self.policy.config.top_k_pages
        sorted_pages = sorted(page_attn.items(), key=lambda x: x[1], reverse=True)
        selected_pages = [pid for pid, _ in sorted_pages[:budget]]

        # Convert pages to positions
        retained_positions = []
        for pid in sorted(selected_pages):
            start = pid * page_size
            end = min(start + page_size, seq_len)
            retained_positions.extend(range(start, end))

        generated_ids, _ = self.model_wrapper.generate_with_pruned_kv(
            query_input_ids=query_input_ids,
            past_key_values=past_key_values,
            retained_positions=retained_positions,
            max_new_tokens=max_new_tokens,
        )

        debug_info = {
            "page_size": page_size,
            "top_k_pages": budget,
            "retained_positions": len(retained_positions),
            "compression_ratio": len(retained_positions) / seq_len,
        }
        return generated_ids[0].tolist(), debug_info
