"""
Perplexity Evaluation with Sparse KV Cache Policies.

Directly measures the closed-loop generative-quality degradation caused by
KV-cache pruning. This answers the reviewer criticism:
  "Recovery rate ↑ does not imply generation quality ↑."

Methodology
-----------
1. Prefill the full text and extract per-layer attention (via hooks).
2. Run the target policy to select active chunks.
3. Build a custom 4-D attention mask where inactive chunks are masked out
   (value = 0) while causal structure and active chunks remain visible.
4. Forward the text through the model with this sparse mask.
5. Manually compute next-token cross-entropy:  
   loss = CE( logits[:-1], labels[1:] )
6. Perplexity = exp(mean loss).

This is expensive (O(L²) mask memory) but it is the most faithful
approximation of "what would the model output if only these chunks were
present in the KV cache?".

Reference:
  - Bengio et al., "A Neural Probabilistic Language Model", JMLR 2003
    (standard perplexity definition)
"""

from __future__ import annotations

import gc
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _build_sparse_attention_mask(
    seq_len: int,
    selected_positions: List[int],
    device: str = "cuda",
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Build a 4-D attention mask [1, 1, seq_len, seq_len] where:
      - Causal structure is preserved (query i can only see key j <= i).
      - Only selected_positions are visible.
      - Masked positions = 0, visible positions = 1.
    """
    selected_set = set(selected_positions)
    mask = torch.zeros(1, 1, seq_len, seq_len, dtype=dtype, device=device)
    for q in range(seq_len):
        for k_pos in range(q + 1):
            if k_pos in selected_set:
                mask[0, 0, q, k_pos] = 1.0
    return mask


class SparseKVPerplexityEvaluator:
    """Evaluate perplexity under sparse KV retention policies."""

    def __init__(
        self,
        model,
        tokenizer,
        chunk_size: int = 64,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.device = device
        self.model.eval()

    def _extract_attention_for_policy(
        self,
        input_ids: torch.Tensor,
    ) -> np.ndarray:
        """
        Run a forward pass with output_attentions=True and return
        the last-layer attention averaged over batch/heads/query positions.
        Shape: [kv_len]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_attentions=True,
                use_cache=False,
            )
        if not outputs.attentions:
            raise RuntimeError("Model did not return attention weights")
        last_attn = outputs.attentions[-1]  # [B, H, Q, KV]
        # Average over batch, heads, query positions
        attn_1d = last_attn.float().mean(dim=(0, 1, 2)).cpu().numpy()
        attn_1d = np.nan_to_num(attn_1d, nan=0.0)
        del outputs, last_attn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return attn_1d

    def _chunks_from_attention(
        self,
        attn_1d: np.ndarray,
        seq_len: int,
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Aggregate 1-D attention into per-chunk masses."""
        boundaries = []
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            boundaries.append((start, end))

        num_chunks = len(boundaries)
        chunk_masses = np.zeros(num_chunks)
        for cid, (s, e) in enumerate(boundaries):
            if e <= len(attn_1d):
                chunk_masses[cid] = float(attn_1d[s:e].sum())

        total = chunk_masses.sum()
        if total > 0:
            chunk_masses /= total
        return boundaries, chunk_masses

    def _select_positions_from_policy(
        self,
        boundaries: List[Tuple[int, int]],
        chunk_masses: np.ndarray,
        budget_ratio: float,
        policy,
        step: int = 0,
    ) -> List[int]:
        """Run the policy and expand selected chunks to token positions."""
        num_chunks = len(boundaries)
        budget_chunks = max(1, int(num_chunks * budget_ratio))

        # Anchor chunks: first and last
        anchor_ids = [0, num_chunks - 1]

        # Policy input
        attn_dict = {i: float(chunk_masses[i]) for i in range(num_chunks)}

        # Handle different policy signatures
        if hasattr(policy, "select_active_chunks_rich"):
            try:
                raw_selected = policy.select_active_chunks_rich(
                    num_chunks, budget_chunks, attn_dict, anchor_ids, step,
                )
            except Exception:
                raw_selected = policy.select_active_chunks(
                    num_chunks, budget_chunks, attn_dict, anchor_ids, step,
                )
        else:
            raw_selected = policy.select_active_chunks(
                num_chunks, budget_chunks, attn_dict, anchor_ids, step,
            )

        # Enforce budget
        selected_set = set(anchor_ids)
        max_total = len(anchor_ids) + budget_chunks
        for cid in raw_selected:
            if len(selected_set) >= max_total:
                break
            selected_set.add(cid)

        # Expand chunks to positions
        selected_positions = []
        for cid in sorted(selected_set):
            s, e = boundaries[cid]
            selected_positions.extend(range(s, e))
        return selected_positions

    def evaluate_text(
        self,
        text: str,
        policy,
        budget_ratio: float,
    ) -> Dict[str, float]:
        """
        Compute perplexity for a single text under the given policy.
        """
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) < 2:
            return {"perplexity": float("nan"), "nll": float("nan"), "num_tokens": len(tokens)}

        seq_len = len(tokens)
        input_ids = torch.tensor([tokens], device=self.device)

        # 1. Extract attention for policy decision
        attn_1d = self._extract_attention_for_policy(input_ids)
        boundaries, chunk_masses = self._chunks_from_attention(attn_1d, seq_len)

        # 2. Policy selects chunks
        selected_positions = self._select_positions_from_policy(
            boundaries, chunk_masses, budget_ratio, policy, step=0,
        )

        # 3. Build sparse 4-D attention mask
        sparse_mask = _build_sparse_attention_mask(
            seq_len, selected_positions, device=self.device, dtype=torch.float32
        )

        # 4. Forward with sparse mask
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=sparse_mask)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        # 5. Compute next-token CE manually
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )
        nll = float(loss.item())
        perplexity = float(math.exp(nll))

        # Cleanup
        del logits, outputs, sparse_mask, input_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "perplexity": perplexity,
            "nll": nll,
            "num_tokens": seq_len,
            "num_chunks": len(boundaries),
            "selected_chunks": len(set(pos // self.chunk_size for pos in selected_positions)),
            "budget_ratio": budget_ratio,
        }

    def evaluate_batch(
        self,
        texts: List[str],
        policy,
        budget_ratio: float,
    ) -> Dict[str, float]:
        """
        Evaluate perplexity over a batch of texts and return aggregates.
        """
        perplexities = []
        nlls = []
        token_counts = []

        for idx, text in enumerate(texts):
            try:
                result = self.evaluate_text(text, policy, budget_ratio)
                if not math.isnan(result["nll"]):
                    perplexities.append(result["perplexity"])
                    nlls.append(result["nll"])
                    token_counts.append(result["num_tokens"])
            except Exception as e:
                logger.warning(f"Perplexity eval failed for text {idx}: {e}")

            # Aggressive cleanup between samples
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not nlls:
            return {
                "mean_perplexity": float("nan"),
                "median_perplexity": float("nan"),
                "mean_nll": float("nan"),
                "total_tokens": 0,
                "num_texts": 0,
            }

        # Token-weighted mean NLL is the correct aggregate perplexity
        total_tokens = sum(token_counts)
        weighted_nll = sum(n * t for n, t in zip(nlls, token_counts)) / total_tokens

        return {
            "mean_perplexity": float(np.mean(perplexities)),
            "median_perplexity": float(np.median(perplexities)),
            "token_weighted_perplexity": float(math.exp(weighted_nll)),
            "mean_nll": float(np.mean(nlls)),
            "weighted_nll": weighted_nll,
            "total_tokens": total_tokens,
            "num_texts": len(nlls),
        }
