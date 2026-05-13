"""
H2O (Heavy Hitter Oracle) Baseline.

H2O: Heavy-Hitter Oracle for Accurate KV Cache Compression
Reference: https://arxiv.org/abs/2306.14022

Key idea: Keep KV entries with highest cumulative attention scores.
"""

import torch
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from src.attention.kv_cache_manager import SparseKVCacheManager

logger = logging.getLogger(__name__)


@dataclass
class H2OKVCache:
    """H2O-style KV cache with heavy hitter tracking."""
    # Cumulative attention scores per position
    cumulative_scores: torch.Tensor  # [seq_len]
    
    # Positions sorted by score (descending)
    sorted_positions: List[int] = None
    
    def __post_init__(self):
        if self.sorted_positions is None:
            self._resort()
    
    def _resort(self):
        """Sort positions by cumulative score."""
        scores = self.cumulative_scores.cpu().numpy()
        self.sorted_positions = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
    
    def update_scores(self, new_attention: torch.Tensor):
        """Update cumulative scores with new attention weights."""
        # new_attention: [seq_len] - attention weights to each KV position
        self.cumulative_scores += new_attention.cpu()
        self._resort()
    
    def get_heavy_hitters(self, budget: int) -> List[int]:
        """Get top-k positions by cumulative attention."""
        return sorted(self.sorted_positions[:budget])


class H2OPolicy:
    """
    H2O: Heavy-Hitter Oracle retention policy.
    
    Keeps entries with highest cumulative attention scores.
    The "oracle" comes from using actual attention weights
to determine importance.
    """
    
    def __init__(
        self,
        hh_ratio: float = 0.1,
        recent_ratio: float = 0.1,
    ):
        """
        Initialize H2O policy.
        
        Args:
            hh_ratio: Ratio of heavy hitters to keep
            recent_ratio: Ratio of recent tokens to keep
                        (H2O keeps both heavy hitters + recent window)
        """
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.cumulative_scores: Optional[torch.Tensor] = None
        
        logger.info(f"H2O policy: hh_ratio={hh_ratio}, recent_ratio={recent_ratio}")
    
    def initialize(self, seq_len: int):
        """Initialize cumulative scores."""
        self.cumulative_scores = torch.zeros(seq_len)
    
    def update(
        self,
        attention_weights: torch.Tensor,
        query_position: int,
    ):
        """
        Update cumulative scores with new attention.
        
        Args:
            attention_weights: [kv_len] attention weights
            query_position: Current query position
        """
        if self.cumulative_scores is None:
            self.initialize(attention_weights.shape[0])
        
        # Accumulate attention scores
        self.cumulative_scores[:len(attention_weights)] += attention_weights.cpu()
    
    def select_retained_positions(
        self,
        seq_len: int,
        budget: Optional[int] = None,
    ) -> List[int]:
        """
        Select positions to retain based on H2O policy.
        
        Args:
            seq_len: Sequence length
            budget: Total budget (if None, uses hh_ratio + recent_ratio)
            
        Returns:
            Sorted list of retained positions
        """
        if self.cumulative_scores is None:
            # No scores yet, keep all
            return list(range(seq_len))
        
        if budget is None:
            budget = int(seq_len * (self.hh_ratio + self.recent_ratio))
        
        # Split budget between heavy hitters and recent
        hh_budget = int(budget * (self.hh_ratio / (self.hh_ratio + self.recent_ratio)))
        recent_budget = budget - hh_budget
        
        # Get heavy hitters
        scores = self.cumulative_scores[:seq_len].cpu().numpy()
        hh_positions = set()
        if hh_budget > 0:
            top_indices = scores.argsort()[-hh_budget:]
            hh_positions = set(top_indices.tolist())
        
        # Get recent positions
        recent_positions = set(range(max(0, seq_len - recent_budget), seq_len))
        
        # Union
        retained = sorted(hh_positions | recent_positions)
        
        return retained
    
    def apply_to_kv_cache(
        self,
        full_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        retained_positions: List[int],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Apply H2O policy to KV cache.
        
        Args:
            full_kv: Full KV cache from prefill
            retained_positions: Positions to retain
            
        Returns:
            Pruned KV cache
        """
        pos_tensor = torch.tensor(retained_positions, dtype=torch.long)
        
        pruned_kv = []
        for layer_idx in range(len(full_kv)):
            k, v = full_kv[layer_idx]
            pruned_k = k[:, :, pos_tensor.to(k.device), :]
            pruned_v = v[:, :, pos_tensor.to(v.device), :]
            pruned_kv.append((pruned_k, pruned_v))
        
        return tuple(pruned_kv)


class H2ORunner:
    """
    End-to-end runner for H2O baseline.
    
    Runs generation with H2O-style KV cache compression.
    """
    
    def __init__(
        self,
        model_wrapper,
        hh_ratio: float = 0.1,
        recent_ratio: float = 0.1,
    ):
        """
        Initialize H2O runner.
        
        Args:
            model_wrapper: ModelWrapper instance
            hh_ratio: Heavy hitter ratio
            recent_ratio: Recent window ratio
        """
        self.model_wrapper = model_wrapper
        self.policy = H2OPolicy(hh_ratio, recent_ratio)
        
        logger.info(f"H2ORunner: hh_ratio={hh_ratio}, recent_ratio={recent_ratio}")
    
    def run(
        self,
        context_input_ids: torch.Tensor,
        query_input_ids: torch.Tensor,
        max_new_tokens: int = 50,
    ) -> Tuple[List[int], Dict]:
        """
        Run generation with H2O compression.
        
        Args:
            context_input_ids: Context token IDs
            query_input_ids: Query token IDs  
            max_new_tokens: Max tokens to generate
            
        Returns:
            (generated_ids, debug_info)
        """
        # Prefill to get full KV and attention
        full_input_ids = torch.cat([context_input_ids, query_input_ids], dim=1)
        context_len = context_input_ids.shape[1]
        
        logits, past_key_values = self.model_wrapper.prefill(full_input_ids)
        
        # Extract attention from prefill
        with torch.no_grad():
            outputs = self.model_wrapper.model(
                input_ids=full_input_ids,
                output_attentions=True,
                use_cache=True,
            )
            
            # Aggregate attention across layers
            if outputs.attentions:
                # Use last layer attention
                last_attn = outputs.attentions[-1]  # [batch, heads, q, kv]
                # Average over heads, take last query position
                avg_attn = last_attn[0, :, -1, :].mean(dim=0)  # [kv_len]
                
                # Initialize and update H2O policy
                self.policy.initialize(avg_attn.shape[0])
                self.policy.update(avg_attn, full_input_ids.shape[1] - 1)
        
        # Select retained positions
        seq_len = full_input_ids.shape[1]
        retained_positions = self.policy.select_retained_positions(seq_len)
        
        # Generate with pruned KV
        generated_ids, _ = self.model_wrapper.generate_with_pruned_kv(
            query_input_ids=query_input_ids,
            past_key_values=past_key_values,
            retained_positions=retained_positions,
            max_new_tokens=max_new_tokens,
        )
        
        debug_info = {
            "hh_ratio": self.policy.hh_ratio,
            "recent_ratio": self.policy.recent_ratio,
            "retained_positions": len(retained_positions),
            "compression_ratio": len(retained_positions) / seq_len,
        }
        
        return generated_ids[0].tolist(), debug_info
