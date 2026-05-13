"""
SnapKV Baseline.

SnapKV: Efficient LLM Inference with Adaptive KV Caching
Reference: https://arxiv.org/abs/2404.14469

Key idea: Observe attention patterns in an observation window,
then select KV entries with highest attention in that window.
This is more adaptive than H2O because it doesn't rely on
cumulative history.
"""

import torch
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SnapKVConfig:
    """Configuration for SnapKV."""
    observation_tokens: int = 64  # Number of tokens to observe
    retention_ratio: float = 0.3   # Ratio of KV to keep after observation
    
    def retention_budget(self, seq_len: int) -> int:
        """Compute retention budget."""
        return int(seq_len * self.retention_ratio)


class SnapKVPolicy:
    """
    SnapKV retention policy.
    
    Process:
    1. Observe attention patterns for first N tokens
    2. Score each KV position by attention received in observation window
    3. Keep top-K KV positions by score
    4. During generation, maintain only these selected positions
    """
    
    def __init__(
        self,
        observation_tokens: int = 64,
        retention_ratio: float = 0.3,
    ):
        """
        Initialize SnapKV policy.
        
        Args:
            observation_tokens: Number of tokens in observation window
            retention_ratio: Fraction of KV to retain after observation
        """
        self.config = SnapKVConfig(observation_tokens, retention_ratio)
        self.observation_scores: Optional[torch.Tensor] = None
        self.selected_positions: Optional[List[int]] = None
        
        logger.info(
            f"SnapKV: observation_tokens={observation_tokens}, "
            f"retention_ratio={retention_ratio}"
        )
    
    def observe(
        self,
        observation_attention: torch.Tensor,
    ):
        """
        Observe attention during observation window.
        
        Args:
            observation_attention: [num_obs_tokens, seq_len] attention weights
                where each row is attention from one observation query
        """
        # Sum attention across observation queries
        # This gives total attention received by each KV position
        self.observation_scores = observation_attention.sum(dim=0).cpu()  # [seq_len]
    
    def select_positions(self, seq_len: int) -> List[int]:
        """
        Select KV positions to retain based on observation.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Sorted list of retained positions
        """
        if self.observation_scores is None:
            logger.warning("SnapKV: No observation scores, keeping all")
            return list(range(seq_len))
        
        budget = self.config.retention_budget(seq_len)
        
        # Get top positions by observation score
        scores = self.observation_scores[:seq_len].numpy()
        top_indices = scores.argsort()[-budget:]
        
        self.selected_positions = sorted(top_indices.tolist())
        return self.selected_positions
    
    def get_pooling_features(
        self,
        hidden_states: torch.Tensor,
        positions: List[int],
    ) -> torch.Tensor:
        """
        Extract pooling features for selected positions.
        
        SnapKV can use pooling-based features as an alternative
        to attention-based selection.
        
        Args:
            hidden_states: [seq_len, hidden_dim]
            positions: Selected positions
            
        Returns:
            Pooled features [num_positions, hidden_dim]
        """
        return hidden_states[positions]


class SnapKVRunner:
    """
    End-to-end runner for SnapKV baseline.
    """
    
    def __init__(
        self,
        model_wrapper,
        observation_tokens: int = 64,
        retention_ratio: float = 0.3,
    ):
        """
        Initialize SnapKV runner.
        
        Args:
            model_wrapper: ModelWrapper instance
            observation_tokens: Observation window size
            retention_ratio: Retention ratio after observation
        """
        self.model_wrapper = model_wrapper
        self.policy = SnapKVPolicy(observation_tokens, retention_ratio)
        
        logger.info(f"SnapKVRunner: obs_tokens={observation_tokens}, ratio={retention_ratio}")
    
    def run(
        self,
        context_input_ids: torch.Tensor,
        query_input_ids: torch.Tensor,
        max_new_tokens: int = 50,
    ) -> Tuple[List[int], Dict]:
        """
        Run generation with SnapKV compression.
        
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
        
        # Run with attention output for observation
        with torch.no_grad():
            outputs = self.model_wrapper.model(
                input_ids=full_input_ids,
                output_attentions=True,
                use_cache=True,
            )
            
            past_key_values = outputs.past_key_values
            
            # Extract attention for observation window
            if outputs.attentions:
                # Use first few tokens as observation
                num_obs = min(self.policy.config.observation_tokens, full_input_ids.shape[1])
                
                # Aggregate attention from observation queries
                obs_attention = []
                for layer_attn in outputs.attentions:
                    # layer_attn: [batch, heads, q, kv]
                    # Average over heads, take first num_obs queries
                    avg_attn = layer_attn[0, :, :num_obs, :].mean(dim=0)  # [num_obs, kv]
                    obs_attention.append(avg_attn)
                
                # Average over layers
                obs_attention = torch.stack(obs_attention).mean(dim=0).cpu()
                
                # Observe
                self.policy.observe(obs_attention)
        
        # Select positions based on observation
        seq_len = full_input_ids.shape[1]
        retained_positions = self.policy.select_positions(seq_len)
        
        # Generate with pruned KV
        generated_ids, _ = self.model_wrapper.generate_with_pruned_kv(
            query_input_ids=query_input_ids,
            past_key_values=past_key_values,
            retained_positions=retained_positions,
            max_new_tokens=max_new_tokens,
        )
        
        debug_info = {
            "observation_tokens": self.policy.config.observation_tokens,
            "retention_ratio": self.policy.config.retention_ratio,
            "retained_positions": len(retained_positions),
            "compression_ratio": len(retained_positions) / seq_len,
        }
        
        return generated_ids[0].tolist(), debug_info


class PerLayerSnapKV(SnapKVPolicy):
    """
    Extension: Per-layer SnapKV.
    
    Different layers may need different KV retention policies.
    Early layers (syntactic) vs late layers (semantic) have
    different attention patterns.
    """
    
    def __init__(
        self,
        observation_tokens: int = 64,
        base_retention_ratio: float = 0.3,
        layer_decay: float = 0.9,  # Later layers keep less
    ):
        super().__init__(observation_tokens, base_retention_ratio)
        self.layer_decay = layer_decay
        self.per_layer_scores: Dict[int, torch.Tensor] = {}
    
    def observe_per_layer(
        self,
        layer_attentions: Dict[int, torch.Tensor],
    ):
        """
        Observe attention per layer.
        
        Args:
            layer_attentions: {layer_idx: [num_obs, seq_len]}
        """
        for layer_idx, attn in layer_attentions.items():
            self.per_layer_scores[layer_idx] = attn.sum(dim=0).cpu()
    
    def select_positions_per_layer(
        self,
        seq_len: int,
        num_layers: int,
    ) -> Dict[int, List[int]]:
        """
        Select positions per layer with decay.
        
        Returns:
            {layer_idx: positions}
        """
        result = {}
        
        for layer_idx in range(num_layers):
            if layer_idx not in self.per_layer_scores:
                # No scores, keep all
                result[layer_idx] = list(range(seq_len))
                continue
            
            # Apply layer decay
            decay = self.layer_decay ** layer_idx
            effective_ratio = self.config.retention_ratio * decay
            budget = int(seq_len * effective_ratio)
            
            scores = self.per_layer_scores[layer_idx][:seq_len].numpy()
            top_indices = scores.argsort()[-budget:]
            result[layer_idx] = sorted(top_indices.tolist())
        
        return result
