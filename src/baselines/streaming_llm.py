"""
StreamingLLM Baseline.

StreamingLLM: Efficient Streaming Language Models with Attention Sinks
Reference: https://arxiv.org/abs/2309.05353

Key idea: Keep initial "sink" tokens (first few tokens) plus a recent window.
The sink tokens act as attention "sinks" that absorb attention without
needing to have high semantic importance.
"""

import torch
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SinkConfig:
    """Configuration for attention sink."""
    num_sink_tokens: int = 4  # Number of initial tokens to always keep
    window_size: int = 1024   # Size of recent window
    
    def total_budget(self) -> int:
        return self.num_sink_tokens + self.window_size


class StreamingLLMPolicy:
    """
    StreamingLLM retention policy.
    
    Keeps:
    1. First N "sink" tokens (attention sinks)
    2. Most recent W tokens (sliding window)
    
    Discards everything in between.
    """
    
    def __init__(
        self,
        num_sink_tokens: int = 4,
        window_size: int = 1024,
    ):
        """
        Initialize StreamingLLM policy.
        
        Args:
            num_sink_tokens: Number of initial sink tokens to keep
            window_size: Size of recent sliding window
        """
        self.config = SinkConfig(num_sink_tokens, window_size)
        
        logger.info(
            f"StreamingLLM: sink_tokens={num_sink_tokens}, window={window_size}"
        )
    
    def select_retained_positions(
        self,
        seq_len: int,
    ) -> List[int]:
        """
        Select positions to retain based on StreamingLLM policy.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Sorted list of retained positions
        """
        sink = list(range(min(self.config.num_sink_tokens, seq_len)))
        
        # Recent window
        window_start = max(0, seq_len - self.config.window_size)
        window = list(range(window_start, seq_len))
        
        # If there's overlap between sink and window, handle it
        retained = sorted(set(sink) | set(window))
        
        return retained
    
    def should_evict(self, position: int, current_seq_len: int) -> bool:
        """
        Check if a position should be evicted.
        
        Args:
            position: Token position
            current_seq_len: Current sequence length
            
        Returns:
            True if position should be evicted
        """
        # Never evict sink tokens
        if position < self.config.num_sink_tokens:
            return False
        
        # Never evict recent window
        if position >= current_seq_len - self.config.window_size:
            return False
        
        # Everything else is evicted
        return True


class StreamingLLMRunner:
    """
    End-to-end runner for StreamingLLM baseline.
    """
    
    def __init__(
        self,
        model_wrapper,
        num_sink_tokens: int = 4,
        window_size: int = 1024,
    ):
        """
        Initialize StreamingLLM runner.
        
        Args:
            model_wrapper: ModelWrapper instance
            num_sink_tokens: Number of sink tokens
            window_size: Recent window size
        """
        self.model_wrapper = model_wrapper
        self.policy = StreamingLLMPolicy(num_sink_tokens, window_size)
        
        logger.info(f"StreamingLLMRunner: sink={num_sink_tokens}, window={window_size}")
    
    def run(
        self,
        context_input_ids: torch.Tensor,
        query_input_ids: torch.Tensor,
        max_new_tokens: int = 50,
    ) -> Tuple[List[int], Dict]:
        """
        Run generation with StreamingLLM compression.
        
        Args:
            context_input_ids: Context token IDs
            query_input_ids: Query token IDs
            max_new_tokens: Max tokens to generate
            
        Returns:
            (generated_ids, debug_info)
        """
        # Prefill to get full KV
        full_input_ids = torch.cat([context_input_ids, query_input_ids], dim=1)
        context_len = context_input_ids.shape[1]
        
        logits, past_key_values = self.model_wrapper.prefill(full_input_ids)
        
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
            "num_sink_tokens": self.policy.config.num_sink_tokens,
            "window_size": self.policy.config.window_size,
            "retained_positions": len(retained_positions),
            "compression_ratio": len(retained_positions) / seq_len,
        }
        
        return generated_ids[0].tolist(), debug_info


class StreamingLLMWithPositionBias(StreamingLLMPolicy):
    """
    Extension: StreamingLLM with learned position bias.
    
    Instead of fixed sink tokens, learn which initial positions
    are most important as attention sinks.
    """
    
    def __init__(
        self,
        num_sink_tokens: int = 4,
        window_size: int = 1024,
        learnable_positions: bool = False,
    ):
        super().__init__(num_sink_tokens, window_size)
        self.learnable_positions = learnable_positions
        
        # Position importance scores (learned)
        if learnable_positions:
            self.position_scores = torch.zeros(num_sink_tokens)
        else:
            self.position_scores = None
    
    def update_position_scores(self, attention_patterns: List[torch.Tensor]):
        """
        Update position importance scores from observed attention.
        
        Args:
            attention_patterns: List of attention weight tensors
        """
        if not self.learnable_positions:
            return
        
        # Average attention to each sink position
        for attn in attention_patterns:
            # attn: [kv_len]
            for i in range(min(self.config.num_sink_tokens, len(attn))):
                self.position_scores[i] += attn[i].item()
        
        # Renormalize
        self.position_scores /= len(attention_patterns)
