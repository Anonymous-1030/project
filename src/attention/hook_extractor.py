"""
Real Attention Weight Extraction via Forward Hooks.

Replaces the fake L2-norm/cosine-similarity proxy with genuine per-head,
per-layer attention scores captured during the forward pass.

Supports: Qwen2, Llama, Mistral architecture families.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Architecture-specific attention module paths
_ATTN_MODULE_PATHS = {
    "qwen2": "model.layers.{}.self_attn",
    "llama": "model.layers.{}.self_attn",
    "mistral": "model.layers.{}.self_attn",
}


@dataclass
class ChunkAttentionInfo:
    """Per-chunk attention mass aggregated across heads/layers."""
    chunk_id: int
    token_start: int
    token_end: int
    attention_mass: float  # Sum of attention weights pointing to this chunk
    per_head_mass: Optional[List[float]] = None
    per_layer_mass: Optional[List[float]] = None


@dataclass
class AttentionCaptureResult:
    """Complete attention capture from a forward pass."""
    # Raw per-layer attention: layer_idx -> [batch, heads, q_len, kv_len]
    layer_attention: Dict[int, torch.Tensor] = field(default_factory=dict)
    # Aggregated chunk-level attention
    chunk_attention: Optional[List[ChunkAttentionInfo]] = None
    # KV cache from the forward pass
    past_key_values: Optional[Any] = None
    # Logits
    logits: Optional[torch.Tensor] = None


def _resolve_module(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted path to a submodule."""
    parts = path.split(".")
    mod = model
    for part in parts:
        mod = getattr(mod, part)
    return mod


def _detect_architecture(model: nn.Module) -> str:
    """Detect model architecture from config."""
    model_type = getattr(model.config, "model_type", "").lower()
    if "qwen2" in model_type or "qwen" in model_type:
        return "qwen2"
    elif "llama" in model_type:
        return "llama"
    elif "mistral" in model_type:
        return "mistral"
    else:
        logger.warning(f"Unknown model_type '{model_type}', defaulting to llama")
        return "llama"


def _get_num_kv_heads(model: nn.Module) -> int:
    """Get number of KV heads (for GQA models)."""
    config = model.config
    if hasattr(config, "num_key_value_heads"):
        return config.num_key_value_heads
    return config.num_attention_heads


class AttentionHookExtractor:
    """
    Extracts real attention weights from transformer models via forward hooks.

    Usage:
        extractor = AttentionHookExtractor(model)
        with extractor.capture() as capture:
            outputs = model(input_ids, attention_mask=mask, output_attentions=True)
        attn_data = capture.layer_attention  # {layer_idx: Tensor}
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.arch = _detect_architecture(model)
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.num_kv_heads = _get_num_kv_heads(model)
        self.head_dim = model.config.hidden_size // self.num_heads

        self._hooks: List[Any] = []
        self._capture_data: Dict[int, torch.Tensor] = {}

        logger.info(
            f"AttentionHookExtractor: arch={self.arch}, "
            f"layers={self.num_layers}, heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, head_dim={self.head_dim}"
        )

    def _make_hook(self, layer_idx: int):
        """Create a forward hook that captures attention weights."""
        def hook_fn(module, args, output):
            # For HuggingFace models with output_attentions=True,
            # output is (attn_output, attn_weights, past_kv)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                    self._capture_data[layer_idx] = attn_weights.detach().cpu()
        return hook_fn

    def _register_hooks(self) -> None:
        """Register forward hooks on attention modules."""
        attn_path_template = _ATTN_MODULE_PATHS[self.arch]
        for layer_idx in range(self.num_layers):
            path = attn_path_template.format(layer_idx)
            try:
                attn_module = _resolve_module(self.model, path)
                hook = attn_module.register_forward_hook(self._make_hook(layer_idx))
                self._hooks.append(hook)
            except AttributeError:
                logger.warning(f"Could not find attention module at {path}")

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @contextmanager
    def capture(self):
        """
        Context manager for capturing attention weights.

        Yields an AttentionCaptureResult that is populated after
        the forward pass completes inside the context.
        """
        result = AttentionCaptureResult()
        self._capture_data = {}
        self._register_hooks()
        try:
            yield result
        finally:
            result.layer_attention = dict(self._capture_data)
            self._remove_hooks()
            self._capture_data = {}

    def extract_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> AttentionCaptureResult:
        """
        Run a forward pass and extract real attention weights.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            layer_indices: Which layers to capture (None = all)

        Returns:
            AttentionCaptureResult with per-layer attention tensors
        """
        with self.capture() as result:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    use_cache=True,
                )
            result.logits = outputs.logits
            result.past_key_values = outputs.past_key_values

        # Filter to requested layers
        if layer_indices is not None:
            result.layer_attention = {
                k: v for k, v in result.layer_attention.items()
                if k in layer_indices
            }

        return result

    def extract_and_aggregate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_boundaries: Optional[List[Tuple[int, int]]] = None,
        query_token_range: Optional[Tuple[int, int]] = None,
        aggregation: str = "mean_heads_mean_layers",
    ) -> Tuple[AttentionCaptureResult, Dict[int, float]]:
        """
        Extract attention and aggregate to per-token or per-chunk scores.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            chunk_boundaries: List of (start, end) token positions for chunks
            query_token_range: (start, end) for query tokens whose attention to aggregate
            aggregation: How to aggregate across heads and layers

        Returns:
            (AttentionCaptureResult, per_position_or_chunk_attention_mass)
        """
        result = self.extract_attention(input_ids, attention_mask)

        if not result.layer_attention:
            logger.warning("No attention weights captured — model may use FlashAttention")
            return result, {}

        seq_len = input_ids.shape[1]

        # Determine query range — by default, use the last token (autoregressive query)
        if query_token_range is None:
            q_start, q_end = seq_len - 1, seq_len
        else:
            q_start, q_end = query_token_range

        # Aggregate attention across layers and heads
        # Shape per layer: [batch, heads, q_len, kv_len]
        accumulated = torch.zeros(seq_len, dtype=torch.float32)
        n_layers = 0

        for layer_idx, attn in result.layer_attention.items():
            # attn: [batch, heads, q_len, kv_len]
            # Select query positions and average over heads
            q_attn = attn[0, :, q_start:q_end, :seq_len]  # [heads, q_range, kv_len]
            # Mean over heads
            head_mean = q_attn.mean(dim=0)  # [q_range, kv_len]
            # Mean over query positions
            per_kv = head_mean.mean(dim=0)  # [kv_len]
            accumulated += per_kv
            n_layers += 1

        if n_layers > 0:
            accumulated /= n_layers

        # If chunk boundaries provided, aggregate to chunk level
        if chunk_boundaries is not None:
            chunk_masses = {}
            for chunk_idx, (c_start, c_end) in enumerate(chunk_boundaries):
                c_end_clamped = min(c_end, seq_len)
                if c_start < c_end_clamped:
                    chunk_masses[chunk_idx] = float(accumulated[c_start:c_end_clamped].sum())
                else:
                    chunk_masses[chunk_idx] = 0.0
            return result, chunk_masses
        else:
            # Return per-token attention mass
            token_masses = {i: float(accumulated[i]) for i in range(seq_len)}
            return result, token_masses

    def get_per_head_chunk_attention(
        self,
        result: AttentionCaptureResult,
        chunk_boundaries: List[Tuple[int, int]],
        query_token_idx: int = -1,
        layer_idx: int = -1,
    ) -> Dict[int, List[float]]:
        """
        Get per-head attention mass for each chunk (for per-head policies like SnapKV).

        Args:
            result: Previously captured attention
            chunk_boundaries: Chunk (start, end) list
            query_token_idx: Which query token to use (-1 = last)
            layer_idx: Which layer (-1 = last)

        Returns:
            chunk_id -> [head_0_mass, head_1_mass, ...]
        """
        if layer_idx == -1:
            layer_idx = max(result.layer_attention.keys()) if result.layer_attention else 0

        attn = result.layer_attention.get(layer_idx)
        if attn is None:
            return {}

        seq_len = attn.shape[-1]
        if query_token_idx == -1:
            query_token_idx = attn.shape[2] - 1

        # attn: [batch, heads, q_len, kv_len]
        q_attn = attn[0, :, query_token_idx, :seq_len]  # [heads, kv_len]
        num_heads = q_attn.shape[0]

        chunk_head_masses = {}
        for chunk_idx, (c_start, c_end) in enumerate(chunk_boundaries):
            c_end = min(c_end, seq_len)
            if c_start < c_end:
                per_head = [float(q_attn[h, c_start:c_end].sum()) for h in range(num_heads)]
            else:
                per_head = [0.0] * num_heads
            chunk_head_masses[chunk_idx] = per_head

        return chunk_head_masses
