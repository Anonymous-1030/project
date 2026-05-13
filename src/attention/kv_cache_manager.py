"""
Sparse KV Cache Manager with RoPE-correct position tracking.

This is the core infrastructure that enables ProSE's retention policy
to actually affect model output. When tokens are pruned from the KV cache,
their original position IDs must be preserved for RoPE correctness.

Supports:
- Token-level position tracking for RoPE-correct pruning
- Chunk-level tier tracking (anchor/tail/promoted)
- Efficient pruning and restoration operations
- Memory accounting (bytes in HBM vs Host DRAM)
- GQA-aware shapes (num_kv_heads != num_attention_heads)
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class KVTier(str, Enum):
    """Storage tier for KV entries."""
    HBM_ANCHOR = "hbm_anchor"       # Full fidelity, always in HBM
    HBM_PROMOTED = "hbm_promoted"   # Promoted from tail, full fidelity in HBM
    DRAM_TAIL = "dram_tail"         # Compressed, in host DRAM
    EVICTED = "evicted"             # No longer stored


@dataclass
class KVSlice:
    """Metadata for a contiguous slice of KV entries."""
    token_start: int          # Original token position (for RoPE)
    token_end: int
    tier: KVTier
    chunk_id: Optional[str] = None
    logical_bytes: int = 0


@dataclass
class PrunedKVCache:
    """
    A pruned KV cache ready for generation.

    Contains only the retained KV entries with their original position IDs
    preserved for RoPE correctness.
    """
    # Per-layer pruned KV: layer_idx -> (K, V)
    # K shape: [batch, num_kv_heads, retained_len, head_dim]
    # V shape: [batch, num_kv_heads, retained_len, head_dim]
    kv_per_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]]

    # Original position IDs for the retained tokens [batch, retained_len]
    position_ids: torch.Tensor

    # Attention mask for the retained tokens [batch, retained_len]
    attention_mask: torch.Tensor

    # Mapping from pruned index to original token position
    retained_positions: List[int]

    # Memory accounting
    hbm_bytes: int = 0
    total_entries: int = 0
    original_entries: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.original_entries == 0:
            return 1.0
        return self.total_entries / self.original_entries

    @property
    def num_layers(self) -> int:
        return len(self.kv_per_layer)

    def to_transformers_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Convert to HuggingFace transformers past_key_values format."""
        max_layer = max(self.kv_per_layer.keys()) + 1
        result = []
        for i in range(max_layer):
            if i in self.kv_per_layer:
                result.append(self.kv_per_layer[i])
            else:
                # Should not happen in practice
                raise ValueError(f"Missing layer {i} in pruned KV cache")
        return tuple(result)


class SparseKVCacheManager:
    """
    Manages a per-layer sparse KV cache with position tracking.

    This is the bridge between ProSE's retention policy and the actual
    model generation. It takes the full KV cache from prefill and produces
    a pruned cache that only contains retained tokens.

    Key invariant: position IDs in the pruned cache always correspond
    to the original token positions, ensuring RoPE correctness.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Full KV cache from prefill
        self._full_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
        self._seq_len: int = 0

        # Chunk metadata
        self._chunk_boundaries: List[Tuple[int, int]] = []  # (start, end) per chunk
        self._chunk_tiers: Dict[int, KVTier] = {}

    def load_from_prefill(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ) -> None:
        """
        Load full KV cache from prefill pass.

        Args:
            past_key_values: HuggingFace format (layer_0_kv, layer_1_kv, ...)
                where each is (K, V) with shape [batch, heads, seq_len, head_dim]
        """
        self._full_kv = past_key_values
        if past_key_values and len(past_key_values) > 0:
            self._seq_len = past_key_values[0][0].shape[2]
        logger.debug(f"Loaded KV cache: {self.num_layers} layers, seq_len={self._seq_len}")

    def set_chunk_boundaries(
        self,
        boundaries: List[Tuple[int, int]],
    ) -> None:
        """Set chunk boundaries for chunk-level operations."""
        self._chunk_boundaries = boundaries

    def prune_by_token_positions(
        self,
        retained_positions: List[int],
        device: Optional[torch.device] = None,
    ) -> PrunedKVCache:
        """
        Create a pruned KV cache containing only the specified token positions.

        This is the fundamental operation. The retained positions keep their
        original position IDs for RoPE correctness.

        Args:
            retained_positions: Sorted list of original token positions to keep
            device: Target device for the pruned cache

        Returns:
            PrunedKVCache ready for generation
        """
        if self._full_kv is None:
            raise RuntimeError("No KV cache loaded. Call load_from_prefill first.")

        if not retained_positions:
            raise ValueError("retained_positions cannot be empty")

        retained_positions = sorted(set(retained_positions))
        pos_tensor = torch.tensor(retained_positions, dtype=torch.long)

        kv_per_layer = {}
        hbm_bytes = 0

        for layer_idx in range(len(self._full_kv)):
            full_k, full_v = self._full_kv[layer_idx]
            # full_k shape: [batch, num_kv_heads, seq_len, head_dim]

            target_device = device or full_k.device

            # Index select on the sequence dimension
            pruned_k = full_k[:, :, pos_tensor, :].to(target_device)
            pruned_v = full_v[:, :, pos_tensor, :].to(target_device)

            kv_per_layer[layer_idx] = (pruned_k, pruned_v)
            hbm_bytes += pruned_k.nelement() * pruned_k.element_size()
            hbm_bytes += pruned_v.nelement() * pruned_v.element_size()

        # Build position IDs (original positions, not sequential)
        position_ids = torch.tensor(
            [retained_positions], dtype=torch.long,
            device=device or self._full_kv[0][0].device,
        )

        # Build attention mask (all ones for retained positions)
        attention_mask = torch.ones(
            1, len(retained_positions), dtype=torch.long,
            device=device or self._full_kv[0][0].device,
        )

        return PrunedKVCache(
            kv_per_layer=kv_per_layer,
            position_ids=position_ids,
            attention_mask=attention_mask,
            retained_positions=retained_positions,
            hbm_bytes=hbm_bytes,
            total_entries=len(retained_positions),
            original_entries=self._seq_len,
        )

    def prune_by_chunks(
        self,
        anchor_chunk_ids: List[int],
        promoted_chunk_ids: Optional[List[int]] = None,
        include_query_tokens: Optional[Tuple[int, int]] = None,
        device: Optional[torch.device] = None,
    ) -> PrunedKVCache:
        """
        Create a pruned KV cache by specifying which chunks to retain.

        Args:
            anchor_chunk_ids: Chunk indices to keep as anchors
            promoted_chunk_ids: Chunk indices promoted from tail
            include_query_tokens: (start, end) range of query tokens to always include
            device: Target device

        Returns:
            PrunedKVCache
        """
        if not self._chunk_boundaries:
            raise RuntimeError("Chunk boundaries not set. Call set_chunk_boundaries first.")

        retained_positions: Set[int] = set()

        # Add anchor chunks
        for chunk_id in anchor_chunk_ids:
            if 0 <= chunk_id < len(self._chunk_boundaries):
                start, end = self._chunk_boundaries[chunk_id]
                retained_positions.update(range(start, min(end, self._seq_len)))

        # Add promoted chunks
        if promoted_chunk_ids:
            for chunk_id in promoted_chunk_ids:
                if 0 <= chunk_id < len(self._chunk_boundaries):
                    start, end = self._chunk_boundaries[chunk_id]
                    retained_positions.update(range(start, min(end, self._seq_len)))

        # Always include query tokens
        if include_query_tokens:
            q_start, q_end = include_query_tokens
            retained_positions.update(range(q_start, min(q_end, self._seq_len)))

        return self.prune_by_token_positions(
            sorted(retained_positions), device=device,
        )

    def get_chunk_token_positions(self, chunk_ids: List[int]) -> List[int]:
        """Convert chunk IDs to token positions."""
        positions = []
        for chunk_id in chunk_ids:
            if 0 <= chunk_id < len(self._chunk_boundaries):
                start, end = self._chunk_boundaries[chunk_id]
                positions.extend(range(start, min(end, self._seq_len)))
        return sorted(set(positions))

    def compute_memory_breakdown(
        self,
        anchor_chunk_ids: List[int],
        promoted_chunk_ids: List[int],
        tail_chunk_ids: List[int],
    ) -> Dict[str, Any]:
        """
        Compute memory breakdown by tier.

        Returns bytes used in HBM (anchors + promoted) vs DRAM (tail).
        """
        bytes_per_entry = 2 * self.num_kv_heads * self.head_dim * self.num_layers
        # Factor of 2 for K + V, multiply by dtype size
        element_size = 2  # float16
        bytes_per_token = bytes_per_entry * element_size

        anchor_tokens = sum(
            min(end, self._seq_len) - start
            for cid in anchor_chunk_ids
            if 0 <= cid < len(self._chunk_boundaries)
            for start, end in [self._chunk_boundaries[cid]]
        )
        promoted_tokens = sum(
            min(end, self._seq_len) - start
            for cid in promoted_chunk_ids
            if 0 <= cid < len(self._chunk_boundaries)
            for start, end in [self._chunk_boundaries[cid]]
        )
        tail_tokens = sum(
            min(end, self._seq_len) - start
            for cid in tail_chunk_ids
            if 0 <= cid < len(self._chunk_boundaries)
            for start, end in [self._chunk_boundaries[cid]]
        )

        return {
            "hbm_anchor_bytes": anchor_tokens * bytes_per_token,
            "hbm_promoted_bytes": promoted_tokens * bytes_per_token,
            "hbm_total_bytes": (anchor_tokens + promoted_tokens) * bytes_per_token,
            "dram_tail_bytes": tail_tokens * bytes_per_token,
            "total_bytes": (anchor_tokens + promoted_tokens + tail_tokens) * bytes_per_token,
            "anchor_tokens": anchor_tokens,
            "promoted_tokens": promoted_tokens,
            "tail_tokens": tail_tokens,
            "total_tokens": self._seq_len,
            "hbm_ratio": (anchor_tokens + promoted_tokens) / max(self._seq_len, 1),
        }

    def release_full_kv(self) -> None:
        """Release the full KV cache to free memory."""
        self._full_kv = None
        self._seq_len = 0
