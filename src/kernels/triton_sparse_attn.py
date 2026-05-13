"""
Triton Block-Sparse Attention Kernel for ProSE.

This is the core hardware contribution for HPCA:
- Custom CUDA kernel via Triton for sparse attention
- Block-sparse mask (aligned to retention policy chunks)
- Optimized for decode-phase (single query token)

The kernel supports:
1. Variable block sizes (matching chunk sizes)
2. Sparse mask specified as list of active blocks
3. Efficient memory access patterns for HBM
4. Compute/memory overlap via pipelining
"""

import torch
import torch.nn.functional as F
import logging
from typing import Optional, List, Tuple, Dict
from functools import lru_cache

logger = logging.getLogger(__name__)

# Try to import Triton
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    logger.warning("Triton not available. Using fallback PyTorch implementation.")


# Kernel configuration
DEFAULT_BLOCK_SIZE = 64
DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 2


def is_triton_available() -> bool:
    """Check if Triton is available."""
    return TRITON_AVAILABLE


# =============================================================================
# Triton Kernels
# =============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _block_sparse_attn_fwd_kernel(
        # Input pointers
        Q, K, V,  # [batch, heads, seq_len, head_dim]
        Out,  # [batch, heads, seq_len, head_dim]
        # Sparse mask info
        BlockMask,  # [num_blocks] bool mask for which blocks are active
        BlockIndices,  # [num_active_blocks] indices of active blocks
        NumActiveBlocks,  # scalar
        # Strides
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        # Dimensions
        BATCH, HEADS, SEQ_LEN, HEAD_DIM,
        BLOCK_SIZE: tl.constexpr,
        NUM_WARPS: tl.constexpr,
    ):
        """
        Forward pass of block-sparse attention.
        
        Each kernel instance handles one (batch, head, query_block).
        Only computes attention for active blocks (specified by BlockMask).
        """
        # Get program IDs
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        q_block_idx = tl.program_id(2)
        
        # Compute query position
        q_start = q_block_idx * BLOCK_SIZE
        
        # Initialize accumulators for online softmax
        m_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float('inf')  # max
        l_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  # sum
        acc = tl.zeros([BLOCK_SIZE, HEAD_DIM], dtype=tl.float32)
        
        # Load query block
        q_offs_m = q_start + tl.arange(0, BLOCK_SIZE)
        q_offs_k = tl.arange(0, HEAD_DIM)
        q_ptrs = Q + (
            batch_idx * stride_qb + 
            head_idx * stride_qh + 
            q_offs_m[:, None] * stride_qm + 
            q_offs_k[None, :] * stride_qk
        )
        
        # Mask for valid query positions
        q_mask = q_offs_m < SEQ_LEN
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        
        # Iterate over active KV blocks only
        num_active = tl.load(NumActiveBlocks)
        
        for block_idx_idx in range(0, num_active):
            # Get KV block index
            kv_block_idx = tl.load(BlockIndices + block_idx_idx)
            kv_start = kv_block_idx * BLOCK_SIZE
            
            # Check if this block should be attended to
            # In causal attention, kv_block_idx <= q_block_idx
            # For sparse attention, we also check BlockMask
            should_compute = kv_start <= q_start
            
            if should_compute:
                # Load K block
                k_offs_n = kv_start + tl.arange(0, BLOCK_SIZE)
                k_offs_k = tl.arange(0, HEAD_DIM)
                k_ptrs = K + (
                    batch_idx * stride_kb +
                    head_idx * stride_kh +
                    k_offs_n[None, :] * stride_kn +
                    k_offs_k[:, None] * stride_kk
                )
                
                k_mask = k_offs_n < SEQ_LEN
                k = tl.load(k_ptrs, mask=k_mask[None, :], other=0.0)
                
                # Compute QK^T
                qk = tl.dot(q, k)  # [BLOCK_SIZE, BLOCK_SIZE]
                
                # Apply causal mask within block
                offs_m = tl.arange(0, BLOCK_SIZE)
                offs_n = tl.arange(0, BLOCK_SIZE)
                causal_mask = offs_m[:, None] >= (kv_start - q_start + offs_n[None, :])
                qk = tl.where(causal_mask, qk, float('-inf'))
                
                # Compute softmax
                m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
                p = tl.exp(qk - m_ij[:, None])
                l_ij = tl.sum(p, axis=1)
                
                # Update running statistics
                alpha = tl.exp(m_i - m_ij)
                l_i = l_i * alpha + l_ij
                
                # Load V block
                v_ptrs = V + (
                    batch_idx * stride_vb +
                    head_idx * stride_vh +
                    k_offs_n[:, None] * stride_vn +
                    k_offs_k[None, :] * stride_vk
                )
                v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
                
                # Update accumulator
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                m_i = m_ij
        
        # Normalize
        acc = acc / l_i[:, None]
        
        # Store output
        o_ptrs = Out + (
            batch_idx * stride_ob +
            head_idx * stride_oh +
            q_offs_m[:, None] * stride_om +
            q_offs_k[None, :] * stride_ok
        )
        tl.store(o_ptrs, acc, mask=q_mask[:, None])


    @triton.jit
    def _decode_sparse_attn_fwd_kernel(
        # Input pointers (single query token)
        Q,  # [batch, heads, 1, head_dim]
        K, V,  # [batch, heads, seq_len, head_dim]
        Out,  # [batch, heads, 1, head_dim]
        # Block mask
        BlockMask,  # [num_blocks]
        BlockIndices,  # [num_active_blocks]
        NumActiveBlocks,
        # Strides
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        # Dimensions
        BATCH, HEADS, SEQ_LEN, HEAD_DIM,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Optimized kernel for decode phase (single query token).
        
        This is much faster than the full kernel for generation
        since we only have one query position.
        """
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        
        # Load single query
        q_offs_k = tl.arange(0, HEAD_DIM)
        q_ptrs = Q + (
            batch_idx * stride_qb +
            head_idx * stride_qh +
            q_offs_k[None, :]
        )
        q = tl.load(q_ptrs)  # [1, HEAD_DIM]
        
        # Initialize accumulators
        m_i = -float('inf')
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        
        # Iterate over active blocks
        num_active = tl.load(NumActiveBlocks)
        
        for block_idx_idx in range(0, num_active):
            kv_block_idx = tl.load(BlockIndices + block_idx_idx)
            kv_start = kv_block_idx * BLOCK_SIZE
            
            # Load K block
            k_offs_n = kv_start + tl.arange(0, BLOCK_SIZE)
            k_offs_k = tl.arange(0, HEAD_DIM)
            k_ptrs = K + (
                batch_idx * stride_kb +
                head_idx * stride_kh +
                k_offs_n[None, :] * stride_kn +
                k_offs_k[:, None] * stride_kk
            )
            
            k_mask = k_offs_n < SEQ_LEN
            k = tl.load(k_ptrs, mask=k_mask[None, :], other=0.0)
            
            # Compute attention scores
            qk = tl.dot(q, k)  # [1, BLOCK_SIZE]
            qk = tl.where(k_mask, qk, float('-inf'))
            
            # Softmax
            m_ij = tl.maximum(m_i, tl.max(qk))
            p = tl.exp(qk - m_ij)
            l_ij = tl.sum(p)
            
            # Update
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            
            # Load V
            v_ptrs = V + (
                batch_idx * stride_vb +
                head_idx * stride_vh +
                k_offs_n[:, None] * stride_vn +
                k_offs_k[None, :] * stride_vk
            )
            v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
            
            acc = acc * alpha + tl.dot(p.to(v.dtype), v)
            m_i = m_ij
        
        # Normalize and store
        acc = acc / l_i
        
        o_ptrs = Out + (
            batch_idx * stride_ob +
            head_idx * stride_oh +
            q_offs_k[None, :]
        )
        tl.store(o_ptrs, acc[None, :])


# =============================================================================
# PyTorch Interface
# =============================================================================

class BlockSparseAttention:
    """
    Block-sparse attention with Triton kernel acceleration.
    
    This is the main interface for using sparse attention in ProSE.
    It handles:
    1. Converting chunk-level retention policy to block mask
    2. Launching optimized Triton kernels
    3. Fallback to PyTorch if Triton unavailable
    """
    
    def __init__(
        self,
        block_size: int = 64,
        use_triton: bool = True,
    ):
        """
        Initialize block-sparse attention.
        
        Args:
            block_size: Block size for sparsity (should match chunk size)
            use_triton: Whether to use Triton kernels if available
        """
        self.block_size = block_size
        self.use_triton = use_triton and TRITON_AVAILABLE
        
        if self.use_triton:
            logger.info(f"BlockSparseAttention using Triton (block_size={block_size})")
        else:
            logger.info("BlockSparseAttention using PyTorch fallback")
    
    def create_block_mask(
        self,
        seq_len: int,
        active_positions: List[int],
    ) -> torch.Tensor:
        """
        Create block mask from active token positions.
        
        Args:
            seq_len: Total sequence length
            active_positions: List of token positions that should be attended to
            
        Returns:
            Block mask [num_blocks] (True = block is active)
        """
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        block_mask = torch.zeros(num_blocks, dtype=torch.bool, device="cuda")
        
        for pos in active_positions:
            block_idx = pos // self.block_size
            if block_idx < num_blocks:
                block_mask[block_idx] = True
        
        return block_mask
    
    def forward(
        self,
        query: torch.Tensor,  # [batch, heads, q_len, head_dim]
        key: torch.Tensor,    # [batch, heads, kv_len, head_dim]
        value: torch.Tensor,  # [batch, heads, kv_len, head_dim]
        block_mask: torch.Tensor,  # [num_blocks] bool
        is_decode: bool = False,  # True if q_len == 1
    ) -> torch.Tensor:
        """
        Forward pass of block-sparse attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            block_mask: Which blocks are active
            is_decode: Whether this is decode phase (single query)
            
        Returns:
            Output tensor [batch, heads, q_len, head_dim]
        """
        if self.use_triton and query.is_cuda:
            return self._forward_triton(query, key, value, block_mask, is_decode)
        else:
            return self._forward_pytorch(query, key, value, block_mask)
    
    def _forward_triton(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_mask: torch.Tensor,
        is_decode: bool,
    ) -> torch.Tensor:
        """Triton kernel forward pass."""
        batch, heads, q_len, head_dim = query.shape
        kv_len = key.shape[2]
        
        # Get active block indices
        active_blocks = torch.where(block_mask)[0].to(torch.int32)
        num_active = torch.tensor([len(active_blocks)], dtype=torch.int32, device="cuda")
        
        # Output tensor
        output = torch.empty_like(query)
        
        if is_decode and q_len == 1:
            # Use optimized decode kernel
            grid = (batch, heads)
            
            # Pad head_dim to nearest power of 2 for Triton
            HEAD_DIM_PAD = 2 ** ((head_dim - 1).bit_length())
            
            _decode_sparse_attn_fwd_kernel[grid](
                query, key, value, output,
                block_mask, active_blocks, num_active,
                query.stride(0), query.stride(1), query.stride(2), query.stride(3),
                key.stride(0), key.stride(1), key.stride(2), key.stride(3),
                value.stride(0), value.stride(1), value.stride(2), value.stride(3),
                output.stride(0), output.stride(1), output.stride(2), output.stride(3),
                batch, heads, kv_len, head_dim,
                BLOCK_SIZE=self.block_size,
                num_warps=DEFAULT_NUM_WARPS,
            )
        else:
            # Use full kernel
            num_q_blocks = (q_len + self.block_size - 1) // self.block_size
            grid = (batch, heads, num_q_blocks)
            
            _block_sparse_attn_fwd_kernel[grid](
                query, key, value, output,
                block_mask, active_blocks, num_active,
                query.stride(0), query.stride(1), query.stride(2), query.stride(3),
                key.stride(0), key.stride(1), key.stride(2), key.stride(3),
                value.stride(0), value.stride(1), value.stride(2), value.stride(3),
                output.stride(0), output.stride(1), output.stride(2), output.stride(3),
                batch, heads, kv_len, head_dim,
                BLOCK_SIZE=self.block_size,
                NUM_WARPS=DEFAULT_NUM_WARPS,
            )
        
        return output
    
    def _forward_pytorch(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_mask: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch fallback forward pass."""
        batch, heads, q_len, head_dim = query.shape
        kv_len = key.shape[2]
        
        # Expand block mask to token mask
        num_blocks = len(block_mask)
        token_mask = torch.zeros(kv_len, dtype=torch.bool, device=query.device)
        
        for block_idx in range(num_blocks):
            if block_mask[block_idx]:
                start = block_idx * self.block_size
                end = min(start + self.block_size, kv_len)
                token_mask[start:end] = True
        
        # Apply mask to K and V
        key_masked = key[:, :, token_mask, :]
        value_masked = value[:, :, token_mask, :]
        
        # Standard attention on masked KV
        scale = head_dim ** -0.5
        scores = torch.matmul(query, key_masked.transpose(-2, -1)) * scale
        
        # Causal mask
        if q_len > 1:
            causal_mask = torch.triu(
                torch.ones(q_len, key_masked.shape[2], device=query.device),
                diagonal=1,
            ).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, value_masked)
        
        return output


class BlockMaskBuilder:
    """
    Builder for creating block masks from retention policies.
    
    Converts ProSE's chunk-level retention decisions into
    block masks suitable for the Triton kernel.
    """
    
    def __init__(self, block_size: int = 64):
        """
        Initialize builder.
        
        Args:
            block_size: Block size (should match chunk size)
        """
        self.block_size = block_size
    
    def from_retention_policy(
        self,
        seq_len: int,
        anchor_chunk_ids: List[int],
        promoted_chunk_ids: List[int],
        chunk_boundaries: List[Tuple[int, int]],
        query_len: int = 1,
    ) -> torch.Tensor:
        """
        Create block mask from ProSE retention policy.
        
        Args:
            seq_len: Total sequence length
            anchor_chunk_ids: Chunks kept as anchors
            promoted_chunk_ids: Chunks promoted from tail
            chunk_boundaries: (start, end) for each chunk
            query_len: Length of query portion (always kept)
            
        Returns:
            Block mask [num_blocks]
        """
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        mask = torch.zeros(num_blocks, dtype=torch.bool, device="cuda")
        
        # Active chunks = anchors + promoted
        active_chunks = set(anchor_chunk_ids) | set(promoted_chunk_ids)
        
        # Mark blocks for active chunks
        for chunk_id in active_chunks:
            if chunk_id < len(chunk_boundaries):
                start, end = chunk_boundaries[chunk_id]
                start_block = start // self.block_size
                end_block = (end + self.block_size - 1) // self.block_size
                mask[start_block:end_block] = True
        
        # Always include query portion (last blocks)
        if query_len > 0:
            query_start = seq_len - query_len
            query_start_block = query_start // self.block_size
            mask[query_start_block:] = True
        
        return mask
    
    def from_two_tier_state(
        self,
        seq_len: int,
        memory_manager,
        query_len: int = 1,
    ) -> torch.Tensor:
        """
        Create block mask from TwoTierMemoryManager state.
        
        Args:
            seq_len: Sequence length
            memory_manager: TwoTierMemoryManager instance
            query_len: Query length
            
        Returns:
            Block mask
        """
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        mask = torch.zeros(num_blocks, dtype=torch.bool, device="cuda")
        
        # Only HBM-resident chunks are visible to attention
        for chunk_id, tier in memory_manager.chunk_tiers.items():
            if tier.name == "HBM" and chunk_id in memory_manager.chunk_positions:
                start, end = memory_manager.chunk_positions[chunk_id]
                start_block = start // self.block_size
                end_block = (end + self.block_size - 1) // self.block_size
                mask[start_block:end_block] = True
        
        # Include query portion
        if query_len > 0:
            query_start = seq_len - query_len
            query_start_block = query_start // self.block_size
            mask[query_start_block:] = True
        
        return mask
    
    def compute_sparsity(self, block_mask: torch.Tensor) -> float:
        """
        Compute sparsity ratio from block mask.
        
        Returns:
            Fraction of blocks that are inactive (0-1)
        """
        return 1.0 - (block_mask.sum() / len(block_mask)).item()


# =============================================================================
# Performance Benchmarking
# =============================================================================

def benchmark_sparse_attention(
    seq_lens: List[int] = [1024, 4096, 16384, 65536],
    sparsity_levels: List[float] = [0.0, 0.5, 0.9, 0.95],
    head_dim: int = 128,
    num_heads: int = 8,
    num_runs: int = 10,
) -> Dict:
    """
    Benchmark sparse attention performance.
    
    Returns:
        Results dict with timing info
    """
    import time
    
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for seq_len in seq_lens:
        for sparsity in sparsity_levels:
            # Create random data
            batch = 1
            q = torch.randn(batch, num_heads, 1, head_dim, device=device)
            k = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
            v = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
            
            # Create block mask with given sparsity
            num_blocks = (seq_len + 64 - 1) // 64
            num_active = int(num_blocks * (1 - sparsity))
            block_mask = torch.zeros(num_blocks, dtype=torch.bool, device=device)
            block_mask[:num_active] = True
            
            # Initialize sparse attention
            sparse_attn = BlockSparseAttention(block_size=64, use_triton=True)
            
            # Warmup
            for _ in range(3):
                _ = sparse_attn.forward(q, k, v, block_mask, is_decode=True)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            for _ in range(num_runs):
                _ = sparse_attn.forward(q, k, v, block_mask, is_decode=True)
                if device == "cuda":
                    torch.cuda.synchronize()
            elapsed = time.time() - start
            
            results.append({
                "seq_len": seq_len,
                "sparsity": sparsity,
                "time_ms": elapsed / num_runs * 1000,
                "tokens_per_sec": seq_len / (elapsed / num_runs),
            })
    
    return {"results": results}


if __name__ == "__main__":
    # Run benchmarks
    print("Benchmarking sparse attention...")
    results = benchmark_sparse_attention()
    
    print("\nResults:")
    print("-" * 60)
    print(f"{'Seq Len':>10} {'Sparsity':>10} {'Time (ms)':>12} {'Tokens/s':>12}")
    print("-" * 60)
    for r in results["results"]:
        print(f"{r['seq_len']:>10} {r['sparsity']:>10.1%} "
              f"{r['time_ms']:>12.3f} {r['tokens_per_sec']:>12.0f}")
