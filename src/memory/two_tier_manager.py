"""
Two-Tier Memory Model for HBM + Host DRAM/CXL.

This is the core hardware contribution for HPCA:
- Tier 1 (HBM): Hot KV entries (anchors + promoted) at full precision
- Tier 2 (DRAM/CXL): Cold KV entries (tail) compressed

Key features:
- Bandwidth-aware promotion (models PCIe/CXL bandwidth)
- Async prefetch with compute overlap
- Memory capacity modeling
"""

import torch
import logging
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


@contextmanager
def _nullcontext():
    """No-op context manager for Python < 3.7 compatibility."""
    yield


logger = logging.getLogger(__name__)


# Bandwidth constants (GB/s)
DEFAULT_HBM_BANDWIDTH = 3350  # H100 HBM3
DEFAULT_PCIE_BANDWIDTH = 32   # PCIe Gen5 x16
DEFAULT_CXL_BANDWIDTH = 64    # CXL 3.0


class MemoryTier(Enum):
    """Memory tier enum."""
    HBM = "hbm"           # GPU high bandwidth memory
    HOST_DRAM = "dram"    # Host DRAM via PCIe/CXL
    CXL = "cxl"           # CXL-attached memory


@dataclass
class BandwidthSpec:
    """Bandwidth specification for a memory system."""
    hbm_bw_gbps: float = DEFAULT_HBM_BANDWIDTH
    pcie_bw_gbps: float = DEFAULT_PCIE_BANDWIDTH
    cxl_bw_gbps: float = DEFAULT_CXL_BANDWIDTH
    
    # Latency in microseconds
    hbm_latency_us: float = 0.1
    pcie_latency_us: float = 5.0
    cxl_latency_us: float = 0.5
    
    def promotion_time_us(self, bytes_to_promote: int, via: str = "pcie") -> float:
        """
        Calculate time to promote data from host to HBM.
        
        Args:
            bytes_to_promote: Number of bytes to promote
            via: "pcie" or "cxl"
            
        Returns:
            Time in microseconds
        """
        gb = bytes_to_promote / (1024 ** 3)
        
        if via == "pcie":
            transfer_time = gb / self.pcie_bw_gbps * 1e6  # Convert to us
            latency = self.pcie_latency_us
        elif via == "cxl":
            transfer_time = gb / self.cxl_bw_gbps * 1e6
            latency = self.cxl_latency_us
        else:
            raise ValueError(f"Unknown via: {via}")
        
        return latency + transfer_time


@dataclass
class TieredKVEntry:
    """A single KV entry in the two-tier system."""
    layer_idx: int
    head_idx: int
    position: int  # Original token position for RoPE
    
    # Data (may be in HBM or host)
    key: torch.Tensor  # [head_dim]
    value: torch.Tensor  # [head_dim]
    
    # Tier location
    tier: MemoryTier = MemoryTier.HBM
    
    # Compression info
    is_compressed: bool = False
    compression_ratio: float = 1.0
    
    # Access tracking
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    
    def access(self):
        """Record an access to this entry."""
        self.last_access_time = time.time()
        self.access_count += 1


@dataclass
class PromotionRequest:
    """A request to promote chunks from DRAM to HBM."""
    chunk_ids: List[int]
    priority: float = 1.0  # Higher = more urgent
    request_time: float = field(default_factory=time.time)
    
    # Estimated compute time for current step (for overlap planning)
    estimated_compute_us: float = 100.0


@dataclass
class PromotionResult:
    """Result of a promotion operation."""
    promoted_chunks: List[int]
    failed_chunks: List[int]
    promotion_time_us: float = 0.0
    bytes_promoted: int = 0
    overlapped: bool = False  # Whether promotion overlapped with compute
    transfer_path: str = "pcie"
    prefetch_depth: int = 0
    estimated_hidden_us: float = 0.0


@dataclass
class PrefetchDecision:
    """A protocol-level decision for promotion path and prefetch depth."""
    transfer_path: str
    prefetch_depth: int
    expected_transfer_us: float
    expected_overlap_us: float
    expected_exposed_us: float
    chunk_ids: List[int] = field(default_factory=list)


class CompressedTailStore:
    """
    Compressed storage for tail KV entries in host DRAM.
    
    Supports multiple compression schemes:
    - int8/int4 quantization
    - Centroid-based (vector quantization)
    - Sparse representation (store only non-zero entries)
    """
    
    def __init__(
        self,
        compression: str = "int8",
        device: str = "cpu",
    ):
        """
        Initialize compressed tail store.
        
        Args:
            compression: "none", "int8", "int4", "centroid"
            device: Device for storage (usually "cpu")
        """
        self.compression = compression
        self.device = device
        
        # Chunk storage: chunk_id -> compressed data
        self._chunks: Dict[int, Dict[str, Any]] = {}
        
        # Compression ratio achieved
        self._compression_ratio = 1.0 if compression == "none" else 0.25
        
        logger.info(f"CompressedTailStore: compression={compression}")
    
    def store_chunk(
        self,
        chunk_id: int,
        k_tensor: torch.Tensor,  # [num_layers, num_heads, chunk_len, head_dim]
        v_tensor: torch.Tensor,
    ) -> int:
        """
        Store a chunk in compressed form.
        
        Returns:
            Bytes stored
        """
        chunk_record: Dict[str, Any] = {
            "shape": k_tensor.shape,
            "compression": self.compression,
        }

        if self.compression == "none":
            stored_k = k_tensor.to(self.device)
            stored_v = v_tensor.to(self.device)
            bytes_stored = (stored_k.numel() + stored_v.numel()) * 2  # fp16
            chunk_record.update({"k": stored_k, "v": stored_v})
            
        elif self.compression == "int8":
            # Simple per-channel quantization
            stored_k, k_scale = self._quantize_int8(k_tensor)
            stored_v, v_scale = self._quantize_int8(v_tensor)
            bytes_stored = (stored_k.numel() + stored_v.numel()) * 1  # int8
            bytes_stored += (k_scale.numel() + v_scale.numel()) * 4  # fp32 scales
            chunk_record.update({
                "k": stored_k,
                "v": stored_v,
                "k_scale": k_scale,
                "v_scale": v_scale,
            })
            
        elif self.compression == "int4":
            # 4-bit quantization (pack 2 values per byte)
            stored_k, k_scale = self._quantize_int4(k_tensor)
            stored_v, v_scale = self._quantize_int4(v_tensor)
            bytes_stored = (stored_k.numel() + stored_v.numel()) // 2
            bytes_stored += (k_scale.numel() + v_scale.numel()) * 4
            chunk_record.update({
                "k": stored_k,
                "v": stored_v,
                "k_scale": k_scale,
                "v_scale": v_scale,
            })
            
        elif self.compression == "centroid":
            # K-means vector quantization
            stored_k = self._vector_quantize(k_tensor)
            stored_v = self._vector_quantize(v_tensor)
            bytes_stored = stored_k["indices"].numel() * 2  # 16-bit indices
            bytes_stored += stored_k["centroids"].numel() * 2  # fp16 centroids
            bytes_stored += stored_v["indices"].numel() * 2
            bytes_stored += stored_v["centroids"].numel() * 2
            chunk_record.update({"k": stored_k, "v": stored_v})
        else:
            raise ValueError(f"Unknown compression: {self.compression}")

        self._chunks[chunk_id] = chunk_record
        
        return bytes_stored
    
    def retrieve_chunk(
        self,
        chunk_id: int,
        target_device: str = "cuda",
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve and decompress a chunk.
        
        Returns:
            (k_tensor, v_tensor) or None if not found
        """
        if chunk_id not in self._chunks:
            return None
        
        chunk = self._chunks[chunk_id]
        
        if self.compression == "none":
            k = chunk["k"].to(target_device)
            v = chunk["v"].to(target_device)
            
        elif self.compression in ("int8", "int4"):
            k = self._dequantize(chunk["k"], chunk["k_scale"], target_device)
            v = self._dequantize(chunk["v"], chunk["v_scale"], target_device)
            
        elif self.compression == "centroid":
            k = self._dequantize_vectors(chunk["k"], target_device)
            v = self._dequantize_vectors(chunk["v"], target_device)
        
        return k, v
    
    def _quantize_int8(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to int8 with per-channel scaling."""
        # Compute per-channel (head) scales
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        scale = (x_max - x_min) / 255.0
        
        # Quantize
        x_quant = ((x - x_min) / scale).to(torch.uint8)
        return x_quant, scale
    
    def _quantize_int4(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to int4 (pack 2 values per byte)."""
        # Similar to int8 but with 4 bits
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        scale = (x_max - x_min) / 15.0
        
        x_quant = ((x - x_min) / scale).to(torch.uint8)
        # Pack two 4-bit values into one byte
        x_packed = (x_quant[:, :, :, ::2] << 4) | x_quant[:, :, :, 1::2]
        return x_packed, scale
    
    def _vector_quantize(self, x: torch.Tensor) -> Dict:
        """
        Vector quantization using k-means on head-dimension vectors.

        Groups vectors into clusters via mini-batch k-means, stores
        cluster indices (int16) + centroid codebook (fp16).
        Reconstruction: lookup centroids by index.

        Compression ratio ≈ head_dim * 2 bytes / (2 bytes index + codebook amortized)
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1]).float()  # [N, D]
        n_vectors = x_flat.shape[0]
        dim = x_flat.shape[-1]

        n_centroids = min(256, max(16, n_vectors // 4))
        device = x_flat.device

        # Mini-batch k-means (3 iterations suffice for KV cache vectors)
        # Initialize centroids via k-means++ style: random sample
        perm = torch.randperm(n_vectors, device=device)[:n_centroids]
        centroids = x_flat[perm].clone()  # [K, D]

        for _ in range(3):
            # Assignment: find nearest centroid for each vector
            # Compute distances: ||x - c||^2 = ||x||^2 - 2*x@c^T + ||c||^2
            x_sq = (x_flat ** 2).sum(dim=1, keepdim=True)      # [N, 1]
            c_sq = (centroids ** 2).sum(dim=1, keepdim=True).T  # [1, K]
            dists = x_sq - 2 * x_flat @ centroids.T + c_sq     # [N, K]
            indices = dists.argmin(dim=1)                        # [N]

            # Update: recompute centroids as cluster means
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(n_centroids, device=device)
            new_centroids.index_add_(0, indices, x_flat)
            counts.index_add_(0, indices, torch.ones(n_vectors, device=device))
            mask = counts > 0
            new_centroids[mask] /= counts[mask].unsqueeze(1)
            # Keep old centroids for empty clusters
            new_centroids[~mask] = centroids[~mask]
            centroids = new_centroids

        # Final assignment
        x_sq = (x_flat ** 2).sum(dim=1, keepdim=True)
        c_sq = (centroids ** 2).sum(dim=1, keepdim=True).T
        dists = x_sq - 2 * x_flat @ centroids.T + c_sq
        indices = dists.argmin(dim=1).to(torch.int16)

        return {
            "indices": indices.cpu(),
            "centroids": centroids.to(x.dtype).cpu(),
            "shape": orig_shape,
        }
    
    def _dequantize(self, x_quant, scale, device):
        """Dequantize int8/int4 to fp16."""
        return x_quant.to(device).float() * scale.to(device)
    
    def _dequantize_vectors(self, vq_data: Dict, device: str) -> torch.Tensor:
        """Dequantize vector quantized data."""
        indices = vq_data["indices"].to(device)
        centroids = vq_data["centroids"].to(device)
        shape = vq_data["shape"]
        
        # Lookup centroids
        x = centroids[indices]
        return x.reshape(shape)


class TwoTierMemoryManager:
    """
    Manager for two-tier KV memory system.
    
    This is the core hardware contribution for HPCA submission:
    - Models HBM capacity and bandwidth
    - Models PCIe/CXL promotion bandwidth
    - Schedules async promotions to overlap with compute
    """
    
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        hbm_capacity_gb: float = 40.0,
        dram_capacity_gb: float = 512.0,
        bandwidth_spec: Optional[BandwidthSpec] = None,
        tail_compression: str = "int8",
        device: str = "cuda",
    ):
        """
        Initialize two-tier memory manager.
        
        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV heads (for GQA)
            head_dim: Dimension per head
            hbm_capacity_gb: HBM capacity in GB
            dram_capacity_gb: Host DRAM capacity in GB
            bandwidth_spec: Bandwidth specification
            tail_compression: Compression for tail entries
            device: Primary device (HBM)
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hbm_capacity_bytes = int(hbm_capacity_gb * 1024 ** 3)
        self.dram_capacity_bytes = int(dram_capacity_gb * 1024 ** 3)
        self.bandwidth = bandwidth_spec or BandwidthSpec()
        self.device = device
        
        # Bytes per token
        self.bytes_per_token = (
            2 * num_layers * num_kv_heads * head_dim * 2  # fp16, K+V
        )
        
        # HBM-resident KV cache (anchors + promoted)
        self.hbm_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # chunk_id -> (K, V) where K/V shape: [layers, heads, tokens, dim]
        
        # DRAM-resident compressed tail
        self.tail_store = CompressedTailStore(compression=tail_compression)
        
        # Chunk metadata
        self.chunk_tiers: Dict[int, MemoryTier] = {}
        self.chunk_positions: Dict[int, Tuple[int, int]] = {}  # chunk_id -> (start, end)
        
        # Promotion queue
        self.promotion_queue: deque[PromotionRequest] = deque()
        
        # Statistics
        self.stats = {
            "hbm_bytes_used": 0,
            "dram_bytes_used": 0,
            "total_promotions": 0,
            "promotion_bytes": 0,
            "promotion_time_us": 0.0,
            "pcie_promotions": 0,
            "cxl_promotions": 0,
            "cxl_prefetch_hits": 0,
            "prefetch_depth_sum": 0,
        }
        
        logger.info(
            f"TwoTierMemoryManager initialized:\n"
            f"  HBM capacity: {hbm_capacity_gb} GB\n"
            f"  DRAM capacity: {dram_capacity_gb} GB\n"
            f"  Bytes per token: {self.bytes_per_token}\n"
            f"  Max tokens in HBM: {self.hbm_capacity_bytes // self.bytes_per_token}"
        )

    def estimate_chunk_transfer_bytes(self, chunk_ids: List[int]) -> int:
        """Estimate transfer bytes for a list of chunks from metadata/positions."""
        total_bytes = 0
        for cid in chunk_ids:
            if cid in self.hbm_kv:
                k, v = self.hbm_kv[cid]
                total_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
            elif cid in self.chunk_positions:
                start, end = self.chunk_positions[cid]
                total_bytes += max(0, end - start) * self.bytes_per_token
        return total_bytes

    def select_transfer_path(
        self,
        bytes_to_promote: int,
        estimated_compute_us: float,
        prefer_cxl_for_prefetch: bool = True,
    ) -> PrefetchDecision:
        """
        Choose PCIe vs CXL and a prefetch depth.

        Heuristic:
        - Prefer the path with lower exposed latency within the decode compute window.
        - Allow deeper prefetch for low-latency CXL when transfer cannot be fully hidden.
        """
        pcie_time = self.bandwidth.promotion_time_us(bytes_to_promote, via="pcie")
        cxl_time = self.bandwidth.promotion_time_us(bytes_to_promote, via="cxl")

        pcie_exposed = max(0.0, pcie_time - estimated_compute_us)
        cxl_exposed = max(0.0, cxl_time - estimated_compute_us)

        if prefer_cxl_for_prefetch and cxl_exposed <= pcie_exposed:
            transfer_path = "cxl"
            expected_transfer = cxl_time
            exposed = cxl_exposed
        else:
            transfer_path = "pcie"
            expected_transfer = pcie_time
            exposed = pcie_exposed

        overlap = min(expected_transfer, estimated_compute_us)
        # Adapt lookahead depth to exposed latency pressure.
        if expected_transfer <= estimated_compute_us:
            depth = 1
        else:
            depth = min(4, max(2, int(round(expected_transfer / max(estimated_compute_us, 1.0)))))

        return PrefetchDecision(
            transfer_path=transfer_path,
            prefetch_depth=depth,
            expected_transfer_us=expected_transfer,
            expected_overlap_us=overlap,
            expected_exposed_us=exposed,
        )
    
    def allocate_chunks(
        self,
        full_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        chunk_boundaries: List[Tuple[int, int]],
        anchor_chunk_ids: List[int],
        promoted_chunk_ids: List[int],
        tail_chunk_ids: List[int],
    ) -> Dict[str, int]:
        """
        Allocate chunks to appropriate memory tiers.
        
        Args:
            full_kv: Full KV cache from prefill
            chunk_boundaries: (start, end) for each chunk
            anchor_chunk_ids: Chunks to keep in HBM as anchors
            promoted_chunk_ids: Chunks to promote to HBM
            tail_chunk_ids: Chunks to store compressed in DRAM
            
        Returns:
            Memory breakdown dict
        """
        # Convert full_kv to per-layer format
        # full_kv: tuple of (K, V) per layer
        # Each K/V: [batch, heads, seq_len, head_dim]
        
        seq_len = full_kv[0][0].shape[2]
        
        # Allocate anchors to HBM
        hbm_bytes = 0
        for cid in anchor_chunk_ids:
            if cid < len(chunk_boundaries):
                start, end = chunk_boundaries[cid]
                self._store_in_hbm(cid, full_kv, start, end)
                self.chunk_tiers[cid] = MemoryTier.HBM
                self.chunk_positions[cid] = (start, end)
                hbm_bytes += (end - start) * self.bytes_per_token
        
        # Allocate promoted to HBM
        for cid in promoted_chunk_ids:
            if cid < len(chunk_boundaries):
                start, end = chunk_boundaries[cid]
                self._store_in_hbm(cid, full_kv, start, end)
                self.chunk_tiers[cid] = MemoryTier.HBM
                self.chunk_positions[cid] = (start, end)
                hbm_bytes += (end - start) * self.bytes_per_token
        
        # Allocate tail to DRAM (compressed)
        dram_bytes = 0
        for cid in tail_chunk_ids:
            if cid < len(chunk_boundaries):
                start, end = chunk_boundaries[cid]
                bytes_stored = self._store_in_dram(cid, full_kv, start, end)
                self.chunk_tiers[cid] = MemoryTier.HOST_DRAM
                self.chunk_positions[cid] = (start, end)
                dram_bytes += bytes_stored
        
        self.stats["hbm_bytes_used"] = hbm_bytes
        self.stats["dram_bytes_used"] = dram_bytes
        
        return {
            "hbm_bytes": hbm_bytes,
            "dram_bytes": dram_bytes,
            "hbm_chunks": len(anchor_chunk_ids) + len(promoted_chunk_ids),
            "dram_chunks": len(tail_chunk_ids),
        }
    
    def _store_in_hbm(
        self,
        chunk_id: int,
        full_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        start: int,
        end: int,
    ):
        """Store a chunk in HBM."""
        # Extract slice from each layer
        k_list = []
        v_list = []
        
        for layer_idx, (k, v) in enumerate(full_kv):
            # k: [batch, heads, seq_len, dim]
            k_slice = k[:, :, start:end, :].clone()
            v_slice = v[:, :, start:end, :].clone()
            k_list.append(k_slice)
            v_list.append(v_slice)
        
        # Stack into single tensor per chunk
        chunk_k = torch.cat(k_list, dim=0)  # [layers, batch, heads, tokens, dim]
        chunk_v = torch.cat(v_list, dim=0)
        
        self.hbm_kv[chunk_id] = (chunk_k, chunk_v)
    
    def _store_in_dram(
        self,
        chunk_id: int,
        full_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        start: int,
        end: int,
    ) -> int:
        """Store a chunk in DRAM (compressed). Returns bytes stored."""
        k_list = []
        v_list = []
        
        for layer_idx, (k, v) in enumerate(full_kv):
            k_slice = k[:, :, start:end, :].clone()
            v_slice = v[:, :, start:end, :].clone()
            k_list.append(k_slice)
            v_list.append(v_slice)
        
        chunk_k = torch.cat(k_list, dim=0)
        chunk_v = torch.cat(v_list, dim=0)
        
        return self.tail_store.store_chunk(chunk_id, chunk_k, chunk_v)
    
    def promote_chunks(
        self,
        chunk_ids: List[int],
        async_prefetch: bool = True,
        estimated_compute_us: float = 100.0,
        preferred_path: Optional[str] = None,
    ) -> PromotionResult:
        """
        Promote chunks from DRAM to HBM.

        Uses CUDA events for accurate GPU-side timing of the actual
        host→device DMA transfer, rather than analytical modeling.
        If async_prefetch is True and a non-default CUDA stream is available,
        the transfer is launched on a secondary stream to overlap with compute.

        Args:
            chunk_ids: Chunks to promote
            async_prefetch: Whether to use async prefetch on a secondary stream
            estimated_compute_us: Estimated compute time (used as fallback
                                  when CUDA is unavailable)

        Returns:
            PromotionResult with measured timing info
        """
        bytes_estimate = self.estimate_chunk_transfer_bytes(chunk_ids)
        decision = self.select_transfer_path(bytes_estimate, estimated_compute_us)
        if preferred_path in {"pcie", "cxl"}:
            decision.transfer_path = preferred_path
            decision.expected_transfer_us = self.bandwidth.promotion_time_us(
                bytes_estimate, via=preferred_path
            )
            decision.expected_overlap_us = min(decision.expected_transfer_us, estimated_compute_us)
            decision.expected_exposed_us = max(0.0, decision.expected_transfer_us - estimated_compute_us)

        result = PromotionResult(
            promoted_chunks=[],
            failed_chunks=[],
            transfer_path=decision.transfer_path,
            prefetch_depth=decision.prefetch_depth,
            estimated_hidden_us=decision.expected_overlap_us,
        )
        total_bytes = 0

        use_cuda = torch.cuda.is_available() and self.device.startswith("cuda")

        # Set up CUDA event timing if available
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Use a secondary stream for async prefetch
            if async_prefetch:
                transfer_stream = torch.cuda.Stream()
            else:
                transfer_stream = None

            start_event.record()

            ctx = torch.cuda.stream(transfer_stream) if transfer_stream else _nullcontext()
            with ctx:
                for cid in chunk_ids:
                    if cid not in self.chunk_tiers:
                        result.failed_chunks.append(cid)
                        continue
                    if self.chunk_tiers[cid] == MemoryTier.HBM:
                        continue

                    kv = self.tail_store.retrieve_chunk(cid, target_device=self.device)
                    if kv is None:
                        result.failed_chunks.append(cid)
                        continue

                    k, v = kv
                    self.hbm_kv[cid] = (k, v)
                    self.chunk_tiers[cid] = MemoryTier.HBM
                    result.promoted_chunks.append(cid)
                    total_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()

            # Synchronize the transfer stream before recording end event
            if transfer_stream:
                transfer_stream.synchronize()

            end_event.record()
            torch.cuda.synchronize()

            transfer_time_us = start_event.elapsed_time(end_event) * 1000.0  # ms → µs

            # Overlap determination: if we used a secondary stream, the
            # transfer ran concurrently with the default stream's compute.
            can_overlap = async_prefetch and transfer_stream is not None and (
                transfer_time_us < estimated_compute_us
            )

        else:
            # CPU fallback: use analytical model
            for cid in chunk_ids:
                if cid not in self.chunk_tiers:
                    result.failed_chunks.append(cid)
                    continue
                if self.chunk_tiers[cid] == MemoryTier.HBM:
                    continue

                kv = self.tail_store.retrieve_chunk(cid, target_device=self.device)
                if kv is None:
                    result.failed_chunks.append(cid)
                    continue

                k, v = kv
                self.hbm_kv[cid] = (k, v)
                self.chunk_tiers[cid] = MemoryTier.HBM
                result.promoted_chunks.append(cid)
                total_bytes += k.numel() * 2 + v.numel() * 2  # assume fp16

            transfer_time_us = self.bandwidth.promotion_time_us(total_bytes, via=decision.transfer_path)
            can_overlap = async_prefetch and (transfer_time_us < estimated_compute_us)

        result.bytes_promoted = total_bytes
        # If overlapped, the transfer cost is hidden behind compute
        result.promotion_time_us = 0.0 if can_overlap else transfer_time_us
        result.overlapped = can_overlap

        self.stats["total_promotions"] += len(result.promoted_chunks)
        self.stats["promotion_bytes"] += total_bytes
        self.stats["promotion_time_us"] += result.promotion_time_us
        self.stats["prefetch_depth_sum"] += result.prefetch_depth
        if result.transfer_path == "cxl":
            self.stats["cxl_promotions"] += len(result.promoted_chunks)
            if result.overlapped:
                self.stats["cxl_prefetch_hits"] += len(result.promoted_chunks)
        else:
            self.stats["pcie_promotions"] += len(result.promoted_chunks)

        return result
    
    def build_attention_mask(
        self,
        active_chunk_ids: List[int],
        seq_len: int,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Build attention mask for active chunks.
        
        Args:
            active_chunk_ids: Chunks currently in HBM
            seq_len: Total sequence length
            device: Device for mask
            
        Returns:
            Attention mask [seq_len] (True = can attend)
        """
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        
        for cid in active_chunk_ids:
            if cid in self.chunk_positions:
                start, end = self.chunk_positions[cid]
                mask[start:end] = True
        
        return mask
    
    def get_memory_breakdown(self) -> Dict[str, Any]:
        """Get current memory usage breakdown."""
        hbm_chunks = sum(1 for t in self.chunk_tiers.values() if t == MemoryTier.HBM)
        dram_chunks = sum(1 for t in self.chunk_tiers.values() if t == MemoryTier.HOST_DRAM)
        
        return {
            "hbm_bytes": self.stats["hbm_bytes_used"],
            "dram_bytes": self.stats["dram_bytes_used"],
            "hbm_chunks": hbm_chunks,
            "dram_chunks": dram_chunks,
            "hbm_utilization": self.stats["hbm_bytes_used"] / self.hbm_capacity_bytes,
            "total_promotions": self.stats["total_promotions"],
            "promotion_time_ms": self.stats["promotion_time_us"] / 1000,
            "pcie_promotions": self.stats["pcie_promotions"],
            "cxl_promotions": self.stats["cxl_promotions"],
            "cxl_prefetch_hit_rate": self.stats["cxl_prefetch_hits"] / max(self.stats["cxl_promotions"], 1),
            "avg_prefetch_depth": self.stats["prefetch_depth_sum"] / max(self.stats["total_promotions"], 1),
        }
    
    def compute_optimal_anchor_ratio(
        self,
        seq_len: int,
        target_hbm_utilization: float = 0.8,
    ) -> float:
        """
        Compute optimal anchor ratio given bandwidth constraints.
        
        This is the analytical model for HPCA paper:
        Given promotion bandwidth B and decode time T_decode per step,
        how many anchors do we need to keep hot so that promotion
        doesn't become a bottleneck?
        
        Args:
            seq_len: Sequence length
            target_hbm_utilization: Target HBM utilization
            
        Returns:
            Optimal anchor ratio (0-1)
        """
        # Maximum tokens that fit in HBM at target utilization
        max_hbm_tokens = int(
            (self.hbm_capacity_bytes * target_hbm_utilization) / self.bytes_per_token
        )
        
        # Ratio needed to fit in HBM
        ratio = max_hbm_tokens / seq_len if seq_len > 0 else 1.0
        
        # Clamp to reasonable range
        return max(0.05, min(0.5, ratio))


class PromotionPrefetchEngine:
    """
    Prefetch engine for overlapping promotion with compute.
    
    In decode phase, we have compute-bound GEMM operations.
    This engine predicts which chunks will be needed in step N+1
    and prefetches them during step N's compute.
    """
    
    def __init__(
        self,
        memory_manager: TwoTierMemoryManager,
        lookahead_steps: int = 1,
    ):
        """
        Initialize prefetch engine.
        
        Args:
            memory_manager: TwoTierMemoryManager instance
            lookahead_steps: How many steps ahead to prefetch
        """
        self.memory_manager = memory_manager
        self.lookahead_steps = lookahead_steps
        
        # Predicted future chunk needs
        self.predicted_needs: List[List[int]] = []
        
        # Ongoing async transfers (simulated)
        self.pending_transfers: Dict[int, float] = {}  # chunk_id -> completion_time
        self.last_decision: Optional[PrefetchDecision] = None
    
    def predict_future_needs(
        self,
        current_step: int,
        recent_attention_patterns: List[Dict[int, float]],
    ) -> List[int]:
        """
        Predict which chunks will be needed in future steps.
        
        Simple heuristic: chunks with increasing attention trend
        are likely to be needed soon.
        
        Args:
            current_step: Current generation step
            recent_attention_patterns: List of chunk_attention dicts
            
        Returns:
            List of chunk_ids predicted to be needed
        """
        if not recent_attention_patterns:
            return []
        
        # Compute attention trend for each chunk
        chunk_trends = {}
        chunk_values = {}
        
        for pattern in recent_attention_patterns:
            for cid, mass in pattern.items():
                if cid not in chunk_values:
                    chunk_values[cid] = []
                chunk_values[cid].append(mass)
        
        for cid, values in chunk_values.items():
            if len(values) >= 2:
                # Simple trend: increasing or high value
                trend = values[-1] - values[0]
                chunk_trends[cid] = (trend, values[-1])
        
        # Predict: chunks with positive trend or high current attention
        predicted = [
            cid for cid, (trend, current) in chunk_trends.items()
            if trend > 0 or current > 0.1
        ]
        
        # Filter to chunks not already in HBM
        predicted = [
            cid for cid in predicted
            if self.memory_manager.chunk_tiers.get(cid) != MemoryTier.HBM
        ]
        
        return predicted[:4]  # Limit prefetch to 4 chunks at a time
    
    def schedule_prefetch(
        self,
        predicted_chunks: List[int],
        current_compute_time_us: float,
    ) -> PromotionResult:
        """
        Schedule prefetch of predicted chunks.
        
        Args:
            predicted_chunks: Chunks to prefetch
            current_compute_time_us: Time for current compute (for overlap)
            
        Returns:
            PromotionResult
        """
        # Check which can be overlapped
        bytes_to_promote = self.memory_manager.estimate_chunk_transfer_bytes(predicted_chunks)
        self.last_decision = self.memory_manager.select_transfer_path(
            bytes_to_promote=bytes_to_promote,
            estimated_compute_us=current_compute_time_us,
            prefer_cxl_for_prefetch=True,
        )
        self.last_decision.chunk_ids = list(predicted_chunks)

        result = self.memory_manager.promote_chunks(
            chunk_ids=predicted_chunks,
            async_prefetch=True,
            estimated_compute_us=current_compute_time_us,
            preferred_path=self.last_decision.transfer_path,
        )
        
        return result
