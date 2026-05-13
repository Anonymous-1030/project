"""
Burst-and-Stick Promotion (BSP) - Burst Expansion Module.

This module implements burst expansion around selected chunks.
- Selected chunk(s) expand to a local burst window before transfer
- Support: no burst, radius=1, radius=2
- Configurable burst selection strategy
"""

import time
import logging
from typing import Dict, List, Set, Tuple, Optional

from src.core_types import (
    ChunkMetadata, BurstResult, ChunkTier
)
from src.config import BurstConfig

logger = logging.getLogger(__name__)


class BurstExpander:
    """
    Burst expansion around selected chunks.
    
    For each selected "core" chunk, expand to include neighboring chunks
    within a configurable radius. This helps capture local context that
    may be relevant even if the exact chunk wasn't scored highest.
    """
    
    def __init__(self, config: BurstConfig):
        self.config = config
        logger.info(
            f"BurstExpander initialized: enabled={config.enabled}, "
            f"radius={config.radius}"
        )
    
    def expand(
        self,
        selected_ids: List[str],
        all_chunks: Dict[str, ChunkMetadata],
        current_step: int,
    ) -> BurstResult:
        """
        Expand selected chunks to burst window.
        
        Args:
            selected_ids: List of chunk IDs selected by scheduler
            all_chunks: Dictionary of all chunk metadata
            current_step: Current decode step
            
        Returns:
            BurstResult with expanded burst set
        """
        start_time = time.time()
        
        if not self.config.enabled or self.config.radius == 0:
            # No burst expansion
            return BurstResult(
                request_id="",
                step=current_step,
                input_ids=selected_ids,
                burst_ids=selected_ids,
                core_ids=selected_ids,
                expansion_ids=[],
                burst_radius={cid: 0 for cid in selected_ids},
                n_input=len(selected_ids),
                n_burst_total=len(selected_ids),
                n_expansion=0,
                burst_latency_us=(time.time() - start_time) * 1e6,
            )
        
        # Build position index for fast neighbor lookup
        position_index = self._build_position_index(all_chunks)
        
        # Expand each selected chunk
        burst_ids = set(selected_ids)
        core_ids = list(selected_ids)
        expansion_ids = []
        burst_radius: Dict[str, int] = {}
        
        # Get request_id from first valid chunk
        request_id = ""
        for core_id in selected_ids:
            chunk = all_chunks.get(core_id)
            if chunk:
                request_id = chunk.request_id
                break
        
        for core_id in selected_ids:
            core_chunk = all_chunks.get(core_id)
            if core_chunk is None:
                continue
            
            # Core chunk has radius 0
            burst_radius[core_id] = 0
            
            # Find neighbors within radius
            neighbors = self._find_neighbors(
                core_chunk, all_chunks, position_index, self.config.radius
            )
            
            for neighbor_id, radius in neighbors:
                if neighbor_id not in burst_ids:
                    burst_ids.add(neighbor_id)
                    expansion_ids.append(neighbor_id)
                    burst_radius[neighbor_id] = radius
        
        # Convert to list with stable ordering
        burst_ids_ordered = self._order_burst_set(
            core_ids, expansion_ids, all_chunks
        )
        
        latency_us = (time.time() - start_time) * 1e6
        
        return BurstResult(
            request_id=request_id,
            step=current_step,
            input_ids=selected_ids,
            burst_ids=burst_ids_ordered,
            core_ids=core_ids,
            expansion_ids=expansion_ids,
            burst_radius=burst_radius,
            n_input=len(selected_ids),
            n_burst_total=len(burst_ids_ordered),
            n_expansion=len(expansion_ids),
            burst_latency_us=latency_us,
        )
    
    def _build_position_index(
        self,
        all_chunks: Dict[str, ChunkMetadata],
    ) -> Dict[str, List[Tuple[int, int, str]]]:
        """
        Build position index for fast neighbor lookup.
        
        Returns:
            Dictionary mapping request_id to sorted list of (start, end, chunk_id)
        """
        index: Dict[str, List[Tuple[int, int, str]]] = {}
        
        for chunk_id, chunk in all_chunks.items():
            if chunk.request_id not in index:
                index[chunk.request_id] = []
            index[chunk.request_id].append(
                (chunk.token_start, chunk.token_end, chunk_id)
            )
        
        # Sort each request's chunks by position
        for request_id in index:
            index[request_id].sort(key=lambda x: x[0])
        
        return index
    
    def _find_neighbors(
        self,
        core_chunk: ChunkMetadata,
        all_chunks: Dict[str, ChunkMetadata],
        position_index: Dict[str, List[Tuple[int, int, str]]],
        max_radius: int,
    ) -> List[Tuple[str, int]]:
        """
        Find neighboring chunks within radius.
        
        Args:
            core_chunk: The center chunk
            all_chunks: All chunk metadata
            position_index: Position index for lookup
            max_radius: Maximum radius to search
            
        Returns:
            List of (neighbor_id, radius) tuples
        """
        neighbors = []
        request_chunks = position_index.get(core_chunk.request_id, [])
        
        # Find core chunk index in sorted list
        core_idx = None
        for i, (start, end, chunk_id) in enumerate(request_chunks):
            if chunk_id == core_chunk.chunk_id:
                core_idx = i
                break
        
        if core_idx is None:
            return neighbors
        
        # Search forward and backward
        chunk_size = core_chunk.num_tokens
        
        # Search forward (higher positions)
        for i in range(core_idx + 1, len(request_chunks)):
            start, end, chunk_id = request_chunks[i]
            
            # Compute distance in chunk units
            distance_tokens = start - core_chunk.token_end
            distance_chunks = max(1, distance_tokens / max(chunk_size, 1))
            
            if distance_chunks <= max_radius:
                radius = int(distance_chunks)
                neighbors.append((chunk_id, radius))
            else:
                break
        
        # Search backward (lower positions)
        for i in range(core_idx - 1, -1, -1):
            start, end, chunk_id = request_chunks[i]
            
            # Compute distance in chunk units
            distance_tokens = core_chunk.token_start - end
            distance_chunks = max(1, distance_tokens / max(chunk_size, 1))
            
            if distance_chunks <= max_radius:
                radius = int(distance_chunks)
                neighbors.append((chunk_id, radius))
            else:
                break
        
        return neighbors
    
    def _order_burst_set(
        self,
        core_ids: List[str],
        expansion_ids: List[str],
        all_chunks: Dict[str, ChunkMetadata],
    ) -> List[str]:
        """
        Order burst set with cores first, then expansions by position.
        
        Args:
            core_ids: Core chunk IDs
            expansion_ids: Expansion chunk IDs
            all_chunks: All chunk metadata
            
        Returns:
            Ordered list of chunk IDs
        """
        if self.config.burst_selection == "contiguous":
            # Order by position in sequence
            all_ids = core_ids + expansion_ids
            
            # Sort by token start position
            def get_position(chunk_id):
                chunk = all_chunks.get(chunk_id)
                if chunk:
                    return chunk.token_start
                return 0
            
            return sorted(all_ids, key=get_position)
        
        elif self.config.burst_selection == "symmetric":
            # Core first, then symmetric expansions
            ordered = list(core_ids)
            
            # Group expansions by radius
            # For simplicity, just append expansions sorted by position
            expansion_chunks = [
                all_chunks.get(cid) for cid in expansion_ids
                if cid in all_chunks
            ]
            expansion_chunks.sort(key=lambda c: c.token_start)
            ordered.extend([c.chunk_id for c in expansion_chunks])
            
            return ordered
        
        else:
            # Default: cores first, then expansions
            return core_ids + expansion_ids


class IdentityBurstExpander(BurstExpander):
    """
    Identity burst expander (no expansion).
    
    Used for ablation studies.
    """
    
    def __init__(self):
        config = BurstConfig(enabled=False, radius=0)
        super().__init__(config)
