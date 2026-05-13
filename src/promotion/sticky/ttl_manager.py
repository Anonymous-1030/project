"""
Sticky Residency TTL Manager for ProSE-X 2.0.

Implements sticky residency with TTL:
- Promoted chunks persist for configurable minimum steps unless explicitly evicted
- TTL refresh policy is explicit and logged
- Configurable TTL values: 2, 4, 8 steps
"""

import time
import logging
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

from src.core_types import (
    ChunkMetadata, StickyResult, StickyUpdate, ChunkState
)
from src.config import BurstConfig

logger = logging.getLogger(__name__)


@dataclass
class StickyEntry:
    """Internal entry for sticky residency tracking."""
    chunk_id: str
    original_ttl: int
    current_ttl: int
    promotion_step: int
    refresh_count: int = 0


class StickyTTLManager:
    """
    Manages sticky residency with TTL for promoted chunks.
    
    Features:
    - TTL assignment on promotion
    - TTL refresh on access/recompute
    - TTL decay (optional)
    - Explicit expiration
    - Comprehensive logging
    """
    
    def __init__(self, config: BurstConfig):
        self.config = config
        
        # Active sticky entries
        self._entries: Dict[str, StickyEntry] = {}
        
        logger.info(
            f"StickyTTLManager initialized: enabled={config.sticky_enabled}, "
            f"default_ttl={config.default_ttl}"
        )
    
    def update(
        self,
        newly_promoted_ids: List[str],
        accessed_ids: List[str],
        all_chunks: Dict[str, ChunkMetadata],
        current_step: int,
    ) -> StickyResult:
        """
        Update sticky residency state.
        
        Args:
            newly_promoted_ids: Chunks newly promoted this step
            accessed_ids: Chunks accessed this step (for TTL refresh)
            all_chunks: All chunk metadata
            current_step: Current decode step
            
        Returns:
            StickyResult with updated TTL state
        """
        start_time = time.time()
        
        ttl_updates: List[StickyUpdate] = []
        
        # 1. Add newly promoted chunks
        for chunk_id in newly_promoted_ids:
            if not self.config.sticky_enabled:
                # Sticky disabled - set TTL to 0
                ttl = 0
            else:
                ttl = self.config.default_ttl
            
            if chunk_id in self._entries:
                # Already sticky - refresh TTL
                old_ttl = self._entries[chunk_id].current_ttl
                new_ttl = min(
                    self.config.ttl_refresh_max if self.config.ttl_refresh_max > 0 else ttl,
                    max(ttl, old_ttl)
                )
                self._entries[chunk_id].current_ttl = new_ttl
                
                ttl_updates.append(StickyUpdate(
                    chunk_id=chunk_id,
                    old_ttl=old_ttl,
                    new_ttl=new_ttl,
                    update_type="refresh"
                ))
            else:
                # New entry
                self._entries[chunk_id] = StickyEntry(
                    chunk_id=chunk_id,
                    original_ttl=ttl,
                    current_ttl=ttl,
                    promotion_step=current_step,
                )
                
                ttl_updates.append(StickyUpdate(
                    chunk_id=chunk_id,
                    old_ttl=0,
                    new_ttl=ttl,
                    update_type="new"
                ))
        
        # 2. Refresh TTL for accessed chunks
        if self.config.sticky_enabled and self.config.ttl_refresh_policy != "none":
            for chunk_id in accessed_ids:
                if chunk_id in self._entries:
                    old_ttl = self._entries[chunk_id].current_ttl
                    
                    # Compute new TTL based on refresh policy
                    if self.config.ttl_refresh_policy == "access":
                        # Refresh to original TTL
                        new_ttl = self._entries[chunk_id].original_ttl
                    elif self.config.ttl_refresh_policy == "recompute":
                        # Increase TTL up to max
                        new_ttl = min(
                            self.config.ttl_refresh_max,
                            old_ttl + 1
                        )
                    else:
                        continue
                    
                    self._entries[chunk_id].current_ttl = new_ttl
                    self._entries[chunk_id].refresh_count += 1
                    
                    ttl_updates.append(StickyUpdate(
                        chunk_id=chunk_id,
                        old_ttl=old_ttl,
                        new_ttl=new_ttl,
                        update_type="refresh"
                    ))
        
        # 3. Decay TTL (optional)
        if self.config.enable_ttl_decay and self.config.sticky_enabled:
            for chunk_id, entry in list(self._entries.items()):
                old_ttl = entry.current_ttl
                new_ttl = int(old_ttl * self.config.ttl_decay_rate)
                
                if new_ttl != old_ttl:
                    entry.current_ttl = new_ttl
                    ttl_updates.append(StickyUpdate(
                        chunk_id=chunk_id,
                        old_ttl=old_ttl,
                        new_ttl=new_ttl,
                        update_type="decay"
                    ))
        
        # 4. Decrement TTL and expire
        expired_ids = []
        if self.config.sticky_enabled:
            for chunk_id, entry in list(self._entries.items()):
                if chunk_id in newly_promoted_ids or chunk_id in accessed_ids:
                    # Don't decrement just-promoted or accessed chunks
                    continue
                
                old_ttl = entry.current_ttl
                new_ttl = old_ttl - 1
                
                if new_ttl <= 0:
                    # Expire
                    expired_ids.append(chunk_id)
                    del self._entries[chunk_id]
                    
                    ttl_updates.append(StickyUpdate(
                        chunk_id=chunk_id,
                        old_ttl=old_ttl,
                        new_ttl=0,
                        update_type="expire"
                    ))
                else:
                    entry.current_ttl = new_ttl
        else:
            # Sticky disabled - expire all immediately
            expired_ids = list(self._entries.keys())
            for chunk_id in expired_ids:
                entry = self._entries[chunk_id]
                ttl_updates.append(StickyUpdate(
                    chunk_id=chunk_id,
                    old_ttl=entry.current_ttl,
                    new_ttl=0,
                    update_type="expire"
                ))
            self._entries.clear()
        
        # 5. Build current promoted set
        promoted_ids = list(self._entries.keys())
        ttl_values = {e.chunk_id: e.current_ttl for e in self._entries.values()}
        
        # Count refreshed
        n_refreshed = sum(1 for u in ttl_updates if u.update_type == "refresh")
        avg_ttl = sum(ttl_values.values()) / max(len(ttl_values), 1)
        
        latency_us = (time.time() - start_time) * 1e6
        
        result = StickyResult(
            request_id="",  # Will be filled by caller
            step=current_step,
            promoted_ids=promoted_ids,
            ttl_values=ttl_values,
            ttl_updates=ttl_updates,
            expired_ids=expired_ids,
            n_promoted=len(promoted_ids),
            n_expired=len(expired_ids),
            n_refreshed=n_refreshed,
            avg_ttl=avg_ttl,
            sticky_latency_us=latency_us,
        )
        
        if self.config.log_ttl_updates:
            logger.debug(
                f"StickyTTL step {current_step}: "
                f"promoted={result.n_promoted}, expired={result.n_expired}, "
                f"avg_ttl={avg_ttl:.1f}"
            )
        
        return result
    
    def is_sticky(self, chunk_id: str) -> bool:
        """Check if a chunk is currently sticky."""
        return chunk_id in self._entries
    
    def get_ttl(self, chunk_id: str) -> int:
        """Get current TTL for a chunk (0 if not sticky)."""
        entry = self._entries.get(chunk_id)
        return entry.current_ttl if entry else 0
    
    def force_expire(self, chunk_id: str) -> bool:
        """Force expire a chunk (for explicit eviction)."""
        if chunk_id in self._entries:
            del self._entries[chunk_id]
            return True
        return False
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about sticky residency."""
        if not self._entries:
            return {
                "n_sticky": 0,
                "avg_ttl": 0.0,
                "max_ttl": 0,
                "min_ttl": 0,
            }
        
        ttls = [e.current_ttl for e in self._entries.values()]
        return {
            "n_sticky": len(self._entries),
            "avg_ttl": sum(ttls) / len(ttls),
            "max_ttl": max(ttls),
            "min_ttl": min(ttls),
        }
    
    def clear(self) -> None:
        """Clear all sticky entries."""
        self._entries.clear()


class NoStickyManager(StickyTTLManager):
    """
    No sticky residency (for ablation).
    
    All TTLs are set to 0, chunks expire immediately after selection.
    """
    
    def __init__(self):
        config = BurstConfig(sticky_enabled=False, default_ttl=0)
        super().__init__(config)
