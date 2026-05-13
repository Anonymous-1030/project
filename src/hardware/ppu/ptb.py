"""
Promotion Target Buffer (PTB) — branch-target-buffer-inspired hardware cache.

Analogous to a Branch Target Buffer (BTB) in CPU branch prediction:
- Caches recently successful promotion targets (chunk address + metadata)
- Enables speculative prefetch before the scorer runs
- On PHT predict-promote + PTB hit → immediate DMA without waiting for scorer

Area at 7nm: 32 entries × 16B = 512B SRAM + tag CAM + LRU ≈ 0.005 mm², 4 mW
Latency: 1 cycle (parallel with PHT lookup)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PTBConfig:
    """PTB hardware configuration."""

    num_entries: int = 32
    associativity: int = 32          # fully associative by default
    tag_bits: int = 16               # signature tag for validation
    eviction_policy: str = "lru"     # "lru" or "fifo"
    entry_bytes: int = 16            # per-entry storage cost
    max_age_steps: int = 50          # evict entries older than this


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PTBEntry:
    """Single PTB entry: cached promotion target."""

    chunk_id: str
    chunk_address: int       # logical address for DMA
    utility_score: float     # last known utility
    last_success_step: int   # step when last successfully promoted
    metadata_tag: int        # signature for validation
    valid: bool = True
    lru_counter: int = 0     # for LRU eviction (higher = more recent)


@dataclass
class PTBLookupResult:
    """Result of a PTB lookup."""

    hit: bool
    entry: Optional[PTBEntry] = None
    signature: int = 0


# ---------------------------------------------------------------------------
# Promotion Target Buffer
# ---------------------------------------------------------------------------

class PromotionTargetBuffer:
    """Hardware model for the Promotion Target Buffer.

    Structural analogy to branch prediction:
      Branch Prediction          KV Promotion Prediction
      ─────────────────          ───────────────────────
      BTB caches branch targets  PTB caches promotion targets
      PC tag match               Signature tag match
      Speculative fetch          Speculative prefetch via CXL/PCIe
      BTB miss → stall           PTB miss → wait for scorer
    """

    def __init__(self, config: PTBConfig):
        self.config = config
        self._entries: List[Optional[PTBEntry]] = [None] * config.num_entries
        self._global_lru = 0  # monotonic counter for LRU ordering

        # Statistics
        self._lookups = 0
        self._hits = 0
        self._inserts = 0
        self._evictions = 0
        self._age_evictions = 0

    # ── Lookup ─────────────────────────────────────────────────────────

    def lookup(self, signature: int) -> PTBLookupResult:
        """Check if a promotion target is cached for this signature.

        Tag match: signature & tag_mask == entry.metadata_tag
        """
        self._lookups += 1
        tag = signature & ((1 << self.config.tag_bits) - 1)

        for i, entry in enumerate(self._entries):
            if entry is not None and entry.valid and entry.metadata_tag == tag:
                self._hits += 1
                # Update LRU
                self._global_lru += 1
                entry.lru_counter = self._global_lru
                return PTBLookupResult(hit=True, entry=entry, signature=signature)

        return PTBLookupResult(hit=False, signature=signature)

    # ── Insert ─────────────────────────────────────────────────────────

    def insert(
        self,
        signature: int,
        chunk_id: str,
        chunk_address: int,
        utility_score: float,
        step: int,
    ) -> None:
        """Insert a successful promotion target into the PTB."""
        self._inserts += 1
        tag = signature & ((1 << self.config.tag_bits) - 1)
        self._global_lru += 1

        new_entry = PTBEntry(
            chunk_id=chunk_id,
            chunk_address=chunk_address,
            utility_score=utility_score,
            last_success_step=step,
            metadata_tag=tag,
            valid=True,
            lru_counter=self._global_lru,
        )

        # Check if tag already exists → update in place
        for i, entry in enumerate(self._entries):
            if entry is not None and entry.valid and entry.metadata_tag == tag:
                self._entries[i] = new_entry
                return

        # Find empty slot
        for i, entry in enumerate(self._entries):
            if entry is None or not entry.valid:
                self._entries[i] = new_entry
                return

        # Evict based on policy
        victim_idx = self._find_victim()
        self._entries[victim_idx] = new_entry
        self._evictions += 1

    # ── Invalidation ───────────────────────────────────────────────────

    def invalidate(self, signature: int) -> bool:
        """Invalidate entry on misprediction. Returns True if found."""
        tag = signature & ((1 << self.config.tag_bits) - 1)
        for entry in self._entries:
            if entry is not None and entry.valid and entry.metadata_tag == tag:
                entry.valid = False
                return True
        return False

    # ── Age-based eviction ─────────────────────────────────────────────

    def age_entries(self, current_step: int) -> List[str]:
        """Evict entries older than max_age_steps. Returns evicted chunk_ids."""
        evicted = []
        for i, entry in enumerate(self._entries):
            if entry is None or not entry.valid:
                continue
            age = current_step - entry.last_success_step
            if age > self.config.max_age_steps:
                evicted.append(entry.chunk_id)
                entry.valid = False
                self._age_evictions += 1
        return evicted

    # ── Speculative prefetch targets ───────────────────────────────────

    def get_speculative_prefetch_targets(
        self, current_step: int
    ) -> List[PTBEntry]:
        """Return all valid PTB entries suitable for speculative prefetch.

        Sorted by utility_score descending (highest utility first).
        """
        targets = []
        for entry in self._entries:
            if entry is None or not entry.valid:
                continue
            age = current_step - entry.last_success_step
            if age <= self.config.max_age_steps:
                targets.append(entry)
        targets.sort(key=lambda e: e.utility_score, reverse=True)
        return targets

    # ── Eviction policy ────────────────────────────────────────────────

    def _find_victim(self) -> int:
        """Find victim index for eviction."""
        if self.config.eviction_policy == "lru":
            return self._find_lru_victim()
        else:  # fifo
            return self._find_fifo_victim()

    def _find_lru_victim(self) -> int:
        """LRU: evict entry with smallest lru_counter."""
        min_lru = float("inf")
        victim = 0
        for i, entry in enumerate(self._entries):
            if entry is None:
                return i
            if not entry.valid:
                return i
            if entry.lru_counter < min_lru:
                min_lru = entry.lru_counter
                victim = i
        return victim

    def _find_fifo_victim(self) -> int:
        """FIFO: evict entry with oldest last_success_step."""
        oldest_step = float("inf")
        victim = 0
        for i, entry in enumerate(self._entries):
            if entry is None:
                return i
            if not entry.valid:
                return i
            if entry.last_success_step < oldest_step:
                oldest_step = entry.last_success_step
                victim = i
        return victim

    # ── Statistics ─────────────────────────────────────────────────────

    @property
    def occupancy(self) -> int:
        return sum(1 for e in self._entries if e is not None and e.valid)

    def stats(self) -> Dict[str, Any]:
        """Return hit rate, occupancy, age distribution."""
        hit_rate = self._hits / max(1, self._lookups)
        valid_entries = [e for e in self._entries if e is not None and e.valid]
        ages = []
        if valid_entries:
            max_step = max(e.last_success_step for e in valid_entries)
            ages = [max_step - e.last_success_step for e in valid_entries]

        return {
            "lookups": self._lookups,
            "hits": self._hits,
            "hit_rate": hit_rate,
            "inserts": self._inserts,
            "evictions": self._evictions,
            "age_evictions": self._age_evictions,
            "occupancy": self.occupancy,
            "capacity": self.config.num_entries,
            "avg_age": sum(ages) / max(1, len(ages)) if ages else 0.0,
            "max_age": max(ages) if ages else 0,
        }

    def reset_stats(self) -> None:
        self._lookups = 0
        self._hits = 0
        self._inserts = 0
        self._evictions = 0
        self._age_evictions = 0
