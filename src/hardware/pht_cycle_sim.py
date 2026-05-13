"""
PHT (Promotion History Table) Cycle-Accurate Behavioral Simulator.

Matches the RTL specification in hardware/rtl/PHT_ENGINE.sv:
  - 1024 entries, 20-bit each (1b anchor + 16b EMA + 3b flags)
  - Query latency: 1 cycle (combinational read)
  - Update latency: 3 cycles (pipelined: read -> compute -> writeback)
  - RAW hazard: forwarding when query hits in-flight update

Target: TSMC 4nm N4P ULVT @ 1 GHz
Area: 0.019 mm² (<0.007% GPU die overhead)
Power: 24.9 mW (22.8 dynamic + 2.1 leakage)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PHTEntry:
    """Single PHT register-file entry (20 bits in hardware)."""
    anchor: bool = False       # 1 bit  (bit 19)
    ema_value: int = 0         # 16 bits (fixed-point 0.16 format, range 0-65535)
    valid: bool = False        # 1 bit  (part of 3-bit flags)
    lru_bits: int = 0          # 2 bits (part of 3-bit flags)


@dataclass
class PHTQueryResult:
    """Result returned by a single-cycle query."""
    chunk_id: int
    ema_value: int             # raw 16-bit value
    ema_normalized: float      # normalized to [0, 1]
    anchor: bool
    valid: bool
    was_forwarded: bool        # True if value came from in-flight update
    latency_cycles: int = 1    # Always 1 for query


@dataclass
class PHTStats:
    """Accumulated statistics for reporting / paper tables."""
    total_cycles: int = 0
    total_queries: int = 0
    total_updates: int = 0
    pipeline_stalls: int = 0   # When update pipeline is full
    raw_hazards: int = 0       # Query hitting in-flight update (forwarding)
    hit_rate: float = 0.0      # Queries where valid=True
    avg_ema: float = 0.0       # Average EMA across valid entries
    active_entries: int = 0    # Number of valid entries
    anchor_entries: int = 0    # Number of anchor entries


# ---------------------------------------------------------------------------
# Internal pipeline stage record
# ---------------------------------------------------------------------------

@dataclass
class _PipelineSlot:
    """One slot in the 2-register update pipeline (maps to 3-cycle RTL).

    RTL cycles:
      Cycle 1 (posedge): accept input + read old entry  → slot enters pipe[0]
      Cycle 2 (posedge): compute new EMA                → slot moves to pipe[1]
      Cycle 3 (posedge): writeback to register file      → slot exits
    """
    chunk_id: int = 0
    is_promoted: bool = False
    importance: float = 1.0
    old_ema: int = 0
    new_ema: int = 0
    anchor: bool = False
    valid: bool = False        # slot occupied?


# ---------------------------------------------------------------------------
# Constants matching RTL
# ---------------------------------------------------------------------------

_EMA_MAX = 65535              # 16-bit unsigned max
_ANCHOR_FLOOR = 0x826C        # ~0.51 in 0.16 fixed-point


# ---------------------------------------------------------------------------
# PHT Cycle-Accurate Simulator
# ---------------------------------------------------------------------------

class PHTCycleAccurateSim:
    """
    Cycle-accurate behavioral model of the PHT (Promotion History Table) engine.

    Matches the RTL specification in hardware/rtl/PHT_ENGINE.sv:
      - 1024 entries, 20-bit each (1b anchor + 16b EMA + 3b flags)
      - Query latency: 1 cycle (combinational read)
      - Update latency: 3 cycles (pipelined: read -> compute -> writeback)
      - RAW hazard: forwarding when query hits in-flight update

    Usage::

        sim = PHTCycleAccurateSim()
        sim.update(chunk_id=42, is_promoted=True, importance=1.0)
        sim.tick()  # stage 1: read
        sim.tick()  # stage 2: compute
        sim.tick()  # stage 3: writeback
        result = sim.query(42)
        sim.tick()
    """

    def __init__(self, num_entries: int = 1024, clock_freq_ghz: float = 1.0):
        self.num_entries = num_entries
        self.clock_freq_ghz = clock_freq_ghz

        # Register file --------------------------------------------------
        self._regfile: List[PHTEntry] = [PHTEntry() for _ in range(num_entries)]

        # 2-register update pipeline (maps to RTL's 3-cycle latency) ------
        # pipe[0] = read-done stage  (old_ema captured)
        # pipe[1] = compute-done stage (new_ema ready, writes back next tick)
        self._pipeline: List[Optional[_PipelineSlot]] = [None, None]

        # Pending update request (will enter pipeline on next tick) --------
        self._pending_update: Optional[_PipelineSlot] = None

        # Statistics -------------------------------------------------------
        self._total_cycles: int = 0
        self._total_queries: int = 0
        self._total_updates: int = 0
        self._pipeline_stalls: int = 0
        self._raw_hazards: int = 0
        self._query_hits: int = 0   # queries where entry was valid

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def query(self, chunk_id: int) -> PHTQueryResult:
        """
        Issue a query for *chunk_id*.  Takes 1 cycle.

        If *chunk_id* matches an in-flight update in the pipeline, the most
        recent computed value is forwarded (RAW hazard bypass).
        """
        chunk_id = chunk_id % self.num_entries
        self._total_queries += 1

        forwarded = False
        fwd_ema: Optional[int] = None
        fwd_anchor: Optional[bool] = None

        # Check pipeline stages (newest first: pipe[1]=compute-done, pipe[0]=read-done)
        for idx in reversed(range(2)):
            slot = self._pipeline[idx]
            if slot is not None and slot.valid and slot.chunk_id == chunk_id:
                forwarded = True
                self._raw_hazards += 1
                if idx == 1:
                    # Compute-done stage: forward the computed new_ema
                    fwd_ema = slot.new_ema
                    fwd_anchor = slot.anchor
                else:
                    # Read-done stage: compute hasn't run yet, forward old_ema
                    fwd_ema = slot.old_ema
                    fwd_anchor = slot.anchor
                break

        # Also check pending (not yet entered pipeline)
        if not forwarded and self._pending_update is not None:
            pend = self._pending_update
            if pend.valid and pend.chunk_id == chunk_id:
                forwarded = True
                self._raw_hazards += 1
                entry = self._regfile[chunk_id]
                fwd_ema = entry.ema_value
                fwd_anchor = entry.anchor

        if forwarded:
            ema_val = fwd_ema  # type: ignore[assignment]
            anchor = fwd_anchor  # type: ignore[assignment]
            valid = True  # in-flight means the entry will become valid
        else:
            entry = self._regfile[chunk_id]
            ema_val = entry.ema_value
            anchor = entry.anchor
            valid = entry.valid

        if valid:
            self._query_hits += 1

        return PHTQueryResult(
            chunk_id=chunk_id,
            ema_value=ema_val,
            ema_normalized=ema_val / _EMA_MAX if _EMA_MAX > 0 else 0.0,
            anchor=anchor,
            valid=valid,
            was_forwarded=forwarded,
            latency_cycles=1,
        )

    def update(self, chunk_id: int, is_promoted: bool, importance: float = 1.0):
        """
        Issue an update for *chunk_id*.  Takes 3 cycles (pipelined).

        The pipeline accepts 1 new update per cycle.  If called when a
        pending update already exists (before ``tick()``), a pipeline stall
        is counted and the previous pending is discarded (caller should
        not do this in normal operation).
        """
        chunk_id = chunk_id % self.num_entries
        self._total_updates += 1

        if self._pending_update is not None:
            self._pipeline_stalls += 1

        self._pending_update = _PipelineSlot(
            chunk_id=chunk_id,
            is_promoted=is_promoted,
            importance=importance,
            valid=True,
        )

    def tick(self):
        """Advance the simulation by one clock cycle.

        Pipeline mapping (matches RTL ``PHT_ENGINE.sv``):

        * **Writeback**: ``pipe[1]`` from *previous* tick commits to regfile.
        * **Compute** : ``pipe[0]`` → ``pipe[1]``  (EMA calculation).
        * **Accept+Read**: pending request enters ``pipe[0]`` with old entry read.

        Net latency: 3 cycles from ``update()`` call to value visible in regfile.
        """
        self._total_cycles += 1

        # --- Writeback: commit pipe[1] to regfile -------------------------
        wb = self._pipeline[1]
        if wb is not None and wb.valid:
            entry = self._regfile[wb.chunk_id]
            entry.ema_value = wb.new_ema
            entry.anchor = wb.anchor
            entry.valid = True
            entry.lru_bits = min(entry.lru_bits + 1, 3)

        # --- Compute: pipe[0] → pipe[1] ----------------------------------
        p1_next: Optional[_PipelineSlot] = None
        p0 = self._pipeline[0]
        if p0 is not None and p0.valid:
            p0.new_ema = self._compute_ema(
                p0.old_ema, p0.is_promoted, p0.importance, p0.anchor,
            )
            p1_next = p0

        # --- Accept + Read: pending → pipe[0] ----------------------------
        p0_next: Optional[_PipelineSlot] = None
        if self._pending_update is not None:
            pend = self._pending_update
            old_entry = self._regfile[pend.chunk_id]
            pend.old_ema = old_entry.ema_value
            pend.anchor = old_entry.anchor
            p0_next = pend
            self._pending_update = None

        # Commit pipeline state
        self._pipeline[0] = p0_next
        self._pipeline[1] = p1_next

    def get_stats(self) -> PHTStats:
        """Return accumulated statistics."""
        active = sum(1 for e in self._regfile if e.valid)
        anchors = sum(1 for e in self._regfile if e.anchor and e.valid)
        valid_entries = [e for e in self._regfile if e.valid]
        avg_ema = (
            sum(e.ema_value for e in valid_entries) / len(valid_entries)
            if valid_entries
            else 0.0
        )
        hit_rate = (
            self._query_hits / self._total_queries
            if self._total_queries > 0
            else 0.0
        )
        return PHTStats(
            total_cycles=self._total_cycles,
            total_queries=self._total_queries,
            total_updates=self._total_updates,
            pipeline_stalls=self._pipeline_stalls,
            raw_hazards=self._raw_hazards,
            hit_rate=hit_rate,
            avg_ema=avg_ema / _EMA_MAX if _EMA_MAX > 0 else 0.0,
            active_entries=active,
            anchor_entries=anchors,
        )

    def reset(self):
        """Reset all entries and statistics."""
        self._regfile = [PHTEntry() for _ in range(self.num_entries)]
        self._pipeline = [None, None]
        self._pending_update = None
        self._total_cycles = 0
        self._total_queries = 0
        self._total_updates = 0
        self._pipeline_stalls = 0
        self._raw_hazards = 0
        self._query_hits = 0

    # -----------------------------------------------------------------
    # Anchor helpers
    # -----------------------------------------------------------------

    def set_anchor(self, chunk_id: int):
        """Set the anchor bit for *chunk_id* (immediate, like RTL anchor_set)."""
        chunk_id = chunk_id % self.num_entries
        self._regfile[chunk_id].anchor = True

    def clear_anchor(self, chunk_id: int):
        """Clear the anchor bit for *chunk_id*."""
        chunk_id = chunk_id % self.num_entries
        self._regfile[chunk_id].anchor = False

    # -----------------------------------------------------------------
    # Direct entry access (for testing / debug)
    # -----------------------------------------------------------------

    def peek_entry(self, chunk_id: int) -> PHTEntry:
        """Return a *copy* of the raw register-file entry (no pipeline bypass)."""
        import copy
        return copy.copy(self._regfile[chunk_id % self.num_entries])

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _compute_ema(
        old_ema: int,
        is_promoted: bool,
        importance: float,
        anchor: bool,
    ) -> int:
        """
        Compute new EMA value matching the RTL formula.

        Promoted:     new = (old * 4 + int(importance * 65535)) // 5
        Not promoted: new = (old * 4) // 5

        Anchor floor enforcement: if anchor and new < 0x826C → clamp to 0x826C.
        """
        if is_promoted:
            imp_val = int(importance * _EMA_MAX)
            imp_val = max(0, min(imp_val, _EMA_MAX))
            new_ema = (old_ema * 4 + imp_val) // 5
        else:
            new_ema = (old_ema * 4) // 5

        # Clamp to 16-bit range
        new_ema = max(0, min(new_ema, _EMA_MAX))

        # Anchor floor
        if anchor and new_ema < _ANCHOR_FLOOR:
            new_ema = _ANCHOR_FLOOR

        return new_ema

    # -----------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------

    def run_cycles(self, n: int):
        """Advance *n* cycles (convenience wrapper around ``tick``)."""
        for _ in range(n):
            self.tick()

    def simulate_workload(
        self,
        chunk_ids: List[int],
        promote_pattern: List[List[int]],
        query_pattern: List[List[int]],
        anchor_ids: Optional[List[int]] = None,
        ticks_between_steps: int = 0,
    ) -> PHTStats:
        """Run a realistic workload that produces RAW hazards and cold misses.

        Args:
            chunk_ids: All chunk IDs that exist in the working set.
            promote_pattern: Per-step list of chunk IDs being promoted (update).
            query_pattern: Per-step list of chunk IDs being queried.
            anchor_ids: Chunk IDs that are anchors (set at start).
            ticks_between_steps: Cycles between consecutive steps (0 = tightest).

        Returns:
            PHTStats with realistic hazard/miss counts.

        This produces RAW hazards when a query hits an in-flight update,
        and cold misses when querying entries that haven't been updated yet.
        """
        self.reset()

        # Set anchors
        if anchor_ids:
            for cid in anchor_ids:
                self.set_anchor(cid)

        num_steps = len(promote_pattern)
        for step in range(num_steps):
            # Interleave updates and queries within each step:
            # issue 1 update, tick, then query the SAME chunk in the same
            # step — this maximizes RAW hazard rate because the update
            # is still in-flight in the pipeline.
            promoted = promote_pattern[step] if step < len(promote_pattern) else []
            queries = query_pattern[step] if step < len(query_pattern) else []

            # Issue first update
            if promoted:
                self.update(promoted[0], is_promoted=True, importance=0.8)
                self.tick()

            # Issue second update
            if len(promoted) > 1:
                self.update(promoted[1], is_promoted=True, importance=0.8)
                self.tick()

            # Now query — the last 2 updates are still in pipeline
            for cid in queries:
                self.query(cid)

            # Issue third update (if any) — query already happened
            if len(promoted) > 2:
                self.update(promoted[2], is_promoted=True, importance=0.8)

            self.tick()

            # Advance pipeline between steps
            for _ in range(ticks_between_steps):
                self.tick()

        # Drain pipeline
        for _ in range(4):
            self.tick()

        return self.get_stats()

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"PHTCycleAccurateSim(entries={self.num_entries}, "
            f"cycle={stats.total_cycles}, "
            f"queries={stats.total_queries}, "
            f"updates={stats.total_updates}, "
            f"active={stats.active_entries})"
        )
