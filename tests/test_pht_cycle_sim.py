"""
Unit tests for the PHT Cycle-Accurate Behavioral Simulator.

Tests cover:
  1. Basic query on empty entry
  2. Single update + query after pipeline drain
  3. Pipeline throughput (3 concurrent updates)
  4. RAW hazard forwarding
  5. EMA decay with alternating promoted / not-promoted
  6. Full-capacity test (1024 entries)
  7. Anchor floor enforcement
  8. Statistics collection
  9. Reset behaviour
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from hardware.pht_cycle_sim import (
    PHTCycleAccurateSim,
    PHTEntry,
    PHTQueryResult,
    PHTStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drain_pipeline(sim: PHTCycleAccurateSim, cycles: int = 4):
    """Tick enough cycles to flush the 3-stage pipeline."""
    for _ in range(cycles):
        sim.tick()


# ---------------------------------------------------------------------------
# 1. Basic query: empty entry → valid=False, ema=0
# ---------------------------------------------------------------------------

class TestBasicQuery:
    def test_empty_entry(self):
        sim = PHTCycleAccurateSim()
        result = sim.query(0)
        sim.tick()
        assert result.valid is False
        assert result.ema_value == 0
        assert result.was_forwarded is False
        assert result.latency_cycles == 1

    def test_out_of_range_wraps(self):
        sim = PHTCycleAccurateSim(num_entries=1024)
        r1 = sim.query(1024)  # should wrap to 0
        r2 = sim.query(0)
        assert r1.chunk_id == r2.chunk_id == 0


# ---------------------------------------------------------------------------
# 2. Single update + query after pipeline drain
# ---------------------------------------------------------------------------

class TestSingleUpdate:
    def test_promote_with_full_importance(self):
        sim = PHTCycleAccurateSim()
        sim.update(chunk_id=42, is_promoted=True, importance=1.0)
        # Need 3 ticks for the update to write back
        sim.tick()  # stage 0 → read
        sim.tick()  # stage 1 → compute
        sim.tick()  # stage 2 → writeback

        result = sim.query(42)
        sim.tick()

        # EMA: (0 * 4 + 65535) // 5 = 13107
        assert result.valid is True
        assert result.ema_value == 13107
        assert result.was_forwarded is False

    def test_promote_with_partial_importance(self):
        sim = PHTCycleAccurateSim()
        sim.update(chunk_id=10, is_promoted=True, importance=0.5)
        _drain_pipeline(sim)

        result = sim.query(10)
        sim.tick()

        # EMA: (0 * 4 + int(0.5 * 65535)) // 5 = (0 + 32767) // 5 = 6553
        assert result.ema_value == 6553

    def test_not_promoted_from_zero(self):
        sim = PHTCycleAccurateSim()
        sim.update(chunk_id=5, is_promoted=False)
        _drain_pipeline(sim)

        result = sim.query(5)
        sim.tick()

        # EMA: (0 * 4) // 5 = 0
        assert result.ema_value == 0
        assert result.valid is True  # entry was touched


# ---------------------------------------------------------------------------
# 3. Pipeline throughput: 3 different entries, 1 update per cycle
# ---------------------------------------------------------------------------

class TestPipelineThroughput:
    def test_three_concurrent_updates(self):
        sim = PHTCycleAccurateSim()

        # Issue 3 updates on consecutive cycles
        sim.update(chunk_id=0, is_promoted=True, importance=1.0)
        sim.tick()
        sim.update(chunk_id=1, is_promoted=True, importance=1.0)
        sim.tick()
        sim.update(chunk_id=2, is_promoted=True, importance=1.0)
        sim.tick()

        # After 3 more ticks all should be written back
        sim.tick()
        sim.tick()

        # All 3 entries should now be valid with same EMA
        for cid in (0, 1, 2):
            r = sim.query(cid)
            assert r.valid is True
            assert r.ema_value == 13107, f"chunk {cid}: {r.ema_value}"

        stats = sim.get_stats()
        assert stats.total_updates == 3
        assert stats.pipeline_stalls == 0


# ---------------------------------------------------------------------------
# 4. RAW forwarding: query while update is in-flight
# ---------------------------------------------------------------------------

class TestRAWForwarding:
    def test_forward_from_compute_stage(self):
        sim = PHTCycleAccurateSim()
        sim.update(chunk_id=7, is_promoted=True, importance=1.0)
        sim.tick()  # accept+read → pipe[0] (old_ema captured)
        sim.tick()  # compute → pipe[1] (new_ema = 13107)

        # Query while update is at pipe[1] — computed value available
        result = sim.query(7)
        assert result.was_forwarded is True
        assert result.ema_value == 13107  # forwarded computed value

        stats = sim.get_stats()
        assert stats.raw_hazards == 1

    def test_forward_from_read_stage(self):
        sim = PHTCycleAccurateSim()
        sim.update(chunk_id=7, is_promoted=True, importance=1.0)
        sim.tick()  # accept+read → pipe[0] (old_ema captured, compute not done)

        result = sim.query(7)
        assert result.was_forwarded is True
        # At pipe[0] (read only), we forward old_ema which is 0
        assert result.ema_value == 0

    def test_forward_latest_in_pipeline(self):
        """When same chunk_id is in multiple stages, forward the newest."""
        sim = PHTCycleAccurateSim()
        # First update
        sim.update(chunk_id=3, is_promoted=True, importance=1.0)
        sim.tick()
        # Second update to same entry before first completes
        sim.update(chunk_id=3, is_promoted=True, importance=1.0)
        sim.tick()

        # Now stage 1 has 1st update (computed), stage 0 has 2nd update (read)
        # query should see the newest (stage 1 = index 1, which is more recent
        # in reversed iteration)
        result = sim.query(3)
        assert result.was_forwarded is True


# ---------------------------------------------------------------------------
# 5. EMA decay: alternating promoted / not-promoted
# ---------------------------------------------------------------------------

class TestEMADecay:
    def test_promote_then_decay(self):
        sim = PHTCycleAccurateSim()

        # Promote once
        sim.update(chunk_id=0, is_promoted=True, importance=1.0)
        _drain_pipeline(sim)
        r = sim.query(0)
        sim.tick()
        ema_after_promote = r.ema_value
        assert ema_after_promote == 13107  # (0*4 + 65535) // 5

        # Decay (not promoted)
        sim.update(chunk_id=0, is_promoted=False)
        _drain_pipeline(sim)
        r = sim.query(0)
        sim.tick()
        ema_after_decay1 = r.ema_value
        assert ema_after_decay1 == (13107 * 4) // 5  # 10485

        # Decay again
        sim.update(chunk_id=0, is_promoted=False)
        _drain_pipeline(sim)
        r = sim.query(0)
        sim.tick()
        ema_after_decay2 = r.ema_value
        assert ema_after_decay2 == (10485 * 4) // 5  # 8388

    def test_multiple_promotes_converge(self):
        sim = PHTCycleAccurateSim()
        ema = 0
        for _ in range(20):
            sim.update(chunk_id=0, is_promoted=True, importance=1.0)
            _drain_pipeline(sim)
            r = sim.query(0)
            sim.tick()
            ema = r.ema_value

        # After many promotes, EMA should converge close to 65535
        assert ema > 60000, f"Expected convergence toward 65535, got {ema}"


# ---------------------------------------------------------------------------
# 6. Full capacity: update all 1024 entries
# ---------------------------------------------------------------------------

class TestFullCapacity:
    def test_all_entries(self):
        sim = PHTCycleAccurateSim(num_entries=1024)

        for cid in range(1024):
            sim.update(chunk_id=cid, is_promoted=True, importance=1.0)
            sim.tick()

        # Drain remaining pipeline
        _drain_pipeline(sim)

        stats = sim.get_stats()
        assert stats.total_updates == 1024
        assert stats.active_entries == 1024

        # Spot-check a few
        for cid in [0, 511, 1023]:
            r = sim.query(cid)
            assert r.valid is True
            assert r.ema_value == 13107


# ---------------------------------------------------------------------------
# 7. Anchor floor enforcement
# ---------------------------------------------------------------------------

class TestAnchor:
    def test_anchor_prevents_decay_below_floor(self):
        sim = PHTCycleAccurateSim()

        # Promote entry to get a decent EMA
        for _ in range(10):
            sim.update(chunk_id=0, is_promoted=True, importance=1.0)
            _drain_pipeline(sim)

        # Set anchor
        sim.set_anchor(0)

        # Decay many times
        for _ in range(50):
            sim.update(chunk_id=0, is_promoted=False)
            _drain_pipeline(sim)

        r = sim.query(0)
        sim.tick()
        # Should not go below anchor floor 0x826C = 33388
        assert r.ema_value >= 0x826C, f"Expected >= {0x826C}, got {r.ema_value}"
        assert r.anchor is True

    def test_anchor_clear(self):
        sim = PHTCycleAccurateSim()
        sim.set_anchor(5)
        assert sim.peek_entry(5).anchor is True
        sim.clear_anchor(5)
        assert sim.peek_entry(5).anchor is False


# ---------------------------------------------------------------------------
# 8. Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_stats_after_operations(self):
        sim = PHTCycleAccurateSim()

        sim.update(0, True, 1.0)
        sim.tick()  # cycle 1: accept+read entry 0
        sim.update(1, True, 1.0)
        sim.tick()  # cycle 2: compute entry 0, accept+read entry 1
        sim.tick()  # cycle 3: writeback entry 0, compute entry 1
        sim.tick()  # cycle 4: writeback entry 1

        sim.query(0)
        sim.query(1)
        sim.query(999)  # miss (not valid)
        sim.tick()  # cycle 5

        stats = sim.get_stats()
        assert stats.total_updates == 2
        assert stats.total_queries == 3
        assert stats.total_cycles == 5
        assert stats.active_entries == 2
        # 2 hits out of 3 queries
        assert abs(stats.hit_rate - 2 / 3) < 1e-6


# ---------------------------------------------------------------------------
# 9. Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_everything(self):
        sim = PHTCycleAccurateSim()

        sim.update(0, True, 1.0)
        _drain_pipeline(sim)
        sim.query(0)
        sim.tick()

        sim.reset()

        stats = sim.get_stats()
        assert stats.total_cycles == 0
        assert stats.total_queries == 0
        assert stats.total_updates == 0
        assert stats.active_entries == 0

        r = sim.query(0)
        assert r.valid is False
        assert r.ema_value == 0


# ---------------------------------------------------------------------------
# Run with pytest
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
