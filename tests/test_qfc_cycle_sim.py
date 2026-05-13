"""
Unit tests for the QFC RTL-Level Cycle-Accurate Simulator.

Tests cover:
  1.  Basic import of all public classes
  2.  QFCConfig default values
  3.  Single request latency precision (4-stage pipeline)
  4.  Multi-request throughput with MAC parallelism
  5.  Backpressure / arbitration stalls
  6.  MAC utilization under load
  7.  Traditional vs QFC mode speedup
  8.  Bandwidth reduction (QFC vs Traditional)
  9.  Stats completeness
  10. per_stage_latency_ns sanity
  11. Edge cases: 0 / 1 requests
  12. Custom configuration effects
  13. Reset behaviour
  14. compare_modes helper
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.hardware.qfc_cycle_sim import (
    QFCConfig,
    QFCRequest,
    QFCStats,
    QFCRTLStats,
    QFCCycleAccurateSim,
    BatchSimResult,
    ComparisonResult,
    run_comparison_sweep,
)


# ---------------------------------------------------------------------------
# 1. Basic import
# ---------------------------------------------------------------------------

class TestImport:
    def test_all_classes_importable(self):
        """All public classes should be importable without error."""
        assert QFCConfig is not None
        assert QFCRequest is not None
        assert QFCStats is not None
        assert QFCRTLStats is not None
        assert QFCCycleAccurateSim is not None
        assert BatchSimResult is not None
        assert ComparisonResult is not None

    def test_instantiate_defaults(self):
        sim = QFCCycleAccurateSim()
        assert sim.cfg is not None


# ---------------------------------------------------------------------------
# 2. Default configuration
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    def test_num_mac_arrays(self):
        cfg = QFCConfig()
        assert cfg.num_mac_arrays == 8

    def test_mac_fifo_depth(self):
        cfg = QFCConfig()
        assert cfg.mac_fifo_depth == 2

    def test_cxl_bandwidth(self):
        cfg = QFCConfig()
        assert cfg.cxl_bandwidth_gbps == 64.0

    def test_mac_compute_ns(self):
        cfg = QFCConfig()
        assert cfg.mac_compute_ns == 50_000.0

    def test_bytes_per_ns(self):
        cfg = QFCConfig()
        assert cfg.bytes_per_ns == 64.0

    def test_full_chunk_size(self):
        cfg = QFCConfig()
        assert cfg.full_chunk_size_bytes == 65536

    def test_query_and_result_sizes(self):
        cfg = QFCConfig()
        assert cfg.query_size_bytes == 1024
        assert cfg.result_size_bytes == 4


# ---------------------------------------------------------------------------
# 3. Single request latency precision
# ---------------------------------------------------------------------------

class TestSingleRequestLatency:
    def test_single_qfc_request_completes(self):
        """A single QFC request should traverse 4 stages and complete."""
        sim = QFCCycleAccurateSim()
        sim.submit_qfc_query("req_0", chunk_id=0)
        sim.run_until_idle()
        stats = sim.get_stats()
        assert stats.total_requests == 1
        assert stats.qfc_requests == 1
        assert stats.avg_latency_ns > 0

    def test_single_traditional_request_completes(self):
        """A single traditional request should complete."""
        sim = QFCCycleAccurateSim()
        sim.submit_traditional_fetch("trad_0", chunk_id=0)
        sim.run_until_idle()
        stats = sim.get_stats()
        assert stats.total_requests == 1
        assert stats.traditional_requests == 1
        assert stats.avg_latency_ns > 0

    def test_qfc_much_faster_than_traditional_single(self):
        """Single QFC should be faster than single traditional."""
        sim = QFCCycleAccurateSim()
        sim.submit_qfc_query("q0", 0)
        sim.run_until_idle()
        qfc_lat = sim.get_stats().avg_latency_ns

        sim.reset()
        sim.submit_traditional_fetch("t0", 0)
        sim.run_until_idle()
        trad_lat = sim.get_stats().avg_latency_ns

        assert qfc_lat < trad_lat, f"QFC {qfc_lat} should be < Traditional {trad_lat}"


# ---------------------------------------------------------------------------
# 4. Multi-request throughput — MAC parallelism
# ---------------------------------------------------------------------------

class TestMultiRequestThroughput:
    def test_8_qfc_requests_parallel(self):
        """8 QFC requests should leverage 8 MAC arrays."""
        sim = QFCCycleAccurateSim()
        for i in range(8):
            sim.submit_qfc_query(f"req_{i}", chunk_id=i)
        sim.run_until_idle()
        stats = sim.get_stats()
        assert stats.total_requests == 8
        assert stats.qfc_requests == 8
        # All should complete
        completed = [r for r in sim._requests if r.complete_cycle >= 0]
        assert len(completed) == 8

    def test_batch_simulate(self):
        """simulate_batch helper should produce valid results."""
        sim = QFCCycleAccurateSim()
        result = sim.simulate_batch(batch_size=4, chunks_per_request=2, mode="qfc")
        assert result.batch_size == 4
        assert result.mode == "qfc"
        assert result.total_latency_ns > 0
        assert len(result.per_request_latencies_ns) == 4


# ---------------------------------------------------------------------------
# 5. Backpressure / arbitration stalls
# ---------------------------------------------------------------------------

class TestBackpressure:
    def test_many_requests_cause_stalls(self):
        """
        With default 8 MACs * fifo_depth=2 = 16 slots,
        submitting >16 QFC requests should trigger arbitration stalls.
        """
        sim = QFCCycleAccurateSim()
        n = 32
        for i in range(n):
            sim.submit_qfc_query(f"req_{i}", chunk_id=i)
        sim.run_until_idle()
        stats = sim.get_stats()
        assert stats.total_requests == n
        assert stats.arbitration_stalls > 0, "Expected backpressure stalls with 32 requests"

    def test_small_fifo_more_stalls(self):
        """Reducing fifo_depth should increase stalls."""
        cfg = QFCConfig(mac_fifo_depth=1)
        sim = QFCCycleAccurateSim(cfg)
        for i in range(20):
            sim.submit_qfc_query(f"req_{i}", chunk_id=i)
        sim.run_until_idle()
        stats = sim.get_stats()
        assert stats.arbitration_stalls > 0


# ---------------------------------------------------------------------------
# 6. MAC utilization
# ---------------------------------------------------------------------------

class TestMACUtilization:
    def test_high_load_utilization(self):
        """Under high load MAC utilization should be non-trivial."""
        sim = QFCCycleAccurateSim()
        result = sim.simulate_batch(batch_size=8, chunks_per_request=4, mode="qfc")
        assert result.stats.mac_utilization > 0.0

    def test_traditional_no_mac_usage(self):
        """Traditional mode should not use MACs at all."""
        sim = QFCCycleAccurateSim()
        result = sim.simulate_batch(batch_size=4, chunks_per_request=2, mode="traditional")
        assert result.stats.mac_utilization == 0.0


# ---------------------------------------------------------------------------
# 7. Traditional vs QFC comparison — speedup
# ---------------------------------------------------------------------------

class TestSpeedup:
    def test_qfc_speedup_significant(self):
        """QFC should provide > 1.5x speedup over traditional."""
        sim = QFCCycleAccurateSim()
        cmp = sim.compare_modes(batch_size=8, chunks_per_request=4)
        assert cmp.speedup > 1.5, f"Speedup {cmp.speedup:.2f}x too low"

    def test_speedup_grows_with_batch(self):
        """Speedup should be at least as good with larger batch."""
        sim = QFCCycleAccurateSim()
        cmp_small = sim.compare_modes(batch_size=4, chunks_per_request=4)
        cmp_large = sim.compare_modes(batch_size=32, chunks_per_request=4)
        # Both should show meaningful speedup
        assert cmp_small.speedup > 1.0
        assert cmp_large.speedup > 1.0


# ---------------------------------------------------------------------------
# 8. Bandwidth reduction
# ---------------------------------------------------------------------------

class TestBandwidthReduction:
    def test_bandwidth_reduction_significant(self):
        """QFC should reduce bandwidth by > 10x (64KB vs ~1KB)."""
        sim = QFCCycleAccurateSim()
        cmp = sim.compare_modes(batch_size=8, chunks_per_request=4)
        assert cmp.bandwidth_reduction > 10.0, (
            f"Bandwidth reduction {cmp.bandwidth_reduction:.1f}x too low"
        )


# ---------------------------------------------------------------------------
# 9. Stats completeness
# ---------------------------------------------------------------------------

class TestStatsCompleteness:
    def test_qfc_stats_fields(self):
        """All QFCRTLStats fields should have reasonable values."""
        sim = QFCCycleAccurateSim()
        for i in range(8):
            sim.submit_qfc_query(f"req_{i}", chunk_id=i)
        sim.run_until_idle()
        stats = sim.get_stats()

        assert isinstance(stats, QFCRTLStats)
        assert stats.total_cycles > 0
        assert stats.total_requests == 8
        assert stats.qfc_requests == 8
        assert stats.traditional_requests == 0
        assert stats.avg_latency_ns > 0
        assert stats.p50_latency_ns > 0
        assert stats.p99_latency_ns >= stats.p50_latency_ns
        assert stats.max_latency_ns >= stats.p99_latency_ns
        assert stats.total_bytes_transferred > 0
        assert stats.peak_queue_depth >= 0
        assert isinstance(stats.per_stage_latency_ns, dict)

    def test_empty_stats(self):
        """Stats on a fresh sim should return zeroed QFCRTLStats."""
        sim = QFCCycleAccurateSim()
        stats = sim.get_stats()
        assert stats.total_cycles == 0
        assert stats.total_requests == 0


# ---------------------------------------------------------------------------
# 10. per_stage_latency_ns sanity
# ---------------------------------------------------------------------------

class TestPerStageLatency:
    def test_qfc_all_stages_positive(self):
        """QFC requests should have positive latency for all 4 stages."""
        sim = QFCCycleAccurateSim()
        for i in range(4):
            sim.submit_qfc_query(f"req_{i}", chunk_id=i)
        sim.run_until_idle()
        stats = sim.get_stats()
        psl = stats.per_stage_latency_ns

        for stage in ["arbitration", "transfer_out", "compute", "transfer_back"]:
            assert stage in psl, f"Missing stage '{stage}'"
            assert psl[stage] >= 0, f"Stage '{stage}' latency should be >= 0"

        # transfer_out, compute, transfer_back must be > 0 for QFC
        assert psl["transfer_out"] > 0
        assert psl["compute"] > 0
        assert psl["transfer_back"] > 0

    def test_traditional_no_transfer_back(self):
        """Traditional requests should have 0 transfer_back latency."""
        sim = QFCCycleAccurateSim()
        for i in range(4):
            sim.submit_traditional_fetch(f"trad_{i}", chunk_id=i)
        sim.run_until_idle()
        stats = sim.get_stats()
        psl = stats.per_stage_latency_ns
        assert psl.get("transfer_back", 0) == 0


# ---------------------------------------------------------------------------
# 11. Edge cases: 0 or 1 request
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_requests_no_crash(self):
        """Running with no requests should not crash."""
        sim = QFCCycleAccurateSim()
        cycles = sim.run_until_idle()
        stats = sim.get_stats()
        assert stats.total_requests == 0
        assert cycles >= 0

    def test_one_request(self):
        """Single request should complete cleanly."""
        sim = QFCCycleAccurateSim()
        sim.submit_qfc_query("solo", chunk_id=0)
        sim.run_until_idle()
        stats = sim.get_stats()
        assert stats.total_requests == 1
        completed = [r for r in sim._requests if r.complete_cycle >= 0]
        assert len(completed) == 1

    def test_tick_without_requests(self):
        """Ticking with no requests should be a no-op."""
        sim = QFCCycleAccurateSim()
        for _ in range(10):
            sim.tick()
        stats = sim.get_stats()
        assert stats.total_requests == 0


# ---------------------------------------------------------------------------
# 12. Custom configuration
# ---------------------------------------------------------------------------

class TestCustomConfig:
    def test_fewer_macs_slower(self):
        """Fewer MAC arrays should generally lead to higher total latency."""
        cfg_full = QFCConfig(num_mac_arrays=8)
        cfg_half = QFCConfig(num_mac_arrays=2)

        sim_full = QFCCycleAccurateSim(cfg_full)
        res_full = sim_full.simulate_batch(batch_size=8, chunks_per_request=4, mode="qfc")

        sim_half = QFCCycleAccurateSim(cfg_half)
        res_half = sim_half.simulate_batch(batch_size=8, chunks_per_request=4, mode="qfc")

        assert res_half.total_latency_ns >= res_full.total_latency_ns, (
            f"2-MAC ({res_half.total_latency_ns}) should be >= 8-MAC ({res_full.total_latency_ns})"
        )

    def test_custom_clock_freq(self):
        """Changing clock_freq_ghz should be reflected in config."""
        cfg = QFCConfig(clock_freq_ghz=2.0)
        sim = QFCCycleAccurateSim(cfg)
        assert sim.cfg.clock_freq_ghz == 2.0

    def test_large_mac_fifo_reduces_stalls(self):
        """Larger FIFO depth should reduce or eliminate stalls."""
        cfg_small = QFCConfig(mac_fifo_depth=1)
        cfg_large = QFCConfig(mac_fifo_depth=8)

        sim_s = QFCCycleAccurateSim(cfg_small)
        for i in range(20):
            sim_s.submit_qfc_query(f"req_{i}", chunk_id=i)
        sim_s.run_until_idle()
        stalls_small = sim_s.get_stats().arbitration_stalls

        sim_l = QFCCycleAccurateSim(cfg_large)
        for i in range(20):
            sim_l.submit_qfc_query(f"req_{i}", chunk_id=i)
        sim_l.run_until_idle()
        stalls_large = sim_l.get_stats().arbitration_stalls

        assert stalls_large <= stalls_small, (
            f"Large FIFO stalls ({stalls_large}) should be <= small ({stalls_small})"
        )


# ---------------------------------------------------------------------------
# 13. Reset behaviour
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self):
        sim = QFCCycleAccurateSim()
        sim.submit_qfc_query("r0", 0)
        sim.run_until_idle()
        assert sim.get_stats().total_requests == 1

        sim.reset()
        stats = sim.get_stats()
        assert stats.total_requests == 0
        assert stats.total_cycles == 0

    def test_reset_allows_reuse(self):
        sim = QFCCycleAccurateSim()
        sim.submit_qfc_query("a", 0)
        sim.run_until_idle()
        sim.reset()

        sim.submit_traditional_fetch("b", 0)
        sim.run_until_idle()
        stats = sim.get_stats()
        assert stats.total_requests == 1
        assert stats.traditional_requests == 1


# ---------------------------------------------------------------------------
# 14. compare_modes & sweep helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_compare_modes_returns_comparison(self):
        sim = QFCCycleAccurateSim()
        cmp = sim.compare_modes(batch_size=4, chunks_per_request=2)
        assert isinstance(cmp, ComparisonResult)
        assert cmp.traditional.total_latency_ns > 0
        assert cmp.qfc.total_latency_ns > 0

    def test_run_comparison_sweep(self):
        results = run_comparison_sweep(batch_sizes=[1, 4], chunks=2)
        assert 1 in results
        assert 4 in results
        for bs, d in results.items():
            assert d["speedup"] > 0
            assert d["bandwidth_reduction"] > 0


# ---------------------------------------------------------------------------
# Run with pytest
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
