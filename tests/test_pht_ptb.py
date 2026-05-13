"""Unit tests for PHT (Promotion History Table) and PTB (Promotion Target Buffer)."""

import unittest
import numpy as np

from src.core_types import ChunkMetadata, ChunkTier, ChunkState, QueryContext
from src.config import PPUConfig
from src.hardware.ppu.pht import PHTConfig, PHTEntry, PromotionHistoryTable
from src.hardware.ppu.ptb import PTBConfig, PromotionTargetBuffer
from src.hardware.ppu.pht_ptb_cacti import PHTCACTIModel
from src.hardware.ppu.pht_ptb_integration import (
    PHTAugmentedFeatureExtractor,
    PHTAugmentedPPU,
)
from src.hardware.ppu.pht_ptb_simulator import PHTAugmentedCycleSimulator


def _make_chunk(chunk_id: str, position_ratio: float = 0.5,
                promoted_count: int = 0, access_count: int = 0) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=chunk_id,
        request_id="req_0",
        token_start=int(position_ratio * 1000),
        token_end=int(position_ratio * 1000) + 512,
        position_ratio=position_ratio,
        num_tokens=512,
        logical_bytes=4096,
        signature=np.random.randn(64).astype(np.float32),
        signature_hex=None,
        is_section_boundary=False,
        is_title_adjacent=False,
        is_code_block=False,
        section_id=None,
        tier=ChunkTier.TAIL,
        state=ChunkState.ACTIVE,
        creation_step=0,
        last_access_step=0,
        last_promotion_step=0,
        promoted_count=promoted_count,
        access_count=access_count,
        sticky_ttl=0,
        sticky_original_ttl=0,
        extra={},
    )


def _make_query(step: int = 0) -> QueryContext:
    return QueryContext(
        request_id="req_0",
        step=step,
        query_summary=np.random.randn(64).astype(np.float32),
        query_tokens=None,
        query_text=None,
        query_signature=np.random.randn(64).astype(np.float32),
        extracted_entities=None,
        query_length=32,
        active_anchor_ids=["anchor_0", "anchor_1"],
        recent_anchor_ids=["anchor_0"],
        steps_since_start=step,
        generation_length=0,
    )


class TestPHTEntry(unittest.TestCase):
    """Test PHT entry saturating counter behavior."""

    def test_initial_state(self):
        entry = PHTEntry(counter=1)
        self.assertFalse(entry.predict_promote())
        self.assertEqual(entry.counter, 1)

    def test_saturation_upper(self):
        entry = PHTEntry(counter=3)
        entry.update(was_useful=True)
        self.assertEqual(entry.counter, 3)  # saturates at 3

    def test_saturation_lower(self):
        entry = PHTEntry(counter=0)
        entry.update(was_useful=False)
        self.assertEqual(entry.counter, 0)  # saturates at 0

    def test_increment(self):
        entry = PHTEntry(counter=1)
        entry.update(was_useful=True)
        self.assertEqual(entry.counter, 2)
        self.assertTrue(entry.predict_promote())

    def test_decrement(self):
        entry = PHTEntry(counter=2)
        entry.update(was_useful=False)
        self.assertEqual(entry.counter, 1)
        self.assertFalse(entry.predict_promote())

    def test_confidence(self):
        # Extremes should have high confidence
        self.assertGreater(PHTEntry(counter=0).confidence, 0.3)
        self.assertGreater(PHTEntry(counter=3).confidence, 0.3)
        # Middle values should have lower confidence
        self.assertLess(PHTEntry(counter=1).confidence, 0.5)


class TestPromotionHistoryTable(unittest.TestCase):
    """Test PHT prediction and update logic."""

    def setUp(self):
        self.config = PHTConfig(num_entries=256, counter_bits=2)
        self.pht = PromotionHistoryTable(self.config)
        self.chunk = _make_chunk("chunk_0", position_ratio=0.3)
        self.query = _make_query(step=10)

    def test_signature_determinism(self):
        sig1 = self.pht.compute_signature(self.chunk, self.query)
        sig2 = self.pht.compute_signature(self.chunk, self.query)
        self.assertEqual(sig1, sig2)

    def test_signature_range(self):
        sig = self.pht.compute_signature(self.chunk, self.query)
        self.assertGreaterEqual(sig, 0)
        self.assertLess(sig, self.config.num_entries)

    def test_initial_prediction_conservative(self):
        result = self.pht.predict(self.chunk, self.query)
        # Initial counter=1, threshold=2 → should NOT predict promote
        self.assertFalse(result.predict_promote)

    def test_update_to_promote(self):
        # Update twice with was_useful=True: 1→2→3
        self.pht.update(self.chunk, self.query, was_useful=True)
        self.pht.update(self.chunk, self.query, was_useful=True)
        result = self.pht.predict(self.chunk, self.query)
        self.assertTrue(result.predict_promote)

    def test_update_to_not_promote(self):
        # Update with was_useful=False: 1→0
        self.pht.update(self.chunk, self.query, was_useful=False)
        result = self.pht.predict(self.chunk, self.query)
        self.assertFalse(result.predict_promote)

    def test_feature_value_range(self):
        feat = self.pht.get_prediction_feature(self.chunk, self.query)
        self.assertGreaterEqual(feat, 0.0)
        self.assertLessEqual(feat, 1.0)

    def test_batch_predict(self):
        chunks = [_make_chunk(f"chunk_{i}", i / 10.0) for i in range(5)]
        results = self.pht.batch_predict(chunks, self.query)
        self.assertEqual(len(results), 5)

    def test_stats(self):
        self.pht.predict(self.chunk, self.query)
        stats = self.pht.stats()
        self.assertEqual(stats["total_predictions"], 1)
        self.assertEqual(stats["num_entries"], 256)

    def test_decay(self):
        # Push counter to 3
        self.pht.update(self.chunk, self.query, was_useful=True)
        self.pht.update(self.chunk, self.query, was_useful=True)
        sig = self.pht.compute_signature(self.chunk, self.query)
        self.assertEqual(self.pht._table[sig].counter, 3)
        # Decay should bring 3→2
        self.pht.decay_all()
        self.assertEqual(self.pht._table[sig].counter, 2)


class TestPromotionTargetBuffer(unittest.TestCase):
    """Test PTB lookup, insert, eviction."""

    def setUp(self):
        self.config = PTBConfig(num_entries=4, max_age_steps=10)
        self.ptb = PromotionTargetBuffer(self.config)

    def test_empty_lookup(self):
        result = self.ptb.lookup(signature=42)
        self.assertFalse(result.hit)

    def test_insert_and_lookup(self):
        self.ptb.insert(42, "chunk_0", 1000, 0.9, step=5)
        result = self.ptb.lookup(42)
        self.assertTrue(result.hit)
        self.assertEqual(result.entry.chunk_id, "chunk_0")

    def test_invalidate(self):
        self.ptb.insert(42, "chunk_0", 1000, 0.9, step=5)
        self.ptb.invalidate(42)
        result = self.ptb.lookup(42)
        self.assertFalse(result.hit)

    def test_lru_eviction(self):
        # Fill all 4 entries
        for i in range(4):
            self.ptb.insert(i, f"chunk_{i}", i * 100, 0.5, step=i)
        # Insert 5th → should evict LRU (entry 0)
        self.ptb.insert(99, "chunk_99", 9900, 0.8, step=10)
        result = self.ptb.lookup(0)
        self.assertFalse(result.hit)
        result = self.ptb.lookup(99)
        self.assertTrue(result.hit)

    def test_age_eviction(self):
        self.ptb.insert(42, "chunk_0", 1000, 0.9, step=0)
        evicted = self.ptb.age_entries(current_step=100)
        self.assertIn("chunk_0", evicted)
        result = self.ptb.lookup(42)
        self.assertFalse(result.hit)

    def test_speculative_targets(self):
        self.ptb.insert(1, "chunk_1", 100, 0.9, step=5)
        self.ptb.insert(2, "chunk_2", 200, 0.5, step=5)
        targets = self.ptb.get_speculative_prefetch_targets(current_step=6)
        self.assertEqual(len(targets), 2)
        # Should be sorted by utility descending
        self.assertEqual(targets[0].chunk_id, "chunk_1")

    def test_stats(self):
        self.ptb.insert(1, "chunk_1", 100, 0.9, step=0)
        self.ptb.lookup(1)
        self.ptb.lookup(999)  # miss
        stats = self.ptb.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["lookups"], 2)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)


class TestPHTCACTIModel(unittest.TestCase):
    """Test area/power estimation."""

    def test_area_within_budget(self):
        pht_cfg = PHTConfig(num_entries=1024, counter_bits=2)
        ptb_cfg = PTBConfig(num_entries=32)
        model = PHTCACTIModel(pht_cfg, ptb_cfg)
        report = model.estimate()
        self.assertTrue(report.fits_budget)
        self.assertLess(report.total_area_mm2, 0.03)

    def test_power_reasonable(self):
        pht_cfg = PHTConfig()
        ptb_cfg = PTBConfig()
        model = PHTCACTIModel(pht_cfg, ptb_cfg)
        report = model.estimate()
        self.assertLess(report.total_power_mw, 20.0)
        self.assertGreater(report.total_power_mw, 0.0)


class TestPHTAugmentedPPU(unittest.TestCase):
    """Test the 5-stage augmented PPU pipeline."""

    def setUp(self):
        self.ppu_config = PPUConfig(lut_index_bits=10)
        self.pht_config = PHTConfig(num_entries=256)
        self.ptb_config = PTBConfig(num_entries=8)
        self.ppu = PHTAugmentedPPU(
            self.ppu_config, self.pht_config, self.ptb_config
        )
        self.chunk = _make_chunk("chunk_0")
        self.query = _make_query(step=5)
        self.all_chunks = {"chunk_0": self.chunk}

    def test_process_candidate(self):
        result = self.ppu.process_candidate(
            self.chunk, self.query, self.all_chunks, attention_mass=0.5
        )
        self.assertEqual(result.chunk_id, "chunk_0")
        self.assertIn("pht_prediction", result.metadata)
        self.assertIn("ptb_hit", result.metadata)

    def test_end_step_updates_pht(self):
        # Process and promote
        self.ppu.process_candidate(
            self.chunk, self.query, self.all_chunks,
            attention_mass=0.5, enqueue_threshold=0.0,
        )
        # End step: chunk was accessed
        stats = self.ppu.end_step(
            promoted_chunk_ids=["chunk_0"],
            accessed_chunk_ids=["chunk_0"],
            query=self.query,
            all_chunks=self.all_chunks,
        )
        self.assertGreater(stats["pht_updates"], 0)


class TestCycleSimulator(unittest.TestCase):
    """Test the 5-stage cycle simulator."""

    def test_pipeline_timing(self):
        config = PPUConfig()
        sim = PHTAugmentedCycleSimulator(
            config, PHTConfig(), PTBConfig()
        )
        # 5 stages, each 1 cycle → pipeline depth = 5
        self.assertEqual(sim.pipeline_depth, 5)
        self.assertEqual(sim.ii, 1)

    def test_latency_scaling(self):
        config = PPUConfig(clock_frequency_ghz=1.5)
        sim = PHTAugmentedCycleSimulator(
            config, PHTConfig(), PTBConfig()
        )
        # 10 candidates: (10-1)*1 + 5 = 14 cycles
        # At 1.5 GHz: 14 / 1.5 ≈ 9.33 ns
        latency = sim.latency_ns(10)
        self.assertAlmostEqual(latency, 14 / 1.5, places=2)

    def test_simulate_trace(self):
        config = PPUConfig()
        sim = PHTAugmentedCycleSimulator(
            config, PHTConfig(), PTBConfig()
        )
        chunks = [_make_chunk(f"c_{i}", i / 5.0) for i in range(5)]
        query = _make_query(step=0)
        all_chunks = {c.chunk_id: c for c in chunks}

        trace = sim.simulate(query, chunks, all_chunks)
        self.assertEqual(trace.num_candidates, 5)
        self.assertGreater(trace.total_cycles, 0)
        # 5 stages × 5 candidates = 25 events
        self.assertEqual(len(trace.events), 25)


if __name__ == "__main__":
    unittest.main()
