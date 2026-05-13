"""Validation tests for the PPU stack."""

import unittest
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import ProSEXv2Config
from src.core_types import ChunkMetadata, QueryContext, ChunkTier
from src.hardware.ppu import (
    PromotionPredictionUnit,
    CACTIModel,
    PPUCycleSimulator,
    LUTDistiller,
    PPUIntegratedPromotionPipeline,
)


class TestPPUStack(unittest.TestCase):
    def setUp(self):
        self.config = ProSEXv2Config()
        self.config.eabs.max_chunks_per_step = 2
        self.config.eabs.exploration_ratio = 0.0
        self.config.ppu.integration_mode = "standalone"
        self.query = QueryContext(
            request_id="req",
            step=10,
            query_signature=np.ones(4, dtype=np.float32),
            query_tokens=[1, 2, 3],
        )
        self.anchor = ChunkMetadata(
            chunk_id="anchor",
            request_id="req",
            token_start=0,
            token_end=511,
            position_ratio=0.0,
            num_tokens=512,
            logical_bytes=4096,
            tier=ChunkTier.ANCHOR,
            signature=np.ones(4, dtype=np.float32),
        )
        self.tail_chunks = [
            ChunkMetadata(
                chunk_id="tail_0",
                request_id="req",
                token_start=512,
                token_end=1023,
                position_ratio=0.2,
                num_tokens=512,
                logical_bytes=4096,
                tier=ChunkTier.TAIL,
                signature=np.ones(4, dtype=np.float32),
                last_access_step=9,
                promoted_count=2,
                access_count=2,
                extra={"token_ids": [1, 2, 3]},
            ),
            ChunkMetadata(
                chunk_id="tail_1",
                request_id="req",
                token_start=1024,
                token_end=1535,
                position_ratio=0.8,
                num_tokens=512,
                logical_bytes=4096,
                tier=ChunkTier.TAIL,
                signature=np.zeros(4, dtype=np.float32),
                last_access_step=0,
                promoted_count=1,
                access_count=0,
                extra={"token_ids": [9, 10]},
            ),
        ]

    def test_ppu_core_process_candidate(self):
        # Dynamically size LUT and fill with a linear ramp so every index
        # produces a distinct, non-trivial utility value.
        n = 1 << self.config.ppu.lut_index_bits
        lut_values = [int(i * 255 / max(n - 1, 1)) for i in range(n)]
        ppu = PromotionPredictionUnit(self.config.ppu, lut_values=lut_values)
        all_chunks = {c.chunk_id: c for c in [self.anchor] + self.tail_chunks}
        result = ppu.process_candidate(self.tail_chunks[0], self.query, all_chunks, attention_mass=0.9)
        self.assertGreaterEqual(result.utility, 0.0)
        self.assertLessEqual(result.utility, 1.0)
        self.assertIsInstance(result.lut_index, int)
        # With a ramp LUT and non-trivial features, utility should be > 0
        self.assertGreater(result.utility, 0.0, "Ramp LUT should produce non-zero utility for valid features")

    def test_cacti_model_report(self):
        report = CACTIModel(self.config.ppu).estimate()
        self.assertGreater(report.total_area_mm2, 0.0)
        self.assertGreater(report.total_power_mw, 0.0)
        self.assertGreater(len(report.components), 0)

    def test_lut_distillation(self):
        distiller = LUTDistiller(self.config.ppu)
        features = np.array([
            [1.0, 1.0, 0.1, 1.0],
            [0.0, 0.2, 0.9, 0.0],
            [0.7, 0.8, 0.2, 0.5],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.1, 0.7], dtype=np.float32)
        lut, report = distiller.distill(features, scores)
        self.assertEqual(lut.shape[0], 1 << self.config.ppu.lut_index_bits)
        self.assertEqual(report.num_samples, 3)

    def test_cycle_simulator(self):
        sim = PPUCycleSimulator(self.config.ppu)
        all_chunks = {c.chunk_id: c for c in [self.anchor] + self.tail_chunks}
        trace = sim.simulate(self.query, self.tail_chunks, all_chunks, attention_masses={"tail_0": 1.0})
        self.assertEqual(trace.processed_candidates, 2)
        self.assertGreater(trace.total_cycles, 0)

    def test_ppu_pipeline_runs(self):
        pipe = PPUIntegratedPromotionPipeline(self.config)
        result = pipe.run(
            self.query,
            tail_chunks=self.tail_chunks,
            anchor_chunks=[self.anchor],
            promoted_chunks=[],
            budget_bytes=8192,
            attention_masses={"tail_0": 1.0, "tail_1": 0.1},
        )
        self.assertIsNotNone(result.scorer_result)
        self.assertIn(result.scorer_result.scorer_mode, ("ppu_lut", "ppu_pht_ptb"))
        self.assertIsNotNone(getattr(result, "ppu_trace", None))


if __name__ == "__main__":
    unittest.main()