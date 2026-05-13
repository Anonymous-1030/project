"""
Test EvidenceDecomposer: verify 5-dim decomposition from ODUS-X score algebra.

Key invariants:
1. Sum of 5 dimensions equals original score (within floating point tolerance)
2. Temporal dimension is dominant in stable mode
3. Semantic dimension is dominant in reactive mode
4. Mode switching changes dimension distributions
"""

import unittest

import numpy as np

from src.eval.causal.layer1_cei import EvidenceDecomposer
from src.core_types import EvidenceDimension


class TestEvidenceDecomposer(unittest.TestCase):

    def setUp(self):
        self.decomposer = EvidenceDecomposer()
        self.cue_values = {
            "recency": 0.7,
            "ewma": 0.5,
            "window": 0.4,
            "position": 0.6,
            "anchor": 0.3,
            "similarity": 0.8,
            "lexical": 0.6,
            "history": 0.4,
            "pht": 0.3,
            "anchor_bonus": 0.0,
            "promoted": 0.5,
        }

    def test_decomposition_sum_equals_score(self):
        """Verify dim sum equals total score in all three modes."""
        for mode in ["stable", "mixed", "reactive"]:
            ev = self.decomposer.decompose("test_chunk", self.cue_values, mode=mode)
            dim_sum = ev.e_temp + ev.e_struct + ev.e_sem + ev.e_hist + ev.e_press
            self.assertAlmostEqual(
                dim_sum, ev.score, places=5,
                msg=f"Mode {mode}: dim sum {dim_sum:.6f} != score {ev.score:.6f}"
            )

    def test_temporal_dominates_stable_mode(self):
        """Temporal cues should dominate in stable (streaming) mode."""
        ev = self.decomposer.decompose("test_chunk", self.cue_values, mode="stable")
        dims = {
            EvidenceDimension.TEMPORAL: ev.e_temp,
            EvidenceDimension.STRUCTURAL: ev.e_struct,
            EvidenceDimension.SEMANTIC: ev.e_sem,
            EvidenceDimension.HISTORICAL: ev.e_hist,
        }
        dominant = max(dims, key=dims.get)
        self.assertEqual(
            dominant, EvidenceDimension.TEMPORAL,
            f"Expected TEMPORAL dominant in stable mode, got {dominant.value}"
        )

    def test_semantic_dominates_reactive_mode(self):
        """Semantic cues should dominate in reactive (needle/drift) mode."""
        ev = self.decomposer.decompose("test_chunk", self.cue_values, mode="reactive")
        dims = {
            EvidenceDimension.TEMPORAL: ev.e_temp,
            EvidenceDimension.STRUCTURAL: ev.e_struct,
            EvidenceDimension.SEMANTIC: ev.e_sem,
            EvidenceDimension.HISTORICAL: ev.e_hist,
        }
        dominant = max(dims, key=dims.get)
        self.assertEqual(
            dominant, EvidenceDimension.SEMANTIC,
            f"Expected SEMANTIC dominant in reactive mode, got {dominant.value}"
        )

    def test_mode_switching_changes_dimension_distribution(self):
        """Mode switch from stable to reactive should shift weight from temp to sem."""
        ev_stable = self.decomposer.decompose("test", self.cue_values, mode="stable")
        ev_reactive = self.decomposer.decompose("test", self.cue_values, mode="reactive")

        # Temporal should decrease in reactive mode
        self.assertLess(ev_reactive.e_temp, ev_stable.e_temp)

        # Semantic should increase in reactive mode
        self.assertGreater(ev_reactive.e_sem, ev_stable.e_sem)

    def test_batch_decomposition(self):
        """Batch decomposition should produce same results as individual."""
        chunk_ids = [f"c{i}" for i in range(10)]
        cue_batch = [self.cue_values.copy() for _ in range(10)]
        modes = ["stable"] * 10

        batch_results = self.decomposer.decompose_batch(chunk_ids, cue_batch, modes)
        individual_results = [
            self.decomposer.decompose(cid, cues, mode)
            for cid, cues, mode in zip(chunk_ids, cue_batch, modes)
        ]

        for i, (batch, indiv) in enumerate(zip(batch_results, individual_results)):
            self.assertAlmostEqual(batch.score, indiv.score, places=5)

    def test_budget_pressure_contribution(self):
        """Budget pressure should contribute to e_press dimension."""
        ev_no_pressure = self.decomposer.decompose(
            "test", self.cue_values, mode="stable", budget_pressure=0.0
        )
        ev_high_pressure = self.decomposer.decompose(
            "test", self.cue_values, mode="stable", budget_pressure=0.9
        )

        self.assertGreater(ev_high_pressure.e_press, ev_no_pressure.e_press)

    def test_global_means_computation(self):
        """Global means should be sensible."""
        evs = self.decomposer.decompose_batch(
            [f"c{i}" for i in range(20)],
            [{k: float(np.random.beta(2, 5)) for k in self.cue_values} for _ in range(20)],
            ["stable"] * 20,
        )

        means = self.decomposer.compute_global_means(evs)
        for dim in EvidenceDimension:
            self.assertGreaterEqual(means[dim], 0.0)
            self.assertLessEqual(means[dim], 1.0)

    def test_dominant_dimension_identifies_largest(self):
        """EvidenceVector.dominant_dimension() should identify the max dim."""
        from src.core_types import EvidenceVector
        ev = EvidenceVector(
            chunk_id="test",
            e_temp=0.3,
            e_struct=0.1,
            e_sem=0.5,
            e_hist=0.2,
            e_press=0.0,
            score=1.1,
            mode="stable",
        )
        self.assertEqual(ev.dominant_dimension(), EvidenceDimension.SEMANTIC)

    def test_all_dimensions_non_negative(self):
        """All evidence dimensions should be non-negative."""
        for mode in ["stable", "mixed", "reactive"]:
            ev = self.decomposer.decompose("test", self.cue_values, mode=mode)
            self.assertGreaterEqual(ev.e_temp, 0.0)
            self.assertGreaterEqual(ev.e_struct, 0.0)
            self.assertGreaterEqual(ev.e_sem, 0.0)
            self.assertGreaterEqual(ev.e_hist, 0.0)
            self.assertGreaterEqual(ev.e_press, 0.0)


if __name__ == "__main__":
    unittest.main()
