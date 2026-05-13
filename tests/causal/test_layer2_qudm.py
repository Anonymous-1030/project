"""
Test Layer 2: Query-Utility Disentanglement Matrix (QUDM).

Verifies:
1. Four quadrants partition correctly
2. QCDR is in [0, 1]
3. Locality trap quadrant has measurable admission rate
4. Analytical mode produces consistent results
"""

import unittest

import numpy as np

from src.config import CausalVerificationConfig
from src.core_types import QuadrantLabel
from src.eval.causal.layer2_qudm import (
    ReuseMetricCalculator,
    QuadrantPartitioner,
    QUDMLayerRunner,
)
from src.eval.causal.causal_metrics import compute_qcdr


class TestQuadrantPartitioner(unittest.TestCase):

    def setUp(self):
        self.partitioner = QuadrantPartitioner(
            reuse_quantile=0.5,
            utility_quantile=0.5,
        )
        rng = np.random.RandomState(42)
        self.reuse_scores = rng.beta(3, 3, size=200)
        self.utility_scores = rng.beta(3, 3, size=200)

    def test_all_chunks_classified(self):
        """Every chunk should be assigned to exactly one quadrant."""
        quadrants, labels = self.partitioner.partition(
            self.reuse_scores, self.utility_scores
        )

        total = sum(len(idxs) for idxs in quadrants.values())
        self.assertEqual(total, len(self.reuse_scores))

        # Verify no overlap
        all_indices = set()
        for idxs in quadrants.values():
            for idx in idxs:
                self.assertNotIn(idx, all_indices)
                all_indices.add(idx)

    def test_qcdr_bounded(self):
        """QCDR should always be in [0, 1]."""
        for i in range(10):
            alpha_2 = np.random.random()
            alpha_3 = np.random.random()
            qcdr = compute_qcdr(alpha_2, alpha_3)
            self.assertGreaterEqual(qcdr, 0.0)
            self.assertLessEqual(qcdr, 1.0)

    def test_qcdr_zero_when_no_locality_traps(self):
        """QCDR should be 0 when there are no locality trap admissions."""
        qcdr = compute_qcdr(0.0, 0.5)
        self.assertEqual(qcdr, 0.0)

    def test_qcdr_one_when_no_long_range(self):
        """QCDR should be 1 when long-range admissions are zero."""
        qcdr = compute_qcdr(0.5, 0.0)
        self.assertEqual(qcdr, 1.0)

    def test_quadrant_labels_correct(self):
        """Each label should correspond to the correct quadrant type."""
        # Verify named labels match QuadrantLabel enum
        expected_labels = {
            "high_reuse_high_utility",
            "high_reuse_low_utility",
            "low_reuse_high_utility",
            "low_reuse_low_utility",
        }
        actual_labels = {q.value for q in QuadrantLabel}
        self.assertEqual(actual_labels, expected_labels)


class TestQUDMLayerRunner(unittest.TestCase):

    def setUp(self):
        self.config = CausalVerificationConfig()
        self.runner = QUDMLayerRunner(self.config)

    def test_analytical_mode_produces_valid_metrics(self):
        """Analytical mode should produce valid quadrant metrics."""
        metrics, qcdr, pass_fail = self.runner.run_analytical(
            num_chunks=100, seed=42
        )

        self.assertEqual(len(metrics), 4)  # 4 quadrants
        self.assertGreaterEqual(qcdr, 0.0)
        self.assertLessEqual(qcdr, 1.0)
        self.assertIsInstance(pass_fail, bool)

        for m in metrics:
            self.assertGreaterEqual(m.admission_rate, 0.0)
            self.assertLessEqual(m.admission_rate, 1.0)
            self.assertGreaterEqual(m.num_chunks, 0)

    def test_high_reuse_high_admission(self):
        """High-reuse chunks should have higher admission rates."""
        metrics, _, _ = self.runner.run_analytical(num_chunks=200, seed=42)

        high_r_high_u = next(
            m for m in metrics if m.quadrant == QuadrantLabel.HIGH_R_HIGH_U
        )
        low_r_low_u = next(
            m for m in metrics if m.quadrant == QuadrantLabel.LOW_R_LOW_U
        )

        # Hot quadrants should be admitted more than cold quadrants
        self.assertGreaterEqual(
            high_r_high_u.admission_rate,
            low_r_low_u.admission_rate,
        )

    def test_reproducible_with_same_seed(self):
        """Same seed should produce same results."""
        metrics1, qcdr1, _ = self.runner.run_analytical(seed=42)
        metrics2, qcdr2, _ = self.runner.run_analytical(seed=42)

        self.assertAlmostEqual(qcdr1, qcdr2, places=5)

        for m1, m2 in zip(metrics1, metrics2):
            self.assertAlmostEqual(m1.admission_rate, m2.admission_rate, places=5)
            self.assertEqual(m1.num_chunks, m2.num_chunks)


if __name__ == "__main__":
    unittest.main()
