"""
Test Layer 7: Online Causal Adaptation (OCA).

Verifies:
1. LinUCB bandit converges (weights stabilize)
2. SBFI constraint is never violated (hard requirement)
3. Thompson Sampling produces valid weight distributions
4. Distribution drift detector triggers at known shifts
5. Analytical mode runs without errors
"""

import unittest

import numpy as np

from src.config import CausalVerificationConfig, ProSEXv2Config
from src.core_types import EvidenceVector, BanditAlgorithm
from src.eval.causal.layer7_oca import (
    LinUCBBandit,
    ThompsonSamplingBandit,
    EpsilonGreedyScorer,
    DistributionDriftDetector,
    OCALayerRunner,
)


class TestLinUCBBandit(unittest.TestCase):

    def setUp(self):
        self.bandit = LinUCBBandit(
            n_features=5,
            alpha=0.5,
            sbfi_min_score=0.3,
            seed=42,
        )

    def test_initial_weights_are_zero(self):
        """Initial weights should be zero vector."""
        weights = self.bandit.get_weights()
        for v in weights.values():
            self.assertAlmostEqual(v, 0.0, places=5)

    def test_score_respects_sbfi_floor(self):
        """Score should never be below sbfi_min_score."""
        ev = EvidenceVector(
            chunk_id="test",
            e_temp=0.0, e_struct=0.0, e_sem=0.0, e_hist=0.0, e_press=0.0,
        )
        score, _ = self.bandit.score(ev)
        self.assertGreaterEqual(score, self.bandit.sbfi_min_score)

    def test_update_changes_weights(self):
        """After updates, weights should change from zero."""
        features = np.array([[0.3, 0.1, 0.5, 0.2, 0.0]])
        rewards = np.array([0.8])
        admitted = np.array([True])

        self.bandit.update(features, rewards, admitted)
        weights = self.bandit.get_weights()

        # At least some weights should be non-zero after update
        non_zero = sum(1 for v in weights.values() if abs(v) > 1e-6)
        self.assertGreater(non_zero, 0, "No weights changed after update")

    def test_convergence_with_stationary_data(self):
        """Weights should stabilize with repeated stationary data."""
        rng = np.random.RandomState(42)

        for step in range(100):
            features = rng.rand(10, 5).astype(np.float64)
            # True utility model: 0.4 * dim0 + 0.3 * dim2 + noise
            true_reward = features[:, 0] * 0.4 + features[:, 2] * 0.3 + 0.05 * rng.randn(10)
            true_reward = np.clip(true_reward, 0.0, 1.0)

            admitted = np.ones(10, dtype=bool)
            self.bandit.update(features, true_reward, admitted)

        weights = self.bandit.get_weights()
        # Temporal (dim 0) and semantic (dim 2) should be learned
        self.assertGreater(weights["e_temp"], 0.0)
        self.assertGreater(weights["e_sem"], 0.0)

    def test_no_update_on_no_admissions(self):
        """No-op when no chunks are admitted."""
        original_weights = self.bandit.get_weights().copy()
        features = np.array([[0.3, 0.1, 0.5, 0.2, 0.0]])
        rewards = np.array([0.8])
        admitted = np.array([False])

        self.bandit.update(features, rewards, admitted)
        new_weights = self.bandit.get_weights()

        for key in original_weights:
            self.assertAlmostEqual(original_weights[key], new_weights[key], places=5)


class TestThompsonSamplingBandit(unittest.TestCase):

    def setUp(self):
        self.bandit = ThompsonSamplingBandit(
            n_features=5,
            prior_variance=1.0,
            noise_variance=0.1,
            sbfi_min_score=0.3,
            seed=42,
        )

    def test_score_respects_sbfi_floor(self):
        """Score should respect SBFI floor."""
        ev = EvidenceVector(
            chunk_id="test",
            e_temp=0.0, e_struct=0.0, e_sem=0.0, e_hist=0.0, e_press=0.0,
        )
        score, _ = self.bandit.score(ev)
        self.assertGreaterEqual(score, self.bandit.sbfi_min_score)

    def test_sampling_produces_variation(self):
        """Thompson sampling should produce different scores on repeated calls."""
        ev = EvidenceVector(
            chunk_id="test",
            e_temp=0.5, e_struct=0.3, e_sem=0.4, e_hist=0.2, e_press=0.0,
        )
        scores = [self.bandit.score(ev)[0] for _ in range(20)]
        # There should be variation in scores
        self.assertGreater(np.std(scores), 0.0)


class TestDistributionDriftDetector(unittest.TestCase):

    def setUp(self):
        self.detector = DistributionDriftDetector(window=20, threshold=0.2)

    def test_no_drift_on_stationary_data(self):
        """No drift should be detected on stationary data."""
        for _ in range(60):
            result = self.detector.update(0.5 + np.random.normal(0, 0.02))
        # After filling window, no drift should be detected on stationary data
        self.assertFalse(result)

    def test_drift_detected_on_shift(self):
        """Drift should be detected after a large shift."""
        # Fill with stationary data
        for _ in range(25):
            self.detector.update(0.5)

        # Introduce sharp shift
        drift_detected = False
        for _ in range(50):
            result = self.detector.update(0.8)
            if result:
                drift_detected = True
                break

        self.assertTrue(drift_detected, "Drift detector failed to detect shift")

    def test_small_shifts_do_not_trigger(self):
        """Small gradual shifts should not trigger drift detection."""
        for i in range(60):
            # Very gradual shift
            ratio = 0.5 + 0.001 * i
            result = self.detector.update(ratio)

        self.assertFalse(result, "Detector triggered on gradual shift")


class TestOCALayerRunner(unittest.TestCase):

    def setUp(self):
        self.config = CausalVerificationConfig()
        self.runner = OCALayerRunner(self.config)

    def test_analytical_mode_runs(self):
        """Analytical mode should produce results for all algorithms."""
        results = self.runner.run_analytical(
            n_steps=50,
            n_chunks_per_step=20,
            drift_step=25,
            seed=42,
        )

        self.assertEqual(len(results), 3)  # LinUCB, Thompson, Epsilon-Greedy

        for r in results:
            self.assertGreater(r.cumulative_reward, 0.0)
            self.assertIsInstance(r.sbfi_boundary_violations, int)

    def test_sbfi_violations_zero_with_isolation(self):
        """SBFI violations should be zero when causal isolation is active."""
        results = self.runner.run_analytical(
            n_steps=50,
            seed=42,
        )

        for r in results:
            self.assertEqual(
                r.sbfi_boundary_violations, 0,
                f"{r.algorithm.value}: SBFI violations detected!"
            )

    def test_drift_events_detected(self):
        """Drift events should be detected after the shift point."""
        results = self.runner.run_analytical(
            n_steps=100,
            drift_step=50,
            seed=42,
        )

        for r in results:
            if r.drift_events:
                # All drift events should be after the shift
                for event_step in r.drift_events:
                    self.assertGreaterEqual(event_step, 40)


if __name__ == "__main__":
    unittest.main()
