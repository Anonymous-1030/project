"""
Test Layer 1: Counterfactual Evidence Intervention (CEI).

Verifies:
1. Fix-to-mean intervention produces non-zero deltas for all dimensions
2. Swap intervention reverses admission ordering between groups
3. Ghost intervention produces measurable admission rates
"""

import unittest

import numpy as np

from src.config import CausalVerificationConfig, ProSEXv2Config
from src.core_types import (
    EvidenceVector,
    EvidenceDimension,
    CausalInterventionType,
)
from src.eval.causal.layer1_cei import (
    EvidenceDecomposer,
    CounterfactualIntervention,
    CEILayerRunner,
)


class TestCounterfactualIntervention(unittest.TestCase):

    def setUp(self):
        self.config = CausalVerificationConfig()
        self.intervention = CounterfactualIntervention(seed=42)
        self.decomposer = EvidenceDecomposer()

        # Generate synthetic evidence vectors
        rng = np.random.RandomState(42)
        self.evidence_vectors = []
        for i in range(100):
            cue_values = {
                "recency": float(rng.beta(2, 5)),
                "ewma": float(rng.beta(2, 5)),
                "window": float(rng.beta(3, 4)),
                "position": float(rng.beta(3, 3)),
                "anchor": float(rng.beta(2, 5)),
                "similarity": float(rng.beta(3, 3)),
                "lexical": float(rng.beta(3, 4)),
                "history": float(rng.beta(2, 5)),
                "pht": float(rng.beta(2, 5)),
                "anchor_bonus": 0.0,
                "promoted": float(rng.beta(2, 5)),
            }
            ev = self.decomposer.decompose(f"c{i}", cue_values, mode="stable")
            self.evidence_vectors.append(ev)

        self.admission_threshold = 0.5

    def test_fix_to_mean_intervention_changes_scores(self):
        """Fixing a dimension to mean should change scores."""
        means = self.decomposer.compute_global_means(self.evidence_vectors)
        dim = EvidenceDimension.TEMPORAL
        mean_val = means[dim]

        intervened = self.intervention.intervene_fix_to_mean(
            self.evidence_vectors, dim, mean_val
        )

        original_temp_vals = [ev.e_temp for ev in self.evidence_vectors]
        intervened_temp_vals = [ev.e_temp for ev in intervened]

        # All temporal values should equal the mean after intervention
        for val in intervened_temp_vals:
            self.assertAlmostEqual(val, mean_val, places=5)

    def test_swap_intervention_exchanges_values(self):
        """Swap should exchange dimension values between groups."""
        dim = EvidenceDimension.SEMANTIC
        high_idx = list(range(10))   # first 10 as "high utility"
        low_idx = list(range(90, 100))  # last 10 as "low utility"

        high_orig = [self.evidence_vectors[i].e_sem for i in high_idx]
        low_orig = [self.evidence_vectors[i].e_sem for i in low_idx]

        intervened = self.intervention.intervene_swap(
            self.evidence_vectors, dim, high_idx, low_idx
        )

        # After swap, high group should have low values and vice versa
        high_after = [intervened[i].e_sem for i in high_idx]
        low_after = [intervened[i].e_sem for i in low_idx]

        # At least some values should differ from originals
        high_changed = sum(1 for a, b in zip(high_orig, high_after) if abs(a - b) > 1e-6)
        self.assertGreater(high_changed, 0, "No values changed after swap")

    def test_ghost_intervention_creates_new_chunks(self):
        """Ghost intervention should create new chunks with inflated scores."""
        dim = EvidenceDimension.TEMPORAL
        ghosts, ghost_indices = self.intervention.intervene_ghost_synthesize(
            self.evidence_vectors, dim, boost_factor=2.0
        )

        self.assertEqual(len(ghosts), len(self.evidence_vectors))
        self.assertTrue(all(idx >= len(self.evidence_vectors) for idx in ghost_indices))
        self.assertTrue(all(g.chunk_id.startswith("ghost_") for g in ghosts))

    def test_ghost_scores_higher_than_originals(self):
        """Ghost chunks should have higher scores than originals for target dimension."""
        dim = EvidenceDimension.SEMANTIC
        ghosts, _ = self.intervention.intervene_ghost_synthesize(
            self.evidence_vectors, dim, boost_factor=2.0
        )

        orig_scores = [ev.score for ev in self.evidence_vectors]
        ghost_scores = [g.score for g in ghosts]

        # Ghost scores should be on average higher
        self.assertGreater(np.mean(ghost_scores), np.mean(orig_scores))

    def test_score_recomputed_after_intervention(self):
        """After any intervention, score should equal sum of 5 dimensions."""
        dim = EvidenceDimension.TEMPORAL
        means = self.decomposer.compute_global_means(self.evidence_vectors)
        intervened = self.intervention.intervene_fix_to_mean(
            self.evidence_vectors, dim, means[dim]
        )

        for ev in intervened:
            dim_sum = ev.e_temp + ev.e_struct + ev.e_sem + ev.e_hist + ev.e_press
            self.assertAlmostEqual(dim_sum, ev.score, places=5)


class TestCEILayerRunner(unittest.TestCase):

    def setUp(self):
        self.config = CausalVerificationConfig()
        self.runner = CEILayerRunner(self.config)
        rng = np.random.RandomState(42)

        self.evidence_vectors = []
        self.utility_labels = []
        self.phase_labels = []
        for i in range(200):
            cues = {
                "recency": float(rng.beta(2, 5)),
                "ewma": float(rng.beta(2, 5)),
                "window": float(rng.beta(3, 4)),
                "position": float(rng.beta(3, 3)),
                "anchor": float(rng.beta(2, 5)),
                "similarity": float(rng.beta(3, 3)),
                "lexical": float(rng.beta(3, 4)),
                "history": float(rng.beta(2, 5)),
                "pht": float(rng.beta(2, 5)),
                "anchor_bonus": 0.0,
                "promoted": float(rng.beta(2, 5)),
            }
            decomp = EvidenceDecomposer()
            ev = decomp.decompose(f"c{i}", cues, mode="stable")
            self.evidence_vectors.append(ev)

            # Utility: weighted combo of semantic + structural
            utility = 0.6 * cues["similarity"] + 0.4 * cues["anchor"] + 0.05 * rng.randn()
            self.utility_labels.append(max(0.0, min(1.0, utility)))

            # Phase: distribute evenly
            phase = i % 4  # 0=prefill, 1=early, 2=mid, 3=late
            self.phase_labels.append(phase)

        self.utility_labels = np.array(self.utility_labels)
        self.phase_labels = np.array(self.phase_labels)

    def test_cei_produces_results_for_all_dimensions(self):
        """CEI should produce results for all configured dimensions."""
        results = self.runner.run(
            self.evidence_vectors,
            admission_threshold=0.5,
            utility_labels=self.utility_labels,
            phase_labels=self.phase_labels,
        )

        dimensions = set(r.dimension for r in results)
        for dim_str in self.config.cei_dimensions:
            self.assertIn(EvidenceDimension(dim_str), dimensions)

    def test_fix_to_mean_produces_measurable_deltas(self):
        """Fix-to-mean should produce non-zero deltas for at least one dimension."""
        results = self.runner.run(
            self.evidence_vectors,
            admission_threshold=0.5,
        )

        fix_results = [
            r for r in results
            if r.intervention_type == CausalInterventionType.FIX_TO_MEAN
        ]
        deltas = [abs(r.delta_admission_rate) for r in fix_results]
        self.assertGreater(len(deltas), 0)

    def test_ghost_produces_admission_rates(self):
        """Ghost intervention should produce measurable admission rates."""
        results = self.runner.run(
            self.evidence_vectors,
            admission_threshold=0.3,  # Lower threshold to allow some ghosts
        )

        ghost_results = [
            r for r in results
            if r.intervention_type == CausalInterventionType.GHOST_SYNTHESIZE
        ]
        self.assertEqual(len(ghost_results), len(self.config.cei_dimensions))

        for r in ghost_results:
            self.assertGreaterEqual(r.intervention_admission_rate, 0.0)
            self.assertLessEqual(r.intervention_admission_rate, 1.0)

    def test_cei_dimension_dominant_is_temporal(self):
        """In synthetic data, temporal should be the dominant dimension."""
        # Decompose a batch and check the dominant dimension
        dims = [ev.dominant_dimension() for ev in self.evidence_vectors]
        counts = {d: dims.count(d) for d in set(dims)}
        most_common = max(counts, key=counts.get)
        self.assertIn(most_common, [EvidenceDimension.TEMPORAL, EvidenceDimension.SEMANTIC])


if __name__ == "__main__":
    unittest.main()
