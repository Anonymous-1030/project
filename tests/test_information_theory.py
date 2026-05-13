"""Unit tests for the Information-Theoretic Framework (Theorems 1-5)."""

import unittest
import math
import numpy as np

from src.theory.promotion_information_theory import (
    PromotionInformationGain,
    TemporalPromotionHardness,
    KPTWInstance,
    CompetitiveRatioAnalysis,
    PHTAccuracyBound,
    PHTRegretImprovement,
)
from src.theory.pig_estimator import PIGEstimator, PIGEvaluator


class TestPIG(unittest.TestCase):
    """Test Theorem 1: Promotion Information Gain."""

    def setUp(self):
        self.pig = PromotionInformationGain(vocab_size=32000)
        self.rng = np.random.RandomState(42)
        self.n_chunks = 20
        # Random attention distribution (normalized)
        raw = self.rng.exponential(1.0, self.n_chunks)
        self.attention = raw / raw.sum()
        # Random embeddings
        self.embeddings = self.rng.randn(self.n_chunks, 64).astype(np.float32)

    def test_pig_non_negative(self):
        """Property P1: PIG(c, W) >= 0."""
        for c in range(self.n_chunks):
            ws = [i for i in range(self.n_chunks) if i != c][:5]
            result = self.pig.compute_pig(
                self.attention, c, ws, self.embeddings
            )
            self.assertGreaterEqual(result.pig_value, -1e-10)

    def test_pig_bounded(self):
        """Property P4: Σ PIG(c, W) ≤ H(Y)."""
        ws = []
        total_pig = 0.0
        for c in range(self.n_chunks):
            result = self.pig.compute_pig(
                self.attention, c, ws, self.embeddings
            )
            total_pig += result.pig_value
        # PIG values are attention-mass based, so sum ≤ 1.0
        self.assertLessEqual(total_pig, 1.0 + 1e-6)

    def test_pig_submodularity(self):
        """Property P2: PIG(c, W) ≥ PIG(c, W') for W ⊆ W'."""
        result = self.pig.verify_submodularity(
            self.attention, self.embeddings, num_samples=200, seed=42
        )
        self.assertEqual(result["violations"], 0,
                         f"Submodularity violated {result['violations']} times")

    def test_pig_zero_for_zero_attention(self):
        """Property P3: PIG = 0 when chunk has no attention."""
        attn = np.zeros(self.n_chunks)
        attn[0] = 1.0  # all attention on chunk 0
        result = self.pig.compute_pig(attn, 5, [0], self.embeddings)
        self.assertAlmostEqual(result.pig_value, 0.0, places=6)

    def test_pig_set_computation(self):
        results = self.pig.compute_pig_set(
            self.attention, [0, 1, 2], [3, 4], self.embeddings
        )
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertGreaterEqual(r.pig_value, 0.0)


class TestNPHardness(unittest.TestCase):
    """Test Theorem 2: NP-Hardness reduction."""

    def setUp(self):
        self.hardness = TemporalPromotionHardness()

    def test_reduction_correctness(self):
        """Verify KPTW → TKVP reduction produces valid instance."""
        kptw = KPTWInstance(
            num_items=5,
            profits=[10.0, 20.0, 15.0, 25.0, 30.0],
            weights=[3.0, 5.0, 4.0, 6.0, 7.0],
            time_windows=[(0, 2), (1, 3), (0, 4), (2, 4), (3, 4)],
            capacity=10.0,
            num_timesteps=5,
        )
        tkvp = self.hardness.construct_reduction(kptw)

        self.assertEqual(tkvp.num_chunks, 5)
        self.assertEqual(tkvp.num_steps, 5)
        self.assertEqual(tkvp.bandwidth_budget, 10.0)

        # Check utility matrix: item 0 has profit 10 in steps 0-2
        self.assertAlmostEqual(tkvp.utilities[0, 0], 10.0)
        self.assertAlmostEqual(tkvp.utilities[2, 0], 10.0)
        self.assertAlmostEqual(tkvp.utilities[3, 0], 0.0)  # outside window

    def test_greedy_solver(self):
        """Test greedy solver produces feasible solution."""
        kptw = KPTWInstance(
            num_items=3,
            profits=[10.0, 20.0, 15.0],
            weights=[5.0, 5.0, 5.0],
            time_windows=[(0, 2), (0, 2), (0, 2)],
            capacity=10.0,
            num_timesteps=3,
        )
        tkvp = self.hardness.construct_reduction(kptw)
        solution, total = self.hardness.solve_greedy(tkvp)

        self.assertEqual(len(solution), 3)
        self.assertGreater(total, 0.0)

        # Verify bandwidth constraint
        for t, s_t in enumerate(solution):
            used = sum(tkvp.transfer_costs[c] for c in s_t)
            self.assertLessEqual(used, tkvp.bandwidth_budget + 1e-10)

    def test_reduction_verification(self):
        """Verify solution maps back correctly."""
        kptw = KPTWInstance(
            num_items=3,
            profits=[10.0, 20.0, 15.0],
            weights=[5.0, 5.0, 5.0],
            time_windows=[(0, 2), (0, 2), (0, 2)],
            capacity=10.0,
            num_timesteps=3,
        )
        tkvp = self.hardness.construct_reduction(kptw)
        solution, _ = self.hardness.solve_greedy(tkvp)

        result = self.hardness.verify_reduction(kptw, solution)
        self.assertTrue(result["valid"])
        self.assertEqual(result["bandwidth_violations"], 0)


class TestCompetitiveRatio(unittest.TestCase):
    """Test Theorem 3: Tight competitive ratio."""

    def setUp(self):
        self.analysis = CompetitiveRatioAnalysis()
        self.rng = np.random.RandomState(42)

    def test_theoretical_ratio(self):
        self.assertAlmostEqual(
            self.analysis.theoretical_ratio, 1.0 - 1.0 / math.e, places=4
        )

    def test_greedy_ratio_above_bound(self):
        """Greedy should achieve at least 1-1/e of optimal."""
        n = 15
        attn = self.rng.exponential(1.0, n)
        attn /= attn.sum()
        emb = self.rng.randn(n, 32).astype(np.float32)

        result = self.analysis.compute_greedy_ratio(
            attn, emb, budget_chunks=3
        )
        self.assertTrue(result["ratio_exceeds_bound"],
                        f"Ratio {result['empirical_ratio']:.4f} < "
                        f"bound {self.analysis.theoretical_ratio:.4f}")

    def test_lower_bound_demonstration(self):
        """Max-k-Cover instances should approach 1-1/e."""
        result = self.analysis.verify_lower_bound(
            num_elements=20, num_sets=30, k=3, seed=42
        )
        # Greedy ratio should be close to 1-1/e (within tolerance)
        self.assertGreater(result["ratio"], 0.5)


class TestPHTAccuracyBound(unittest.TestCase):
    """Test Theorem 4: PHT prediction accuracy bound."""

    def setUp(self):
        self.bound = PHTAccuracyBound()

    def test_bound_components(self):
        result = self.bound.compute_bound(
            delta_locality=0.7,
            num_entries=1024,
            counter_bits=2,
            num_active_chunks=100,
        )
        self.assertGreater(result["intrinsic_component"], 0.0)
        self.assertGreater(result["aliasing_component"], 0.0)
        self.assertGreater(result["quantization_component"], 0.0)
        self.assertLessEqual(result["misprediction_bound"], 1.0)

    def test_more_entries_reduces_aliasing(self):
        r1 = self.bound.compute_bound(0.7, 256, 2, 100)
        r2 = self.bound.compute_bound(0.7, 4096, 2, 100)
        self.assertGreater(
            r1["aliasing_component"], r2["aliasing_component"]
        )

    def test_more_bits_reduces_quantization(self):
        r1 = self.bound.compute_bound(0.7, 1024, 1, 100)
        r2 = self.bound.compute_bound(0.7, 1024, 4, 100)
        self.assertGreater(
            r1["quantization_component"], r2["quantization_component"]
        )

    def test_adaptation_reduces_error(self):
        r1 = self.bound.compute_bound(0.7, 1024, 2, 100, adaptation_steps=0)
        r2 = self.bound.compute_bound(0.7, 1024, 2, 100, adaptation_steps=20)
        self.assertGreater(
            r1["misprediction_bound"], r2["misprediction_bound"]
        )

    def test_empirical_validation(self):
        outcomes = [True, True, False, True, True, False, True, True, True, False]
        predictions = [True, True, True, True, False, False, True, True, False, False]
        result = self.bound.empirical_validation(outcomes, predictions)
        self.assertEqual(result["num_samples"], 10)
        self.assertGreater(result["empirical_rate"], 0.0)


class TestPHTRegretImprovement(unittest.TestCase):
    """Test Theorem 5: PHT-augmented regret improvement."""

    def setUp(self):
        self.regret = PHTRegretImprovement()

    def test_pht_reduces_regret(self):
        T, V_T = 10000, 5000.0
        base = self.regret.compute_regret_without_pht(T, V_T)
        pht = self.regret.compute_regret_with_pht(T, V_T, misprediction_rate=0.3)
        self.assertLess(pht, base)

    def test_perfect_pht(self):
        """m=0 → regret = O(√(T log T)), much less than baseline."""
        T, V_T = 10000, 5000.0
        result = self.regret.compute_improvement_ratio(T, V_T, 0.0)
        self.assertGreater(result["reduction_percent"], 50.0)

    def test_useless_pht(self):
        """m=1 → regret = baseline (no improvement)."""
        T, V_T = 10000, 5000.0
        result = self.regret.compute_improvement_ratio(T, V_T, 1.0)
        self.assertAlmostEqual(result["ratio"], 1.0, places=1)

    def test_sweep(self):
        results = self.regret.sweep_misprediction_rates(T=1000, V_T=500.0)
        self.assertEqual(len(results), 11)
        # Reduction should decrease as m increases
        reductions = [r["reduction_percent"] for r in results]
        self.assertGreater(reductions[0], reductions[-1])


class TestPIGEstimator(unittest.TestCase):
    """Test practical PIG estimation."""

    def setUp(self):
        self.estimator = PIGEstimator()
        self.rng = np.random.RandomState(42)
        self.n = 10
        raw = self.rng.exponential(1.0, self.n)
        self.attn = raw / raw.sum()
        self.emb = self.rng.randn(self.n, 32).astype(np.float32)

    def test_rank_by_pig(self):
        ranked = self.estimator.rank_by_pig(
            self.attn, list(range(5)), [5, 6], self.emb
        )
        self.assertEqual(len(ranked), 5)
        # Should be sorted descending
        for i in range(len(ranked) - 1):
            self.assertGreaterEqual(ranked[i][1], ranked[i + 1][1])

    def test_optimal_set(self):
        result = self.estimator.compute_pig_optimal_set(
            self.attn, list(range(8)), [8, 9], budget_chunks=3,
            chunk_embeddings=self.emb,
        )
        self.assertLessEqual(result.budget_used, 3)
        self.assertGreater(result.total_pig, 0.0)


class TestPIGEvaluator(unittest.TestCase):
    """Test PIG evaluation across steps."""

    def test_aggregate(self):
        evaluator = PIGEvaluator()
        rng = np.random.RandomState(42)
        n = 10

        for step in range(5):
            raw = rng.exponential(1.0, n)
            attn = raw / raw.sum()
            emb = rng.randn(n, 32).astype(np.float32)
            evaluator.evaluate_step(
                step=step,
                attention_masses=attn,
                actual_promoted_indices=[0, 1],
                candidate_indices=list(range(8)),
                working_set_indices=[8, 9],
                budget_chunks=3,
                chunk_embeddings=emb,
            )

        agg = evaluator.aggregate()
        self.assertEqual(agg["num_steps"], 5)
        self.assertGreater(agg["total_pig_achieved"], 0.0)
        self.assertGreaterEqual(agg["avg_step_efficiency"], 0.0)
        self.assertLessEqual(agg["avg_step_efficiency"], 1.0)


if __name__ == "__main__":
    unittest.main()
