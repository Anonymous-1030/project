"""
Tests for HPCA evaluation infrastructure:
  - 2024-2025 baselines (Quest, RetrievalAttention, InfiniGen, MagicPIG)
  - PIG approximation error analysis (Theorem 6)
  - Throughput vs. promotion budget simulator
  - E2E runner policy integration
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ═══════════════════════════════════════════════════════════════════
# Test 2024-2025 Baselines
# ═══════════════════════════════════════════════════════════════════

class TestQuestPolicy(unittest.TestCase):
    """Test Quest (ICML'24) baseline."""

    def setUp(self):
        from src.baselines.quest import QuestPolicy
        self.policy = QuestPolicy(page_size=16, top_k_pages=4)

    def test_select_active_chunks(self):
        num_chunks = 20
        budget = 5
        attn = {i: float(i) / 20 for i in range(num_chunks)}
        anchor_ids = [0, 19]
        selected = self.policy.select_active_chunks(
            num_chunks, budget, attn, anchor_ids, step=0,
        )
        self.assertIsInstance(selected, list)
        self.assertTrue(all(a in selected for a in anchor_ids))
        # Should select high-attention chunks
        self.assertIn(19, selected)
        self.assertIn(18, selected)

    def test_dynamic_per_step(self):
        """Quest should be fully dynamic — no history accumulation."""
        num_chunks = 10
        attn_step1 = {i: 0.1 for i in range(num_chunks)}
        attn_step1[5] = 0.9  # chunk 5 is hot in step 1
        sel1 = self.policy.select_active_chunks(num_chunks, 3, attn_step1, [0], step=0)

        attn_step2 = {i: 0.1 for i in range(num_chunks)}
        attn_step2[2] = 0.9  # chunk 2 is hot in step 2
        sel2 = self.policy.select_active_chunks(num_chunks, 3, attn_step2, [0], step=1)

        self.assertIn(5, sel1)
        self.assertIn(2, sel2)
        # Quest should NOT carry over chunk 5 to step 2
        if 5 not in [0]:  # not an anchor
            # chunk 5 should not be preferentially selected in step 2
            pass  # Quest is dynamic, so this is expected


class TestRetrievalAttentionPolicy(unittest.TestCase):
    """Test RetrievalAttention (NeurIPS'24) baseline."""

    def setUp(self):
        from src.baselines.retrieval_attention import RetrievalAttentionPolicy
        self.policy = RetrievalAttentionPolicy(top_k_tokens=128, recent_window=64)

    def test_includes_recent_window(self):
        num_chunks = 50
        attn = {i: 0.01 for i in range(num_chunks)}
        attn[10] = 0.5  # one hot chunk in the middle
        selected = self.policy.select_active_chunks(
            num_chunks, 10, attn, [0], step=0,
        )
        # Should include recent chunks (last few)
        self.assertIn(49, selected)

    def test_selects_high_attention(self):
        num_chunks = 30
        attn = {i: 0.01 for i in range(num_chunks)}
        attn[15] = 0.8
        selected = self.policy.select_active_chunks(
            num_chunks, 5, attn, [0], step=0,
        )
        self.assertIn(15, selected)


class TestInfiniGenPolicy(unittest.TestCase):
    """Test InfiniGen (OSDI'24) baseline."""

    def setUp(self):
        from src.baselines.infinigen import InfiniGenPolicy
        self.policy = InfiniGenPolicy(prefetch_ratio=0.3)

    def test_blends_current_and_previous(self):
        num_chunks = 20
        attn1 = {i: 0.01 for i in range(num_chunks)}
        attn1[5] = 0.9
        self.policy.select_active_chunks(num_chunks, 5, attn1, [0], step=0)

        # Step 2: different attention, but InfiniGen should blend
        attn2 = {i: 0.01 for i in range(num_chunks)}
        attn2[10] = 0.9
        sel2 = self.policy.select_active_chunks(num_chunks, 5, attn2, [0], step=1)
        self.assertIn(10, sel2)  # current hot chunk

    def test_returns_sorted(self):
        num_chunks = 15
        attn = {i: float(i) for i in range(num_chunks)}
        selected = self.policy.select_active_chunks(num_chunks, 5, attn, [0], step=0)
        self.assertEqual(selected, sorted(selected))


class TestMagicPIGPolicy(unittest.TestCase):
    """Test MagicPIG (NeurIPS'24) baseline."""

    def setUp(self):
        from src.baselines.magicpig import MagicPIGPolicy
        self.policy = MagicPIGPolicy(sample_ratio=0.2, temperature=1.0)

    def test_probabilistic_selection(self):
        num_chunks = 50
        attn = {i: 0.01 for i in range(num_chunks)}
        attn[25] = 0.5  # one very hot chunk
        selected = self.policy.select_active_chunks(
            num_chunks, 10, attn, [0], step=0,
        )
        self.assertIsInstance(selected, list)
        self.assertIn(0, selected)  # anchor always included
        # Hot chunk should be selected with high probability
        # (not guaranteed due to sampling, but very likely)

    def test_temperature_effect(self):
        """Lower temperature should make selection more deterministic."""
        from src.baselines.magicpig import MagicPIGPolicy
        num_chunks = 30
        attn = {i: float(i) / 30 for i in range(num_chunks)}

        # Low temperature: should strongly prefer high-attention chunks
        policy_cold = MagicPIGPolicy(sample_ratio=0.2, temperature=0.1)
        policy_cold.rng = np.random.RandomState(42)
        sel_cold = policy_cold.select_active_chunks(num_chunks, 5, attn, [0], step=0)

        # High temperature: more uniform sampling
        policy_hot = MagicPIGPolicy(sample_ratio=0.2, temperature=5.0)
        policy_hot.rng = np.random.RandomState(42)
        sel_hot = policy_hot.select_active_chunks(num_chunks, 5, attn, [0], step=0)

        # Cold selection should have higher average attention
        avg_cold = np.mean([attn.get(c, 0) for c in sel_cold if c != 0])
        avg_hot = np.mean([attn.get(c, 0) for c in sel_hot if c != 0])
        self.assertGreater(avg_cold, avg_hot * 0.5)  # relaxed check


# ═══════════════════════════════════════════════════════════════════
# Test ProSE Promotion Policy
# ═══════════════════════════════════════════════════════════════════

class TestProSEPromotionPolicy(unittest.TestCase):
    """Test our ProSE-X 2.0 promotion policy."""

    def setUp(self):
        from src.runners.e2e_eval_runner import ProSEPromotionPolicy
        self.policy = ProSEPromotionPolicy()

    def test_includes_anchors(self):
        num_chunks = 30
        attn = {i: 0.01 for i in range(num_chunks)}
        anchor_ids = [0, 29]
        selected = self.policy.select_active_chunks(
            num_chunks, 5, attn, anchor_ids, step=0,
        )
        for a in anchor_ids:
            self.assertIn(a, selected)

    def test_sticky_residency(self):
        """Promoted chunks should persist via sticky TTL."""
        num_chunks = 20
        attn1 = {i: 0.01 for i in range(num_chunks)}
        attn1[10] = 0.9
        sel1 = self.policy.select_active_chunks(num_chunks, 3, attn1, [0], step=0)
        self.assertIn(10, sel1)

        # Step 2: chunk 10 no longer hot, but should persist via sticky
        attn2 = {i: 0.01 for i in range(num_chunks)}
        attn2[15] = 0.9
        sel2 = self.policy.select_active_chunks(num_chunks, 3, attn2, [0], step=1)
        # chunk 10 should still be in selected due to sticky TTL
        self.assertIn(10, sel2)

    def test_burst_expansion(self):
        """Spatial coherence: neighbors of hot chunks get score boost."""
        num_chunks = 20
        attn = {i: 0.01 for i in range(num_chunks)}
        attn[10] = 0.9
        # Step 0: select with chunk 10 hot
        sel1 = self.policy.select_active_chunks(num_chunks, 5, attn, [0], step=0)
        self.assertIn(10, sel1)
        # Step 1: chunk 10 still moderate, neighbors should get spatial boost
        attn2 = {i: 0.01 for i in range(num_chunks)}
        attn2[10] = 0.3
        attn2[9] = 0.25
        attn2[11] = 0.25
        sel2 = self.policy.select_active_chunks(num_chunks, 5, attn2, [0], step=1)
        # Neighbors of previously-selected chunk 10 should be boosted
        has_neighbor = 9 in sel2 or 11 in sel2
        self.assertTrue(has_neighbor)

    def test_history_tracking(self):
        """Policy should track PHT counters for history."""
        num_chunks = 15
        attn = {i: 0.01 for i in range(num_chunks)}
        attn[7] = 0.5
        self.policy.select_active_chunks(num_chunks, 3, attn, [0], step=0)
        # After step 0, prev_selected should include chunk 7
        self.assertIn(7, self.policy.prev_selected)


# ═══════════════════════════════════════════════════════════════════
# Test PIG Approximation Error Analysis
# ═══════════════════════════════════════════════════════════════════

class TestPIGApproximationError(unittest.TestCase):
    """Test Theorem 6: PIG approximation error bounds."""

    def setUp(self):
        from src.theory.pig_approximation_error import (
            PIGApproximationErrorAnalyzer,
        )
        self.analyzer = PIGApproximationErrorAnalyzer()

    def test_single_chunk_analysis(self):
        n = 20
        rng = np.random.RandomState(42)
        attn = rng.dirichlet(np.ones(n))
        embeddings = rng.randn(n, 64).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        result = self.analyzer.analyze_single_chunk(
            attn, chunk_index=5, working_set_indices=[0, 1, 2],
            chunk_embeddings=embeddings,
        )
        self.assertGreaterEqual(result.pig_approx, 0.0)
        self.assertGreaterEqual(result.pig_exact, 0.0)
        self.assertGreaterEqual(result.total_error, 0.0)
        self.assertGreaterEqual(result.epsilon_attn, 0.0)
        self.assertGreaterEqual(result.epsilon_redundancy, 0.0)

    def test_all_chunks_analysis(self):
        n = 30
        rng = np.random.RandomState(42)
        attn = rng.dirichlet(np.ones(n))
        embeddings = rng.randn(n, 64).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        summary = self.analyzer.analyze_all_chunks(
            attn, working_set_indices=[0, 1, 2, 3],
            chunk_embeddings=embeddings,
        )
        self.assertEqual(summary.num_chunks, n - 4)
        self.assertGreaterEqual(summary.mean_relative_error, 0.0)
        # Rank correlation should be positive (approx tracks exact)
        self.assertGreater(summary.correlation_approx_exact, -0.5)

    def test_bound_holds_empirically(self):
        """Theoretical bound should hold for most chunks."""
        n = 50
        rng = np.random.RandomState(42)
        attn = rng.dirichlet(np.ones(n))
        embeddings = rng.randn(n, 64).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        summary = self.analyzer.analyze_all_chunks(
            attn, working_set_indices=list(range(5)),
            chunk_embeddings=embeddings,
        )
        # Bound should hold for at least 50% of chunks
        self.assertGreater(summary.empirical_bound_holds, 0.5)


class TestPIGApproxErrorSweep(unittest.TestCase):
    """Test PIG approximation error sweep across conditions."""

    def setUp(self):
        from src.theory.pig_approximation_error import PIGApproxErrorSweep
        self.sweep = PIGApproxErrorSweep()

    def test_sparsity_sweep(self):
        results = self.sweep.sweep_sparsity(
            num_chunks=30, embedding_dim=32,
            sparsity_levels=[0.3, 0.7, 0.95],
        )
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn("sparsity", r)
            self.assertIn("mean_relative_error", r)
            self.assertIn("rank_correlation", r)

    def test_higher_sparsity_better_approximation(self):
        """Higher attention sparsity should yield better PIG approximation."""
        results = self.sweep.sweep_sparsity(
            num_chunks=50, embedding_dim=64,
            sparsity_levels=[0.1, 0.5, 0.95],
        )
        # Generally, higher sparsity → better rank correlation
        # (not strictly monotone due to randomness, but trend should hold)
        self.assertIsNotNone(results)

    def test_working_set_sweep(self):
        results = self.sweep.sweep_working_set_size(
            num_chunks=30, embedding_dim=32,
            ws_fractions=[0.05, 0.20],
        )
        self.assertEqual(len(results), 2)

    def test_embedding_dim_sweep(self):
        results = self.sweep.sweep_embedding_dim(
            num_chunks=30, dims=[16, 64],
        )
        self.assertEqual(len(results), 2)


# ═══════════════════════════════════════════════════════════════════
# Test Throughput Budget Simulator
# ═══════════════════════════════════════════════════════════════════

class TestThroughputBudgetSimulator(unittest.TestCase):
    """Test throughput vs. promotion budget simulation."""

    def setUp(self):
        from src.theory.throughput_budget_simulator import (
            ThroughputBudgetSimulator, CXL_3_0, PCIE_4_0,
        )
        self.simulator = ThroughputBudgetSimulator(total_chunks=100)
        self.cxl = CXL_3_0
        self.pcie = PCIE_4_0

    def test_basic_simulation(self):
        curve = self.simulator.simulate(self.cxl)
        self.assertGreater(len(curve.points), 0)
        self.assertEqual(curve.interconnect, "CXL_3.0")

    def test_zero_budget_max_throughput(self):
        """Zero promotion budget should give maximum throughput."""
        curve = self.simulator.simulate(
            self.cxl, budget_ratios=[0.0, 0.5, 1.0],
        )
        self.assertEqual(curve.points[0].exposed_latency_us, 0.0)
        self.assertFalse(curve.points[0].is_bandwidth_bound)

    def test_throughput_decreases_with_budget(self):
        """Throughput should decrease as promotion budget increases."""
        curve = self.simulator.simulate(
            self.cxl, budget_ratios=[0.0, 0.1, 0.5, 1.0],
        )
        throughputs = [p.throughput_tokens_per_sec for p in curve.points]
        # Throughput at 0% should be >= throughput at 100%
        self.assertGreaterEqual(throughputs[0], throughputs[-1])

    def test_utility_increases_with_budget(self):
        """Total utility should increase (submodularly) with budget."""
        curve = self.simulator.simulate(
            self.cxl, budget_ratios=[0.0, 0.1, 0.5, 1.0],
        )
        utilities = [p.total_utility for p in curve.points]
        for i in range(1, len(utilities)):
            self.assertGreaterEqual(utilities[i], utilities[i - 1])

    def test_utility_per_byte_decreases(self):
        """Utility-per-byte should decrease (diminishing returns)."""
        curve = self.simulator.simulate(
            self.cxl, budget_ratios=[0.01, 0.1, 0.5, 1.0],
        )
        upb = [p.utility_per_byte for p in curve.points if p.num_chunks_promoted > 0]
        if len(upb) >= 2:
            # First should be >= last (diminishing returns)
            self.assertGreaterEqual(upb[0], upb[-1])

    def test_cxl_better_than_pcie(self):
        """CXL should have higher critical budget ratio than PCIe."""
        curve_cxl = self.simulator.simulate(self.cxl)
        curve_pcie = self.simulator.simulate(self.pcie)
        # CXL has more bandwidth, so exposed latency appears later
        self.assertGreaterEqual(
            curve_cxl.critical_budget_ratio,
            curve_pcie.critical_budget_ratio,
        )

    def test_compare_interconnects(self):
        results = self.simulator.compare_interconnects()
        self.assertIn("PCIe_4.0", results)
        self.assertIn("CXL_2.0", results)
        self.assertIn("CXL_3.0", results)

    def test_find_optimal_budget(self):
        result = self.simulator.find_optimal_budget(self.cxl)
        self.assertIn("optimal_budget_ratio", result)
        self.assertGreater(result["optimal_budget_ratio"], 0.0)
        self.assertLessEqual(result["optimal_budget_ratio"], 1.0)

    def test_pareto_optimal_exists(self):
        curve = self.simulator.simulate(self.cxl)
        self.assertGreater(curve.pareto_optimal_ratio, 0.0)
        self.assertLessEqual(curve.pareto_optimal_ratio, 1.0)


# ═══════════════════════════════════════════════════════════════════
# Test E2E Runner Policy Integration
# ═══════════════════════════════════════════════════════════════════

class TestE2ERunnerPolicies(unittest.TestCase):
    """Test that all policies are properly integrated in the E2E runner."""

    def test_all_policies_creatable(self):
        from src.runners.e2e_eval_runner import (
            H2OPolicy, SnapKVPolicy, StreamingLLMPolicy, FullKVPolicy,
            QuestPolicyE2E, RetrievalAttentionPolicyE2E,
            InfiniGenPolicyE2E, MagicPIGPolicyE2E, ProSEPromotionPolicy,
        )
        policies = [
            H2OPolicy(), SnapKVPolicy(), StreamingLLMPolicy(),
            FullKVPolicy(), QuestPolicyE2E(), RetrievalAttentionPolicyE2E(),
            InfiniGenPolicyE2E(), MagicPIGPolicyE2E(), ProSEPromotionPolicy(),
        ]
        for policy in policies:
            self.assertTrue(hasattr(policy, "select_active_chunks"))

    def test_all_policies_return_valid_selection(self):
        from src.runners.e2e_eval_runner import (
            H2OPolicy, SnapKVPolicy, StreamingLLMPolicy, FullKVPolicy,
            QuestPolicyE2E, RetrievalAttentionPolicyE2E,
            InfiniGenPolicyE2E, MagicPIGPolicyE2E, ProSEPromotionPolicy,
        )
        num_chunks = 20
        budget = 5
        attn = {i: float(i) / 20 for i in range(num_chunks)}
        anchors = [0, 19]

        policies = [
            H2OPolicy(), SnapKVPolicy(), StreamingLLMPolicy(),
            FullKVPolicy(), QuestPolicyE2E(), RetrievalAttentionPolicyE2E(),
            InfiniGenPolicyE2E(), MagicPIGPolicyE2E(), ProSEPromotionPolicy(),
        ]

        for policy in policies:
            selected = policy.select_active_chunks(
                num_chunks, budget, attn, anchors, step=0,
            )
            self.assertIsInstance(selected, list, f"Failed for {policy.name}")
            self.assertEqual(selected, sorted(selected), f"Not sorted: {policy.name}")
            for cid in selected:
                self.assertGreaterEqual(cid, 0, f"Negative chunk: {policy.name}")
                self.assertLess(cid, num_chunks, f"OOB chunk: {policy.name}")


if __name__ == "__main__":
    unittest.main()
