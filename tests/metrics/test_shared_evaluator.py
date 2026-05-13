"""
Unit tests for Shared Evaluator and metrics.

Ensures metric computations are correct and consistent.
"""

import unittest
from src.eval.shared.metrics import (
    compute_conditional_recovery,
    compute_useful_promote_ratio,
    compute_burst_gain,
    compute_candidate_recall_at_k,
    compute_budget_utilization,
    compute_latency_statistics,
)
from src.eval.shared.evaluator import SharedEvaluator


class TestConditionalRecovery(unittest.TestCase):
    """Test conditional recovery computation."""
    
    def test_perfect_recovery(self):
        """Test when gold is always recovered."""
        step_results = [
            {"gold": {"a"}, "visible": {"a", "b"}},
            {"gold": {"a"}, "visible": {"a", "c"}},
        ]
        
        def gold_accessor(step):
            return step["gold"]
        
        def visibility_accessor(step):
            return step["visible"]
        
        result = compute_conditional_recovery(
            step_results, gold_accessor, visibility_accessor
        )
        
        self.assertEqual(result.conditional_recovery, 1.0)
        self.assertEqual(result.steps_with_gold, 2)
        self.assertEqual(result.steps_with_recovery, 2)
    
    def test_no_recovery(self):
        """Test when gold is never recovered."""
        step_results = [
            {"gold": {"a"}, "visible": {"b", "c"}},
            {"gold": {"a"}, "visible": {"d", "e"}},
        ]
        
        def gold_accessor(step):
            return step["gold"]
        
        def visibility_accessor(step):
            return step["visible"]
        
        result = compute_conditional_recovery(
            step_results, gold_accessor, visibility_accessor
        )
        
        self.assertEqual(result.conditional_recovery, 0.0)
        self.assertEqual(result.steps_with_recovery, 0)
        self.assertEqual(result.steps_with_miss, 2)
    
    def test_partial_recovery(self):
        """Test partial recovery."""
        step_results = [
            {"gold": {"a"}, "visible": {"a", "b"}},  # recovered
            {"gold": {"a"}, "visible": {"b", "c"}},  # not recovered
        ]
        
        def gold_accessor(step):
            return step["gold"]
        
        def visibility_accessor(step):
            return step["visible"]
        
        result = compute_conditional_recovery(
            step_results, gold_accessor, visibility_accessor
        )
        
        self.assertEqual(result.conditional_recovery, 0.5)
        self.assertEqual(result.steps_with_gold, 2)
        self.assertEqual(result.steps_with_recovery, 1)
    
    def test_no_gold_steps_excluded(self):
        """Test that steps without gold are excluded from conditional."""
        step_results = [
            {"gold": {"a"}, "visible": {"a", "b"}},  # recovered
            {"gold": set(), "visible": {"b", "c"}},  # no gold
        ]
        
        def gold_accessor(step):
            return step["gold"]
        
        def visibility_accessor(step):
            return step["visible"]
        
        result = compute_conditional_recovery(
            step_results, gold_accessor, visibility_accessor
        )
        
        self.assertEqual(result.conditional_recovery, 1.0)  # 1/1
        self.assertEqual(result.steps_with_gold, 1)


class TestUsefulPromoteRatio(unittest.TestCase):
    """Test useful promote ratio computation."""
    
    def test_all_useful(self):
        """Test when all promotions are useful."""
        promoted_units = [
            {
                "bytes_transferred": 1000,
                "future_access_count": 5,
                "max_attention_weight": 0.05,
            },
            {
                "bytes_transferred": 2000,
                "future_access_count": 3,
                "max_attention_weight": 0.02,
            },
        ]
        
        result = compute_useful_promote_ratio(
            promoted_units,
            usefulness_criteria="attention_access_based",
            attention_threshold=0.01,
        )
        
        self.assertEqual(result.total_promoted_bytes, 3000)
        self.assertEqual(result.useful_bytes_attention_based, 3000)
        self.assertEqual(result.upr_attention_based, 1.0)
    
    def test_none_useful(self):
        """Test when no promotions are useful."""
        promoted_units = [
            {
                "bytes_transferred": 1000,
                "future_access_count": 0,
                "max_attention_weight": 0.0,
            },
            {
                "bytes_transferred": 2000,
                "future_access_count": 0,
                "max_attention_weight": 0.0,
            },
        ]
        
        result = compute_useful_promote_ratio(
            promoted_units,
            usefulness_criteria="attention_access_based",
            attention_threshold=0.01,
        )
        
        self.assertEqual(result.total_promoted_bytes, 3000)
        self.assertEqual(result.useful_bytes_attention_based, 0)
        self.assertEqual(result.upr_attention_based, 0.0)
    
    def test_partial_useful(self):
        """Test partial usefulness."""
        promoted_units = [
            {
                "bytes_transferred": 1000,
                "future_access_count": 5,
                "max_attention_weight": 0.05,
            },
            {
                "bytes_transferred": 2000,
                "future_access_count": 0,
                "max_attention_weight": 0.0,
            },
        ]
        
        result = compute_useful_promote_ratio(
            promoted_units,
            usefulness_criteria="attention_access_based",
            attention_threshold=0.01,
        )
        
        self.assertEqual(result.total_promoted_bytes, 3000)
        self.assertEqual(result.useful_bytes_attention_based, 1000)
        self.assertAlmostEqual(result.upr_attention_based, 1/3, places=5)
    
    def test_gold_overlap_useful(self):
        """Test gold overlap based usefulness."""
        promoted_units = [
            {
                "bytes_transferred": 1000,
                "gold_overlap_tokens": 10,  # has overlap
            },
            {
                "bytes_transferred": 2000,
                "gold_overlap_tokens": 0,  # no overlap
            },
        ]
        
        result = compute_useful_promote_ratio(promoted_units)
        
        self.assertEqual(result.useful_bytes_gold_based, 1000)
        self.assertAlmostEqual(result.upr_gold_based, 1/3, places=5)


class TestBurstGain(unittest.TestCase):
    """Test burst gain computation."""
    
    def test_positive_gain(self):
        """Test positive burst gain."""
        with_burst = {
            "conditional_recovery": 0.8,
            "total_promoted_bytes": 10_000_000,  # 10 MB
        }
        without_burst = {
            "conditional_recovery": 0.6,
            "total_promoted_bytes": 5_000_000,  # 5 MB
        }
        
        result = compute_burst_gain(with_burst, without_burst)
        
        self.assertEqual(result.recovery_with_burst, 0.8)
        self.assertEqual(result.recovery_without_burst, 0.6)
        self.assertEqual(result.bytes_with_burst, 10_000_000)
        self.assertEqual(result.bytes_without_burst, 5_000_000)
        # Gain = (0.8 - 0.6) / 5 MB = 0.04 per MB
        self.assertAlmostEqual(result.burst_gain, 0.04, places=5)
    
    def test_negative_gain(self):
        """Test negative burst gain (burst hurts)."""
        with_burst = {
            "conditional_recovery": 0.5,
            "total_promoted_bytes": 10_000_000,
        }
        without_burst = {
            "conditional_recovery": 0.6,
            "total_promoted_bytes": 5_000_000,
        }
        
        result = compute_burst_gain(with_burst, without_burst)
        
        # Gain = (0.5 - 0.6) / 5 MB = -0.02 per MB
        self.assertAlmostEqual(result.burst_gain, -0.02, places=5)
    
    def test_zero_bytes_delta(self):
        """Test when bytes don't change (edge case)."""
        with_burst = {
            "conditional_recovery": 0.8,
            "total_promoted_bytes": 5_000_000,
        }
        without_burst = {
            "conditional_recovery": 0.6,
            "total_promoted_bytes": 5_000_000,
        }
        
        result = compute_burst_gain(with_burst, without_burst)
        
        self.assertEqual(result.burst_gain, 0.0)


class TestCandidateRecallAtK(unittest.TestCase):
    """Test candidate recall@K computation."""
    
    def test_perfect_recall(self):
        """Test when all gold is recalled."""
        candidates = ["a", "b", "c", "d"]
        gold = {"a", "b"}
        
        result = compute_candidate_recall_at_k(candidates, gold, [1, 2, 5])
        
        self.assertEqual(result[1], 0.5)  # a is recalled
        self.assertEqual(result[2], 1.0)  # a, b both recalled
        self.assertEqual(result[5], 1.0)  # all gold recalled
    
    def test_partial_recall(self):
        """Test partial recall."""
        candidates = ["a", "c", "d", "e"]
        gold = {"a", "b"}
        
        result = compute_candidate_recall_at_k(candidates, gold, [1, 2, 5])
        
        self.assertEqual(result[1], 0.5)  # only a recalled
        self.assertEqual(result[2], 0.5)  # still only a recalled
        self.assertEqual(result[5], 0.5)  # b never recalled
    
    def test_no_gold(self):
        """Test when there's no gold."""
        candidates = ["a", "b", "c"]
        gold = set()
        
        result = compute_candidate_recall_at_k(candidates, gold, [1, 5])
        
        self.assertEqual(result[1], 0.0)
        self.assertEqual(result[5], 0.0)


class TestBudgetUtilization(unittest.TestCase):
    """Test budget utilization computation."""
    
    def test_full_utilization(self):
        """Test 100% utilization."""
        result = compute_budget_utilization(1000, 1000)
        self.assertEqual(result, 1.0)
    
    def test_partial_utilization(self):
        """Test partial utilization."""
        result = compute_budget_utilization(500, 1000)
        self.assertEqual(result, 0.5)
    
    def test_zero_budget(self):
        """Test edge case of zero budget."""
        result = compute_budget_utilization(0, 0)
        self.assertEqual(result, 0.0)
    
    def test_over_utilization(self):
        """Test that utilization is capped at 1.0."""
        result = compute_budget_utilization(1500, 1000)
        self.assertEqual(result, 1.0)


class TestLatencyStatistics(unittest.TestCase):
    """Test latency statistics computation."""
    
    def test_basic_stats(self):
        """Test basic latency statistics."""
        latencies = [1000, 2000, 3000, 4000, 5000]  # microseconds
        
        result = compute_latency_statistics(latencies)
        
        self.assertAlmostEqual(result["mean_ms"], 3.0, places=5)  # 3000 us = 3 ms
        self.assertAlmostEqual(result["p50_ms"], 3.0, places=5)
    
    def test_empty_list(self):
        """Test empty latency list."""
        result = compute_latency_statistics([])
        
        self.assertEqual(result["mean_ms"], 0.0)
        self.assertEqual(result["p95_ms"], 0.0)


class TestSharedEvaluator(unittest.TestCase):
    """Test SharedEvaluator class."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        config = {"test": "config"}
        evaluator = SharedEvaluator(config, experiment_id="test_exp")
        
        self.assertEqual(evaluator.experiment_id, "test_exp")
        self.assertEqual(evaluator.config, config)
    
    def test_add_step_result(self):
        """Test adding step results."""
        evaluator = SharedEvaluator({}, "test")
        
        evaluator.add_step_result(
            step=1,
            gold_chunk_ids={"gold1"},
            visible_chunk_ids={"gold1", "other"},
            candidate_chunk_ids=["gold1", "cand2"],
            selected_chunk_ids=["gold1"],
            promoted_chunk_ids=["gold1"],
            budget_bytes=10000,
            used_bytes=5000,
            latency_us=100.0,
        )
        
        self.assertEqual(len(evaluator.step_results), 1)
        self.assertEqual(evaluator.step_results[0]["step"], 1)


if __name__ == "__main__":
    unittest.main()
