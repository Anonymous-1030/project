"""
Unit tests for scheduler veto behavior fix.

These tests ensure the scheduler cannot illegally veto all candidates
when valid candidates exist and budget is available.

PATCH TEST (2026-03-24):
- Two candidates, top candidate score = 1.0, budget available.
- Expected: selected >= 1.
- Failing this test should block merge.
"""

import unittest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import EABSConfig
from src.promotion.scheduler.eabs import ExplorationAwareBudgetScheduler
from src.core_types import (
    ChunkMetadata, QueryContext, ScoredCandidate, ScorerResult,
    ChunkTier
)


class TestSchedulerVetoFix(unittest.TestCase):
    """Tests for scheduler veto behavior fix."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = EABSConfig(
            exploration_ratio=0.2,
            max_chunks_per_step=5,
            budget_ratio_of_tail=0.1,
            min_score_threshold=0.5,  # Threshold higher than candidate score
            confidence_threshold=0.5,
        )
        
        self.scheduler = ExplorationAwareBudgetScheduler(self.config)
        
        # Create test chunks
        self.chunks = {
            "chunk_1": ChunkMetadata(
                chunk_id="chunk_1",
                request_id="test_request",
                token_start=0,
                token_end=512,
                position_ratio=0.1,
                num_tokens=512,
                logical_bytes=65536,
                tier=ChunkTier.TAIL,
            ),
            "chunk_2": ChunkMetadata(
                chunk_id="chunk_2",
                request_id="test_request",
                token_start=512,
                token_end=1024,
                position_ratio=0.3,
                num_tokens=512,
                logical_bytes=65536,
                tier=ChunkTier.TAIL,
            ),
        }
        
        self.query = QueryContext(
            request_id="test_request",
            step=0,
        )
    
    def test_mandatory_exploit_fallback(self):
        """
        PATCH TEST: Top candidate score = 1.0, budget available.
        
        Expected: selected >= 1, even if below threshold.
        
        This test must pass - failing blocks merge.
        """
        # Create candidates with score below threshold
        # BUT top candidate has score = 1.0 which is above threshold
        candidates = [
            ScoredCandidate(chunk_id="chunk_1", score=1.0, confidence=0.9),
            ScoredCandidate(chunk_id="chunk_2", score=0.8, confidence=0.8),
        ]
        
        scorer_result = ScorerResult(
            request_id="test_request",
            step=0,
            candidates=candidates,
            n_input_candidates=2,
            n_scored=2,
            n_above_threshold=2,
            score_threshold=0.5,
            scorer_mode="test",
            scorer_latency_us=1.0,
        )
        
        # Budget available for at least one chunk
        budget_bytes = 100000
        
        result = self.scheduler.schedule(
            scorer_result, self.query, self.chunks, budget_bytes
        )
        
        # INVARIANT: Must select at least 1 candidate
        self.assertGreaterEqual(
            len(result.selected_ids), 
            1,
            f"CRITICAL: Scheduler selected 0 candidates when {len(candidates)} valid "
            f"candidates exist with scores above threshold and budget available. "
            f"Top candidate score = {candidates[0].score}, threshold = {self.config.min_score_threshold}"
        )
        
        # Additional checks
        self.assertGreaterEqual(
            result.n_exploit, 
            1,
            "Must select at least 1 exploit candidate"
        )
    
    def test_no_veto_when_candidates_above_threshold(self):
        """
        Test that scheduler does not veto when candidates are above threshold.
        """
        candidates = [
            ScoredCandidate(chunk_id="chunk_1", score=0.9, confidence=0.9),
            ScoredCandidate(chunk_id="chunk_2", score=0.7, confidence=0.8),
        ]
        
        scorer_result = ScorerResult(
            request_id="test_request",
            step=0,
            candidates=candidates,
            n_input_candidates=2,
            n_scored=2,
            n_above_threshold=2,
            score_threshold=0.5,
            scorer_mode="test",
            scorer_latency_us=1.0,
        )
        
        budget_bytes = 100000
        
        result = self.scheduler.schedule(
            scorer_result, self.query, self.chunks, budget_bytes
        )
        
        # Should select candidates
        self.assertGreaterEqual(len(result.selected_ids), 1)
        self.assertEqual(result.n_dropped_low_score, 0)
        self.assertEqual(result.n_dropped_low_confidence, 0)
    
    def test_advisory_mode_selects_below_threshold(self):
        """
        Test that advisory mode selects candidates even when below threshold.
        
        PATCHED BEHAVIOR: Score threshold is advisory, not veto.
        Candidates should be selected even if below threshold.
        """
        # Create candidates ALL below threshold
        candidates = [
            ScoredCandidate(chunk_id="chunk_1", score=0.3, confidence=0.9),  # Below 0.5
            ScoredCandidate(chunk_id="chunk_2", score=0.2, confidence=0.8),  # Below 0.5
        ]
        
        scorer_result = ScorerResult(
            request_id="test_request",
            step=0,
            candidates=candidates,
            n_input_candidates=2,
            n_scored=2,
            n_above_threshold=0,
            score_threshold=0.5,
            scorer_mode="test",
            scorer_latency_us=1.0,
        )
        
        budget_bytes = 100000
        
        result = self.scheduler.schedule(
            scorer_result, self.query, self.chunks, budget_bytes
        )
        
        # PATCHED: Should select candidates even below threshold (advisory mode)
        self.assertGreaterEqual(
            len(result.selected_ids), 
            1,
            "Advisory mode should select candidates even below threshold"
        )
        
        # Top candidate (chunk_1) should be selected
        self.assertIn(
            "chunk_1",
            result.selected_ids,
            "Top candidate should be selected even with score below threshold"
        )
    
    def test_illegal_zero_selection_invariant(self):
        """
        Test the invariant: candidates > 0 and selected == 0 and budget > 0 is illegal.
        
        This should never happen after the patch.
        """
        candidates = [
            ScoredCandidate(chunk_id="chunk_1", score=1.0, confidence=0.9),
        ]
        
        scorer_result = ScorerResult(
            request_id="test_request",
            step=0,
            candidates=candidates,
            n_input_candidates=1,
            n_scored=1,
            n_above_threshold=1,
            score_threshold=0.5,
            scorer_mode="test",
            scorer_latency_us=1.0,
        )
        
        # Budget available
        budget_bytes = 100000
        
        result = self.scheduler.schedule(
            scorer_result, self.query, self.chunks, budget_bytes
        )
        
        # INVARIANT: Cannot have candidates > 0 but selected == 0 when budget available
        if len(candidates) > 0 and budget_bytes > 0:
            self.assertGreaterEqual(
                len(result.selected_ids),
                1,
                "INVARIANT VIOLATION: illegal_zero_selection"
            )
            
            # Check no invariant violations in decision log
            if hasattr(result, '_decision_log'):
                self.assertNotIn(
                    "illegal_zero_selection",
                    result._decision_log.invariant_violations,
                    "Should not have illegal_zero_selection violation"
                )
    
    def test_scheduler_veto_on_gold_rank1_invariant(self):
        """
        Test the invariant: gold_rank == 1 and selected == 0 is illegal.
        """
        # Simulate gold chunk at rank 1
        candidates = [
            ScoredCandidate(chunk_id="chunk_1", score=1.0, confidence=0.9),  # Gold
            ScoredCandidate(chunk_id="chunk_2", score=0.8, confidence=0.8),
        ]
        
        scorer_result = ScorerResult(
            request_id="test_request",
            step=0,
            candidates=candidates,
            n_input_candidates=2,
            n_scored=2,
            n_above_threshold=2,
            score_threshold=0.5,
            scorer_mode="test",
            scorer_latency_us=1.0,
        )
        # Mark chunk_1 as gold
        scorer_result.gold_chunk_ids = {"chunk_1"}
        
        budget_bytes = 100000
        
        result = self.scheduler.schedule(
            scorer_result, self.query, self.chunks, budget_bytes
        )
        
        # Gold is at rank 1 (top), so must be selected
        if result.n_dropped_budget == 0:  # If not dropped due to budget
            self.assertGreaterEqual(
                len(result.selected_ids),
                1,
                "INVARIANT VIOLATION: scheduler_veto_on_gold_rank1"
            )
            
            # chunk_1 should be selected
            self.assertIn(
                "chunk_1",
                result.selected_ids,
                "Gold chunk at rank 1 must be selected"
            )
    
    def test_budget_constrains_selection(self):
        """
        Test that budget is still respected.
        
        Should not select if budget is truly exhausted.
        """
        candidates = [
            ScoredCandidate(chunk_id="chunk_1", score=1.0, confidence=0.9),
            ScoredCandidate(chunk_id="chunk_2", score=0.9, confidence=0.9),
        ]
        
        scorer_result = ScorerResult(
            request_id="test_request",
            step=0,
            candidates=candidates,
            n_input_candidates=2,
            n_scored=2,
            n_above_threshold=2,
            score_threshold=0.5,
            scorer_mode="test",
            scorer_latency_us=1.0,
        )
        
        # Budget too small for any chunk (chunk size is 65536 bytes)
        budget_bytes = 1000  # Less than chunk size
        
        result = self.scheduler.schedule(
            scorer_result, self.query, self.chunks, budget_bytes
        )
        
        # Should not select due to budget constraint
        self.assertEqual(len(result.selected_ids), 0, 
                        "Should not select when budget < chunk size")
        # At least one candidate should be marked as dropped due to budget
        self.assertGreaterEqual(result.n_dropped_budget, 1,
                               "Should record budget constraint drops")


if __name__ == "__main__":
    unittest.main()
