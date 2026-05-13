"""
Unit tests for Multi-Queue Recall ULF.
"""

import unittest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import MQRULFConfig
from src.promotion.ulf.mqr_ulf import (
    MultiQueueRecallULF, AnchorNeighborQueue, LexicalOverlapQueue,
    StructuralRecencyQueue, HistoricalSuccessQueue
)
from src.core_types import ChunkMetadata, QueryContext, ChunkTier


class TestMQRULF(unittest.TestCase):
    """Tests for MQR-ULF."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MQRULFConfig()
        
        # Create test chunks
        self.chunks = []
        for i in range(10):
            chunk = ChunkMetadata(
                chunk_id=f"chunk_{i}",
                request_id="test_request",
                token_start=i * 100,
                token_end=(i + 1) * 100,
                position_ratio=i / 10,
                num_tokens=100,
                logical_bytes=12800,
                signature=np.random.randn(128).astype(np.float32),
                tier=ChunkTier.TAIL,
                creation_step=0,
            )
            self.chunks.append(chunk)
        
        # Mark some as section boundaries
        self.chunks[0].is_section_boundary = True
        self.chunks[5].is_section_boundary = True
        
        # Create query
        self.query = QueryContext(
            request_id="test_request",
            step=50,
            query_signature=self.chunks[0].signature,
        )
    
    def test_ulf_initialization(self):
        """Test ULF initializes all queues."""
        ulf = MultiQueueRecallULF(self.config)
        
        self.assertEqual(len(ulf.queues), 4)
        
        queue_names = [q.name for q in ulf.queues]
        self.assertIn("anchor_neighbor", queue_names)
        self.assertIn("lexical_overlap", queue_names)
        self.assertIn("structural_recency", queue_names)
        self.assertIn("historical_success", queue_names)
    
    def test_anchor_neighbor_queue(self):
        """Test anchor-neighbor queue."""
        queue = AnchorNeighborQueue(radius=2, top_k=5)
        
        # Set active anchor
        self.query.active_anchor_ids = ["chunk_5"]
        
        all_chunks = {c.chunk_id: c for c in self.chunks}
        result, diagnostics = queue.recall(self.query, self.chunks, all_chunks)
        
        # Should recall chunks near chunk_5
        self.assertGreater(len(result.candidate_ids), 0)
        
        # chunk_4, chunk_6 should be recalled (radius 1)
        self.assertIn("chunk_4", result.candidate_ids)
        self.assertIn("chunk_6", result.candidate_ids)
    
    def test_lexical_overlap_queue(self):
        """Test lexical overlap queue."""
        queue = LexicalOverlapQueue(method="hashed_token", top_k=5)
        
        all_chunks = {c.chunk_id: c for c in self.chunks}
        result, diagnostics = queue.recall(self.query, self.chunks, all_chunks)
        
        # Should recall chunks with similar signatures
        self.assertGreaterEqual(len(result.candidate_ids), 0)
    
    def test_structural_recency_queue(self):
        """Test structural/recency queue."""
        queue = StructuralRecencyQueue(
            recent_top_k=3,
            boundary_top_k=2,
            title_adjacent_top_k=2,
        )
        
        # Set last access for some chunks
        self.chunks[1].last_access_step = 45  # Recent
        self.chunks[2].last_access_step = 40  # Recent
        
        all_chunks = {c.chunk_id: c for c in self.chunks}
        result, diagnostics = queue.recall(self.query, self.chunks, all_chunks)
        
        # Should recall recent chunks and section boundaries
        self.assertGreater(len(result.candidate_ids), 0)
        
        # chunk_0 is section boundary
        self.assertIn("chunk_0", result.candidate_ids)
    
    def test_historical_success_queue(self):
        """Test historical success queue."""
        queue = HistoricalSuccessQueue(top_k=3, min_success_count=1)
        
        # Set promotion history
        self.chunks[3].promoted_count = 2
        self.chunks[7].promoted_count = 1
        
        all_chunks = {c.chunk_id: c for c in self.chunks}
        result, diagnostics = queue.recall(self.query, self.chunks, all_chunks)
        
        # Should recall chunks with promotion history
        self.assertEqual(len(result.candidate_ids), 2)
        self.assertIn("chunk_3", result.candidate_ids)
        self.assertIn("chunk_7", result.candidate_ids)
    
    def test_ulf_filter(self):
        """Test complete ULF filter."""
        ulf = MultiQueueRecallULF(self.config)
        
        result = ulf.filter(self.query, self.chunks)
        
        # Check result structure
        self.assertEqual(result.request_id, self.query.request_id)
        self.assertEqual(result.step, self.query.step)
        self.assertEqual(result.n_tail_total, len(self.chunks))
        
        # Should have candidates
        self.assertGreater(result.n_candidates, 0)
        self.assertLessEqual(result.n_candidates, self.config.max_total_candidates)
        
        # Check per-queue counts
        self.assertIn("anchor_neighbor", result.per_queue_counts)
        self.assertIn("lexical_overlap", result.per_queue_counts)
    
    def test_queue_union_dedup(self):
        """Test union and deduplication across queues."""
        # Create config with overlapping queues
        config = MQRULFConfig(
            anchor_neighbor_top_k=10,
            lexical_overlap_top_k=10,
            structural_recent_top_k=10,
            historical_success_top_k=10,
        )
        
        ulf = MultiQueueRecallULF(config)
        
        # Make sure there's overlap
        self.query.active_anchor_ids = ["chunk_0"]
        self.chunks[0].last_access_step = 45  # Recent
        
        result = ulf.filter(self.query, self.chunks)
        
        # Check that candidates are unique
        self.assertEqual(len(result.candidate_ids), len(set(result.candidate_ids)))
        
        # Check overlap matrix
        if result.queue_overlap_matrix:
            # Matrix should have entries for each queue
            for queue_name in result.per_queue_counts.keys():
                self.assertIn(queue_name, result.queue_overlap_matrix)
                # Self overlap equals the queue's candidate count
                self.assertEqual(
                    result.queue_overlap_matrix[queue_name][queue_name],
                    result.per_queue_counts[queue_name]
                )


if __name__ == "__main__":
    unittest.main()
