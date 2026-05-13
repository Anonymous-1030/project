"""
Unit tests for Burst and Sticky modules.
"""

import unittest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import BurstConfig
from src.promotion.burst.burst_expand import BurstExpander, IdentityBurstExpander
from src.promotion.sticky.ttl_manager import StickyTTLManager, NoStickyManager
from src.core_types import ChunkMetadata, ChunkTier


class TestBurstExpander(unittest.TestCase):
    """Tests for Burst expansion."""
    
    def setUp(self):
        """Set up test fixtures."""
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
                tier=ChunkTier.TAIL,
                creation_step=0,
            )
            self.chunks.append(chunk)
        
        self.all_chunks = {c.chunk_id: c for c in self.chunks}
    
    def test_burst_disabled(self):
        """Test burst expansion disabled."""
        config = BurstConfig(enabled=False, radius=0)
        expander = BurstExpander(config)
        
        selected = ["chunk_3"]
        result = expander.expand(selected, self.all_chunks, current_step=5)
        
        self.assertEqual(result.n_input, 1)
        self.assertEqual(result.n_burst_total, 1)
        self.assertEqual(result.n_expansion, 0)
        self.assertEqual(result.burst_ids, selected)
    
    def test_burst_radius_1(self):
        """Test burst expansion with radius 1."""
        config = BurstConfig(enabled=True, radius=1)
        expander = BurstExpander(config)
        
        selected = ["chunk_3"]
        result = expander.expand(selected, self.all_chunks, current_step=5)
        
        # Should include chunk_2, chunk_3, chunk_4
        self.assertIn("chunk_2", result.burst_ids)
        self.assertIn("chunk_3", result.burst_ids)
        self.assertIn("chunk_4", result.burst_ids)
        
        # Core should be preserved
        self.assertEqual(result.core_ids, selected)
        
        # Expansion should be chunk_2, chunk_4
        self.assertIn("chunk_2", result.expansion_ids)
        self.assertIn("chunk_4", result.expansion_ids)
    
    def test_burst_radius_2(self):
        """Test burst expansion with radius 2."""
        config = BurstConfig(enabled=True, radius=2)
        expander = BurstExpander(config)
        
        selected = ["chunk_3"]
        result = expander.expand(selected, self.all_chunks, current_step=5)
        
        # Radius 2 should include more chunks than radius 1
        # Core is chunk_3, plus neighbors within distance 2
        self.assertGreater(len(result.burst_ids), 1)  # At least core
        self.assertIn("chunk_3", result.burst_ids)  # Core always included
    
    def test_burst_multiple_cores(self):
        """Test burst expansion with multiple core chunks."""
        config = BurstConfig(enabled=True, radius=1)
        expander = BurstExpander(config)
        
        selected = ["chunk_2", "chunk_7"]
        result = expander.expand(selected, self.all_chunks, current_step=5)
        
        # Should merge bursts around both cores
        self.assertIn("chunk_1", result.burst_ids)  # Near chunk_2
        self.assertIn("chunk_3", result.burst_ids)  # Near chunk_2
        self.assertIn("chunk_6", result.burst_ids)  # Near chunk_7
        self.assertIn("chunk_8", result.burst_ids)  # Near chunk_7


class TestStickyTTLManager(unittest.TestCase):
    """Tests for Sticky TTL manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BurstConfig(
            sticky_enabled=True,
            default_ttl=4,
            ttl_refresh_policy="access",
        )
    
    def test_sticky_assignment(self):
        """Test TTL assignment on promotion."""
        manager = StickyTTLManager(self.config)
        
        all_chunks = {"chunk_1": None, "chunk_2": None}
        
        result = manager.update(
            newly_promoted_ids=["chunk_1"],
            accessed_ids=[],
            all_chunks=all_chunks,
            current_step=0,
        )
        
        self.assertEqual(result.n_promoted, 1)
        self.assertEqual(result.ttl_values["chunk_1"], 4)
    
    def test_ttl_decay(self):
        """Test TTL decay over steps."""
        manager = StickyTTLManager(self.config)
        
        all_chunks = {"chunk_1": None}
        
        # Step 0: Promote
        result = manager.update(
            newly_promoted_ids=["chunk_1"],
            accessed_ids=[],
            all_chunks=all_chunks,
            current_step=0,
        )
        self.assertEqual(result.ttl_values.get("chunk_1"), 4)
        
        # Step 1-3: TTL decays 4 -> 3 -> 2 -> 1, chunk still sticky
        for step in range(1, 4):
            result = manager.update(
                newly_promoted_ids=[],
                accessed_ids=[],
                all_chunks=all_chunks,
                current_step=step,
            )
            self.assertIn("chunk_1", result.promoted_ids)
        
        # Step 4: TTL reaches 0, chunk expires
        result = manager.update(
            newly_promoted_ids=[],
            accessed_ids=[],
            all_chunks=all_chunks,
            current_step=4,
        )
        
        self.assertIn("chunk_1", result.expired_ids)
        self.assertEqual(result.n_expired, 1)
    
    def test_ttl_refresh(self):
        """Test TTL refresh on access."""
        manager = StickyTTLManager(self.config)
        
        all_chunks = {"chunk_1": None}
        
        # Promote
        manager.update(
            newly_promoted_ids=["chunk_1"],
            accessed_ids=[],
            all_chunks=all_chunks,
            current_step=0,
        )
        
        # Decay twice
        manager.update(
            newly_promoted_ids=[],
            accessed_ids=[],
            all_chunks=all_chunks,
            current_step=1,
        )
        
        # Access and refresh
        result = manager.update(
            newly_promoted_ids=[],
            accessed_ids=["chunk_1"],
            all_chunks=all_chunks,
            current_step=2,
        )
        
        # TTL should be refreshed
        self.assertEqual(result.ttl_values.get("chunk_1"), 4)
        self.assertEqual(result.n_refreshed, 1)
    
    def test_no_sticky(self):
        """Test no sticky mode."""
        manager = NoStickyManager()
        
        all_chunks = {"chunk_1": None}
        
        result = manager.update(
            newly_promoted_ids=["chunk_1"],
            accessed_ids=[],
            all_chunks=all_chunks,
            current_step=0,
        )
        
        # Should expire immediately
        self.assertEqual(result.n_promoted, 0)
        self.assertEqual(result.n_expired, 1)


if __name__ == "__main__":
    unittest.main()
