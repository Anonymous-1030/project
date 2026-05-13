"""
ProSE-X 3.0 Innovations: Four advanced modules addressing core limitations.

1. ABA  - Adaptive Bandwidth Arbitrage: mode-switching by measured bandwidth
2. HES  - Hierarchical Evidence Synthesis: CXL-side neural micro-scorer
3. GBS  - Ghost Block Synthesis: approximate KV reconstruction for missing blocks
4. SDAP - Speculative Decode-Aware Promotion: draft-model attention as free evidence
"""

from src.innovations.aba import AdaptiveBandwidthArbitrage, ABAMode
from src.innovations.hes import HierarchicalEvidenceSynthesis
from src.innovations.gbs import GhostBlockSynthesizer
from src.innovations.sdap import SpeculativeDecodeAwarePromotion

__all__ = [
    "AdaptiveBandwidthArbitrage",
    "ABAMode",
    "HierarchicalEvidenceSynthesis",
    "GhostBlockSynthesizer",
    "SpeculativeDecodeAwarePromotion",
]
