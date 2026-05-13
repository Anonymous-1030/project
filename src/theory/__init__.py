"""
Theoretical foundations for ProSE HPCA contribution.

This package provides:
- Formal KV Cache Roofline model with rigorous derivation (kv_roofline_formal.py)
- Optimal promotion strategy analysis (optimal_promotion.py)
- Attention-driven prefetch theory (attention_prefetch_theory.py)
- Promotion efficiency bounds with submodularity (promotion_efficiency_bound.py)
- Non-stationary promotion regret bound (nonstationary_regret.py)
- Bandwidth-delay product theorem for KV promotion (bandwidth_delay_product.py)
"""

from .kv_roofline_formal import FormalKVCacheRoofline
from .optimal_promotion import OptimalPromotionSolver, GreedyApproximation
from .attention_prefetch_theory import AttentionPrefetchModel
from .nonstationary_regret import NonStationaryRegretAnalyzer
from .bandwidth_delay_product import BDPAnalyzer
