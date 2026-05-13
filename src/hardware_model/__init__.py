"""
Hardware-level architectural models for KV-Cache-Aware Memory Controller (KCMC).

This package provides:
- KCMC area/power/latency estimation (area_power.py)
- Cycle-analytical performance model v1 (performance_model.py)
- Cycle-analytical performance model v2 with queuing + contention (performance_model_v2.py)
- Near-CXL decompressor hardware model (near_cxl_decompressor.py)
- KCMC microarchitecture with CXL.mem coherence (kcmc_microarch.py)
- PPU (Promotion Processing Unit) pipeline model (ppu_pipeline.py)
- gem5-compatible CXL memory simulation (gem5_cxl_sim.py)
- Multi-tenant promotion with fairness (multi_tenant.py)
"""

from .area_power import KCMCHardwareModel, ComponentAreaPower
from .performance_model import CycleAnalyticalModel, LatencyBreakdown
from .near_cxl_decompressor import NearCXLDecompressor
from .performance_model_v2 import CycleAnalyticalModelV2, LatencyBreakdownV2
from .kcmc_microarch import KCMCMicroarchitecture, KCMCConfig, KCMCSimulationResult
from .ppu_pipeline import PPUConfig, PPUPipelineSimulator, compute_ppu_hardware
from .gem5_cxl_sim import KVPromotionTraceSimulator, TraceGenerator
from .multi_tenant import MultiTenantSimulator, FairPromotionAllocator
