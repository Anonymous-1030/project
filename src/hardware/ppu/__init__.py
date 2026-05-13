"""
Promotion Prediction Unit (PPU) — On-chip Hardware Accelerator for KV Cache Promotion.

This package implements the hardware contribution for ProSE-X's HPCA submission:
a low-area, low-power on-chip unit that replaces the software ODUS MLP with a
quantized lookup table (LUT) for single-cycle utility prediction.

=== PHYSICAL DEPLOYMENT DECLARATION (§3.x) ===

  PHT + PTB + PPU are integrated INSIDE the GPU's L2 cache controller
  (or equivalently, the GPU's internal memory controller partition).

  Physical topology:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                          GPU Die                                    │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐       ┌──────────┐      │
    │  │  SM  0   │  │  SM  1   │  │  SM  N   │  ...  │  SM  M   │      │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘       └────┬─────┘      │
    │       │              │              │                   │            │
    │       └──────────────┴──────┬───────┴───────────────────┘            │
    │                             │  L2 / Memory Controller               │
    │                    ┌────────┴────────┐                               │
    │                    │   PPU (5-stage) │  ← on-chip, 1-cycle decision │
    │                    │   + PHT (1024e) │  ← on-chip, 1-cycle lookup  │
    │                    │   + PTB (32e)   │  ← on-chip, 1-cycle lookup  │
    │                    │   + MMRF        │  ← on-chip, Flash-Decoding   │
    │                    └────────┬────────┘                               │
    │                             │                                        │
    ═════════════════════════════╪═════════════════ CXL Link ══════════════
                                  │  (250ns one-way, 64 GB/s full-duplex)
                    ┌────────────┴────────────┐
                    │  CXL Memory Controller  │
                    │  + QFC (8 MAC arrays)   │  ← off-chip, ~50us compute
                    │  + CXL-attached DRAM    │
                    └─────────────────────────┘

  KEY LATENCY DISTINCTION:
    - ON-CHIP (fast path): PHT query = 1 cycle (~1ns @ 1GHz)
      PPU full pipeline = 5 cycles (~5ns)
      This is the "should we promote?" decision latency.
    - OFF-CHIP (slow path): QFC query = 250ns (CXL RTT) + 50us (MAC compute)
      + 250ns (return) ≈ 50.5us total
      This is the "what's the attention score?" remote compute latency.

  The fast path (PHT/PTB/PPU) runs entirely within the GPU die and does NOT
  cross the CXL link.  The slow path (QFC) runs on the CXL memory controller
  and is only invoked when the PPU needs remote attention scores — which is
  the *exception*, not the common case.  In the common case, PHT + PTB
  provide a speculative prefetch decision in 1 cycle, confirmed by the
  PPU's 5-cycle LUT lookup, without any CXL traffic.

v2.1: Flash-Decoding compatible MMRF-based attention mass ingestion.
The PPU no longer assumes token-level attention aggregation (incompatible with
FlashAttention's tiled online-softmax).  Instead, chunk-level masses are derived
as a zero-cost byproduct of Flash-Decoding's split-K reduction and delivered via
a Memory-Mapped Register File (MMRF).

v2.2: Probation mechanism for speculative prefetch mispredictions.
Speculatively prefetched blocks are tagged as "Probation" (trial period). If
the PPU's slow-path LUT scoring disagrees with the PHT prediction, or if the
block is not accessed within N decode steps, it is immediately marked as a
low-priority eviction candidate, preventing HBM pollution from wasted prefetches.

Modules:
    flash_decode_interface — Flash-Decoding epilogue exporter + MMRF receiver
    ppu_core            — Core PPU microarchitecture (MMRF, LUT, counters, features, DMA)
    cacti_model         — CACTI-based area/power/latency estimation (5-stage)
    ppu_simulator       — Cycle-level performance simulator (5-stage)
    lut_distill         — Distillation pipeline: ODUS MLP → quantized LUT
    ppu_pipeline        — PPU-integrated promotion pipeline (drop-in replacement)
    ppu_evaluation      — Evaluation framework comparing PPU vs software ODUS
    pht                 — Promotion History Table (branch-predictor-inspired)
    ptb                 — Promotion Target Buffer (branch-target-buffer-inspired)
    pht_ptb_integration — PHT/PTB integration with PPU pipeline (6-stage)
    pht_ptb_simulator   — Cycle simulator for 6-stage augmented pipeline
    pht_ptb_cacti       — Area/power model for PHT + PTB
    pht_ptb_explorer    — Design space exploration for PHT/PTB parameters
"""

from src.hardware.ppu.flash_decode_interface import (
    ChunkLSEPair,
    FlashDecodeEpilogueExporter,
    FlashDecodeTimingModel,
    MMRFConfig,
    MMRFReceiver,
    MMRFReceiveResult,
    ReductionResult,
)
from src.hardware.ppu.ppu_core import (
    PromotionPredictionUnit,
    AttentionMassCounterBank,
    FeatureExtractor,
    UtilityLUT,
    DMARequestQueue,
    DMARequest,
    QuantizedFeatures,
    PPUResult,
)
from src.hardware.ppu.cacti_model import CACTIModel, AreaPowerReport
from src.hardware.ppu.ppu_simulator import PPUCycleSimulator, SimulationTrace
from src.hardware.ppu.lut_distill import LUTDistiller, DistillationReport
from src.hardware.ppu.ppu_pipeline import PPUIntegratedPromotionPipeline
from src.hardware.ppu.ppu_evaluation import PPUEvaluationFramework, PPUComparisonResult
from src.hardware.ppu.pht import PromotionHistoryTable, PHTConfig, PHTEntry, PHTResult
from src.hardware.ppu.ptb import PromotionTargetBuffer, PTBConfig, PTBEntry, PTBLookupResult
from src.hardware.ppu.pht_ptb_integration import (
    PHTAugmentedFeatureExtractor,
    PTBAugmentedPrefetchEngine,
    PHTAugmentedPPU,
)
from src.hardware.ppu.pht_ptb_cacti import PHTCACTIModel, PHTAreaPowerReport
from src.hardware.ppu.pht_ptb_simulator import PHTAugmentedCycleSimulator
from src.hardware.ppu.pht_ptb_explorer import PHTDesignSpaceExplorer, PHTDesignPoint
from src.hardware.ppu.continuous_batching import (
    ContinuousBatchingController,
    ContinuousBatchingConfig,
    PingPongStateManager,
    PingPongConfig,
    DoorbellArbiter,
    DoorbellConfig,
    HardwareBlockTableWalker,
    HWBTWConfig,
    CBAreaPowerReport,
    estimate_cb_area_power,
)

__all__ = [
    # Flash-Decoding interface (v2.1)
    "ChunkLSEPair",
    "FlashDecodeEpilogueExporter",
    "FlashDecodeTimingModel",
    "MMRFConfig",
    "MMRFReceiver",
    "MMRFReceiveResult",
    "ReductionResult",
    # Core PPU
    "PromotionPredictionUnit",
    "AttentionMassCounterBank",
    "FeatureExtractor",
    "UtilityLUT",
    "DMARequestQueue",
    "DMARequest",
    "QuantizedFeatures",
    "PPUResult",
    "CACTIModel",
    "AreaPowerReport",
    "PPUCycleSimulator",
    "SimulationTrace",
    "LUTDistiller",
    "DistillationReport",
    "PPUIntegratedPromotionPipeline",
    "PPUEvaluationFramework",
    "PPUComparisonResult",
    "PromotionHistoryTable",
    "PHTConfig",
    "PHTEntry",
    "PHTResult",
    "PromotionTargetBuffer",
    "PTBConfig",
    "PTBEntry",
    "PTBLookupResult",
    "PHTAugmentedFeatureExtractor",
    "PTBAugmentedPrefetchEngine",
    "PHTAugmentedPPU",
    "PHTCACTIModel",
    "PHTAreaPowerReport",
    "PHTAugmentedCycleSimulator",
    "PHTDesignSpaceExplorer",
    "PHTDesignPoint",
    # Continuous Batching (v3.0)
    "ContinuousBatchingController",
    "ContinuousBatchingConfig",
    "PingPongStateManager",
    "PingPongConfig",
    "DoorbellArbiter",
    "DoorbellConfig",
    "HardwareBlockTableWalker",
    "HWBTWConfig",
    "CBAreaPowerReport",
    "estimate_cb_area_power",
]
