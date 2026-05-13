"""
Cycle-level simulator for the PHT/PTB-augmented PPU pipeline (v2.1).

Extends PPUCycleSimulator for the 6-stage pipeline:
  Stage 1: mmrf_receive        — MMRF format cast (1 cycle)
  Stage 2: counter_update      — Attention mass counter update (1 cycle)
  Stage 3: feature_extract+PHT — Feature extraction || PHT lookup (parallel, 1 cycle)
  Stage 4: lut_lookup          — Utility LUT read (1 cycle)
  Stage 5: dma_enqueue         — DMA request enqueue (1 cycle)
  Stage 6: ptb_update          — PTB bookkeeping (1 cycle)

v2.1: Added mmrf_receive as Stage 1 for Flash-Decoding compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from src.config import PPUConfig
from src.core_types import ChunkMetadata, QueryContext
from src.hardware.ppu.pht import PHTConfig
from src.hardware.ppu.ptb import PTBConfig
from src.hardware.ppu.pht_ptb_integration import PHTAugmentedPPU


_AUGMENTED_STAGE_NAMES = (
    "mmrf_receive",
    "counter_update",
    "feature_extract_pht",  # parallel: feature extract + PHT lookup
    "lut_lookup",
    "dma_enqueue",
    "ptb_update",
)


@dataclass
class PipelineEvent:
    """A single pipeline event in the simulation trace."""

    cycle: int
    stage: str
    chunk_id: str
    detail: str = ""


@dataclass
class AugmentedSimulationTrace:
    """Complete simulation trace for the augmented pipeline."""

    events: List[PipelineEvent]
    results: List[Dict[str, Any]]
    total_cycles: int
    initiation_interval: int
    throughput_chunks_per_cycle: float
    stage_costs: Dict[str, int]
    num_candidates: int
    pht_predictions: int
    ptb_hits: int
    mmrf_receive_cycles: int = 1


class PHTAugmentedCycleSimulator:
    """Cycle simulator for the PHT/PTB-augmented PPU pipeline.

    6-stage pipeline (v2.1):
      Stage 1: mmrf_receive       (1 cycle, FP16→Q0.15 format cast)
      Stage 2: counter_update     (1 cycle)
      Stage 3: feature_extract_pht (max(feature, pht) = 1 cycle, parallel)
      Stage 4: lut_lookup          (1 cycle)
      Stage 5: dma_enqueue         (1 cycle)
      Stage 6: ptb_update          (1 cycle)

    Initiation Interval (II) = max(stage_costs) = 1 cycle (fully pipelined)
    Total latency for N candidates = (N-1) × II + sum(stage_costs)
    """

    def __init__(
        self,
        config: PPUConfig,
        pht_config: PHTConfig,
        ptb_config: PTBConfig,
        ppu: Optional[PHTAugmentedPPU] = None,
    ):
        self.config = config
        self.pht_config = pht_config
        self.ptb_config = ptb_config

        # Stage costs in cycles (v2.1: mmrf_receive added)
        self.stage_costs = {
            "mmrf_receive": 1,  # Always 1 cycle (format cast only)
            "counter_update": config.counter_update_cycles,
            "feature_extract_pht": max(
                config.feature_extract_cycles, 1  # PHT lookup = 1 cycle
            ),
            "lut_lookup": config.lut_lookup_cycles,
            "dma_enqueue": config.dma_enqueue_cycles,
            "ptb_update": 1,  # PTB update = 1 cycle
        }

        self.ii = max(self.stage_costs.values())
        self.pipeline_depth = sum(self.stage_costs.values())

        # Optional functional PPU for actual computation
        self._ppu = ppu

    def simulate(
        self,
        query: QueryContext,
        candidates: List[ChunkMetadata],
        all_chunks: Dict[str, ChunkMetadata],
        attention_masses: Optional[Dict[str, float]] = None,
        enqueue_threshold: float = 0.0,
    ) -> AugmentedSimulationTrace:
        """Simulate the 6-stage pipeline for all candidates."""
        if attention_masses is None:
            attention_masses = {}

        events: List[PipelineEvent] = []
        results: List[Dict[str, Any]] = []
        n = len(candidates)

        pht_predictions = 0
        ptb_hits = 0

        # Stage 1 (MMRF): feed masses into the PPU's MMRF buffer
        if self._ppu is not None:
            self._ppu.begin_step(attention_masses)

        for i, chunk in enumerate(candidates):
            entry_cycle = i * self.ii
            cumulative = entry_cycle

            # Walk through all 6 stages
            for stage_name in _AUGMENTED_STAGE_NAMES:
                cost = self.stage_costs[stage_name]
                detail = ""
                if stage_name == "mmrf_receive":
                    detail = "FP16→Q0.15 format cast"
                elif stage_name == "feature_extract_pht":
                    detail = "parallel: feature_extract || pht_lookup"

                events.append(PipelineEvent(
                    cycle=cumulative,
                    stage=stage_name,
                    chunk_id=chunk.chunk_id,
                    detail=detail,
                ))
                cumulative += cost

            # Functional simulation if PPU available
            if self._ppu is not None:
                mass = attention_masses.get(chunk.chunk_id, 0.0)
                ppu_result = self._ppu.process_candidate(
                    chunk, query, all_chunks, mass, enqueue_threshold
                )
                result_dict = {
                    "chunk_id": chunk.chunk_id,
                    "utility": ppu_result.utility,
                    "confidence": ppu_result.confidence,
                    "dma_enqueued": ppu_result.dma_enqueued,
                    "pht_prediction": ppu_result.metadata.get("pht_prediction", False),
                    "ptb_hit": ppu_result.metadata.get("ptb_hit", False),
                    "entry_cycle": entry_cycle,
                    "exit_cycle": cumulative,
                }
                results.append(result_dict)

                if ppu_result.metadata.get("pht_prediction", False):
                    pht_predictions += 1
                if ppu_result.metadata.get("ptb_hit", False):
                    ptb_hits += 1
            else:
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "entry_cycle": entry_cycle,
                    "exit_cycle": cumulative,
                })

        total_cycles = (n - 1) * self.ii + self.pipeline_depth if n > 0 else 0
        throughput = n / max(1, total_cycles)

        return AugmentedSimulationTrace(
            events=events,
            results=results,
            total_cycles=total_cycles,
            initiation_interval=self.ii,
            throughput_chunks_per_cycle=throughput,
            stage_costs=dict(self.stage_costs),
            num_candidates=n,
            pht_predictions=pht_predictions,
            ptb_hits=ptb_hits,
            mmrf_receive_cycles=1,
        )

    def latency_ns(self, num_candidates: int) -> float:
        """Compute total latency in nanoseconds."""
        if num_candidates <= 0:
            return 0.0
        total_cycles = (num_candidates - 1) * self.ii + self.pipeline_depth
        period_ns = 1.0 / self.config.clock_frequency_ghz
        return total_cycles * period_ns
