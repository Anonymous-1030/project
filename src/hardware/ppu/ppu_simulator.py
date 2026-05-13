"""Cycle-level simulator for the PPU promotion datapath (v2.1).

Models a 5-stage pipeline with MMRF-based attention mass ingestion:

  Stage 1: mmrf_receive     — MMRF format cast (1 cycle)
  Stage 2: counter_update   — Attention mass counter update (1 cycle)
  Stage 3: feature_extract  — 4D quantized feature extraction (1 cycle)
  Stage 4: lut_lookup       — Utility LUT read (1 cycle)
  Stage 5: dma_enqueue      — DMA request enqueue (1 cycle)

A new candidate enters the pipeline every clock cycle once it is full
(steady-state throughput = 1 result/cycle).  Total latency for N
candidates = N + (stages - 1).

The MMRF receive stage models the single-cycle FP16→Q0.15 format cast
that occurs after the Flash-Decoding reduction kernel writes chunk
masses to the memory-mapped register file.  See flash_decode_interface.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config import PPUConfig
from src.core_types import ChunkMetadata, QueryContext
from src.hardware.ppu.ppu_core import PromotionPredictionUnit, PPUResult


@dataclass
class SimulationEvent:
    cycle: int
    stage: str
    chunk_id: str
    detail: str


@dataclass
class SimulationTrace:
    total_cycles: int
    processed_candidates: int
    events: List[SimulationEvent] = field(default_factory=list)
    results: List[PPUResult] = field(default_factory=list)
    queue_depth_peak: int = 0
    # Pipeline-aware metrics
    pipeline_stages: int = 5
    pipeline_startup_cycles: int = 0
    steady_state_throughput: float = 0.0  # candidates per cycle
    # MMRF metrics
    mmrf_receive_cycles: int = 1  # Always 1 (format cast only)

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_cycles": self.total_cycles,
            "processed_candidates": self.processed_candidates,
            "queue_depth_peak": self.queue_depth_peak,
            "pipeline_stages": self.pipeline_stages,
            "pipeline_startup_cycles": self.pipeline_startup_cycles,
            "steady_state_throughput": self.steady_state_throughput,
            "mmrf_receive_cycles": self.mmrf_receive_cycles,
            "events": [e.__dict__ for e in self.events],
        }


# The five pipeline stages and their corresponding config fields.
_STAGE_NAMES = (
    "mmrf_receive",
    "counter_update",
    "feature_extract",
    "lut_lookup",
    "dma_enqueue",
)


class PPUCycleSimulator:
    """Deterministic pipeline simulator using config cycle costs.

    Models a proper pipelined datapath: stage costs define the per-stage
    latency, and candidates overlap in the pipeline.  In steady state
    the throughput is limited by the slowest stage (1 candidate per
    max-stage-cost cycles).

    v2.1: Added mmrf_receive as Stage 1 (1 cycle, FP16→Q0.15 format cast).
    The MMRF stage is always 1 cycle because it only performs a shift-based
    format conversion — the actual data arrival from the GPU reduction
    kernel is modeled separately in FlashDecodeTimingModel.
    """

    def __init__(self, config: PPUConfig, ppu: Optional[PromotionPredictionUnit] = None):
        self.config = config
        self.ppu = ppu or PromotionPredictionUnit(config)

    def simulate(
        self,
        query: QueryContext,
        candidates: List[ChunkMetadata],
        all_chunks: Dict[str, ChunkMetadata],
        attention_masses: Optional[Dict[str, float]] = None,
        enqueue_threshold: float = 0.0,
    ) -> SimulationTrace:
        attention_masses = attention_masses or {}
        events: List[SimulationEvent] = []
        results: List[PPUResult] = []
        queue_depth_peak = 0

        stage_costs = [
            1,  # mmrf_receive: always 1 cycle (format cast)
            max(1, self.config.counter_update_cycles),
            max(1, self.config.feature_extract_cycles),
            max(1, self.config.lut_lookup_cycles),
            max(1, self.config.dma_enqueue_cycles),
        ]
        num_stages = len(stage_costs)

        if not candidates:
            self.ppu.end_step()
            return SimulationTrace(
                total_cycles=0,
                processed_candidates=0,
                pipeline_stages=num_stages,
            )

        # Stage 1 (MMRF): begin_step feeds all masses into the MMRF buffer
        self.ppu.begin_step(attention_masses)

        # Pipeline simulation: each candidate enters the pipeline on the
        # next cycle after the previous candidate clears stage 0.  The
        # initiation interval (II) is the cost of the slowest stage.
        initiation_interval = max(stage_costs)

        for idx, chunk in enumerate(candidates):
            # Pipeline issue time: candidate idx enters at cycle idx * II.
            issue_cycle = idx * initiation_interval

            # Walk through each pipeline stage.
            stage_start = issue_cycle
            for stage_name, cost in zip(_STAGE_NAMES, stage_costs):
                events.append(SimulationEvent(
                    cycle=stage_start,
                    stage=stage_name,
                    chunk_id=chunk.chunk_id,
                    detail=f"cost={cost}",
                ))
                stage_start += cost

            # commit_cycle is the cycle at which the last stage finishes.
            commit_cycle = stage_start

            # Stages 2-5: run the PPU logic (functional model; timing is
            # captured by the events above).
            result = self.ppu.process_candidate(
                chunk=chunk,
                query=query,
                all_chunks=all_chunks,
                attention_mass=attention_masses.get(chunk.chunk_id, 0.0),
                enqueue_threshold=enqueue_threshold,
            )
            results.append(result)
            queue_depth_peak = max(queue_depth_peak, len(self.ppu.dma_queue.snapshot()))

            events.append(SimulationEvent(
                cycle=commit_cycle,
                stage="commit",
                chunk_id=chunk.chunk_id,
                detail=f"utility={result.utility:.3f}, enqueued={result.dma_enqueued}",
            ))

        # Total cycles = last candidate's commit cycle
        last_issue = (len(candidates) - 1) * initiation_interval
        total_cycles = last_issue + sum(stage_costs)
        startup_cycles = sum(stage_costs)  # pipeline fill time
        throughput = len(candidates) / max(total_cycles, 1)

        self.ppu.end_step()
        return SimulationTrace(
            total_cycles=total_cycles,
            processed_candidates=len(candidates),
            events=events,
            results=results,
            queue_depth_peak=queue_depth_peak,
            pipeline_stages=num_stages,
            pipeline_startup_cycles=startup_cycles,
            steady_state_throughput=throughput,
            mmrf_receive_cycles=1,
        )
