"""PPU-integrated promotion pipeline with optional PHT/PTB augmentation (v3.0).

v2.1: Flash-Decoding compatible MMRF-based attention mass ingestion.
v3.0: Continuous batching support (ping-pong state, doorbell, HW-BTW).
Standard pipeline: 5 stages (mmrf → counter → feature → LUT → DMA).
Augmented pipeline: 6 stages (+ PHT parallel + PTB update).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Any

from src.config import ProSEXv2Config
from src.core_types import (
    ChunkMetadata,
    QueryContext,
    PromotionPipelineResult,
    ScoredCandidate,
    ScorerResult,
)
from src.promotion.pipeline import PromotionPipeline
from src.hardware.ppu.ppu_core import PromotionPredictionUnit
from src.hardware.ppu.ppu_simulator import PPUCycleSimulator
from src.hardware.ppu.cacti_model import CACTIModel
from src.hardware.ppu.pht import PHTConfig, PromotionHistoryTable
from src.hardware.ppu.ptb import PTBConfig, PromotionTargetBuffer
from src.hardware.ppu.pht_ptb_integration import PHTAugmentedPPU
from src.hardware.ppu.pht_ptb_simulator import PHTAugmentedCycleSimulator
from src.hardware.ppu.pht_ptb_cacti import PHTCACTIModel
from src.hardware.ppu.continuous_batching import (
    ContinuousBatchingConfig,
    ContinuousBatchingController,
    PingPongConfig,
    DoorbellConfig,
    HWBTWConfig,
)


class PPUIntegratedPromotionPipeline(PromotionPipeline):
    """Drop-in pipeline variant using the PPU instead of software ODUS scoring.

    v3.0: Continuous batching support (ping-pong state, doorbell, HW-BTW).
    When cb_enabled is set in PPUConfig, the pipeline instantiates a
    ContinuousBatchingController for multi-sequence state management.

    When PHT/PTB are enabled in config.ppu, uses the 6-stage augmented pipeline
    (PHTAugmentedPPU) with speculative prefetch and history-based prediction.
    Otherwise falls back to the standard 5-stage PPU.
    """

    def __init__(self, config: ProSEXv2Config, lut_values: Optional[List[int]] = None):
        super().__init__(config)
        self._pht_enabled = getattr(config.ppu, "pht_enabled", False)
        self._ptb_enabled = getattr(config.ppu, "ptb_enabled", False)

        if self._pht_enabled or self._ptb_enabled:
            # Build PHT/PTB configs from PPUConfig fields
            pht_cfg = PHTConfig(
                num_entries=getattr(config.ppu, "pht_num_entries", 1024),
                counter_bits=getattr(config.ppu, "pht_counter_bits", 2),
                prediction_threshold=getattr(config.ppu, "pht_prediction_threshold", 2),
                position_hash_bits=getattr(config.ppu, "pht_position_hash_bits", 8),
                layer_hash_bits=getattr(config.ppu, "pht_layer_hash_bits", 4),
                context_hash_bits=getattr(config.ppu, "pht_context_hash_bits", 4),
                enable_periodic_decay=getattr(config.ppu, "pht_enable_periodic_decay", False),
                decay_interval_steps=getattr(config.ppu, "pht_decay_interval_steps", 100),
            )
            ptb_cfg = PTBConfig(
                num_entries=getattr(config.ppu, "ptb_num_entries", 32),
                associativity=getattr(config.ppu, "ptb_associativity", 32),
                tag_bits=getattr(config.ppu, "ptb_tag_bits", 16),
                eviction_policy=getattr(config.ppu, "ptb_eviction_policy", "lru"),
                entry_bytes=getattr(config.ppu, "ptb_entry_bytes", 16),
                max_age_steps=getattr(config.ppu, "ptb_max_age_steps", 50),
            )
            self.augmented_ppu = PHTAugmentedPPU(
                config.ppu, pht_cfg, ptb_cfg, lut_values=lut_values,
            )
            self.augmented_simulator = PHTAugmentedCycleSimulator(
                config.ppu, pht_cfg, ptb_cfg, ppu=self.augmented_ppu,
            )
            self.pht_ptb_area_model = PHTCACTIModel(pht_cfg, ptb_cfg)
            self.ppu = self.augmented_ppu  # for get_ppu_summary compat
        else:
            self.augmented_ppu = None
            self.augmented_simulator = None
            self.pht_ptb_area_model = None
            self.ppu = PromotionPredictionUnit(config.ppu, lut_values=lut_values)

        self.simulator = PPUCycleSimulator(config.ppu, ppu=self.ppu if not self._pht_enabled else None)
        self.area_model = CACTIModel(config.ppu)
        self._last_ppu_trace = None

        # v3.0: Continuous batching controller
        self._cb_enabled = getattr(config.ppu, "cb_enabled", False)
        self.cb_controller: Optional[ContinuousBatchingController] = None
        if self._cb_enabled:
            cb_config = ContinuousBatchingConfig(
                enabled=True,
                max_batch_size=getattr(config.ppu, "cb_max_batch_size", 64),
                pingpong=PingPongConfig(
                    buffer_size_kb=getattr(config.ppu, "cb_pingpong_buffer_kb", 13.0),
                    swap_dma_bandwidth_gbps=getattr(config.ppu, "cb_swap_dma_bandwidth_gbps", 1000.0),
                ),
                doorbell=DoorbellConfig(
                    ring_depth=getattr(config.ppu, "cb_doorbell_depth", 128),
                    pull_bandwidth_gbps=getattr(config.ppu, "cb_pull_bandwidth_gbps", 200.0),
                ),
                hw_btw=HWBTWConfig(
                    max_sequences=getattr(config.ppu, "cb_btw_max_sequences", 256),
                    max_blocks_per_seq=getattr(config.ppu, "cb_btw_max_blocks_per_seq", 8192),
                    lookup_latency_cycles=getattr(config.ppu, "cb_btw_lookup_latency_cycles", 2),
                ),
            )
            self.cb_controller = ContinuousBatchingController(cb_config)

    def run(
        self,
        query: QueryContext,
        tail_chunks: List[ChunkMetadata],
        anchor_chunks: List[ChunkMetadata],
        promoted_chunks: List[ChunkMetadata],
        budget_bytes: Optional[int] = None,
        gold_chunk_ids: Optional[Set[str]] = None,
        attention_masses: Optional[Dict[str, float]] = None,
    ) -> PromotionPipelineResult:
        total_start = time.time()
        all_chunks = {c.chunk_id: c for c in tail_chunks + anchor_chunks + promoted_chunks}
        self._chunks = all_chunks
        query.active_anchor_ids = [c.chunk_id for c in anchor_chunks]

        ulf_result = self.ulf.filter(query, tail_chunks, gold_chunk_ids=gold_chunk_ids, all_chunks=all_chunks)
        scorer_result = self._ppu_score(ulf_result.candidate_ids, query, all_chunks, attention_masses or {})

        if budget_bytes is None:
            budget_bytes = self._compute_budget(tail_chunks)
        scheduler_result = self.scheduler.schedule(scorer_result, query, all_chunks, budget_bytes)
        burst_result = self.burst.expand(scheduler_result.selected_ids, all_chunks, query.step)
        accessed_ids = [c.chunk_id for c in promoted_chunks]
        sticky_result = self.sticky.update(burst_result.burst_ids, accessed_ids, all_chunks, query.step)
        sticky_result.request_id = query.request_id

        # PHT/PTB end-of-step update: feed promotion outcomes back
        if self.augmented_ppu is not None:
            self.augmented_ppu.end_step(
                promoted_chunk_ids=scheduler_result.selected_ids,
                accessed_chunk_ids=accessed_ids,
                query=query,
                all_chunks=all_chunks,
            )

        final_visible = self._compute_final_visible(anchor_chunks, promoted_chunks, sticky_result.promoted_ids)
        final_active_bytes = sum(all_chunks[cid].logical_bytes for cid in final_visible if cid in all_chunks)
        final_promoted_bytes = sum(all_chunks[cid].logical_bytes for cid in sticky_result.promoted_ids if cid in all_chunks)
        result = PromotionPipelineResult(
            request_id=query.request_id,
            step=query.step,
            ulf_result=ulf_result,
            scorer_result=scorer_result,
            scheduler_result=scheduler_result,
            burst_result=burst_result,
            sticky_result=sticky_result,
            final_visible_ids=final_visible,
            final_active_bytes=final_active_bytes,
            final_promoted_bytes=final_promoted_bytes,
            total_latency_us=(time.time() - total_start) * 1e6,
            ppu_trace=self._last_ppu_trace,
            ppu_area_power=self.area_model.estimate(),
        )
        return result

    def _ppu_score(
        self,
        candidate_ids: List[str],
        query: QueryContext,
        all_chunks: Dict[str, ChunkMetadata],
        attention_masses: Dict[str, float],
    ) -> ScorerResult:
        start = time.time()
        candidates = [all_chunks[cid] for cid in candidate_ids if cid in all_chunks]

        # v2.1: MMRF begin_step is called by the simulators internally.
        # The simulator models the full pipeline including Stage 1 (MMRF receive).

        if self.augmented_simulator is not None:
            # Use 6-stage PHT/PTB-augmented pipeline
            trace = self.augmented_simulator.simulate(
                query=query,
                candidates=candidates,
                all_chunks=all_chunks,
                attention_masses=attention_masses,
                enqueue_threshold=0.0,
            )
            self._last_ppu_trace = trace
            scored = [
                ScoredCandidate(
                    chunk_id=r["chunk_id"],
                    score=r.get("utility", 0.0),
                    confidence=r.get("confidence", 0.5),
                    feature_vector=None,
                    score_components={
                        "ppu_utility": r.get("utility", 0.0),
                        "pht_prediction": r.get("pht_prediction", False),
                        "ptb_hit": r.get("ptb_hit", False),
                    },
                )
                for r in trace.results
            ]
        else:
            # Standard 5-stage PPU pipeline
            trace = self.simulator.simulate(
                query=query,
                candidates=candidates,
                all_chunks=all_chunks,
                attention_masses=attention_masses,
                enqueue_threshold=0.0,
            )
            self._last_ppu_trace = trace
            scored = [
                ScoredCandidate(
                    chunk_id=r.chunk_id,
                    score=r.utility,
                    confidence=r.confidence,
                    feature_vector=None,
                    score_components={
                        "ppu_utility": r.utility,
                        "lut_index": float(r.lut_index),
                        "counter_value": float(r.counter_value),
                    },
                )
                for r in trace.results
            ]

        scored.sort(key=lambda c: c.score, reverse=True)
        return ScorerResult(
            request_id=query.request_id,
            step=query.step,
            candidates=scored,
            n_input_candidates=len(candidate_ids),
            n_scored=len(scored),
            n_above_threshold=sum(1 for c in scored if c.score >= 0.5),
            score_threshold=0.5,
            scorer_mode="ppu_pht_ptb" if self.augmented_ppu else "ppu_lut",
            scorer_latency_us=(time.time() - start) * 1e6,
        )

    def get_ppu_summary(self) -> Dict[str, Any]:
        report = self.area_model.estimate()
        summary: Dict[str, Any] = {
            "counter_entries": len(self.ppu.counters.snapshot()),
            "area_power": report.to_dict(),
            "last_trace": None,
        }
        if hasattr(self.ppu, "dma_queue"):
            summary["queue_stats"] = self.ppu.dma_queue.stats()
        if self.augmented_ppu is not None:
            summary["pht_stats"] = self.augmented_ppu.pht.stats()
            summary["ptb_stats"] = self.augmented_ppu.ptb.stats()
            summary["prefetch_stats"] = self.augmented_ppu.prefetch_engine.stats()
            if self.pht_ptb_area_model is not None:
                pht_ptb_report = self.pht_ptb_area_model.estimate()
                summary["pht_ptb_area_power"] = {
                    "total_area_mm2": pht_ptb_report.total_area_mm2,
                    "total_power_mw": pht_ptb_report.total_power_mw,
                    "fits_budget": pht_ptb_report.fits_budget,
                }
        # v3.0: Continuous batching stats
        if self.cb_controller is not None:
            summary["continuous_batching"] = self.cb_controller.stats()
        return summary
