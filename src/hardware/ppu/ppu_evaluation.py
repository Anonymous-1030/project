"""Evaluation utilities comparing PPU and software scoring paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config import ProSEXv2Config
from src.core_types import ChunkMetadata, QueryContext
from src.promotion.pipeline import PromotionPipeline
from src.hardware.ppu.ppu_pipeline import PPUIntegratedPromotionPipeline


@dataclass
class PPUComparisonResult:
    request_id: str
    step: int
    software_selected: List[str]
    ppu_selected: List[str]
    overlap: float
    software_latency_us: float
    ppu_latency_us: float
    speedup: float
    metadata: Dict[str, object] = field(default_factory=dict)


class PPUEvaluationFramework:
    """Run side-by-side comparisons between baseline and PPU pipelines."""

    def __init__(self, config: ProSEXv2Config, lut_values: Optional[List[int]] = None):
        self.config = config
        self.software_pipeline = PromotionPipeline(config)
        self.ppu_pipeline = PPUIntegratedPromotionPipeline(config, lut_values=lut_values)

    def compare_once(
        self,
        query: QueryContext,
        tail_chunks: List[ChunkMetadata],
        anchor_chunks: List[ChunkMetadata],
        promoted_chunks: List[ChunkMetadata],
        budget_bytes: Optional[int] = None,
        attention_masses: Optional[Dict[str, float]] = None,
    ) -> PPUComparisonResult:
        sw = self.software_pipeline.run(query, tail_chunks, anchor_chunks, promoted_chunks, budget_bytes=budget_bytes)
        ppu = self.ppu_pipeline.run(
            query,
            tail_chunks,
            anchor_chunks,
            promoted_chunks,
            budget_bytes=budget_bytes,
            attention_masses=attention_masses,
        )
        sw_sel = sw.scheduler_result.selected_ids if sw.scheduler_result else []
        ppu_sel = ppu.scheduler_result.selected_ids if ppu.scheduler_result else []
        inter = len(set(sw_sel) & set(ppu_sel))
        union = len(set(sw_sel) | set(ppu_sel))
        overlap = inter / max(1, union)
        speedup = sw.total_latency_us / max(ppu.total_latency_us, 1e-6)
        return PPUComparisonResult(
            request_id=query.request_id,
            step=query.step,
            software_selected=sw_sel,
            ppu_selected=ppu_sel,
            overlap=overlap,
            software_latency_us=sw.total_latency_us,
            ppu_latency_us=ppu.total_latency_us,
            speedup=speedup,
            metadata={
                "ppu_trace_cycles": ppu.ppu_trace.total_cycles if ppu.ppu_trace else None,
                "ppu_area_power": self.ppu_pipeline.area_model.estimate().to_dict(),
            },
        )
