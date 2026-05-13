"""CACTI-style area/power estimator for the PPU (v3.0).

Pipeline stages modeled (5-stage base):
  1. mmrf_receiver       — MMRF input FIFO + reorder buffer + FP16→Q0.15 caster
  2. attention_counter_bank — Saturating counters with shift-based decay
  3. feature_extractor   — 4D quantized feature extraction
  4. utility_lut         — Single-cycle SRAM LUT lookup
  5. dma_arbiter         — Priority queue with coalescing

Continuous batching additions (v3.0):
  - pingpong_sram        — 2×13KB double-buffered state SRAM
  - doorbell_arbiter     — 128-entry ring buffer + pull DMA
  - hw_btw               — Hardware Block Table Walker for PagedAttention
  - swap_dma             — Dedicated state swap DMA engine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.config import PPUConfig


@dataclass
class AreaPowerComponent:
    name: str
    area_mm2: float
    power_mw: float
    latency_ns: float
    detail: str = ""


@dataclass
class AreaPowerReport:
    technology_node_nm: int
    frequency_ghz: float
    total_area_mm2: float
    total_power_mw: float
    critical_path_ns: float
    # Pipeline-aware timing: clock period = slowest stage, not sum.
    pipeline_clock_period_ns: float = 0.0
    end_to_end_latency_ns: float = 0.0
    achievable_frequency_ghz: float = 0.0
    num_pipeline_stages: int = 5
    components: List[AreaPowerComponent] = field(default_factory=list)
    # v3.0: Continuous batching additions (off-pipeline, not on critical path)
    cb_area_mm2: float = 0.0
    cb_power_mw: float = 0.0
    # Grand total including CB
    grand_total_area_mm2: float = 0.0
    grand_total_power_mw: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "technology_node_nm": self.technology_node_nm,
            "frequency_ghz": self.frequency_ghz,
            "total_area_mm2": self.total_area_mm2,
            "total_power_mw": self.total_power_mw,
            "critical_path_ns": self.critical_path_ns,
            "pipeline_clock_period_ns": self.pipeline_clock_period_ns,
            "end_to_end_latency_ns": self.end_to_end_latency_ns,
            "achievable_frequency_ghz": self.achievable_frequency_ghz,
            "num_pipeline_stages": self.num_pipeline_stages,
            "cb_area_mm2": self.cb_area_mm2,
            "cb_power_mw": self.cb_power_mw,
            "grand_total_area_mm2": self.grand_total_area_mm2,
            "grand_total_power_mw": self.grand_total_power_mw,
            "components": [c.__dict__ for c in self.components],
        }


class CACTIModel:
    """Simple analytical model aligned with CACTI-style SRAM sizing."""

    def __init__(self, config: PPUConfig):
        self.config = config

    def estimate(self) -> AreaPowerReport:
        entries = 1 << self.config.lut_index_bits
        lut_bits = entries * self.config.lut_output_bits
        counter_bits = self.config.num_counter_entries * self.config.counter_bits
        dma_meta_bits = self.config.dma_queue_depth * (64 + self.config.dma_priority_bits)

        # Stage 1: MMRF Receiver (Flash-Decoding interface)
        # - 32-deep input FIFO (16-bit entries)
        # - Reorder buffer: N_chunks × 16-bit + N_chunks × 1-bit valid bitmap
        # - FP16→Q0.15 format caster (shift logic, no multiplier)
        mmrf_fifo_bits = 32 * 16  # 32-entry × 16-bit FIFO
        mmrf_reorder_bits = self.config.num_counter_entries * 17  # 16-bit data + 1-bit valid
        mmrf_total_bits = mmrf_fifo_bits + mmrf_reorder_bits
        mmrf = self._sram_component(
            "mmrf_receiver", mmrf_total_bits, banks=1, base_latency_ns=0.20,
        )
        # Override: MMRF is mostly register file + shift logic, not dense SRAM
        node_scale = self.config.technology_node_nm / 7.0
        mmrf.area_mm2 = max(0.0004, 0.0003 * node_scale)
        mmrf.power_mw = max(0.1, 0.3 * node_scale)
        mmrf.detail = (
            f"32-deep FIFO + {self.config.num_counter_entries}-entry reorder buffer "
            f"+ FP16→Q0.15 format caster. Zero-overhead Flash-Decoding interface."
        )

        # Stage 2: Attention counter bank
        lut = self._sram_component("utility_lut", lut_bits, banks=self.config.lut_sram_banks, base_latency_ns=0.65)
        counters = self._sram_component(
            "attention_counter_bank",
            counter_bits,
            banks=self.config.counter_sram_banks,
            base_latency_ns=0.55,
        )
        dma = self._sram_component("dma_queue_metadata", dma_meta_bits, banks=1, base_latency_ns=0.45)

        # Stage 3: Feature extractor
        extractor = AreaPowerComponent(
            name="feature_extractor",
            area_mm2=0.004 * (7.0 / max(self.config.technology_node_nm, 1)),
            power_mw=1.2,
            latency_ns=0.35,
            detail="Comparator/add/quantization logic for 4 features.",
        )

        # Stage 5: DMA arbiter
        arbiter = AreaPowerComponent(
            name="dma_arbiter",
            area_mm2=0.003,
            power_mw=0.9,
            latency_ns=0.25,
            detail="Request prioritization and coalescing control.",
        )
        components = [mmrf, lut, counters, dma, extractor, arbiter]
        total_area = sum(c.area_mm2 for c in components)
        total_power = sum(c.power_mw for c in components)

        # Pipeline timing analysis.
        # The PPU is a 5-stage pipeline:
        #   mmrf_receiver → counter_bank → feature_extractor → LUT → DMA arbiter
        # Clock period = slowest stage (determines achievable frequency).
        # End-to-end latency = sum of all stages (single-request fill time).
        pipeline_stage_names = {
            "mmrf_receiver", "attention_counter_bank",
            "feature_extractor", "utility_lut", "dma_arbiter",
        }
        pipeline_stages = [c for c in components if c.name in pipeline_stage_names]
        clock_period_ns = max(c.latency_ns for c in pipeline_stages)
        end_to_end_ns = sum(c.latency_ns for c in pipeline_stages)
        achievable_freq = 1.0 / clock_period_ns if clock_period_ns > 0 else 0.0

        # v3.0: Continuous batching components (off-pipeline, not on critical path)
        cb_area = 0.0
        cb_power = 0.0
        if getattr(self.config, "cb_enabled", False):
            cb_node = self.config.technology_node_nm / 7.0
            # Ping-pong SRAM: 2 × 13KB = 26KB
            pp_kb = getattr(self.config, "cb_pingpong_buffer_kb", 13.0) * 2
            pp_area = max(0.005, 0.0015 * pp_kb * cb_node)
            pp_power = max(1.0, pp_kb * 0.5 * 1.5 * 0.08)
            # Doorbell: 1KB ring buffer + arbiter FSM
            db_area = 0.002 * cb_node
            db_power = 0.7 * cb_node
            # HW-BTW: address ALU + descriptor formatter
            btw_area = 0.0008 * cb_node
            btw_power = 0.8 * cb_node
            # Swap DMA engine
            swap_area = 0.002 * cb_node
            swap_power = 1.0 * cb_node

            cb_area = pp_area + db_area + btw_area + swap_area
            cb_power = pp_power + db_power + btw_power + swap_power

            components.extend([
                AreaPowerComponent(
                    name="pingpong_sram",
                    area_mm2=pp_area, power_mw=pp_power, latency_ns=0.0,
                    detail=f"2×{pp_kb/2:.0f}KB double-buffered state SRAM for context switching.",
                ),
                AreaPowerComponent(
                    name="doorbell_arbiter",
                    area_mm2=db_area, power_mw=db_power, latency_ns=0.0,
                    detail="128-entry ring buffer + async pull DMA front-end.",
                ),
                AreaPowerComponent(
                    name="hw_btw",
                    area_mm2=btw_area, power_mw=btw_power, latency_ns=0.0,
                    detail="Hardware Block Table Walker for PagedAttention scatter/gather.",
                ),
                AreaPowerComponent(
                    name="swap_dma",
                    area_mm2=swap_area, power_mw=swap_power, latency_ns=0.0,
                    detail="Dedicated DMA engine for ping-pong state swaps.",
                ),
            ])

        return AreaPowerReport(
            technology_node_nm=self.config.technology_node_nm,
            frequency_ghz=self.config.clock_frequency_ghz,
            total_area_mm2=total_area,
            total_power_mw=total_power,
            critical_path_ns=clock_period_ns,
            pipeline_clock_period_ns=clock_period_ns,
            end_to_end_latency_ns=end_to_end_ns,
            achievable_frequency_ghz=achievable_freq,
            num_pipeline_stages=len(pipeline_stages),
            components=components,
            cb_area_mm2=cb_area,
            cb_power_mw=cb_power,
            grand_total_area_mm2=total_area + cb_area,
            grand_total_power_mw=total_power + cb_power,
        )

    def _sram_component(self, name: str, bits: int, banks: int, base_latency_ns: float) -> AreaPowerComponent:
        kb = bits / 8192.0
        node_scale = self.config.technology_node_nm / 7.0
        area = max(0.0008, 0.0015 * kb * node_scale / max(1, banks ** 0.5))
        power = max(0.15, kb * self.config.sram_read_energy_pj * self.config.clock_frequency_ghz * 0.08)
        return AreaPowerComponent(
            name=name,
            area_mm2=area,
            power_mw=power,
            latency_ns=base_latency_ns,
            detail=f"{bits} bits across {banks} bank(s).",
        )
