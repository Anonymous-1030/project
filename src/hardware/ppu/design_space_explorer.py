"""PPU Design Space Exploration and Sensitivity Analysis for HPCA.

Generates the data for:
- Figure 7: PPU area vs. LUT index bits (design space)
- Figure 8: CXL bandwidth sensitivity (16/32/64/128 GB/s)
- Figure 9: Area-accuracy Pareto frontier
- Table 3: PPU configuration comparison
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

from src.config import PPUConfig
from src.hardware.ppu.cacti_model import CACTIModel, AreaPowerReport
from src.theory.promotion_overlap_theory import (
    PromotionOverlapTheory,
    HardwareParams,
)


@dataclass
class DesignPoint:
    """A single point in the PPU design space."""
    label: str
    config: PPUConfig
    area_mm2: float
    power_mw: float
    lut_entries: int
    counter_entries: int
    dma_depth: int
    clock_period_ns: float
    achievable_freq_ghz: float
    # Accuracy proxy (higher LUT entries → better approximation)
    accuracy_proxy: float
    # Normalized metrics
    area_normalized: float = 0.0
    power_normalized: float = 0.0


@dataclass
class SensitivityResult:
    """Result of a sensitivity sweep."""
    sweep_param: str
    sweep_values: List[float]
    area_mm2: List[float]
    power_mw: List[float]
    accuracy_proxy: List[float]
    throughput_proxy: List[float]
    pareto_indices: List[int]


@dataclass
class CXLSensitivityResult:
    """Result of CXL bandwidth sensitivity analysis."""
    bandwidths_gbps: List[float]
    critical_budgets_bytes: List[int]
    critical_budgets_chunks: List[int]
    exposed_latencies_us: List[float]
    throughputs: List[float]
    fully_hidden_flags: List[bool]


class PPUDesignSpaceExplorer:
    """Systematic exploration of PPU design space for HPCA evaluation."""

    # ------------------------------------------------------------------ #
    # LUT index bits sweep (Figure 7)
    # ------------------------------------------------------------------ #

    def sweep_lut_index_bits(
        self,
        bits_range: Optional[List[int]] = None,
    ) -> SensitivityResult:
        """Sweep LUT index bits: 6, 8, 10, 12, 14, 16."""
        if bits_range is None:
            bits_range = [6, 8, 10, 12, 14, 16]

        areas, powers, accuracies, throughputs = [], [], [], []

        for bits in bits_range:
            cfg = PPUConfig(lut_index_bits=bits)
            report = CACTIModel(cfg).estimate()
            areas.append(report.total_area_mm2)
            powers.append(report.total_power_mw)
            # Accuracy proxy: more entries → better approximation of ODUS
            # Empirical fit: accuracy ≈ 1 - C / sqrt(entries)
            entries = 1 << bits
            accuracies.append(1.0 - 2.0 / math.sqrt(entries))
            # Throughput: 1 / clock_period
            throughputs.append(report.achievable_frequency_ghz)

        pareto = self._extract_pareto_2d(
            [(a, -acc) for a, acc in zip(areas, accuracies)]
        )

        return SensitivityResult(
            sweep_param="lut_index_bits",
            sweep_values=[float(b) for b in bits_range],
            area_mm2=areas,
            power_mw=powers,
            accuracy_proxy=accuracies,
            throughput_proxy=throughputs,
            pareto_indices=pareto,
        )

    # ------------------------------------------------------------------ #
    # Counter bank size sweep
    # ------------------------------------------------------------------ #

    def sweep_counter_entries(
        self,
        entries_range: Optional[List[int]] = None,
    ) -> SensitivityResult:
        if entries_range is None:
            entries_range = [128, 256, 512, 1024, 2048]

        areas, powers, accuracies, throughputs = [], [], [], []

        for n in entries_range:
            cfg = PPUConfig(num_counter_entries=n)
            report = CACTIModel(cfg).estimate()
            areas.append(report.total_area_mm2)
            powers.append(report.total_power_mw)
            # More counters → fewer evictions → better tracking
            accuracies.append(min(1.0, 0.7 + 0.3 * math.log2(n) / math.log2(2048)))
            throughputs.append(report.achievable_frequency_ghz)

        pareto = self._extract_pareto_2d(
            [(a, -acc) for a, acc in zip(areas, accuracies)]
        )

        return SensitivityResult(
            sweep_param="counter_entries",
            sweep_values=[float(n) for n in entries_range],
            area_mm2=areas,
            power_mw=powers,
            accuracy_proxy=accuracies,
            throughput_proxy=throughputs,
            pareto_indices=pareto,
        )

    # ------------------------------------------------------------------ #
    # DMA queue depth sweep
    # ------------------------------------------------------------------ #

    def sweep_dma_queue_depth(
        self,
        depths: Optional[List[int]] = None,
    ) -> SensitivityResult:
        if depths is None:
            depths = [2, 4, 8, 16, 32]

        areas, powers, accuracies, throughputs = [], [], [], []

        for d in depths:
            cfg = PPUConfig(dma_queue_depth=d)
            report = CACTIModel(cfg).estimate()
            areas.append(report.total_area_mm2)
            powers.append(report.total_power_mw)
            # Deeper queue → fewer drops → better promotion coverage
            accuracies.append(min(1.0, 0.6 + 0.4 * (1 - 1.0 / d)))
            throughputs.append(report.achievable_frequency_ghz)

        pareto = self._extract_pareto_2d(
            [(a, -acc) for a, acc in zip(areas, accuracies)]
        )

        return SensitivityResult(
            sweep_param="dma_queue_depth",
            sweep_values=[float(d) for d in depths],
            area_mm2=areas,
            power_mw=powers,
            accuracy_proxy=accuracies,
            throughput_proxy=throughputs,
            pareto_indices=pareto,
        )

    # ------------------------------------------------------------------ #
    # Technology node sweep
    # ------------------------------------------------------------------ #

    def sweep_technology_node(
        self,
        nodes: Optional[List[int]] = None,
    ) -> SensitivityResult:
        if nodes is None:
            nodes = [3, 5, 7, 10, 14]

        areas, powers, accuracies, throughputs = [], [], [], []

        for nm in nodes:
            cfg = PPUConfig(technology_node_nm=nm)
            report = CACTIModel(cfg).estimate()
            areas.append(report.total_area_mm2)
            powers.append(report.total_power_mw)
            accuracies.append(0.92)  # Accuracy independent of node
            throughputs.append(report.achievable_frequency_ghz)

        return SensitivityResult(
            sweep_param="technology_node_nm",
            sweep_values=[float(n) for n in nodes],
            area_mm2=areas,
            power_mw=powers,
            accuracy_proxy=accuracies,
            throughput_proxy=throughputs,
            pareto_indices=[],
        )

    # ------------------------------------------------------------------ #
    # CXL bandwidth sensitivity (Figure 8)
    # ------------------------------------------------------------------ #

    def sweep_cxl_bandwidth(
        self,
        bandwidths_gbps: Optional[List[float]] = None,
        compute_time_us: float = 55.0,
        promoted_bytes: int = 3 * 1048576,  # 3 chunks
        bytes_per_chunk: int = 1048576,
    ) -> CXLSensitivityResult:
        if bandwidths_gbps is None:
            bandwidths_gbps = [16.0, 32.0, 48.0, 64.0, 96.0, 128.0]

        critical_budgets, critical_chunks = [], []
        exposed_lats, throughputs, hidden_flags = [], [], []

        for bw in bandwidths_gbps:
            hw = HardwareParams(cxl_bandwidth_gbps=bw)
            theory = PromotionOverlapTheory(hw)

            cb = theory.compute_critical_budget(compute_time_us, "cxl", bytes_per_chunk)
            critical_budgets.append(cb["critical_budget_bytes"])
            critical_chunks.append(cb["critical_budget_chunks"])

            ol = theory.compute_exposed_latency(promoted_bytes, compute_time_us, "cxl")
            exposed_lats.append(ol["exposed_latency_us"])
            hidden_flags.append(ol["fully_hidden"])

            tp = theory.throughput_model(promoted_bytes, compute_time_us, "cxl")
            throughputs.append(tp["throughput_actual_steps_per_s"])

        return CXLSensitivityResult(
            bandwidths_gbps=bandwidths_gbps,
            critical_budgets_bytes=critical_budgets,
            critical_budgets_chunks=critical_chunks,
            exposed_latencies_us=exposed_lats,
            throughputs=throughputs,
            fully_hidden_flags=hidden_flags,
        )

    # ------------------------------------------------------------------ #
    # Full design space exploration (Table 3)
    # ------------------------------------------------------------------ #

    def explore_design_space(self) -> List[DesignPoint]:
        """Enumerate representative PPU configurations."""
        configs = {
            "PPU-Tiny": PPUConfig(
                lut_index_bits=6, num_counter_entries=128,
                dma_queue_depth=4, technology_node_nm=7,
            ),
            "PPU-Small": PPUConfig(
                lut_index_bits=8, num_counter_entries=256,
                dma_queue_depth=4, technology_node_nm=7,
            ),
            "PPU-Base": PPUConfig(
                lut_index_bits=8, num_counter_entries=512,
                dma_queue_depth=8, technology_node_nm=7,
            ),
            "PPU-Large": PPUConfig(
                lut_index_bits=10, num_counter_entries=1024,
                dma_queue_depth=8, technology_node_nm=7,
            ),
            "PPU-XL": PPUConfig(
                lut_index_bits=12, num_counter_entries=2048,
                dma_queue_depth=16, technology_node_nm=7,
            ),
        }

        points: List[DesignPoint] = []
        for label, cfg in configs.items():
            report = CACTIModel(cfg).estimate()
            entries = 1 << cfg.lut_index_bits
            acc_proxy = 1.0 - 2.0 / math.sqrt(entries)

            points.append(DesignPoint(
                label=label,
                config=cfg,
                area_mm2=report.total_area_mm2,
                power_mw=report.total_power_mw,
                lut_entries=entries,
                counter_entries=cfg.num_counter_entries,
                dma_depth=cfg.dma_queue_depth,
                clock_period_ns=report.pipeline_clock_period_ns,
                achievable_freq_ghz=report.achievable_frequency_ghz,
                accuracy_proxy=acc_proxy,
            ))

        # Normalize
        max_area = max(p.area_mm2 for p in points)
        max_power = max(p.power_mw for p in points)
        for p in points:
            p.area_normalized = p.area_mm2 / max_area if max_area > 0 else 0
            p.power_normalized = p.power_mw / max_power if max_power > 0 else 0

        return points

    # ------------------------------------------------------------------ #
    # GPU overhead analysis
    # ------------------------------------------------------------------ #

    def gpu_overhead_analysis(
        self,
        config: Optional[PPUConfig] = None,
        gpu_die_area_mm2: float = 814.0,  # H100 SXM
        gpu_tdp_w: float = 700.0,
    ) -> Dict[str, float]:
        """Compute PPU overhead relative to GPU die."""
        cfg = config or PPUConfig()
        report = CACTIModel(cfg).estimate()

        return {
            "ppu_area_mm2": report.total_area_mm2,
            "gpu_die_area_mm2": gpu_die_area_mm2,
            "area_overhead_pct": report.total_area_mm2 / gpu_die_area_mm2 * 100,
            "ppu_power_mw": report.total_power_mw,
            "gpu_tdp_w": gpu_tdp_w,
            "power_overhead_pct": report.total_power_mw / (gpu_tdp_w * 1000) * 100,
            "pipeline_stages": report.num_pipeline_stages,
            "clock_period_ns": report.pipeline_clock_period_ns,
            "achievable_freq_ghz": report.achievable_frequency_ghz,
        }

    # ------------------------------------------------------------------ #
    # Run all analyses (for paper figure generation)
    # ------------------------------------------------------------------ #

    def run_all(self) -> Dict[str, object]:
        """Run complete design space exploration for paper."""
        return {
            "lut_sweep": self.sweep_lut_index_bits().__dict__,
            "counter_sweep": self.sweep_counter_entries().__dict__,
            "dma_sweep": self.sweep_dma_queue_depth().__dict__,
            "node_sweep": self.sweep_technology_node().__dict__,
            "cxl_sweep": self.sweep_cxl_bandwidth().__dict__,
            "design_points": [p.__dict__ for p in self.explore_design_space()],
            "gpu_overhead": self.gpu_overhead_analysis(),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _extract_pareto_2d(
        self, points: List[Tuple[float, float]]
    ) -> List[int]:
        """Extract Pareto-optimal indices (minimize both dimensions)."""
        n = len(points)
        pareto = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if points[j][0] <= points[i][0] and points[j][1] <= points[i][1]:
                    if points[j][0] < points[i][0] or points[j][1] < points[i][1]:
                        dominated = True
                        break
            if not dominated:
                pareto.append(i)
        return pareto
