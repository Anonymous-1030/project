"""
Design space exploration for PHT/PTB parameters.

Generates data for HPCA evaluation figures:
- PHT entries vs prediction accuracy proxy
- PTB entries vs prefetch hit rate proxy
- Combined area vs accuracy Pareto frontier
- PHT/PTB configuration comparison table
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from src.hardware.ppu.pht import PHTConfig
from src.hardware.ppu.ptb import PTBConfig
from src.hardware.ppu.pht_ptb_cacti import PHTCACTIModel


@dataclass
class PHTDesignPoint:
    """A single point in the PHT/PTB design space."""

    label: str
    pht_entries: int
    ptb_entries: int
    counter_bits: int
    pht_area_mm2: float
    ptb_area_mm2: float
    total_area_mm2: float
    total_power_mw: float
    prediction_accuracy_proxy: float
    prefetch_hit_rate_proxy: float


@dataclass
class SweepResult:
    """Result of a parameter sweep."""

    parameter_name: str
    values: List[Any]
    areas: List[float]
    powers: List[float]
    accuracy_proxies: List[float]


class PHTDesignSpaceExplorer:
    """Sweep PHT/PTB parameters for HPCA evaluation figures."""

    def __init__(
        self,
        technology_node_nm: int = 7,
        clock_frequency_ghz: float = 1.5,
        delta_locality: float = 0.7,
        num_active_chunks: int = 100,
    ):
        self.tech_nm = technology_node_nm
        self.clock_ghz = clock_frequency_ghz
        self.delta_locality = delta_locality
        self.num_active = num_active_chunks

    def _accuracy_proxy(
        self, num_entries: int, counter_bits: int, adaptation_steps: int = 20
    ) -> float:
        """Estimate prediction accuracy from Theorem 4 bound.

        accuracy = 1 - misprediction_bound
        """
        intrinsic = 1.0 - self.delta_locality
        n_alias = self.num_active * (self.num_active - 1) / (2.0 * num_entries)
        aliasing = n_alias / num_entries
        tau = 2.0 ** counter_bits
        quant = 2.0 ** (1 - counter_bits) * math.exp(-adaptation_steps / tau)
        bound = intrinsic + aliasing + quant
        return max(0.0, 1.0 - min(1.0, bound))

    def _hit_rate_proxy(self, ptb_entries: int, max_age: int = 50) -> float:
        """Estimate PTB hit rate based on entry count and temporal locality."""
        # More entries → higher coverage of active promotion targets
        # Diminishing returns modeled as 1 - e^{-entries/scale}
        scale = self.num_active * 0.3
        coverage = 1.0 - math.exp(-ptb_entries / scale)
        return coverage * self.delta_locality

    def sweep_pht_entries(
        self, entries_range: Optional[List[int]] = None
    ) -> SweepResult:
        """Sweep PHT entries: area vs accuracy."""
        if entries_range is None:
            entries_range = [256, 512, 1024, 2048, 4096]

        areas, powers, accuracies = [], [], []
        for n in entries_range:
            pht_cfg = PHTConfig(num_entries=n)
            ptb_cfg = PTBConfig()
            model = PHTCACTIModel(pht_cfg, ptb_cfg, self.tech_nm, self.clock_ghz)
            report = model.estimate()
            areas.append(report.pht_area_mm2)
            powers.append(report.total_power_mw)
            accuracies.append(self._accuracy_proxy(n, 2))

        return SweepResult(
            parameter_name="pht_entries",
            values=entries_range,
            areas=areas,
            powers=powers,
            accuracy_proxies=accuracies,
        )

    def sweep_ptb_entries(
        self, entries_range: Optional[List[int]] = None
    ) -> SweepResult:
        """Sweep PTB entries: area vs hit rate."""
        if entries_range is None:
            entries_range = [8, 16, 32, 64, 128]

        areas, powers, hit_rates = [], [], []
        for n in entries_range:
            pht_cfg = PHTConfig()
            ptb_cfg = PTBConfig(num_entries=n)
            model = PHTCACTIModel(pht_cfg, ptb_cfg, self.tech_nm, self.clock_ghz)
            report = model.estimate()
            areas.append(report.ptb_area_mm2)
            powers.append(report.total_power_mw)
            hit_rates.append(self._hit_rate_proxy(n))

        return SweepResult(
            parameter_name="ptb_entries",
            values=entries_range,
            areas=areas,
            powers=powers,
            accuracy_proxies=hit_rates,
        )

    def sweep_counter_bits(
        self, bits_range: Optional[List[int]] = None
    ) -> SweepResult:
        """Sweep counter bits: accuracy vs area."""
        if bits_range is None:
            bits_range = [1, 2, 3, 4]

        areas, powers, accuracies = [], [], []
        for k in bits_range:
            pht_cfg = PHTConfig(counter_bits=k)
            ptb_cfg = PTBConfig()
            model = PHTCACTIModel(pht_cfg, ptb_cfg, self.tech_nm, self.clock_ghz)
            report = model.estimate()
            areas.append(report.pht_area_mm2)
            powers.append(report.total_power_mw)
            accuracies.append(self._accuracy_proxy(1024, k))

        return SweepResult(
            parameter_name="counter_bits",
            values=bits_range,
            areas=areas,
            powers=powers,
            accuracy_proxies=accuracies,
        )

    def explore_design_space(self) -> List[PHTDesignPoint]:
        """Enumerate representative PHT/PTB configurations."""
        configs = [
            ("PHT-Tiny",  256,  16, 2),
            ("PHT-Small", 512,  32, 2),
            ("PHT-Base",  1024, 32, 2),
            ("PHT-Large", 2048, 64, 2),
            ("PHT-XL",    4096, 128, 2),
        ]

        points = []
        for label, pht_n, ptb_n, k in configs:
            pht_cfg = PHTConfig(num_entries=pht_n, counter_bits=k)
            ptb_cfg = PTBConfig(num_entries=ptb_n)
            model = PHTCACTIModel(pht_cfg, ptb_cfg, self.tech_nm, self.clock_ghz)
            report = model.estimate()

            points.append(PHTDesignPoint(
                label=label,
                pht_entries=pht_n,
                ptb_entries=ptb_n,
                counter_bits=k,
                pht_area_mm2=report.pht_area_mm2,
                ptb_area_mm2=report.ptb_area_mm2,
                total_area_mm2=report.total_area_mm2,
                total_power_mw=report.total_power_mw,
                prediction_accuracy_proxy=self._accuracy_proxy(pht_n, k),
                prefetch_hit_rate_proxy=self._hit_rate_proxy(ptb_n),
            ))

        return points

    def combined_area_budget_analysis(self) -> Dict[str, Any]:
        """Verify PHT+PTB fits within area budget at 7nm."""
        points = self.explore_design_space()
        return {
            "design_points": [
                {
                    "label": p.label,
                    "total_area_mm2": p.total_area_mm2,
                    "total_power_mw": p.total_power_mw,
                    "fits_0.02_budget": p.total_area_mm2 < 0.02,
                    "fits_0.03_budget": p.total_area_mm2 < 0.03,
                    "accuracy": p.prediction_accuracy_proxy,
                    "hit_rate": p.prefetch_hit_rate_proxy,
                }
                for p in points
            ],
            "recommended": "PHT-Base",
            "recommended_area": next(
                p.total_area_mm2 for p in points if p.label == "PHT-Base"
            ),
        }
