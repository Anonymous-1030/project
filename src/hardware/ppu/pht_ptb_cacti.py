"""
CACTI-style area/power model for PHT and PTB.

Produces AreaPowerComponent-compatible estimates for integration
with the existing CACTIModel in cacti_model.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from src.hardware.ppu.pht import PHTConfig
from src.hardware.ppu.ptb import PTBConfig


@dataclass
class AreaPowerComponent:
    """Single hardware component area/power estimate."""

    name: str
    area_mm2: float
    power_mw: float
    latency_ns: float
    description: str


@dataclass
class PHTAreaPowerReport:
    """Complete area/power report for PHT + PTB."""

    components: List[AreaPowerComponent]
    total_area_mm2: float
    total_power_mw: float
    technology_node_nm: int
    pht_area_mm2: float
    ptb_area_mm2: float
    fits_budget: bool  # True if total < 0.03 mm²


class PHTCACTIModel:
    """CACTI-style area/power model for PHT + PTB.

    Uses analytical SRAM models calibrated to CACTI 7.0 at 7nm.
    """

    def __init__(
        self,
        pht_config: PHTConfig,
        ptb_config: PTBConfig,
        technology_node_nm: int = 7,
        clock_frequency_ghz: float = 1.5,
        sram_read_energy_pj: float = 0.5,
    ):
        self.pht_config = pht_config
        self.ptb_config = ptb_config
        self.tech_nm = technology_node_nm
        self.clock_ghz = clock_frequency_ghz
        self.sram_energy_pj = sram_read_energy_pj
        self._node_scale = technology_node_nm / 7.0  # normalize to 7nm

    def estimate_pht(self) -> AreaPowerComponent:
        """PHT: counter SRAM + hash unit + comparators.

        SRAM: num_entries × counter_bits / 8 bytes
        Hash unit: XOR tree for signature computation
        Comparators + control logic
        """
        # SRAM for counters
        sram_bytes = (
            self.pht_config.num_entries * self.pht_config.counter_bits
        ) / 8.0
        sram_kb = sram_bytes / 1024.0

        # SRAM area: calibrated to CACTI 7nm
        # ~0.0015 mm² per KB at 7nm for small SRAMs
        sram_area = max(0.0005, 0.0015 * sram_kb * self._node_scale)

        # Hash unit: XOR tree for 16-bit signature
        # ~0.001 mm² at 7nm (simple combinational logic)
        hash_area = 0.001 * self._node_scale

        # Comparators + mux + control
        control_area = 0.002 * self._node_scale

        total_area = sram_area + hash_area + control_area

        # Power: SRAM read + hash + control
        sram_power = sram_kb * self.sram_energy_pj * self.clock_ghz * 0.08
        logic_power = 1.5 * self._node_scale  # mW for hash + control
        total_power = max(1.0, sram_power + logic_power)

        # Latency: single cycle at target frequency
        latency_ns = 1.0 / self.clock_ghz

        return AreaPowerComponent(
            name="PHT (Promotion History Table)",
            area_mm2=round(total_area, 5),
            power_mw=round(total_power, 2),
            latency_ns=round(latency_ns, 3),
            description=(
                f"{self.pht_config.num_entries} entries × "
                f"{self.pht_config.counter_bits}-bit counters, "
                f"{sram_bytes:.0f}B SRAM + hash unit"
            ),
        )

    def estimate_ptb(self) -> AreaPowerComponent:
        """PTB: entry SRAM + tag CAM + LRU logic.

        SRAM: num_entries × entry_bytes
        Tag CAM: fully-associative tag comparison
        LRU: pseudo-LRU or counter-based
        """
        # SRAM for entries
        sram_bytes = self.ptb_config.num_entries * self.ptb_config.entry_bytes
        sram_kb = sram_bytes / 1024.0

        sram_area = max(0.0008, 0.0015 * sram_kb * self._node_scale)

        # Tag CAM: ~0.0001 mm² per entry at 7nm for small CAMs
        cam_area = (
            self.ptb_config.num_entries * 0.00005 * self._node_scale
        )

        # LRU logic: counter comparators
        lru_area = 0.001 * self._node_scale

        total_area = sram_area + cam_area + lru_area

        # Power
        sram_power = sram_kb * self.sram_energy_pj * self.clock_ghz * 0.08
        cam_power = self.ptb_config.num_entries * 0.05 * self._node_scale  # mW
        lru_power = 0.5 * self._node_scale
        total_power = max(1.5, sram_power + cam_power + lru_power)

        latency_ns = 1.0 / self.clock_ghz

        return AreaPowerComponent(
            name="PTB (Promotion Target Buffer)",
            area_mm2=round(total_area, 5),
            power_mw=round(total_power, 2),
            latency_ns=round(latency_ns, 3),
            description=(
                f"{self.ptb_config.num_entries} entries × "
                f"{self.ptb_config.entry_bytes}B, "
                f"fully-associative with "
                f"{self.ptb_config.eviction_policy.upper()} eviction"
            ),
        )

    def estimate(self) -> PHTAreaPowerReport:
        """Full area/power report for PHT + PTB."""
        pht = self.estimate_pht()
        ptb = self.estimate_ptb()
        components = [pht, ptb]

        total_area = sum(c.area_mm2 for c in components)
        total_power = sum(c.power_mw for c in components)

        return PHTAreaPowerReport(
            components=components,
            total_area_mm2=round(total_area, 5),
            total_power_mw=round(total_power, 2),
            technology_node_nm=self.tech_nm,
            pht_area_mm2=pht.area_mm2,
            ptb_area_mm2=ptb.area_mm2,
            fits_budget=total_area < 0.03,
        )
