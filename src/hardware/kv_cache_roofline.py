"""
KV Cache Roofline Model for ProSE-X 2.0.

Core HPCA theoretical contribution: an analytical model analogous to the
classical Roofline model (Williams et al., 2009) but adapted for KV cache
management in heterogeneous memory systems.

Key insight: KV cache promotion is bounded by TWO ceilings:
  1. COMPUTE ceiling: sparse attention throughput (FLOP/s)
  2. BANDWIDTH ceiling: host->HBM transfer throughput (GB/s)

The "Operational Intensity" axis measures how much useful attention mass
is recovered per byte promoted (Attention-Mass / Byte), analogous to
FLOP/Byte in the classical model.

This model enables:
  - Predicting whether a given budget ratio is compute- or bandwidth-bound
  - Deriving the optimal promotion budget for a given hardware config
  - Comparing ProSE vs baselines on the same roofline chart
  - CXL-aware prefetch protocol (Fix 10): intelligent path selection
    between PCIe and CXL with latency-aware prefetch depth adaptation
  - Bandwidth-utility Pareto analysis (Fix 11): sweep promotion budgets
    to identify Pareto-optimal operating points for paper figures
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum

from src.theory.kv_roofline_formal import (
    FormalKVCacheRoofline,
    ChunkUtilityRecord,
)
from src.theory.optimal_promotion import PromotionItem, OptimalPromotionSolver

logger = logging.getLogger(__name__)


@dataclass
class HardwareSpec:
    """Hardware specification for roofline modeling."""
    name: str = "H100-PCIe"

    # Compute specs
    sparse_attn_tflops: float = 50.0
    dense_attn_tflops: float = 30.0

    # Memory specs
    hbm_capacity_gb: float = 80.0
    hbm_bandwidth_gbps: float = 3350.0

    # Interconnect specs
    pcie_bandwidth_gbps: float = 32.0
    pcie_latency_us: float = 5.0
    cxl_bandwidth_gbps: float = 64.0
    cxl_latency_us: float = 0.5

    # Effective bandwidth (accounting for protocol overhead)
    pcie_efficiency: float = 0.75
    cxl_efficiency: float = 0.80


HARDWARE_PRESETS = {
    "H100-SXM": HardwareSpec(
        name="H100-SXM",
        sparse_attn_tflops=60.0, dense_attn_tflops=40.0,
        hbm_capacity_gb=80.0, hbm_bandwidth_gbps=3350.0,
        pcie_bandwidth_gbps=64.0, cxl_bandwidth_gbps=64.0,
    ),
    "H100-PCIe": HardwareSpec(),
    "A100-80G": HardwareSpec(
        name="A100-80G",
        sparse_attn_tflops=35.0, dense_attn_tflops=20.0,
        hbm_capacity_gb=80.0, hbm_bandwidth_gbps=2039.0,
        pcie_bandwidth_gbps=32.0, cxl_bandwidth_gbps=32.0,
    ),
    "L40S": HardwareSpec(
        name="L40S",
        sparse_attn_tflops=25.0, dense_attn_tflops=15.0,
        hbm_capacity_gb=48.0, hbm_bandwidth_gbps=864.0,
        pcie_bandwidth_gbps=32.0, cxl_bandwidth_gbps=32.0,
    ),
}


@dataclass
class RooflinePoint:
    """A single point on the roofline chart."""
    label: str
    operational_intensity: float   # Recovered attention mass per byte promoted
    throughput: float              # Effective tokens/s or attention ops/s
    is_bandwidth_bound: bool
    is_compute_bound: bool
    ridge_point: float
    utilization: float             # Fraction of peak achieved [0, 1]


@dataclass
class RooflineAnalysis:
    """Complete roofline analysis result."""
    hardware: HardwareSpec
    interconnect: str

    compute_ceiling: float
    bandwidth_ceiling: float
    ridge_point: float

    points: List[RooflinePoint] = field(default_factory=list)
    prose_utilization: float = 0.0
    baseline_utilizations: Dict[str, float] = field(default_factory=dict)


@dataclass
class BandwidthUtilityPoint:
    """One operating point in the bandwidth-utility tradeoff space."""
    label: str
    budget_ratio: float
    promoted_bytes: int
    utility: float
    utility_per_byte: float
    throughput: float
    exposed_latency_us: float
    interconnect: str
    is_pareto_optimal: bool = False


@dataclass
class ParetoFrontierAnalysis:
    """Result of sweeping operating points and extracting the Pareto frontier."""
    hardware: HardwareSpec
    interconnect: str
    points: List[BandwidthUtilityPoint] = field(default_factory=list)
    pareto_frontier: List[BandwidthUtilityPoint] = field(default_factory=list)
    recommended_point: Optional[BandwidthUtilityPoint] = None


class KVCacheRoofline:
    """
    KV Cache Roofline analytical model.

    Attainable Performance = min(
        Compute_Ceiling,
        Bandwidth_Ceiling x Operational_Intensity
    )

    Operational Intensity (OI) = recovered_attention_mass / bytes_promoted

    Higher OI means each promoted byte contributes more useful attention,
    indicating a smarter promotion policy.
    """

    def __init__(self, hardware: Optional[HardwareSpec] = None):
        self.hardware = hardware or HardwareSpec()
        self.formal_model = FormalKVCacheRoofline()

    def compute_operational_intensity(
        self,
        recovered_attention_mass: float,
        bytes_promoted: int,
    ) -> float:
        """
        OI = total recovered attention mass / bytes promoted.

        Args:
            recovered_attention_mass: Total attention mass recovered by
                                      promoted chunks (sum over decode steps)
            bytes_promoted: Total bytes transferred from host to HBM
        """
        if bytes_promoted <= 0:
            return 0.0
        return recovered_attention_mass / bytes_promoted

    def compute_formal_operational_intensity(
        self,
        records: Iterable[ChunkUtilityRecord],
    ) -> float:
        """Rigorous OI: sum_i utility_i * attention_mass_i / sum_i bytes_i."""
        return self.formal_model.compute_operational_intensity(records)

    def solve_optimal_promotion(
        self,
        items: Iterable[PromotionItem],
        bandwidth_budget: int,
        hbm_budget: int,
    ) -> Dict[str, object]:
        """Dual-constrained admission solved as a small 0-1 knapsack variant."""
        solver = OptimalPromotionSolver()
        return solver.solve(list(items), bandwidth_budget=bandwidth_budget, hbm_budget=hbm_budget)

    def compute_ceilings(
        self,
        interconnect: str = "pcie",
    ) -> Tuple[float, float, float]:
        """
        Compute the two roofline ceilings and ridge point.

        Returns:
            (compute_ceiling_gflops, bandwidth_ceiling_gbps, ridge_oi)
        """
        compute_ceiling = self.hardware.sparse_attn_tflops * 1e3  # GFLOP/s

        if interconnect == "pcie":
            bw = self.hardware.pcie_bandwidth_gbps * self.hardware.pcie_efficiency
        elif interconnect == "cxl":
            bw = self.hardware.cxl_bandwidth_gbps * self.hardware.cxl_efficiency
        else:
            raise ValueError(f"Unknown interconnect: {interconnect}")

        ridge_point = compute_ceiling / bw if bw > 0 else float('inf')
        return compute_ceiling, bw, ridge_point

    def analyze(
        self,
        method_results: Dict[str, Dict[str, float]],
        interconnect: str = "pcie",
    ) -> RooflineAnalysis:
        """
        Run roofline analysis for multiple methods.

        Args:
            method_results: method_name -> {
                "recovered_attention_mass": float,
                "bytes_promoted": int,
                "effective_throughput": float,
            }
            interconnect: "pcie" or "cxl"

        Returns:
            RooflineAnalysis with all methods plotted
        """
        compute_ceil, bw_ceil, ridge = self.compute_ceilings(interconnect)

        analysis = RooflineAnalysis(
            hardware=self.hardware,
            interconnect=interconnect,
            compute_ceiling=compute_ceil,
            bandwidth_ceiling=bw_ceil,
            ridge_point=ridge,
        )

        for method_name, results in method_results.items():
            oi = self.compute_operational_intensity(
                results.get("recovered_attention_mass", 0.0),
                results.get("bytes_promoted", 1),
            )
            throughput = results.get("effective_throughput", 0.0)

            bw_limited = bw_ceil * oi
            is_bw_bound = oi < ridge
            theoretical_max = min(compute_ceil, bw_limited) if oi > 0 else compute_ceil
            utilization = throughput / theoretical_max if theoretical_max > 0 else 0.0

            point = RooflinePoint(
                label=method_name,
                operational_intensity=oi,
                throughput=throughput,
                is_bandwidth_bound=is_bw_bound,
                is_compute_bound=not is_bw_bound,
                ridge_point=ridge,
                utilization=min(1.0, utilization),
            )
            analysis.points.append(point)

            if method_name.lower() == "prose":
                analysis.prose_utilization = point.utilization
            else:
                analysis.baseline_utilizations[method_name] = point.utilization

        return analysis

    def derive_optimal_budget(
        self,
        seq_len: int,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        interconnect: str = "pcie",
        decode_time_us: float = 100.0,
    ) -> Dict[str, float]:
        """
        Derive the theoretically optimal promotion budget.

        At the ridge point, compute and bandwidth are balanced.
        The optimal promotion budget B* satisfies:
            B* = BW_eff x T_decode
        """
        bytes_per_token = 2 * num_kv_heads * head_dim * 2  # K+V, fp16

        if interconnect == "pcie":
            bw_eff = self.hardware.pcie_bandwidth_gbps * self.hardware.pcie_efficiency
            latency = self.hardware.pcie_latency_us
        else:
            bw_eff = self.hardware.cxl_bandwidth_gbps * self.hardware.cxl_efficiency
            latency = self.hardware.cxl_latency_us

        available_time_us = max(0, decode_time_us - latency)
        max_bytes = bw_eff * 1e9 * (available_time_us / 1e6)
        max_tokens = int(max_bytes / bytes_per_token)
        optimal_ratio = max_tokens / seq_len if seq_len > 0 else 0.0

        hbm_max_tokens = int(self.hardware.hbm_capacity_gb * 1e9 * 0.8 / bytes_per_token)
        hbm_ratio = hbm_max_tokens / seq_len if seq_len > 0 else 1.0
        effective_ratio = min(optimal_ratio, hbm_ratio)

        return {
            "optimal_promote_ratio": effective_ratio,
            "bandwidth_limited_tokens": max_tokens,
            "hbm_limited_tokens": hbm_max_tokens,
            "bottleneck": "bandwidth" if optimal_ratio < hbm_ratio else "capacity",
            "max_overlappable_bytes": max_bytes,
            "decode_time_us": decode_time_us,
            "interconnect": interconnect,
        }

    def generate_roofline_data(
        self,
        analysis: RooflineAnalysis,
        oi_range: Optional[Tuple[float, float]] = None,
        num_points: int = 100,
    ) -> Dict[str, Any]:
        """Generate data for plotting the roofline chart."""
        if oi_range is None:
            oi_range = (1e-6, analysis.ridge_point * 4)

        oi_values = [
            oi_range[0] * (oi_range[1] / oi_range[0]) ** (i / (num_points - 1))
            for i in range(num_points)
        ]

        envelope = [
            min(analysis.compute_ceiling, analysis.bandwidth_ceiling * oi)
            for oi in oi_values
        ]

        return {
            "oi_values": oi_values,
            "envelope": envelope,
            "compute_ceiling": analysis.compute_ceiling,
            "bandwidth_ceiling": analysis.bandwidth_ceiling,
            "ridge_point": analysis.ridge_point,
            "method_points": [
                {
                    "label": p.label,
                    "oi": p.operational_intensity,
                    "throughput": p.throughput,
                    "bound": "bandwidth" if p.is_bandwidth_bound else "compute",
                    "utilization": p.utilization,
                }
                for p in analysis.points
            ],
        }

    def sweep_promotion_budgets(
        self,
        seq_len: int,
        utility_fn: Callable[[float], float],
        throughput_fn: Optional[Callable[[float], float]] = None,
        budget_ratios: Optional[List[float]] = None,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        interconnect: str = "pcie",
        decode_time_us: float = 100.0,
        label_prefix: str = "prose",
    ) -> ParetoFrontierAnalysis:
        """
        Sweep budget ratios and build a bandwidth-utility Pareto set.

        Utility is user-provided to keep the model generic across evaluation
        workloads. Throughput can also be provided; otherwise we derive a simple
        throughput proxy penalized by exposed transfer latency.
        """
        if budget_ratios is None:
            budget_ratios = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]

        bytes_per_token = 2 * num_kv_heads * head_dim * 2
        points: List[BandwidthUtilityPoint] = []

        for ratio in budget_ratios:
            promoted_tokens = max(1, int(seq_len * ratio))
            promoted_bytes = promoted_tokens * bytes_per_token

            utility = max(0.0, float(utility_fn(ratio)))
            utility_per_byte = utility / max(promoted_bytes, 1)

            if interconnect == "pcie":
                transfer_us = self.hardware.pcie_latency_us + (
                    (promoted_bytes / (1024 ** 3)) / max(self.hardware.pcie_bandwidth_gbps * self.hardware.pcie_efficiency, 1e-9)
                ) * 1e6
            elif interconnect == "cxl":
                transfer_us = self.hardware.cxl_latency_us + (
                    (promoted_bytes / (1024 ** 3)) / max(self.hardware.cxl_bandwidth_gbps * self.hardware.cxl_efficiency, 1e-9)
                ) * 1e6
            else:
                raise ValueError(f"Unknown interconnect: {interconnect}")

            exposed_latency_us = max(0.0, transfer_us - decode_time_us)
            if throughput_fn is not None:
                throughput = float(throughput_fn(ratio))
            else:
                throughput = 1e6 / max(decode_time_us + exposed_latency_us, 1.0)

            points.append(BandwidthUtilityPoint(
                label=f"{label_prefix}@{ratio:.2f}",
                budget_ratio=ratio,
                promoted_bytes=promoted_bytes,
                utility=utility,
                utility_per_byte=utility_per_byte,
                throughput=throughput,
                exposed_latency_us=exposed_latency_us,
                interconnect=interconnect,
            ))

        frontier = self.extract_pareto_frontier(points)
        recommended = None
        if frontier:
            recommended = max(
                frontier,
                key=lambda p: (p.utility_per_byte * 0.4 + p.utility * 0.4 + p.throughput * 0.2)
            )

        return ParetoFrontierAnalysis(
            hardware=self.hardware,
            interconnect=interconnect,
            points=points,
            pareto_frontier=frontier,
            recommended_point=recommended,
        )

    def extract_pareto_frontier(
        self,
        points: List[BandwidthUtilityPoint],
    ) -> List[BandwidthUtilityPoint]:
        """Keep points not dominated in both utility and throughput."""
        frontier: List[BandwidthUtilityPoint] = []
        for point in points:
            dominated = False
            for other in points:
                if other is point:
                    continue
                no_worse = other.utility >= point.utility and other.throughput >= point.throughput
                strictly_better = other.utility > point.utility or other.throughput > point.throughput
                if no_worse and strictly_better:
                    dominated = True
                    break
            point.is_pareto_optimal = not dominated
            if not dominated:
                frontier.append(point)
        frontier.sort(key=lambda p: (p.promoted_bytes, -p.utility_per_byte))
        return frontier

    def summarize_pareto_analysis(self, analysis: ParetoFrontierAnalysis) -> Dict[str, Any]:
        """Return a compact summary for logging/reporting."""
        recommended = analysis.recommended_point
        return {
            "hardware": analysis.hardware.name,
            "interconnect": analysis.interconnect,
            "num_points": len(analysis.points),
            "pareto_count": len(analysis.pareto_frontier),
            "recommended": None if recommended is None else {
                "label": recommended.label,
                "budget_ratio": recommended.budget_ratio,
                "utility": recommended.utility,
                "utility_per_byte": recommended.utility_per_byte,
                "throughput": recommended.throughput,
                "exposed_latency_us": recommended.exposed_latency_us,
            },
        }
