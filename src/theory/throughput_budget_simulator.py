"""
Throughput vs. Promotion Budget Simulator.

Models the key HPCA insight: as promotion budget increases,
utility improves but utility-per-byte declines, and throughput
degrades once transfer time exceeds compute time.

Generates data for Figure 2 of the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class InterconnectSpec:
    """Hardware interconnect specification."""
    name: str
    bandwidth_gbps: float       # GB/s
    base_latency_us: float      # per-transfer setup latency
    max_outstanding: int = 16   # max concurrent DMA requests

    @property
    def bandwidth_bytes_per_us(self) -> float:
        return self.bandwidth_gbps * 1e9 / 1e6


# Standard interconnect configurations
PCIE_4_0 = InterconnectSpec("PCIe_4.0", 32.0, 1.5, 8)
CXL_2_0 = InterconnectSpec("CXL_2.0", 64.0, 0.8, 16)
CXL_3_0 = InterconnectSpec("CXL_3.0", 128.0, 0.4, 32)
HBM_3 = InterconnectSpec("HBM3", 3200.0, 0.01, 256)


@dataclass
class ThroughputPoint:
    """Single point on the throughput-budget curve."""
    budget_ratio: float
    num_chunks_promoted: int
    transfer_bytes: int
    transfer_time_us: float
    compute_time_us: float
    exposed_latency_us: float
    total_step_time_us: float
    throughput_tokens_per_sec: float
    total_utility: float
    utility_per_byte: float
    marginal_utility: float     # utility gain from last chunk
    is_bandwidth_bound: bool


@dataclass
class ThroughputCurve:
    """Complete throughput curve for one interconnect."""
    interconnect: str
    points: List[ThroughputPoint]
    critical_budget_ratio: float  # where exposed latency first appears
    pareto_optimal_ratio: float   # best utility-per-byte × throughput


class ThroughputBudgetSimulator:
    """Simulates throughput vs. promotion budget tradeoff.

    Models the three-way tension:
    1. More promotion → higher utility (submodular growth)
    2. More promotion → more transfer bytes → lower throughput
    3. Transfer time > compute time → exposed latency → throughput cliff
    """

    def __init__(
        self,
        total_chunks: int = 1000,
        chunk_size_tokens: int = 64,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        dtype_bytes: int = 2,       # fp16
        compute_time_us: float = 300.0,  # typical decode step
    ):
        self.total_chunks = total_chunks
        self.chunk_bytes = chunk_size_tokens * 2 * num_kv_heads * head_dim * dtype_bytes
        self.compute_time_us = compute_time_us

    def simulate(
        self,
        interconnect: InterconnectSpec,
        budget_ratios: Optional[List[float]] = None,
        attention_distribution: Optional[np.ndarray] = None,
    ) -> ThroughputCurve:
        """Simulate throughput curve for a given interconnect.

        Args:
            interconnect: Hardware spec
            budget_ratios: Promotion budget ratios to sweep
            attention_distribution: Per-chunk attention masses (for utility)
        """
        if budget_ratios is None:
            budget_ratios = [
                0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15,
                0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0,
            ]

        if attention_distribution is None:
            # Default: Zipf-like attention (realistic for LLMs)
            attention_distribution = self._generate_zipf_attention(self.total_chunks)

        # Sort chunks by attention mass (greedy promotion order)
        sorted_indices = np.argsort(attention_distribution)[::-1]
        sorted_masses = attention_distribution[sorted_indices]

        points = []
        prev_utility = 0.0

        for ratio in budget_ratios:
            n_promoted = int(ratio * self.total_chunks)
            transfer_bytes = n_promoted * self.chunk_bytes

            # Transfer time with pipelining
            if n_promoted == 0:
                transfer_time = 0.0
            else:
                # Pipelined: first chunk latency + remaining at bandwidth
                first_chunk_time = (
                    interconnect.base_latency_us
                    + self.chunk_bytes / interconnect.bandwidth_bytes_per_us
                )
                if n_promoted > 1:
                    remaining_time = (
                        (n_promoted - 1) * self.chunk_bytes
                        / interconnect.bandwidth_bytes_per_us
                    )
                else:
                    remaining_time = 0.0
                # With outstanding requests, overlap some transfers
                pipeline_factor = min(n_promoted, interconnect.max_outstanding)
                transfer_time = first_chunk_time + remaining_time / max(pipeline_factor, 1)

            # Exposed latency
            exposed = max(0.0, transfer_time - self.compute_time_us)

            # Total step time
            total_step = self.compute_time_us + exposed

            # Throughput
            throughput = 1e6 / total_step if total_step > 0 else float('inf')

            # Utility: submodular (sum of sorted attention masses with diminishing returns)
            if n_promoted > 0:
                cumulative_masses = np.cumsum(sorted_masses[:n_promoted])
                # Submodular utility: 1 - product(1 - mass_i)
                # Approximated as sum with diminishing factor
                total_utility = float(cumulative_masses[-1])
                marginal = float(sorted_masses[n_promoted - 1]) if n_promoted > 0 else 0.0
            else:
                total_utility = 0.0
                marginal = 0.0

            utility_per_byte = total_utility / max(transfer_bytes, 1)

            points.append(ThroughputPoint(
                budget_ratio=ratio,
                num_chunks_promoted=n_promoted,
                transfer_bytes=transfer_bytes,
                transfer_time_us=transfer_time,
                compute_time_us=self.compute_time_us,
                exposed_latency_us=exposed,
                total_step_time_us=total_step,
                throughput_tokens_per_sec=throughput,
                total_utility=total_utility,
                utility_per_byte=utility_per_byte,
                marginal_utility=marginal,
                is_bandwidth_bound=exposed > 0,
            ))

            prev_utility = total_utility

        # Find critical budget ratio (first exposed latency)
        critical = 1.0
        for p in points:
            if p.exposed_latency_us > 0:
                critical = p.budget_ratio
                break

        # Find Pareto-optimal ratio (best utility × throughput product)
        best_product = 0.0
        pareto_ratio = 0.0
        for p in points:
            product = p.total_utility * p.throughput_tokens_per_sec
            if product > best_product:
                best_product = product
                pareto_ratio = p.budget_ratio

        return ThroughputCurve(
            interconnect=interconnect.name,
            points=points,
            critical_budget_ratio=critical,
            pareto_optimal_ratio=pareto_ratio,
        )

    def compare_interconnects(
        self,
        interconnects: Optional[List[InterconnectSpec]] = None,
        budget_ratios: Optional[List[float]] = None,
    ) -> Dict[str, ThroughputCurve]:
        """Compare throughput curves across interconnects."""
        if interconnects is None:
            interconnects = [PCIE_4_0, CXL_2_0, CXL_3_0]

        results = {}
        for ic in interconnects:
            results[ic.name] = self.simulate(ic, budget_ratios)
        return results

    def find_optimal_budget(
        self,
        interconnect: InterconnectSpec,
        min_throughput_fraction: float = 0.9,
    ) -> Dict[str, Any]:
        """Find the optimal promotion budget that maximizes utility
        while maintaining at least min_throughput_fraction of peak throughput.
        """
        curve = self.simulate(interconnect)
        peak_throughput = max(p.throughput_tokens_per_sec for p in curve.points)
        threshold = peak_throughput * min_throughput_fraction

        best_utility = 0.0
        best_ratio = 0.0
        for p in curve.points:
            if p.throughput_tokens_per_sec >= threshold and p.total_utility > best_utility:
                best_utility = p.total_utility
                best_ratio = p.budget_ratio

        return {
            "optimal_budget_ratio": best_ratio,
            "utility_at_optimal": best_utility,
            "throughput_at_optimal": next(
                p.throughput_tokens_per_sec for p in curve.points
                if p.budget_ratio == best_ratio
            ),
            "peak_throughput": peak_throughput,
            "throughput_retention": min_throughput_fraction,
            "critical_budget_ratio": curve.critical_budget_ratio,
        }

    @staticmethod
    def _generate_zipf_attention(n: int, alpha: float = 1.2, seed: int = 42) -> np.ndarray:
        """Generate Zipf-like attention distribution (realistic for LLMs)."""
        rng = np.random.RandomState(seed)
        ranks = np.arange(1, n + 1, dtype=np.float64)
        masses = 1.0 / np.power(ranks, alpha)
        # Add noise
        masses *= (1.0 + 0.1 * rng.randn(n))
        masses = np.maximum(masses, 0.0)
        masses /= masses.sum()
        return masses
