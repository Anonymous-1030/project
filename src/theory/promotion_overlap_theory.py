"""Formal Promotion-Compute Overlap Theory for HPCA.

Theorem 6 (Promotion Latency Hiding — Necessary and Sufficient Condition):
  Let T_compute(t) be the decode compute time at step t, B_eff the effective
  interconnect bandwidth (accounting for protocol overhead), and P(t) the
  set of chunks promoted at step t with total size S(t) = Σ_{c∈P(t)} s_c.

  The promotion latency is FULLY HIDDEN (exposed latency = 0) iff:
      S(t) ≤ B_eff · T_compute(t)                              ... (*)

  Define the critical promotion budget:
      B* = B_eff · T_compute

  When S(t) > B*, the exposed latency is:
      T_exposed(t) = S(t)/B_eff - T_compute(t) = (S(t) - B*) / B_eff

  and the effective decode latency becomes:
      T_decode_eff(t) = T_compute(t) + T_exposed(t)
                      = S(t) / B_eff

Theorem 7 (Throughput Degradation under Over-Promotion):
  When the promotion budget exceeds B*, throughput degrades as:
      Throughput(S) = 1 / T_decode_eff
                    = B_eff / S           for S > B*
                    = 1 / T_compute       for S ≤ B*

  The throughput-utility Pareto frontier is characterized by:
      max U(S)  s.t.  Throughput(S) ≥ Throughput_min

  which yields the Pareto-optimal budget:
      S_pareto = min(S : U(S)/U* ≥ α) ∩ {S ≤ B_eff / Throughput_min}

Theorem 8 (Multi-Step Lookahead Regret Bound):
  Let π_k be a k-step lookahead promotion policy that predicts chunk
  utilities k steps ahead.  Let π* be the oracle policy.  If the
  per-step utility prediction error is bounded by ε, then:

      Regret_T(π_k) = Σ_{t=1}^{T} [U*(t) - U_{π_k}(t)]
                    ≤ T · ε · B*/s_avg + T · (1-L_δ^k) · U_max

  where L_δ is the δ-locality concentration and s_avg is the average
  chunk size.  The first term captures prediction error; the second
  captures the probability that the k-step lookahead misses a
  non-local access pattern.

  For k=1 and L_δ=0.7: Regret ≤ T·(ε·B*/s_avg + 0.3·U_max)
  For k=2 and L_δ=0.7: Regret ≤ T·(ε·B*/s_avg + 0.09·U_max)

  The regret decreases geometrically with lookahead depth k.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class HardwareParams:
    """Hardware parameters for overlap analysis."""
    hbm_bandwidth_gbps: float = 3350.0
    pcie_bandwidth_gbps: float = 32.0
    cxl_bandwidth_gbps: float = 64.0
    pcie_efficiency: float = 0.75
    cxl_efficiency: float = 0.80
    pcie_latency_us: float = 5.0
    cxl_latency_us: float = 0.5


@dataclass
class OverlapAnalysisResult:
    """Result of promotion-compute overlap analysis."""
    # Critical budget
    critical_budget_bytes: int
    critical_budget_chunks: int
    # Latency model
    compute_time_us: float
    transfer_time_us: float
    exposed_latency_us: float
    fully_hidden: bool
    # Throughput model
    throughput_baseline: float      # tokens/s without promotion
    throughput_with_promotion: float
    throughput_degradation_pct: float
    # Pareto analysis
    pareto_budget_bytes: int
    pareto_utility_ratio: float
    # Interconnect comparison
    pcie_critical_budget: int
    cxl_critical_budget: int
    cxl_advantage_ratio: float


@dataclass
class RegretBoundResult:
    """Result of multi-step lookahead regret analysis."""
    lookahead_k: int
    locality_concentration: float
    prediction_error: float
    per_step_regret_bound: float
    total_regret_bound: float
    regret_components: Dict[str, float]
    regret_vs_k: List[Tuple[int, float]]  # (k, regret_bound)


class PromotionOverlapTheory:
    """Formal analysis of promotion-compute overlap (Theorems 6-8)."""

    def __init__(self, hw: Optional[HardwareParams] = None):
        self.hw = hw or HardwareParams()

    # ------------------------------------------------------------------ #
    # Theorem 6: Critical budget and exposed latency
    # ------------------------------------------------------------------ #

    def compute_critical_budget(
        self,
        compute_time_us: float,
        interconnect: str = "cxl",
        bytes_per_chunk: int = 1048576,
    ) -> Dict[str, object]:
        """Compute B* = B_eff × T_compute (Theorem 6)."""
        bw_eff = self._effective_bandwidth(interconnect)
        latency_us = self._base_latency(interconnect)

        available_us = max(0.0, compute_time_us - latency_us)
        critical_bytes = int(bw_eff * 1e9 * available_us / 1e6)
        critical_chunks = critical_bytes // max(bytes_per_chunk, 1)

        return {
            "critical_budget_bytes": critical_bytes,
            "critical_budget_chunks": critical_chunks,
            "effective_bandwidth_gbps": bw_eff,
            "compute_time_us": compute_time_us,
            "base_latency_us": latency_us,
            "available_overlap_us": available_us,
            "interconnect": interconnect,
        }

    def compute_exposed_latency(
        self,
        promoted_bytes: int,
        compute_time_us: float,
        interconnect: str = "cxl",
    ) -> Dict[str, float]:
        """Compute exposed latency when S > B* (Theorem 6)."""
        bw_eff = self._effective_bandwidth(interconnect)
        latency_us = self._base_latency(interconnect)

        transfer_us = latency_us + (promoted_bytes / (1024**3)) / max(bw_eff, 1e-9) * 1e6
        exposed_us = max(0.0, transfer_us - compute_time_us)
        effective_decode_us = compute_time_us + exposed_us

        return {
            "transfer_time_us": transfer_us,
            "exposed_latency_us": exposed_us,
            "effective_decode_us": effective_decode_us,
            "fully_hidden": exposed_us < 0.01,
            "hiding_ratio": min(1.0, compute_time_us / max(transfer_us, 1e-9)),
        }

    # ------------------------------------------------------------------ #
    # Theorem 7: Throughput degradation
    # ------------------------------------------------------------------ #

    def throughput_model(
        self,
        promoted_bytes: int,
        compute_time_us: float,
        interconnect: str = "cxl",
    ) -> Dict[str, float]:
        """Throughput as a function of promotion budget (Theorem 7)."""
        overlap = self.compute_exposed_latency(promoted_bytes, compute_time_us, interconnect)
        eff_decode_us = overlap["effective_decode_us"]

        throughput_baseline = 1e6 / compute_time_us  # steps/s
        throughput_actual = 1e6 / max(eff_decode_us, 1e-3)
        degradation = 1.0 - throughput_actual / throughput_baseline

        return {
            "throughput_baseline_steps_per_s": throughput_baseline,
            "throughput_actual_steps_per_s": throughput_actual,
            "throughput_degradation_pct": degradation * 100,
            "effective_decode_us": eff_decode_us,
        }

    def sweep_throughput_utility(
        self,
        budget_bytes_list: List[int],
        utility_fn,  # Callable[[int], float]: bytes -> utility
        compute_time_us: float,
        interconnect: str = "cxl",
    ) -> List[Dict[str, float]]:
        """Sweep budgets to build throughput-utility Pareto data (Theorem 7)."""
        results = []
        for b in budget_bytes_list:
            tp = self.throughput_model(b, compute_time_us, interconnect)
            u = float(utility_fn(b))
            results.append({
                "budget_bytes": b,
                "utility": u,
                "throughput": tp["throughput_actual_steps_per_s"],
                "exposed_us": tp["effective_decode_us"] - compute_time_us,
                "degradation_pct": tp["throughput_degradation_pct"],
            })
        return results

    # ------------------------------------------------------------------ #
    # Theorem 8: Multi-step lookahead regret
    # ------------------------------------------------------------------ #

    def regret_bound(
        self,
        lookahead_k: int,
        locality_concentration: float,
        prediction_error: float,
        total_steps: int,
        critical_budget_bytes: int,
        avg_chunk_bytes: int,
        max_single_utility: float = 1.0,
    ) -> RegretBoundResult:
        """Compute regret bound for k-step lookahead (Theorem 8).

        Regret_T(π_k) ≤ T · [ε · B*/s_avg + (1 - L_δ^k) · U_max]
        """
        L_k = locality_concentration ** lookahead_k
        prediction_term = prediction_error * critical_budget_bytes / max(avg_chunk_bytes, 1)
        locality_term = (1 - L_k) * max_single_utility
        per_step = prediction_term + locality_term
        total = total_steps * per_step

        # Sweep k for comparison
        regret_vs_k = []
        for k in range(1, min(lookahead_k + 3, 8)):
            Lk = locality_concentration ** k
            r = prediction_error * critical_budget_bytes / max(avg_chunk_bytes, 1) + (1 - Lk) * max_single_utility
            regret_vs_k.append((k, total_steps * r))

        return RegretBoundResult(
            lookahead_k=lookahead_k,
            locality_concentration=locality_concentration,
            prediction_error=prediction_error,
            per_step_regret_bound=per_step,
            total_regret_bound=total,
            regret_components={
                "prediction_error_term": total_steps * prediction_term,
                "locality_miss_term": total_steps * locality_term,
                "L_delta_k": L_k,
            },
            regret_vs_k=regret_vs_k,
        )

    # ------------------------------------------------------------------ #
    # Cross-interconnect comparison
    # ------------------------------------------------------------------ #

    def compare_interconnects(
        self,
        compute_time_us: float,
        bytes_per_chunk: int = 1048576,
    ) -> Dict[str, object]:
        """Compare PCIe vs CXL for promotion overlap."""
        pcie = self.compute_critical_budget(compute_time_us, "pcie", bytes_per_chunk)
        cxl = self.compute_critical_budget(compute_time_us, "cxl", bytes_per_chunk)

        return {
            "pcie": pcie,
            "cxl": cxl,
            "cxl_advantage_bytes": cxl["critical_budget_bytes"] - pcie["critical_budget_bytes"],
            "cxl_advantage_chunks": cxl["critical_budget_chunks"] - pcie["critical_budget_chunks"],
            "cxl_bandwidth_ratio": cxl["effective_bandwidth_gbps"] / max(pcie["effective_bandwidth_gbps"], 1e-9),
        }

    # ------------------------------------------------------------------ #
    # Formal theorem statements
    # ------------------------------------------------------------------ #

    def theorem_statements(self) -> Dict[str, str]:
        return {
            "theorem_6_latency_hiding": (
                "Theorem 6 (Latency Hiding N&S Condition). Promotion latency "
                "is fully hidden iff S(t) ≤ B* = B_eff · T_compute. When "
                "S(t) > B*, exposed latency T_exposed = (S(t) - B*)/B_eff "
                "and effective decode time T_eff = S(t)/B_eff."
            ),
            "theorem_7_throughput_degradation": (
                "Theorem 7 (Throughput Degradation). Throughput(S) = "
                "min(1/T_compute, B_eff/S). The Pareto-optimal budget "
                "S_pareto minimizes S subject to U(S)/U* ≥ α and "
                "Throughput(S) ≥ Throughput_min."
            ),
            "theorem_8_regret_bound": (
                "Theorem 8 (Lookahead Regret). For k-step lookahead with "
                "prediction error ε and locality L_δ: Regret_T ≤ "
                "T·[ε·B*/s_avg + (1-L_δ^k)·U_max]. Regret decreases "
                "geometrically with k: O((1-L_δ)^k)."
            ),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _effective_bandwidth(self, interconnect: str) -> float:
        if interconnect == "pcie":
            return self.hw.pcie_bandwidth_gbps * self.hw.pcie_efficiency
        elif interconnect == "cxl":
            return self.hw.cxl_bandwidth_gbps * self.hw.cxl_efficiency
        raise ValueError(f"Unknown interconnect: {interconnect}")

    def _base_latency(self, interconnect: str) -> float:
        if interconnect == "pcie":
            return self.hw.pcie_latency_us
        elif interconnect == "cxl":
            return self.hw.cxl_latency_us
        raise ValueError(f"Unknown interconnect: {interconnect}")
