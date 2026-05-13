"""Non-Stationary Promotion Regret Bound for HPCA.

Theorem 6 (Non-Stationary Promotion Regret):
  Model KV promotion as an online learning problem with switching costs.
  At each decode step t ∈ [T], the system selects a promotion set S_t ⊆ Tail
  of total size ≤ B bytes.  The per-step utility u_t(S) is revealed *after*
  the decision (adversarial / non-stationary attention).

  Define:
    - Dynamic regret:  R_T = Σ_t [u_t(S*_t) - u_t(S_t)]
    - Switching cost:  C_T = λ Σ_t |S_t △ S_{t-1}| · bytes / bandwidth
    - Path length:     P_T = Σ_t ‖u_t - u_{t-1}‖_∞  (total variation)
    - Attention var:   V_T = Σ_t Σ_c |a_t(c) - a_{t-1}(c)|  (L1 variation)

  Lower bound (any online policy):
      R_T + C_T ≥ Ω(√(P_T · T))

  Upper bound (Burst-and-Stick with TTL τ):
      R_T + C_T ≤ O(√(P_T · T) + T/τ + τ · V_T)

  Optimal TTL:
      τ* = (T / V_T)^{1/3}   →   R_T + C_T ≤ O(T^{2/3} · V_T^{1/3})

  Proof sketch:
    The lower bound follows from the Ω(√(P_T · T)) dynamic regret lower
    bound for online convex optimization (Zinkevich, 2003) extended with
    switching costs (Chen et al., NeurIPS 2020).  The upper bound decomposes
    into: (a) utility loss from stale promotions within a TTL window ≤ τ·V_T,
    (b) switching cost amortized over τ steps ≤ T/τ · c_switch, and
    (c) base dynamic regret ≤ O(√(P_T · T)) from the greedy scorer.

Theorem 7 (Switching Cost Amortization via Sticky TTL):
  The sticky TTL mechanism amortizes the per-switch transfer cost c_sw over
  τ consecutive steps.  Per-step amortized cost:
      c_amort = c_sw / τ = chunk_bytes / (bandwidth · τ)
  When τ ≥ BDP / chunk_bytes, the switching cost is fully hidden behind
  compute, yielding zero exposed switching overhead.

References:
  - Zinkevich, "Online Convex Programming", ICML 2003
  - Chen et al., "Online Learning with Switching Costs", NeurIPS 2020
  - Daniely et al., "Strongly Adaptive Online Learning", ICML 2015
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class RegretBoundResult:
    """Result of regret bound computation."""
    T: int                          # Number of decode steps
    path_length: float              # P_T
    total_variation: float          # V_T
    ttl: int                        # τ used
    optimal_ttl: float              # τ*
    lower_bound: float              # Ω(√(P_T · T))
    upper_bound: float              # O(√(P_T·T) + T/τ + τ·V_T)
    utility_loss_term: float        # τ · V_T
    switching_cost_term: float      # T / τ
    base_regret_term: float         # √(P_T · T)
    amortized_switch_cost: float    # per-step


@dataclass
class SwitchingCostAnalysis:
    """Detailed switching cost breakdown."""
    total_switches: int
    total_bytes_transferred: int
    total_switching_cost_us: float
    amortized_per_step_us: float
    hidden_by_compute: float       # Fraction hidden behind compute
    exposed_switching_us: float    # Switching cost not hidden


@dataclass
class EmpiricalRegretResult:
    """Empirical regret measured from actual policy execution."""
    T: int
    cumulative_oracle_utility: float
    cumulative_policy_utility: float
    cumulative_regret: float
    cumulative_switching_cost: float
    total_cost: float              # regret + λ·switching_cost
    per_step_regret: List[float] = field(default_factory=list)
    per_step_switching: List[float] = field(default_factory=list)


class NonStationaryRegretAnalyzer:
    """Analyzer for non-stationary promotion regret bounds.

    Computes theoretical bounds and empirical regret for the
    Burst-and-Stick promotion policy under non-stationary attention.
    """

    def __init__(
        self,
        switching_cost_lambda: float = 1.0,
        chunk_bytes: int = 65536,
        bandwidth_gbps: float = 64.0,
    ):
        self.lam = switching_cost_lambda
        self.chunk_bytes = chunk_bytes
        self.bandwidth_gbps = bandwidth_gbps
        # Per-chunk transfer cost in microseconds
        self._switch_cost_us = (
            chunk_bytes / (bandwidth_gbps * 1024**3)
        ) * 1e6

    # ------------------------------------------------------------------
    # Path length and variation from traces
    # ------------------------------------------------------------------
    def compute_path_length(
        self, utility_sequence: Sequence[Dict[int, float]],
    ) -> float:
        """Compute path length P_T = Σ_t ‖u_t - u_{t-1}‖_∞.

        Args:
            utility_sequence: List of {chunk_id: utility} dicts, one per step.
        """
        P = 0.0
        for t in range(1, len(utility_sequence)):
            all_chunks = set(utility_sequence[t].keys()) | set(utility_sequence[t - 1].keys())
            max_diff = 0.0
            for c in all_chunks:
                u_t = utility_sequence[t].get(c, 0.0)
                u_prev = utility_sequence[t - 1].get(c, 0.0)
                max_diff = max(max_diff, abs(u_t - u_prev))
            P += max_diff
        return P

    def compute_total_variation(
        self, attention_patterns: Sequence[Dict[int, float]],
    ) -> float:
        """Compute total variation V_T = Σ_t Σ_c |a_t(c) - a_{t-1}(c)|.

        Args:
            attention_patterns: List of {chunk_id: attention_mass} per step.
        """
        V = 0.0
        for t in range(1, len(attention_patterns)):
            all_chunks = (
                set(attention_patterns[t].keys()) |
                set(attention_patterns[t - 1].keys())
            )
            for c in all_chunks:
                a_t = attention_patterns[t].get(c, 0.0)
                a_prev = attention_patterns[t - 1].get(c, 0.0)
                V += abs(a_t - a_prev)
        return V

    # ------------------------------------------------------------------
    # Theoretical bounds
    # ------------------------------------------------------------------
    def compute_regret_bound(
        self, T: int, P_T: float, V_T: float, ttl: int,
    ) -> RegretBoundResult:
        """Compute theoretical regret bounds for given parameters.

        Args:
            T: Number of decode steps.
            P_T: Path length of utility sequence.
            V_T: Total variation of attention patterns.
            ttl: Sticky TTL value τ.
        """
        ttl = max(ttl, 1)
        base_regret = math.sqrt(max(P_T * T, 0.0))
        utility_loss = ttl * V_T
        switching_term = T / ttl * self._switch_cost_us * self.lam
        upper = base_regret + utility_loss + switching_term
        lower = 0.5 * base_regret  # Constant factor in Ω

        opt_ttl = self.optimal_ttl(T, V_T)
        amortized = self._switch_cost_us / ttl

        return RegretBoundResult(
            T=T, path_length=P_T, total_variation=V_T,
            ttl=ttl, optimal_ttl=opt_ttl,
            lower_bound=lower, upper_bound=upper,
            utility_loss_term=utility_loss,
            switching_cost_term=switching_term,
            base_regret_term=base_regret,
            amortized_switch_cost=amortized,
        )

    def optimal_ttl(self, T: int, V_T: float) -> float:
        """Compute optimal TTL τ* = (T / V_T)^{1/3}.

        When V_T = 0 (stationary), τ* → ∞ (never switch).
        """
        if V_T <= 0:
            return float(T)  # Never switch if stationary
        return (T / V_T) ** (1.0 / 3.0)

    def regret_vs_ttl_curve(
        self, T: int, P_T: float, V_T: float,
        ttl_range: Optional[List[int]] = None,
    ) -> List[Dict[str, float]]:
        """Sweep TTL values and compute regret bound for each.

        Returns list of {ttl, upper_bound, utility_loss, switching_cost}.
        """
        if ttl_range is None:
            ttl_range = [1, 2, 4, 8, 16, 32, 64]

        results = []
        for ttl in ttl_range:
            bound = self.compute_regret_bound(T, P_T, V_T, ttl)
            results.append({
                "ttl": ttl,
                "upper_bound": bound.upper_bound,
                "utility_loss": bound.utility_loss_term,
                "switching_cost": bound.switching_cost_term,
                "base_regret": bound.base_regret_term,
            })
        return results

    # ------------------------------------------------------------------
    # Empirical regret from traces
    # ------------------------------------------------------------------
    def compute_empirical_regret(
        self,
        oracle_utilities: Sequence[float],
        policy_utilities: Sequence[float],
        promotion_sets: Optional[Sequence[set]] = None,
    ) -> EmpiricalRegretResult:
        """Compute empirical regret from actual policy execution.

        Args:
            oracle_utilities: Per-step utility of oracle (best hindsight).
            policy_utilities: Per-step utility of the actual policy.
            promotion_sets: Per-step set of promoted chunk IDs (for switching cost).
        """
        T = len(oracle_utilities)
        per_step_regret = []
        per_step_switching = []
        cum_regret = 0.0
        cum_switching = 0.0

        for t in range(T):
            r = oracle_utilities[t] - policy_utilities[t]
            per_step_regret.append(r)
            cum_regret += r

            if promotion_sets and t > 0:
                sym_diff = len(promotion_sets[t] ^ promotion_sets[t - 1])
                sc = sym_diff * self._switch_cost_us * self.lam
            else:
                sc = 0.0
            per_step_switching.append(sc)
            cum_switching += sc

        return EmpiricalRegretResult(
            T=T,
            cumulative_oracle_utility=sum(oracle_utilities),
            cumulative_policy_utility=sum(policy_utilities),
            cumulative_regret=cum_regret,
            cumulative_switching_cost=cum_switching,
            total_cost=cum_regret + cum_switching,
            per_step_regret=per_step_regret,
            per_step_switching=per_step_switching,
        )

    # ------------------------------------------------------------------
    # Switching cost analysis
    # ------------------------------------------------------------------
    def switching_cost_analysis(
        self,
        promotion_sets: Sequence[set],
        compute_window_us: float = 50.0,
    ) -> SwitchingCostAnalysis:
        """Analyze switching costs from a promotion trace.

        Args:
            promotion_sets: Per-step set of promoted chunk IDs.
            compute_window_us: Compute time available to hide transfers.
        """
        total_switches = 0
        total_bytes = 0

        for t in range(1, len(promotion_sets)):
            sym_diff = len(promotion_sets[t] ^ promotion_sets[t - 1])
            total_switches += sym_diff
            total_bytes += sym_diff * self.chunk_bytes

        total_cost_us = total_switches * self._switch_cost_us
        T = len(promotion_sets)
        amortized = total_cost_us / max(T, 1)

        # How much is hidden behind compute?
        per_step_switches = total_switches / max(T - 1, 1)
        per_step_transfer_us = per_step_switches * self._switch_cost_us
        hidden_frac = min(1.0, compute_window_us / max(per_step_transfer_us, 1e-9))
        exposed = max(0.0, per_step_transfer_us - compute_window_us) * (T - 1)

        return SwitchingCostAnalysis(
            total_switches=total_switches,
            total_bytes_transferred=total_bytes,
            total_switching_cost_us=total_cost_us,
            amortized_per_step_us=amortized,
            hidden_by_compute=hidden_frac,
            exposed_switching_us=exposed,
        )

    # ------------------------------------------------------------------
    # Validation: check empirical vs theoretical
    # ------------------------------------------------------------------
    def validate_bound(
        self,
        attention_patterns: Sequence[Dict[int, float]],
        oracle_utilities: Sequence[float],
        policy_utilities: Sequence[float],
        promotion_sets: Sequence[set],
        ttl: int,
    ) -> Dict[str, float]:
        """Validate that empirical regret is within theoretical bound.

        Returns dict with theoretical bound, empirical cost, and ratio.
        """
        T = len(oracle_utilities)
        P_T = self.compute_path_length(
            [{c: v for c, v in p.items()} for p in attention_patterns]
        )
        V_T = self.compute_total_variation(attention_patterns)

        bound = self.compute_regret_bound(T, P_T, V_T, ttl)
        empirical = self.compute_empirical_regret(
            oracle_utilities, policy_utilities, promotion_sets
        )

        return {
            "T": T,
            "P_T": P_T,
            "V_T": V_T,
            "ttl": ttl,
            "optimal_ttl": bound.optimal_ttl,
            "theoretical_upper": bound.upper_bound,
            "empirical_total_cost": empirical.total_cost,
            "empirical_regret": empirical.cumulative_regret,
            "empirical_switching": empirical.cumulative_switching_cost,
            "bound_satisfied": empirical.total_cost <= bound.upper_bound * 1.1,
            "tightness_ratio": empirical.total_cost / max(bound.upper_bound, 1e-9),
        }
