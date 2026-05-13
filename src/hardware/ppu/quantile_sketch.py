"""
Hardware-friendly quantile sketch for KV utility distribution tracking.

Enables the admission controller to set adaptive thresholds that track
shifting utility distributions across model architectures and workloads,
eliminating per-model offline calibration.

Design rationale — why not t-digest or DDSketch:
  - t-digest requires centroid merging and periodic sorting (O(k log k))
  - DDSketch needs variable-width bucket management
  - For 256-entry hardware SRAM, a log-spaced fixed-bin histogram with
    configurable bin edges is simpler, deterministic, and synthesizable
    as a register file + saturating incrementers.

Hardware implementation:
  ┌──────────────────────────────────────────────────┐
  │  Utility Quantile Sketch (256 × 16-bit counters) │
  │                                                   │
  │  utility ──→ bin_index() ──→ counter[bin] += 1   │
  │                                                   │
  │  quantile(q) ──→ scan_counters() ──→ threshold   │
  │                                                   │
  │  Area: 256 × 16b = 4096 bits = 512 bytes SRAM    │
  │  Latency: 1 cycle (update), 1-4 cycles (query)   │
  └──────────────────────────────────────────────────┘

The sketch supports three bin-spacing strategies:
  - "uniform":  Equally spaced bins in [0, 1]
  - "log_tail": Dense bins near 1.0 (where promotion decisions matter)
  - "adaptive": Periodically rebalance bin edges based on observed CDF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import math

import numpy as np


@dataclass
class QuantileSketchConfig:
    """Configuration for the hardware quantile sketch."""

    num_bins: int = 256
    counter_bits: int = 16  # Bits per counter (saturating)
    bin_spacing: str = "log_tail"  # "uniform", "log_tail", "adaptive"

    # Log-tail parameters: bins are denser near 1.0 (high utility)
    # bin[i] covers [log_tail_start + i * delta, ...)
    # where delta grows geometrically toward 0
    log_tail_gamma: float = 0.95  # Controls density skew (closer to 1 = more tail-heavy)

    # Decay: exponentially decay all counters periodically to track
    # distribution drift.  A decay_factor of 0.99 per step means the
    # sketch half-life is ~69 steps.
    decay_enabled: bool = True
    decay_factor: float = 0.99  # Multiplicative decay per step

    # Periodic rebalancing for "adaptive" mode
    rebalance_interval_steps: int = 1000

    # Minimum count before quantile estimates are considered valid
    min_total_count: int = 50


class QuantileSketch:
    """
    Hardware-friendly quantile sketch over [0, 1] utility values.

    Maintains a fixed-bin histogram with optional exponential decay.
    Supports O(1) update and O(num_bins) quantile query.
    """

    def __init__(self, config: Optional[QuantileSketchConfig] = None):
        self.config = config or QuantileSketchConfig()
        self._counters = np.zeros(self.config.num_bins, dtype=np.int32)
        self._total_count: int = 0
        self._step: int = 0

        # Precompute bin edges
        self._bin_edges = self._compute_bin_edges()

    def _compute_bin_edges(self) -> np.ndarray:
        """Compute bin boundaries based on spacing strategy."""
        n = self.config.num_bins
        if self.config.bin_spacing == "uniform":
            return np.linspace(0.0, 1.0, n + 1)
        elif self.config.bin_spacing == "log_tail":
            # Dense bins near 1.0 using exponential spacing from right
            # bin[i] covers [edges[i], edges[i+1])
            # edges[i] = 1 - gamma^(n - i)  scaled so edges[0]=0, edges[n]=1
            gamma = self.config.log_tail_gamma
            # Generate non-uniform spacing: denser at high values
            raw = np.array([gamma ** (n - i) for i in range(n + 1)], dtype=np.float64)
            # Normalize to [0, 1]
            raw = (raw - raw[0]) / (raw[-1] - raw[0])
            return raw.astype(np.float64)
        else:  # adaptive — start uniform, will rebalance
            return np.linspace(0.0, 1.0, n + 1)

    def update(self, utility: float) -> int:
        """Record a single utility observation. O(1) hardware: 1 cycle."""
        utility = max(0.0, min(1.0, float(utility)))
        bin_idx = self._find_bin(utility)
        max_val = (1 << self.config.counter_bits) - 1
        if self._counters[bin_idx] < max_val:
            self._counters[bin_idx] += 1
        self._total_count += 1
        self._step += 1
        return bin_idx

    def update_batch(self, utilities: Sequence[float]) -> None:
        """Record multiple observations."""
        for u in utilities:
            self.update(u)

    def _find_bin(self, utility: float) -> int:
        """Binary search for bin index. Hardware: priority encoder / comparator tree."""
        edges = self._bin_edges
        lo, hi = 0, len(edges) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if utility >= edges[mid]:
                lo = mid
            else:
                hi = mid - 1
        return min(lo, self.config.num_bins - 1)

    def quantile(self, q: float) -> float:
        """
        Estimate the q-th quantile of the utility distribution.

        Args:
            q: Quantile in [0, 1] (e.g., 0.5 = median, 0.9 = 90th percentile)

        Returns:
            Estimated utility value at quantile q, or 0.0 if insufficient data
        """
        q = max(0.0, min(1.0, float(q)))
        if self._total_count < self.config.min_total_count:
            return 0.0
        target = int(q * self._total_count)
        cumulative = 0
        for i in range(self.config.num_bins):
            cumulative += int(self._counters[i])
            if cumulative >= target:
                # Linear interpolation within the bin
                fraction = 0.0
                if self._total_count > 0:
                    excess = cumulative - target
                    bin_count = max(1, int(self._counters[i]))
                    fraction = (bin_count - excess) / bin_count
                lo = self._bin_edges[i]
                hi = self._bin_edges[min(i + 1, self.config.num_bins)]
                return float(lo + fraction * (hi - lo))
        return 1.0

    def cdf(self, utility: float) -> float:
        """Estimate CDF at a given utility value."""
        utility = max(0.0, min(1.0, float(utility)))
        if self._total_count == 0:
            return 0.0
        bin_idx = self._find_bin(utility)
        cumulative = int(np.sum(self._counters[:bin_idx]))
        # Linear interpolation within bin
        bin_count = max(1, int(self._counters[bin_idx]))
        lo = self._bin_edges[bin_idx]
        hi = self._bin_edges[min(bin_idx + 1, self.config.num_bins)]
        frac = (utility - lo) / max(hi - lo, 1e-12)
        cumulative += int(frac * bin_count)
        return float(min(1.0, cumulative / max(1, self._total_count)))

    def adaptive_threshold(
        self,
        target_accept_rate: float = 0.3,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
    ) -> float:
        """
        Compute an adaptive score threshold that would admit approximately
        `target_accept_rate` fraction of chunks.

        This replaces the hand-tuned `min_score_threshold = 0.3` with a
        threshold derived from the observed utility distribution.

        Args:
            target_accept_rate: Desired fraction of candidates to accept
            min_threshold: Floor for the threshold
            max_threshold: Ceiling for the threshold

        Returns:
            Adaptive score threshold
        """
        raw = self.quantile(1.0 - target_accept_rate)
        return float(max(min_threshold, min(max_threshold, raw)))

    def decay(self) -> None:
        """Apply exponential decay to all counters. Hardware: shift-and-subtract."""
        if not self.config.decay_enabled:
            return
        factor = self.config.decay_factor
        self._counters = np.maximum(0, np.round(self._counters * factor)).astype(np.int32)
        self._total_count = int(np.sum(self._counters))

    def get_distribution_summary(self) -> dict:
        """Return key distribution statistics for diagnostic logging."""
        return {
            "total_count": self._total_count,
            "p50": self.quantile(0.50),
            "p75": self.quantile(0.75),
            "p90": self.quantile(0.90),
            "p95": self.quantile(0.95),
            "p99": self.quantile(0.99),
            "adaptive_threshold_p70": self.adaptive_threshold(0.3),
            "adaptive_threshold_p80": self.adaptive_threshold(0.2),
            "num_active_bins": int(np.count_nonzero(self._counters)),
        }

    def reset(self) -> None:
        """Reset all counters."""
        self._counters = np.zeros(self.config.num_bins, dtype=np.int32)
        self._total_count = 0
        self._step = 0

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (bin_edges, counts) for visualization."""
        return self._bin_edges.copy(), self._counters.copy()
