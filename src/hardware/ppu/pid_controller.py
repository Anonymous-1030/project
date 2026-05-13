"""
PID controller for adaptive admission threshold regulation.

Replaces the static `min_score_threshold` with a closed-loop controller
that tracks queue pressure and Service Level Objective (SLO) metrics.

Motivation:
  The current ODUS-X / EABS pipeline uses a hand-tuned constant threshold
  (min_score_threshold = 0.3) that cannot adapt to:
    - Bursty workloads causing transient HBM saturation
    - Model architecture changes (different KV sizes, attention patterns)
    - Multi-tenant interference (competing promotion streams)

Control architecture:
  ┌──────────┐    ┌─────────────┐    ┌──────────────┐
  │  Queue   │───→│  PID Ctrl   │───→│  Threshold   │───→ admission
  │ Pressure │    │  Kp,Ki,Kd   │    │  clamp[0,1]  │    decision
  └──────────┘    └─────────────┘    └──────────────┘
       ↑                                   │
       └─────────── feedback ──────────────┘

Measured variable:  queue_pressure = dma_queue_depth / max_depth
Setpoint:           target_pressure (default 0.7 — keep queue ~70% full)
Control variable:   admission_threshold (raises to admit fewer, lowers to admit more)

Hardware implementation:
  Standard discrete PID with:
    - Anti-windup clamping on the integral term
    - Derivative term computed on measurement (not error) to avoid
      "derivative kick" on setpoint changes
    - Fixed-point arithmetic friendly (all multiplies are float, but
      could be Q15.16 fixed-point)
    - 1 cycle per update (3 multiplies + 4 adds)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PIDConfig:
    """Configuration for the discrete PID controller."""

    # PID gains
    kp: float = 0.5   # Proportional gain
    ki: float = 0.1   # Integral gain
    kd: float = 0.05  # Derivative gain

    # Setpoint: target queue pressure [0, 1]
    # 0.7 means we want the DMA queue 70% full on average
    target_pressure: float = 0.7

    # Anti-windup: clamp integral term to [-windup_limit, +windup_limit]
    integral_windup_limit: float = 1.0

    # Output limits
    output_min: float = 0.05  # Don't set threshold below 0.05
    output_max: float = 0.95  # Don't set threshold above 0.95

    # Derivative low-pass filter coefficient (0 = no filtering, 1 = no derivative)
    derivative_filter_alpha: float = 0.1

    # Use derivative-on-measurement (True) vs derivative-on-error (False)
    derivative_on_measurement: bool = True

    # Initial threshold value
    initial_threshold: float = 0.3

    # Reverse-acting: True when higher output → less inflow (admission control).
    # In our case, higher threshold → fewer chunks admitted → lower queue pressure.
    # So the error is measured - setpoint (raise threshold when queue is too full).
    reverse_acting: bool = True

    # SLO target: maximum acceptable miss rate
    # If miss_rate exceeds this, the setpoint is temporarily lowered
    slo_miss_rate_target: float = 0.05  # 5% miss rate
    slo_enabled: bool = True
    slo_setpoint_adjustment: float = 0.1  # How much to lower setpoint when SLO violated


class PIDController:
    """
    Discrete PID controller for adaptive admission threshold.

    Usage:
        pid = PIDController(PIDConfig())
        for each step:
            pressure = measure_queue_pressure()
            threshold = pid.step(pressure)
            # Use `threshold` as admission score cutoff
            pid.record_miss(is_miss=True)  # Optional SLO tracking
    """

    def __init__(self, config: Optional[PIDConfig] = None):
        self.config = config or PIDConfig()
        self.reset()

    def reset(self) -> None:
        """Reset all controller state."""
        self._integral: float = 0.0
        self._prev_measurement: float = self.config.target_pressure
        self._prev_error: float = 0.0
        self._filtered_derivative: float = 0.0
        self._output: float = self.config.initial_threshold
        self._step_count: int = 0

        # SLO tracking
        self._miss_window: list = []  # Rolling window of recent miss flags
        self._miss_window_size: int = 100

    def step(
        self,
        measured_pressure: float,
        dt: float = 1.0,
    ) -> float:
        """
        Compute one PID step.

        Args:
            measured_pressure: Current queue pressure [0, 1]
            dt: Time delta (default 1.0 for per-step updates)

        Returns:
            New admission threshold [output_min, output_max]
        """
        cfg = self.config
        self._step_count += 1

        # Clamp measurement
        measured = max(0.0, min(1.0, float(measured_pressure)))

        # --- Error computation ---
        setpoint = cfg.target_pressure

        # SLO-aware setpoint adjustment
        if cfg.slo_enabled and len(self._miss_window) >= 10:
            recent_miss_rate = sum(self._miss_window) / len(self._miss_window)
            if recent_miss_rate > cfg.slo_miss_rate_target:
                # More misses than SLO allows → lower setpoint to admit more
                setpoint -= cfg.slo_setpoint_adjustment
                setpoint = max(0.3, setpoint)  # Don't go below 0.3

        # Reverse-acting: higher output → less inflow.
        # When measured > setpoint (queue too full), raise threshold.
        # error = measured - setpoint (not setpoint - measured).
        if cfg.reverse_acting:
            error = measured - setpoint
        else:
            error = setpoint - measured

        # --- Proportional term ---
        p_term = cfg.kp * error

        # --- Integral term with anti-windup ---
        self._integral += cfg.ki * error * dt
        self._integral = max(-cfg.integral_windup_limit,
                             min(cfg.integral_windup_limit, self._integral))
        i_term = self._integral

        # --- Derivative term ---
        if cfg.derivative_on_measurement:
            # Derivative on measurement (avoids kick on setpoint changes)
            d_raw = -(measured - self._prev_measurement) / max(dt, 1e-6)
        else:
            d_raw = (error - self._prev_error) / max(dt, 1e-6)

        # Low-pass filter the derivative
        alpha = cfg.derivative_filter_alpha
        self._filtered_derivative = (
            (1.0 - alpha) * self._filtered_derivative + alpha * d_raw
        )
        d_term = cfg.kd * self._filtered_derivative

        # --- Combine and clamp ---
        raw_output = cfg.initial_threshold + p_term + i_term + d_term
        self._output = max(cfg.output_min, min(cfg.output_max, raw_output))

        # --- Update state ---
        self._prev_measurement = measured
        self._prev_error = error

        return self._output

    def record_outcome(self, was_miss: bool) -> None:
        """
        Record whether a promotion decision resulted in a miss
        (chunk was needed but not promoted).  Used for SLO tracking.
        """
        self._miss_window.append(1.0 if was_miss else 0.0)
        if len(self._miss_window) > self._miss_window_size:
            self._miss_window.pop(0)

    def record_batch_outcomes(self, misses: int, total: int) -> None:
        """Record batch outcomes in one call."""
        if total <= 0:
            return
        miss_rate = misses / total
        # Extend rolling window proportionally
        for _ in range(total):
            self._miss_window.append(1.0 if len(self._miss_window) < misses else 0.0)
        # Truncate
        if len(self._miss_window) > self._miss_window_size:
            self._miss_window = self._miss_window[-self._miss_window_size:]

    @property
    def current_threshold(self) -> float:
        return self._output

    @property
    def integral_term(self) -> float:
        return self._integral

    @property
    def derivative_term(self) -> float:
        return self._filtered_derivative

    def get_state(self) -> dict:
        """Return controller state for logging/debugging."""
        recent_miss_rate = (
            sum(self._miss_window) / max(1, len(self._miss_window))
            if self._miss_window else 0.0
        )
        return {
            "threshold": round(self._output, 4),
            "integral": round(self._integral, 4),
            "derivative": round(self._filtered_derivative, 4),
            "setpoint": round(self.config.target_pressure, 3),
            "step": self._step_count,
            "recent_miss_rate": round(recent_miss_rate, 4),
            "miss_window_size": len(self._miss_window),
        }
