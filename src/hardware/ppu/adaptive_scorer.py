"""
Hardware-Adaptive Admission Scorer (HAAS) — v1.0.

Replaces the static ODUS-X lookup tables and the one-time-distilled PPU LUT
with a hardware scoring accelerator that performs ONLINE LEARNING without
consuming SM cycles.

=== MOTIVATION ===

ODUS-X uses three fixed weight tables (stable/mixed/reactive) hand-tuned for
a single model architecture.  The PPU LUT is statically distilled once from
the ODUS MLP and never updated.  Both break under workload drift, new model
families, or different attention modes — recovery drops to 0.45.

HAAS replaces this with four hardware components:

  ┌──────────────────────────────────────────────────────────────┐
  │           Hardware-Adaptive Admission Scorer (HAAS)           │
  │                                                               │
  │  ┌──────────────────┐   ┌─────────────────────────────────┐  │
  │  │  Systolic Array   │   │  SGD Weight Adapter             │  │
  │  │  (1×11 MAC lane)  │   │  w_i += lr * reward * f_i       │  │
  │  │  score = Σ w_i·f_i│←──│  reward = commit (+1) / abort(-1)│ │
  │  └──────┬───────────┘   └─────────────────────────────────┘  │
  │         │                                                     │
  │         ▼                                                     │
  │  ┌──────────────────┐   ┌─────────────────────────────────┐  │
  │  │  PID Controller   │   │  Quantile Sketch                │  │
  │  │  adaptive θ(t)    │   │  Utility CDF → distribution-    │  │
  │  │  tracks queue     │   │  aware threshold initialization  │  │
  │  │  pressure + SLO   │   │                                  │  │
  │  └──────────────────┘   └─────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────────┘

=== SYSTOLIC ARRAY MICROARCHITECTURE ===

A 1×N MAC (multiply-accumulate) lane that computes:
    score = σ(Σ w_i · f_i)   where σ is sigmoid (optional)

Hardware model:
  - N = 11 features (matching ODUS-X cue dimensions)
  - Weights stored in a 11-entry × 16-bit register file
  - Features streamed from the FeatureExtractor
  - 1 MAC per cycle → 11 cycles for full dot-product (serial)
  - OR: 11 parallel MACs → 1 cycle (SIMD, 11× area)
  - Configurable: SIMD width = [1, 11] (trades area vs latency)

The systolic array is essentially a dot-product engine (DPE). We use
the term "systolic" loosely — at this scale (11 weights), the dataflow
is closer to a SIMD reduction tree.  For authenticity, we model:
  - Weight-stationary: weights pre-loaded, features broadcast
  - Output-stationary: partial sum accumulates at the output register

=== SGD ONLINE WEIGHT ADAPTATION ===

Uses the verification result (promote + access = success; promote + no-access
= waste) as an IMPLICIT REWARD SIGNAL — no labels needed.

Perceptron-style update rule:
    w_i(t+1) = w_i(t) + η · reward · f_i(t)

where:
  - reward ∈ {+1, -1}:  +1 if promoted chunk was accessed (commit success)
                         -1 if promoted chunk was NOT accessed (budget wasted)
  - η: learning rate (hardware-friendly: power-of-2 shift)
  - f_i: feature value for cue i at decision time

Hardware: each weight update is 1 multiply-add per weight per chunk.
With 11 weights and 5 chunks/step, that's 55 MACs per step — negligible
even at 1 MAC/cycle (55 cycles vs millions of GPU cycles/step).

Key design property: the SGD loop runs ENTIRELY in the PPU hardware and
NEVER touches the GPU SMs.  This is the core claim — online learning
without consuming compute cycles.

=== PID + QUANTILE THRESHOLD MANAGEMENT ===

The admission threshold θ(t) adapts via two mechanisms:
  1. PID controller: tracks DMA queue pressure, raises θ when queue is
     too full, lowers θ when queue drains (anti-windup included)
  2. Quantile sketch: periodically seeds θ using the observed utility
     CDF, so that approximately `target_accept_rate` fraction of
     candidates pass the threshold

These two mechanisms COMPOSE: the quantile sketch provides a distribution-
aware baseline, and the PID adds closed-loop regulation around it.

=== AREA AND TIMING ESTIMATES (TSMC N7) ===

  Component              | Entries | Bits  | Area (μm²) | Latency
  -----------------------|---------|-------|------------|--------
  Weight Register File   | 11      | 16    | ~200       | 1 cycle
  SIMD MAC lane (11×)    | 11      | —     | ~1500      | 1 cycle
  SGD update logic       | —       | —     | ~800       | 11 cycles
  PID controller         | —       | —     | ~300       | 1 cycle
  Quantile sketch SRAM   | 256     | 16    | ~3500      | 1 cycle
  ───────────────────────|─────────|───────|────────────|────────
  TOTAL (HAAS)           | —       | —     | ~6300 μm²  | ~12 cycles

  At 1.5 GHz: ~8 ns inference, ~80 ns SGD update.  Negligible vs
  ~50 μs CXL promotion latency and ~10 ms decode step time.

Reference: PPU LUT alone = ~1800 μm² (256×8b SRAM).  HAAS adds ~4500 μm²
for online learning — a 3.5× area increase but eliminates offline
calibration entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import math

import numpy as np

from src.hardware.ppu.quantile_sketch import (
    QuantileSketch,
    QuantileSketchConfig,
)
from src.hardware.ppu.pid_controller import (
    PIDController,
    PIDConfig,
)


# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────


@dataclass
class HAASConfig:
    """Configuration for the Hardware-Adaptive Admission Scorer."""

    # --- Feature dimensions ---
    num_features: int = 11  # Must match ODUS-X cue count

    # --- Systolic Array / SIMD DPE ---
    simd_width: int = 11  # Number of parallel MACs (1=serial, 11=full parallel)
    weight_bits: int = 16  # Weight precision (Q8.8 fixed-point)
    feature_bits: int = 8  # Feature precision (Q0.8)
    score_bits: int = 16  # Output score precision (Q8.8)
    use_sigmoid: bool = False  # Apply sigmoid to dot-product output

    # --- Initial Weights ---
    # Default: initialized from ODUS-X "mixed" mode for warm start
    initial_weights: Optional[List[float]] = None
    # If None, use these defaults (ODUS-X mixed mode, normalized):
    # [recency, position, similarity, lexical, anchor,
    #  promoted, history, ewma, window, pht, anchor_bonus]

    # --- SGD Online Learning ---
    sgd_enabled: bool = True
    sgd_learning_rate: float = 0.01  # η — keep small for stability
    sgd_learning_rate_decay: float = 0.999  # Per-step multiplicative decay
    sgd_min_lr: float = 0.001  # Floor for learning rate
    sgd_weight_decay: float = 0.0  # L2 regularization (0 = disabled)
    sgd_batch_size: int = 1  # Number of samples before weight update

    # Reward model
    reward_promote_hit: float = 1.0   # Chunk promoted AND accessed
    reward_promote_miss: float = -1.0  # Chunk promoted but NOT accessed
    reward_skip_hit: float = -0.5     # Chunk NOT promoted but WOULD have been used
    reward_skip_miss: float = 0.1     # Chunk NOT promoted and NOT used (correct skip)

    # --- PID Threshold Controller ---
    pid_config: Optional[PIDConfig] = None  # If None, uses defaults

    # --- Quantile Sketch ---
    sketch_config: Optional[QuantileSketchConfig] = None
    sketch_target_accept_rate: float = 0.3  # Admit ~30% of candidates

    # --- Threshold Composition ---
    # How to combine PID output and quantile-based threshold
    # "pid_only" / "quantile_only" / "blend" / "min" / "max"
    threshold_mode: str = "blend"
    blend_alpha: float = 0.7  # Weight on PID (1-alpha on quantile)

    # --- Operation ---
    # Feature names matching ODUS-X cue ordering
    feature_names: List[str] = field(default_factory=lambda: [
        "recency", "position", "similarity", "lexical", "anchor",
        "promoted", "history", "ewma", "window", "pht", "anchor_bonus",
    ])

    # --- Logging ---
    log_weight_updates: bool = True
    log_threshold_adaptation: bool = True
    log_reward_statistics: bool = True


# ────────────────────────────────────────────────────────────────
# Systolic Dot-Product Engine (DPE)
# ────────────────────────────────────────────────────────────────


class SystolicDPE:
    """
    1×N weight-stationary dot-product engine.

    Computes: score = Σ(w_i · f_i) for i ∈ [0, N)

    Microarchitecture:
      - Weight register file: N entries × W bits
      - Feature broadcast bus: N features streamed per cycle
      - SIMD width: S ∈ [1, N] parallel multipliers
      - Accumulator: single register, log₂(N·2^W·2^F) bits wide

    With SIMD width = N (full parallel): 1 cycle
    With SIMD width = 1 (serial):        N cycles

    Hardware cost: S multipliers + S adders + 1 accumulator register
    """

    def __init__(self, config: HAASConfig):
        self.config = config
        self.num_features = config.num_features
        self.simd_width = min(config.simd_width, config.num_features)

        # Weight register file: stored as float in simulation,
        # but modeled as Q8.8 fixed-point in hardware
        self._weights: np.ndarray = self._init_weights()

        # Minimum weight delta for update (hardware precision floor)
        self._weight_epsilon = 1.0 / (1 << (config.weight_bits - 1))

    def _init_weights(self) -> np.ndarray:
        """Initialize weights from config or ODUS-X mixed-mode defaults."""
        if self.config.initial_weights is not None:
            w = np.array(self.config.initial_weights, dtype=np.float64)
            if len(w) != self.num_features:
                raise ValueError(
                    f"initial_weights has {len(w)} entries, expected {self.num_features}"
                )
            return w

        # ODUS-X "mixed" mode weights (from odus_x.py)
        default_weights = np.array([
            0.15,  # recency
            0.05,  # position
            0.15,  # similarity
            0.10,  # lexical
            0.10,  # anchor
            0.10,  # promoted
            0.10,  # history
            0.10,  # ewma
            0.05,  # window
            0.05,  # pht
            0.05,  # anchor_bonus
        ], dtype=np.float64)
        # Normalize to unit sum for stability
        return default_weights / default_weights.sum()

    def compute(self, features: np.ndarray) -> float:
        """
        Compute dot-product score for a single feature vector.

        Args:
            features: Array of shape (num_features,) in [0, 1]

        Returns:
            Score in [0, 1]
        """
        if len(features) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {len(features)}"
            )
        raw = float(np.dot(self._weights, features))
        if self.config.use_sigmoid:
            raw = 1.0 / (1.0 + math.exp(-raw))
        return max(0.0, min(1.0, raw))

    def compute_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Compute scores for a batch of feature vectors.

        Args:
            feature_matrix: Array of shape (batch_size, num_features)

        Returns:
            Scores of shape (batch_size,)
        """
        raw = feature_matrix @ self._weights
        if self.config.use_sigmoid:
            raw = 1.0 / (1.0 + np.exp(-raw))
        return np.clip(raw, 0.0, 1.0)

    def update_weights(self, gradients: np.ndarray) -> None:
        """Apply accumulated gradients to weights (single-cycle register write)."""
        self._weights += gradients
        # Clamp to reasonable range
        self._weights = np.clip(self._weights, -2.0, 2.0)
        # Apply weight decay
        if self.config.sgd_weight_decay > 0:
            self._weights *= (1.0 - self.config.sgd_weight_decay)

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    def get_weight_precision(self) -> float:
        """Return minimum representable weight delta."""
        return self._weight_epsilon

    def estimate_cycles(self, batch_size: int = 1) -> int:
        """
        Estimate computation cycles.

        With SIMD width S and N features:
          cycles = ceil(N / S) * batch_size
        """
        cycles_per_dot = math.ceil(self.num_features / self.simd_width)
        return cycles_per_dot * batch_size


# ────────────────────────────────────────────────────────────────
# SGD Online Weight Adapter
# ────────────────────────────────────────────────────────────────


@dataclass
class SGDState:
    """Per-step state for SGD adapter."""
    step: int = 0
    current_lr: float = 0.01
    total_reward: float = 0.0
    total_updates: int = 0
    cumulant_gradient: np.ndarray = field(default=None)  # shape (num_features,)
    batch_count: int = 0


class SGDWeightAdapter:
    """
    Online weight adaptation using SGD with implicit reward signal.

    Reward model (fully causal — uses only information available at runtime):
      - Chunk was promoted AND later accessed  → +1.0 (correct promotion)
      - Chunk was promoted but NOT accessed    → -1.0 (wasted bandwidth)
      - Chunk was NOT promoted but WAS accessed → -0.5 (missed opportunity)
      - Chunk was NOT promoted and NOT accessed → +0.1 (correct skip)

    The adapter ACCUMULATES gradients over `batch_size` samples, then
    applies the mean gradient — this is mini-batch SGD in hardware.
    """

    def __init__(self, config: HAASConfig):
        self.config = config
        self._lr = config.sgd_learning_rate
        self.state = SGDState(
            cumulant_gradient=np.zeros(config.num_features, dtype=np.float64),
        )

    def record_outcome(
        self,
        features: np.ndarray,
        reward: float,
        score: float,
    ) -> None:
        """
        Record a single outcome for gradient accumulation.

        The gradient is:
          ∇_w L = -reward · features   (for squared error)
          OR
          ∇_w L = -(reward - score) · features  (if using score residual)

        We use the simpler perceptron rule:
          w_i += lr * reward * f_i

        where reward ∈ {+1, -1} is the binary outcome signal.

        Args:
            features: Feature vector at decision time (num_features,)
            reward: Outcome signal
            score: Predicted score (unused in simple rule, kept for residual variant)
        """
        if not self.config.sgd_enabled:
            return

        # Perceptron-style gradient: move weights toward features that
        # preceded positive outcomes, away from those that preceded negative
        gradient = self._lr * reward * features
        self.state.cumulant_gradient += gradient
        self.state.batch_count += 1
        self.state.total_reward += reward

    def flush(self, dpe: SystolicDPE) -> int:
        """
        Apply accumulated gradient to the DPE weight register file.

        Returns:
            Number of updates applied (0 if batch was empty)
        """
        if self.state.batch_count == 0:
            return 0

        # Apply mean gradient
        mean_grad = self.state.cumulant_gradient / self.state.batch_count
        dpe.update_weights(mean_grad)

        self.state.total_updates += 1
        self.state.step += 1
        n = self.state.batch_count

        # Reset accumulator
        self.state.cumulant_gradient = np.zeros(self.config.num_features, dtype=np.float64)
        self.state.batch_count = 0

        # Learning rate decay
        self._lr = max(
            self.config.sgd_min_lr,
            self._lr * self.config.sgd_learning_rate_decay,
        )
        self.state.current_lr = self._lr

        return n

    def compute_reward(
        self,
        was_promoted: bool,
        was_accessed: bool,
    ) -> float:
        """Map (promoted, accessed) outcome to scalar reward."""
        cfg = self.config
        if was_promoted and was_accessed:
            return cfg.reward_promote_hit
        elif was_promoted and not was_accessed:
            return cfg.reward_promote_miss
        elif not was_promoted and was_accessed:
            return cfg.reward_skip_hit
        else:  # not promoted and not accessed
            return cfg.reward_skip_miss

    @property
    def learning_rate(self) -> float:
        return self._lr


# ────────────────────────────────────────────────────────────────
# Hardware-Adaptive Admission Scorer (top-level)
# ────────────────────────────────────────────────────────────────


@dataclass
class HAASResult:
    """Single-candidate HAAS inference result."""
    chunk_id: str
    score: float
    confidence: float
    features: Dict[str, float]
    weights: Dict[str, float]
    quantile_threshold: float
    pid_threshold: float
    effective_threshold: float
    admitted: bool
    latency_cycles: int


@dataclass
class HAASStepResult:
    """HAAS result for a full step."""
    candidates: List[HAASResult]
    num_admitted: int
    num_rejected: int
    effective_threshold: float
    pid_state: dict
    sketch_summary: dict
    weight_snapshot: Dict[str, float]
    learning_rate: float
    sgd_updates_applied: int
    total_latency_cycles: int


class HardwareAdaptiveAdmissionScorer:
    """
    Top-level hardware-adaptive scorer integrating DPE + SGD + PID + Sketch.

    Usage:
        haas = HardwareAdaptiveAdmissionScorer(HAASConfig())
        for each decode step:
            # 1. Compute features for candidates (from existing FeatureExtractor)
            feature_vectors = ...

            # 2. Score and admit
            result = haas.score_and_admit(
                chunk_ids, feature_matrix, queue_pressure
            )

            # 3. After step completes (next step's beginning), record outcomes
            haas.record_step_outcomes(
                promoted_ids, accessed_ids, feature_vectors_at_decision_time
            )
    """

    def __init__(self, config: Optional[HAASConfig] = None):
        self.config = config or HAASConfig()

        # Sub-components
        self.dpe = SystolicDPE(self.config)
        self.sgd = SGDWeightAdapter(self.config)
        self.pid = PIDController(self.config.pid_config)
        self.sketch = QuantileSketch(self.config.sketch_config)

        # Per-step state for outcome recording
        self._pending_decisions: Dict[str, Tuple[np.ndarray, float]] = {}
        # chunk_id → (features_at_decision_time, score_at_decision_time)

        # Counters
        self._total_steps: int = 0
        self._total_promoted: int = 0
        self._total_accessed: int = 0
        self._total_hits: int = 0  # promoted AND accessed

    # ── Main Inference Path ────────────────────────────────────

    def score_and_admit(
        self,
        chunk_ids: List[str],
        feature_matrix: np.ndarray,
        queue_pressure: float = 0.5,
    ) -> HAASStepResult:
        """
        Full scoring + admission for one decode step.

        Args:
            chunk_ids: Candidate chunk IDs (length B)
            feature_matrix: Features (B, num_features)
            queue_pressure: Current DMA queue fill ratio [0, 1]

        Returns:
            HAASStepResult
        """
        self._total_steps += 1
        B = len(chunk_ids)
        if B == 0:
            return HAASStepResult(
                candidates=[], num_admitted=0, num_rejected=0,
                effective_threshold=0.0, pid_state={}, sketch_summary={},
                weight_snapshot={}, learning_rate=self.sgd.learning_rate,
                sgd_updates_applied=0, total_latency_cycles=0,
            )

        # Step 1: Dot-product scoring (DPE, 1-11 cycles)
        dpe_cycles = self.dpe.estimate_cycles(B)
        if feature_matrix.ndim == 1:
            feature_matrix = feature_matrix.reshape(1, -1)
        scores = self.dpe.compute_batch(feature_matrix)

        # Step 2: Compute adaptive threshold
        pid_threshold = self.pid.step(queue_pressure)
        quantile_threshold = self.sketch.adaptive_threshold(
            target_accept_rate=self.config.sketch_target_accept_rate,
        )
        effective_threshold = self._compose_threshold(pid_threshold, quantile_threshold)

        # Step 3: Admit/reject
        candidates = []
        for i in range(B):
            score = float(scores[i])
            features = {
                self.config.feature_names[j]: float(feature_matrix[i, j])
                for j in range(min(self.config.num_features, feature_matrix.shape[1]))
            }
            admitted = score >= effective_threshold
            confidence = 0.5 + 0.5 * abs(score - 0.5) * 2.0

            result = HAASResult(
                chunk_id=chunk_ids[i],
                score=score,
                confidence=confidence,
                features=features,
                weights=dict(zip(
                    self.config.feature_names,
                    self.dpe.get_weights(),
                )),
                quantile_threshold=quantile_threshold,
                pid_threshold=pid_threshold,
                effective_threshold=effective_threshold,
                admitted=admitted,
                latency_cycles=dpe_cycles,
            )
            candidates.append(result)

            # Store for outcome recording
            if admitted:
                self._pending_decisions[chunk_ids[i]] = (
                    feature_matrix[i].copy(), score
                )

        # Step 4: Update quantile sketch with observed scores
        self.sketch.update_batch(scores)
        self.sketch.decay()

        num_admitted = sum(1 for c in candidates if c.admitted)

        return HAASStepResult(
            candidates=candidates,
            num_admitted=num_admitted,
            num_rejected=B - num_admitted,
            effective_threshold=effective_threshold,
            pid_state=self.pid.get_state(),
            sketch_summary=self.sketch.get_distribution_summary(),
            weight_snapshot=dict(zip(
                self.config.feature_names,
                [round(float(w), 4) for w in self.dpe.get_weights()],
            )),
            learning_rate=self.sgd.learning_rate,
            sgd_updates_applied=self.sgd.state.total_updates,
            total_latency_cycles=dpe_cycles + 2,  # +2 for threshold + admit
        )

    # ── Outcome Recording & Online Learning ─────────────────────

    def record_step_outcomes(
        self,
        promoted_ids: List[str],
        accessed_chunk_ids: set,
        feature_snapshots: Optional[Dict[str, np.ndarray]] = None,
    ) -> int:
        """
        Record which promoted chunks were actually accessed.

        Call this at the START of the next decode step, when the attention
        operation has revealed which chunks were actually used.

        Args:
            promoted_ids: Chunks promoted in the previous step
            accessed_chunk_ids: Chunks that received non-zero attention
            feature_snapshots: Optional pre-recorded feature vectors.
                               If None, uses self._pending_decisions.

        Returns:
            Number of SGD updates applied
        """
        update_count = 0

        # Record outcomes for promoted chunks
        for cid in promoted_ids:
            was_accessed = cid in accessed_chunk_ids
            reward = self.sgd.compute_reward(
                was_promoted=True, was_accessed=was_accessed,
            )

            # Get features from pending decisions or snapshots
            if feature_snapshots and cid in feature_snapshots:
                features = feature_snapshots[cid]
            elif cid in self._pending_decisions:
                features, score = self._pending_decisions[cid]
            else:
                continue  # Can't update without features

            self.sgd.record_outcome(features, reward, 0.0)
            update_count += 1

            # Track statistics
            self._total_promoted += 1
            if was_accessed:
                self._total_hits += 1

        # Record SLO miss for PID: chunks that were accessed but NOT promoted
        accessed_set = set(accessed_chunk_ids)
        promoted_set = set(promoted_ids)
        missed = accessed_set - promoted_set
        if missed:
            self.pid.record_batch_outcomes(
                misses=len(missed),
                total=len(accessed_set),
            )
        elif accessed_set:
            self.pid.record_batch_outcomes(misses=0, total=len(accessed_set))

        # Flush accumulated gradients to DPE
        n_flushed = self.sgd.flush(self.dpe)

        # Clear pending decisions
        self._pending_decisions.clear()

        self._total_accessed += len(accessed_chunk_ids)

        return n_flushed

    # ── Helpers ─────────────────────────────────────────────────

    def _compose_threshold(
        self,
        pid_threshold: float,
        quantile_threshold: float,
    ) -> float:
        """Combine PID and quantile thresholds per the configured strategy."""
        mode = self.config.threshold_mode
        alpha = self.config.blend_alpha

        if mode == "pid_only":
            return pid_threshold
        elif mode == "quantile_only":
            return quantile_threshold
        elif mode == "blend":
            return alpha * pid_threshold + (1.0 - alpha) * quantile_threshold
        elif mode == "min":
            return min(pid_threshold, quantile_threshold)
        elif mode == "max":
            return max(pid_threshold, quantile_threshold)
        else:
            return pid_threshold

    def get_weights(self) -> Dict[str, float]:
        """Return current weight vector as a named dict."""
        return dict(zip(self.config.feature_names, self.dpe.get_weights()))

    def get_stats(self) -> dict:
        """Return comprehensive scorer statistics."""
        hit_rate = self._total_hits / max(1, self._total_promoted)
        return {
            "total_steps": self._total_steps,
            "total_promoted": self._total_promoted,
            "total_hits": self._total_hits,
            "hit_rate": round(hit_rate, 4),
            "total_accessed": self._total_accessed,
            "learning_rate": round(self.sgd.learning_rate, 6),
            "sgd_updates": self.sgd.state.total_updates,
            **self.pid.get_state(),
            **self.sketch.get_distribution_summary(),
            "weights": {
                k: round(float(v), 4)
                for k, v in zip(self.config.feature_names, self.dpe.get_weights())
            },
        }

    def reset(self) -> None:
        """Reset all state (for new evaluation run)."""
        self.dpe = SystolicDPE(self.config)
        self.sgd = SGDWeightAdapter(self.config)
        self.pid = PIDController(self.config.pid_config)
        self.sketch = QuantileSketch(self.config.sketch_config)
        self._pending_decisions.clear()
        self._total_steps = 0
        self._total_promoted = 0
        self._total_accessed = 0
        self._total_hits = 0
