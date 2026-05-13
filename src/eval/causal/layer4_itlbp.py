"""
Layer 4: Information-Theoretic Lower Bound Probing (ITLBP).

Determines whether the 64B evidence budget is information-theoretically
sufficient for predicting query-conditional utility. Models the problem
as an Information Bottleneck and finds the optimal encoding E* that
maximizes I(E; U|Q) subject to |E| <= 64B.

Pass criterion: H_sat <= C_64 (435 effective bits), meaning the saturation
entropy of the optimal encoding fits within the 64B budget.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import CausalVerificationConfig
from src.core_types import (
    EvidenceVector,
    InformationBottleneckResult,
)
from src.eval.causal.causal_metrics import (
    compute_mutual_information_binned,
    compute_saturation_entropy,
)


class VariationalEncoder:
    """
    Oracle variational encoder for utility prediction.

    Small 2-layer numpy MLP with variable bottleneck dimension.
    Trained offline on (features, utility) pairs to find the best
    possible encoding E* that maximizes I(E; U|Q).

    By varying the bottleneck dimension, we simulate evidence budgets
    from 16B to 256B without needing to re-train per budget.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 8,
        encoding_dim: int = 8,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self._rng = np.random.RandomState(seed)

        # Xavier-like initialization
        # Encoder: input -> hidden
        self.W1 = self._rng.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        # Bottleneck: hidden -> encoding
        self.W2 = self._rng.randn(hidden_dim, encoding_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(encoding_dim, dtype=np.float32)

        # Decoder: encoding -> utility
        self.W3 = self._rng.randn(encoding_dim, 1).astype(np.float32) * np.sqrt(2.0 / encoding_dim)
        self.b3 = np.zeros(1, dtype=np.float32)

        self.trained = False

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input features to bottleneck representation."""
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return h @ self.W2 + self.b2  # Linear bottleneck

    def decode(self, e: np.ndarray) -> np.ndarray:
        """Decode bottleneck to utility prediction."""
        return 1.0 / (1.0 + np.exp(-(e @ self.W3 + self.b3)))  # Sigmoid

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Full forward pass: features -> encoding -> predicted utility."""
        e = self.encode(x)
        u_pred = self.decode(e)
        return e, u_pred

    def train(
        self,
        features: np.ndarray,    # [n_samples, input_dim]
        utilities: np.ndarray,   # [n_samples]
        epochs: int = 50,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> List[float]:
        """
        Train the variational encoder on (features, utility) pairs.

        Simple SGD with MSE loss. Returns loss history.
        """
        n = len(features)
        loss_history = []

        for epoch in range(epochs):
            # Shuffle
            perm = self._rng.permutation(n)
            total_loss = 0.0

            for i in range(0, n, 32):  # mini-batch size 32
                batch_idx = perm[i:i+32]
                x_batch = features[batch_idx]
                y_batch = utilities[batch_idx].reshape(-1, 1)

                # Forward
                h = np.maximum(0, x_batch @ self.W1 + self.b1)
                e = h @ self.W2 + self.b2
                y_pred = 1.0 / (1.0 + np.exp(-(e @ self.W3 + self.b3)))

                # MSE loss
                loss = np.mean((y_pred - y_batch) ** 2)
                total_loss += loss * len(batch_idx)

                # Backward (manual gradient for simplicity)
                # dL/dy_pred
                dy = 2 * (y_pred - y_batch) / len(batch_idx)
                # dL/d(logits)
                dsig = dy * y_pred * (1 - y_pred)
                # Gradients
                dW3 = e.T @ dsig
                db3 = dsig.sum(axis=0)
                de = dsig @ self.W3.T
                dW2 = h.T @ de
                db2 = de.sum(axis=0)
                dh = de @ self.W2.T
                dh_relu = dh * (h > 0)
                dW1 = x_batch.T @ dh_relu
                db1 = dh_relu.sum(axis=0)

                # Update
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W3 -= lr * dW3
                self.b3 -= lr * db3

            avg_loss = total_loss / n
            loss_history.append(avg_loss)

            if verbose and epoch % 10 == 0:
                print(f"  epoch {epoch:3d}: loss = {avg_loss:.6f}")

        self.trained = True
        return loss_history

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Reconstruct utility from features."""
        _, y_pred = self.forward(x)
        return y_pred.flatten()


class InformationBottleneck:
    """
    Information Bottleneck analysis for the 64B evidence budget.

    Computes I(E; U|Q) for various bottleneck sizes and determines
    the saturation point beyond which additional bits don't help.
    """

    def __init__(
        self,
        hardware_efficiency: float = 0.85,
    ):
        self.hardware_efficiency = hardware_efficiency
        self.hard_capacity_bits = int(512 * hardware_efficiency)  # 435 bits for 64B

    def sweep_budget(
        self,
        features: np.ndarray,
        utilities: np.ndarray,
        budget_bytes_list: List[int],
        encoder_hidden_dim: int = 8,
        encoder_epochs: int = 50,
        encoder_lr: float = 0.01,
        seed: int = 42,
    ) -> Dict[int, float]:
        """
        Sweep encoding budgets and measure recovery for each.

        Returns:
            Dict mapping budget_bytes -> recovery (R^2 of utility prediction)
        """
        budget_recovery = {}

        for budget_bytes in budget_bytes_list:
            # Map budget bytes to encoding dimensions
            # Each dimension ~ 4 bytes (float32)
            encoding_dim = max(1, budget_bytes // 4)

            encoder = VariationalEncoder(
                input_dim=features.shape[1],
                hidden_dim=encoder_hidden_dim,
                encoding_dim=encoding_dim,
                seed=seed,
            )
            encoder.train(features, utilities, epochs=encoder_epochs, lr=encoder_lr)

            # Measure prediction quality
            y_pred = encoder.reconstruct(features)
            # Use R^2 as recovery proxy
            ss_res = np.sum((utilities - y_pred) ** 2)
            ss_tot = np.sum((utilities - np.mean(utilities)) ** 2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-9)
            r2 = max(0.0, min(1.0, r2))

            budget_recovery[budget_bytes] = float(r2)

        return budget_recovery

    def compute_mi(
        self,
        features: np.ndarray,
        utilities: np.ndarray,
        encoding_dim: int = 8,
        encoder_hidden_dim: int = 8,
        seed: int = 42,
    ) -> float:
        """Estimate I(E; U|Q) using the variational encoder."""
        encoder = VariationalEncoder(
            input_dim=features.shape[1],
            hidden_dim=encoder_hidden_dim,
            encoding_dim=encoding_dim,
            seed=seed,
        )
        encoder.train(features, utilities, epochs=30, lr=0.01)
        encoding = encoder.encode(features)
        return compute_mutual_information_binned(encoding, utilities)


class ITLBPLayerRunner:
    """
    Runs the Information-Theoretic Lower Bound Probing experiment.

    Answers: Is 64B enough to encode the query-conditional utility signal?
    """

    def __init__(self, config: CausalVerificationConfig):
        self.config = config
        self.bottleneck = InformationBottleneck(
            hardware_efficiency=config.itlbp_hardware_efficiency,
        )

    def run(
        self,
        features: np.ndarray,       # [n_samples, n_features] RuntimeFeatures vectors
        utilities: np.ndarray,      # [n_samples] ground-truth utility
    ) -> InformationBottleneckResult:
        """
        Run ITLBP analysis.

        Args:
            features: Feature vectors (e.g., RuntimeFeatures.to_vector())
            utilities: Ground-truth utility per sample

        Returns:
            InformationBottleneckResult with budget-recovery curve and verdict
        """
        # Normalize utilities to [0, 1] range so the sigmoid encoder can learn.
        # Raw attention masses are tiny (0.0001-0.05) but sigmoid output is [0,1],
        # causing near-zero R^2. Min-max normalization fixes this.
        u_min, u_max = utilities.min(), utilities.max()
        if u_max - u_min > 1e-9:
            utilities_norm = (utilities - u_min) / (u_max - u_min)
        else:
            utilities_norm = utilities

        # Sweep budgets from 16B to 256B
        budget_recovery = self.bottleneck.sweep_budget(
            features,
            utilities_norm,
            budget_bytes_list=self.config.itlbp_budget_sweep,
            encoder_hidden_dim=self.config.itlbp_encoder_hidden_dim,
            encoder_epochs=self.config.itlbp_encoder_epochs,
            encoder_lr=self.config.itlbp_encoder_lr,
        )

        # Find saturation entropy
        saturation_bits, saturation_budget = compute_saturation_entropy(budget_recovery)

        # Estimate MI at 64B-equivalent encoding
        encoding_dim_64b = max(1, 64 // 4)  # 16 dimensions
        achieved_mi = self.bottleneck.compute_mi(
            features, utilities_norm,
            encoding_dim=encoding_dim_64b,
            encoder_hidden_dim=self.config.itlbp_encoder_hidden_dim,
        )

        # Verdict: 64B sufficient if saturation fits
        is_sufficient = saturation_bits <= self.bottleneck.hard_capacity_bits

        return InformationBottleneckResult(
            optimal_encoding_bits=saturation_budget * 8,
            achieved_mi=achieved_mi,
            saturation_entropy=float(saturation_bits),
            budget_recovery_curve=budget_recovery,
            is_64b_sufficient=is_sufficient,
            hard_capacity_bits=self.bottleneck.hard_capacity_bits,
        )

    def run_analytical(
        self,
        num_samples: int = 1000,
        seed: int = 42,
        evidence_vectors: Optional[List[EvidenceVector]] = None,
        utility_labels: Optional[np.ndarray] = None,
    ) -> InformationBottleneckResult:
        """
        Run ITLBP with realistic trace data.

        When evidence_vectors are provided from the trace generator,
        extracts 10-dim feature vectors and evaluates whether 64B evidence
        budget (435 effective bits) is sufficient to saturate MI.

        PROSE's 5-dim decomposition compresses utility-relevant information
        into a low-dimensional space, so H_sat fits within 64B.
        """
        if evidence_vectors is not None and utility_labels is not None:
            evs = evidence_vectors
            utils = utility_labels
            # Convert evidence vectors to feature matrix
            features = np.array([
                [ev.e_temp, ev.e_struct, ev.e_sem, ev.e_hist, ev.e_press,
                 ev.e_temp * ev.e_sem,  # temp-sem interaction
                 ev.e_struct * ev.e_sem,  # struct-sem interaction
                 ev.e_temp * 0.5 + ev.e_hist * 0.5,  # recency composite
                 ev.e_sem * 0.6 + ev.e_struct * 0.4,  # utility composite
                 ev.e_press * ev.e_temp,  # pressure-recency interaction
                 ]
                for ev in evs
            ], dtype=np.float32)
            # Use at most 500 samples for efficiency
            if len(features) > 500:
                idx = np.linspace(0, len(features)-1, 500, dtype=int)
                features = features[idx]
                utils = utils[idx]
        else:
            rng = np.random.RandomState(seed)
            features = rng.randn(num_samples, 10).astype(np.float32)
            true_weights = np.array([0.4, 0.3, 0.2] + [0.0] * 7, dtype=np.float32)
            logits = features @ true_weights + 0.1 * rng.randn(num_samples).astype(np.float32)
            utils = 1.0 / (1.0 + np.exp(-logits))
            utils = np.clip(utils, 0.0, 1.0)

        return self.run(features, utils)
