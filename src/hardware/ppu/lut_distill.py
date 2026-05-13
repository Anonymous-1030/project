"""Distillation from ODUS-style runtime features to a compact LUT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.config import PPUConfig


@dataclass
class DistillationReport:
    num_samples: int
    num_lut_entries: int
    occupied_entries: int
    mse: float
    mae: float
    coverage: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_samples": self.num_samples,
            "num_lut_entries": self.num_lut_entries,
            "occupied_entries": self.occupied_entries,
            "mse": self.mse,
            "mae": self.mae,
            "coverage": self.coverage,
            "notes": self.notes,
        }


class LUTDistiller:
    """Collapse feature/utility pairs into a quantized LUT representation."""

    def __init__(self, config: PPUConfig):
        self.config = config
        self.num_entries = 1 << config.lut_index_bits
        self.max_output = (1 << config.lut_output_bits) - 1

    def distill(
        self,
        feature_matrix: Sequence[Sequence[float]],
        target_scores: Sequence[float],
    ) -> Tuple[np.ndarray, DistillationReport]:
        features = np.asarray(feature_matrix, dtype=np.float32)
        targets = np.asarray(target_scores, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] < 4:
            raise ValueError("feature_matrix must be [N, >=4] with the first 4 columns = recency/similarity/position/history")
        if len(features) != len(targets):
            raise ValueError("feature_matrix and target_scores must have the same number of samples")

        lut_acc = np.zeros(self.num_entries, dtype=np.float64)
        lut_cnt = np.zeros(self.num_entries, dtype=np.int32)
        idxs = np.array([self._pack_row(row[:4]) for row in features], dtype=np.int32)
        for idx, target in zip(idxs, targets):
            lut_acc[idx] += float(np.clip(target, 0.0, 1.0))
            lut_cnt[idx] += 1

        table = np.zeros(self.num_entries, dtype=np.uint8)
        global_mean = float(np.clip(targets.mean() if len(targets) else 0.0, 0.0, 1.0))
        fill = int(round(global_mean * self.max_output))
        table[:] = fill
        occupied = int(np.count_nonzero(lut_cnt))
        nonzero_idx = lut_cnt > 0
        table[nonzero_idx] = np.clip(np.round((lut_acc[nonzero_idx] / lut_cnt[nonzero_idx]) * self.max_output), 0, self.max_output).astype(np.uint8)

        preds = table[idxs].astype(np.float32) / float(self.max_output or 1)
        errors = preds - np.clip(targets, 0.0, 1.0)
        report = DistillationReport(
            num_samples=len(targets),
            num_lut_entries=self.num_entries,
            occupied_entries=occupied,
            mse=float(np.mean(np.square(errors))) if len(errors) else 0.0,
            mae=float(np.mean(np.abs(errors))) if len(errors) else 0.0,
            coverage=float(occupied / max(1, self.num_entries)),
            notes=[
                f"quantization={self.config.distill_quantization_method}",
                "fallback fill = global mean utility",
            ],
        )
        return table, report

    def distill_from_odus_weights(self, odus_weights_path: str, num_samples: Optional[int] = None) -> Tuple[np.ndarray, DistillationReport]:
        """Generate calibration data from a saved ODUS MLP checkpoint.

        The ODUS MLP takes a 10-dim feature vector where the first 4
        dimensions are [recency, similarity, position, history] — exactly
        the features the PPU hardware LUT is indexed by.  The remaining
        6 dimensions (lexical_overlap, anchor_distance, promoted_distance,
        promoted_count_norm, is_section_boundary, is_title_adjacent) are
        secondary signals.

        To produce a faithful LUT we must evaluate the MLP with
        *realistic* secondary features rather than random noise.  We
        sweep the 4 PPU features uniformly and set the 6 secondary
        features to their observed median values (conservative defaults).
        """
        try:
            import torch
        except Exception as exc:  # pragma: no cover - only when torch missing
            raise RuntimeError("torch is required for distill_from_odus_weights") from exc

        sample_count = int(num_samples or self.config.distill_num_calibration_samples)
        checkpoint = torch.load(odus_weights_path, map_location="cpu")
        state_dict = checkpoint.get("model_state", checkpoint)
        layer_indices = sorted({
            int(k.split(".")[1]) for k in state_dict.keys()
            if k.startswith("network.") and "weight" in k
        })
        weights: List[np.ndarray] = []
        biases: List[np.ndarray] = []
        for idx in layer_indices:
            w_key = f"network.{idx}.weight"
            b_key = f"network.{idx}.bias"
            if w_key in state_dict and b_key in state_dict:
                weights.append(state_dict[w_key].numpy().T)
                biases.append(state_dict[b_key].numpy())
        if not weights:
            raise ValueError("No ODUS layers found in checkpoint")

        input_dim = weights[0].shape[0]
        rng = np.random.default_rng(42)

        # Build calibration inputs: sweep the 4 PPU features uniformly,
        # fix secondary features to realistic median defaults.
        #
        # PPU feature order:  [recency, similarity, position, history]
        # ODUS feature order: [position, recency, similarity, lexical,
        #                      anchor_dist, promoted_dist, promo_count_norm,
        #                      success_rate, is_section_boundary, is_title_adjacent]
        #
        # PPU similarity is in [0, 1] (affine-mapped from cosine [-1, 1]).
        # ODUS similarity is raw cosine in [-1, 1].
        # We must reverse the affine map when feeding the MLP: odus_sim = ppu_sim * 2 - 1

        ppu_features = rng.random((sample_count, 4), dtype=np.float32)
        # ppu_features columns: [recency, similarity, position, history]

        # Map PPU features → ODUS feature ordering (first 10 dims).
        odus_inputs = np.zeros((sample_count, max(input_dim, 10)), dtype=np.float32)
        odus_inputs[:, 0] = ppu_features[:, 2]  # position → ODUS col 0
        odus_inputs[:, 1] = ppu_features[:, 0]  # recency  → ODUS col 1
        odus_inputs[:, 2] = ppu_features[:, 1] * 2.0 - 1.0  # similarity [0,1]→[-1,1] → ODUS col 2
        # Secondary features: conservative median defaults
        odus_inputs[:, 3] = 0.05   # lexical_overlap
        odus_inputs[:, 4] = 5.0    # distance_to_nearest_anchor (chunk units)
        odus_inputs[:, 5] = 5.0    # distance_to_promoted
        odus_inputs[:, 6] = 0.1    # promoted_count / 5.0
        odus_inputs[:, 7] = ppu_features[:, 3]  # history (success_rate) → ODUS col 7
        odus_inputs[:, 8] = 0.0    # is_section_boundary
        odus_inputs[:, 9] = 0.0    # is_title_adjacent

        # Truncate to match actual MLP input dimension.
        inputs = odus_inputs[:, :input_dim]

        # Forward pass through the ODUS MLP.
        x = inputs
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = x @ w + b
            if i < len(weights) - 1:
                x = np.maximum(0, x)
        scores = 1.0 / (1.0 + np.exp(-x[:, 0]))

        return self.distill(ppu_features, scores)

    def _pack_row(self, row: Sequence[float]) -> int:
        bits = max(1, self.config.lut_index_bits // 4)
        levels = (1 << bits) - 1
        vals = [int(round(float(np.clip(v, 0.0, 1.0)) * levels)) for v in row]
        return (
            (vals[0] << (3 * bits))
            | (vals[1] << (2 * bits))
            | (vals[2] << bits)
            | vals[3]
        ) & ((1 << self.config.lut_index_bits) - 1)
