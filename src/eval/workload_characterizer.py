"""
Workload Characterizer — Quantify Real LLM Attention Access Patterns.

Answers the reviewer question:
  "You have not proven that needle-heavy access dominates real LLM inference."

We analyze real attention traces extracted from Qwen models and classify each
decode step into one of four categories:
  - sequential    : top-attention chunks follow a monotonic stride
  - localized     : high attention mass is concentrated in a small window
  - needle_heavy  : attention jumps abruptly to distant chunks (high entropy / low locality)
  - uniform       : attention is broadly distributed

Metrics exported:
  - Gini coefficient of chunk attention (locality)
  - Attention entropy (unpredictability)
  - Sequential-bias score (autocorrelation of top-K chunk indices)
  - Pattern-class distribution across steps
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AccessPatternClassification:
    dominant_pattern: str  # "sequential", "localized", "needle_heavy", "uniform"
    sequential_score: float
    localized_score: float
    needle_heavy_score: float
    uniformity_score: float
    gini: float
    entropy_bits: float
    top_k_drift: float


@dataclass
class WorkloadCharacterizationReport:
    total_steps: int
    pattern_distribution: Dict[str, float]
    mean_gini: float
    mean_entropy_bits: float
    mean_sequential_bias: float
    mean_top_k_drift: float
    per_step_classifications: List[AccessPatternClassification] = field(default_factory=list)


def _gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient: 0 = perfectly uniform, 1 = perfectly concentrated."""
    values = np.array(values, dtype=float).flatten()
    if values.sum() == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n + 1) * values))) / (n * cumsum[-1]) - (n + 1) / n


def _attention_entropy(attn: np.ndarray) -> float:
    """Shannon entropy in bits of the attention distribution."""
    p = np.array(attn, dtype=float)
    p = p[p > 0]
    if p.sum() == 0:
        return 0.0
    p = p / p.sum()
    return -np.sum(p * np.log2(p))


def _sequential_bias_score(top_k_history: List[List[int]]) -> float:
    """
    Measure how often the top-K chunk index follows a consistent stride.
    Returns a score in [0, 1].
    """
    if len(top_k_history) < 3:
        return 0.0

    strides = []
    for i in range(1, len(top_k_history)):
        # Use the median top-K index as the representative position
        prev = float(np.median(top_k_history[i - 1]))
        curr = float(np.median(top_k_history[i]))
        strides.append(curr - prev)

    if not strides:
        return 0.0

    # Fraction of consecutive strides that are equal (or 1-step monotonic)
    consistent = sum(1 for i in range(1, len(strides)) if abs(strides[i] - strides[i - 1]) <= 1)
    return consistent / max(1, len(strides) - 1)


class WorkloadCharacterizer:
    """Characterize attention-trace access patterns."""

    def __init__(self, top_k_ratio: float = 0.10):
        self.top_k_ratio = top_k_ratio

    def classify_step(
        self,
        chunk_attention: np.ndarray,
        prev_chunk_attention: Optional[np.ndarray] = None,
        top_k_history: Optional[List[List[int]]] = None,
    ) -> AccessPatternClassification:
        """Classify a single decode step."""
        n = len(chunk_attention)
        k = max(1, int(n * self.top_k_ratio))

        gini = _gini_coefficient(chunk_attention)
        entropy = _attention_entropy(chunk_attention)
        max_entropy = math.log2(max(n, 2))
        normalized_entropy = entropy / max_entropy  # [0, 1]

        # Top-K indices this step
        top_k = list(np.argsort(chunk_attention)[::-1][:k])

        # Localized: high Gini (>0.7) and low entropy (<0.5 max)
        localized_score = gini * (1.0 - normalized_entropy)

        # Uniform: low Gini (<0.3) and high entropy (>0.7 max)
        uniformity_score = (1.0 - gini) * normalized_entropy

        # Sequential bias
        seq_score = 0.0
        if top_k_history is not None and len(top_k_history) >= 2:
            seq_score = _sequential_bias_score(top_k_history + [top_k])

        # Needle-heavy: low sequential bias, high drift from previous step, moderate Gini
        drift = 0.0
        if prev_chunk_attention is not None and len(prev_chunk_attention) == n:
            # Jaccard distance between top-K sets
            prev_top_k = set(np.argsort(prev_chunk_attention)[::-1][:k])
            curr_top_k = set(top_k)
            union = len(prev_top_k | curr_top_k)
            inter = len(prev_top_k & curr_top_k)
            drift = 1.0 - (inter / union) if union > 0 else 0.0

        # Needle-heavy score: high drift + not localized + not uniform
        needle_score = drift * (1.0 - localized_score) * (1.0 - uniformity_score)

        # Adjust sequential score downward if drift is high
        seq_score *= (1.0 - drift)

        scores = {
            "sequential": seq_score,
            "localized": localized_score,
            "needle_heavy": needle_score,
            "uniform": uniformity_score,
        }
        dominant = max(scores, key=scores.get)

        return AccessPatternClassification(
            dominant_pattern=dominant,
            sequential_score=seq_score,
            localized_score=localized_score,
            needle_heavy_score=needle_score,
            uniformity_score=uniformity_score,
            gini=gini,
            entropy_bits=entropy,
            top_k_drift=drift,
        )

    def characterize_trace(
        self,
        chunk_attention_sequence: List[np.ndarray],
    ) -> WorkloadCharacterizationReport:
        """Characterize a full decode-time attention trace."""
        classifications: List[AccessPatternClassification] = []
        top_k_history: List[List[int]] = []

        for t, attn in enumerate(chunk_attention_sequence):
            prev_attn = chunk_attention_sequence[t - 1] if t > 0 else None
            n = len(attn)
            k = max(1, int(n * self.top_k_ratio))
            top_k = list(np.argsort(attn)[::-1][:k])

            cls = self.classify_step(attn, prev_attn, top_k_history if t > 0 else None)
            classifications.append(cls)
            top_k_history.append(top_k)

        total = len(classifications)
        dist = {
            "sequential": sum(1 for c in classifications if c.dominant_pattern == "sequential") / total,
            "localized": sum(1 for c in classifications if c.dominant_pattern == "localized") / total,
            "needle_heavy": sum(1 for c in classifications if c.dominant_pattern == "needle_heavy") / total,
            "uniform": sum(1 for c in classifications if c.dominant_pattern == "uniform") / total,
        }

        return WorkloadCharacterizationReport(
            total_steps=total,
            pattern_distribution=dist,
            mean_gini=float(np.mean([c.gini for c in classifications])),
            mean_entropy_bits=float(np.mean([c.entropy_bits for c in classifications])),
            mean_sequential_bias=float(np.mean([c.sequential_score for c in classifications])),
            mean_top_k_drift=float(np.mean([c.top_k_drift for c in classifications])),
            per_step_classifications=classifications,
        )

    def characterize_from_hpca_traces(
        self,
        traces: List[Dict],
    ) -> WorkloadCharacterizationReport:
        """
        Convenience wrapper for traces produced by
        `hpca_eval_orchestrator.AttentionTraceExtractor`.
        """
        all_attn = []
        for trace in traces:
            chunk_attn = trace.get("chunk_attention", None)
            if chunk_attn is not None:
                all_attn.append(np.array(chunk_attn))
        if not all_attn:
            # If only a single trace dict is provided, wrap it
            if "chunk_attention" in traces:
                all_attn.append(np.array(traces["chunk_attention"]))
        return self.characterize_trace(all_attn)
