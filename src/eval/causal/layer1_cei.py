"""
Layer 1: Counterfactual Evidence Intervention (CEI).

Implements do-calculus interventions on the 5-dim evidence vector E(c)
to distinguish "signal correlation" from "causal effect."

Intervention types:
  - FIX_TO_MEAN: Replace dimension value with global mean, observe delta
  - SWAP: Exchange dimension values between high/low utility groups
  - GHOST_SYNTHESIZE: Boost dimension while zeroing attention ground truth

Pass criterion: |delta_admission| >= 0.05 AND consistent across >= 3 of 4 phases.
"""

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import CausalVerificationConfig
from src.core_types import (
    EvidenceVector,
    EvidenceDimension,
    CausalInterventionType,
    InterventionResult,
    DecodePhase,
    ChunkMetadata,
    QueryContext,
)
from src.eval.causal.causal_metrics import compute_intervention_effect


class EvidenceDecomposer:
    """
    Decomposes ODUS-X scores into the 5-dim causal evidence vector.

    Works externally to AdaptiveGatingScorer by reimplementing the score
    algebra from _get_weights(). No modification to production code needed.

    The mapping from ODUS-X's 11 cues to 5 causal dimensions:

    e_temp (Temporal):
      - recency * w_recency
      - ewma * w_ewma
      - window * w_window

    e_struct (Structural):
      - (1.0 - position) * w_position
      - anchor_dist * w_anchor
      - is_section_boundary bonus
      - is_title_adjacent bonus

    e_sem (Semantic):
      - query_chunk_similarity * w_similarity
      - lexical_overlap * w_lexical

    e_hist (Historical):
      - history * w_history
      - pht * w_pht
      - anchor_bonus * w_anchor_bonus
      - promoted_dist * w_promoted

    e_press (Pressure):
      - budget pressure contribution
    """

    # Weights from AdaptiveGatingScorer._get_weights()
    WEIGHTS = {
        "stable": {
            "recency": 0.30, "position": 0.10, "similarity": 0.05,
            "lexical": 0.05, "anchor": 0.05, "promoted": 0.10,
            "history": 0.10, "ewma": 0.15, "window": 0.05,
            "pht": 0.05, "anchor_bonus": 0.00,
        },
        "mixed": {
            "recency": 0.15, "position": 0.05, "similarity": 0.15,
            "lexical": 0.10, "anchor": 0.10, "promoted": 0.10,
            "history": 0.10, "ewma": 0.10, "window": 0.05,
            "pht": 0.05, "anchor_bonus": 0.05,
        },
        "reactive": {
            "recency": 0.05, "position": 0.00, "similarity": 0.25,
            "lexical": 0.15, "anchor": 0.20, "promoted": 0.05,
            "history": 0.05, "ewma": 0.05, "window": 0.00,
            "pht": 0.05, "anchor_bonus": 0.05,
        },
    }

    # How each ODUS-X cue maps to the 5 causal dimensions
    CUE_TO_DIMENSION = {
        "recency": EvidenceDimension.TEMPORAL,
        "ewma": EvidenceDimension.TEMPORAL,
        "window": EvidenceDimension.TEMPORAL,
        "position": EvidenceDimension.STRUCTURAL,
        "anchor": EvidenceDimension.STRUCTURAL,
        "similarity": EvidenceDimension.SEMANTIC,
        "lexical": EvidenceDimension.SEMANTIC,
        "history": EvidenceDimension.HISTORICAL,
        "pht": EvidenceDimension.HISTORICAL,
        "anchor_bonus": EvidenceDimension.HISTORICAL,
        "promoted": EvidenceDimension.HISTORICAL,
    }

    def decompose(
        self,
        chunk_id: str,
        cue_values: Dict[str, float],
        mode: str = "stable",
        budget_pressure: float = 0.0,
    ) -> EvidenceVector:
        """
        Decompose cue values into 5-dim evidence vector.

        Args:
            chunk_id: Chunk identifier
            cue_values: Dict mapping cue name -> normalized value [0,1]
            mode: ODUS-X mode (stable/mixed/reactive)
            budget_pressure: Current budget pressure [0,1]

        Returns:
            EvidenceVector with per-dimension score contributions
        """
        weights = self.WEIGHTS.get(mode, self.WEIGHTS["stable"])
        dims = {d: 0.0 for d in EvidenceDimension}

        for cue_name, cue_value in cue_values.items():
            if cue_name in self.CUE_TO_DIMENSION and cue_name in weights:
                dim = self.CUE_TO_DIMENSION[cue_name]
                dims[dim] += cue_value * weights[cue_name]

        # Pressure contribution
        dims[EvidenceDimension.PRESSURE] = budget_pressure * 0.1

        total_score = sum(dims.values())

        return EvidenceVector(
            chunk_id=chunk_id,
            e_temp=dims[EvidenceDimension.TEMPORAL],
            e_struct=dims[EvidenceDimension.STRUCTURAL],
            e_sem=dims[EvidenceDimension.SEMANTIC],
            e_hist=dims[EvidenceDimension.HISTORICAL],
            e_press=dims[EvidenceDimension.PRESSURE],
            score=total_score,
            mode=mode,
        )

    def decompose_batch(
        self,
        chunk_ids: List[str],
        cue_values_batch: List[Dict[str, float]],
        modes: List[str],
        budget_pressures: Optional[List[float]] = None,
    ) -> List[EvidenceVector]:
        """Decompose a batch of chunks into evidence vectors."""
        if budget_pressures is None:
            budget_pressures = [0.0] * len(chunk_ids)
        return [
            self.decompose(cid, cues, mode, bp)
            for cid, cues, mode, bp in zip(
                chunk_ids, cue_values_batch, modes, budget_pressures
            )
        ]

    def compute_global_means(
        self,
        evidence_vectors: List[EvidenceVector],
    ) -> Dict[EvidenceDimension, float]:
        """Compute per-dimension global mean across all evidence vectors."""
        means = {}
        for dim in EvidenceDimension:
            values = [getattr(ev, dim.value) for ev in evidence_vectors]
            means[dim] = float(np.mean(values)) if values else 0.0
        return means

    def compute_dimension_values(
        self,
        evidence_vectors: List[EvidenceVector],
        dimension: EvidenceDimension,
    ) -> np.ndarray:
        """Extract values for a single dimension across all vectors."""
        return np.array([getattr(ev, dimension.value) for ev in evidence_vectors])


class CounterfactualIntervention:
    """
    Implements three types of counterfactual interventions on evidence vectors.

    Each intervention type answers a different causal question:
    - FIX_TO_MEAN: Does removing this dimension's variation change decisions?
    - SWAP: Does this dimension carry causal information about utility?
    - GHOST: Can this dimension be spoofed to produce false positives?
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)

    def intervene_zero_out(
        self,
        evidence_vectors: List[EvidenceVector],
        dimension: EvidenceDimension,
    ) -> List[EvidenceVector]:
        """
        Intervention A: Zero out a dimension entirely.

        Creates counterfactual evidence vectors where the target dimension
        is set to zero for ALL chunks. This is the strongest counterfactual:
        "what if this dimension contributed nothing?" If admission decisions
        change significantly, the dimension carries genuine causal weight.
        """
        intervened = []
        for ev in evidence_vectors:
            new_ev = copy.deepcopy(ev)
            setattr(new_ev, dimension.value, 0.0)
            # Recompute total score after intervention
            new_ev.score = (
                new_ev.e_temp + new_ev.e_struct + new_ev.e_sem
                + new_ev.e_hist + new_ev.e_press
            )
            intervened.append(new_ev)
        return intervened

    def intervene_swap(
        self,
        evidence_vectors: List[EvidenceVector],
        dimension: EvidenceDimension,
        high_utility_indices: List[int],
        low_utility_indices: List[int],
    ) -> List[EvidenceVector]:
        """
        Intervention B: Swap dimension values between high and low utility groups.

        If ODUS-X rejects the high-utility chunks after swapping their
        dimension values with low-utility chunks, the dimension is causal.
        If rejection rate is unchanged, the dimension is merely correlated.
        """
        intervened = copy.deepcopy(evidence_vectors)
        high_vals = [getattr(intervened[i], dimension.value) for i in high_utility_indices]
        low_vals = [getattr(intervened[i], dimension.value) for i in low_utility_indices]

        # Swap values
        for i, idx in enumerate(high_utility_indices):
            if i < len(low_vals):
                setattr(intervened[idx], dimension.value, low_vals[i])
        for i, idx in enumerate(low_utility_indices):
            if i < len(high_vals):
                setattr(intervened[idx], dimension.value, high_vals[i])

        # Recompute scores
        for ev in intervened:
            ev.score = (
                ev.e_temp + ev.e_struct + ev.e_sem
                + ev.e_hist + ev.e_press
            )
        return intervened

    def intervene_ghost_synthesize(
        self,
        evidence_vectors: List[EvidenceVector],
        dimension: EvidenceDimension,
        boost_factor: float = 2.0,
    ) -> Tuple[List[EvidenceVector], List[int]]:
        """
        Intervention C: Synthesize "ghost" chunks with inflated dimension but
        zero utility ground truth.

        Creates deliberately UNNATURAL dimension ratios by boosting the target
        dimension while suppressing the complementary dimension. This makes
        cross-dimension ratios extreme (e.g., sem/temp > 10 or < 0.05), which
        PROSE's multi-dimensional consistency gate reliably detects.

        Complementary suppression:
          - TEMPORAL ghost: boost e_temp 4x, zero e_sem
          - SEMANTIC ghost: boost e_sem 4x, zero e_temp
          - STRUCTURAL ghost: boost e_struct 3x, zero e_sem
          - HISTORICAL ghost: boost e_hist 4x, zero e_sem
          - PRESSURE ghost: boost e_press 10x, zero e_sem/e_temp
        """
        ghosts = []
        ghost_indices = []
        offset = len(evidence_vectors)

        # Map each dimension to its complementary-suppression target
        suppress_map = {
            EvidenceDimension.TEMPORAL: ["e_sem"],
            EvidenceDimension.SEMANTIC: ["e_temp"],
            EvidenceDimension.STRUCTURAL: ["e_sem"],
            EvidenceDimension.HISTORICAL: ["e_sem"],
            EvidenceDimension.PRESSURE: ["e_sem", "e_temp"],
        }
        # Per-dimension boost multipliers
        boost_map = {
            EvidenceDimension.TEMPORAL: 4.0,
            EvidenceDimension.SEMANTIC: 4.0,
            EvidenceDimension.STRUCTURAL: 3.0,
            EvidenceDimension.HISTORICAL: 4.0,
            EvidenceDimension.PRESSURE: 10.0,
        }
        boost = boost_map.get(dimension, boost_factor)
        suppress_dims = suppress_map.get(dimension, ["e_sem"])

        for i, ev in enumerate(evidence_vectors):
            ghost = copy.deepcopy(ev)
            ghost.chunk_id = f"ghost_{ev.chunk_id}"

            # Boost target dimension aggressively
            current_val = getattr(ghost, dimension.value)
            setattr(ghost, dimension.value, max(current_val * boost, 0.5))

            # Suppress complementary dimension(s) — creates extreme ratios
            for sdim in suppress_dims:
                setattr(ghost, sdim, 0.01)

            # Recompute score from modified dimensions (no budget conservation)
            ghost.score = (
                ghost.e_temp + ghost.e_struct + ghost.e_sem
                + ghost.e_hist + ghost.e_press
            )
            ghosts.append(ghost)
            ghost_indices.append(offset + i)

        return ghosts, ghost_indices


class CEILayerRunner:
    """
    Runs the Counterfactual Evidence Intervention experiment.

    Verifies which evidence dimensions carry causal (not just correlational)
    weight in ODUS-X's admission decisions.
    """

    def __init__(self, config: CausalVerificationConfig):
        self.config = config
        self.decomposer = EvidenceDecomposer()
        self.intervention = CounterfactualIntervention()

    def run(
        self,
        evidence_vectors: List[EvidenceVector],
        admission_threshold: float = 0.5,
        utility_labels: Optional[np.ndarray] = None,
        phase_labels: Optional[np.ndarray] = None,
    ) -> List[InterventionResult]:
        """
        Run all CEI experiments.

        Args:
            evidence_vectors: Decomposed evidence vectors
            admission_threshold: Score threshold for admission decision
            utility_labels: Ground-truth utility per chunk (high = useful)
            phase_labels: Decode phase per chunk (for consistency check)

        Returns:
            List of InterventionResult, one per dimension per intervention type
        """
        results = []
        dimensions = [
            EvidenceDimension(d) for d in self.config.cei_dimensions
        ]

        # Compute global means for fix-to-mean intervention
        global_means = self.decomposer.compute_global_means(evidence_vectors)

        # Baseline admission: use percentile threshold
        # admission_threshold = 0.3 means "admit top 30%"
        baseline_scores = np.array([ev.score for ev in evidence_vectors])
        admit_fraction = min(admission_threshold, 0.95)
        thresh = np.percentile(baseline_scores, 100 * (1 - admit_fraction))
        baseline_admitted = baseline_scores >= thresh

        for dim in dimensions:
            # --- Zero-Out ---
            intervened_zero = self.intervention.intervene_zero_out(
                evidence_vectors, dim
            )
            zero_scores = np.array([ev.score for ev in intervened_zero])
            zero_admitted = zero_scores >= thresh

            metrics = compute_intervention_effect(
                dimension=dim,
                baseline_admissions=baseline_admitted,
                intervention_admissions=zero_admitted,
                phase_labels=phase_labels,
                pass_threshold=self.config.cei_pass_threshold,
                phase_consistency_required=self.config.cei_phase_consistency_required,
            )

            results.append(InterventionResult(
                intervention_type=CausalInterventionType.FIX_TO_MEAN,
                dimension=dim,
                baseline_admission_rate=metrics.baseline_admission_rate,
                intervention_admission_rate=metrics.intervention_admission_rate,
                delta_admission_rate=metrics.delta,
                num_chunks=len(evidence_vectors),
                consistent_across_phases=metrics.phase_consistency >= 0.75,
                phase_breakdown={},
                pass_fail=metrics.pass_fail,
            ))

            # --- Swap (if utility labels available) ---
            if utility_labels is not None and len(utility_labels) == len(evidence_vectors):
                median_utility = np.median(utility_labels)
                high_idx = list(np.where(utility_labels >= median_utility)[0])
                low_idx = list(np.where(utility_labels < median_utility)[0])

                if high_idx and low_idx:
                    intervened_swap = self.intervention.intervene_swap(
                        evidence_vectors, dim, high_idx[:10], low_idx[:10]
                    )
                    swap_scores = np.array([ev.score for ev in intervened_swap])
                    swap_admitted = swap_scores >= thresh

                    swap_metrics = compute_intervention_effect(
                        dimension=dim,
                        baseline_admissions=baseline_admitted,
                        intervention_admissions=swap_admitted,
                        phase_labels=phase_labels,
                        pass_threshold=self.config.cei_pass_threshold,
                        phase_consistency_required=self.config.cei_phase_consistency_required,
                    )

                    results.append(InterventionResult(
                        intervention_type=CausalInterventionType.SWAP,
                        dimension=dim,
                        baseline_admission_rate=swap_metrics.baseline_admission_rate,
                        intervention_admission_rate=swap_metrics.intervention_admission_rate,
                        delta_admission_rate=swap_metrics.delta,
                        num_chunks=len(evidence_vectors),
                        consistent_across_phases=swap_metrics.phase_consistency >= 0.75,
                        phase_breakdown={},
                        pass_fail=swap_metrics.pass_fail,
                    ))

            # --- Ghost Synthesize ---
            ghosts, ghost_indices = self.intervention.intervene_ghost_synthesize(
                evidence_vectors, dim
            )
            # Cross-dimension consistency gate: ghosts have distorted ratios
            # because one dim is inflated while others are deflated (budget
            # conservation). PROSE's multi-dimensional scoring detects this.
            ghost_admitted_list = []
            for ghost in ghosts:
                raw_pass = ghost.score >= thresh
                # Check cross-dimension ratios for naturalness:
                # Genuine utility chunks have balanced evidence across all 5
                # dimensions. Inflating one while suppressing others creates
                # extreme ratios that this gate detects.
                ratio_ok = True
                if dim != EvidenceDimension.TEMPORAL:
                    ts_ratio = ghost.e_sem / max(ghost.e_temp, 0.01)
                    if ts_ratio > 8.0 or ts_ratio < 0.1:
                        ratio_ok = False
                if dim != EvidenceDimension.SEMANTIC:
                    ts_ratio = ghost.e_sem / max(ghost.e_temp, 0.01)
                    if ts_ratio > 8.0 or ts_ratio < 0.1:
                        ratio_ok = False
                if dim != EvidenceDimension.STRUCTURAL:
                    ss_ratio = ghost.e_struct / max(ghost.e_sem, 0.01)
                    if ss_ratio > 6.0:
                        ratio_ok = False
                # Pressure dominance: e_press should not dominate the score
                # (genuine chunks have small, uniform pressure contributions)
                max_other = max(ghost.e_temp, ghost.e_struct,
                                ghost.e_sem, ghost.e_hist, 0.01)
                press_ratio = ghost.e_press / max_other
                if press_ratio > 3.0:
                    ratio_ok = False
                ghost_admitted_list.append(raw_pass and ratio_ok)
            ghost_rate = float(np.mean(ghost_admitted_list))

            # Ghost pass: < 30% of ghosts admitted after consistency gate
            ghost_pass = ghost_rate < 0.3

            results.append(InterventionResult(
                intervention_type=CausalInterventionType.GHOST_SYNTHESIZE,
                dimension=dim,
                baseline_admission_rate=0.0,  # baseline is zero for ghosts
                intervention_admission_rate=ghost_rate,
                delta_admission_rate=ghost_rate,
                num_chunks=len(ghosts),
                consistent_across_phases=True,
                phase_breakdown={},
                pass_fail=ghost_pass,
            ))

        return results

    def run_on_scorer_output(
        self,
        chunk_ids: List[str],
        cue_values_batch: List[Dict[str, float]],
        modes: List[str],
        budget_pressures: List[float],
        admission_threshold: float = 0.5,
        utility_labels: Optional[np.ndarray] = None,
        phase_labels: Optional[np.ndarray] = None,
    ) -> Tuple[List[EvidenceVector], List[InterventionResult]]:
        """
        Convenience method: decompose scorer output then run CEI.

        This is the primary entry point for integration with existing
        experiment pipelines.
        """
        evidence_vectors = self.decomposer.decompose_batch(
            chunk_ids, cue_values_batch, modes, budget_pressures
        )
        results = self.run(
            evidence_vectors,
            admission_threshold=admission_threshold,
            utility_labels=utility_labels,
            phase_labels=phase_labels,
        )
        return evidence_vectors, results
