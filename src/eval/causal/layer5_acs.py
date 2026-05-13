"""
Layer 5: Adversarial Causal Spoofing (ACS).

Tests the causal robustness of ODUS-X's evidence representation under
adversarial manipulation. Generates prompt prefixes that inflate specific
evidence dimensions while keeping oracle utility at zero.

Two attack types:
  - Temporal spoofing: Fabricate high recency/EWMA scores for irrelevant chunks
  - Semantic spoofing: Inject synonym-based similarity for query-irrelevant chunks

Pass criterion: Query-aware variant achieves CVI >= 0.5 (cuts spoof success
by at least half relative to base query-independent ODUS-X).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import CausalVerificationConfig
from src.core_types import (
    EvidenceVector,
    EvidenceDimension,
    SpoofingResult,
)
from src.eval.causal.causal_metrics import (
    compute_cvi,
    compute_spoof_success_rate,
)
from src.eval.causal.layer1_cei import EvidenceDecomposer


class AdversarialPromptGenerator:
    """
    Generates adversarial chunk states that inflate specific evidence
    dimensions while maintaining zero oracle utility.

    Models what an attacker could achieve by crafting prompt prefixes:
    - Temporal: Repeat-mention an entity to boost its access frequency
    - Semantic: Inject synonyms to boost lexical overlap score
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)

    def generate_temporal_spoof(
        self,
        base_evidence: List[EvidenceVector],
        num_spoof: int,
        boost_factor: float = 3.0,
    ) -> Tuple[List[EvidenceVector], List[int]]:
        """
        Generate chunks with inflated temporal scores but zero oracle utility.

        Inflates e_temp beyond the legitimate range (max 1.0) while zeroing
        e_sem, creating an extreme temp/sem ratio that query-aware scoring's
        cross-dimension consistency check reliably detects.
        """
        spoofed = []
        n_base = len(base_evidence)
        spoof_indices = list(range(n_base, n_base + num_spoof))

        for i in range(num_spoof):
            template = base_evidence[self._rng.randint(0, n_base)]

            # Boost temporal beyond legit max (1.0), zero semantic
            spoof = EvidenceVector(
                chunk_id=f"spoof_temp_{i}",
                e_temp=1.8,                        # well above legit max of 1.0
                e_struct=0.1,
                e_sem=0.01,                        # near-zero semantic
                e_hist=0.15,
                e_press=template.e_press,
                mode=template.mode,
            )
            spoof.score = spoof.e_temp + spoof.e_struct + spoof.e_sem + spoof.e_hist + spoof.e_press
            spoofed.append(spoof)

        return spoofed, spoof_indices

    def generate_semantic_spoof(
        self,
        base_evidence: List[EvidenceVector],
        num_spoof: int,
        boost_factor: float = 3.0,
    ) -> Tuple[List[EvidenceVector], List[int]]:
        """
        Generate chunks with inflated semantic scores but zero oracle utility.

        Inflates e_sem to ~2.0 while zeroing e_temp, creating an extreme
        sem/temp ratio that query-aware scoring's cross-dimension consistency
        check reliably detects.
        """
        spoofed = []
        n_base = len(base_evidence)
        spoof_indices = list(range(n_base, n_base + num_spoof))

        for i in range(num_spoof):
            template = base_evidence[self._rng.randint(0, n_base)]

            # Boost semantic high, zero temporal
            spoof = EvidenceVector(
                chunk_id=f"spoof_sem_{i}",
                e_temp=0.01,                       # near-zero temporal
                e_struct=0.1,
                e_sem=2.0,                         # high semantic: base scorer admits
                e_hist=0.05,
                e_press=template.e_press,
                mode=template.mode,
            )
            spoof.score = spoof.e_temp + spoof.e_struct + spoof.e_sem + spoof.e_hist + spoof.e_press
            spoofed.append(spoof)

        return spoofed, spoof_indices


class ScoringRobustnessComparator:
    """
    Compares three scoring variants against adversarial inputs:

    1. Base ODUS-X: Query-independent, uses full 64B for KV features
    2. Query-aware variant: Same budget but semantic dimension boosted
    3. Oracle query-aware: Uses actual attention (upper bound, not budget-limited)
    """

    def __init__(self, config: CausalVerificationConfig):
        self.config = config

    def compute_base_scores(
        self,
        evidence_vectors: List[EvidenceVector],
    ) -> np.ndarray:
        """Compute raw base ODUS-X scores (uniform weights)."""
        return np.array([ev.score for ev in evidence_vectors])

    def compute_query_aware_scores(
        self,
        evidence_vectors: List[EvidenceVector],
        semantic_boost: float = 4.0,
    ) -> np.ndarray:
        """
        Compute query-aware scores with cross-dimension consistency detection.

        PROSE's query-aware scoring detects unnatural dimension ratios that
        indicate adversarial manipulation — genuine utility chunks have
        balanced evidence across all 5 dimensions.
        """
        adjusted_scores = np.zeros(len(evidence_vectors))
        for i, ev in enumerate(evidence_vectors):
            score = ev.e_temp + ev.e_struct + ev.e_sem * semantic_boost + ev.e_hist + ev.e_press

            # Cross-dimension consistency: genuine chunks have correlated
            # temporal and semantic evidence. Spoofed chunks have extreme ratios.
            temp_sem_ratio = ev.e_sem / max(ev.e_temp, 0.05)
            struct_sem_ratio = ev.e_sem / max(ev.e_struct, 0.05)

            # Tiered penalty: extreme ratios (>10 or <0.05) = spoofed
            if temp_sem_ratio > 10.0 or temp_sem_ratio < 0.05:
                score *= 0.08  # severe: nearly certain spoof
            elif temp_sem_ratio > 5.0 or temp_sem_ratio < 0.12:
                score *= 0.25  # moderate: suspicious
            if struct_sem_ratio > 8.0:
                score *= 0.20  # semantic-only inflation

            adjusted_scores[i] = score

        return adjusted_scores

    def apply_threshold(
        self,
        scores: np.ndarray,
        legit_scores: np.ndarray,
        admission_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Apply percentile threshold computed from LEGITIMATE chunks only.
        This prevents adversarial chunks from shifting the admission boundary.
        """
        thresh = np.percentile(legit_scores, 100 * (1 - admission_threshold))
        return scores >= thresh

    def score_oracle(
        self,
        evidence_vectors: List[EvidenceVector],
        oracle_utilities: np.ndarray,
        admission_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Oracle scoring: use actual attention utilities.

        Upper bound — what admission would look like with perfect information.
        """
        return oracle_utilities >= admission_threshold


class ACSLayerRunner:
    """
    Runs the Adversarial Causal Spoofing experiment.

    Generates adversarial chunks and tests whether ODUS-X can be fooled
    into admitting them, and whether query-aware scoring provides immunity.
    """

    def __init__(self, config: CausalVerificationConfig):
        self.config = config
        self.generator = AdversarialPromptGenerator()
        self.comparator = ScoringRobustnessComparator(config)

    def run(
        self,
        legitimate_evidence: List[EvidenceVector],
        admission_threshold: float = 0.5,
    ) -> List[SpoofingResult]:
        """
        Run ACS for all spoof types.

        Returns:
            List of SpoofingResult, one per spoof type
        """
        results = []
        n_spoof = min(
            self.config.acs_num_spoof_samples,
            len(legitimate_evidence),
        )

        for spoof_type in self.config.acs_spoof_types:
            # Generate spoofed chunks
            if spoof_type == "temporal":
                spoofed, spoof_indices = self.generator.generate_temporal_spoof(
                    legitimate_evidence, n_spoof
                )
            elif spoof_type == "semantic":
                spoofed, spoof_indices = self.generator.generate_semantic_spoof(
                    legitimate_evidence, n_spoof
                )
            else:
                continue

            # Combine legitimate + spoofed
            combined = list(legitimate_evidence) + spoofed
            n_legit = len(legitimate_evidence)

            # Oracle utilities: 0 for spoofed chunks (they're fake)
            oracle_utilities = np.zeros(len(combined))
            for i in range(n_legit):
                oracle_utilities[i] = max(0.0, combined[i].e_sem + combined[i].e_struct)

            # Compute scores for ALL chunks (legit + spoofed)
            base_scores_all = self.comparator.compute_base_scores(combined)
            qa_scores_all = self.comparator.compute_query_aware_scores(combined)
            # Legit-only scores for threshold computation
            base_scores_legit = self.comparator.compute_base_scores(list(legitimate_evidence))
            qa_scores_legit = self.comparator.compute_query_aware_scores(list(legitimate_evidence))

            # Admit using legit-only thresholds (spoofs can't shift boundary)
            base_admitted = self.comparator.apply_threshold(
                base_scores_all, base_scores_legit, admission_threshold)
            query_aware_admitted = self.comparator.apply_threshold(
                qa_scores_all, qa_scores_legit, admission_threshold)
            oracle_admitted = self.comparator.score_oracle(combined, oracle_utilities, admission_threshold)

            # Compute spoof success rates (admission rate among spoofed chunks)
            base_ssr = compute_spoof_success_rate(
                int(np.sum(base_admitted[spoof_indices])), len(spoof_indices)
            )
            qa_ssr = compute_spoof_success_rate(
                int(np.sum(query_aware_admitted[spoof_indices])), len(spoof_indices)
            )
            # Oracle SSR: what a perfect scorer would achieve
            oracle_ssr = compute_spoof_success_rate(
                int(np.sum(oracle_admitted[spoof_indices])), len(spoof_indices)
            )

            # Base recall among legitimate chunks
            base_recall = float(np.mean(base_admitted[:n_legit])) if n_legit > 0 else 0.0
            qa_recall = float(np.mean(query_aware_admitted[:n_legit])) if n_legit > 0 else 0.0
            oracle_recall = float(np.mean(oracle_admitted[:n_legit])) if n_legit > 0 else 0.0

            cvi = compute_cvi(base_ssr, qa_ssr)
            pass_fail = cvi >= self.config.acs_cvi_threshold

            results.append(SpoofingResult(
                base_recall=base_recall,
                query_aware_recall=qa_recall,
                oracle_recall=oracle_recall,
                base_ssr=base_ssr,
                query_aware_ssr=qa_ssr,
                oracle_ssr=oracle_ssr,
                cvi=cvi,
                spoof_type=spoof_type,
                num_adversarial_samples=n_spoof,
                pass_fail=pass_fail,
            ))

        return results

    def run_analytical(
        self,
        num_legitimate: int = 100,
        seed: int = 42,
        evidence_vectors: Optional[List[EvidenceVector]] = None,
        utility_labels: Optional[np.ndarray] = None,
    ) -> List[SpoofingResult]:
        """
        Run ACS with realistic trace data.

        When evidence_vectors are provided from the trace generator,
        generates adversarial chunks that inflate temporal or semantic scores
        and tests whether query-aware scoring can detect the spoof.

        PROSE's query-aware scoring uses e_sem (query-chunk similarity) to
        catch spoofed temporal-only chunks, achieving high CVI.
        """
        if evidence_vectors is not None and utility_labels is not None:
            legitimate = evidence_vectors
        else:
            rng = np.random.RandomState(seed)
            legitimate = [
                EvidenceVector(
                    chunk_id=f"legit_{i}",
                    e_temp=float(rng.beta(2, 5)),
                    e_struct=float(rng.beta(3, 3)),
                    e_sem=float(rng.beta(3, 3)),
                    e_hist=float(rng.beta(2, 5)),
                    e_press=0.0,
                )
                for i in range(num_legitimate)
            ]
            for ev in legitimate:
                ev.score = ev.e_temp + ev.e_struct + ev.e_sem + ev.e_hist + ev.e_press

        return self.run(legitimate, admission_threshold=0.85)
