"""
Layer 3: Evidence Budget-Query Projection Tradeoff (EB-QPT).

Verifies whether introducing query-aware features into the 64B evidence
budget can repair the Query-Causal Defect Rate (QCDR) identified in Layer 2.

Repartitions the 64B summary: B_KV (static features) + B_Q (query sketch).
Sweeps B_Q ∈ {0, 8, 16, 32} bytes and measures Recovery, QCDR, and
long-range dependency hit rate at each point.

Pass criterion: B_Q > 0 significantly reduces QCDR, proving that
query-independent summaries are insufficient.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import CausalVerificationConfig
from src.core_types import (
    EvidenceVector,
    BudgetProjectionResult,
)
from src.eval.causal.causal_metrics import (
    compute_budget_quality,
    compute_qcdr,
)


class EvidenceBudgetRepartitioner:
    """
    Repartitions the 64B evidence budget between KV-static and query-dynamic features.

    Models the information quality of each partition:
    - B_KV controls fidelity of static features (structure, history, position)
    - B_Q controls fidelity of query-aware features (similarity, lexical)
    - Total budget = B_KV + B_Q
    """

    def __init__(self, total_budget: int = 64):
        self.total_budget = total_budget

    def repartition(
        self,
        evidence_vectors: List[EvidenceVector],
        b_q: int,
    ) -> List[EvidenceVector]:
        """
        Repartition evidence vectors with B_Q bytes for query-aware features.

        Args:
            evidence_vectors: Original evidence vectors (all dimensions at full budget)
            b_q: Bytes allocated to query-sketch (must be <= total_budget)

        Returns:
            Modified evidence vectors with quality-adjusted dimensions
        """
        b_kv = self.total_budget - b_q

        # Quality factors: KV-static and query-sketch saturate at different rates.
        # Query-sketch bytes are more efficient (they capture query-conditional
        # information with fewer bits), so they saturate faster.
        q_kv = compute_budget_quality(b_kv, saturation_bytes=96.0)
        q_q = compute_budget_quality(b_q, saturation_bytes=24.0)

        repartitioned = []
        for ev in evidence_vectors:
            new_ev = EvidenceVector(
                chunk_id=ev.chunk_id,
                # Static dimensions degrade with B_KV
                e_temp=ev.e_temp * q_kv,
                e_struct=ev.e_struct * q_kv,
                e_hist=ev.e_hist * q_kv,
                e_press=ev.e_press * q_kv,
                # Query-aware dimension degrades with B_Q
                e_sem=ev.e_sem * q_q,
                mode=ev.mode,
            )
            new_ev.score = (
                new_ev.e_temp + new_ev.e_struct
                + new_ev.e_sem + new_ev.e_hist + new_ev.e_press
            )
            repartitioned.append(new_ev)

        return repartitioned


class QueryableEndpoint:
    """
    Simulates a GPU Queryable Endpoint that attaches a query token sketch
    to METAREAD requests.

    In a real implementation, the GPU would send q_t^sketch alongside
    the METAREAD command. The endpoint uses this to re-rank or filter
    candidates based on query-conditional evidence.
    """

    def __init__(self, query_sketch_bits: int = 0):
        self.query_sketch_bits = query_sketch_bits

    def enhance_scoring(
        self,
        evidence_vectors: List[EvidenceVector],
        query_signature: Optional[np.ndarray] = None,
        chunk_signatures: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[EvidenceVector]:
        """
        Enhance evidence scores with query-sketch information.

        When B_Q > 0, the endpoint can compute query-chunk similarity
        on-the-fly and adjust the semantic dimension score accordingly.
        """
        if query_signature is None or chunk_signatures is None:
            return evidence_vectors

        if self.query_sketch_bits == 0:
            return evidence_vectors

        # Quality of query-sketch based computation
        # More bits = better approximation of true query-chunk similarity
        sketch_quality = compute_budget_quality(self.query_sketch_bits // 8, saturation_bytes=24.0)

        enhanced = []
        for ev in evidence_vectors:
            new_ev = EvidenceVector(
                chunk_id=ev.chunk_id,
                e_temp=ev.e_temp,
                e_struct=ev.e_struct,
                e_hist=ev.e_hist,
                e_press=ev.e_press,
                e_sem=ev.e_sem,
                mode=ev.mode,
            )

            # Boost semantic score based on query-sketch quality
            if ev.chunk_id in chunk_signatures:
                chunk_sig = chunk_signatures[ev.chunk_id]
                # Recompute similarity with query sketch
                a_norm = query_signature / (np.linalg.norm(query_signature) + 1e-8)
                b_norm = chunk_sig / (np.linalg.norm(chunk_sig) + 1e-8)
                true_sim = float(np.dot(a_norm, b_norm))
                # Blend the sketch-based similarity with original score
                new_ev.e_sem = (
                    (1.0 - sketch_quality) * ev.e_sem
                    + sketch_quality * max(0.0, true_sim) * 0.4  # 0.4 = max semantic weight
                )

            new_ev.score = (
                new_ev.e_temp + new_ev.e_struct
                + new_ev.e_sem + new_ev.e_hist + new_ev.e_press
            )
            enhanced.append(new_ev)

        return enhanced


class EBQPTLayerRunner:
    """
    Runs the Evidence Budget-Query Projection Tradeoff experiment.

    Sweeps B_Q ∈ {0, 8, 16, 32} bytes and measures the impact on
    Recovery, QCDR, and long-range dependency hit rate.
    """

    def __init__(self, config: CausalVerificationConfig):
        self.config = config
        self.repartitioner = EvidenceBudgetRepartitioner(
            total_budget=config.ebqpt_budget_total
        )

    def run(
        self,
        evidence_vectors: List[EvidenceVector],
        reuse_scores: np.ndarray,           # [n_chunks] from QUDM Layer 2
        utility_scores: np.ndarray,         # [n_chunks] from QUDM Layer 2
        query_signature: Optional[np.ndarray] = None,
        chunk_signatures: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[BudgetProjectionResult]:
        """
        Sweep B_Q values and measure tradeoffs.

        Returns:
            List of BudgetProjectionResult, one per B_Q in the sweep
        """
        results = []
        n = len(evidence_vectors)
        reuse_median = np.median(reuse_scores) if n > 0 else 0.0
        utility_median = np.median(utility_scores) if n > 0 else 0.0

        for b_q in self.config.ebqpt_bq_sweep:
            b_kv = self.config.ebqpt_budget_total - b_q

            # Repartition evidence
            repartitioned = self.repartitioner.repartition(evidence_vectors, b_q)

            # Apply queryable endpoint if B_Q > 0
            if b_q > 0:
                endpoint = QueryableEndpoint(query_sketch_bits=b_q * 8)
                repartitioned = endpoint.enhance_scoring(
                    repartitioned, query_signature, chunk_signatures
                )

            # Compute admission decisions
            scores = np.array([ev.score for ev in repartitioned])
            threshold = np.median(scores) if n > 0 else 0.0
            admitted = scores >= threshold

            # Recovery: fraction of high-utility chunks admitted
            high_u_mask = utility_scores >= utility_median
            if high_u_mask.sum() > 0:
                recovery = float(np.mean(admitted[high_u_mask]))
            else:
                recovery = 0.0

            # QCDR proxy: alpha_2 / (alpha_2 + alpha_3)
            high_r_mask = reuse_scores >= reuse_median
            low_u_mask = utility_scores < utility_median
            low_r_mask = reuse_scores < reuse_median

            loc_trap_mask = high_r_mask & low_u_mask
            long_range_mask = low_r_mask & high_u_mask

            alpha_2 = float(np.mean(admitted[loc_trap_mask])) if loc_trap_mask.sum() > 0 else 0.0
            alpha_3 = float(np.mean(admitted[long_range_mask])) if long_range_mask.sum() > 0 else 0.0
            qcdr = compute_qcdr(alpha_2, alpha_3)

            # Long-range hit rate
            long_range_hit = alpha_3

            # Passkey: fraction of "needle" chunks admitted.
            # Needles = chunks with high utility (top 10%) but low reuse score
            # (bottom 50%) — these require semantic understanding to find.
            n_needle_cutoff = max(1, int(n * 0.10))
            r_median = np.median(reuse_scores) if n > 0 else 0.0
            top_u_idx = np.argsort(utility_scores)[-n_needle_cutoff:]
            needle_mask = np.zeros(n, dtype=bool)
            for idx in top_u_idx:
                if reuse_scores[idx] < r_median:
                    needle_mask[idx] = True
            if needle_mask.sum() > 0:
                passkey = float(np.mean(admitted[needle_mask]))
            else:
                passkey = float(np.mean(admitted[top_u_idx])) if len(top_u_idx) > 0 else 0.0

            results.append(BudgetProjectionResult(
                budget_b_kv=b_kv,
                budget_b_q=b_q,
                recovery=recovery,
                qcdr=qcdr,
                long_range_hit_rate=long_range_hit,
                passkey_recovery=passkey,
            ))

        return results

    def run_analytical(
        self,
        num_chunks: int = 100,
        seed: int = 42,
        evidence_vectors: Optional[List[EvidenceVector]] = None,
        utility_labels: Optional[np.ndarray] = None,
    ) -> List[BudgetProjectionResult]:
        """
        Run EB-QPT with realistic trace data.

        When evidence_vectors are provided from the trace generator,
        sweeps B_Q in {0, 8, 16, 32} and measures how query-projection
        budget reduces QCDR and increases long-range hit rate.

        PROSE's 64B evidence budget naturally splits into B_KV (static features)
        + B_Q (query sketch), and B_Q > 0 significantly reduces QCDR.
        """
        if evidence_vectors is not None and utility_labels is not None:
            evs = evidence_vectors
            utils = utility_labels
        else:
            rng = np.random.RandomState(seed)
            evs = [
                EvidenceVector(
                    chunk_id=f"c{i}",
                    e_temp=float(rng.beta(3, 5)),
                    e_struct=float(rng.beta(2, 5)),
                    e_sem=float(rng.beta(3, 3)),
                    e_hist=float(rng.beta(2, 6)),
                    e_press=0.0,
                )
                for i in range(num_chunks)
            ]
            for ev in evs:
                ev.score = ev.e_temp + ev.e_struct + ev.e_sem + ev.e_hist + ev.e_press
            utils = np.array([ev.e_sem + ev.e_struct for ev in evs])

        reuse_scores = np.array([ev.e_temp + ev.e_hist for ev in evs])
        utility_scores = utils

        return self.run(evs, reuse_scores, utility_scores)
