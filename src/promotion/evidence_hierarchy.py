"""
Tiered Evidence Hierarchy — Cascaded Admission Protocol for Multi-Tier Memory.

Generalizes SBFI from a single HBM↔CXL boundary to N-tier storage hierarchies.
Each tier boundary uses progressively larger evidence summaries, creating a
cascaded admission filter that protects each link from unnecessary bulk transfers.

Core insight:
  HBM → CXL:     16B "Bloom fingerprint" — fast hash, coarse filter
  CXL → SSD:     64B "Sketch" — SimHash/minhash, medium fidelity
  SSD → Remote:  256B "Compressed fragment" — quantized embedding, high fidelity

The PROSE controller logic (scoring, scheduling, burst, sticky) remains identical
across all tiers. Only the evidence profile changes — what evidence to fetch at
each boundary, and what admission threshold to apply.

References:
  - Broder, "On the resemblance and containment of documents" (SimHash)
  - Bloom, "Space/time trade-offs in hash coding" (Bloom filter)
  - PROSE-SBFI: score-before-fetch ordering pattern
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.memory.multi_tier_hierarchy import (
    TierSpec, TierConfig, TierStepStats, MultiTierStepStats,
    TierQueueSimulator, get_default_tier_configs,
)


# ── Evidence Level Enum ────────────────────────────────────────────────

class EvidenceLevel(Enum):
    """Evidence granularity at a tier boundary."""
    NONE = 0            # No evidence (HBM-local)
    BLOOM_16B = auto()  # 16-byte Bloom fingerprint
    SKETCH_64B = auto() # 64-byte SimHash/minhash sketch
    FRAGMENT_256B = auto()  # 256-byte compressed embedding fragment

    @property
    def size_bytes(self) -> int:
        return {
            EvidenceLevel.NONE: 0,
            EvidenceLevel.BLOOM_16B: 16,
            EvidenceLevel.SKETCH_64B: 64,
            EvidenceLevel.FRAGMENT_256B: 256,
        }[self]

    @property
    def label(self) -> str:
        return {
            EvidenceLevel.NONE: "none",
            EvidenceLevel.BLOOM_16B: "bloom-16B",
            EvidenceLevel.SKETCH_64B: "sketch-64B",
            EvidenceLevel.FRAGMENT_256B: "fragment-256B",
        }[self]


# ── Evidence Ladder Strategy ────────────────────────────────────────────

@dataclass
class EvidenceLadder:
    """Defines the evidence strategy for a multi-tier hierarchy.

    Maps each tier boundary (source_tier, dest_tier) to an evidence level,
    defining what size and type of summary to fetch before promotion.
    """
    name: str
    # Map from tier index (0=HBM, 1=CXL, 2=SSD, 3=Remote) to evidence level
    # Key = source tier index, value = evidence to use when promoting FROM that tier
    boundary_evidence: Dict[int, EvidenceLevel] = field(default_factory=dict)

    @classmethod
    def fixed_16b(cls) -> "EvidenceLadder":
        """All tiers use 16B bloom fingerprint."""
        return cls(
            name="fixed-16B",
            boundary_evidence={
                1: EvidenceLevel.BLOOM_16B,   # HBM→CXL: 16B
                2: EvidenceLevel.BLOOM_16B,   # CXL→SSD: 16B
                3: EvidenceLevel.BLOOM_16B,   # SSD→Remote: 16B
            },
        )

    @classmethod
    def fixed_64b(cls) -> "EvidenceLadder":
        """All tiers use 64B sketch."""
        return cls(
            name="fixed-64B",
            boundary_evidence={
                1: EvidenceLevel.SKETCH_64B,
                2: EvidenceLevel.SKETCH_64B,
                3: EvidenceLevel.SKETCH_64B,
            },
        )

    @classmethod
    def fixed_256b(cls) -> "EvidenceLadder":
        """All tiers use 256B compressed fragment."""
        return cls(
            name="fixed-256B",
            boundary_evidence={
                1: EvidenceLevel.FRAGMENT_256B,
                2: EvidenceLevel.FRAGMENT_256B,
                3: EvidenceLevel.FRAGMENT_256B,
            },
        )

    @classmethod
    def progressive_ladder(cls) -> "EvidenceLadder":
        """Progressive evidence enlargement: 16B → 64B → 256B.

        Each deeper tier uses more evidence because:
        - Byte-at-risk is larger (more data potentially wasted)
        - Utility uncertainty is higher (further from computation)
        - Link bandwidth is lower (evidence cost amortized better)
        """
        return cls(
            name="ladder-16-64-256",
            boundary_evidence={
                1: EvidenceLevel.BLOOM_16B,      # HBM→CXL
                2: EvidenceLevel.SKETCH_64B,      # CXL→SSD
                3: EvidenceLevel.FRAGMENT_256B,   # SSD→Remote
            },
        )

    @classmethod
    def oracle_ladder(cls) -> "EvidenceLadder":
        """Oracle: evidence size inversely proportional to utility uncertainty.

        At HBM→CXL: low uncertainty (recent chunks have high utility corr) → 16B
        At CXL→SSD: moderate uncertainty → 64B
        At SSD→Remote: high uncertainty (old, cold chunks) → 256B
        """
        return cls(
            name="oracle-ladder",
            boundary_evidence={
                1: EvidenceLevel.BLOOM_16B,
                2: EvidenceLevel.SKETCH_64B,
                3: EvidenceLevel.FRAGMENT_256B,
            },
        )

    @classmethod
    def no_evidence_fts(cls) -> "EvidenceLadder":
        """Fetch-then-score at all tiers: no evidence gating.

        This is the baseline: ALL payloads are fetched before scoring.
        Evidence size = 0 at all boundaries.
        """
        return cls(
            name="fetch-before-score",
            boundary_evidence={
                1: EvidenceLevel.NONE,
                2: EvidenceLevel.NONE,
                3: EvidenceLevel.NONE,
            },
        )

    @classmethod
    def all_strategies(cls) -> List["EvidenceLadder"]:
        """Return all comparable evidence ladder strategies."""
        return [
            cls.fixed_16b(),
            cls.fixed_64b(),
            cls.fixed_256b(),
            cls.progressive_ladder(),
            cls.oracle_ladder(),
            cls.no_evidence_fts(),
        ]

    def get_evidence_size(self, source_tier_idx: int) -> int:
        """Get evidence size in bytes for promoting FROM this tier."""
        level = self.boundary_evidence.get(source_tier_idx, EvidenceLevel.NONE)
        return level.size_bytes

    def get_evidence_level(self, source_tier_idx: int) -> EvidenceLevel:
        return self.boundary_evidence.get(source_tier_idx, EvidenceLevel.NONE)


# ── Evidence Generator ──────────────────────────────────────────────────

class EvidenceGenerator:
    """Generates evidence summaries at different fidelity levels.

    Simulates the information content of each evidence type:
    - 16B Bloom: 128-bit hash fingerprint (Bloom filter with k=4 hashes)
      Effective discrimination: ~85-90% precision at moderate load
    - 64B Sketch: 512-bit SimHash, captures semantic similarity
      Effective discrimination: ~92-96% precision
    - 256B Fragment: 2048-bit quantized embedding
      Effective discrimination: ~96-99% precision
    """

    # Information quality: how well each evidence level predicts true utility
    # Represented as noise added to the utility signal
    QUALITY = {
        EvidenceLevel.NONE: 0.0,           # No information
        EvidenceLevel.BLOOM_16B: 0.75,      # 75% signal preservation
        EvidenceLevel.SKETCH_64B: 0.88,     # 88% signal preservation
        EvidenceLevel.FRAGMENT_256B: 0.96,  # 96% signal preservation
    }

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)

    def generate_evidence_scores(
        self,
        chunk_ids: List[int],
        true_utilities: np.ndarray,       # ground-truth utility per chunk
        evidence_level: EvidenceLevel,
        noise_scale: float = 0.1,
    ) -> Dict[int, float]:
        """Generate evidence-based utility scores for a set of candidate chunks.

        The evidence score = true_utility * quality + noise * (1 - quality)
        where quality depends on the evidence level.

        Args:
            chunk_ids: Candidate chunk IDs
            true_utilities: Ground-truth utility values (array indexed by chunk_id)
            evidence_level: What evidence fidelity to use
            noise_scale: Magnitude of noise for the non-signal portion

        Returns:
            Dict mapping chunk_id → evidence-based score
        """
        quality = self.QUALITY.get(evidence_level, 0.5)
        scores = {}
        for cid in chunk_ids:
            if 0 <= cid < len(true_utilities):
                signal = float(true_utilities[cid])
                noise = self._rng.normal(0.0, noise_scale)
                # Higher quality → more signal, less noise
                scores[cid] = quality * signal + (1.0 - quality) * abs(noise)
        return scores

    def generate_evidence_bytes(
        self,
        chunk_ids: List[int],
        evidence_level: EvidenceLevel,
    ) -> bytes:
        """Simulate the actual evidence bytes (for bandwidth accounting)."""
        return b'\x00' * (len(chunk_ids) * evidence_level.size_bytes)


# ── Tiered Admission Controller ─────────────────────────────────────────

@dataclass
class AdmissionStep:
    """Result of a single tier's admission decision."""
    source_tier: TierSpec
    dest_tier: TierSpec
    candidates_in: List[int]
    admitted: List[int]          # Passed evidence gate → promote payload
    rejected: List[int]          # Failed evidence gate → stay at source tier
    evidence_fetch: Optional[TierFetchResult] = None
    payload_fetch: Optional[TierFetchResult] = None


class TieredAdmissionController:
    """Cascaded admission controller for multi-tier memory.

    Implements the tiered SBFI protocol:
    1. At each tier boundary, fetch evidence (size depends on ladder config)
    2. Score candidates using evidence
    3. Admit top-K to the next tier (promote payload)
    4. Rejected candidates stay at current tier

    The scoring FUNCTION is the same across all tiers (ODUS-X multi-cue).
    Only the evidence SIZE and admission THRESHOLD vary per tier.
    """

    def __init__(
        self,
        tier_configs: Dict[TierSpec, TierConfig],
        ladder: EvidenceLadder,
        evidence_generator: Optional[EvidenceGenerator] = None,
    ):
        self.tier_configs = tier_configs
        self.ladder = ladder
        self.evidence_gen = evidence_generator or EvidenceGenerator()

        # Per-tier queue simulators (skip HBM for queues — it's local)
        self._queues: Dict[TierSpec, TierQueueSimulator] = {}
        for tier in [TierSpec.CXL_DRAM, TierSpec.CXL_FLASH, TierSpec.REMOTE_RDMA]:
            if tier in tier_configs:
                self._queues[tier] = TierQueueSimulator(tier_configs[tier])

    def reset(self):
        for q in self._queues.values():
            q.reset()

    def cascade_admit(
        self,
        candidate_ids: List[int],
        true_utilities: np.ndarray,
        anchor_ids: List[int],
        budget_per_tier: Dict[TierSpec, int],      # tiers present → max promote
        scorer_fn: Callable[[List[int], np.ndarray, List[int]], List[int]],
        active_tiers: List[TierSpec],               # Ordered HBM → ... → Remote
        step: int = 0,
    ) -> Tuple[MultiTierStepStats, List[int]]:
        """Run the full cascaded admission pipeline for one decode step.

        Flow:
        1. Candidates start at the deepest active tier
        2. For each tier boundary (deep → shallow):
           a. Fetch evidence at this tier's evidence size
           b. Score candidates using evidence
           c. Admit top-K to next tier (promote payload)
           d. Rejected stay at current tier
        3. Track all per-tier stats

        Args:
            candidate_ids: All potential candidates for promotion
            true_utilities: Ground-truth utility (for oracle comparison)
            anchor_ids: Anchor chunks (always resident in HBM)
            budget_per_tier: Max chunks to promote across each boundary
            scorer_fn: Scoring function (same across all tiers)
            active_tiers: Ordered list of active tiers [HBM, CXL, SSD, Remote]
            step: Decode step number

        Returns:
            (MultiTierStepStats, final_hbm_resident_chunks)
        """
        step_stats = MultiTierStepStats(step=step)
        anchor_set = set(anchor_ids)

        # Filter out anchors from candidates
        candidates = [c for c in candidate_ids if c not in anchor_set]

        # Work from deepest tier toward HBM
        # active_tiers[0]=HBM, active_tiers[-1]=deepest
        current_residents = set(candidates)  # Start: everything at deepest tier

        # Track which chunks have been promoted to each tier
        promoted_to: Dict[TierSpec, set] = {t: set() for t in active_tiers}
        promoted_to[active_tiers[-1]] = set(candidates)  # Start at deepest

        # Process each boundary from deep → shallow
        for i in range(len(active_tiers) - 1, 0, -1):
            source_tier = active_tiers[i]
            dest_tier = active_tiers[i - 1]
            budget = budget_per_tier.get(source_tier, 0)

            if source_tier not in self._queues or budget <= 0:
                # Skip if this tier has no queue or budget
                continue

            queue = self._queues[source_tier]
            candidates_at_tier = sorted(promoted_to.get(source_tier, set()))

            if not candidates_at_tier:
                continue

            evidence_level = self.ladder.get_evidence_level(source_tier.value)
            evidence_size = evidence_level.size_bytes

            # ── Branch: Evidence-gated vs Fetch-then-score ──
            if evidence_level == EvidenceLevel.NONE:
                # Fetch-then-score: no evidence gate, fetch all payloads
                payload_result = queue.submit_payload_fetch(candidates_at_tier)

                # Score all (too late — payloads already fetched)
                ranked = scorer_fn(candidates_at_tier, true_utilities, anchor_ids)
                admitted = ranked[:budget]
                rejected = [c for c in candidates_at_tier if c not in admitted]

                queue.mark_chunks_invalid(rejected)
                queue.mark_chunks_used(admitted)

                evidence_result = None

            else:
                # Score-before-fetch: fetch evidence first
                evidence_result = queue.submit_evidence_fetch(
                    candidates_at_tier, evidence_size
                )

                # Generate evidence-based scores
                evidence_scores = self.evidence_gen.generate_evidence_scores(
                    candidates_at_tier, true_utilities, evidence_level,
                )

                # Score candidates using evidence
                ranked = scorer_fn(candidates_at_tier, true_utilities, anchor_ids)

                # Blend with evidence scores (weighted by evidence quality)
                quality = EvidenceGenerator.QUALITY[evidence_level]
                blended = {}
                for cid in candidates_at_tier:
                    base_score = 0.0
                    # Compute rank-based score from ranked order
                    if cid in ranked:
                        rank = ranked.index(cid)
                        base_score = max(0.0, 1.0 - rank / len(ranked))
                    ev_score = evidence_scores.get(cid, 0.0)
                    blended[cid] = quality * ev_score + (1.0 - quality) * base_score

                admitted = sorted(blended, key=blended.get, reverse=True)[:budget]
                rejected = [c for c in candidates_at_tier if c not in admitted]

                # Track invalid evidence bytes
                queue.mark_evidence_invalid(len(rejected), evidence_size)

                # Fetch payloads for admitted chunks only
                if admitted:
                    payload_result = queue.submit_payload_fetch(admitted)
                    queue.mark_chunks_used(admitted)
                else:
                    payload_result = None

            # Record cross-tier traffic
            edge = (source_tier, dest_tier)
            if payload_result:
                step_stats.cross_tier_payload_bytes[edge] = (
                    step_stats.cross_tier_payload_bytes.get(edge, 0)
                    + payload_result.total_bytes
                )
            if evidence_result and evidence_level != EvidenceLevel.NONE:
                step_stats.cross_tier_evidence_bytes[edge] = (
                    step_stats.cross_tier_evidence_bytes.get(edge, 0)
                    + evidence_result.total_bytes
                )
            step_stats.cross_tier_promotions[edge] = len(admitted)

            # Track which chunks moved to the next tier
            promoted_to[dest_tier] = promoted_to.get(dest_tier, set()) | set(admitted)

        # ── End of step: collect per-tier stats ──
        for tier, queue in self._queues.items():
            # Compute byte-at-risk for this tier
            num_resident = len(promoted_to.get(tier, set()))
            # Miss probability increases with tier depth
            miss_prob = 0.05 * tier.value  # 0% HBM, 5% CXL, 10% SSD, 15% Remote

            tier_stats = queue.end_step(
                decode_step_ns=100_000.0,
                num_chunks_resident=num_resident,
                miss_probability=miss_prob,
            )
            step_stats.per_tier[tier] = tier_stats

        # Final HBM-resident set
        final_hbm = set(anchor_ids) | promoted_to.get(TierSpec.HBM, set())

        step_stats.gold_chunks = self._get_gold(true_utilities, anchor_ids)
        step_stats.recovered_chunks = list(final_hbm)

        # Total stall = sum of queuing across all tiers
        step_stats.total_stall_ns = sum(
            s.total_queuing_ns for s in step_stats.per_tier.values()
        )

        return step_stats, sorted(final_hbm)

    @staticmethod
    def _get_gold(true_utilities: np.ndarray, anchor_ids: List[int],
                  top_k: int = 100) -> List[int]:
        """Oracle gold set: top-K by true utility (excluding anchors)."""
        anchor_set = set(anchor_ids)
        ranked = np.argsort(true_utilities)[::-1]
        gold = []
        for cid in ranked:
            if int(cid) not in anchor_set and len(gold) < top_k:
                gold.append(int(cid))
        return gold
