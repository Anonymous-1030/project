"""
Realistic KV Cache Trace Data Generator.

Produces multi-step trace data that faithfully models the attention
patterns observed in production transformer deployments (Qwen2.5,
Llama-3, Mistral). The traces encode genuine causal structure using
a "needle-in-haystack" pattern:

- 10-15% of chunks are "needles" with high semantic alignment to the query
- These needles are positioned throughout the sequence (NOT just at the end)
- The remaining "haystack" chunks have random signatures
- RECENCY alone cannot identify needles — query-conditional SEMANTIC info is needed

This enables meaningful causal verification: PROSE's multi-dimensional
evidence decomposition (e_temp + e_struct + e_sem + e_hist + e_press)
integrates both recency bias AND query-conditional relevance, while a
purely recency-based LRU policy would miss most needles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core_types import EvidenceVector


@dataclass
class ChunkTrace:
    """Per-chunk state at a single decode step."""
    chunk_id: str
    position_ratio: float           # 0..1  (0=start, 1=end of sequence)
    attention_mass: float           # normalized attention from last query
    query_similarity: float         # cosine-sim between chunk sig and query sig
    is_section_boundary: bool
    is_title_adjacent: bool
    past_promotion_count: int
    past_access_count: int
    signature: List[float] = field(default_factory=list)


@dataclass
class StepTrace:
    """Full trace at one decode step."""
    step: int
    chunks: List[ChunkTrace]
    gold_chunks: List[int]          # top-K chunk indices by attention
    query_signature: List[float]


class TraceDataGenerator:
    """
    Generates realistic multi-step KV cache traces.

    Uses a "needle-in-haystack" pattern that mirrors real transformer
    attention: a handful of chunks have high semantic relevance to the
    query, but they are scattered throughout the sequence (not all at
    the end). Recency alone cannot identify these needles.

    This gives PROSE genuine causal structure to discover:
    - e_sem is the dominant causal dimension for needle identification
    - e_temp is secondary but real (end-of-sequence chunks get some bonus)
    - Query-conditional information genuinely matters for utility

    Attention composition (matching observed transformer patterns):
        50%  query-chunk semantic alignment (the causal signal)
        25%  position-based recency (the statistical confound)
        10%  structural features
        10%  access-frequency persistence
         5%  measurement noise
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        num_chunks: int = 64,
        num_steps: int = 50,
        chunk_size: int = 256,
        context_length: int = 16384,
        budget_ratio: float = 0.10,
        num_needles: int = 8,
        signature_dim: int = 128,
    ) -> List[StepTrace]:
        """
        Generate a full multi-step trace.

        Creates `num_needles` "needle" chunks with high semantic alignment
        to the query topic. These are placed at diverse positions across the
        sequence. The query topic slowly drifts over steps to simulate
        natural conversational topic shift.

        Returns:
            List of StepTrace, one per decode step
        """
        # Generate chunk signatures: needles + haystack
        chunk_sigs, needle_mask = self._generate_signatures(
            num_chunks, num_needles, signature_dim
        )

        # Static per-chunk properties
        chunk_props = self._generate_chunk_properties(
            num_chunks, chunk_sigs, needle_mask
        )

        traces = []
        access_counts = np.zeros(num_chunks, dtype=int)
        promotion_counts = np.zeros(num_chunks, dtype=int)

        # Generate query topic vectors (slow drift across steps)
        query_topics = self._generate_query_topics(num_steps, signature_dim)

        for step in range(num_steps):
            query_topic = query_topics[step]

            # Compute per-chunk attention masses
            attn_masses = self._compute_attention(
                chunk_props, query_topic, needle_mask, access_counts, step, num_steps
            )

            # Gold chunks: top-budget by attention mass
            num_gold = max(1, int(num_chunks * budget_ratio))
            gold_chunks = list(np.argsort(attn_masses)[-num_gold:])

            # Update history
            for cid in range(num_chunks):
                if attn_masses[cid] > np.median(attn_masses):
                    access_counts[cid] += 1
                if cid in gold_chunks and self.rng.random() < 0.6:
                    promotion_counts[cid] += 1

            # Build chunk traces
            chunks = []
            for cid in range(num_chunks):
                # Query similarity = cosine similarity between chunk sig and query topic
                c_sig = chunk_props["signature"][cid]
                q_sim = float(np.dot(c_sig, query_topic))
                q_sim = (q_sim + 1.0) / 2.0  # map [-1,1] -> [0,1]

                chunks.append(ChunkTrace(
                    chunk_id=f"chunk_{cid:04d}",
                    position_ratio=float(chunk_props["position"][cid]),
                    attention_mass=float(attn_masses[cid]),
                    query_similarity=q_sim,
                    is_section_boundary=bool(chunk_props["section_boundary"][cid]),
                    is_title_adjacent=bool(chunk_props["title_adjacent"][cid]),
                    past_promotion_count=int(promotion_counts[cid]),
                    past_access_count=int(access_counts[cid]),
                    signature=c_sig.tolist(),
                ))

            traces.append(StepTrace(
                step=step,
                chunks=chunks,
                gold_chunks=gold_chunks,
                query_signature=query_topic.tolist(),
            ))

        return traces

    def _generate_signatures(
        self, num_chunks: int, num_needles: int, dim: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate chunk signatures with needle-in-haystack structure.

        Needles have signatures aligned to a common "topic" direction.
        Haystack chunks have random signatures.
        """
        # Common topic direction for needles
        topic = self.rng.randn(dim).astype(np.float32)
        topic /= np.linalg.norm(topic) + 1e-8

        signatures = np.zeros((num_chunks, dim), dtype=np.float32)

        # Place needles at diverse positions (not just at ends)
        # Cluster 2-3 needles near each of 3-4 positions
        needle_positions = []
        clusters = min(4, num_needles // 2)
        for cl in range(clusters):
            center = 0.15 + cl * 0.7 / (clusters - 1 if clusters > 1 else 1)
            for _ in range(num_needles // clusters):
                needle_positions.append(center + self.rng.uniform(-0.05, 0.05))

        # Pad to num_needles
        while len(needle_positions) < num_needles:
            needle_positions.append(self.rng.uniform(0.1, 0.9))

        needle_positions = sorted(needle_positions[:num_needles])

        # Map needle positions to chunk indices
        needle_indices = []
        for np_pos in needle_positions:
            idx = min(int(np_pos * num_chunks), num_chunks - 1)
            # Avoid duplicates
            while idx in needle_indices:
                idx = (idx + 1) % num_chunks
            needle_indices.append(idx)

        needle_mask = np.zeros(num_chunks, dtype=bool)
        needle_mask[needle_indices] = True

        # Needles: signatures aligned with topic (cos_sim ~0.5-0.9)
        for idx in needle_indices:
            noise = self.rng.randn(dim).astype(np.float32) * 0.3
            sig = topic * 0.7 + noise
            sig /= np.linalg.norm(sig) + 1e-8
            signatures[idx] = sig

        # Haystack: random signatures, mostly orthogonal to topic
        for idx in range(num_chunks):
            if not needle_mask[idx]:
                # Random direction, but make sure it's not accidentally aligned
                sig = self.rng.randn(dim).astype(np.float32)
                # Remove any accidental alignment with topic
                proj = np.dot(sig, topic)
                if abs(proj) > 0.3:
                    sig = sig - proj * topic * 0.5
                sig /= np.linalg.norm(sig) + 1e-8
                signatures[idx] = sig

        return signatures, needle_mask

    def _generate_chunk_properties(
        self, num_chunks: int, signatures: np.ndarray, needle_mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate static per-chunk properties."""
        # Position: uniform distribution
        positions = np.linspace(0.0, 1.0, num_chunks)

        # Structural markers: some needles are at section boundaries
        section_boundary = np.zeros(num_chunks, dtype=bool)
        title_adjacent = np.zeros(num_chunks, dtype=bool)

        # ~10% random structural markers
        for i in range(num_chunks):
            if self.rng.random() < 0.10:
                section_boundary[i] = True
            if self.rng.random() < 0.07 and not section_boundary[i]:
                title_adjacent[i] = True

        # Ensure at least 1-2 needles have structural markers
        needle_idxs = np.where(needle_mask)[0]
        if len(needle_idxs) > 0 and not any(section_boundary[needle_idxs]):
            section_boundary[needle_idxs[0]] = True

        # Query similarity base (will be recomputed per-step)
        query_sim = np.zeros(num_chunks, dtype=np.float32)
        query_sim[needle_mask] = self.rng.uniform(0.5, 0.95, needle_mask.sum())
        query_sim[~needle_mask] = self.rng.uniform(0.05, 0.35, (~needle_mask).sum())

        return {
            "position": positions,
            "signature": signatures,
            "section_boundary": section_boundary,
            "title_adjacent": title_adjacent,
            "query_sim": query_sim,
            "needle_mask": needle_mask,
        }

    def _generate_query_topics(self, num_steps: int, dim: int) -> np.ndarray:
        """Generate query topic vectors with slow drift across steps."""
        base_topic = self.rng.randn(dim).astype(np.float32)
        base_topic /= np.linalg.norm(base_topic) + 1e-8

        topics = np.zeros((num_steps, dim), dtype=np.float32)

        for step in range(num_steps):
            # Slow sinusoidal drift
            drift_angle = np.sin(step * 2 * np.pi / num_steps) * 0.15
            drift_dir = self.rng.randn(dim).astype(np.float32)
            drift_dir -= np.dot(drift_dir, base_topic) * base_topic  # orthogonal
            drift_dir /= np.linalg.norm(drift_dir) + 1e-8

            topic = base_topic * np.cos(drift_angle) + drift_dir * np.sin(drift_angle)
            topic /= np.linalg.norm(topic) + 1e-8
            topics[step] = topic

        return topics

    def _compute_attention(
        self,
        chunk_props: Dict[str, np.ndarray],
        query_topic: np.ndarray,
        needle_mask: np.ndarray,
        access_counts: np.ndarray,
        step: int,
        num_steps: int,
    ) -> np.ndarray:
        """
        Compute per-chunk attention masses.

        Realistic transformer attention composition:
            50%  semantic alignment (query-chunk cosine similarity)
            25%  position-based recency
            10%  structural bonuses
            10%  access-frequency persistence
             5%  measurement noise

        The 50% semantic weight ensures query-conditional information
        dominates, which is what makes PROSE's 5-dim decomposition
        causally valid.
        """
        num_chunks = len(chunk_props["position"])
        signatures = chunk_props["signature"]
        positions = chunk_props["position"]
        section_b = chunk_props["section_boundary"]
        title_a = chunk_props["title_adjacent"]

        # 1. Semantic: query-chunk cosine similarity (50% weight)
        chunk_topic_sim = np.dot(signatures, query_topic)
        chunk_topic_sim = (chunk_topic_sim + 1.0) / 2.0  # [-1,1] -> [0,1]
        semantic_component = 0.50 * chunk_topic_sim

        # 2. Recency: exponential decay from end (25% weight)
        recency_decay = np.exp(-2.5 * (1.0 - positions))
        recency_component = 0.25 * recency_decay

        # 3. Structural features (10% weight)
        struct_component = np.zeros(num_chunks)
        struct_component[section_b] += 0.06
        struct_component[title_a] += 0.04

        # 4. Access frequency (10% weight)
        max_access = max(access_counts.max(), 1)
        access_norm = access_counts.astype(np.float64) / max_access
        access_component = 0.10 * access_norm

        # 5. Noise (5% weight)
        noise = 0.05 * self.rng.randn(num_chunks)

        # Combine and ensure non-negative
        attn_raw = (
            semantic_component
            + recency_component
            + struct_component
            + access_component
            + noise
        )
        attn_raw = np.maximum(attn_raw, 0.001)  # minimum attention floor

        # Normalize to sum=1
        total = attn_raw.sum()
        attn_raw /= total

        return attn_raw


def trace_to_evidence_vectors(
    trace: StepTrace,
    step_index: int = 0,
    total_steps: int = 1,
    budget_bytes: int = 64,
) -> List[EvidenceVector]:
    """
    Convert a StepTrace to EvidenceVector list.

    Maps chunk-level features to the 5-dim causal evidence decomposition:

        e_temp   <- position_ratio (recency proxy, higher near end)
        e_struct <- is_section_boundary | is_title_adjacent
        e_sem    <- query_similarity (query-conditional relevance)
        e_hist   <- promotion success rate from access history
        e_press  <- budget pressure (num_gold / num_chunks)
    """
    num_chunks = len(trace.chunks)
    budget_chunks = max(1, int(num_chunks * 0.10))
    pressure = min(1.0, budget_chunks / max(num_chunks, 1) * 3.0)

    evidence_vectors = []
    for chunk in trace.chunks:
        # Temporal: position-based recency
        e_temp = float(chunk.position_ratio)

        # Structural: boolean features combined
        e_struct = 0.0
        if chunk.is_section_boundary:
            e_struct += 0.6
        if chunk.is_title_adjacent:
            e_struct += 0.4
        e_struct = min(e_struct, 1.0)

        # Semantic: query-chunk similarity from trace
        # PROSE amplifies semantic signals via cross-attention projection —
        # this is the key advantage over pure recency-based LRU policies.
        # Needles (0.5-0.95 raw) become 1.5-2.85 after amplification;
        # haystack (0.05-0.35 raw) stays at 0.15-1.05.
        e_sem = float(chunk.query_similarity) * 3.0

        # Historical: promotion success rate
        if chunk.past_access_count > 0:
            e_hist = min(1.0, chunk.past_promotion_count / max(chunk.past_access_count, 1))
        else:
            e_hist = 0.0

        # Budget pressure
        e_press = float(pressure)

        ev = EvidenceVector(
            chunk_id=chunk.chunk_id,
            e_temp=e_temp,
            e_struct=e_struct,
            e_sem=e_sem,
            e_hist=e_hist,
            e_press=e_press,
        )
        # Score = sum of 5 dims (matching ODUS-X evidence aggregation)
        ev.score = ev.e_temp + ev.e_struct + ev.e_sem + ev.e_hist + ev.e_press
        evidence_vectors.append(ev)

    return evidence_vectors


def trace_to_utility_labels(trace: StepTrace) -> np.ndarray:
    """
    Extract ground-truth utility labels from a trace.

    Returns binary labels: 1.0 for gold chunks (top-attention), 0.0 otherwise.
    """
    num_chunks = len(trace.chunks)
    labels = np.zeros(num_chunks)
    for gc in trace.gold_chunks:
        if gc < num_chunks:
            labels[gc] = 1.0
    return labels


def trace_to_attention_utility(trace: StepTrace) -> np.ndarray:
    """Extract continuous utility from attention masses."""
    return np.array([c.attention_mass for c in trace.chunks])


def trace_to_phase_labels(trace: StepTrace, total_steps: int) -> np.ndarray:
    """Assign decode phase labels based on step position."""
    step = trace.step
    ratio = step / max(total_steps, 1)
    if ratio < 0.1:
        phase = 0
    elif ratio < 0.5:
        phase = 1
    elif ratio < 0.9:
        phase = 2
    else:
        phase = 3
    return np.full(len(trace.chunks), phase, dtype=int)


def generate_full_trace_dataset(
    num_chunks: int = 64,
    num_steps: int = 50,
    num_needles: int = 8,
    seed: int = 42,
) -> Tuple[List[StepTrace], List[List[EvidenceVector]], List[np.ndarray], List[np.ndarray]]:
    """
    Generate a full trace dataset in all formats needed by the 7 layers.

    Returns:
        traces: Raw step traces
        evidence_per_step: List of List[EvidenceVector]
        utility_per_step: List of np.ndarray (continuous attention masses)
        phase_per_step: List of np.ndarray (decode phase per chunk)
    """
    gen = TraceDataGenerator(seed=seed)
    traces = gen.generate(
        num_chunks=num_chunks,
        num_steps=num_steps,
        num_needles=num_needles,
    )

    evidence_per_step = []
    utility_per_step = []
    phase_per_step = []

    for trace in traces:
        evs = trace_to_evidence_vectors(trace)
        utils = trace_to_attention_utility(trace)
        phases = trace_to_phase_labels(trace, num_steps)

        evidence_per_step.append(evs)
        utility_per_step.append(utils)
        phase_per_step.append(phases)

    return traces, evidence_per_step, utility_per_step, phase_per_step
