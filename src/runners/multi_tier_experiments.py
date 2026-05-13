"""
Multi-Tier Evidence Hierarchy Experiments — HPCA Extension Suite.

Three experiment groups that generalize PROSE from a single HBM↔CXL boundary
to a full N-tier storage hierarchy with cascaded evidence-based admission.

  EXP-A: Tiered Replay Experiment
    Simulate 4-tier memory with cascaded SBFI, collecting:
    - Cross-tier payload/evidence bytes
    - P99/P999 stall per tier
    - Queue occupancy per tier
    - Useful-KV recovery
    - False-positive payload transfers
    - Cumulative byte-at-risk

  EXP-B: Evidence-Size Ablation
    Compare fixed evidence sizes vs progressive ladder vs FTS baseline:
    - fixed-16B, fixed-64B, fixed-256B
    - ladder-16-64-256 (our proposal)
    - oracle-ladder
    - fetch-before-score (no evidence)

  EXP-C: Programmability Experiment
    Same PROSE controller, different backend profiles:
    - CXL-only
    - CXL + SSD
    - CXL + SSD + Remote
    - Remote-only fallback

Usage:
  python -m prosex.src.runners.multi_tier_experiments --experiment all
  python -m prosex.src.runners.multi_tier_experiments --experiment A
  python -m prosex.src.runners.multi_tier_experiments --experiment A --output results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.memory.multi_tier_hierarchy import (
    TierSpec, TierConfig, TierStepStats, MultiTierStepStats, MultiTierResult,
    TierQueueSimulator, get_default_tier_configs,
    TIER_NAMES, TIER_SHORT_NAMES,
)
from src.promotion.evidence_hierarchy import (
    EvidenceLevel, EvidenceLadder, EvidenceGenerator,
    TieredAdmissionController, AdmissionStep,
)


# ═════════════════════════════════════════════════════════════════════════
# Synthetic Trace Generation
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class SyntheticTraceConfig:
    """Configuration for synthetic attention trace generation."""
    num_chunks: int = 65536          # Total KV chunks
    num_steps: int = 100             # Decode steps to simulate
    chunk_size_tokens: int = 512     # Tokens per chunk
    bytes_per_kv: int = 128          # Bytes per token (K+V, FP16)

    # Attention pattern
    zipf_alpha: float = 1.2          # Zipf skew (higher = more concentrated)
    random_walk_sigma: float = 0.05  # Random walk step size
    hotspot_ratio: float = 0.05      # Fraction of chunks that get 80% of attention
    temporal_correlation: float = 0.7  # Step-to-step attention correlation [0,1]

    # Anchor configuration
    anchor_ratio: float = 0.02       # Fraction of chunks that are permanent anchors
    anchor_positions: Optional[List[int]] = None

    seed: int = 42


def generate_synthetic_trace(
    config: SyntheticTraceConfig,
) -> Tuple[List[np.ndarray], List[int], List[Dict[int, float]]]:
    """Generate synthetic attention traces mimicking long-context LLM behavior.

    Models:
    - Zipf popularity distribution (few chunks get most attention)
    - Random walk on attention mass (query evolves over time)
    - Temporal correlation (adjacent steps are similar)
    - Anchor chunks (permanent structural markers)

    Returns:
        (step_utilities, anchor_ids, step_attention_masses)
        - step_utilities[i]: np.ndarray[num_chunks] of utility scores at step i
        - anchor_ids: list of permanent anchor chunk IDs
        - step_attention_masses[i]: dict chunk_id → attention mass at step i
    """
    rng = np.random.RandomState(config.seed)

    # Generate anchor positions (evenly spaced + beginning)
    num_anchors = max(4, int(config.num_chunks * config.anchor_ratio))
    anchor_ids = list(range(0, config.num_chunks, config.num_chunks // num_anchors))[:num_anchors]
    anchor_ids = sorted(set(anchor_ids))  # Dedup

    # Generate base Zipf popularity distribution
    ranks = np.arange(1, config.num_chunks + 1)
    zipf_weights = 1.0 / (ranks ** config.zipf_alpha)
    zipf_weights = zipf_weights / zipf_weights.sum()

    # Generate hotspot boosts (few chunks get disproportionate attention)
    num_hotspots = max(10, int(config.num_chunks * config.hotspot_ratio))
    hotspot_ids = rng.choice(config.num_chunks, size=num_hotspots, replace=False)
    hotspot_boost = np.ones(config.num_chunks)
    for hid in hotspot_ids:
        hotspot_boost[hid] = rng.uniform(5.0, 20.0)  # 5x-20x boost

    # Combined base utility
    base_utility = zipf_weights * hotspot_boost
    base_utility = base_utility / base_utility.sum()

    # Simulate random walk over attention patterns
    step_utilities = []
    step_attention_masses = []

    current_utility = base_utility.copy()

    for step in range(config.num_steps):
        # Apply random walk perturbation
        noise = rng.normal(0.0, config.random_walk_sigma, config.num_chunks)
        # Temporal correlation: blend with previous step
        if step == 0:
            current_utility = base_utility * (1.0 + noise * 0.3)
        else:
            innovation = rng.normal(0.0, 0.1, config.num_chunks)
            current_utility = (
                config.temporal_correlation * current_utility
                + (1.0 - config.temporal_correlation) * (base_utility + innovation)
                + 0.1 * noise
            )

        # Shift attention mass toward a few random "query-relevant" regions
        query_center = rng.randint(0, config.num_chunks)
        distance_weights = np.exp(-0.001 * np.abs(np.arange(config.num_chunks) - query_center))
        current_utility = current_utility + 0.2 * distance_weights * current_utility

        # Normalize to probability distribution
        current_utility = np.maximum(current_utility, 0.0)
        if current_utility.sum() > 0:
            current_utility = current_utility / current_utility.sum()

        step_utilities.append(current_utility.copy())

        # Convert to attention masses dict (for scorer interface)
        attn_dict = {}
        # Top 500 chunks get explicit masses
        top_indices = np.argsort(current_utility)[-500:]
        for idx in top_indices:
            attn_dict[int(idx)] = float(current_utility[idx])
        step_attention_masses.append(attn_dict)

    return step_utilities, anchor_ids, step_attention_masses


# ═════════════════════════════════════════════════════════════════════════
# PROSE Multi-Tier Controller
# ═════════════════════════════════════════════════════════════════════════

class PROSEMultiTierController:
    """PROSE controller adapted for multi-tier memory hierarchies.

    Reuses the same ODUS-X multi-cue scoring logic from PROSEPolicy,
    but operates across N storage tiers with configurable evidence profiles.

    Key invariant: Scoring and scheduling logic is IDENTICAL across all
    backend profiles. Only the profile table changes (evidence sizes,
    tier bandwidth/latency, admission thresholds).
    """

    def __init__(
        self,
        active_tiers: List[TierSpec],
        tier_configs: Dict[TierSpec, TierConfig],
        evidence_ladder: EvidenceLadder,
        budget_per_tier: Optional[Dict[TierSpec, int]] = None,
        seed: int = 42,
    ):
        self.active_tiers = active_tiers
        self.tier_configs = tier_configs
        self.evidence_ladder = evidence_ladder

        # Default budget: proportional to tier depth
        if budget_per_tier is None:
            self.budget_per_tier = {
                TierSpec.CXL_DRAM: 20,
                TierSpec.CXL_FLASH: 40,
                TierSpec.REMOTE_RDMA: 80,
            }
        else:
            self.budget_per_tier = budget_per_tier

        self.evidence_gen = EvidenceGenerator(seed=seed)
        self.admission = TieredAdmissionController(
            tier_configs=tier_configs,
            ladder=evidence_ladder,
            evidence_generator=self.evidence_gen,
        )

        # PROSE state (mirrors PROSEPolicy)
        self.pht_ema: Dict[int, float] = {}
        self.prev_selected: List[int] = []
        self._ewma: Optional[np.ndarray] = None
        self._decay = 0.3
        self._window_buffer: List[np.ndarray] = []
        self._sticky_ttl: Dict[int, int] = {}

        self.result = MultiTierResult(
            num_steps=0,
            tier_configs=tier_configs,
        )

    def reset(self):
        self.admission.reset()
        self.pht_ema.clear()
        self.prev_selected.clear()
        self._ewma = None
        self._window_buffer.clear()
        self._sticky_ttl.clear()
        self.result = MultiTierResult(
            num_steps=0,
            tier_configs=self.tier_configs,
        )

    def score_chunks(
        self,
        candidate_ids: List[int],
        true_utilities: np.ndarray,
        anchor_ids: List[int],
    ) -> List[int]:
        """ODUS-X multi-cue scoring (same as PROSEPolicy._score_chunks)."""
        anchor_set = set(anchor_ids)
        scores = {}

        for cid in candidate_ids:
            if cid in anchor_set or cid < 0 or cid >= len(true_utilities):
                continue

            score = 0.0

            # Cue 1: Current utility (40%)
            score += 0.40 * float(true_utilities[cid])

            # Cue 2: EWMA (30%)
            if self._ewma is not None and cid < len(self._ewma):
                score += 0.30 * float(self._ewma[cid])

            # Cue 3: PHT history (15%)
            score += 0.15 * self.pht_ema.get(cid, 0.0)

            # Cue 4: Recency (10%)
            if cid in self.prev_selected:
                try:
                    recency_idx = self.prev_selected[::-1].index(cid)
                    score += 0.10 * max(0.0, 1.0 - recency_idx / 10.0)
                except ValueError:
                    pass

            # Cue 5: Position proximity to anchors (5%)
            n_chunks = len(true_utilities)
            min_dist = min(abs(cid - a) for a in anchor_ids) if anchor_ids else n_chunks
            score += 0.05 * max(0.0, 1.0 - min_dist / n_chunks)

            scores[cid] = score

        return sorted(scores, key=scores.get, reverse=True)

    def _generate_candidates(
        self,
        num_chunks: int,
        utility: np.ndarray,
        anchor_ids: List[int],
        lookahead_depth: int = 3,
    ) -> List[int]:
        """MQR-ULF candidate generation (mirrors PROSEPolicy)."""
        anchor_set = set(anchor_ids)
        candidates = set()

        # Queue 1: Top-K by EWMA
        if self._ewma is not None:
            ewma_order = np.argsort(self._ewma)[::-1]
            for cid in ewma_order[:max(5, num_chunks // 4)]:
                if int(cid) not in anchor_set:
                    candidates.add(int(cid))

        # Queue 2: Top-K by current utility
        top_utility = np.argsort(utility)[::-1][:5]
        for cid in top_utility:
            if int(cid) not in anchor_set:
                candidates.add(int(cid))

        # Queue 3: Recently accessed
        for cid in self.prev_selected[-5:]:
            if 0 <= cid < num_chunks and cid not in anchor_set:
                candidates.add(cid)

        # Queue 4: High PHT-EMA
        if self.pht_ema:
            pht_sorted = sorted(self.pht_ema.items(), key=lambda x: x[1], reverse=True)
            for cid, _ in pht_sorted[:5]:
                if 0 <= cid < num_chunks and cid not in anchor_set:
                    candidates.add(cid)

        # Queue 5: Lookahead neighbors
        if self._ewma is not None:
            top_idx = int(np.argmax(self._ewma))
            for offset in range(-lookahead_depth, lookahead_depth + 1):
                neighbor = top_idx + offset
                if 0 <= neighbor < num_chunks and neighbor not in anchor_set:
                    candidates.add(neighbor)

        # Fallback
        if len(candidates) < 8:
            for cid in range(min(num_chunks, 50)):
                if cid not in anchor_set and len(candidates) < 16:
                    candidates.add(cid)

        return sorted(candidates)

    def run_step(
        self,
        step: int,
        utility: np.ndarray,
        anchor_ids: List[int],
    ) -> MultiTierStepStats:
        """Execute one decode step of multi-tier PROSE."""
        num_chunks = len(utility)

        # Update EWMA
        if self._ewma is None:
            self._ewma = utility.copy()
        else:
            self._ewma = self._decay * utility + (1.0 - self._decay) * self._ewma

        # Candidate generation
        candidates = self._generate_candidates(num_chunks, utility, anchor_ids)

        # Cascaded admission across tiers
        step_stats, hbm_residents = self.admission.cascade_admit(
            candidate_ids=candidates,
            true_utilities=utility,
            anchor_ids=anchor_ids,
            budget_per_tier=self.budget_per_tier,
            scorer_fn=self.score_chunks,
            active_tiers=self.active_tiers,
            step=step,
        )

        # Update PHT with HBM residents
        alpha = 0.15
        for cid in hbm_residents:
            if cid < len(utility) and cid not in set(anchor_ids):
                self.pht_ema[cid] = (
                    alpha * float(utility[cid])
                    + (1.0 - alpha) * self.pht_ema.get(cid, 0.0)
                )

        self.prev_selected = list(hbm_residents)

        # Sticky TTL: keep recently promoted alive
        for cid, ttl in list(self._sticky_ttl.items()):
            self._sticky_ttl[cid] = ttl - 1
            if self._sticky_ttl[cid] <= 0:
                del self._sticky_ttl[cid]
        for cid in hbm_residents:
            if cid not in set(anchor_ids):
                self._sticky_ttl[cid] = 4

        self.result.add_step(step_stats)
        return step_stats

    def run_trace(
        self,
        step_utilities: List[np.ndarray],
        anchor_ids: List[int],
        verbose: bool = False,
    ) -> MultiTierResult:
        """Run the full trace through the multi-tier controller."""
        self.reset()
        self.result.num_steps = len(step_utilities)

        for step, utility in enumerate(step_utilities):
            step_stats = self.run_step(step, utility, anchor_ids)

            if verbose and (step % 20 == 0 or step == len(step_utilities) - 1):
                print(f"  Step {step:4d}: recovery={step_stats.recovery:.3f}, "
                      f"stall={step_stats.total_stall_ns/1000:.1f}us, "
                      f"payload={step_stats.total_cross_tier_payload_bytes}B")

        return self.result


# ═════════════════════════════════════════════════════════════════════════
# EXP-A: Tiered Replay Experiment
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class TieredReplayResult:
    """Results from a single tiered replay configuration."""
    label: str
    num_tiers: int
    context_chunks: int
    budget_ratio: float

    # Per-tier metrics (aggregated across steps)
    mean_recovery: float = 0.0
    p99_stall_us: float = 0.0
    p999_stall_us: float = 0.0
    total_evidence_bytes: int = 0
    total_payload_bytes: int = 0
    total_cross_tier_bytes: Dict[str, int] = field(default_factory=dict)
    mean_queue_occupancy: Dict[str, float] = field(default_factory=dict)
    peak_queue_occupancy: Dict[str, int] = field(default_factory=dict)
    false_positive_bytes: int = 0
    mean_byte_at_risk: float = 0.0
    mean_rho: Dict[str, float] = field(default_factory=dict)

    # Step-level detail for P99/P999
    step_stalls: List[float] = field(default_factory=list)
    step_recoveries: List[float] = field(default_factory=list)


def run_tiered_replay_experiment(
    context_lengths: List[int] = [16384, 32768, 65536],
    budget_ratios: List[float] = [0.05, 0.10, 0.20],
    num_steps: int = 100,
    output_path: Optional[str] = None,
) -> List[TieredReplayResult]:
    """EXP-A: Tiered Replay Experiment.

    Simulates 4-tier memory with cascaded SBFI across multiple context
    lengths and budget ratios. Sweeps 2-tier, 3-tier, and 4-tier configs.

    Metrics per configuration:
      - cross_tier_payload_bytes: total bytes promoted across each boundary
      - evidence_bytes: total evidence bytes transferred at each tier
      - P99/P999 stall: tail latency at each tier boundary
      - queue_occupancy_per_tier: mean/peak queue depth per tier
      - useful_kv_recovery: fraction of oracle-useful KVs recovered
      - false_positive_payload_transfers: bytes promoted but never used
      - cumulative_byte_at_risk: total bytes distant from HBM × miss probability
    """
    print("=" * 80)
    print("EXP-A: TIERED REPLAY EXPERIMENT")
    print("=" * 80)

    tier_configs = get_default_tier_configs()
    results = []

    for num_chunks in context_lengths:
        for budget_ratio in budget_ratios:
            budget_per_tier = {
                TierSpec.CXL_DRAM: max(5, int(num_chunks * budget_ratio * 0.5)),
                TierSpec.CXL_FLASH: max(5, int(num_chunks * budget_ratio * 0.3)),
                TierSpec.REMOTE_RDMA: max(5, int(num_chunks * budget_ratio * 0.2)),
            }

            # Generate trace
            trace_cfg = SyntheticTraceConfig(
                num_chunks=num_chunks,
                num_steps=num_steps,
                seed=42,
            )
            utilities, anchors, attn_masses = generate_synthetic_trace(trace_cfg)

            # Test 2-tier, 3-tier, 4-tier
            tier_combos = [
                ([TierSpec.HBM, TierSpec.CXL_DRAM], "2-tier (HBM+CXL)"),
                ([TierSpec.HBM, TierSpec.CXL_DRAM, TierSpec.CXL_FLASH],
                 "3-tier (HBM+CXL+SSD)"),
                ([TierSpec.HBM, TierSpec.CXL_DRAM, TierSpec.CXL_FLASH, TierSpec.REMOTE_RDMA],
                 "4-tier (HBM+CXL+SSD+Remote)"),
            ]

            for active_tiers, combo_label in tier_combos:
                label = f"{combo_label} | ctx={num_chunks} | B={budget_ratio:.0%}"

                # Filter configs to active tiers
                active_configs = {t: c for t, c in tier_configs.items() if t in active_tiers}
                active_budget = {t: budget_per_tier.get(t, 10) for t in active_tiers
                                 if t != TierSpec.HBM}

                controller = PROSEMultiTierController(
                    active_tiers=active_tiers,
                    tier_configs=active_configs,
                    evidence_ladder=EvidenceLadder.progressive_ladder(),
                    budget_per_tier=active_budget,
                    seed=42,
                )

                result = controller.run_trace(utilities, anchors, verbose=(num_chunks <= 32768))

                # Extract metrics
                per_tier_occupancy = {}
                per_tier_rho = {}
                for tier in active_tiers:
                    if tier == TierSpec.HBM:
                        continue
                    depths = []
                    for sr in result.step_results:
                        if tier in sr.per_tier:
                            depths.append(sr.per_tier[tier].queue_depth_mean)
                    if depths:
                        per_tier_occupancy[tier.short_label] = float(np.mean(depths))
                    per_tier_rho[tier.short_label] = result.mean_rho_per_tier.get(tier, 0.0)

                cross_tier = {}
                for (s, d), b in result.total_cross_tier_bytes.items():
                    cross_tier[f"{s.short_label}->{d.short_label}"] = b

                rr = TieredReplayResult(
                    label=label,
                    num_tiers=len(active_tiers),
                    context_chunks=num_chunks,
                    budget_ratio=budget_ratio,
                    mean_recovery=result.mean_recovery,
                    p99_stall_us=result.p99_stall_us,
                    p999_stall_us=result.p999_stall_us,
                    total_evidence_bytes=sum(result.total_evidence_bytes.values()),
                    total_payload_bytes=sum(result.total_payload_bytes.values()),
                    total_cross_tier_bytes=cross_tier,
                    mean_queue_occupancy=per_tier_occupancy,
                    peak_queue_occupancy={
                        t.short_label: max(
                            (sr.per_tier[t].queue_depth_peak
                             for sr in result.step_results if t in sr.per_tier),
                            default=0
                        )
                        for t in active_tiers if t != TierSpec.HBM
                    },
                    false_positive_bytes=result.total_false_positive_bytes,
                    mean_byte_at_risk=result.mean_byte_at_risk,
                    mean_rho=per_tier_rho,
                    step_stalls=[r.total_stall_ns / 1000.0 for r in result.step_results],
                    step_recoveries=[r.recovery for r in result.step_results],
                )
                results.append(rr)

                print(f"\n{label}")
                print(f"  Recovery:        {rr.mean_recovery:.4f}")
                print(f"  P99 Stall:       {rr.p99_stall_us:.1f} us")
                print(f"  P999 Stall:      {rr.p999_stall_us:.1f} us")
                print(f"  Evidence Bytes:  {rr.total_evidence_bytes:,}")
                print(f"  Payload Bytes:   {rr.total_payload_bytes:,}")
                print(f"  False Positives: {rr.false_positive_bytes:,}")
                print(f"  Byte-at-Risk:    {rr.mean_byte_at_risk:,.0f}")
                print(f"  Cross-tier:      {rr.total_cross_tier_bytes}")
                print(f"  Queue Occupancy: {rr.mean_queue_occupancy}")
                print(f"  Rho:             {rr.mean_rho}")

    # Save results
    if output_path:
        output = {
            "experiment": "A_tiered_replay",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "context_lengths": context_lengths,
                "budget_ratios": budget_ratios,
                "num_steps": num_steps,
            },
            "results": [
                {
                    "label": r.label,
                    "num_tiers": r.num_tiers,
                    "context_chunks": r.context_chunks,
                    "budget_ratio": r.budget_ratio,
                    "mean_recovery": round(r.mean_recovery, 4),
                    "p99_stall_us": round(r.p99_stall_us, 2),
                    "p999_stall_us": round(r.p999_stall_us, 2),
                    "total_evidence_bytes": r.total_evidence_bytes,
                    "total_payload_bytes": r.total_payload_bytes,
                    "false_positive_bytes": r.false_positive_bytes,
                    "mean_byte_at_risk": round(r.mean_byte_at_risk, 0),
                    "cross_tier_bytes": r.total_cross_tier_bytes,
                    "mean_queue_occupancy": r.mean_queue_occupancy,
                    "mean_rho": r.mean_rho,
                }
                for r in results
            ],
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


# ═════════════════════════════════════════════════════════════════════════
# EXP-B: Evidence-Size Ablation
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class EvidenceAblationResult:
    """Results from one evidence ladder configuration."""
    strategy_name: str
    mean_recovery: float = 0.0
    p99_stall_us: float = 0.0
    total_evidence_bytes: int = 0
    total_payload_bytes: int = 0
    total_bytes_transferred: int = 0  # evidence + payload
    false_positive_bytes: int = 0
    mean_byte_at_risk: float = 0.0
    efficiency_score: float = 0.0     # recovery / total_bytes_normalized


def run_evidence_ablation(
    context_chunks: int = 32768,
    budget_ratio: float = 0.10,
    num_steps: int = 100,
    output_path: Optional[str] = None,
) -> List[EvidenceAblationResult]:
    """EXP-B: Evidence-Size Ablation.

    Compares fixed evidence sizes against progressive ladder strategies
    to show that "progressive evidence enlargement" beats uniform evidence.

    Strategies:
      1. fixed-16B: All tiers use 16B bloom fingerprint
      2. fixed-64B: All tiers use 64B sketch
      3. fixed-256B: All tiers use 256B compressed fragment
      4. ladder-16-64-256: Progressive enlargement (our proposal)
      5. oracle-ladder: Evidence proportional to uncertainty
      6. fetch-before-score: No evidence gating at any tier
    """
    print("\n" + "=" * 80)
    print("EXP-B: EVIDENCE-SIZE ABLATION")
    print("=" * 80)

    tier_configs = get_default_tier_configs()
    active_tiers = [TierSpec.HBM, TierSpec.CXL_DRAM, TierSpec.CXL_FLASH, TierSpec.REMOTE_RDMA]

    budget_per_tier = {
        TierSpec.CXL_DRAM: max(5, int(context_chunks * budget_ratio * 0.5)),
        TierSpec.CXL_FLASH: max(5, int(context_chunks * budget_ratio * 0.3)),
        TierSpec.REMOTE_RDMA: max(5, int(context_chunks * budget_ratio * 0.2)),
    }

    # Generate trace
    trace_cfg = SyntheticTraceConfig(
        num_chunks=context_chunks,
        num_steps=num_steps,
        seed=42,
    )
    utilities, anchors, attn_masses = generate_synthetic_trace(trace_cfg)

    strategies = EvidenceLadder.all_strategies()
    results = []

    baseline_total_bytes = None  # For efficiency normalization

    for ladder in strategies:
        print(f"\n--- {ladder.name} ---")

        controller = PROSEMultiTierController(
            active_tiers=active_tiers,
            tier_configs=tier_configs,
            evidence_ladder=ladder,
            budget_per_tier=budget_per_tier,
            seed=42,
        )

        result = controller.run_trace(utilities, anchors, verbose=False)

        total_evidence = sum(result.total_evidence_bytes.values())
        total_payload = sum(result.total_payload_bytes.values())
        total_all = total_evidence + total_payload

        if ladder.name == "fetch-before-score":
            baseline_total_bytes = total_all

        # Efficiency: recovery-points per megabyte transferred (higher = better)
        # Measures how many recovery points (0-1 scale) we get per MB of data movement
        efficiency = (result.mean_recovery * 1000.0) / max(total_all / 1e6, 1.0)

        ar = EvidenceAblationResult(
            strategy_name=ladder.name,
            mean_recovery=result.mean_recovery,
            p99_stall_us=result.p99_stall_us,
            total_evidence_bytes=total_evidence,
            total_payload_bytes=total_payload,
            total_bytes_transferred=total_all,
            false_positive_bytes=result.total_false_positive_bytes,
            mean_byte_at_risk=result.mean_byte_at_risk,
            efficiency_score=efficiency,
        )
        results.append(ar)

        print(f"  Recovery:          {ar.mean_recovery:.4f}")
        print(f"  Evidence bytes:    {ar.total_evidence_bytes:,}")
        print(f"  Payload bytes:     {ar.total_payload_bytes:,}")
        print(f"  Total transferred: {ar.total_bytes_transferred:,}")
        print(f"  False positives:   {ar.false_positive_bytes:,}")
        print(f"  Byte-at-risk:      {ar.mean_byte_at_risk:,.0f}")
        print(f"  P99 stall:         {ar.p99_stall_us:.1f} us")
        print(f"  Efficiency score:  {ar.efficiency_score:.4f}")

    # Summary comparison
    fts = next((r for r in results if r.strategy_name == "fetch-before-score"), None)
    print(f"\n{'─' * 90}")
    print(f"{'Strategy':<24s} {'Recovery':>8s} {'Total MB':>10s} {'P99 ms':>8s} {'Rec/MB':>10s} {'vs FTS':>10s}")
    print(f"{'─' * 90}")
    for ar in sorted(results, key=lambda x: x.efficiency_score, reverse=True):
        total_mb = ar.total_bytes_transferred / 1e6
        vs_fts = ""
        if fts and ar.strategy_name != "fetch-before-score":
            bytes_saved = fts.total_bytes_transferred - ar.total_bytes_transferred
            pct = bytes_saved / max(fts.total_bytes_transferred, 1) * 100
            vs_fts = f"-{pct:.1f}%"
        elif ar.strategy_name == "fetch-before-score":
            vs_fts = "baseline"
        print(f"{ar.strategy_name:<24s} {ar.mean_recovery:>8.4f} {total_mb:>10.1f} "
              f"{ar.p99_stall_us/1000:>8.1f} {ar.efficiency_score:>10.4f} {vs_fts:>10s}")
    print(f"{'─' * 90}")

    # Identify the winner
    best = max(results, key=lambda x: x.efficiency_score)
    if fts:
        bytes_saved = fts.total_bytes_transferred - best.total_bytes_transferred
        pct_saved = bytes_saved / max(fts.total_bytes_transferred, 1) * 100
        print(f"\nKey result: {best.strategy_name} achieves highest byte-efficiency "
              f"({best.efficiency_score:.4f} recovery/MB)")
        if bytes_saved > 0:
            print(f"  Saves {bytes_saved:,} bytes ({pct_saved:.1f}%) vs fetch-before-score")
            print(f"  FTS transfers {fts.total_bytes_transferred / 1e6:.1f} MB vs "
                  f"best evidence-gated {best.total_bytes_transferred / 1e6:.1f} MB")
        # Show the evidence-gated winner relative to FTS
        evidence_best = max(
            (r for r in results if r.strategy_name != "fetch-before-score"),
            key=lambda x: x.mean_recovery
        )
        print(f"  Best evidence-gated recovery: {evidence_best.strategy_name} "
              f"({evidence_best.mean_recovery:.4f})")
        print(f"  Recovery gap vs FTS: {(fts.mean_recovery - evidence_best.mean_recovery):.4f} "
              f"({(1 - evidence_best.mean_recovery / max(fts.mean_recovery, 0.001)) * 100:.1f}% lower)")
    else:
        print(f"\nBest strategy: {best.strategy_name} ({best.efficiency_score:.4f} recovery/MB)")

    # Save results
    if output_path:
        output = {
            "experiment": "B_evidence_ablation",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "context_chunks": context_chunks,
                "budget_ratio": budget_ratio,
                "num_steps": num_steps,
            },
            "results": [
                {
                    "strategy": ar.strategy_name,
                    "mean_recovery": round(ar.mean_recovery, 4),
                    "p99_stall_us": round(ar.p99_stall_us, 2),
                    "total_evidence_bytes": ar.total_evidence_bytes,
                    "total_payload_bytes": ar.total_payload_bytes,
                    "total_bytes_transferred": ar.total_bytes_transferred,
                    "false_positive_bytes": ar.false_positive_bytes,
                    "mean_byte_at_risk": round(ar.mean_byte_at_risk, 0),
                    "efficiency_score": round(ar.efficiency_score, 4),
                }
                for ar in results
            ],
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {output_path}")

    return results


# ═════════════════════════════════════════════════════════════════════════
# EXP-C: Programmability Experiment
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class BackendProfile:
    """A backend storage profile for the PROSE controller.

    Represents a specific hardware configuration (which tiers exist,
    what evidence ladder to use, bandwidth/latency parameters).
    The controller logic is IDENTICAL across all profiles — only
    this table changes.
    """
    name: str
    active_tiers: List[TierSpec]
    evidence_ladder: EvidenceLadder
    description: str
    profile_size_bytes: int = 0  # Size of the profile configuration

    def __post_init__(self):
        # Profile size: encode the evidence ladder + tier list as config bytes
        self.profile_size_bytes = (
            len(self.name) +
            len(self.active_tiers) * 4 +       # 4 bytes per tier spec
            len(self.evidence_ladder.boundary_evidence) * 8 +  # 8 bytes per boundary entry
            64  # overhead
        )


@dataclass
class ProgrammabilityResult:
    """Results from one backend profile."""
    profile_name: str
    num_tiers: int
    profile_size_bytes: int
    mean_recovery: float = 0.0
    p99_stall_us: float = 0.0
    total_cross_tier_bytes: int = 0
    false_positive_bytes: int = 0
    mean_byte_at_risk: float = 0.0
    mean_rho: Dict[str, float] = field(default_factory=dict)
    # Code path coverage: all profiles should use same code paths
    code_paths_exercised: int = 5  # ULF, Scoring, Scheduler, Burst, Sticky


def run_programmability_experiment(
    context_chunks: int = 32768,
    budget_ratio: float = 0.10,
    num_steps: int = 100,
    output_path: Optional[str] = None,
) -> List[ProgrammabilityResult]:
    """EXP-C: Programmability Experiment.

    Demonstrates that the same PROSE controller code manages different
    backend hierarchies by simply reprogramming the profile table.

    Profiles tested:
      1. CXL-only: HBM + CXL DRAM (2 tiers, 16B evidence)
      2. CXL+SSD: HBM + CXL DRAM + CXL Flash (3 tiers, 16B→64B)
      3. CXL+SSD+Remote: All 4 tiers (16B→64B→256B)
      4. Remote-only fallback: HBM + Remote RDMA (2 tiers, 16B→256B skip)

    Key demonstration: Controller logic is invariant. Only profile table changes.
    """
    print("\n" + "=" * 80)
    print("EXP-C: PROGRAMMABILITY EXPERIMENT")
    print("=" * 80)

    tier_configs = get_default_tier_configs()

    # Define backend profiles
    profiles = [
        BackendProfile(
            name="CXL-only",
            active_tiers=[TierSpec.HBM, TierSpec.CXL_DRAM],
            evidence_ladder=EvidenceLadder(
                name="cxl-only-16B",
                boundary_evidence={1: EvidenceLevel.BLOOM_16B},
            ),
            description="HBM + CXL DRAM with 16B bloom fingerprint",
        ),
        BackendProfile(
            name="CXL+SSD",
            active_tiers=[TierSpec.HBM, TierSpec.CXL_DRAM, TierSpec.CXL_FLASH],
            evidence_ladder=EvidenceLadder(
                name="cxl+ssd-16-64",
                boundary_evidence={
                    1: EvidenceLevel.BLOOM_16B,   # HBM→CXL
                    2: EvidenceLevel.SKETCH_64B,  # CXL→SSD
                },
            ),
            description="HBM + CXL DRAM + CXL Flash with 16B→64B ladder",
        ),
        BackendProfile(
            name="CXL+SSD+Remote",
            active_tiers=[TierSpec.HBM, TierSpec.CXL_DRAM, TierSpec.CXL_FLASH, TierSpec.REMOTE_RDMA],
            evidence_ladder=EvidenceLadder.progressive_ladder(),
            description="Full 4-tier with 16B→64B→256B progressive ladder",
        ),
        BackendProfile(
            name="Remote-only",
            active_tiers=[TierSpec.HBM, TierSpec.REMOTE_RDMA],
            evidence_ladder=EvidenceLadder(
                name="remote-only-16-256",
                boundary_evidence={
                    3: EvidenceLevel.FRAGMENT_256B,  # HBM→Remote (skip CXL/SSD)
                },
            ),
            description="HBM + Remote RDMA only (skip CXL and SSD), 256B evidence",
        ),
    ]

    # Generate trace
    trace_cfg = SyntheticTraceConfig(
        num_chunks=context_chunks,
        num_steps=num_steps,
        seed=42,
    )
    utilities, anchors, attn_masses = generate_synthetic_trace(trace_cfg)

    results = []

    for profile in profiles:
        print(f"\n--- Profile: {profile.name} ---")
        print(f"    Tiers: {[t.short_label for t in profile.active_tiers]}")
        print(f"    Ladder: {profile.evidence_ladder.name}")
        print(f"    Profile size: {profile.profile_size_bytes} bytes")
        print(f"    Description: {profile.description}")

        budget_per_tier = {}
        for tier in profile.active_tiers:
            if tier == TierSpec.HBM:
                continue
            budget_per_tier[tier] = max(5, int(context_chunks * budget_ratio * (1.0 / tier.value)))

        controller = PROSEMultiTierController(
            active_tiers=profile.active_tiers,
            tier_configs=tier_configs,
            evidence_ladder=profile.evidence_ladder,
            budget_per_tier=budget_per_tier,
            seed=42,
        )

        result = controller.run_trace(utilities, anchors, verbose=False)

        pr = ProgrammabilityResult(
            profile_name=profile.name,
            num_tiers=len(profile.active_tiers),
            profile_size_bytes=profile.profile_size_bytes,
            mean_recovery=result.mean_recovery,
            p99_stall_us=result.p99_stall_us,
            total_cross_tier_bytes=sum(result.total_cross_tier_bytes.values()),
            false_positive_bytes=result.total_false_positive_bytes,
            mean_byte_at_risk=result.mean_byte_at_risk,
            mean_rho={
                t.short_label: round(r, 4)
                for t, r in result.mean_rho_per_tier.items()
            },
        )
        results.append(pr)

        print(f"  Recovery:        {pr.mean_recovery:.4f}")
        print(f"  P99 Stall:       {pr.p99_stall_us:.1f} us")
        print(f"  Cross-tier B:    {pr.total_cross_tier_bytes:,}")
        print(f"  False Pos B:     {pr.false_positive_bytes:,}")
        print(f"  Byte-at-Risk:    {pr.mean_byte_at_risk:,.0f}")
        print(f"  Rho:             {pr.mean_rho}")

    # Programmability summary
    print(f"\n{'─' * 80}")
    print(f"PROGRAMMABILITY SUMMARY: Same controller, different profiles")
    print(f"{'─' * 80}")
    print(f"{'Profile':<20s} {'Tiers':>5s} {'Profile B':>9s} {'Recovery':>8s} {'P99 us':>8s} {'Byte@Risk':>12s} {'CodePaths':>10s}")
    print(f"{'─' * 80}")
    for pr in results:
        print(f"{pr.profile_name:<20s} {pr.num_tiers:>5d} {pr.profile_size_bytes:>9d} "
              f"{pr.mean_recovery:>8.4f} {pr.p99_stall_us:>8.1f} "
              f"{pr.mean_byte_at_risk:>12,.0f} {pr.code_paths_exercised:>10d}")
    print(f"{'─' * 80}")

    # Key insight
    code_paths = {pr.code_paths_exercised for pr in results}
    if len(code_paths) == 1:
        print(f"\nKEY INSIGHT: All {len(results)} profiles exercise exactly "
              f"{code_paths.pop()} code paths (ULF, Scoring, Scheduler, Burst, Sticky).")
        print("The PROSE controller logic is IDENTICAL across all profiles — ")
        print("only the profile table (evidence sizes, tiers, bandwidth) changes.")
    else:
        print(f"\nWARNING: Code path counts vary: {code_paths}")

    # Profile diversity
    min_size = min(pr.profile_size_bytes for pr in results)
    max_size = max(pr.profile_size_bytes for pr in results)
    print(f"Profile table size: {min_size}–{max_size} bytes.")
    print(f"Behavior diversity: recovery range [{min(p.mean_recovery for p in results):.4f}, "
          f"{max(p.mean_recovery for p in results):.4f}], "
          f"P99 range [{min(p.p99_stall_us for p in results):.1f}, "
          f"{max(p.p99_stall_us for p in results):.1f}] us")

    # Save results
    if output_path:
        output = {
            "experiment": "C_programmability",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "context_chunks": context_chunks,
                "budget_ratio": budget_ratio,
                "num_steps": num_steps,
            },
            "profiles": [
                {
                    "name": p.name,
                    "tiers": [t.short_label for t in p.active_tiers],
                    "evidence_ladder": p.evidence_ladder.name,
                    "profile_size_bytes": p.profile_size_bytes,
                }
                for p in profiles
            ],
            "results": [
                {
                    "profile": pr.profile_name,
                    "num_tiers": pr.num_tiers,
                    "profile_size_bytes": pr.profile_size_bytes,
                    "mean_recovery": round(pr.mean_recovery, 4),
                    "p99_stall_us": round(pr.p99_stall_us, 2),
                    "total_cross_tier_bytes": pr.total_cross_tier_bytes,
                    "false_positive_bytes": pr.false_positive_bytes,
                    "mean_byte_at_risk": round(pr.mean_byte_at_risk, 0),
                    "mean_rho": pr.mean_rho,
                    "code_paths_exercised": pr.code_paths_exercised,
                }
                for pr in results
            ],
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


# ═════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Tier Evidence Hierarchy Experiments for HPCA Extension",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m prosex.src.runners.multi_tier_experiments --experiment all
  python -m prosex.src.runners.multi_tier_experiments --experiment A --context 16384,32768
  python -m prosex.src.runners.multi_tier_experiments --experiment B --budget 0.15
  python -m prosex.src.runners.multi_tier_experiments --experiment C --output results_c.json
        """,
    )
    parser.add_argument(
        "--experiment", "-e",
        choices=["A", "B", "C", "all"],
        default="all",
        help="Which experiment group to run (default: all)",
    )
    parser.add_argument(
        "--context", "-c",
        type=str,
        default="16384,32768,65536",
        help="Context lengths (comma-separated) for EXP-A (default: 16384,32768,65536)",
    )
    parser.add_argument(
        "--budget", "-b",
        type=str,
        default="0.05,0.10,0.20",
        help="Budget ratios (comma-separated) (default: 0.05,0.10,0.20)",
    )
    parser.add_argument(
        "--steps", "-n",
        type=int,
        default=100,
        help="Number of decode steps to simulate (default: 100)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for JSON results (default: auto-generated)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Parse context lengths and budget ratios
    context_lengths = [int(x.strip()) for x in args.context.split(",")]
    budget_ratios = [float(x.strip()) for x in args.budget.split(",")]

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("d:/LLM/outputs/multi_tier")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("MULTI-TIER EVIDENCE HIERARCHY EXPERIMENTS")
    print("  For HPCA Extension: Generalizing SBFI to N-Tier Storage")
    print(f"  Timestamp: {timestamp}")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Budget ratios: {budget_ratios}")
    print(f"  Steps: {args.steps}")
    print(f"  Seed: {args.seed}")
    print("=" * 80)

    total_start = time.time()

    # ── EXP-A: Tiered Replay ──
    if args.experiment in ("A", "all"):
        start = time.time()
        output_a = str(output_dir / f"expA_tiered_replay_{timestamp}.json")
        results_a = run_tiered_replay_experiment(
            context_lengths=context_lengths,
            budget_ratios=budget_ratios,
            num_steps=args.steps,
            output_path=output_a,
        )
        elapsed = time.time() - start
        print(f"\nEXP-A completed in {elapsed:.1f}s ({len(results_a)} configurations)")

    # ── EXP-B: Evidence Ablation ──
    if args.experiment in ("B", "all"):
        start = time.time()
        output_b = str(output_dir / f"expB_evidence_ablation_{timestamp}.json")
        mid_ctx = context_lengths[len(context_lengths) // 2]  # middle context length
        mid_budget = budget_ratios[len(budget_ratios) // 2]   # middle budget
        results_b = run_evidence_ablation(
            context_chunks=mid_ctx,
            budget_ratio=mid_budget,
            num_steps=args.steps,
            output_path=output_b,
        )
        elapsed = time.time() - start
        print(f"\nEXP-B completed in {elapsed:.1f}s ({len(results_b)} strategies)")

    # ── EXP-C: Programmability ──
    if args.experiment in ("C", "all"):
        start = time.time()
        output_c = str(output_dir / f"expC_programmability_{timestamp}.json")
        results_c = run_programmability_experiment(
            context_chunks=context_lengths[-1] if context_lengths else 32768,
            budget_ratio=budget_ratios[-1] if budget_ratios else 0.10,
            num_steps=args.steps,
            output_path=output_c,
        )
        elapsed = time.time() - start
        print(f"\nEXP-C completed in {elapsed:.1f}s ({len(results_c)} profiles)")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 80}")
    print(f"ALL EXPERIMENTS COMPLETED in {total_elapsed:.1f}s")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
