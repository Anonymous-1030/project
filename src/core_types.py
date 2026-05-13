"""
Core type definitions for ProSE-X 2.0.

These types are used across modules to ensure consistency.
All dataclasses are designed for:
1. Explicit field definitions
2. Serialization support
3. Auditability
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np
from datetime import datetime


# =============================================================================
# Enums
# =============================================================================

class ChunkTier(str, Enum):
    """Tier of a KV chunk in the retention hierarchy."""
    FULL = "FULL"
    ANCHOR = "ANCHOR"
    TAIL = "TAIL"
    ARCHIVE = "ARCHIVE"
    EVICTED = "EVICTED"
    PROMOTING = "PROMOTING"


class ChunkState(str, Enum):
    """State of a KV chunk."""
    ACTIVE = "ACTIVE"
    EVICTED = "EVICTED"
    PROMOTING = "PROMOTING"
    ACCESSED = "ACCESSED"
    STICKY = "STICKY"  # Sticky residency state


class PromoteDecision(str, Enum):
    """Promotion decision outcomes."""
    PROMOTE = "PROMOTE"
    SKIP = "SKIP"
    KEEP = "KEEP"
    DEFER = "DEFER"


class FailureReason(str, Enum):
    """
    Failure reason enumeration for failure attribution.

    Used to identify exactly why a gold chunk was missed.
    """
    # ULF stage failures
    CANDIDATE_RECALL_FAILURE = "candidate_recall_failure"
    QUEUE_MISS = "queue_miss"  # Not recalled by any queue

    # Scoring stage failures
    SCORING_RANKING_FAILURE = "scoring_ranking_failure"
    LOW_SCORE = "low_score"

    # Scheduler stage failures
    SCHEDULER_BUDGET_CUT = "scheduler_budget_cut"
    LOW_CONFIDENCE = "low_confidence"
    SCORE_THRESHOLD_MISS = "score_threshold_miss"

    # Burst stage failures
    BURST_WINDOW_MISS = "burst_window_miss"

    # Sticky stage failures
    STICKY_EVICTION = "sticky_eviction"
    TTL_EXPIRED = "ttl_expired"

    # Exploration failures
    EXPLORATION_MISS = "exploration_miss"

    # Success
    RECOVERED = "recovered"
    UNKNOWN = "unknown"


# =============================================================================
# Causal Verification Enums (Seven-Layer Framework)
# =============================================================================

class CausalInterventionType(str, Enum):
    """Types of counterfactual interventions for Layer 1 CEI."""
    FIX_TO_MEAN = "fix_to_mean"
    SWAP = "swap"
    GHOST_SYNTHESIZE = "ghost_synthesize"


class EvidenceDimension(str, Enum):
    """
    The 5 dimensions of the decomposed evidence vector E(c).

    Decomposed from ODUS-X's 11-cue scoring into causally distinct groups:
    - TEMPORAL: recency, EWMA, window-buffer (locality-driven)
    - STRUCTURAL: anchor distance, position, section boundary, title adjacency
    - SEMANTIC: query-chunk similarity, lexical overlap
    - HISTORICAL: promotion history, PHT, anchor bonus, promoted distance
    - PRESSURE: budget pressure contribution to score
    """
    TEMPORAL = "e_temp"
    STRUCTURAL = "e_struct"
    SEMANTIC = "e_sem"
    HISTORICAL = "e_hist"
    PRESSURE = "e_press"


class QuadrantLabel(str, Enum):
    """4 quadrants from the Query-Utility Disentanglement Matrix (Layer 2)."""
    HIGH_R_HIGH_U = "high_reuse_high_utility"     # True hot blocks
    HIGH_R_LOW_U = "high_reuse_low_utility"        # Locality trap
    LOW_R_HIGH_U = "low_reuse_high_utility"        # Long-range dependency
    LOW_R_LOW_U = "low_reuse_low_utility"          # True cold blocks


class BanditAlgorithm(str, Enum):
    """Bandit algorithms for Layer 7 Online Causal Adaptation."""
    LINUCB = "linucb"
    THOMPSON_SAMPLING = "thompson_sampling"
    EPSILON_GREEDY = "epsilon_greedy"


class DecodePhase(str, Enum):
    """Decode phase for phase-consistency checks across interventions."""
    PREFILL = "prefill"
    EARLY_DECODE = "early_decode"
    MID_DECODE = "mid_decode"
    LATE_DECODE = "late_decode"


# =============================================================================
# Chunk and Request Types
# =============================================================================

@dataclass
class ChunkMetadata:
    """
    Metadata for a single chunk.
    
    Contains all runtime-available information about a chunk.
    NO oracle information is stored here.
    """
    chunk_id: str
    request_id: str
    
    # Position
    token_start: int
    token_end: int
    position_ratio: float  # Position in sequence [0, 1]
    
    # Size
    num_tokens: int
    logical_bytes: int
    
    # Content signature (for overlap computation)
    signature: Optional[np.ndarray] = None
    signature_hex: Optional[str] = None
    
    # Structural markers
    is_section_boundary: bool = False
    is_title_adjacent: bool = False
    is_code_block: bool = False
    section_id: Optional[str] = None
    
    # Current state
    tier: ChunkTier = ChunkTier.TAIL
    state: ChunkState = ChunkState.ACTIVE
    
    # History
    creation_step: int = 0
    last_access_step: int = -1
    last_promotion_step: int = -1
    promoted_count: int = 0
    access_count: int = 0
    
    # Sticky residency
    sticky_ttl: int = 0  # Current TTL
    sticky_original_ttl: int = 0  # TTL when first promoted
    
    # Promotion status
    @property
    def is_promoted(self) -> bool:
        """Whether chunk is currently promoted (sticky and active)."""
        return self.sticky_ttl > 0 and self.state == ChunkState.ACTIVE
    
    # Extra metadata
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (numpy-free)."""
        return {
            "chunk_id": self.chunk_id,
            "request_id": self.request_id,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "position_ratio": self.position_ratio,
            "num_tokens": self.num_tokens,
            "logical_bytes": self.logical_bytes,
            "signature_hex": self.signature_hex,
            "is_section_boundary": self.is_section_boundary,
            "is_title_adjacent": self.is_title_adjacent,
            "is_code_block": self.is_code_block,
            "section_id": self.section_id,
            "tier": self.tier.value,
            "state": self.state.value,
            "creation_step": self.creation_step,
            "last_access_step": self.last_access_step,
            "last_promotion_step": self.last_promotion_step,
            "promoted_count": self.promoted_count,
            "access_count": self.access_count,
            "sticky_ttl": self.sticky_ttl,
            "sticky_original_ttl": self.sticky_original_ttl,
        }


@dataclass
class QueryContext:
    """
    Query-side context available at runtime.
    
    NO future information or gold labels.
    """
    request_id: str
    step: int
    
    # Query representation
    query_summary: Optional[np.ndarray] = None  # Lightweight embedding
    query_tokens: Optional[List[int]] = None  # Token IDs
    query_text: Optional[str] = None  # Raw text (if available)
    
    # Query features
    query_signature: Optional[np.ndarray] = None
    extracted_entities: Optional[List[str]] = None
    query_length: int = 0
    
    # Active anchors at this step
    active_anchor_ids: List[str] = field(default_factory=list)
    recent_anchor_ids: List[str] = field(default_factory=list)
    
    # Runtime stats
    steps_since_start: int = 0
    generation_length: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "request_id": self.request_id,
            "step": self.step,
            "query_length": self.query_length,
            "active_anchor_ids": self.active_anchor_ids,
            "recent_anchor_ids": self.recent_anchor_ids,
            "steps_since_start": self.steps_since_start,
            "generation_length": self.generation_length,
        }


# =============================================================================
# Pipeline Stage Results
# =============================================================================

@dataclass
class QueueContribution:
    """Contribution from a single recall queue."""
    queue_name: str
    candidate_ids: List[str]
    candidate_scores: List[float]
    
    def __len__(self) -> int:
        return len(self.candidate_ids)


@dataclass
class ULFResult:
    """Result from Multi-Queue Recall ULF."""
    request_id: str
    step: int
    
    # Union candidate set
    candidate_ids: List[str]
    candidate_sources: Dict[str, str]  # chunk_id -> queue_name (primary source)
    
    # Per-queue contributions (for attribution)
    queue_contributions: List[QueueContribution]
    
    # Statistics
    n_tail_total: int
    n_candidates: int
    per_queue_counts: Dict[str, int]
    per_queue_unique: Dict[str, int]  # Unique contributions per queue
    
    # Latency
    ulf_latency_us: float
    
    # Optional
    queue_overlap_matrix: Optional[Dict[str, Dict[str, int]]] = None
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "request_id": self.request_id,
            "step": self.step,
            "candidate_ids": self.candidate_ids,
            "candidate_sources": self.candidate_sources,
            "n_tail_total": self.n_tail_total,
            "n_candidates": self.n_candidates,
            "per_queue_counts": self.per_queue_counts,
            "per_queue_unique": self.per_queue_unique,
            "ulf_latency_us": self.ulf_latency_us,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ScoredCandidate:
    """A candidate with its utility score."""
    chunk_id: str
    score: float
    confidence: float
    feature_vector: Optional[np.ndarray] = None
    
    # Score components (for interpretability)
    score_components: Optional[Dict[str, float]] = None


@dataclass
class ScorerResult:
    """Result from utility scoring."""
    request_id: str
    step: int
    
    # Scored candidates (sorted by score descending)
    candidates: List[ScoredCandidate]
    
    # Statistics
    n_input_candidates: int
    n_scored: int
    n_above_threshold: int
    score_threshold: float
    
    # Mode info
    scorer_mode: str
    
    # Latency
    scorer_latency_us: float
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "request_id": self.request_id,
            "step": self.step,
            "chunk_ids": [c.chunk_id for c in self.candidates],
            "scores": [c.score for c in self.candidates],
            "confidences": [c.confidence for c in self.candidates],
            "n_input_candidates": self.n_input_candidates,
            "n_scored": self.n_scored,
            "n_above_threshold": self.n_above_threshold,
            "score_threshold": self.score_threshold,
            "scorer_mode": self.scorer_mode,
            "scorer_latency_us": self.scorer_latency_us,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SchedulerDecision:
    """Decision for a single chunk."""
    chunk_id: str
    decision: PromoteDecision
    score: float
    confidence: float
    rejection_reason: Optional[FailureReason] = None
    selection_type: Optional[str] = None  # "exploit" or "explore"


@dataclass
class SchedulerResult:
    """Result from EABS scheduler."""
    request_id: str
    step: int
    
    # Selected for promotion
    selected_ids: List[str]
    selected_decisions: List[SchedulerDecision]
    
    # Exploit vs explore split
    exploit_ids: List[str]
    explore_ids: List[str]
    
    # Dropped
    dropped_ids: List[str]
    dropped_decisions: List[SchedulerDecision]
    
    # Budget
    budget_bytes: int
    used_bytes: int
    utilization: float
    
    # Statistics
    n_exploit: int
    n_explore: int
    n_dropped_budget: int
    n_dropped_low_score: int
    n_dropped_low_confidence: int
    
    # Latency
    scheduler_latency_us: float
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "request_id": self.request_id,
            "step": self.step,
            "selected_ids": self.selected_ids,
            "exploit_ids": self.exploit_ids,
            "explore_ids": self.explore_ids,
            "dropped_ids": self.dropped_ids,
            "budget_bytes": self.budget_bytes,
            "used_bytes": self.used_bytes,
            "utilization": self.utilization,
            "n_exploit": self.n_exploit,
            "n_explore": self.n_explore,
            "n_dropped_budget": self.n_dropped_budget,
            "n_dropped_low_score": self.n_dropped_low_score,
            "n_dropped_low_confidence": self.n_dropped_low_confidence,
            "scheduler_latency_us": self.scheduler_latency_us,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BurstResult:
    """Result from burst expansion."""
    request_id: str
    step: int
    
    # Input chunks from scheduler
    input_ids: List[str]
    
    # Expanded burst set
    burst_ids: List[str]  # All chunks in burst window
    core_ids: List[str]  # Original input chunks
    expansion_ids: List[str]  # Added by burst
    
    # Per-chunk burst info
    burst_radius: Dict[str, int]  # chunk_id -> radius from core
    
    # Statistics
    n_input: int
    n_burst_total: int
    n_expansion: int
    
    # Latency
    burst_latency_us: float
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "request_id": self.request_id,
            "step": self.step,
            "input_ids": self.input_ids,
            "burst_ids": self.burst_ids,
            "core_ids": self.core_ids,
            "expansion_ids": self.expansion_ids,
            "n_input": self.n_input,
            "n_burst_total": self.n_burst_total,
            "n_expansion": self.n_expansion,
            "burst_latency_us": self.burst_latency_us,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class StickyUpdate:
    """Single sticky residency update."""
    chunk_id: str
    old_ttl: int
    new_ttl: int
    update_type: str  # "new", "refresh", "decay", "expire"


@dataclass
class StickyResult:
    """Result from sticky TTL management."""
    request_id: str
    step: int
    
    # Promoted chunks with TTL
    promoted_ids: List[str]
    ttl_values: Dict[str, int]  # chunk_id -> current TTL
    
    # TTL updates this step
    ttl_updates: List[StickyUpdate]
    
    # Expired chunks
    expired_ids: List[str]
    
    # Statistics
    n_promoted: int
    n_expired: int
    n_refreshed: int
    avg_ttl: float
    
    # Latency
    sticky_latency_us: float
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "request_id": self.request_id,
            "step": self.step,
            "promoted_ids": self.promoted_ids,
            "ttl_values": self.ttl_values,
            "expired_ids": self.expired_ids,
            "n_promoted": self.n_promoted,
            "n_expired": self.n_expired,
            "n_refreshed": self.n_refreshed,
            "avg_ttl": self.avg_ttl,
            "sticky_latency_us": self.sticky_latency_us,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# End-to-End Pipeline Result
# =============================================================================

@dataclass
class PromotionPipelineResult:
    """
    Complete result from the promotion pipeline.
    
    Contains all intermediate results for full auditability.
    """
    request_id: str
    step: int
    
    # Stage results
    ulf_result: Optional[ULFResult] = None
    scorer_result: Optional[ScorerResult] = None
    scheduler_result: Optional[SchedulerResult] = None
    burst_result: Optional[BurstResult] = None
    sticky_result: Optional[StickyResult] = None
    
    # Final visible set
    final_visible_ids: List[str] = field(default_factory=list)
    final_active_bytes: int = 0
    final_promoted_bytes: int = 0
    
    # Timing
    total_latency_us: float = 0.0

    # PPU hardware accelerator results (populated when PPU pipeline is used)
    ppu_trace: Optional[Any] = None
    ppu_area_power: Optional[Any] = None

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        result = {
            "request_id": self.request_id,
            "step": self.step,
            "ulf_result": self.ulf_result.to_dict() if self.ulf_result else None,
            "scorer_result": self.scorer_result.to_dict() if self.scorer_result else None,
            "scheduler_result": self.scheduler_result.to_dict() if self.scheduler_result else None,
            "burst_result": self.burst_result.to_dict() if self.burst_result else None,
            "sticky_result": self.sticky_result.to_dict() if self.sticky_result else None,
            "final_visible_ids": self.final_visible_ids,
            "final_active_bytes": self.final_active_bytes,
            "final_promoted_bytes": self.final_promoted_bytes,
            "total_latency_us": self.total_latency_us,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.ppu_trace is not None:
            result["ppu_trace"] = self.ppu_trace.to_dict() if hasattr(self.ppu_trace, "to_dict") else str(self.ppu_trace)
        if self.ppu_area_power is not None:
            result["ppu_area_power"] = self.ppu_area_power.to_dict() if hasattr(self.ppu_area_power, "to_dict") else str(self.ppu_area_power)
        return result


# =============================================================================
# Causal Verification Dataclasses (Seven-Layer Framework)
# =============================================================================

@dataclass
class EvidenceVector:
    """
    The 5-dim evidence vector E(c) decomposed from ODUS-X scoring.

    Decomposed from the 11-cue scoring algebra in AdaptiveGatingScorer
    into causally distinct groups. Each dimension is the weighted
    contribution to the final score from that evidence category.
    """
    chunk_id: str
    e_temp: float = 0.0
    e_struct: float = 0.0
    e_sem: float = 0.0
    e_hist: float = 0.0
    e_press: float = 0.0
    score: float = 0.0
    mode: str = "stable"

    def to_array(self) -> "np.ndarray":
        import numpy as np
        return np.array(
            [self.e_temp, self.e_struct, self.e_sem, self.e_hist, self.e_press],
            dtype=np.float32,
        )

    def dominant_dimension(self) -> EvidenceDimension:
        dims = {
            EvidenceDimension.TEMPORAL: self.e_temp,
            EvidenceDimension.STRUCTURAL: self.e_struct,
            EvidenceDimension.SEMANTIC: self.e_sem,
            EvidenceDimension.HISTORICAL: self.e_hist,
            EvidenceDimension.PRESSURE: self.e_press,
        }
        return max(dims, key=dims.get)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "e_temp": self.e_temp,
            "e_struct": self.e_struct,
            "e_sem": self.e_sem,
            "e_hist": self.e_hist,
            "e_press": self.e_press,
            "score": self.score,
            "mode": self.mode,
        }


@dataclass
class InterventionResult:
    """Result of a single counterfactual intervention (Layer 1 CEI)."""
    intervention_type: CausalInterventionType
    dimension: EvidenceDimension
    baseline_admission_rate: float
    intervention_admission_rate: float
    delta_admission_rate: float
    num_chunks: int
    consistent_across_phases: bool
    phase_breakdown: Dict[str, float] = field(default_factory=dict)
    pass_fail: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention_type": self.intervention_type.value,
            "dimension": self.dimension.value,
            "baseline_admission_rate": self.baseline_admission_rate,
            "intervention_admission_rate": self.intervention_admission_rate,
            "delta_admission_rate": self.delta_admission_rate,
            "num_chunks": self.num_chunks,
            "consistent_across_phases": self.consistent_across_phases,
            "phase_breakdown": self.phase_breakdown,
            "pass_fail": self.pass_fail,
        }


@dataclass
class QuadrantMetrics:
    """Quadrant-level metrics from QUDM (Layer 2)."""
    quadrant: QuadrantLabel
    num_chunks: int
    admission_rate: float
    avg_utility: float
    avg_reuse_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quadrant": self.quadrant.value,
            "num_chunks": self.num_chunks,
            "admission_rate": self.admission_rate,
            "avg_utility": self.avg_utility,
            "avg_reuse_score": self.avg_reuse_score,
        }


@dataclass
class BudgetProjectionResult:
    """Evidence Budget-Query Projection Tradeoff sweep result (Layer 3)."""
    budget_b_kv: int
    budget_b_q: int
    recovery: float
    qcdr: float
    long_range_hit_rate: float
    passkey_recovery: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "budget_b_kv": self.budget_b_kv,
            "budget_b_q": self.budget_b_q,
            "recovery": self.recovery,
            "qcdr": self.qcdr,
            "long_range_hit_rate": self.long_range_hit_rate,
            "passkey_recovery": self.passkey_recovery,
        }


@dataclass
class InformationBottleneckResult:
    """Information-Theoretic Lower Bound Probing result (Layer 4)."""
    optimal_encoding_bits: int
    achieved_mi: float
    saturation_entropy: float
    budget_recovery_curve: Dict[int, float] = field(default_factory=dict)
    is_64b_sufficient: bool = False
    hard_capacity_bits: int = 435  # 512 * 0.85

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimal_encoding_bits": self.optimal_encoding_bits,
            "achieved_mi": self.achieved_mi,
            "saturation_entropy": self.saturation_entropy,
            "budget_recovery_curve": self.budget_recovery_curve,
            "is_64b_sufficient": self.is_64b_sufficient,
            "hard_capacity_bits": self.hard_capacity_bits,
        }


@dataclass
class SpoofingResult:
    """Adversarial Causal Spoofing result (Layer 5)."""
    base_recall: float
    query_aware_recall: float
    oracle_recall: float
    base_ssr: float = 0.0
    query_aware_ssr: float = 0.0
    oracle_ssr: float = 0.0
    cvi: float = 0.0
    spoof_type: str = ""
    num_adversarial_samples: int = 0
    pass_fail: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_recall": self.base_recall,
            "query_aware_recall": self.query_aware_recall,
            "oracle_recall": self.oracle_recall,
            "base_ssr": self.base_ssr,
            "query_aware_ssr": self.query_aware_ssr,
            "oracle_ssr": self.oracle_ssr,
            "cvi": self.cvi,
            "spoof_type": self.spoof_type,
            "num_adversarial_samples": self.num_adversarial_samples,
            "pass_fail": self.pass_fail,
        }


@dataclass
class CrossArchitectureResult:
    """Cross-Architectural Causal Transfer result (Layer 6)."""
    architecture: str
    ace_vector: Dict[str, float] = field(default_factory=dict)
    cec_vs_mha: float = 0.0
    dimension_consistencies: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "ace_vector": self.ace_vector,
            "cec_vs_mha": self.cec_vs_mha,
            "dimension_consistencies": self.dimension_consistencies,
        }


@dataclass
class OnlineBanditResult:
    """Online Causal Adaptation result (Layer 7)."""
    algorithm: BanditAlgorithm
    cumulative_reward: float
    cumulative_regret: float
    commit_abort_ratio: List[float] = field(default_factory=list)
    drift_events: List[int] = field(default_factory=list)
    final_weights: Dict[str, float] = field(default_factory=dict)
    sbfi_boundary_violations: int = 0
    pass_fail: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm.value,
            "cumulative_reward": self.cumulative_reward,
            "cumulative_regret": self.cumulative_regret,
            "commit_abort_ratio": self.commit_abort_ratio,
            "drift_events": self.drift_events,
            "final_weights": self.final_weights,
            "sbfi_boundary_violations": self.sbfi_boundary_violations,
            "pass_fail": self.pass_fail,
        }


@dataclass
class CausalVerificationReport:
    """
    Aggregate report across all 7 causal verification layers.

    Collects results, pass/fail per layer, and critical findings.
    """
    experiment_id: str = ""
    layer_results: Dict[int, Any] = field(default_factory=dict)
    pass_fail_summary: Dict[int, bool] = field(default_factory=dict)
    critical_findings: List[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(self.pass_fail_summary.values()) if self.pass_fail_summary else False

    @property
    def n_passed(self) -> int:
        return sum(1 for v in self.pass_fail_summary.values() if v)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "layer_results": {
                str(k): v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self.layer_results.items()
            },
            "pass_fail_summary": {str(k): v for k, v in self.pass_fail_summary.items()},
            "critical_findings": self.critical_findings,
            "all_passed": self.all_passed,
            "n_passed": self.n_passed,
        }
