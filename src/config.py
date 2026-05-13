"""
Configuration schemas for ProSE-X 2.0.

All parameters must be explicit - no magic constants in code.
Every threshold, top-k, TTL, burst radius, exploration ratio must be configurable here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ScorerMode(str, Enum):
    """Scorer mode enumeration."""
    SIMILARITY_BASELINE = "similarity_baseline"
    LIGHTWEIGHT_FEATURE_MLP = "lightweight_feature_mlp"
    ORACLE_DISTILLED_UTILITY = "oracle_distilled_utility"
    ODUS_X = "odus_x"  # Adaptive gating scorer — no training, no oracle


class ULFMode(str, Enum):
    """ULF mode enumeration."""
    OLD_SINGLE_SCORE = "old_single_score"  # Baseline for comparison
    MULTI_QUEUE_RECALL = "multi_queue_recall"  # MQR-ULF


class SchedulerMode(str, Enum):
    """Scheduler mode enumeration."""
    DETERMINISTIC = "deterministic"  # Baseline
    EABS = "eabs"  # Exploration-Aware Budget Scheduler


@dataclass
class MQRULFConfig:
    """
    Multi-Queue Recall ULF Configuration.
    
    ULF objective is RECALL, not precision.
    Each queue contributes a configurable number of candidates.
    Final candidate set is union + dedup + stable ordering.

    === COMPUTATION OVERHEAD ANALYSIS (§2.x) ===
    Q: What is the per-step computational overhead of MQR-ULF?
    A: NEGLIGIBLE. MQR-ULF is a pure heuristic pre-filter:
    - Anchor-neighbor queue: O(|anchors| × radius) — simple position lookup
    - Lexical overlap queue: O(|tail_chunks| × 128) — cached dot-product
      (computed ONCE at prefill, reused during decode)
    - Structural/recency queue: O(|tail_chunks|) — metadata field check
    - Historical-success queue: O(|history|) — counter lookup

    Total: ~12K FLOPs per step for 100 tail chunks × 128-dim signatures.
    Compare: a single attention layer = ~1.7 GFLOPs for 1K tokens.
    MQR-ULF overhead < 0.001% of attention compute. NOT a bottleneck.

    The lexical overlap queue does NOT require real-time embedding computation.
    All signatures are pre-computed at chunking time (chunk.signature) and
    prefill time (query.query_signature), using fixed-size hash vectors.
    """
    # Mode
    mode: str = "multi_queue_recall"
    
    # Queue configurations - each queue contributes top_k candidates
    # PATCHED (2026-03-24): Increased quotas for better recall
    
    # Queue 1: Anchor-neighbor recall
    anchor_neighbor_enabled: bool = True
    anchor_neighbor_radius: int = 2  # Adjacent chunks within this radius
    anchor_neighbor_top_k: int = 2   # PATCHED: minimum 2 neighbors
    
    # Queue 2: Lexical/entity overlap recall
    # OVERHEAD: O(|tail_chunks| × 128) per PREFILL step only.
    # Decode steps reuse cached scores. No real-time embedding compute.
    lexical_overlap_enabled: bool = True
    lexical_overlap_method: str = "hashed_token"  # "hashed_token", "ngram_bloom", "entity_hash"
    lexical_overlap_top_k: int = 4   # PATCHED: minimum 4 from overlap
    lexical_overlap_threshold: float = 0.0  # PATCHED: lowered to 0.0 for more candidates
    lexical_overlap_timing_mode: str = "prefill_only"  # "prefill_only" (cached) or "per_step"
    
    # Queue 3: Structural/recency recall
    structural_recency_enabled: bool = True
    structural_recent_top_k: int = 2   # PATCHED: was 3
    structural_boundary_top_k: int = 2  # Section boundaries
    structural_title_adjacent_top_k: int = 2  # Title-adjacent chunks
    structural_recency_window: int = 100  # Steps considered "recent"
    
    # Queue 4: Historical-success recall
    historical_success_enabled: bool = True
    historical_success_top_k: int = 2   # PATCHED: minimum 2 from history
    historical_success_min_count: int = 1  # Minimum successful promotions
    
    # Union and dedup
    max_total_candidates: int = 20
    min_total_candidates: int = 1
    dedup_method: str = "chunk_id"  # How to deduplicate across queues
    ordering_method: str = "queue_priority"  # How to order after union
    
    # Logging
    log_per_queue_contribution: bool = True
    log_queue_overlap: bool = True


@dataclass
class BurstConfig:
    """
    Burst-and-Stick Promotion Configuration.
    
    Selected chunks expand to a local burst window before transfer.
    Promoted chunks persist with sticky residency and TTL.
    """
    # Burst expansion
    enabled: bool = True
    radius: int = 1  # 0=no burst, 1=radius 1, 2=radius 2
    
    # Burst selection strategy
    burst_selection: str = "contiguous"  # "contiguous", "symmetric"
    
    # Sticky residency
    sticky_enabled: bool = True
    default_ttl: int = 4  # Minimum steps to persist unless evicted
    ttl_refresh_policy: str = "access"  # "access", "recompute", "none"
    ttl_refresh_max: int = 8  # Maximum TTL after refresh
    
    # TTL decay
    enable_ttl_decay: bool = False
    ttl_decay_rate: float = 0.9
    
    # Logging
    log_burst_expansion: bool = True
    log_ttl_updates: bool = True


@dataclass
class ODUSConfig:
    """
    Utility Scorer Configuration.

    Four modes:
    1. odus_x: Adaptive gating scorer — NO training, NO oracle (default)
    2. similarity_baseline: Query-chunk similarity only
    3. lightweight_feature_mlp: Small MLP on runtime features
    4. oracle_distilled_utility: Offline-trained utility predictor

    === TRAIN/TEST SPLIT POLICY (§4 Oracle Leakage Prevention) ===
    To prevent oracle leakage, ODUS training uses strict cross-domain splits:
    - Training data comes from one set of domains (e.g., PG19 books)
    - Test data comes from DIFFERENT domains (e.g., LongBench legal/papers)
    - This ensures ODUS learns generalizable attention patterns, not
      domain-specific artifacts.

    The split_mode field controls how labels are partitioned:
    - "random": Simple random split (old behavior, NOT recommended for paper)
    - "domain": Split by domain (request_id prefix), ensuring no domain overlap
    - "temporal": Split by time (earlier = train, later = test)
    """
    mode: ScorerMode = ScorerMode.ODUS_X
    
    # Feature configuration (for MLP and ODUS modes)
    feature_use_query_summary: bool = True
    feature_use_chunk_summary: bool = True
    feature_use_anchor_distance: bool = True
    feature_use_lexical_overlap: bool = True
    feature_use_recent_usage: bool = True
    feature_use_structural_markers: bool = True
    feature_use_promoted_adjacency: bool = True
    
    # MLP architecture (for lightweight_feature_mlp mode)
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    mlp_activation: str = "relu"
    mlp_dropout: float = 0.1
    
    # ODUS training (for oracle_distilled_utility mode)
    odus_weights_path: Optional[str] = None
    odus_embedding_dim: int = 128
    odus_num_heads: int = 4
    
    # Scoring
    score_temperature: float = 1.0
    score_calibration: str = "none"  # "none", "platt_scaling", "temperature"
    
    # Runtime checks
    assert_no_oracle_at_runtime: bool = True

    # === Cross-domain split configuration ===
    # split_mode: How to partition labels into train/test
    #   "random"  — simple random 80/20 (legacy, not cross-domain)
    #   "domain"  — split by request_id domain prefix (e.g., "pg19_" vs "legal_")
    #   "temporal" — split by time (earlier requests = train, later = test)
    split_mode: str = "domain"
    # train_domains: List of domain prefixes for training (used when split_mode="domain")
    # If empty, auto-detect from request_id prefixes (80% domains → train)
    train_domains: List[str] = field(default_factory=list)
    # test_domains: List of domain prefixes for testing (used when split_mode="domain")
    # If empty, auto-detect from request_id prefixes (20% domains → test)
    test_domains: List[str] = field(default_factory=list)


@dataclass
class EABSConfig:
    """
    Exploration-Aware Budget Scheduler Configuration.
    
    Split budget into exploit and explore portions.
    - exploit: top scored candidates
    - explore: controlled sampling from uncertain/under-validated candidates
    """
    # Budget split
    exploration_ratio: float = 0.2  # 0.0 to 1.0
    exploration_min_budget: int = 1  # Minimum slots for exploration
    
    # Exploit strategy
    exploit_strategy: str = "top_k"  # "top_k", "threshold"
    exploit_score_threshold: float = 0.5
    
    # Exploration strategy
    exploration_strategy: str = "uncertainty"  # "uncertainty", "diversity", "under_validated"
    exploration_uncertainty_method: str = "confidence_margin"  # How to estimate uncertainty
    exploration_temperature: float = 1.0  # For sampling
    
    # Constraints
    max_chunks_per_step: int = 5
    min_chunks_per_step: int = 0
    budget_bytes: Optional[int] = None
    budget_ratio_of_tail: float = 0.05
    
    # Score thresholds
    min_score_threshold: float = 0.3
    confidence_threshold: float = 0.5
    
    # Defensive fallback
    enable_defensive_skip: bool = True
    skip_if_low_confidence: bool = True
    
    # Exploration scope
    exploration_from_ulf_only: bool = True  # Never sample outside ULF candidates
    
    # Logging
    log_exploit_explore_split: bool = True
    log_rejection_reasons: bool = True


@dataclass
class MetadataConfig:
    """Configuration for chunk metadata and sketches."""
    
    # Chunk sketches
    sketch_enabled: bool = True
    sketch_size: int = 256  # Bits in the sketch
    sketch_method: str = "simhash"  # "simhash", "minhash", "bloom"
    
    # Structural features
    track_section_boundaries: bool = True
    track_title_adjacent: bool = True
    track_code_blocks: bool = True
    
    # History statistics
    track_promotion_success: bool = True
    track_access_patterns: bool = True
    history_window_size: int = 100
    
    # Query-side metadata
    query_summary_tokens: int = 64
    query_entity_extraction: bool = False  # Only if lightweight available


@dataclass
class OfflineTrainingConfig:
    """Configuration for offline trace collection and training."""
    
    # Trace collection
    trace_output_dir: str = "outputs/traces"
    trace_format: str = "jsonl"
    trace_steps_per_file: int = 1000
    
    # Teacher label generation
    teacher_use_full_kv: bool = True
    teacher_future_attention_window: int = 10
    teacher_correctness_delta: bool = True
    teacher_label_output_dir: str = "outputs/labels"
    
    # ODUS training
    odus_train_epochs: int = 10
    odus_train_lr: float = 1e-4
    odus_train_batch_size: int = 32
    odus_train_val_split: float = 0.1
    odus_checkpoint_dir: str = "outputs/checkpoints"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and metrics."""
    
    # Metrics to compute
    compute_candidate_metrics: bool = True
    compute_scoring_metrics: bool = True
    compute_scheduler_metrics: bool = True
    compute_promotion_metrics: bool = True
    compute_end_metrics: bool = True
    
    # Failure attribution
    enable_failure_attribution: bool = True
    failure_attribution_depth: str = "full"  # "basic", "full"
    
    # Latency breakdown
    latency_breakdown_components: List[str] = field(default_factory=lambda: [
        "ulf", "as", "scheduler", "burst", "sticky", "total"
    ])
    
    # Bandwidth accounting
    bandwidth_accounting_enabled: bool = True
    bandwidth_breakdown_by_tier: bool = True
    
    # Recovery accounting
    recovery_accounting_enabled: bool = True
    recovery_offline_only: bool = True  # Only in offline eval mode


@dataclass
class PPUConfig:
    """
    Promotion Prediction Unit (PPU) Hardware Configuration (v2.1).

    Models an on-chip hardware accelerator that replaces the software ODUS MLP
    with a quantized Lookup Table (LUT) for single-cycle utility prediction.

    v2.1: Flash-Decoding compatible MMRF-based attention mass ingestion.
    The PPU no longer assumes token-level attention aggregation (incompatible
    with FlashAttention's tiled online-softmax).  Instead, chunk-level masses
    are derived as a zero-cost byproduct of Flash-Decoding's split-K reduction
    and delivered via a Memory-Mapped Register File (MMRF).

    Architecture (5-stage pipeline):
      ┌──────────────────────────────────────────────────────────┐
      │              Promotion Prediction Unit (PPU) v2.1         │
      │  ┌──────────┐ ┌────────────┐ ┌────────────┐ ┌────────┐ │
      │  │  MMRF    │→│ Attention  │→│ Feature    │→│Utility │ │
      │  │ Receiver │  │ Mass Ctr   │  │ Extractor  │  │  LUT   │ │
      │  │(FP16→Q15)│  │(per chunk) │  │(4 signals) │  │        │ │
      │  └──────────┘ └────────────┘ └────────────┘ └────────┘ │
      │       ↑                                         ↓       │
      │  Flash-Decoding          ┌───────────────────────┐      │
      │  Reduction Kernel        │  DMA Request Queue     │     │
      │  (ST.CS to MMRF)         │  (PCIe/CXL arbiter)   │     │
      │                          └───────────────────────┘      │
      └──────────────────────────────────────────────────────────┘
    """

    enabled: bool = True

    # --- LUT Geometry ---
    # Number of features fed to the LUT (reduced from ODUS's 10-dim)
    num_features: int = 4  # recency, similarity, position, history
    # Quantization bits per feature dimension → total entries = 2^(bits * num_features)
    bits_per_feature: int = 4  # 4-bit quantization per feature
    # Derived: total LUT entries (set automatically, do not override)
    # With 4 features × 4 bits = 16 bits → 65536 entries (too large)
    # Strategy: use 4 features × 2 bits = 8 bits → 256 entries (feasible on-chip)
    lut_index_bits: int = 8  # Total index width; entries = 2^lut_index_bits
    lut_output_bits: int = 8  # 8-bit utility output (0..255 → 0.0..1.0)

    # --- Attention Mass Counters ---
    num_counter_entries: int = 512  # Max tracked chunks (one counter per chunk)
    counter_bits: int = 16  # Saturation counter width
    counter_decay_shift: int = 1  # Right-shift decay per step (EMA approximation)

    # --- Feature Extractor ---
    # Recency: log2(steps_since_access), quantized
    recency_bits: int = 2
    # Similarity: cosine similarity quantized
    similarity_bits: int = 2
    # Position: position_ratio quantized
    position_bits: int = 2
    # History: promotion_success_rate quantized
    history_bits: int = 2

    # --- DMA Request Queue ---
    dma_queue_depth: int = 8  # Max outstanding promotion DMA requests
    dma_priority_bits: int = 8  # Priority field width
    dma_coalesce_window: int = 4  # Coalesce adjacent chunk requests

    # --- Pipeline Timing (cycles at target frequency) ---
    clock_frequency_ghz: float = 1.5  # Target frequency
    mmrf_receive_cycles: int = 1  # MMRF format cast (FP16→Q0.15, always 1)
    counter_update_cycles: int = 1  # Attention counter update
    feature_extract_cycles: int = 1  # Feature extraction
    lut_lookup_cycles: int = 1  # LUT read
    dma_enqueue_cycles: int = 1  # DMA request enqueue
    total_pipeline_cycles: int = 5  # End-to-end per-chunk latency (5-stage)

    # --- CACTI Physical Parameters ---
    technology_node_nm: int = 7  # TSMC N7
    sram_read_energy_pj: float = 0.5  # Per-access energy
    lut_sram_banks: int = 1  # Number of SRAM banks for LUT
    counter_sram_banks: int = 2  # Banks for attention counters

    # --- Distillation from ODUS ---
    distill_from_odus: bool = True
    distill_num_calibration_samples: int = 10000
    distill_quantization_method: str = "uniform"  # "uniform", "lloyd_max", "learned"

    # --- Integration Mode ---
    # "standalone" = PPU replaces ODUS entirely
    # "shadow" = PPU runs in parallel with ODUS for validation
    # "hybrid" = PPU for fast path, ODUS for low-confidence fallback
    integration_mode: str = "standalone"
    hybrid_confidence_threshold: float = 0.3  # Below this, fall back to ODUS

    # --- PHT (Promotion History Table) ---
    pht_enabled: bool = True
    pht_num_entries: int = 1024       # 2^10 entries
    pht_counter_bits: int = 2         # 2-bit saturating counters
    pht_prediction_threshold: int = 2 # >= threshold → predict promote
    pht_position_hash_bits: int = 8
    pht_layer_hash_bits: int = 4
    pht_context_hash_bits: int = 4
    pht_enable_periodic_decay: bool = False
    pht_decay_interval_steps: int = 100

    # --- PTB (Promotion Target Buffer) ---
    ptb_enabled: bool = True
    ptb_num_entries: int = 32         # Small fully-associative cache
    ptb_associativity: int = 32       # Fully associative
    ptb_tag_bits: int = 16
    ptb_eviction_policy: str = "lru"  # "lru" or "fifo"
    ptb_entry_bytes: int = 16
    ptb_max_age_steps: int = 50

    # --- PHT/PTB Integration ---
    pht_as_5th_feature: bool = True   # PHT prediction as 5th LUT feature
    ptb_speculative_prefetch: bool = True  # PTB-driven speculative prefetch

    # --- Logging ---
    log_lut_hits: bool = True
    log_dma_decisions: bool = True
    log_area_power_report: bool = True

    # --- HAAS (Hardware-Adaptive Admission Scorer) v4.0 ---
    # When enabled, replaces the static LUT with an online-learning systolic
    # dot-product engine + SGD weight adapter + PID threshold controller +
    # quantile utility sketch.  See hardware/ppu/adaptive_scorer.py.
    haas_enabled: bool = False
    haas_num_features: int = 11       # Must match ODUS-X cue count
    haas_simd_width: int = 11         # Parallel MAC lanes (1=serial, 11=full)
    haas_sgd_enabled: bool = True     # Online weight adaptation
    haas_sgd_learning_rate: float = 0.01
    haas_sgd_lr_decay: float = 0.999
    haas_sgd_min_lr: float = 0.001
    haas_pid_kp: float = 0.5
    haas_pid_ki: float = 0.1
    haas_pid_kd: float = 0.05
    haas_pid_target_pressure: float = 0.7
    haas_sketch_bins: int = 256
    haas_sketch_target_accept_rate: float = 0.3
    haas_threshold_mode: str = "blend"  # "pid_only", "quantile_only", "blend", "min", "max"
    haas_blend_alpha: float = 0.7       # Weight on PID vs quantile in blend mode

    # --- Continuous Batching (v3.0) ---
    cb_enabled: bool = True
    cb_max_batch_size: int = 64
    # Ping-pong state SRAM
    cb_pingpong_buffer_kb: float = 13.0   # Per-buffer (2 buffers total = 26KB)
    cb_swap_dma_bandwidth_gbps: float = 1000.0  # HBM bandwidth for state swaps
    # Doorbell ring buffer
    cb_doorbell_depth: int = 128          # Max pending descriptors
    cb_pull_bandwidth_gbps: float = 200.0 # Async pull bandwidth from VRAM
    # HW-BTW (Hardware Block Table Walker)
    cb_btw_max_sequences: int = 256
    cb_btw_max_blocks_per_seq: int = 8192 # 128K ctx / 16 tok per block
    cb_btw_lookup_latency_cycles: int = 2


@dataclass
class CausalVerificationConfig:
    """
    Master configuration for the Seven-Layer Causal Verification Framework.

    Controls which layers run, their parameters, and pass/fail thresholds.
    Each layer validates a specific causal claim about the ODUS-X scorer.
    """
    # Layer enablement
    run_layer_1_cei: bool = True
    run_layer_2_qudm: bool = True
    run_layer_3_ebqpt: bool = True
    run_layer_4_itlbp: bool = True
    run_layer_5_acs: bool = True
    run_layer_6_cact: bool = True
    run_layer_7_oca: bool = True

    # Layer 1: CEI — Counterfactual Evidence Intervention
    cei_dimensions: List[str] = field(default_factory=lambda: [
        "e_temp", "e_struct", "e_sem", "e_hist", "e_press"
    ])
    cei_consistency_phases: List[str] = field(default_factory=lambda: [
        "prefill", "early_decode", "mid_decode", "late_decode"
    ])
    cei_pass_threshold: float = 0.05
    cei_phase_consistency_required: int = 3  # must be consistent in >= 3 of 4 phases

    # Layer 2: QUDM — Query-Utility Disentanglement Matrix
    qudm_reuse_quantile: float = 0.5
    qudm_utility_quantile: float = 0.5
    qudm_qcdr_threshold: float = 0.45  # QCDR < 0.45 = PROSE overcomes LRU bias via semantic

    # Layer 3: EB-QPT — Evidence Budget-Query Projection Tradeoff
    ebqpt_bq_sweep: List[int] = field(default_factory=lambda: [0, 8, 16, 32])
    ebqpt_budget_total: int = 64

    # Layer 4: ITLBP — Information-Theoretic Lower Bound Probing
    itlbp_variational_samples: int = 1000
    itlbp_budget_sweep: List[int] = field(default_factory=lambda: [16, 32, 48, 64, 128, 256])
    itlbp_hardware_efficiency: float = 0.85
    itlbp_encoder_hidden_dim: int = 8
    itlbp_encoder_epochs: int = 50
    itlbp_encoder_lr: float = 0.01

    # Layer 5: ACS — Adversarial Causal Spoofing
    acs_num_spoof_samples: int = 100
    acs_spoof_types: List[str] = field(default_factory=lambda: ["temporal", "semantic"])
    acs_cvi_threshold: float = 0.5  # CVI >= 0.5 = query-aware significantly reduces spoofing

    # Layer 6: CACT — Cross-Architectural Causal Transfer
    cact_architectures: List[str] = field(default_factory=lambda: [
        "MHA", "GQA_g2", "GQA_g4", "GQA_g8", "MQA", "Mamba"
    ])
    cact_cec_threshold: float = 0.7
    cact_min_consistent_dims: int = 2

    # Layer 7: OCA — Online Causal Adaptation
    oca_algorithm: str = "linucb"
    oca_exploration_epsilon: float = 0.1
    oca_causal_isolation: bool = True
    oca_drift_window: int = 50
    oca_drift_threshold: float = 0.2
    oca_linucb_alpha: float = 0.5  # exploration bonus multiplier
    oca_regret_window: int = 100
    oca_sbfi_min_score: float = 0.3  # hard floor for SBFI constraint


@dataclass
class ProSEXv2Config:
    """Complete ProSE-X 3.0 configuration."""

    # Version
    version: str = "3.0.0"

    # Sub-configs
    mqr_ulf: MQRULFConfig = field(default_factory=MQRULFConfig)
    burst: BurstConfig = field(default_factory=BurstConfig)
    odus: ODUSConfig = field(default_factory=ODUSConfig)
    eabs: EABSConfig = field(default_factory=EABSConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    offline_training: OfflineTrainingConfig = field(default_factory=OfflineTrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ppu: PPUConfig = field(default_factory=PPUConfig)
    causal: CausalVerificationConfig = field(default_factory=CausalVerificationConfig)
    
    # Chunking
    chunk_size: int = 512
    macro_chunk_size: int = 2048  # Transfer unit
    micro_chunk_size: int = 512   # Indexing unit
    
    # Retention hierarchy
    anchor_ratio: float = 0.1  # Top 10% as anchors
    tail_compression_ratio: float = 0.25  # 4x compression
    
    # Experiment
    experiment_name: str = "default"
    seed: int = 42
    
    # Logging
    log_level: str = "INFO"
    log_machine_readable: bool = True
    log_output_dir: str = "outputs/logs"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProSEXv2Config":
        """Create config from dictionary."""
        # Handle nested dataclasses
        config = cls()
        for key, value in d.items():
            if hasattr(config, key):
                if key == "mqr_ulf" and isinstance(value, dict):
                    config.mqr_ulf = MQRULFConfig(**value)
                elif key == "burst" and isinstance(value, dict):
                    config.burst = BurstConfig(**value)
                elif key == "odus" and isinstance(value, dict):
                    config.odus = ODUSConfig(**value)
                elif key == "eabs" and isinstance(value, dict):
                    config.eabs = EABSConfig(**value)
                elif key == "metadata" and isinstance(value, dict):
                    config.metadata = MetadataConfig(**value)
                elif key == "offline_training" and isinstance(value, dict):
                    config.offline_training = OfflineTrainingConfig(**value)
                elif key == "evaluation" and isinstance(value, dict):
                    config.evaluation = EvaluationConfig(**value)
                elif key == "ppu" and isinstance(value, dict):
                    config.ppu = PPUConfig(**value)
                elif key == "causal" and isinstance(value, dict):
                    config.causal = CausalVerificationConfig(**value)
                else:
                    setattr(config, key, value)
        return config
