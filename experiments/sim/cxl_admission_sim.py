"""
CXL-attached KV admission simulator used by all rebuttal experiments (C1-C12).

Design notes (grounding the numbers):
  * CXL.mem payload path:
      - Bandwidth     : configurable (default 32 GB/s, sweep 4..64)
      - Transaction-level latency floor for MetaRead: 100 ns intrinsic + queue
      - Payload DMA quantum: 64 KB (non-preemptive once dispatched)
      - MetaRead quantum  : 64 B per candidate
  * Decode step model:
      - per-step candidate count = ctx_len * candidate_fanout_per_tok
      - chunk_size    = 64 KB (16K tokens @ 4B per token)
      - each step has a `decode_slack_us` during which CXL I/O can overlap
  * Scorers (all PCM-compatible if they consume MetaRead only):
      - quest_criticality : min/max-key centroid inner product vs query
      - freqrec           : recency + frequency only (no semantic)
      - odus_x            : paper scorer (linear combo of 5 features)
  * Ordering boundaries:
      - fts               : fetch-then-score (payload first, scorer optional)
      - sw_host           : SW admission in host runtime
      - sw_gpu            : SW admission in persistent GPU kernel
      - iommu_filter      : IOMMU/DPU side-car filter
      - cefe              : hardware CEFE in-line filter (this paper)
  * Each boundary is characterized by:
      - admission_latency_us      : decision time per candidate
      - payload_reorder_allowed   : can the scorer retire a payload descriptor
                                    before the CE DMA dispatches it?
      - contends_with_compute     : does the admission engine steal GPU cycles?

This simulator is deliberately closed-form (queueing + serialization model)
so all 12 experiments reproduce in < 2 minutes on a laptop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #
CHUNK_PAYLOAD_B = 64 * 1024       # 64 KB per KV chunk
META_B          = 64              # 64 B summary per chunk
NS_PER_US       = 1_000.0
GB              = 1e9

# Intrinsic CXL.mem MetaRead latency (matches published SimCXL numbers)
META_INTRINSIC_NS = 110.0


# --------------------------------------------------------------------------- #
# Ground truth: which chunks are actually useful this step                    #
# --------------------------------------------------------------------------- #
@dataclass
class StepGroundTruth:
    """Per-decode-step ground-truth for Recovery@K computation."""
    useful_ids: np.ndarray       # chunk ids that would attend > tau
    candidate_ids: np.ndarray    # chunk ids presented by ULF
    key_centroids: np.ndarray    # [N, d] — used by quest
    recency: np.ndarray          # [N] recency signal 0..1
    frequency: np.ndarray        # [N] frequency signal 0..1
    semantic_sim: np.ndarray     # [N] query-chunk sim 0..1 (ground truth)
    structural: np.ndarray       # [N] structural markers 0..1
    history: np.ndarray          # [N] historical-success 0..1
    pressure: np.ndarray         # [N] budget-pressure 0..1


def synth_step(
    n_candidates: int,
    useful_fraction: float,
    rng: np.random.Generator,
    semantic_signal_strength: float = 0.80,
    useful_dir: Optional[np.ndarray] = None,
) -> StepGroundTruth:
    """
    Synthesize a decode step.  useful ids are the ones whose sem-sim > tau.
    All features have tunable correlation with usefulness.
    The `useful_dir` is the semantic axis along which useful-chunk key-centroids
    concentrate; the caller uses a noisy version as the query direction so that
    Quest's criticality is informative (reviewer C2).
    """
    # Ground-truth usefulness label
    n_useful = max(1, int(round(n_candidates * useful_fraction)))
    perm = rng.permutation(n_candidates)
    useful = np.zeros(n_candidates, dtype=bool)
    useful[perm[:n_useful]] = True

    # Feature generation: useful chunks get higher mean on informative cues,
    # same mean on uninformative cues (this models the reviewer's concern
    # that scorer quality matters).
    def feat(signal: float, useful_mean: float = 0.72, noise_std: float = 0.18):
        base = rng.normal(0.5, noise_std, n_candidates)
        if signal > 0:
            delta = signal * (useful_mean - 0.5)
            base = base + useful.astype(float) * delta
        return np.clip(base, 0.0, 1.0)

    semantic_sim = feat(semantic_signal_strength)
    recency      = feat(0.55)
    frequency    = feat(0.30)
    structural   = feat(0.40)
    history      = feat(0.55)
    pressure     = rng.uniform(0.0, 1.0, n_candidates)

    # Key centroids (d=32) aligned to `useful_dir` for useful chunks
    d = 32
    if useful_dir is None:
        useful_dir = rng.normal(0.0, 1.0, d)
        useful_dir /= (np.linalg.norm(useful_dir) + 1e-9)
    centroids = rng.normal(0.0, 1.0, (n_candidates, d))
    for i in np.where(useful)[0]:
        centroids[i] += 1.4 * useful_dir
    centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)

    return StepGroundTruth(
        useful_ids=np.where(useful)[0],
        candidate_ids=np.arange(n_candidates),
        key_centroids=centroids,
        recency=recency,
        frequency=frequency,
        semantic_sim=semantic_sim,
        structural=structural,
        history=history,
        pressure=pressure,
    )


# --------------------------------------------------------------------------- #
# Scorers                                                                     #
# --------------------------------------------------------------------------- #
def score_none(step: StepGroundTruth) -> np.ndarray:
    return np.zeros(len(step.candidate_ids))


def score_lru(step: StepGroundTruth) -> np.ndarray:
    return step.recency.copy()


def score_freqrec(step: StepGroundTruth) -> np.ndarray:
    return 0.5 * step.recency + 0.5 * step.frequency


def score_quest(step: StepGroundTruth, query_dir: np.ndarray) -> np.ndarray:
    # Query-conditional key-centroid similarity (Quest's core signal).
    return step.key_centroids @ query_dir


def score_odus_x(step: StepGroundTruth, w: Optional[Dict[str, float]] = None) -> np.ndarray:
    # The paper's scorer: linear combination of five features.
    w = w or {"temp": 0.20, "struct": 0.15, "sem": 0.40, "hist": 0.15, "press": 0.10}
    return (
        w["temp"]   * step.recency
        + w["struct"] * step.structural
        + w["sem"]    * step.semantic_sim
        + w["hist"]   * step.history
        + w["press"]  * (1.0 - step.pressure)
    )


SCORER_REGISTRY = {
    "none":     lambda s, **k: score_none(s),
    "lru":      lambda s, **k: score_lru(s),
    "freqrec":  lambda s, **k: score_freqrec(s),
    "quest":    lambda s, **k: score_quest(s, k["query_dir"]),
    "odus_x":   lambda s, **k: score_odus_x(s, k.get("odus_weights")),
}


# --------------------------------------------------------------------------- #
# Ordering boundary cost model                                                #
# --------------------------------------------------------------------------- #
@dataclass
class BoundaryCost:
    name: str
    per_cand_decision_us: float           # time to decide verdict
    per_cand_metaread_us: float           # MetaRead path cost (0 if fetches payload)
    payload_reorder_allowed: bool
    contends_with_compute: bool
    meta_credits: int                     # outstanding MetaRead credits
    notes: str = ""


BOUNDARIES = {
    "fts_none":     BoundaryCost("fts_none",    0.0,  0.0,  False, True,  0,   "fetch-all, no filter"),
    "fts_lru":      BoundaryCost("fts_lru",     0.05, 0.0,  False, True,  0,   "fetch after LRU filter (host)"),
    "fts_freqrec":  BoundaryCost("fts_freqrec", 0.10, 0.0,  False, True,  0,   "fetch after FreqRec (host)"),
    "fts_quest":    BoundaryCost("fts_quest",   0.60, 2.0,  False, True,  64,  "fetch after Quest metadata (host)"),
    "sw_host":      BoundaryCost("sw_host",    47.0,  2.0,  True,  False, 32,  "host-runtime PCM"),
    "sw_gpu":       BoundaryCost("sw_gpu",      5.2,  2.0,  True,  True,  64,  "persistent-kernel PCM"),
    "iommu":        BoundaryCost("iommu",      10.5,  2.0,  True,  False, 128, "IOMMU/DPU filter PCM"),
    "cefe":         BoundaryCost("cefe",        3.9,  1.5,  True,  False, 256, "on-CE hardware PCM"),
    # C9: published-style baselines -------------------------------
    "demand_cxl":   BoundaryCost("demand_cxl",  0.0,  0.0,  False, True,  0,   "LRU-on-miss demand fetch (no prefetch, no scorer)"),
    "lia_style":    BoundaryCost("lia_style",   0.30, 0.0,  False, True,  0,   "LIA-style coarse-scored prefetch"),
}


# --------------------------------------------------------------------------- #
# Per-step closed-loop model                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class StepResult:
    admitted:          np.ndarray
    rejected:          np.ndarray
    committed:         np.ndarray
    useful_admitted:   np.ndarray
    rpe_bytes:         float    # rejected-payload exposure
    meta_bytes:        float
    useful_bytes:      float
    wasted_bytes:      float
    admission_us:      float
    transport_us:      float
    queue_depth_peak:  float
    tok_per_s_bound:   float
    recovery_at_k:     float
    step_tok:          float


@dataclass
class SimConfig:
    cxl_bw_gbs:           float = 32.0
    decode_compute_us:    float = 12_000.0  # 70B-class attention+MLP per step
    decode_slack_us:      float = 8_000.0   # window in which CXL I/O overlaps compute
    budget_per_step:      int   = 64        # admits per step (HBM budget)
    top_k_useful:         int   = 32        # Recovery@K denominator
    n_candidates:         int   = 1024
    useful_fraction:      float = 0.04
    semantic_strength:    float = 0.80
    meta_credits_override: Optional[int] = None
    # FTS pre-filter admit rate (fraction of candidates kept for DMA).
    # Our FTS definition: the pre-filter is lossless enough to keep ≥2× budget.
    fts_prefilter_keep_frac: float = 0.30


def simulate_step(
    step: StepGroundTruth,
    boundary_name: str,
    scorer_name: str,
    cfg: SimConfig,
    query_dir: np.ndarray,
) -> StepResult:
    b = BOUNDARIES[boundary_name]
    n = len(step.candidate_ids)
    link_bps = cfg.cxl_bw_gbs * GB

    # ----- Scorer output at the boundary --------------------------------
    scores = SCORER_REGISTRY[scorer_name](step, query_dir=query_dir,
                                          odus_weights=None)

    # ----- Determine admit / fetched sets ------------------------------
    budget = cfg.budget_per_step
    fetch_style = (boundary_name.startswith("fts") or
                   boundary_name in ("demand_cxl", "lia_style"))
    if fetch_style:
        # FTS: optional pre-filter → DMA → post-filter with same/better scorer.
        if scorer_name == "none":
            fetched_mask = np.ones(n, dtype=bool)
        else:
            keep_frac = cfg.fts_prefilter_keep_frac
            thresh = np.quantile(scores, 1.0 - keep_frac)
            fetched_mask = scores >= thresh
        # After DMA, a post-fetch scorer picks the top-budget to residency.
        # For parity, we let FTS use the SAME scorer (can only help FTS).
        order = np.argsort(scores)[::-1]
        keep_mask = np.zeros(n, dtype=bool)
        kept = 0
        for idx in order:
            if fetched_mask[idx] and kept < budget:
                keep_mask[idx] = True
                kept += 1
        meta_bytes   = 0  # FTS does not pay metadata BW
        admission_us = b.per_cand_decision_us * n            # scorer runs serially
    else:
        # PCM boundaries: MetaRead first, payload only for admitted.
        order = np.argsort(scores)[::-1]
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[order[:budget]] = True
        fetched_mask = keep_mask
        meta_bytes = n * META_B
        mc = cfg.meta_credits_override or max(1, b.meta_credits)
        per_wave_us = b.per_cand_decision_us + max(
            b.per_cand_metaread_us,
            META_INTRINSIC_NS / NS_PER_US,
        )
        n_waves = math.ceil(n / mc)
        admission_us = per_wave_us * n_waves

    fetched_bytes  = int(fetched_mask.sum()) * CHUNK_PAYLOAD_B
    admitted_bytes = int(keep_mask.sum())   * CHUNK_PAYLOAD_B
    useful_bytes   = int((keep_mask &
                          np.isin(step.candidate_ids, step.useful_ids)).sum()
                         ) * CHUNK_PAYLOAD_B
    wasted_bytes   = max(0, fetched_bytes - admitted_bytes)
    rpe_bytes      = wasted_bytes

    # ----- Transport time on CXL link ----------------------------------
    transport_us = (meta_bytes + fetched_bytes) / link_bps * 1e6

    # ----- Wall-clock per decode step ----------------------------------
    # Decode-compute runs in parallel to CXL I/O up to decode_slack_us.
    # Beyond that, excess CXL time extends the step.
    io_us        = max(transport_us, admission_us)
    contention   = admission_us * 0.35 if b.contends_with_compute else 0.0
    overflow_us  = max(0.0, io_us - cfg.decode_slack_us)
    wall_us      = cfg.decode_compute_us + overflow_us + contention
    tok_per_s    = 1e6 / wall_us if wall_us > 0 else 0.0

    # Queue-depth peak as fraction of available link-bytes during the slack
    slack_bytes  = max(1.0, (cfg.decode_slack_us / 1e6) * link_bps)
    queue_depth_peak = fetched_bytes / slack_bytes

    # ----- Recovery@K ---------------------------------------------------
    top_useful = set(int(x) for x in step.useful_ids[: cfg.top_k_useful])
    admitted_useful = set(int(x) for x in np.where(keep_mask)[0]) & top_useful
    recovery_at_k = len(admitted_useful) / max(1, len(top_useful))

    return StepResult(
        admitted          = np.where(keep_mask)[0],
        rejected          = np.where(~keep_mask)[0],
        committed         = np.where(keep_mask)[0],
        useful_admitted   = np.array(sorted(admitted_useful)),
        rpe_bytes         = float(rpe_bytes),
        meta_bytes        = float(meta_bytes),
        useful_bytes      = float(useful_bytes),
        wasted_bytes      = float(wasted_bytes),
        admission_us      = float(admission_us),
        transport_us      = float(transport_us),
        queue_depth_peak  = float(queue_depth_peak),
        tok_per_s_bound   = float(tok_per_s),
        recovery_at_k     = float(recovery_at_k),
        step_tok          = 1.0,
    )


# --------------------------------------------------------------------------- #
# Closed-loop run (many steps, averages)                                      #
# --------------------------------------------------------------------------- #
def run_closed_loop(
    boundary_name: str,
    scorer_name: str,
    cfg: SimConfig,
    n_steps: int = 256,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    attr_map = {
        "tok_per_s":        "tok_per_s_bound",
        "recovery_at_k":    "recovery_at_k",
        "rpe_bytes":        "rpe_bytes",
        "useful_bytes":     "useful_bytes",
        "wasted_bytes":     "wasted_bytes",
        "meta_bytes":       "meta_bytes",
        "admission_us":     "admission_us",
        "transport_us":     "transport_us",
        "queue_depth_peak": "queue_depth_peak",
    }
    totals: Dict[str, List[float]] = {k: [] for k in attr_map}
    for step_i in range(n_steps):
        # Per-step useful_dir drifts slowly → the query tracks it with noise.
        useful_dir = rng.normal(0.0, 1.0, 32)
        useful_dir /= (np.linalg.norm(useful_dir) + 1e-9)
        # Query is a noisy aligned version of useful_dir (models an
        # attention-aware query sketch that partially reveals semantics).
        query_dir = useful_dir + 0.65 * rng.normal(0.0, 1.0, 32)
        query_dir /= (np.linalg.norm(query_dir) + 1e-9)

        step = synth_step(
            cfg.n_candidates,
            cfg.useful_fraction,
            rng,
            semantic_signal_strength=cfg.semantic_strength,
            useful_dir=useful_dir,
        )
        r = simulate_step(step, boundary_name, scorer_name, cfg, query_dir)
        for k, attr in attr_map.items():
            totals[k].append(getattr(r, attr))
    out = {
        "boundary":            boundary_name,
        "scorer":              scorer_name,
        "tok_per_s_mean":      float(np.mean(totals["tok_per_s"])),
        "tok_per_s_p50":       float(np.percentile(totals["tok_per_s"], 50)),
        "tok_per_s_p5":        float(np.percentile(totals["tok_per_s"], 5)),
        "recovery_at_k_mean":  float(np.mean(totals["recovery_at_k"])),
        "rpe_bytes_mean":      float(np.mean(totals["rpe_bytes"])),
        "useful_bytes_mean":   float(np.mean(totals["useful_bytes"])),
        "wasted_bytes_mean":   float(np.mean(totals["wasted_bytes"])),
        "meta_bytes_mean":     float(np.mean(totals["meta_bytes"])),
        "admission_us_mean":   float(np.mean(totals["admission_us"])),
        "transport_us_mean":   float(np.mean(totals["transport_us"])),
        "queue_depth_peak_p50": float(np.percentile(totals["queue_depth_peak"], 50)),
        "queue_depth_peak_p99": float(np.percentile(totals["queue_depth_peak"], 99)),
        "useful_frac_of_fetched": float(
            np.sum(totals["useful_bytes"]) / max(1.0,
            (np.sum(totals["useful_bytes"]) + np.sum(totals["wasted_bytes"])))
        ),
        "n_steps":             n_steps,
        "cfg":                 asdict(cfg),
    }
    return out
