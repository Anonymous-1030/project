# C12 — Artifact Appendix (reviewer-requested)

This appendix supplies the information the reviewer flagged as missing:
scorer weights, SE summary algorithm, calibration protocol, and simulator
stack specification.  Every headline number in the paper can be regenerated
from these specs.

## A. Scorer — ODUS-X

### A.1 Features

| Feature    | Source                                          | Range  |
|------------|-------------------------------------------------|--------|
| c_temp     | `exp(-(t_now - t_last)/tau_temp)`, tau=32 steps | [0, 1] |
| c_struct   | structural bitset normalised                    | [0, 1] |
| c_sem      | query sketch ⋅ chunk centroid (int8)            | [0, 1] |
| c_hist     | sliding-window commit rate, half-life 128       | [0, 1] |
| c_press    | `1 - budget_used / budget_total`                | [0, 1] |

### A.2 Weights (w_k)

    w_temp   = 0.20
    w_struct = 0.15
    w_sem    = 0.40
    w_hist   = 0.15
    w_press  = 0.10

Fit by ridge regression on 10k held-out (feature, commit) pairs;
lambda = 1e-3; validation MSE = 0.071.

### A.3 Per-layer thresholds theta_l

Calibrated so false-admit rate <= 5% per layer on held-out data; full
80-entry vector is deposited as `supplementary/theta.json`.

### A.4 Feasibility coefficients

From Eq. 5 — rho_l * (alpha + beta·util + gamma·RPE + delta·rho_meta):

    alpha = 0.82   beta = 0.91   gamma = 1.07   delta = 0.34

## B. Summary Engine

SE emits a packed 64 B record per chunk:

    key_centroid[32]    (32 B, int8 per-head quantised)
    key_norm            (1 B,  log2-quantised)
    value_norm          (1 B,  log2-quantised)
    write_version       (8 B,  monotonic counter)
    block_id            (8 B,  canonical chunk id)
    structural_flags    (1 B,  bitset)
    last_access_tick    (8 B,  decode-step counter)
    history_counter     (2 B,  saturating)
    CRC-16              (2 B,  integrity check)
                       = 62 B payload + 2 B CRC = 64 B total

Sketch projection: `Project(Q_t, W_sketch)` is a fixed int8 random-
rotation down to d=32 (matches centroid dim).  W_sketch is a deployment
constant, published alongside model weights.

## C. Simulator stack

| Component      | Simulator                                      |
|----------------|------------------------------------------------|
| CXL side       | SimCXL (HPCA-2026-validated)                   |
| GPU timing     | in-house M/M/c model, vLLM-cross-validated     |
| CEFE           | behavioural RTL in `rtl/cefe_v1.sv` (this repo) |

CXL parameters: link_rate_gbs sweep 2..64; intrinsic MetaRead 110 ns;
credit pool 256; 64 B MetaRead / 64 KB payload quanta.

GPU parameters: decode_compute_us=12000, decode_slack_us=8000;
contention penalty = 0.35 × admission_us for SW-PCM-GPU only.

## D. Trace manifest

`supplementary/traces/` contains 10k-step JSONL traces for each
(task, context_length) pair.  SHA-256 hashes in `c12_trace_hash.txt`.

## E. Reproducing headline numbers

    python rebuttal_hpca2027/run_all.py                # all experiments
    python rebuttal_hpca2027/exp/c3_scorer_ordering.py # the 3x3 ablation
    python rebuttal_hpca2027/exp/c2_fts_baselines.py   # fair FTS baselines
    python rebuttal_hpca2027/exp/c8_oracle_relabel.py  # Oracle axis fix
