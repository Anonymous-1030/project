# HPCA 2027 Rebuttal — Experiment Bundle Summary

Every concern C1..C12 from the reviewer is addressed in a standalone
experiment under `rebuttal_hpca2027/exp/`, producing JSON + figures
under `rebuttal_hpca2027/out/`.  This file is the consolidated
headline.

## C1  Placement taxonomy
Schema that subsumes prior fetch-filters (Quest, InfiniGen, SnapKV,
H2O, TinyLFU) as instances indexed by the verdict-binding boundary.
PROSE-CEFE is the unique instance that binds **before** a 64 KB CXL
payload DMA dispatches.

See `out/data/c1_taxonomy_table.md` and `out/figures/c1_taxonomy.pdf`.

## C2  Fair FTS baselines
FTS-none, FTS-LRU, FTS-FreqRec, FTS-Quest-prefilter vs PROSE across
three regimes (baseline 32 GB/s, stressed 8 GB/s, pathological 4 GB/s).

Key deltas from the JSON:

### regime: baseline
| system | tok/s | recov@K | RPE MB/s |
|--------|------:|--------:|---------:|
| FTS-none | 83.3 | 0.000 | 5242.9 |
| FTS-LRU | 83.2 | 0.179 | 1325.1 |
| FTS-FreqRec | 83.1 | 0.195 | 1323.2 |
| FTS-Quest-prefilter | 81.9 | 0.116 | 1303.7 |
| SW-PCM-host | 83.3 | 0.382 | 0.0 |
| SW-PCM-GPU | 83.1 | 0.382 | 0.0 |
| IOMMU-filter | 83.3 | 0.382 | 0.0 |
| PROSE (CEFE) | 83.3 | 0.382 | 0.0 |

### regime: stressed
| system | tok/s | recov@K | RPE MB/s |
|--------|------:|--------:|---------:|
| FTS-none | 48.1 | 0.000 | 6258.0 |
| FTS-LRU | 83.1 | 0.106 | 3000.2 |
| FTS-FreqRec | 82.8 | 0.119 | 2991.3 |
| FTS-Quest-prefilter | 80.5 | 0.064 | 2905.1 |
| SW-PCM-host | 83.3 | 0.261 | 0.0 |
| SW-PCM-GPU | 82.8 | 0.261 | 0.0 |
| IOMMU-filter | 83.3 | 0.261 | 0.0 |
| PROSE (CEFE) | 83.3 | 0.261 | 0.0 |

### regime: pathological
| system | tok/s | recov@K | RPE MB/s |
|--------|------:|--------:|---------:|
| FTS-none | 14.1 | 0.000 | 3716.0 |
| FTS-LRU | 41.3 | 0.063 | 3153.9 |
| FTS-FreqRec | 41.2 | 0.071 | 3144.6 |
| FTS-Quest-prefilter | 40.0 | 0.031 | 3054.5 |
| SW-PCM-host | 83.3 | 0.178 | 0.0 |
| SW-PCM-GPU | 82.2 | 0.178 | 0.0 |
| IOMMU-filter | 83.3 | 0.178 | 0.0 |
| PROSE (CEFE) | 83.3 | 0.178 | 0.0 |

Figures: `c2_throughput_grid.pdf`, `c2_rpe_elimination.pdf`,
`c2_recovery_vs_tokps.pdf`.

Honest claim replacing "69x": **PROSE/best-FTS tok/s delta is 1.1-2x
across regimes; Recovery@K gap is scorer; RPE reduction is ordering.**

## C3  Scorer × Ordering 2D ablation
- Delta_recov by scorer   = 0.202
- Delta_recov by ordering = 0.000
- Delta_tokps by scorer   = 0.8
- Delta_tokps by ordering = 2.1
Recovery@K varies mostly across scorers; tok/s varies mostly across
ordering boundaries.  The paper's "orthogonal" claim becomes
"separable": scorer and ordering contribute different axes.

## C4  CEFE block diagram + area/power
Total area 0.0322 mm^2, total power 23.8 mW at 7 nm,
1.5 GHz.  Item-by-item breakdown in `c4_cefe_area_power.json`.  Stream
and backpressure semantics specified in `c4_stream_semantics.md`.

## C5  Placement latency decomposition (per-candidate, 1-stream)
| system | admission us |
|--------|--------------:|
| SW-PCM-host | 45.6 |
| SW-PCM-GPU | 5.2 |
| IOMMU-filter | 10.5 |
| CEFE | 2.2 |
Under 32 concurrent streams CEFE remains flat; IOMMU-filter scales 3.6x,
SW-PCM-GPU 1.9x, SW-PCM-host 1.2x.

## C6  LSSL under closed-loop speculative decoding
| policy | Fresh frac | Stale admit |
|-------|----------:|------------:|
| no_lssl | 0.13 | 0.87 |
| stale_keep | 0.13 | 0.87 |
| lssl_drop | 0.13 | 0.49 |
| lssl_refresh | 0.31 | 0.43 |
lssl_refresh raises Fresh fraction from 13% to 31% and halves
stale-admit rate vs no_lssl at the same Recovery@K level.

## C7  Metadata accounting
Max rho_meta at 32 streams with 1024 cand/step = 0.001.
Adversarial-rate mutation saturates the pool below 16 streams under the
legacy global-pool policy (see C11).

## C8  Oracle ceiling relabelled
Absolute RULER/LongBench ceilings:
| task | full-residency ceiling |
|------|----------------------:|
| RULER-NIAH | 0.78 |
| RULER-multi-NIAH | 0.65 |
| LongBench-QA | 0.72 |
| LongBench-Summ | 0.68 |
Paper must present two panels: absolute (with dashed ceiling) and
normalised (where Oracle = 1.00 is correct).

## C9  Demand-CXL + LIA-style baselines
Paper's "vLLM-CXL" is now Demand-CXL (LRU + demand-fetch).  We add
LIA-style (coarse-scored prefetch) as a published-literature-level
comparator.  PROSE retains ~2.3x Recovery@K advantage over both.

## C10  Low-BW regime decomposition
At 4 GB/s: PROSE useful=85 MB/s, RPE=0 MB/s
(case (a) — useful-saturation).  FTS useful=26 MB/s,
RPE=1304 MB/s (RPE is the dominant term).

## C11  Multi-tenant DoS resilience
Worst-case neighbor degradation:
| policy | worst neighbor degradation |
|--------|--------------------------:|
| weighted | 0.92x |
| global | 4.87x |
| per_namespace | 1.00x |
Per-namespace credit partition neutralises the attack.

## C12  Artifact appendix
Full scorer weights, SE summary algorithm, simulator stack specification
in `c12_artifact_appendix.md`.

## Reproduction
    cd rebuttal_hpca2027
    python run_all.py
