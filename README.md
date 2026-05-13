# PROSE-X: Copy-Engine Admission Control for CXL-Backed KV-Cache Promotion

PROSE-X places admission control at the CXL Copy Engine front-end (CEFE), enforcing score-before-fetch at the last pre-DMA hardware boundary. When bandwidth is abundant, the same scoring mechanism generates a block-sparse attention mask that converts compute budget into semantic recovery. When one-shot scoring hits its information-theoretic limit, Speculative Evidence Accumulation (SEA) closes the gap by converting idle bandwidth into sequential verification.

## Key Components

- **CEFE** — 5-stage PPU pipeline at the Copy Engine issue queue (0.071 mm², 80 mW at 7nm)
- **HES** — Hierarchical Evidence Synthesis: INT4 micro-embedding scorer on CXL controller SRAM (96 KB, ~2K params)
- **SEA** — Speculative Evidence Accumulation: converts one-shot prediction into sequential verification via speculative fetch + attention probe
- **ABA** — Adaptive Bandwidth Arbitrage: runtime mode selector and speculation budget controller
- **Hardware RTL** — Synthesizable Verilog for PHT and QFC engines

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      Decode Step t                              │
│                                                                │
│  Query ──→ HES Scorer ──→ Evidence Accumulator ──→ State FSM   │
│                ↑                  ↑         ↑          │       │
│                │                  │         │          ▼       │
│           micro-emb          attn fb    verify    ┌────────┐  │
│           (SRAM)            (ADMITTED) (SPECUL.)  │ADMITTED│──→ Attention
│                                                   │SPECUL. │──→ Verify
│                                                   │REJECTED│    (skip)
│                                                   └────────┘  │
│                                                       ↑       │
│                                              ABA bandwidth     │
│                                              → speculation $   │
└────────────────────────────────────────────────────────────────┘
```

## Results

### SEA Convergence: Closing the Oracle Gap

![SEA Convergence](results/sea_convergence.png)

SEA converts idle bandwidth into exploration. At high bandwidth (32 GB/s), speculation budget is large and convergence is fast. At low bandwidth (4 GB/s), convergence is slower but still guaranteed. HES one-shot recall (22.5%) is the floor; SEA lifts it to 75-87.5%.

| Bandwidth | HES One-Shot | SEA @ Step 5 | SEA Steady State | Gap Closed |
|-----------|-------------|-------------|-----------------|------------|
| 32 GB/s | 22.5% | **75.0%** | **87.5%** | 83.9% |
| 8 GB/s | 22.5% | **62.5%** | **87.5%** | 83.9% |
| 4 GB/s | 22.5% | **50.0%** | **75.0%** | 67.7% |

### Recall-Budget Curves: Scored vs Unscored Selection

![Recall Curves](results/pcm_recall_curves.png)

At tight budgets (5-15%), scored sparse attention provides **1.3-3.7x recall lift** over unscored static masks. Static masks plateau because their sparsity pattern is query-agnostic.

### Recall-Throughput Pareto Frontier

![Pareto Frontier](results/pcm_pareto_frontier.png)

PROSE-X provides Pareto points unreachable by static masks: static masks are fast-but-blind; scored enforcement is quality-aware speedup.

### PCM Ablation: Full Grid

**32 GB/s (Compute Enforcement)**

| Context | Config | Gold Recall | Throughput | Latency (μs) |
|---------|--------|-------------|------------|--------------|
| 4096 | Baseline | 0.0% | 2.02x | 206.7 |
| 4096 | HES (scored) | **39.5%** | 2.60x | 163.9 |
| 4096 | Static random | 27.0% | 2.67x | 153.6 |
| 8192 | HES (scored) | **45.0%** | 3.14x | 266.5 |
| 8192 | Static random | 30.0% | 3.20x | 256.0 |
| 16384 | HES (scored) | **52.5%** | 2.88x | 573.1 |
| 16384 | Static random | 28.0% | 2.91x | 563.2 |

**4 GB/s (DMA Enforcement)**

| Context | Config | Gold Recall | Throughput | Latency (μs) |
|---------|--------|-------------|------------|--------------|
| 4096 | HES (scored) | **66.0%** | 1.14x | 360.9 |
| 4096 | Static random | 27.0% | 2.68x | 153.6 |
| 8192 | HES (scored) | **45.0%** | 1.66x | 498.6 |
| 8192 | Static random | 30.0% | 3.22x | 256.0 |
| 16384 | HES (scored) | **46.5%** | 2.19x | 758.5 |
| 16384 | Static random | 28.0% | 2.92x | 563.2 |

### Recall Curve Sweep (8 GB/s, 8192 tokens)

| Budget | HES (Scored) | Static Random | Static Position | Baseline |
|--------|-------------|---------------|-----------------|----------|
| 5% | **38.5%** | 10.5% | 0.0% | 0.0% |
| 10% | **45.0%** | 30.0% | 25.0% | 25.0% |
| 15% | **68.0%** | 40.5% | 50.0% | 25.0% |
| 20% | **70.0%** | 58.0% | 50.0% | 25.0% |

### CEFE Experiments

| | |
|---|---|
| ![Throughput](experiments/out/figures/c2_throughput_grid.png) | ![Latency](experiments/out/figures/c5_latency_stacked.png) |
| End-to-end throughput | Placement latency decomposition |
| ![Concurrency](experiments/out/figures/c5_concurrency_sweep.png) | ![BW Sweep](experiments/out/figures/c10_bw_sweep.png) |
| CEFE under concurrent streams | Low-bandwidth performance |

### Hardware Overhead (7nm)

| Component | Area | Power | Latency |
|-----------|------|-------|---------|
| Admission Predictor | 0.012 mm² | 15 mW | 2.5 ns |
| Prefetch DMA Engine | 0.008 mm² | 12 mW | 1.5 ns |
| Near-CXL Decompressor | 0.045 mm² | 45 mW | 5.0 ns |
| Metadata SRAM | 0.006 mm² | 8 mW | 1.5 ns |
| SEA State (11 KB SRAM) | 0.002 mm² | 3 mW | 1.5 ns |
| **Total** | **0.073 mm²** | **83 mW** | — |

Overhead: **0.009% of H100 area**, **0.012% of TDP**.

### ABA Mode Distribution

| Bandwidth | Transparent | Hybrid | Aggressive | Speculation Budget |
|-----------|-------------|--------|------------|-------------------|
| 32 GB/s | 96% | 4% | 0% | 12% (15 blocks/step) |
| 8 GB/s | 0% | 96% | 4% | 4% (5 blocks/step) |
| 4 GB/s | 0% | 4% | 96% | 1% (1 block/step) |

## Project Structure

```
├── src/                    Core implementation
│   ├── innovations/        SEA, ABA, HES modules
│   ├── promotion/          5-stage pipeline (ULF→ODUS→EABS→Burst→Sticky)
│   ├── memory/             CXL queue simulator, multi-tier hierarchy
│   ├── hardware/           PPU simulator, area/power models
│   ├── bridge/             vLLM integration hooks
│   ├── eval/               Evaluation framework
│   └── runners/            Experiment entry points
├── hardware/               RTL (Verilog/SystemVerilog)
│   ├── rtl/                PHT_ENGINE, QFC_ENGINE, MAC array
│   └── synth/              Synthesis constraints
├── experiments/            Rebuttal experiments (c1–c12)
│   ├── sim/                CXL admission simulator
│   └── out/                Generated figures and data
├── configs/                YAML configurations
├── results/                Benchmark outputs + figures
├── scripts/                Figure generators
├── tests/                  Unit tests
├── docs/                   Architecture documents
│   ├── paper_writing_reference.md
│   └── sea_architecture.md
└── figure/                 Paper figures
```

## Quick Start

```bash
pip install -e .

# SEA convergence experiment (closes the oracle gap)
python -m src.runners.sea_convergence_runner

# PCM ablation (HES-only vs static mask baselines)
python -m src.runners.pcm_ablation_runner

# Full innovation benchmark
python -m src.runners.innovation_benchmark

# Regenerate all rebuttal figures (12 experiments)
cd experiments && python run_all.py
```

## Requirements

- Python >= 3.9
- NumPy >= 1.20
- PyYAML >= 5.4
- Matplotlib >= 3.5

## Citation

```bibtex
@inproceedings{prosex2027,
  title={PROSE-X: Copy-Engine Admission Control for CXL-Backed KV-Cache Promotion},
  author={Anonymous},
  booktitle={HPCA},
  year={2027}
}
```

## License

MIT. See [LICENSE](LICENSE).
