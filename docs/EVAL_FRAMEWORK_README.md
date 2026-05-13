# ProSE-X 2.0 Evaluation Framework

## Quick Start

### 1. Validate Acceptance Gates

Verify the framework is complete:

```bash
python prose_v2/src/runners/validate_acceptance_gates.py
```

Expected output: `ALL GATES PASSED`

### 2. Run Unit Tests

Verify metric computations are correct:

```bash
python -m unittest prose_v2.tests.metrics.test_shared_evaluator -v
```

Expected: 22 tests pass

### 3. Run Regression Baseline

Re-run the frozen baseline:

```bash
python prose_v2/src/runners/regression_baseline.py
```

### 4. Run Ablations

Execute burst ablation:
```bash
python prose_v2/src/runners/burst_ablation.py --seed 42
```

Execute sticky ablation:
```bash
python prose_v2/src/runners/sticky_ablation.py --seed 42
```

## Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EvaluationOrchestrator                    │
│                      (Master Coordinator)                    │
└──────────────┬──────────────────────────────────────────────┘
               │
    ┌──────────┼──────────┬──────────────┬──────────────┐
    │          │          │              │              │
    ▼          ▼          ▼              ▼              ▼
┌────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Shared │ │ Useful │ │ Neighbor │ │ Failure  │ │ Fair     │
│Evaluator│ │ Bytes  │ │ Recall   │ │Attribution│ │Comparison│
│        │ │Account.│ │ Reporter │ │          │ │ Runner   │
└────────┘ └────────┘ └──────────┘ └──────────┘ └──────────┘
    │          │          │              │              │
    │          │          │              │              │
    ▼          ▼          ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Reports                            │
│  - evaluation_result.json                                    │
│  - useful_bytes_report.json                                  │
│  - neighbor_recall_report.json                               │
│  - failure_attribution.json                                  │
│  - burst_ablation.json + .md                                 │
│  - sticky_ablation.json + .md                                │
└─────────────────────────────────────────────────────────────┘
```

## Key Metrics

### Conditional Recovery
```
P(visible | gold exists in universe)
```
- **Numerator**: Steps where gold is visible to attention
- **Denominator**: Steps where gold exists in system

### No-Miss Rate
```
1 - (miss_count / total_steps)
```
- Miss: Gold needed but not visible

### Useful Promote Ratio (UPR)
```
useful_promoted_bytes / total_promoted_bytes
```

Three modes:
1. **attention_access_based**: Chunk accessed with attention > 0.01
2. **gold_overlap_based**: Chunk overlaps gold region (eval only)
3. **recovery_event_based**: Chunk contributes to recovery

### Burst Gain
```
(recovery_with_burst - recovery_without_burst) / (bytes_delta_MB)
```
- Unit: Recovery points per additional MB
- Positive = burst helps more than it costs

## File Reference

### Documentation
- `prose_v2/docs/metrics_spec.md` - Frozen metric definitions
- `prose_v2/docs/evaluation_framework_summary.md` - Complete framework guide
- `prose_v2/docs/EVAL_FRAMEWORK_README.md` - This file

### Core Modules
- `prose_v2/src/eval/shared/evaluator.py` - SharedEvaluator class
- `prose_v2/src/eval/shared/metrics.py` - Metric computation functions
- `prose_v2/src/eval/accounting/useful_bytes.py` - UsefulBytesAccountant
- `prose_v2/src/eval/failure_attribution/attributor.py` - FailureAttributor
- `prose_v2/src/eval/neighbor_recall_report.py` - NeighborRecallReporter

### Runners
- `prose_v2/src/runners/comparison_runner.py` - FairComparisonRunner
- `prose_v2/src/runners/eval_orchestrator.py` - EvaluationOrchestrator
- `prose_v2/src/runners/burst_ablation.py` - Burst ablation study
- `prose_v2/src/runners/sticky_ablation.py` - Sticky TTL ablation
- `prose_v2/src/runners/regression_baseline.py` - Baseline regression
- `prose_v2/src/runners/validate_acceptance_gates.py` - Gate validation

### Tests
- `prose_v2/tests/metrics/test_shared_evaluator.py` - Unit tests (22 tests)

### Outputs
- `outputs/reports/baseline_after_scheduler_fix.json` - Frozen baseline
- `outputs/reports/acceptance_gates_validation.json` - Gate check results

## Answering the Four Questions

### Q1: Does neighbor recall produce useful promotions?

**Component**: NeighborRecallReporter  
**Output**: `outputs/reports/neighbor_recall_report.json`

Look for:
- `ultimately_useful.count` > 0
- `funnel.promotion_to_useful.yield` > 0

### Q2: Does promotion improve Conditional Recovery and No-Miss Rate?

**Component**: SharedEvaluator  
**Output**: `evaluation_result.json`

Look for:
- `conditional_recovery` > baseline
- `no_miss_rate` > baseline

### Q3: What fraction of promoted bytes are useful?

**Component**: UsefulBytesAccountant  
**Output**: `useful_bytes_report.json`

Look for:
- `modes.attention_access_based.upr`
- `modes.gold_overlap_based.upr`
- `modes.recovery_event_based.upr`

### Q4: Is burst truly efficient?

**Component**: Burst Ablation  
**Output**: `burst_ablation.json`

Look for:
- `burst_gain_analysis.radius_1_vs_no_burst.burst_gain_per_mb`
- Positive = efficient, Negative = expensive

## Integration Example

```python
from prose_v2.src.config import ProSEXv2Config
from prose_v2.src.runners.eval_orchestrator import EvaluationOrchestrator

# Create config
config = ProSEXv2Config(seed=42)

# Create orchestrator
orchestrator = EvaluationOrchestrator(
    config=config,
    output_dir="outputs",
    experiment_id="my_experiment"
)

# During simulation
for step in simulation:
    # ... run pipeline ...
    
    context = EvaluationContext(
        config=config,
        step=step,
        gold_chunk_ids=gold_ids,
        visible_chunk_ids=visible_ids,
        # ... other fields
    )
    
    orchestrator.record_step(context, pipeline_result)

# Generate reports
reports = orchestrator.finalize()

# Run ablations
burst_report = orchestrator.run_burst_ablation(run_workload)
sticky_report = orchestrator.run_sticky_ablation(run_workload)
```

## Non-Negotiable Rules

1. **Metric definitions frozen** - See `metrics_spec.md`
2. **No benchmark-specific branches** - Shared evaluator only
3. **No oracle during inference** - Gold labels eval-only
4. **Promoted ≠ useful** - Explicit criteria required
5. **All runs reported** - Not just best-case
6. **No hidden negatives** - All failures visible
7. **Fixed denominator** - Document any changes
8. **Shared evaluator** - All experiments use same code
9. **JSON + Summary** - Machine and human readable
10. **Reproducible** - Config + script = same results

## Support

For questions about the framework:
1. Check `metrics_spec.md` for definitions
2. Check `evaluation_framework_summary.md` for usage
3. Run unit tests to verify components
4. Validate acceptance gates
