# ProSE-X 2.0 Metrics Specification

## NON-NEGOTIABLE RULES

1. **Metric definitions are frozen** - Do not change definitions to improve results.
2. **No benchmark-specific branches** - Use shared evaluator for all experiments.
3. **No oracle information during online inference** - Gold labels only in evaluation paths.
4. **Promoted bytes are NOT automatically useful** - Must satisfy explicit usefulness criteria.
5. **Report all runs** - Not just best-case.
6. **No hidden negative results** - All failures must be visible.
7. **Fixed granularity and denominator** - Document any changes explicitly.
8. **Shared evaluator** - All metrics computed by one module.
9. **Machine-readable logs** - JSON + human summary.
10. **Full reproducibility** - From config + script only.

---

## Core Metric Definitions

### A. Conditional Recovery

**Definition:**
Among cases where gold evidence is retained in tail/visible memory universe, how often does the system successfully make it visible/usable through promotion or existing visibility?

**Formula:**
```
Conditional Recovery = (# steps where gold is recovered | gold exists in universe) / (# steps where gold exists in universe)
```

**Computation:**
1. For each step, determine if gold chunks exist in the tail/promoted universe
2. Check if any gold chunk is in the final visible set (anchors + promoted)
3. Count as success if at least one gold chunk is visible
4. Denominator is steps where gold exists somewhere in the system (not evicted)
5. Numerator is steps where gold is visible to attention

**Important:**
- This is a CONDITIONAL metric - only computed over steps where gold exists
- Gold must be actually visible, not just promoted and stuck in sticky limbo
- Must not include steps where gold was never in the system

---

### B. No-Miss Rate

**Definition:**
Fraction of steps where no evidence miss occurs for the relevant gold unit(s).

**Formula:**
```
No-Miss Rate = (# steps with no miss) / (total steps)
```

**Miss Definition:**
A "miss" occurs when:
1. Gold evidence is needed at this step (determined by evaluation)
2. Gold evidence exists in the system (not evicted)
3. Gold evidence is NOT visible to attention

**Computation:**
1. For each step, check if gold is needed
2. If gold is needed but not visible → count as miss
3. No-Miss Rate = 1 - (miss_count / total_steps)

**Important:**
- Different from Conditional Recovery (which conditions on gold existing)
- No-Miss Rate is over ALL steps
- A step can have no miss either because (a) gold not needed, or (b) gold visible

---

### C. Useful Promote Ratio (UPR)

**Definition:**
Useful promoted bytes / total promoted bytes

**Formula:**
```
UPR = useful_promoted_bytes / total_promoted_bytes
```

**When is a promoted byte "useful"?**

A promoted byte may be counted as useful ONLY if it satisfies at least one explicit criterion:

1. **Attention Access Based** (`attention_access_based`)
   - Chunk receives attention weight > threshold in subsequent steps
   - Must be measured over a window (e.g., next 10 steps)
   - Threshold: attention weight > 0.01 (1% of average)

2. **Gold Overlap Based** (`gold_overlap_based`) - EVALUATION ONLY
   - Chunk overlaps with gold evidence region
   - Overlap computed as intersection of token ranges
   - Must be > 0 tokens overlap
   - **NEVER used in online decisions**

3. **Recovery Event Based** (`recovery_event_based`)
   - Chunk participates in a step where recovery outcome improves
   - Must show causal link: promotion → visibility → successful generation
   - Requires step-by-step causal tracing

**Important:**
- NEVER assume promoted = useful
- Report UPR separately for each accounting mode
- Do not merge modes silently

---

### D. Burst Gain

**Definition:**
Incremental recovery benefit attributable to burst expansion divided by the additional promoted bytes or active bytes introduced by burst.

**Formula:**
```
Burst Gain = (recovery_with_burst - recovery_without_burst) / (bytes_with_burst - bytes_without_burst)
```

**Unit:** Recovery percentage points per additional MB

**Computation:**
1. Run system with burst enabled → get recovery rate R_b and bytes B_b
2. Run system with burst disabled → get recovery rate R_n and bytes B_n
3. Burst Gain = (R_b - R_n) / (B_b - B_n) * 1,000,000 (to get per MB)

**Interpretation:**
- Positive: Burst helps recovery more than it costs
- Negative: Burst hurts (costs more than helps)
- Zero: Burst has no effect

**Important:**
- Must include BOTH benefit AND overhead
- Do not report recovery without byte overhead
- Compare with same seed, workload, budget

---

### E. Candidate Recall@K

**Definition:**
Fraction of gold chunks that appear in the top-K candidates from ULF.

**Formula:**
```
Candidate Recall@K = |gold_chunks ∩ top_k_candidates| / |gold_chunks|
```

**Computation:**
1. Get gold chunk IDs for current step (evaluation only)
2. Get top-K candidates from ULF output
3. Count intersection
4. Divide by total gold chunks

**Important:**
- This is BEFORE scoring/scheduler
- Measures ULF recall quality only
- K values: 1, 5, 10, 20

---

### F. Budget Utilization

**Definition:**
Fraction of promotion budget actually used for transfers.

**Formula:**
```
Budget Utilization = used_bytes / budget_bytes
```

**Components:**
- `budget_bytes`: Maximum bytes allowed for promotion this step
- `used_bytes`: Actual bytes transferred (after burst expansion)

**Important:**
- Non-zero budget utilization is required for promotion to work
- Low utilization (< 10%) indicates scheduler or candidate issues

---

### G. Queue Contribution Breakdown

**Definition:**
Per-queue statistics showing what each recall queue contributes.

**Metrics per queue:**
- `raw_output_count`: Candidates before dedup
- `post_dedup_count`: Candidates after dedup
- `entering_scorer`: Candidates reaching scorer
- `surviving_scheduler`: Candidates selected by scheduler
- `burst_expanded`: Additional chunks from burst expansion
- `ultimately_useful`: Chunks satisfying usefulness criteria

**Queues tracked:**
1. `anchor_neighbor`: Anchor-neighbor recall
2. `lexical_overlap`: Lexical/entity overlap
3. `structural_recency`: Structural/recency
4. `historical_success`: Historical promotion success

---

## Failure Attribution Metrics

### Dominant Failure Reason

**Rule:** Every miss is assigned exactly ONE dominant reason.

**Allowed reasons:**
1. `candidate_miss` - Gold never made it to candidate set (ULF failure)
2. `scorer_rank_miss` - Gold in candidates but ranked too low
3. `scheduler_budget_drop` - Gold ranked well but cut by budget
4. `scheduler_threshold_drop` - Gold cut by score threshold
5. `burst_boundary_miss` - Gold near but outside burst radius
6. `sticky_eviction_miss` - Gold promoted but evicted by TTL
7. `promoted_but_unused` - Gold promoted but not accessed
8. `retention_miss` - Gold evicted from system entirely

**Assignment Rules:**
- Assign the EARLIEST stage where failure could have been prevented
- If gold never candidate → `candidate_miss`
- If gold candidate but score low → `scorer_rank_miss`
- If gold high score but not selected → check budget vs threshold
- Exactly one reason per miss (no multi-label)

---

## Sticky TTL Metrics

### Re-promotion Count

**Definition:**
Number of times a chunk is promoted multiple times due to TTL expiration.

**Formula:**
```
Re-promotion Count = total_promotions - unique_chunks_promoted
```

### Average Residency

**Definition:**
Average number of steps a chunk remains promoted (sticky).

**Formula:**
```
Avg Residency = sum(residency_duration) / unique_chunks_promoted
```

### Eviction Count

**Definition:**
Number of chunks evicted due to TTL expiration (not forced eviction).

---

## Latency Metrics

### Per-Component Latency

Track latency for each pipeline stage:
- `ulf_latency_us`: Multi-Queue Recall ULF
- `scorer_latency_us`: Oracle-Distilled Utility Scorer
- `scheduler_latency_us`: EABS scheduler
- `burst_latency_us`: Burst expansion
- `sticky_latency_us`: Sticky TTL management

### Aggregate Latency

- `mean_latency_us`: Mean across all steps
- `p50_latency_us`: 50th percentile
- `p95_latency_us`: 95th percentile
- `p99_latency_us`: 99th percentile

---

## Reporting Formats

### Machine-Readable (JSON)

All metrics must be output as structured JSON:

```json
{
  "experiment_id": "string",
  "config": { ... },
  "metrics": {
    "conditional_recovery": 0.75,
    "no_miss_rate": 0.82,
    "useful_promote_ratio": {
      "attention_access_based": 0.45,
      "gold_overlap_based": 0.38,
      "recovery_event_based": 0.52
    },
    "burst_gain": 0.03,
    ...
  },
  "per_step": [ ... ],
  "timestamp": "2026-03-24T13:18:00Z"
}
```

### Human-Readable (Markdown)

Summary tables and key findings in Markdown format.

---

## Version History

- v1.0.0 (2026-03-24): Initial frozen metric definitions
