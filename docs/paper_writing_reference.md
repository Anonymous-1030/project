# PROSE-X Paper Writing Reference: §II–§IV Implementation Details

## §II CEFE Architecture

### 2.1 Placement Argument

CEFE (Copy-Engine Front-End Enforcement) places the admission verdict at the last
pre-DMA hardware boundary: the Copy Engine issue queue. Once a 64 KB payload DMA
enters the CXL path, link time and endpoint occupancy cannot be reclaimed even if
the block is later found irrelevant. CEFE prevents this by binding the DMA issue
to a prior score commit.

Key distinction from software SBFI: host/GPU software paths pay coordination costs
(109 tok/s measured); CEFE's hardware-path verdict completes in <4 μs P99 under
32 concurrent streams, enabling 130 tok/s end-to-end.

### 2.2 Microarchitecture: 5-Stage PPU Pipeline

The Promotion Prediction Unit executes at the Copy Engine front-end:

| Stage | Function | Latency |
|-------|----------|---------|
| 1. MMRF Receive | FP16→Q0.15 format cast from GPU reduction kernel | 1 cycle |
| 2. Counter Update | Attention mass counter update | configurable |
| 3. Feature Extract | 4D quantized feature extraction | configurable |
| 4. LUT Lookup | Utility LUT read for admission decision | configurable |
| 5. DMA Enqueue | DMA request enqueue to memory controller | configurable |

Throughput: 1 candidate per initiation interval (= max stage cost).
Total latency for N candidates: N × initiation_interval + Σ(stage_costs).

### 2.3 CXL Link Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Effective bandwidth | 64.0 GB/s | 16 lanes × 64 GT/s / 8 × 0.98 |
| Protocol overhead | 2% | CXL.mem TLP encapsulation |
| Flit size | 256 bytes | CXL 3.0 spec |
| Queue depth | 48 | ASIC-class controller |
| Protocol processing latency | 15 ns | — |
| CXL bridge traversal | 50 ns | — |

### 2.4 DRAM Backend (DDR5-4800)

| Parameter | Value |
|-----------|-------|
| Row hit latency (tCAS) | 40 ns |
| Row miss latency (tRP + tRCD + tCAS) | 120 ns |
| Row buffer hit rate | 30% |
| Bank groups | 4 |
| Sustained bandwidth | 76.8 GB/s (2 channels) |

### 2.5 Service Time Model

Small reads (≤4 KB): per-burst access latency dominates.
Large reads (>4 KB): bandwidth-limited with one row-miss penalty.

```
Serialization = ceil(payload_bytes / 256) × flit_serialization_ns
Protocol = 2 × proto_proc_lat_ns + bridge_lat_ns
Queuing (M/D/1): W = (ρ × S) / (2 × (1 - ρ)), where ρ = λ × S
```

Fetch modes:
- Summary fetch: 64 bytes/chunk (score-before-fetch metadata)
- Payload fetch: 65,536 bytes/chunk (full KV cache block)

### 2.6 Concurrency and Backpressure

- 32-stream concurrent admission tested
- Single-stream CEFE latency: ~2.2 μs
- 32-stream P99: <4 μs
- Backpressure: when queue depth exceeds threshold (>8), ABA switches to
  AGGRESSIVE mode, applying compression and reducing DMA volume

### 2.7 Area and Power (7nm)

| Component | Area (mm²) | Power (mW) | Latency (ns) |
|-----------|-----------|------------|--------------|
| Admission Predictor (256-entry LUT + comparator) | 0.012 | 15 | 2.5 |
| Prefetch DMA Engine (4-entry FIFO + FSM) | 0.008 | 12 | 1.5 |
| Near-CXL Decompressor (64-lane INT4→FP16) | 0.045 | 45 | 5.0 |
| Metadata SRAM (4 KB, 512 chunks × 8B) | 0.006 | 8 | 1.5 |
| **Total** | **0.071** | **80** | — |

Overhead relative to H100: area 0.0087% (of 814 mm²), power 0.011% (of 700 W).

---

## §III HES Scorer

### 3.1 Design Rationale

Current SBFI uses content-agnostic evidence (SimHash/Bloom) that cannot capture
semantic relevance. HES deploys a tiny neural scorer on the CXL controller side,
using quantized micro-embeddings for semantic matching at zero additional PCIe
bandwidth cost.

### 3.2 Micro-Embedding Store

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 12 |
| Quantization | INT4 (4 bits per value) |
| Storage per chunk | 48 bits = 6 bytes |
| Max chunks in SRAM | 2048 |
| Total SRAM budget | 96 KB |

Quantization formula:
```
quantized = round((clip(vec, -1.0, 1.0) + 1.0) × 7.5)   → maps to [0, 15]
```

Dequantization:
```
dequantized = (quantized / 7.5) - 1.0                     → maps back to [-1, 1]
```

### 3.3 TinyMLP Architecture

```
Input (20D) = concat(micro_embedding[12D], query_signature[8D])
    ↓
Hidden (16 units, ReLU)
    ↓
Output (1 unit, sigmoid) → neural_score ∈ [0, 1]
```

Total parameters: ~2,000 (fits in <8 KB).
Inference: single forward pass per candidate, batchable.

### 3.4 Scoring Pipeline (per query)

1. Extract query signature (8D, L2-normalized)
2. For each candidate chunk:
   a. Retrieve INT4 micro-embedding from CXL SRAM
   b. Dequantize to float32
   c. Concatenate with query signature → 20D feature vector
   d. Forward through TinyMLP → neural_score
   e. Compute SimHash baseline: cosine(chunk_sig[:8], query_sig[:8]), mapped to [0,1]
   f. Combined score = 0.6 × neural_score + 0.4 × simhash_score
3. Admit if combined_score > 0.3

### 3.5 Offline Distillation

Trigger: during prefill or periodic background update.
Training data: full attention weights from transformer layers.

```
Targets: attention_mass[chunk] / max(attention_masses)  → normalized to [0, 1]
Loss: MSE(MLP_output, target)
Optimizer: SGD, lr=0.005, epochs=30 (per distillation round)
```

The MLP learns to predict which chunks receive high attention mass,
enabling semantic admission without full attention computation at decode time.

### 3.6 Capacity Planning

At 128K context (256 chunks of 512 tokens):
- Micro-embedding storage: 256 × 6B = 1.5 KB
- Well within 96 KB SRAM budget
- Supports up to 2048 chunks (1M token context at 512 tok/chunk)

At 2048 chunks (max capacity):
- Storage: 2048 × 6B = 12 KB
- Remaining SRAM: 84 KB (available for MLP weights, counters, metadata)

### 3.7 Key Experimental Results (from pcm_ablation)

HES-only recall vs static baselines (8 GB/s, 8192 tokens):

| Budget Ratio | HES (Scored) | Static Random | Static Position |
|-------------|-------------|---------------|-----------------|
| 0.05 | **38.5%** | 10.5% | 0.0% |
| 0.10 | **45.0%** | 30.0% | 25.0% |
| 0.15 | **68.0%** | 40.5% | 50.0% |
| 0.20 | **70.0%** | 58.0% | 50.0% |
| 0.25 | **70.0%** | 71.5% | 75.0% |

Key insight: at tight budgets (≤0.15), HES provides 1.3–3.7× recall lift over
unscored selection. Static masks plateau because their sparsity pattern is
query-agnostic; HES converts budget into semantic recovery.

### 3.8 Limitations (to acknowledge in paper)

- Static-position mask beats HES at 32 GB/s, 4K context (50% vs 39.5%) for
  workloads with fixed-position critical tokens (e.g., passkey tasks).
- HES provides general query-aware scoring without hand-tuned heuristics;
  it is a replaceable scorer instantiation, not a hardwired dependency.
- Better scorers improve recall without changing the enforcement mechanism.

---

## §IV ABA Mode Selection

### 4.1 Design Rationale

At high bandwidth (>16 GB/s), full-KV fetch latency is hidden by compute.
PROSE's value shifts from reducing DMA volume to reducing attention FLOPs.
At low bandwidth (<4 GB/s), DMA volume is the bottleneck.
ABA dynamically selects the enforcement mechanism based on which resource is scarce.

### 4.2 Mode Thresholds

| Mode | Bandwidth Condition | Queue Condition | Enforcement |
|------|-------------------|-----------------|-------------|
| TRANSPARENT | > 16 GB/s | depth < 4 | Scored sparse attention mask |
| HYBRID | 4–16 GB/s | — | Standard PROSE pipeline |
| AGGRESSIVE | < 4 GB/s | OR depth > 8 | Pipeline + compression + prefetch |

### 4.3 Hysteresis Mechanism

- Mode switch requires **3 consecutive steps** at the new mode before committing
- Bandwidth smoothed over 4-step moving average
- Queue depth smoothed over 4-step moving average
- Prevents oscillation at mode boundaries (e.g., bandwidth fluctuating around 16 GB/s)

### 4.4 TRANSPARENT Mode: Scored Sparse Attention

When bandwidth is abundant, CEFE's DMA gating is vacuously satisfied (all fetches
are cheap). The system shifts enforcement to the compute boundary:

1. Score ALL tail chunks via HES (neural + simhash combined score)
2. Select top N chunks by score (N = total_chunks × budget_ratio × 2.5)
3. Inject selected chunk IDs as block-sparse mask into FlashAttention kernel
4. Remaining chunks excluded from QKV computation

Compute savings: (total_chunks - visible_chunks) × 512 × 128 FLOPs per step.

**Why this is NOT just "sparse attention":**
Static sparse masks (random or position-based) exhibit flat recall-budget curves
because their sparsity pattern is query-agnostic. Scored sparse attention binds
visibility to evidence score, converting compute budget into semantic recovery.

Experimental evidence (32 GB/s, 16K context):
- HES scored: 52.5% recall, 2.88× throughput
- Static random: 28.0% recall, 2.91× throughput
- Static position: 25.0% recall, 2.91× throughput

Near-identical throughput, but scored selection provides **1.9× recall lift**.

### 4.5 HYBRID Mode: Standard Pipeline

Standard PROSE promotion pipeline executes:
ULF → Scorer → Scheduler → Burst → Sticky

HES re-scores candidates after ULF filtering, expanding the visible set with
up to 3 additional high-score chunks per step.

### 4.6 AGGRESSIVE Mode: Compression + Prefetch

When bandwidth is critically scarce:
- Compression ratio: 0.5 (halves DMA payload via INT4/INT8 quantized transfer)
- Speculative prefetch: predicts next 3 chunks based on access recency (>0.3)
- Prefetch candidates: chunks with promoted_count > 0 and recent access

### 4.7 Mode Distribution (from experiments)

At 4 GB/s (pathological CXL saturation):
- Transparent: 0%
- Hybrid: 4%
- Aggressive: 96%

At 32 GB/s (high bandwidth):
- Transparent: ~96%
- Hybrid: ~4%
- Aggressive: 0%

Mode switches: typically 1 per 50-step decode sequence (stable after warmup).

### 4.8 Key Experimental Results

**Low bandwidth (4 GB/s) — DMA enforcement dominates:**

| Config | 4K Recall | 8K Recall | Throughput |
|--------|-----------|-----------|------------|
| Baseline | 0.0% | 25.0% | 2.05× |
| HES (scored) | **66.0%** | **45.0%** | 1.14× |
| Static random | 27.0% | 30.0% | 2.68× |

HES sacrifices throughput for recall — this is the correct tradeoff when
bandwidth is scarce and every DMA must count.

**High bandwidth (32 GB/s) — Compute enforcement dominates:**

| Config | 8K Recall | 16K Recall | Throughput |
|--------|-----------|------------|------------|
| Baseline | 25.0% | 50.0% | 2.09–2.18× |
| HES (scored) | **45.0%** | **52.5%** | 2.88–3.14× |
| Static random | 30.0% | 28.0% | 2.91× |

HES achieves both higher recall AND higher throughput than baseline —
scored sparse mask reduces FLOPs while maintaining semantic coverage.

---

## Paper Narrative Summary

**One sentence:** PROSE-X enforces score-before-fetch at the Copy Engine front-end;
when bandwidth is abundant and the DMA gate is vacuously satisfied, the same
scoring mechanism generates a block-sparse attention mask that converts compute
budget into semantic recovery.

**Architecture story (no formalism needed):**
1. CEFE placement: verdict binds before DMA issue (§II)
2. HES scorer: replaceable neural scorer on CXL SRAM, zero-PCIe-roundtrip (§III)
3. ABA binding: bandwidth-aware selection of enforcement mechanism (§IV)
4. Static mask baseline proves scored selection is necessary, not just sparsity (§IV.4)

**What NOT to claim:**
- Do NOT call it a "consistency model" or "PCM" — reviewers see this as overclaiming
- The contribution is CEFE placement + implementation path, not an abstract invariant
- Score-before-fetch is a simple ordering property; state it in one sentence, don't formalize it
- HES is one possible scorer; workload-specific heuristics may beat it on specific tasks
- Acknowledge static-position wins on locality-heavy workloads (passkey)
- Do NOT define axioms, state transitions, or visibility semantics unless you also define
  reorderings, races, revocation, fault handling, and driver/runtime/kernel interactions
  (which would be a separate paper)

---

## §V Discussion: Limitations and Deployment Path

### 5.1 HES Scorer Fragility (审稿人关切 #1)

**问题本质：** HES 是系统的精度瓶颈。ordering guarantee（score-before-fetch）与 scorer quality 解耦，但实际 recall 完全取决于 scorer 的好坏。

**论文中应承认的：**

> "HES is the accuracy bottleneck of the system. The ordering enforcement (CEFE)
> guarantees zero rejected-payload exposure regardless of scorer quality, but
> recall — the fraction of useful blocks that are admitted — is bounded by the
> scorer's ability to predict attention mass from 48-byte micro-embeddings."

**具体问题及回应策略：**

| 审稿人问题 | 回应 | 论文中写什么 |
|-----------|------|-------------|
| 微嵌入如何训练？ | Prefill 阶段从 layer-0 hidden states 蒸馏 | "Micro-embeddings are distilled once during prefill from layer-0 hidden states (30 epochs SGD, <1ms on CXL controller). No runtime retraining." |
| 如何跨租户版本管理？ | 每个 request 独立的 embedding namespace，SRAM 按 request_id 分区 | "Each request maintains an isolated embedding namespace in SRAM (max 2048 chunks per request). Multi-tenant isolation is achieved through SRAM partitioning, not versioning." |
| 领域迁移敏感度？ | 承认这是局限。SimHash fallback (40% weight) 提供 domain-agnostic baseline | "Domain shift degrades the neural component (60% weight) but the SimHash baseline (40% weight) provides content-agnostic fallback. Cross-domain accuracy is a known limitation; the scorer is designed to be replaceable." |
| 新鲜度衰减？ | Epoch-based invalidation。stale embedding 退化为 SimHash-only scoring | "Stale micro-embeddings (epoch expired) fall back to SimHash-only scoring (combined_score = 0.4 × simhash). This is a graceful degradation, not a failure mode — recall drops but no invalid blocks are admitted." |
| 与 oracle 的精度差距？ | 承认。给出数据：HES 75% recall at budget 0.30 vs oracle 100% | "The gap to oracle (75% vs 100% at budget 0.30) represents the cost of scoring from 48-byte summaries rather than full attention computation. Closing this gap requires richer metadata, not a different enforcement mechanism." |

**关键论述（一段话，放在 §III 末尾）：**

> "We deliberately separate enforcement correctness from scorer quality.
> CEFE guarantees that no payload DMA is issued without a positive verdict —
> this holds regardless of whether the scorer is HES, SimHash, or an oracle.
> HES's contribution is improving recall within this guarantee, not providing
> the guarantee itself. A better scorer (e.g., one trained on longer attention
> histories or using 96-byte embeddings) would improve recall without any
> change to the enforcement mechanism. The 25% recall gap to oracle is the
> price of scoring from 48-byte summaries; it is a scorer limitation, not
> an architecture limitation."

### 5.2 Hardware Integration Path (审稿人关切 #2)

**问题本质：** GPU 厂商对 memory controller / copy engine descriptor path 管控严格。论文的 placement argument 在逻辑上成立，但部署路径需要更有力的论证。

**论文中应承认的：**

> "Production deployment of CEFE requires cooperation with the GPU memory
> controller vendor. We do not claim that CEFE can be retrofitted to existing
> GPUs without vendor support."

**具体问题及回应策略：**

| 审稿人问题 | 回应 | 论文中写什么 |
|-----------|------|-------------|
| GPU 厂商管控严格 | 承认。但指出 CXL controller 是独立芯片，不在 GPU die 上 | "CEFE logic resides on the CXL memory controller (e.g., Samsung CMM-D, SK Hynix CMS), not on the GPU die. CXL controllers are designed by memory vendors with programmable firmware paths — the integration barrier is lower than modifying GPU silicon." |
| 虚拟内存交互 | CEFE 操作在物理地址空间，在 IOMMU 翻译之后 | "CEFE operates post-IOMMU translation on physical addresses. The verdict is issued after address resolution, avoiding virtual memory complications." |
| 空完成语义 | 被拒绝的 DMA 不发起，因此无完成事件需要处理 | "Rejected descriptors are never issued to the DMA engine; they are dropped at the verdict latch. No null-completion is generated — the descriptor simply does not enter the issue queue. From the driver's perspective, the transfer was never requested." |
| 流式顺序兼容性 | CEFE 不改变已发起 DMA 的顺序，只过滤哪些被发起 | "CEFE is a filter, not a reorderer. Admitted DMAs proceed in their original descriptor order. Stream ordering and fence semantics are preserved because CEFE only removes descriptors from the stream; it never reorders or delays admitted ones." |

**三种部署路径（按可行性排序）：**

**Path A: CXL Controller Firmware (最可行，当前论文的定位)**
- 逻辑部署在 CXL memory expander 的 controller 固件中
- 不需要修改 GPU
- 面积开销 0.071 mm² 在 CXL controller die 上可忽略
- 限制：只能拦截 CXL.mem 请求，不能拦截 GPU 内部的 HBM 访问

**Path B: SmartNIC/DPU Interposer (中等可行)**
- 在 PCIe/CXL 路径上放置 DPU（如 NVIDIA BlueField）
- DPU 固件实现 verdict logic
- 不需要修改 GPU 或 CXL controller
- 限制：增加一跳延迟（~1-2 μs）

**Path C: GPU Memory Controller Modification (最强但最难)**
- 需要 GPU 厂商合作
- 在 copy engine descriptor path 中加入 verdict latch
- 最低延迟，最高集成度
- 限制：需要 NVIDIA/AMD 合作，不现实作为学术贡献

**论文中的定位：**

> "This paper targets Path A: CXL controller firmware deployment. The verdict
> logic (0.071 mm², 80 mW) fits within the programmable region of production
> CXL controllers. Path C (GPU-integrated) would eliminate the CXL hop but
> requires vendor cooperation beyond the scope of this work. We note that
> recent CXL 3.0 controllers (Samsung CMM-D, Montage MXC) already include
> programmable processing elements for address translation and QoS — CEFE's
> logic is comparable in complexity to these existing functions."

### 5.3 Honest Limitations Summary (放在 §V 或 §VI)

建议在论文末尾用一段集中承认所有局限：

> **Limitations.** (1) HES recall is bounded by micro-embedding quality; the
> 25% gap to oracle at budget 0.30 is the cost of 48-byte summaries.
> Domain-shifted workloads degrade to SimHash-only scoring. (2) Production
> deployment requires CXL controller firmware support; GPU-side integration
> (Path C) is not addressed. (3) Evaluation uses SimCXL simulation, not
> silicon measurements; real CXL controller timing may differ. (4) Static
> position masks outperform HES on workloads with fixed-position critical
> tokens (e.g., passkey retrieval), where locality heuristics suffice.
> (5) Multi-tenant SRAM partitioning limits per-request embedding capacity
> to 2048 chunks; longer contexts require eviction or hierarchical storage.

