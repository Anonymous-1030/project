# ProSE-X 2.0 vLLM WSL 运行指南

本指南介绍如何在 WSL (Windows Subsystem for Linux) 中使用 vLLM 运行 ProSE-X 2.0 并打印详细指标。

## 快速开始

### 方法1: 使用 PowerShell 脚本 (推荐)

从 Windows PowerShell 运行：

```powershell
# 进入项目目录
cd D:\LLM\prose_v2

# 运行基本测试
.\scripts\run_vllm_wsl.ps1 -ContextLength 4096 -BudgetRatioPercent 10

# 完整参数
.\scripts\run_vllm_wsl.ps1 `
    -Model "microsoft/Phi-3-mini-4k-instruct" `
    -ContextLength 8192 `
    -BudgetRatioPercent 10 `
    -WorkloadType "passkey" `
    -OutputDir "outputs/vllm_wsl" `
    -UseProSEv2 `
    -DetailedMetrics
```

### 方法2: 使用 Bash 脚本 (在 WSL 内部)

在 WSL 终端中运行：

```bash
# 进入项目目录
cd /mnt/d/LLM/prose_v2

# 给脚本执行权限
chmod +x scripts/run_vllm_wsl.sh

# 运行基本测试
./scripts/run_vllm_wsl.sh --context-length 4096

# 完整参数
./scripts/run_vllm_wsl.sh \
    --model "microsoft/Phi-3-mini-4k-instruct" \
    --context-length 8192 \
    --budget-ratio 0.1 \
    --workload-type passkey \
    --output-dir outputs/vllm_wsl \
    --num-steps 20

# 禁用 ProSE-X v2 (基线对比)
./scripts/run_vllm_wsl.sh --context-length 4096 --no-prose-v2
```

### 方法3: 直接运行 Python 脚本

```bash
# 激活 conda 环境
conda activate vllm

# 运行
python prose_v2/src/runners/run_vllm_metrics.py \
    --model microsoft/Phi-3-mini-4k-instruct \
    --context-length 4096 \
    --budget-ratio 0.1 \
    --use-prose-v2 \
    --detailed-metrics
```

## 输出指标说明

### 1. ULF (Multi-Queue Recall) 指标

```
[ULF] Multi-Queue Recall
  Candidates: 15/80 (18.8% recall rate)
  Queue Contributions:
    - anchor_neighbor: 5
    - lexical_overlap: 4
    - structural_recency: 3
    - historical_success: 3
  Latency: 45.23 μs
```

- **Candidates**: ULF 召回的候选块数 / Tail 总块数
- **Queue Contributions**: 每个队列贡献的候选数
- **Latency**: ULF 执行时间

### 2. AS (Adaptive Scorer) 指标

```
[AS] Utility Scoring
  Candidates Scored: 15
  Score Distribution: μ=0.652, σ=0.123, max=0.891
  Latency: 12.45 μs
```

- **Score Distribution**: 分数分布 (均值、标准差、最大值)
- **Gold Rank**: 黄金块在候选中的排名 (如果有)

### 3. EABS (Scheduler) 指标

```
[EABS] Exploration-Aware Scheduler
  Selected: 4 (exploit: 3, explore: 1)
  Dropped: budget=0, score=8, conf=3
  Budget Utilization: 78.5%
  Latency: 8.12 μs
```

- **Selected**: 选择的块数 (exploit vs explore 分割)
- **Dropped**: 因各种原因被丢弃的数量
  - `budget`: 超出预算
  - `score`: 分数过低
  - `conf`: 置信度过低
- **Budget Utilization**: 预算利用率

### 4. BSP (Burst-and-Stick) 指标

```
[BSP] Burst-and-Stick
  Burst: 4 → 7 (+3 expansion)
  Sticky: 7 promoted, 1 expired, 2 refreshed
  Avg TTL: 3.5
  Latency: burst=2.34 μs, sticky=1.89 μs
```

- **Burst**: 输入块数 → 爆发扩展后总数 (+扩展数)
- **Sticky**: 提升数 / 过期数 / 刷新数
- **Avg TTL**: 平均生存时间

### 5. Recovery 指标

```
[Recovery] Gold Chunk Tracking
  Recall: @1=✗, @5=✗, @10=✓, @20=✓
  Gold Rank: #8 (score: 0.723)
```

- **Recall @K**: 黄金块是否在前 K 个候选中
- **Gold Rank**: 黄金块的排名和分数

### 6. 最终汇总

```
[Aggregated Statistics]
  Avg ULF Recall Rate: 18.8%
  Avg Budget Utilization: 78.5%
  Avg Burst Expansion: 75.0%

[Gold Recovery]
  Steps with gold @10: 10/20
  Steps with gold @20: 18/20

[Latency Breakdown]
  ULF:        0.90 ms (35.2%)
  Scorer:     0.25 ms (9.8%)
  Scheduler:  0.16 ms (6.3%)
  Burst:      0.05 ms (2.0%)
  Sticky:     0.04 ms (1.6%)
  Total:      2.56 ms
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型名称或路径 | `microsoft/Phi-3-mini-4k-instruct` |
| `--context-length` | 上下文长度 (tokens) | `4096` |
| `--budget-ratio` | 提升预算比例 (0.0-1.0) | `0.1` (10%) |
| `--workload-type` | 工作负载类型 | `passkey` |
| `--num-steps` | 模拟的解码步数 | `20` |
| `--use-prose-v2` | 启用 ProSE-X v2 功能 | 启用 |
| `--detailed-metrics` | 打印每步详细指标 | 启用 |
| `--output-dir` | 输出目录 | `outputs/vllm_wsl` |

## 工作负载类型

### Passkey (默认)
在长上下文中隐藏一个 "passkey"，测试模型能否找到它。

### Multihop
需要多跳推理的问答任务。

### Needle
Needle-in-haystack 测试，在大量无关文本中找到关键信息。

## 预算比例影响

预算比例 (`--budget-ratio`) 控制每步可以提升多少块：

| 比例 | 说明 | 适用场景 |
|------|------|----------|
| 0.0 | 禁用提升 | 基线对比 |
| 0.05 | 5% Tail 预算 | 极低带宽 |
| 0.1 | 10% Tail 预算 | 推荐默认 |
| 0.2 | 20% Tail 预算 | 更高召回 |
| 0.5 | 50% Tail 预算 | 接近全KV |

## 输出文件

运行后会生成以下文件：

```
outputs/vllm_wsl/
└── metrics_YYYYMMDD_HHMMSS.json
```

JSON 文件包含：
- 每步的完整指标
- 队列贡献详情
- 分数分布
- 延迟分解
- 恢复统计

## 故障排除

### Conda 环境未找到

```bash
# 检查 conda 安装
which conda

# 如果未找到，手动初始化
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm
```

### 权限错误

```bash
# 给脚本添加执行权限
chmod +x prose_v2/scripts/run_vllm_wsl.sh
```

### Python 模块未找到

```bash
# 确保在项目根目录
cd /mnt/d/LLM/prose_v2

# 检查 Python 路径
python -c "import sys; print('\n'.join(sys.path))"
```

## 进阶使用

### 批量运行不同配置

```bash
#!/bin/bash
# run_sweep.sh

for budget in 0.0 0.05 0.1 0.2; do
    for ctx in 4096 8192 16384; do
        echo "Running: budget=$budget, context=$ctx"
        ./scripts/run_vllm_wsl.sh \
            --budget-ratio $budget \
            --context-length $ctx \
            --output-dir "outputs/sweep/b${budget}_c${ctx}"
    done
done
```

### 与真实 vLLM 集成

当前脚本使用模拟数据。要与真实 vLLM 集成：

1. 修改 `run_vllm_metrics.py` 中的 `run_vllm_simulation` 函数
2. 替换模拟循环为实际 vLLM 解码循环
3. 从 vLLM 获取真实的 query signatures 和 attention weights

参考 `prose_stage2/prose_vllm/runners/run_recovery_bench.py` 的实现。

## 相关文档

- [ProSE-X 2.0 README](../README.md) - 项目概述
- [Implementation Summary](../IMPLEMENTATION_SUMMARY.md) - 实现详情
- [prose_stage2/WSL_GUIDE.md](../../prose_stage2/WSL_GUIDE.md) - Stage 2 WSL 指南
