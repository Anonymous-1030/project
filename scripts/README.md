# ProSE-X 2.0 WSL vLLM 运行脚本

## 概述

这组脚本用于在 WSL (Windows Subsystem for Linux) 环境中使用 vLLM 运行 ProSE-X 2.0，并打印详细的性能指标。

## 文件说明

| 文件 | 说明 |
|------|------|
| `run_vllm_wsl.ps1` | PowerShell 脚本 (从 Windows 运行) |
| `run_vllm_wsl.sh` | Bash 脚本 (从 WSL 内部运行) |
| `../src/runners/run_vllm_metrics.py` | 主 Python 脚本 |

## 快速开始

### 从 Windows PowerShell 运行

```powershell
cd D:\LLM\prose_v2\scripts

# 基本运行
.\run_vllm_wsl.ps1 -ContextLength 4096

# 完整参数
.\run_vllm_wsl.ps1 `
    -Model "microsoft/Phi-3-mini-4k-instruct" `
    -ContextLength 8192 `
    -BudgetRatioPercent 10 `
    -WorkloadType "passkey" `
    -UseProSEv2 `
    -DetailedMetrics
```

### 从 WSL Bash 运行

```bash
cd /mnt/d/LLM/prose_v2/scripts

# 给执行权限
chmod +x run_vllm_wsl.sh

# 基本运行
./run_vllm_wsl.sh --context-length 4096

# 完整参数
./run_vllm_wsl.sh \
    --model "microsoft/Phi-3-mini-4k-instruct" \
    --context-length 8192 \
    --budget-ratio 0.1 \
    --use-prose-v2 \
    --detailed-metrics
```

### 直接运行 Python 脚本

```bash
# 在 WSL 中
cd /mnt/d/LLM
conda activate vllm

python -m prose_v2.src.runners.run_vllm_metrics \
    --model microsoft/Phi-3-mini-4k-instruct \
    --context-length 4096 \
    --budget-ratio 0.1 \
    --workload-type passkey \
    --use-prose-v2 \
    --detailed-metrics \
    --num-steps 20
```

## 详细指标输出

运行时会输出以下详细指标：

### 1. ULF (Multi-Queue Recall) 指标
- 候选数 / Tail 总数
- 每个队列的贡献数
- 召回率

### 2. AS (Adaptive Scorer) 指标
- 评分的候选数
- 分数分布 (均值、标准差、最大值)
- 黄金块排名 (如果有)

### 3. EABS (Scheduler) 指标
- 选择的块数 (exploit/explore 分割)
- 被丢弃的原因统计
- 预算利用率

### 4. BSP (Burst-and-Stick) 指标
- 爆发扩展统计
- Sticky 提升/过期/刷新数
- 平均 TTL

### 5. 延迟分解
- 各阶段耗时 (ULF/Scorer/Scheduler/Burst/Sticky)
- 总延迟

### 6. 恢复指标
- 黄金块召回 @K
- 恢复成功率

## 输出文件

运行结果保存为 JSON 格式：

```
outputs/vllm_wsl/
└── metrics_YYYYMMDD_HHMMSS.json
```

包含每步的完整指标数据。

## 故障排除

### 编码问题
如果看到乱码，请确保终端使用 UTF-8 编码：

```powershell
# PowerShell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

```bash
# Bash
export LANG=en_US.UTF-8
```

### 模块未找到
确保在项目根目录运行，且 PYTHONPATH 包含项目根目录。

### Conda 环境
确保 `vllm` conda 环境已创建并激活：

```bash
conda activate vllm
```

## 与真实 vLLM 集成

当前脚本使用模拟数据进行测试。要与真实 vLLM 集成：

1. 安装 vLLM: `pip install vllm`
2. 修改 `run_vllm_metrics.py` 中的模拟循环
3. 从 vLLM 获取真实的 query signatures
4. 将 ProSE-X 管道集成到 vLLM 的 decode 循环中

参考 `prose_stage2/prose_vllm/` 目录中的实际集成代码。
