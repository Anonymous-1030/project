#!/usr/bin/env python3
"""
GPU Long-Context Expansion Analysis for ProSE (2x2 layout)

Demonstrates how ProSE eliminates the "long-context illusion" on
consumer (RTX 3090/4090/5090) and datacenter (A100) GPUs.
Layout: 2 rows x 2 cols, single-column friendly.
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. GPU specs
# ---------------------------------------------------------------------------
GPUS = [
    {"name": "RTX 3090",  "vram_gb": 24,  "bw_gbps": 936,  "mem_type": "GDDR6X", "price_usd": 800,  "tdp_w": 350},
    {"name": "RTX 4090",  "vram_gb": 24,  "bw_gbps": 1008, "mem_type": "GDDR6X", "price_usd": 1600, "tdp_w": 450},
    {"name": "RTX 5090",  "vram_gb": 32,  "bw_gbps": 1792, "mem_type": "GDDR7",  "price_usd": 2000, "tdp_w": 575},
    {"name": "A100 40GB", "vram_gb": 40,  "bw_gbps": 1555, "mem_type": "HBM2",   "price_usd": 8000, "tdp_w": 400},
    {"name": "A100 80GB", "vram_gb": 80,  "bw_gbps": 2039, "mem_type": "HBM2e",  "price_usd": 12000,"tdp_w": 400},
]
gpu_names = [g["name"] for g in GPUS]

# ---------------------------------------------------------------------------
# 2. Model & KV-cache assumptions
# ---------------------------------------------------------------------------
# Llama 3.1 8B: 32 layers, 8 KV heads, 128 head_dim
L_LAYERS, N_KV_HEADS, HEAD_DIM = 32, 8, 128
BYTES_FP16 = 2
KV_BYTES_PER_TOKEN = 2 * L_LAYERS * N_KV_HEADS * HEAD_DIM * BYTES_FP16
KV_GB_PER_TOKEN = KV_BYTES_PER_TOKEN / (1024**3)

MODEL_FP16_GB = 16.0
MODEL_INT4_GB = 4.5
META_OVERHEAD = 0.08
CPU_OFFLOAD_BW = 64  # GB/s, PCIe 4.0 x16 effective for CPU-GPU bulk transfer

BUDGETS = [0.05, 0.10, 0.20]
RECOVERY = [0.52, 0.68, 0.82]
C_BUDGET = ["#feb24c", "#fd8d3c", "#f03b20"]  # light, med, dark orange

# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------
def max_ctx_full(vram_gb, weight_gb):
    avail = max(0, vram_gb - weight_gb)
    return int(avail / KV_GB_PER_TOKEN)

def max_ctx_prose(vram_gb, weight_gb, budget, recovery):
    avail = max(0, vram_gb - weight_gb)
    prose_gb_per_tok = KV_GB_PER_TOKEN * budget * (1 + META_OVERHEAD)
    raw_len = int(avail / prose_gb_per_tok)
    return int(raw_len * recovery)

def min_budget_for_context(vram_gb, weight_gb, target_tokens):
    avail = max(0, vram_gb - weight_gb)
    req = (target_tokens * KV_GB_PER_TOKEN * (1 + META_OVERHEAD)) / avail
    return min(req, 1.0)

# ---------------------------------------------------------------------------
# 4. Run analysis
# ---------------------------------------------------------------------------
results = []
for gpu in GPUS:
    vram = gpu["vram_gb"]
    full_fp16 = max_ctx_full(vram, MODEL_FP16_GB)
    full_int4 = max_ctx_full(vram, MODEL_INT4_GB)
    entry = {
        "gpu": gpu["name"], "vram_gb": vram,
        "full_kv_fp16_k": round(full_fp16/1000, 1),
        "full_kv_int4_k": round(full_int4/1000, 1),
        "prose_fp16": [], "prose_int4": []
    }
    for b, r in zip(BUDGETS, RECOVERY):
        eff_fp16 = max_ctx_prose(vram, MODEL_FP16_GB, b, r)
        eff_int4 = max_ctx_prose(vram, MODEL_INT4_GB, b, r)
        entry["prose_fp16"].append({"budget_pct": int(b*100), "eff_k": round(eff_fp16/1000, 1)})
        entry["prose_int4"].append({"budget_pct": int(b*100), "eff_k": round(eff_int4/1000, 1)})
    results.append(entry)

OUT_DIR = "outputs/hpca_fair_hardware"
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "gpu_context_expansion.json"), "w") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------------------
# 5. Figure: 2x2 layout
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 300,
})

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
fig.subplots_adjust(hspace=0.35, wspace=0.30)

C_FULL = "#4a4a4a"

# ===== (a) Max Context: FP16 vs INT4 (grouped bars) =====
ax = axes[0, 0]
x = np.arange(len(gpu_names))
width = 0.22

# INT4 effective context with 10% budget
int4_10 = [r["prose_int4"][1]["eff_k"] for r in results]
# FP16 effective context with 10% budget
fp16_10 = [r["prose_fp16"][1]["eff_k"] for r in results]
# Full-KV INT4
full_int4 = [r["full_kv_int4_k"] for r in results]
# Full-KV FP16
full_fp16 = [r["full_kv_fp16_k"] for r in results]

ax.bar(x - 1.5*width, full_fp16, width, label="Full-KV FP16", color=C_FULL, edgecolor="black", linewidth=0.5, alpha=0.9)
ax.bar(x - 0.5*width, full_int4, width, label="Full-KV INT4", color="#888888", edgecolor="black", linewidth=0.5, alpha=0.9)
ax.bar(x + 0.5*width, fp16_10, width, label="ProSE 10% FP16", color=C_BUDGET[1], edgecolor="black", linewidth=0.5, alpha=0.85)
ax.bar(x + 1.5*width, int4_10, width, label="ProSE 10% INT4", color="#d94801", edgecolor="black", linewidth=0.5, alpha=0.85)

# annotate tallest bars
for xi, v in zip(x, int4_10):
    ax.text(xi + 1.5*width, v + v*0.02, f"{v:.0f}K", ha="center", va="bottom", fontsize=7, rotation=90, fontweight="bold")
for xi, v in zip(x, full_fp16):
    ax.text(xi - 1.5*width, v + v*0.02, f"{v:.0f}K", ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(gpu_names, rotation=20, ha="right")
ax.set_ylabel("Max Context (K tokens)")
ax.set_title("(a) Context Capacity: Full-KV vs ProSE 10%")
ax.set_ylim(0, max(int4_10) * 1.25)
ax.legend(loc="upper left", framealpha=0.95, edgecolor="gray", ncol=2, columnspacing=0.6)
ax.grid(axis="y", linestyle="--", alpha=0.35)

# ===== (b) Absolute Latency @ 128K Context (all GPUs) =====
ax = axes[0, 1]

def lat_full_kv_generic(ctx_len, gpu, weight_gb):
    avail = max(0, gpu["vram_gb"] - weight_gb)
    kv_gb = ctx_len * KV_GB_PER_TOKEN
    base = 80
    if kv_gb <= avail:
        return base + kv_gb * 1000 / gpu["bw_gbps"] * 1000
    else:
        offload = kv_gb - avail
        return base + (avail * 1000 / gpu["bw_gbps"] * 1000) + (offload * 1000 / CPU_OFFLOAD_BW * 1000)

def lat_prose_generic(ctx_len, gpu, weight_gb, budget):
    avail = max(0, gpu["vram_gb"] - weight_gb)
    prose_kv = ctx_len * KV_GB_PER_TOKEN * budget * (1 + META_OVERHEAD)
    base = 80 / 3.8
    if prose_kv <= avail:
        return base + prose_kv * 1000 / gpu["bw_gbps"] * 1000
    else:
        offload = prose_kv - avail
        return base + (avail * 1000 / gpu["bw_gbps"] * 1000) + (offload * 1000 / CPU_OFFLOAD_BW * 1000)

ctx_fixed = 128000
x = np.arange(len(gpu_names))
width = 0.18

lat_full = [lat_full_kv_generic(ctx_fixed, g, MODEL_INT4_GB) / 1000 for g in GPUS]
lat_p05  = [lat_prose_generic(ctx_fixed, g, MODEL_INT4_GB, 0.05) / 1000 for g in GPUS]
lat_p10  = [lat_prose_generic(ctx_fixed, g, MODEL_INT4_GB, 0.10) / 1000 for g in GPUS]
lat_p20  = [lat_prose_generic(ctx_fixed, g, MODEL_INT4_GB, 0.20) / 1000 for g in GPUS]

bars_full = ax.bar(x - 1.5*width, lat_full, width, label="Full-KV", color=C_FULL, edgecolor="black", linewidth=0.5)
bars_05   = ax.bar(x - 0.5*width, lat_p05,  width, label="ProSE 5%",  color=C_BUDGET[0], edgecolor="black", linewidth=0.5)
bars_10   = ax.bar(x + 0.5*width, lat_p10,  width, label="ProSE 10%", color=C_BUDGET[1], edgecolor="black", linewidth=0.5)
bars_20   = ax.bar(x + 1.5*width, lat_p20,  width, label="ProSE 20%", color=C_BUDGET[2], edgecolor="black", linewidth=0.5)

for bar in bars_full:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f"{bar.get_height():.1f}",
            ha="center", va="bottom", fontsize=7, fontweight="bold")
for bar in bars_10:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f"{bar.get_height():.1f}",
            ha="center", va="bottom", fontsize=7, fontweight="bold", color=C_BUDGET[1])

ax.set_xticks(x)
ax.set_xticklabels(gpu_names, rotation=20, ha="right")
ax.set_ylabel("Per-Token Latency (ms) @ 128K context")
ax.set_title("(b) Latency Comparison @ 128K Context (INT4)")
ax.set_ylim(0, max(lat_full) * 1.2)
ax.legend(loc="upper right", framealpha=0.95, edgecolor="gray", ncol=2, columnspacing=0.6)
ax.grid(axis="y", linestyle="--", alpha=0.35)

# Annotation omitted; y-axis label and bar heights are self-explanatory

# ===== (c) Min Budget for Target Context (INT4) =====
ax = axes[1, 0]
TARGETS = [128000, 256000, 512000, 1024000]
T_LABELS = ["128K", "256K", "512K", "1M"]
T_COLORS = ["#1a9850", "#d73027", "#4575b4", "#54278f"]
width = 0.18

for i, (tgt, lbl, c) in enumerate(zip(TARGETS, T_LABELS, T_COLORS)):
    budgets = [min_budget_for_context(g["vram_gb"], MODEL_INT4_GB, tgt) * 100 for g in GPUS]
    # Draw OOM bars at 100% with light gray hatch; feasible bars with normal color
    heights = [v if v < 100 else 100 for v in budgets]
    bar_colors = [c if v < 100 else "#cccccc" for v in budgets]
    bars = ax.bar(x + (i - 1.5) * width, heights, width,
                  label=f"{lbl}", color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, budgets):
        if val < 100:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2.5,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

# Bottom OOM annotation per GPU
for xi, g in zip(x, GPUS):
    oom_count = sum(1 for tgt in TARGETS if min_budget_for_context(g["vram_gb"], MODEL_INT4_GB, tgt) >= 1.0)
    if oom_count > 0:
        label = f"OOM ({oom_count})" if oom_count > 1 else "OOM"
        ax.text(xi, -6, label, ha="center", va="top", fontsize=8.5, fontweight="bold", color="darkred")

ax.set_ylim(-12, 118)

for b, col in zip(BUDGETS, C_BUDGET):
    ax.axhline(b * 100, color=col, linestyle="--", linewidth=0.8, alpha=0.4)

ax.set_xticks(x)
ax.set_xticklabels(gpu_names, rotation=20, ha="right")
ax.set_ylabel("Min KV Budget Required (%)")
ax.set_title("(c) Budget to Fit Target Context (INT4)")
ax.set_ylim(-15, 115)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.95, edgecolor="gray", title="Target")
ax.grid(axis="y", linestyle="--", alpha=0.35)

# ===== (d) Power Efficiency: K tokens / 100W =====
ax = axes[1, 1]
# Power efficiency = effective context length / (TDP / 100)
tdp_100w = [g["tdp_w"] / 100 for g in GPUS]

full_eff_p = [r["full_kv_int4_k"] / t for r, t in zip(results, tdp_100w)]
eff_5_p = [r["prose_int4"][0]["eff_k"] / t for r, t in zip(results, tdp_100w)]
eff_10_p = [r["prose_int4"][1]["eff_k"] / t for r, t in zip(results, tdp_100w)]
eff_20_p = [r["prose_int4"][2]["eff_k"] / t for r, t in zip(results, tdp_100w)]

x = np.arange(len(gpu_names))
width = 0.20

bars_full = ax.bar(x - 1.5*width, full_eff_p, width, label="Full-KV", color=C_FULL, edgecolor="black", linewidth=0.5)
bars_5 = ax.bar(x - 0.5*width, eff_5_p, width, label="ProSE 5%", color=C_BUDGET[0], edgecolor="black", linewidth=0.5)
bars_10 = ax.bar(x + 0.5*width, eff_10_p, width, label="ProSE 10%", color=C_BUDGET[1], edgecolor="black", linewidth=0.5)
bars_20 = ax.bar(x + 1.5*width, eff_20_p, width, label="ProSE 20%", color=C_BUDGET[2], edgecolor="black", linewidth=0.5)

# annotate ProSE bars
for bars in [bars_5, bars_10, bars_20]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + h*0.03,
                f"{h:.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(gpu_names, rotation=20, ha="right")
ax.set_ylabel("Efficiency (K tokens / 100W)")
ax.set_title("(d) Context Efficiency per Watt (INT4)")
ax.set_ylim(0, max(eff_5_p) * 1.22)
ax.legend(loc="upper right", framealpha=0.95, edgecolor="gray", ncol=2, columnspacing=0.6)
ax.grid(axis="y", linestyle="--", alpha=0.35)

# Annotation omitted; title and bar heights are self-explanatory

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gpu_context_expansion.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "gpu_context_expansion.png"), bbox_inches="tight")
print("[Saved] gpu_context_expansion.pdf & .png")

# ---------------------------------------------------------------------------
# 6. Console summary
# ---------------------------------------------------------------------------
print("\n" + "="*75)
print("GPU LONG-CONTEXT EXPANSION  (Llama 3.1 8B, INT4 weights ≈ 4.5 GB)")
print("="*75)
for r in results:
    print(f"\n{r['gpu']:12s}  VRAM={r['vram_gb']:2d}GB")
    print(f"  Full-KV:  FP16={r['full_kv_fp16_k']:>6.1f}K  INT4={r['full_kv_int4_k']:>6.1f}K")
    for s in r["prose_int4"]:
        print(f"  ProSE {s['budget_pct']:2d}% INT4:  eff={s['eff_k']:>7.1f}K")
print("="*75)
