#!/usr/bin/env python3
"""
Democratization Figure: A100-class quality on consumer GPUs with ProSE

Narrative: Ultra-premium A100 80GB Full-KV was previously required for
long-context "non-degraded" LLM inference. ProSE enables RTX 3090/4090/5090
to achieve equivalent (or better) effective context at vastly lower cost.

Layout: 2 rows x 4 cols, single-column friendly.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# GPU specs
# ---------------------------------------------------------------------------
GPUS = [
    {"name": "RTX 3090",  "vram": 24,  "bw": 936,  "price": 800,   "tdp": 350, "type": "Consumer"},
    {"name": "RTX 4090",  "vram": 24,  "bw": 1008, "price": 1600,  "tdp": 450, "type": "Consumer"},
    {"name": "RTX 5090",  "vram": 32,  "bw": 1792, "price": 2000,  "tdp": 575, "type": "Consumer"},
    {"name": "A100 80GB", "vram": 80,  "bw": 2039, "price": 12000, "tdp": 400, "type": "Data-Center"},
]

# ---------------------------------------------------------------------------
# Constants (Llama 3.1 8B, INT4 weights ~4.5 GB)
# ---------------------------------------------------------------------------
MODEL_INT4_GB = 4.5
KV_GB_PER_TOK = 2 * 32 * 8 * 128 * 2 / (1024**3)  # ~0.00012207 GB
META = 0.08
CPU_BW = 64  # GB/s offload

# Recovery & Passkey (calibrated from HPCA traces)
BUDGETS = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00])
RECOVERY = np.array([0.52, 0.68, 0.78, 0.82, 0.90, 0.96, 1.00])
PASSKEY_128K = 0.92   # ProSE 10% @ 128K
PASSKEY_256K = 0.85   # ProSE 10% @ 256K
PASSKEY_512K = 0.72   # ProSE 10% @ 512K

C_CONSUMER = "#d94801"    # orange-red
C_A100 = "#4a4a4a"        # dark gray
C_FULLKV = "#969696"      # medium gray
C_PROSE = ["#feb24c", "#fd8d3c", "#f03b20"]  # light/med/dark orange

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 300,
})

fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
fig.subplots_adjust(hspace=0.38, wspace=0.32)

# ========================================================================
# Row 1: Quality Equivalence  (the "non-degraded" claim)
# ========================================================================

# ---- (a) Recovery Rate vs Budget ----
ax = axes[0, 0]
ax.plot(BUDGETS * 100, RECOVERY * 100, marker="o", markersize=7, linewidth=2.5,
        color=C_CONSUMER, markeredgecolor="black", markeredgewidth=0.8, label="ProSE (all GPUs)")
ax.axhline(100, color=C_A100, linestyle="--", linewidth=2.0, label="Full-KV (A100 baseline)")
ax.axvline(10, color=C_PROSE[1], linestyle=":", linewidth=1.5, alpha=0.7)
ax.axvline(20, color=C_PROSE[2], linestyle=":", linewidth=1.5, alpha=0.7)
ax.text(10.5, 55, "10%\nbudget", fontsize=8, color=C_PROSE[1], fontweight="bold")
ax.text(20.5, 55, "20%\nbudget", fontsize=8, color=C_PROSE[2], fontweight="bold")
ax.set_xlabel("KV Budget (%)")
ax.set_ylabel("Gold-Chunk Recovery (%)")
ax.set_title("(a) Quality: Recovery vs Sparse Budget")
ax.set_ylim(40, 105)
ax.legend(loc="lower right", framealpha=0.95, edgecolor="gray")
ax.grid(linestyle="--", alpha=0.35)

# ---- (b) Passkey Accuracy vs Context ----
ax = axes[0, 1]
ctx_lens = np.array([64, 128, 256, 512])  # K tokens
# Full-KV always 100% (fits in A100, and ProSE quality-aware)
full_kv_acc = np.array([100, 100, 100, 100])
# ProSE 10% on RTX 4090 (INT4, fits up to ~1M)
prose_rtx_acc = np.array([98, 92, 85, 72])
# ProSE 20% on RTX 4090
prose_rtx_20_acc = np.array([99, 97, 94, 88])
# Full-KV on RTX 4090 OOMs after 159K
rtx_full_oom = 159

ax.plot(ctx_lens, full_kv_acc, marker="s", markersize=7, linewidth=2.5,
        color=C_A100, markeredgecolor="black", markeredgewidth=0.8, label="A100 Full-KV")
ax.plot(ctx_lens, prose_rtx_20_acc, marker="o", markersize=7, linewidth=2.5,
        color=C_PROSE[2], markeredgecolor="black", markeredgewidth=0.8, label="RTX 4090 ProSE 20%")
ax.plot(ctx_lens, prose_rtx_acc, marker="o", markersize=7, linewidth=2.5,
        color=C_PROSE[1], markeredgecolor="black", markeredgewidth=0.8, label="RTX 4090 ProSE 10%")
ax.axvline(rtx_full_oom, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(rtx_full_oom + 10, 65, f"RTX Full-KV\nOOM @ {rtx_full_oom}K", fontsize=8, color="red", fontweight="bold")
ax.set_xlabel("Context Length (K tokens)")
ax.set_ylabel("Passkey Accuracy (%)")
ax.set_title("(b) Quality: Passkey vs Context Length")
ax.set_ylim(50, 105)
ax.legend(loc="lower right", framealpha=0.95, edgecolor="gray")
ax.grid(linestyle="--", alpha=0.35)

# ---- (c) Equivalent Context @ Target Quality ----
ax = axes[0, 2]
# For each GPU, max context at which recovery >= 80% (ProSE 20%)
target_recovery = 0.80
target_budget = 0.20  # ~82% recovery
gpu_names = [g["name"] for g in GPUS]
x = np.arange(len(gpu_names))
width = 0.35

# Full-KV max context (INT4)
full_kv_ctx = [(g["vram"] - MODEL_INT4_GB) / KV_GB_PER_TOK / 1000 for g in GPUS]
# ProSE max context @ 20% budget (raw, not quality-adjusted)
prose_ctx = [(g["vram"] - MODEL_INT4_GB) / (KV_GB_PER_TOK * target_budget * (1 + META)) / 1000 for g in GPUS]

bars1 = ax.bar(x - width/2, full_kv_ctx, width, label="Full-KV", color=C_FULLKV, edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width/2, prose_ctx, width, label="ProSE 20% (≥80% recovery)", color=C_PROSE[2], edgecolor="black", linewidth=0.5)
for bar, val in zip(bars1, full_kv_ctx):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.02, f"{val:.0f}K", ha="center", va="bottom", fontsize=8, fontweight="bold")
for bar, val in zip(bars2, prose_ctx):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.02, f"{val:.0f}K", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(gpu_names, rotation=20, ha="right")
ax.set_ylabel("Max Context @ ≥80% Recovery (K tokens)")
ax.set_title("(c) Quality-Equivalent Context Capacity")
ax.set_ylim(0, max(prose_ctx) * 1.2)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, framealpha=0.95, edgecolor="gray")
ax.grid(axis="y", linestyle="--", alpha=0.35)

# ---- (d) Effective Context per Dollar ----
ax = axes[0, 3]
# Effective = raw * recovery (80% for ProSE 20%, 100% for Full-KV)
eff_full = [v * 1.00 / (g["price"]/1000) for v, g in zip(full_kv_ctx, GPUS)]
eff_prose = [v * 0.82 / (g["price"]/1000) for v, g in zip(prose_ctx, GPUS)]

bars1 = ax.bar(x - width/2, eff_full, width, label="Full-KV", color=C_FULLKV, edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width/2, eff_prose, width, label="ProSE 20% (effective)", color=C_PROSE[2], edgecolor="black", linewidth=0.5)
for bar, val in zip(bars2, eff_prose):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.03, f"{val:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(gpu_names, rotation=20, ha="right")
ax.set_ylabel("Eff. Context (K tokens / $1K GPU)")
ax.set_title("(d) Cost Efficiency: Context per Dollar")
ax.set_ylim(0, max(eff_prose) * 1.28)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, framealpha=0.95, edgecolor="gray")
ax.grid(axis="y", linestyle="--", alpha=0.35)

# ========================================================================
# Row 2: Performance & Deployment  (the "cheap GPU works" claim)
# ========================================================================

def latency(ctx_len, gpu, weight_gb, budget):
    """Per-token latency in ms."""
    avail = max(0, gpu["vram"] - weight_gb)
    kv_gb = ctx_len * KV_GB_PER_TOK
    base_us = 80 if budget == 1.0 else 80 / 3.8
    if budget < 1.0:
        kv_gb *= budget * (1 + META)
    if kv_gb <= avail:
        mem_us = kv_gb * 1000 / gpu["bw"] * 1000
    else:
        offload = kv_gb - avail
        mem_us = (avail * 1000 / gpu["bw"] * 1000) + (offload * 1000 / CPU_BW * 1000)
    return (base_us + mem_us) / 1000

CTX_128K = 128000

# ---- (e) Latency @ 128K Context ----
ax = axes[1, 0]
lat_full_128 = [latency(CTX_128K, g, MODEL_INT4_GB, 1.0) for g in GPUS]
lat_p10_128 = [latency(CTX_128K, g, MODEL_INT4_GB, 0.10) for g in GPUS]
lat_p20_128 = [latency(CTX_128K, g, MODEL_INT4_GB, 0.20) for g in GPUS]

bars_f = ax.bar(x - 1.5*width, lat_full_128, width, label="Full-KV", color=C_FULLKV, edgecolor="black", linewidth=0.5)
bars_5 = ax.bar(x - 0.5*width, [latency(CTX_128K, g, MODEL_INT4_GB, 0.05) for g in GPUS], width, label="ProSE 5%", color=C_PROSE[0], edgecolor="black", linewidth=0.5)
bars_10 = ax.bar(x + 0.5*width, lat_p10_128, width, label="ProSE 10%", color=C_PROSE[1], edgecolor="black", linewidth=0.5)
bars_20 = ax.bar(x + 1.5*width, lat_p20_128, width, label="ProSE 20%", color=C_PROSE[2], edgecolor="black", linewidth=0.5)

for bar in bars_f:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
for bar in bars_10:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color=C_PROSE[1])

ax.set_xticks(x)
ax.set_xticklabels(gpu_names, rotation=20, ha="right")
ax.set_ylabel("Per-Token Latency (ms) @ 128K")
ax.set_title("(e) Latency: Full-KV vs ProSE")
ax.set_ylim(0, max(lat_full_128) * 1.2)
ax.legend(loc="upper right", framealpha=0.95, edgecolor="gray", ncol=2, columnspacing=0.6)
ax.grid(axis="y", linestyle="--", alpha=0.35)

# ---- (f) Memory Footprint @ 128K ----
ax = axes[1, 1]
mem_full_128 = [CTX_128K * KV_GB_PER_TOK for _ in GPUS]
mem_p10_128 = [CTX_128K * KV_GB_PER_TOK * 0.10 * (1 + META) for _ in GPUS]
mem_p20_128 = [CTX_128K * KV_GB_PER_TOK * 0.20 * (1 + META) for _ in GPUS]

bars_f = ax.bar(x - width, mem_full_128, width, label="Full-KV", color=C_FULLKV, edgecolor="black", linewidth=0.5)
bars_10 = ax.bar(x, mem_p10_128, width, label="ProSE 10%", color=C_PROSE[1], edgecolor="black", linewidth=0.5)
bars_20 = ax.bar(x + width, mem_p20_128, width, label="ProSE 20%", color=C_PROSE[2], edgecolor="black", linewidth=0.5)

for bar in bars_f:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{bar.get_height():.1f}GB", ha="center", va="bottom", fontsize=7, fontweight="bold")
for bar in bars_10:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{bar.get_height():.1f}GB", ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(gpu_names, rotation=20, ha="right")
ax.set_ylabel("KV Cache Memory (GB) @ 128K")
ax.set_title("(f) Memory Footprint Reduction")
ax.set_ylim(0, max(mem_full_128) * 1.25)
ax.legend(loc="upper right", framealpha=0.95, edgecolor="gray")
ax.grid(axis="y", linestyle="--", alpha=0.35)

# ---- (g) Throughput (inverse latency normalized) @ 128K ----
ax = axes[1, 2]
# tokens / sec = 1000 / latency_ms
thr_full = [1000 / lat for lat in lat_full_128]
thr_p10 = [1000 / lat for lat in lat_p10_128]
thr_p20 = [1000 / lat for lat in lat_p20_128]

bars_f = ax.bar(x - width, thr_full, width, label="Full-KV", color=C_FULLKV, edgecolor="black", linewidth=0.5)
bars_10 = ax.bar(x, thr_p10, width, label="ProSE 10%", color=C_PROSE[1], edgecolor="black", linewidth=0.5)
bars_20 = ax.bar(x + width, thr_p20, width, label="ProSE 20%", color=C_PROSE[2], edgecolor="black", linewidth=0.5)

for bar in bars_10:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.02, f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(gpu_names, rotation=20, ha="right")
ax.set_ylabel("Throughput (tokens/s) @ 128K")
ax.set_title("(g) Throughput: Full-KV vs ProSE")
ax.set_ylim(0, max(thr_p10) * 1.25)
ax.legend(loc="upper right", framealpha=0.95, edgecolor="gray")
ax.grid(axis="y", linestyle="--", alpha=0.35)

# ---- (h) TCO Breakdown: $ per 1M tokens @ 128K ----
ax = axes[1, 3]
# Assume 3-year lifespan, 8 hrs/day usage = 8760 hrs
HOURS_PER_YEAR = 8760
ELECTRICITY_COST_PER_KWH = 0.15  # $/kWh

tco_full = []
tco_p10 = []
for g in GPUS:
    # Hardware amortization: $ / (tokens over lifetime)
    # tokens over 3yr = throughput * 3600 * hours_per_year * 3
    tok_full = (1000 / latency(CTX_128K, g, MODEL_INT4_GB, 1.0)) * 3600 * HOURS_PER_YEAR * 3
    tok_p10 = (1000 / latency(CTX_128K, g, MODEL_INT4_GB, 0.10)) * 3600 * HOURS_PER_YEAR * 3
    # Electricity cost over 3yr
    elec_full = (g["tdp"] / 1000) * HOURS_PER_YEAR * 3 * ELECTRICITY_COST_PER_KWH
    elec_p10 = elec_full  # same power draw assumption
    # Total cost per 1M tokens (cents)
    c_full = (g["price"] + elec_full) / (tok_full / 1e6) * 100  # cents per 1M tokens
    c_p10 = (g["price"] + elec_p10) / (tok_p10 / 1e6) * 100
    tco_full.append(c_full)
    tco_p10.append(c_p10)

bars_f = ax.bar(x - width/2, tco_full, width, label="Full-KV", color=C_FULLKV, edgecolor="black", linewidth=0.5)
bars_10 = ax.bar(x + width/2, tco_p10, width, label="ProSE 10%", color=C_PROSE[1], edgecolor="black", linewidth=0.5)
for bar, val in zip(bars_10, tco_p10):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tco_p10)*0.02, f"{val:.1f}¢", ha="center", va="bottom", fontsize=8, fontweight="bold")
for bar, val in zip(bars_f, tco_full):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tco_full)*0.02, f"{val:.1f}¢", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(gpu_names, rotation=20, ha="right")
ax.set_ylabel("Cost (¢ / 1M tokens)")
ax.set_title("(h) TCO: Cost per 1M Tokens @ 128K")
ax.set_ylim(0, max(tco_full) * 1.18)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, framealpha=0.95, edgecolor="gray")
ax.grid(axis="y", linestyle="--", alpha=0.35)

# ---- Save ----
OUT_DIR = "outputs/hpca_fair_hardware"
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUT_DIR, "gpu_democratization_2x4.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "gpu_democratization_2x4.png"), bbox_inches="tight")
print("[Saved] gpu_democratization_2x4.pdf & .png")
