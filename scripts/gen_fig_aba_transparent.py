"""
Generate compact 2x2 single-column figure for ABA Transparent Mode.

Target: 3.45in wide, ~3.2in tall, 2x2 grid, fits \columnwidth in LaTeX.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Data ─────────────────────────────────────────────────────────────

ctx_labels = ['4K', '8K', '16K']
baseline_speedup = [1.97, 2.16, 2.77]
innovation_speedup = [2.55, 3.07, 2.83]
throughput_lift = [29, 42, 2]

aba_transparent_frac = 0.96
aba_hybrid_frac = 0.04

baseline_total_us = [213.5, 387.5, 591.6]
innovation_total_us = [174.5, 278.8, 585.7]

bw_labels = ['32', '8', '4']
recall_baseline_16k = [25.0, 25.0, 25.0]
recall_innov_16k = [26.0, 38.5, 38.5]

# ── Style ────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 7,
    'axes.labelsize': 7.5,
    'axes.titlesize': 8,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.25,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.4,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
})

fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.0))
fig.subplots_adjust(hspace=0.58, wspace=0.48,
                    left=0.12, right=0.97, top=0.92, bottom=0.08)

# ── (a) Speedup ─────────────────────────────────────────────────────

ax = axes[0, 0]
x = np.arange(3)
w = 0.32
ax.bar(x - w/2, baseline_speedup, w, color='#B0BEC5', edgecolor='#546E7A',
       linewidth=0.4, label='Baseline', zorder=3)
ax.bar(x + w/2, innovation_speedup, w, color='#1E88E5', edgecolor='#0D47A1',
       linewidth=0.4, label='+ Innov.', zorder=3)
ax.axhline(1.0, color='#D32F2F', ls='--', lw=0.7, alpha=0.6, zorder=2)

for i, lift in enumerate(throughput_lift):
    y = max(baseline_speedup[i], innovation_speedup[i]) + 0.15
    ax.text(x[i] + w/2, y, f'+{lift}%', ha='center', fontsize=5.5,
            color='#0D47A1', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(ctx_labels)
ax.set_ylim(0, 3.9)
ax.set_yticks([0, 1, 2, 3])
ax.set_ylabel('Speedup vs Full-KV')
ax.set_title('(a) Throughput @ 32 GB/s', pad=3)
ax.legend(loc='upper left', framealpha=0.9, edgecolor='none',
          handlelength=0.8, handletextpad=0.3, borderpad=0.2, labelspacing=0.2)
ax.grid(axis='y', alpha=0.25, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── (b) ABA Mode Pie ────────────────────────────────────────────────

ax = axes[0, 1]
colors = ['#43A047', '#FFA726']
wedges, texts, autotexts = ax.pie(
    [aba_transparent_frac, aba_hybrid_frac],
    labels=['Transparent', 'Hybrid'],
    autopct='%1.0f%%', colors=colors, startangle=90,
    explode=(0.02, 0),
    textprops={'fontsize': 6.5, 'fontweight': 'bold'},
    pctdistance=0.65, labeldistance=1.15,
    wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
)
for at in autotexts:
    at.set_fontsize(7)
    at.set_fontweight('bold')
ax.set_title('(b) ABA Mode @ 32 GB/s', pad=3)

# ── (c) Latency ─────────────────────────────────────────────────────

ax = axes[1, 0]
x = np.arange(3)
w = 0.32
ax.bar(x - w/2, baseline_total_us, w, color='#FFCDD2', edgecolor='#C62828',
       linewidth=0.4, label='Baseline', zorder=3)
ax.bar(x + w/2, innovation_total_us, w, color='#C8E6C9', edgecolor='#2E7D32',
       linewidth=0.4, label='Innovation', zorder=3)

for i in range(3):
    red = (1 - innovation_total_us[i] / baseline_total_us[i]) * 100
    if red > 1:
        y = max(baseline_total_us[i], innovation_total_us[i]) + 18
        ax.text(x[i], y, f'−{red:.0f}%', ha='center', fontsize=5.5,
                color='#2E7D32', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(ctx_labels)
ax.set_ylabel('Latency (μs)')
ax.set_ylim(0, 720)
ax.set_title('(c) End-to-End Latency', pad=3)
ax.legend(loc='upper left', framealpha=0.9, edgecolor='none',
          handlelength=0.8, handletextpad=0.3, borderpad=0.2, labelspacing=0.2)
ax.grid(axis='y', alpha=0.25, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── (d) Recall ───────────────────────────────────────────────────────

ax = axes[1, 1]
x = np.arange(3)
w = 0.32
ax.bar(x - w/2, recall_baseline_16k, w, color='#B0BEC5', edgecolor='#546E7A',
       linewidth=0.4, label='Baseline', zorder=3)
ax.bar(x + w/2, recall_innov_16k, w, color='#AB47BC', edgecolor='#6A1B9A',
       linewidth=0.4, label='Innovation', zorder=3)

for i in range(3):
    lift = recall_innov_16k[i] - recall_baseline_16k[i]
    if lift > 0:
        ax.text(x[i] + w/2, recall_innov_16k[i] + 1.8, f'+{lift:.0f}%',
                ha='center', fontsize=5.5, color='#6A1B9A', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(bw_labels)
ax.set_xlabel('Bandwidth (GB/s)')
ax.set_ylabel('Recall@Visible (%)')
ax.set_ylim(0, 52)
ax.set_title('(d) Recall @ 16K ctx', pad=3)
ax.legend(loc='upper left', framealpha=0.9, edgecolor='none',
          handlelength=0.8, handletextpad=0.3, borderpad=0.2, labelspacing=0.2)
ax.grid(axis='y', alpha=0.25, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Save ─────────────────────────────────────────────────────────────

output_dir = Path('prose_v2/figure')
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / 'c20_aba_transparent.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig(output_dir / 'c20_aba_transparent.png', bbox_inches='tight', pad_inches=0.02)
plt.close(fig)
print(f"Saved: {output_dir / 'c20_aba_transparent.pdf'}")
print(f"Saved: {output_dir / 'c20_aba_transparent.png'}")
