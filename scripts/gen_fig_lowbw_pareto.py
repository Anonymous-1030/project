"""
Generate compact 2x2 single-column figure for Low-BW Pareto tradeoff.

Figure c21_lowbw_pareto.pdf:
  - Shows HES+GBS closing the candidate-to-visible recall gap at 4 GB/s
  - Pareto tradeoff: more visible chunks = better recall but more bandwidth
  - 3.45in wide, ~3.0in tall, 2x2 grid, 600 DPI
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Data from benchmark (4 GB/s) ────────────────────────────────────

ctx_labels = ['4K', '8K', '16K']

# Recall@Candidates vs Recall@Visible — the "gap"
baseline_recall_cand = [50.0, 0.0, 50.0]
baseline_recall_vis = [0.0, 25.0, 25.0]
innov_recall_cand = [50.0, 0.0, 50.0]
innov_recall_vis = [41.0, 28.0, 38.5]

# Visible chunk count
baseline_visible = [1.0, 4.1, 7.0]
innov_visible = [5.6, 8.7, 13.6]

# Speedup (throughput ratio vs full-KV)
baseline_speedup = [2.05, 2.20, 2.78]
innov_speedup = [0.92, 1.33, 1.75]

# GBS metrics
gbs_sink_rate = 1.0
gbs_ghosts_created = 36
gbs_recovery_rate = 0.10

# HES metrics
hes_admission_rate = 1.0
hes_lift_over_simhash = 0.20

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
fig.subplots_adjust(hspace=0.62, wspace=0.50,
                    left=0.13, right=0.97, top=0.92, bottom=0.08)

# ── (a) Candidate→Visible Recall Gap ────────────────────────────────

ax = axes[0, 0]
x = np.arange(3)
w = 0.28

ax.bar(x - w, baseline_recall_cand, w, color='#BBDEFB', edgecolor='#1565C0',
       linewidth=0.4, label='Cand.', zorder=3)
ax.bar(x, baseline_recall_vis, w, color='#B0BEC5', edgecolor='#546E7A',
       linewidth=0.4, label='Base Vis.', zorder=3)
ax.bar(x + w, innov_recall_vis, w, color='#AB47BC', edgecolor='#6A1B9A',
       linewidth=0.4, label='Innov Vis.', zorder=3)

# Gap annotation only for 16K (index 2), placed to the right to avoid overlap
i = 2
ax.annotate('', xy=(x[i] + w + 0.12, innov_recall_vis[i]),
            xytext=(x[i] + w + 0.12, baseline_recall_cand[i]),
            arrowprops=dict(arrowstyle='<->', color='#D32F2F', lw=0.7))
ax.text(x[i] + w + 0.32, (baseline_recall_cand[i] + innov_recall_vis[i]) / 2,
        f'gap\nclosed', fontsize=5, color='#D32F2F',
        ha='left', va='center', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(ctx_labels)
ax.set_ylabel('Recall (%)')
ax.set_ylim(0, 62)
ax.set_title('(a) Recall Gap @ 4 GB/s', pad=3)
ax.legend(loc='upper left', framealpha=0.9, edgecolor='none',
          handlelength=0.7, handletextpad=0.3, borderpad=0.2,
          labelspacing=0.15, fontsize=5.5, ncol=1)
ax.grid(axis='y', alpha=0.25, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── (b) Pareto: Recall vs Visible Chunks ─────────────────────────────

ax = axes[0, 1]

ax.scatter(baseline_visible, baseline_recall_vis, c='#546E7A', s=24,
           marker='o', zorder=4, label='Baseline')
ax.scatter(innov_visible, innov_recall_vis, c='#AB47BC', s=24,
           marker='D', zorder=4, label='Innovation')

# Connect pairs with arrows
for i in range(3):
    ax.annotate('', xy=(innov_visible[i], innov_recall_vis[i]),
                xytext=(baseline_visible[i], baseline_recall_vis[i]),
                arrowprops=dict(arrowstyle='->', color='#7B1FA2',
                                lw=0.6, connectionstyle='arc3,rad=0.15'))

# Labels offset to avoid point overlap
offsets = [(0.5, -5), (0.5, 3), (0.5, 3)]
for i in range(3):
    ax.text(innov_visible[i] + offsets[i][0],
            innov_recall_vis[i] + offsets[i][1],
            ctx_labels[i], fontsize=5.5, color='#7B1FA2', fontweight='bold')

ax.set_xlabel('Visible Chunks')
ax.set_ylabel('Recall@Visible (%)')
ax.set_xlim(0, 16)
ax.set_ylim(-5, 55)
ax.set_title('(b) Pareto Frontier', pad=3)
ax.legend(loc='lower right', framealpha=0.9, edgecolor='none',
          handlelength=0.8, handletextpad=0.3, borderpad=0.2,
          labelspacing=0.2, markerscale=0.8)
ax.grid(alpha=0.25, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── (c) Throughput Cost ──────────────────────────────────────────────

ax = axes[1, 0]
x = np.arange(3)
w = 0.32

ax.bar(x - w/2, baseline_speedup, w, color='#B0BEC5', edgecolor='#546E7A',
       linewidth=0.4, label='Baseline', zorder=3)
ax.bar(x + w/2, innov_speedup, w, color='#FF7043', edgecolor='#BF360C',
       linewidth=0.4, label='Innovation', zorder=3)

ax.axhline(1.0, color='#D32F2F', ls='--', lw=0.6, alpha=0.5, zorder=2)

for i in range(3):
    cost = (1 - innov_speedup[i] / baseline_speedup[i]) * 100
    y = max(baseline_speedup[i], innov_speedup[i]) + 0.1
    ax.text(x[i], y, f'−{cost:.0f}%', ha='center', fontsize=5.5,
            color='#BF360C', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(ctx_labels)
ax.set_ylabel('Speedup vs Full-KV')
ax.set_ylim(0, 3.5)
ax.set_yticks([0, 1, 2, 3])
ax.set_title('(c) Throughput Cost', pad=3)
ax.legend(loc='upper left', framealpha=0.9, edgecolor='none',
          handlelength=0.8, handletextpad=0.3, borderpad=0.2, labelspacing=0.2)
ax.grid(axis='y', alpha=0.25, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── (d) GBS + HES Contribution ───────────────────────────────────────

ax = axes[1, 1]

metrics_labels = ['Sink\nDetect', 'Ghosts\nCreated', 'Ghost→\nReal', 'HES\nLift']
metrics_values = [gbs_sink_rate * 100, gbs_ghosts_created, gbs_recovery_rate * 100, hes_lift_over_simhash * 100]
colors = ['#43A047', '#1E88E5', '#AB47BC', '#FF7043']

bars = ax.bar(np.arange(4), metrics_values, 0.55, color=colors,
              edgecolor=[c.replace('7', '3') for c in colors],
              linewidth=0.4, zorder=3)

for i, (v, b) in enumerate(zip(metrics_values, bars)):
    unit = '%' if i != 1 else ''
    label = f'{v:.0f}{unit}'
    ax.text(i, v + max(metrics_values) * 0.04, label,
            ha='center', fontsize=6, fontweight='bold', color='#212121')

ax.set_xticks(np.arange(4))
ax.set_xticklabels(metrics_labels)
ax.set_ylabel('Value')
ax.set_ylim(0, max(metrics_values) * 1.2)
ax.set_title('(d) GBS + HES Metrics', pad=3)
ax.grid(axis='y', alpha=0.25, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Save ─────────────────────────────────────────────────────────────

output_dir = Path('prose_v2/figure')
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / 'c21_lowbw_pareto.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig(output_dir / 'c21_lowbw_pareto.png', bbox_inches='tight', pad_inches=0.02)
plt.close(fig)
print(f"Saved: {output_dir / 'c21_lowbw_pareto.pdf'}")
print(f"Saved: {output_dir / 'c21_lowbw_pareto.png'}")
