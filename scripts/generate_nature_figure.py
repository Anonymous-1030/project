#!/usr/bin/env python3
"""
Minimal Nature-style 2x3 figure: PROSE SBFI Irreducibility.

Style: clean lines, sparse annotations, generous whitespace, muted palette.
LaTeX-ready PDF with Type 42 fonts.
"""

import json, os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ═══════════════════════════════════════════════════════════════════════
# LaTeX-ready PDF
# ═══════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "pdf.compression": 9,
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.pad": 2, "ytick.major.pad": 2,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════
LLM_ROOT = r"d:\LLM"
with open(os.path.join(LLM_ROOT, "outputs", "hpca_rebuttal", "expA_sbfi_irreducibility.json")) as f:
    DATA = json.load(f)["results"]

SEQ = np.array([8192, 16384, 32768, 65536])
SEQ_LAB = ["8K", "16K", "32K", "64K"]
X_FINE = np.logspace(np.log10(7500), np.log10(70000), 200)

METHODS = ["prose", "prose_fts", "freqrec_prefetcher", "stream_prefetcher"]

# Muted Nature palette
COLOR = {
    "prose":                 "#C44E52",  # muted red
    "prose_fts":             "#DD8C6A",  # muted orange
    "freqrec_prefetcher":    "#5689C0",  # muted blue
    "stream_prefetcher":     "#AAAAAA",  # gray
}
LABEL = {
    "prose":                 "PROSE (SBFI)",
    "prose_fts":             "PROSE-FTS",
    "freqrec_prefetcher":    "FreqRec-PF",
    "stream_prefetcher":     "StreamPrefetcher",
}
LS = {"prose": "-", "prose_fts": "--", "freqrec_prefetcher": "-", "stream_prefetcher": "-."}
MK = {"prose": "o", "prose_fts": "s", "freqrec_prefetcher": "D", "stream_prefetcher": "^"}
ZO = {"prose": 10, "prose_fts": 9, "freqrec_prefetcher": 8, "stream_prefetcher": 7}
LW = {"prose": 1.2, "prose_fts": 1.0, "freqrec_prefetcher": 0.9, "stream_prefetcher": 0.8}
MS = {"prose": 28, "prose_fts": 24, "freqrec_prefetcher": 22, "stream_prefetcher": 20}


def s(metric):
    return {m: np.array([p[metric] for p in DATA[m]], dtype=float) for m in METHODS}


def spline(xd, yd, xf):
    if len(xd) >= 3:
        cs = CubicSpline(xd, yd, bc_type="natural")
        return np.clip(cs(xf), min(yd)*0.85, max(yd)*1.2)
    return np.interp(xf, xd, yd)


def sat_mm1(rho):
    return np.clip(1.0/(1.0 - np.clip(np.asarray(rho, float), 0.001, 0.95)), 1.0, 20.0)


def draw(ax, d, xf):
    for m in METHODS:
        y = d[m]
        yf = spline(SEQ, y, xf)
        ax.plot(xf, yf, color=COLOR[m], ls=LS[m], lw=LW[m], zorder=ZO[m])
        ax.scatter(SEQ, y, c=COLOR[m], marker=MK[m], s=MS[m],
                   edgecolors="white", linewidths=0.5, zorder=ZO[m]+1)


def cfg(ax, yl, ylim=None, yt=None, xl=True):
    ax.set_ylabel(yl, labelpad=1)
    ax.set_xscale("log")
    ax.set_xticks(SEQ)
    ax.set_xticklabels(SEQ_LAB)
    ax.tick_params(axis="x", which="minor", bottom=False)
    if xl:
        ax.set_xlabel("Sequence Length", labelpad=0)
    if ylim:
        ax.set_ylim(*ylim)
    if yt:
        ax.set_yticks(yt)


# Pre-compute
inv = s("invalid_traffic_ratio")
qut = s("queue_utilization")
p99 = s("p99_latency_us")
tps = s("throughput_tps")
rec = s("mean_recovery")
sat = {m: sat_mm1(qut[m]) for m in METHODS}

# ═══════════════════════════════════════════════════════════════════════
# FIGURE — 2 rows x 3 columns, Nature-style proportions
# ═══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18.3/2.54, 12.5/2.54))
gs = fig.add_gridspec(2, 3, hspace=0.72, wspace=0.42,
                       left=0.08, right=0.98, top=0.94, bottom=0.12)
ax = {(i,j): fig.add_subplot(gs[i,j]) for i in range(2) for j in range(3)}

PANEL = [["a","b","c"],["d","e","f"]]

# ── (a) Invalid Traffic Ratio ──
draw(ax[(0,0)], {m: inv[m]*100 for m in METHODS}, X_FINE)
cfg(ax[(0,0)], "Invalid Traffic (%)", ylim=(-10, 90), yt=[0, 30, 60, 90])
ax[(0,0)].axhline(0, color="#ccc", lw=0.4, ls=":", zorder=0)
# Single sparse annotation — just text, no box
ax[(0,0)].text(11000, 3, "PROSE: 0% waste (invariant)", fontsize=6.5,
               color=COLOR["prose"], fontstyle="italic")
ax[(0,0)].text(38000, 80, "PROSE-FTS: ~70% waste", fontsize=6.5,
               color=COLOR["prose_fts"], fontstyle="italic")

# ── (b) Queue Utilization ──
draw(ax[(0,1)], qut, X_FINE)
cfg(ax[(0,1)], "CXL Queue Utilization  ρ", ylim=(-0.05, 1.10), yt=[0, 0.25, 0.5, 0.75, 1.0])
ax[(0,1)].axhline(1.0, color="#888", lw=0.4, ls="--", alpha=0.3, zorder=0)
ax[(0,1)].text(55000, 0.84, "ρ=0.81", fontsize=6.5, color=COLOR["prose"], fontstyle="italic")

# ── (c) Saturation Multiplier ──
draw(ax[(0,2)], sat, X_FINE)
cfg(ax[(0,2)], "Saturation Multiplier", ylim=(0.8, 7.5))
xr = np.logspace(np.log10(8000), np.log10(66000), 100)
ax[(0,2)].plot(xr, 1+0.065*(xr/1000), color="#bbb", lw=0.4, ls=":", alpha=0.5)
ax[(0,2)].text(48000, 3.2, "Linear", fontsize=5.5, color="#aaa", rotation=10)
ax[(0,2)].text(42000, 5.7, "M/M/1", fontsize=6.5, color=COLOR["prose"], fontstyle="italic")

# ── (d) P99 Latency ──
draw(ax[(1,0)], p99, X_FINE)
cfg(ax[(1,0)], "P99 Latency (μs)", ylim=(0, 2050))
# Inset
ain = ax[(1,0)].inset_axes([0.53, 0.14, 0.43, 0.40])
for m in ["prose", "freqrec_prefetcher", "stream_prefetcher"]:
    ain.plot(SEQ, p99[m], color=COLOR[m], ls=LS[m], lw=1.0, marker=MK[m], markersize=4)
ain.set_xscale("log"); ain.set_xticks(SEQ)
ain.set_xticklabels(SEQ_LAB, fontsize=5); ain.tick_params(labelsize=5, pad=0.5)
ain.set_ylabel("P99 (μs)", fontsize=5.5, labelpad=0)
ain.set_ylim(120, 255)
ax[(1,0)].text(50000, 1800, "5.4x", fontsize=7, color=COLOR["prose_fts"],
               fontstyle="italic", fontweight="bold")
ax[(1,0)].text(28000, 135, "+27%", fontsize=7, color=COLOR["prose"],
               fontweight="bold")

# ── (e) Throughput ──
draw(ax[(1,1)], tps, X_FINE)
cfg(ax[(1,1)], "Throughput (tokens/s)", ylim=(0, 8800))
ax[(1,1)].text(50000, 750, "5.4x collapse", fontsize=7, color=COLOR["prose_fts"],
               fontstyle="italic", fontweight="bold")
ax[(1,1)].text(28000, 7000, "−8%", fontsize=7, color=COLOR["prose"], fontweight="bold")

# ── (f) Mean Recovery ──
draw(ax[(1,2)], rec, X_FINE)
cfg(ax[(1,2)], "Mean Recovery", ylim=(0, 0.82))
ax[(1,2)].text(25000, 0.62, "PROSE = PROSE-FTS\n(same ranking)", fontsize=6.5,
               color="#555", fontstyle="italic", ha="center")
ax[(1,2)].text(52000, 0.10, "6−8x worse\nat 64K", fontsize=6.5,
               color=COLOR["freqrec_prefetcher"], fontstyle="italic")

# ── Panel labels ──
for i in range(2):
    for j in range(3):
        ax[(i,j)].text(-0.12, 1.05, PANEL[i][j], transform=ax[(i,j)].transAxes,
                       fontsize=12, fontweight="bold", va="top", ha="left")

# ── Shared legend at top ──
leg = [Line2D([], [], color=COLOR[m], ls=LS[m], marker=MK[m],
              markersize=4, label=LABEL[m], lw=1.2) for m in METHODS]
fig.legend(handles=leg, loc="upper center", ncol=4, frameon=False,
           fontsize=8, handlelength=1.8, handletextpad=0.5, columnspacing=1.2,
           bbox_to_anchor=(0.5, 0.985))

# ── Save ──
out_png = os.path.join(LLM_ROOT, "outputs", "hpca_rebuttal", "fig_press_compact_invariants.png")
out_pdf = os.path.join(LLM_ROOT, "outputs", "hpca_rebuttal", "fig_press_compact_invariants.pdf")
fig.savefig(out_png, dpi=300, facecolor="white", edgecolor="none")
fig.savefig(out_pdf, dpi=300, facecolor="white", edgecolor="none")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
