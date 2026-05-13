"""
HPCA rebuttal: Sigma Sensitivity — 1×2, dual-Y left panel, high data density.
Generates figure/sigma_sweep.pdf
"""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJ, "outputs", "innovations", "w1_sigma_sensitivity.json")) as f:
    D = json.load(f)

sweep   = D["part_a_sweep"]
cascade = D["part_c_cascade_value"]
sigmas  = np.array([s["sigma"] for s in sweep])

# Individual workloads
wl = {
    "Passkey":    np.array([s["passkey_rec"]    for s in sweep]),
    "Needle":     np.array([s["needle_rec"]     for s in sweep]),
    "Sequential": np.array([s["sequential_rec"] for s in sweep]),
    "RULER":      np.array([s["ruler_rec"]      for s in sweep]),
}
mean_rec  = np.array([s["mean_recovery"] for s in sweep])
min_rec = np.array([min(s["passkey_rec"], s["needle_rec"], s["sequential_rec"], s["ruler_rec"]) for s in sweep])
max_rec = np.array([max(s["passkey_rec"], s["needle_rec"], s["sequential_rec"], s["ruler_rec"]) for s in sweep])
p99s     = np.array([s["p99_us"] for s in sweep])
sbase    = sweep[0]["struct_baseline"]

# Cascade ablation curves
c_sigmas = np.array([c["sigma"] for c in cascade])
c_no_r2  = np.array([c["no_round2"] for c in cascade])
c_no_tmp = np.array([c["no_temporal"] for c in cascade])
c_full   = np.array([c["full"] for c in cascade])

# Structured noise
snoise = D["part_b_structured"]
iid_rec   = snoise["results"][0]["mean_recovery"]
struc_rec = snoise["results"][1]["mean_recovery"]

# ═══════════════════════════════════════════════════════════════════════════
# Style
# ═══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans","Arial"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 5.3, "axes.linewidth": 0.5,
    "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# Colour palette
WL_C = {"Passkey":"#D55E00","Needle":"#0072B2","Sequential":"#009E73","RULER":"#CC79A7"}
MEAN = "#111111"; P99_C = "#332288"; GRAY = "#999999"; LIGHT = "#CCCCCC"
R2_C  = "#E69F00"    # no-Round2 curve — amber
TMP_C = "#56B4E9"    # no-Temporal curve — sky blue

# ═══════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════
fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 3.1))
fig.patch.set_facecolor("white")
plt.subplots_adjust(wspace=0.52)

# ─── LEFT: Dual-Y ────────────────────────────────────────────────────────
axL2 = axL.twinx()

# Spread band (min–max across workloads)
axL.fill_between(sigmas, min_rec, max_rec, color="#E8E8E8", alpha=0.55,
                 edgecolor="none", zorder=0)

# Individual workload curves
for name, vals in wl.items():
    axL.plot(sigmas, vals, color=WL_C[name], lw=0.85, marker=".", ms=2.5,
             alpha=0.82, zorder=3)

# Cascade ablation curves — no-Round2 (amber dashed), no-Temporal (sky dash-dot)
axL.plot(c_sigmas, c_no_r2,  color=R2_C, lw=1.0, ls=(0,(5,3)), marker=".", ms=2, alpha=0.8, zorder=3)
axL.plot(c_sigmas, c_no_tmp, color=TMP_C, lw=1.0, ls=(0,(3,2,1,2)), marker=".", ms=2, alpha=0.8, zorder=3)

# Mean recovery — thick black dominant
axL.plot(sigmas, mean_rec, color=MEAN, lw=2.2, marker="o", ms=4.8,
         mfc="white", mec=MEAN, mew=0.9, zorder=6)

# Structural baseline
axL.axhline(sbase, color=GRAY, lw=0.7, ls=(0,(4.5,3)), zorder=1)

# P99 stall — right axis
axL2.plot(sigmas, p99s, color=P99_C, lw=1.8, marker="D", ms=3.8,
          mfc="white", mec=P99_C, mew=0.8, zorder=5)
axL2.axhline(np.mean(p99s), color=P99_C, lw=0.4, ls=":", alpha=0.45)

# Structured noise markers ★
axL.plot(0.5, iid_rec,    marker="*", ms=10, color=MEAN, zorder=9, clip_on=False)
axL.plot(0.5, struc_rec,  marker="*", ms=10, color="#E69F00", zorder=9, clip_on=False)
axL.plot([0.5,0.5],[iid_rec,struc_rec], color="#E69F00", lw=0.6, ls="--", zorder=2)
axL.annotate(f"Δ={iid_rec-struc_rec:+.3f}", (0.5, struc_rec-0.018),
             fontsize=5.2, color="#E69F00", ha="center", fontweight="bold")

# Labels
axL.set_xlim(0.05, 1.05); axL.set_ylim(0.35, 0.71)
axL.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axL2.set_ylim(52, 88)
axL.set_xlabel("Encoder noise σ", labelpad=2)
axL.set_ylabel("Recovery", labelpad=2, color="#222")
axL2.set_ylabel("P99 stall (μs)", labelpad=2, color=P99_C)
axL.tick_params(axis="y", colors="#222"); axL2.tick_params(axis="y", colors=P99_C)
axL.set_title("(a) Recovery & P99 stall vs encoder noise σ", fontweight="bold", pad=3)
axL.grid(True, axis="y", color="#ECECEC", lw=0.25, zorder=0)
axL.set_axisbelow(True)
for sp in ["top"]: axL.spines[sp].set_visible(False); axL2.spines[sp].set_visible(False)

# Legend — grouped by category
legend_items = [
    # Category 1: Recovery metrics
    Line2D([0],[0], color=MEAN, lw=2.2, marker="o", ms=4.5, mfc="white", mec=MEAN, mew=0.9, label="Mean recovery (full cascade)"),
    Line2D([0],[0], color=R2_C, lw=1.0, ls=(0,(5,3)), label="R1-only (−Round 2)"),
    Line2D([0],[0], color=TMP_C, lw=1.0, ls=(0,(3,2,1,2)), label="R1-Temporal (−R2, −Temp)"),
    Line2D([0],[0], color=GRAY, lw=0.7, ls=(0,(4.5,3)), label="Structural-only"),
    # Category 2: P99
    Line2D([0],[0], color=P99_C, lw=1.8, marker="D", ms=3.8, mfc="white", mec=P99_C, mew=0.8, label="P99 stall (μs, →)"),
    # Category 3: benchmarks
    Line2D([],[], color="#DDDDDD", lw=0, marker="s", ms=0, label="— Workloads —"),
]
for name, c in WL_C.items():
    legend_items.append(Line2D([0],[0], color=c, lw=0.85, marker=".", ms=2.5, label=name))

axL.legend(handles=legend_items, loc="lower right", ncol=2, frameon=True,
           framealpha=0.92, edgecolor="#DDD", fontsize=5.0, borderpad=0.35,
           labelspacing=0.18, handletextpad=0.4, columnspacing=0.4)

# ─── RIGHT: Cascade ablation bar chart ───────────────────────────────────
c5_idx = np.where(np.isclose(np.array([c["sigma"] for c in cascade]), 0.5))[0][0]
c5 = cascade[c5_idx]
bars_vals = [c5["full"], c5["no_temporal"], c5["no_round2"], c5["structural_only"]]
bars_lbls = ["Full\ncascade", "−Temporal\nensemble", "−Round 2\n(re-rank)", "Structural\nonly"]
bars_colors = ["#111111", "#555555", "#999999", "#CCCCCC"]
x_pos = np.arange(len(bars_vals))

axR.bar(x_pos, bars_vals, width=0.54, color=bars_colors, edgecolor="white", lw=0.3, zorder=3)

for x, v in zip(x_pos, bars_vals):
    axR.text(x, v + 0.009, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold", color="#222")

# Add mechanism labels inside/below bars
mechanism_labels = ["M_A+B+C", "M_B+C", "—", "—"]
for x, v, m in zip(x_pos, bars_vals, mechanism_labels):
    if m != "—":
        axR.text(x, v - 0.04, m, ha="center", fontsize=5.3, color="white", fontweight="bold")

# Delta bracket annotations
deltas = [bars_vals[1]-bars_vals[0], bars_vals[2]-bars_vals[1], bars_vals[3]-bars_vals[2]]
delta_labels = ["Temp.", "Round 2", "Struct.\nbaseline"]
for i in range(3):
    mid = i + 0.5
    y_mid = (bars_vals[i] + bars_vals[i+1]) / 2
    axR.annotate(f"{deltas[i]:+.3f}", xy=(mid - 0.08, bars_vals[i+1]),
                 xytext=(mid + 0.18, y_mid), fontsize=6.0, color="#D55E00",
                 fontweight="bold", ha="center", va="center",
                 arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.6))

axR.set_xticks(x_pos); axR.set_xticklabels(bars_lbls, fontsize=5.8)
axR.set_ylim(0.30, 0.74)
axR.set_ylabel("Recovery", labelpad=2)
axR.set_title("(b) Cascade ablation at σ=0.5", fontweight="bold", pad=3)
axR.grid(True, axis="y", color="#ECECEC", lw=0.25, zorder=0)
axR.set_axisbelow(True)
axR.spines["top"].set_visible(False); axR.spines["right"].set_visible(False)

# Footer
fig.text(0.5, -0.03,
    "128K context, 25-budget, watertight policy  |  Gray band = min–max across 4 workloads  |  ★ = structured noise test",
    fontsize=5.2, color="#BBBBBB", ha="center", fontstyle="italic")

# Save
out_dir = os.path.join(PROJ, "outputs", "figure")
os.makedirs(out_dir, exist_ok=True)
for fmt in ["pdf","png"]:
    p = os.path.join(out_dir, f"sigma_sweep.{fmt}")
    fig.savefig(p, format=fmt, dpi=600, bbox_inches="tight", pad_inches=0.06,
                facecolor="white", edgecolor="none")
    print(f"✓ {p}")
