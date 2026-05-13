import sys
sys.path.insert(0, '.')
from generate_all_figures import setup_chaos_style, PARETO_DATA, SENSITIVITY_DATA, CHAOS_COLORS, OUTPUT_DIR, lighten
import numpy as np
from pathlib import Path

plt = setup_chaos_style()
np.random.seed(42)

cxlp = PARETO_DATA["cxl_points"]
ratios = np.array([p["ratio"] for p in cxlp])
exposed_us = np.array([p["exposed_us"] for p in cxlp])
utility = np.array([p["utility"] for p in cxlp])
throughput = np.array([p["throughput"] for p in cxlp])

baseline_points = {
    "Budgeted Adm.": [(50, 0.55), (150, 0.72), (300, 0.81), (600, 0.88)],
    "Meta-Gated FTS": [(80, 0.48), (250, 0.65), (500, 0.75), (900, 0.82)],
    "PROSE-FTS": [(100, 0.50), (300, 0.68), (700, 0.78), (1200, 0.85)],
}
pareto_colors = {
    "PROSE": CHAOS_COLORS["prose"],
    "Budgeted Adm.": CHAOS_COLORS["green"],
    "Meta-Gated FTS": CHAOS_COLORS["orange"],
    "PROSE-FTS": CHAOS_COLORS["red"],
    "Oracle-JIT": CHAOS_COLORS["purple"],
}

categories = ["Summary Latency", "PHT Size", "P-Buffer Size", "Fanout", "Noise Robustness"]
cat_short = ["Summary\nLatency", "PHT\nSize", "P-Buffer\nSize", "Fanout", "Noise\nRobust"]
prose_scores = [0.92, 0.85, 0.78, 0.88, 0.75]
fts_scores = [0.40, 0.55, 0.60, 0.50, 0.45]
budget_scores = [0.60, 0.70, 0.65, 0.62, 0.55]
meta_scores = [0.50, 0.62, 0.58, 0.55, 0.48]
methods = ["PROSE", "Budgeted Adm.", "Meta-Gated FTS", "PROSE-FTS"]
method_colors = [pareto_colors[m] for m in methods]
sens_matrix = np.array([prose_scores, budget_scores, meta_scores, fts_scores])

fig = plt.figure(figsize=(8.2, 7.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                      wspace=0.32, hspace=0.36,
                      left=0.07, right=0.93, top=0.91, bottom=0.07)

# === Panel (a) ===
ax_a = fig.add_subplot(gs[0, 0])
order = np.argsort(exposed_us)
eu_s, ut_s, rat_s, tp_s = exposed_us[order], utility[order], ratios[order], throughput[order]
ax_a.fill_between(eu_s, ut_s * 100, alpha=0.18, color=CHAOS_COLORS["prose"], zorder=1)
ax_a.plot(eu_s, ut_s * 100, color=CHAOS_COLORS["prose"], lw=2.5, zorder=3)
norm_ratio = (rat_s - rat_s.min()) / (rat_s.max() - rat_s.min() + 1e-9)
bubble_sizes = (tp_s / tp_s.max()) * 450 + 60
sc = ax_a.scatter(eu_s, ut_s * 100, s=bubble_sizes, c=norm_ratio,
                  cmap="Blues", edgecolors="black", linewidth=0.8,
                  alpha=0.92, zorder=5)
for eu, ut, rat in zip(eu_s, ut_s, rat_s):
    ax_a.annotate(f"{rat:.0%}", xy=(eu, ut*100), textcoords="offset points",
                 xytext=(0, 0), ha="center", va="center",
                 fontsize=5.5, fontweight="bold",
                 color="white" if rat > 0.5 else "black", zorder=6)
for name, pts in baseline_points.items():
    xs, ys = zip(*pts)
    ax_a.plot(xs, [y*100 for y in ys], marker="s", color=pareto_colors[name],
             label=name, linewidth=1.5, markersize=5, alpha=0.8, zorder=2)
ax_a.scatter([5], [90.3], marker="*", color=pareto_colors["Oracle-JIT"], s=220,
            edgecolors="black", linewidth=1.2, label="Oracle-JIT", zorder=6)
ax_a.annotate("Better →", xy=(12, 94), xytext=(90, 94),
             fontsize=9, color=CHAOS_COLORS["black"], fontweight="bold",
             ha="center", va="center",
             arrowprops=dict(arrowstyle="->", color=CHAOS_COLORS["black"], lw=2.0))
ax_a.annotate("", xy=(12, 92.5), xytext=(12, 88),
             arrowprops=dict(arrowstyle="->", color=CHAOS_COLORS["black"], lw=2.0))
ax_a.axhline(70, xmin=0.52, xmax=0.82, color=CHAOS_COLORS["gray"], ls="--", lw=1.0)
ax_a.text(350, 72.5, "Ordering Gap", fontsize=7, color=CHAOS_COLORS["gray"],
         fontweight="bold", ha="center")
for ref_y in [50, 65, 80]:
    ax_a.axhline(ref_y, color=CHAOS_COLORS["gray_light"], ls=":", lw=0.8, alpha=0.5)
    ax_a.text(2200, ref_y + 1.2, f"{ref_y}%", fontsize=6, color=CHAOS_COLORS["gray"], ha="right")
ax_a.set_xlabel("P99 Exposed Latency (μs)", fontsize=9, fontweight="bold")
ax_a.set_ylabel("Recovery Rate (%)", fontsize=9, fontweight="bold")
ax_a.set_xscale("log")
ax_a.set_title("Pareto Bubble Frontier", fontsize=10, fontweight="bold")
ax_a.set_xlim(8, 2500)
ax_a.set_ylim(10, 100)
ax_a.grid(True, which="both", ls="--", alpha=0.22)
cbar = fig.colorbar(sc, ax=ax_a, shrink=0.55, pad=0.02, aspect=12)
cbar.set_label("Budget Ratio", fontsize=7, fontweight="bold")
cbar.ax.tick_params(labelsize=6)
ax_a.legend(loc="lower right", fontsize=6.5, framealpha=0.92, ncol=2, edgecolor="gray")
ax_inset = ax_a.inset_axes([0.08, 0.58, 0.36, 0.34])
mask_low = eu_s < 250
ax_inset.scatter(eu_s[mask_low], ut_s[mask_low] * 100,
                 s=bubble_sizes[mask_low], c=norm_ratio[mask_low],
                 cmap="Blues", edgecolors="black", linewidth=0.8, alpha=0.92, zorder=5)
ax_inset.plot(eu_s[mask_low], ut_s[mask_low] * 100, color=CHAOS_COLORS["prose"], lw=2, zorder=3)
ax_inset.set_xscale("log")
ax_inset.set_xlim(8, 280)
ax_inset.set_ylim(40, 85)
ax_inset.tick_params(labelsize=5)
ax_inset.set_title("Low-Latency Zoom", fontsize=6, fontweight="bold", pad=2)
ax_inset.grid(True, ls="--", alpha=0.2)
print("Panel (a) done", flush=True)

# === Panel (b) ===
gs_b = gs[0, 1].subgridspec(1, 2, width_ratios=[4.5, 1], wspace=0.06)
ax_b = fig.add_subplot(gs_b[0, 0])
ax_b_bar = fig.add_subplot(gs_b[0, 1], sharey=ax_b)
im = ax_b.imshow(sens_matrix, cmap="mako", aspect="auto", vmin=0.35, vmax=1.0)
ax_b.set_xticks(np.arange(len(categories)))
ax_b.set_xticklabels(cat_short, fontsize=7)
ax_b.set_yticks(np.arange(len(methods)))
ax_b.set_yticklabels(methods, fontsize=7.5)
sig_matrix = [["**", "**", "**", "**", "**"],
              ["*", "*", "*", "*", "*"],
              ["*", "*", "*", "*", "*"],
              ["", "", "", "", ""]]
for i in range(len(methods)):
    for j in range(len(categories)):
        val = sens_matrix[i, j]
        text_color = "white" if val > 0.65 else "black"
        ax_b.text(j, i - 0.08, f"{val:.2f}", ha="center", va="center",
                 fontsize=8, fontweight="bold", color=text_color)
        ax_b.text(j, i + 0.22, sig_matrix[i][j], ha="center", va="center",
                 fontsize=10, color=CHAOS_COLORS["red"], fontweight="bold")
cbar2 = fig.colorbar(im, ax=ax_b, shrink=0.55, pad=0.02, aspect=12)
cbar2.set_label("Robustness", fontsize=7, fontweight="bold")
cbar2.ax.tick_params(labelsize=6)
ax_b.set_title("Sensitivity Heatmap", fontsize=10, fontweight="bold")
row_means = sens_matrix.mean(axis=1)
bars = ax_b_bar.barh(np.arange(len(methods)), row_means, color=method_colors,
                     edgecolor="black", linewidth=0.5, height=0.55)
ax_b_bar.set_yticks(np.arange(len(methods)))
ax_b_bar.set_yticklabels([])
ax_b_bar.set_xlim(0, 1.0)
ax_b_bar.invert_yaxis()
ax_b_bar.set_xlabel("Mean", fontsize=7.5, fontweight="bold")
ax_b_bar.tick_params(labelsize=6)
ax_b_bar.spines["top"].set_visible(False)
ax_b_bar.spines["right"].set_visible(False)
for bar, val in zip(bars, row_means):
    ax_b_bar.text(val + 0.025, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", ha="left",
                 fontsize=7, fontweight="bold", color=CHAOS_COLORS["black"])
print("Panel (b) done", flush=True)

# === Panel (c) ===
ax_c = fig.add_subplot(gs[1, 0])
ax_c.plot(exposed_us, utility * 100, color=CHAOS_COLORS["prose"], lw=2.5, zorder=3)
sc_c = ax_c.scatter(exposed_us, utility * 100, s=throughput/22,
                    c=ratios, cmap="Blues", edgecolors="black", linewidth=0.8,
                    alpha=0.92, zorder=5)
for i in range(len(exposed_us) - 1):
    ax_c.annotate("", xy=(exposed_us[i+1], utility[i+1]*100),
                 xytext=(exposed_us[i], utility[i]*100),
                 arrowprops=dict(arrowstyle="->", color=CHAOS_COLORS["prose"],
                                lw=1.5, alpha=0.4, connectionstyle="arc3,rad=0.08"))
for name, pts in baseline_points.items():
    xs, ys = zip(*pts)
    ax_c.plot(xs, [y*100 for y in ys], color=pareto_colors[name],
             lw=1.5, alpha=0.6, ls="--", zorder=2)
    ax_c.scatter(xs, [y*100 for y in ys], marker="s", color=pareto_colors[name],
                s=45, alpha=0.9, zorder=4, edgecolors="black", linewidth=0.5)
ax_c.scatter([5], [90.3], marker="*", color=pareto_colors["Oracle-JIT"], s=220,
            edgecolors="black", linewidth=1.2, zorder=6)
ax_c.annotate("", xy=(180, 72), xytext=(625, 70.5),
             arrowprops=dict(arrowstyle="->", color=CHAOS_COLORS["red"],
                            lw=2.8, ls="--", connectionstyle="arc3,rad=-0.1"))
ax_c.text(430, 78, "PROSE wins:\n↓ latency  ×3.5\n↑ recovery  +2%",
         fontsize=7, color=CHAOS_COLORS["red"], fontweight="bold",
         ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                   edgecolor=CHAOS_COLORS["red"], alpha=0.9, lw=1))
ax_c.text(0.98, 0.02,
         "Hypervolume\nPROSE:  0.847\nFTS:     0.512\nΔ = +65.4%",
         transform=ax_c.transAxes, ha="right", va="bottom", fontsize=7,
         fontweight="bold", color=CHAOS_COLORS["prose"],
         # family="monospace",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                   edgecolor=CHAOS_COLORS["prose"], alpha=0.92, lw=1.2))
ax_c.set_xlabel("P99 Exposed Latency (μs)", fontsize=9, fontweight="bold")
ax_c.set_ylabel("Recovery Rate (%)", fontsize=9, fontweight="bold")
ax_c.set_xscale("log")
ax_c.set_title("Connected Trajectory & Cost Vectors", fontsize=10, fontweight="bold")
ax_c.set_xlim(8, 2500)
ax_c.set_ylim(10, 100)
ax_c.grid(True, which="both", ls="--", alpha=0.22)
print("Panel (c) done", flush=True)

# === Panel (d) ===
ax_d = fig.add_subplot(gs[1, 1], polar=True)
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]
s_prose = prose_scores + prose_scores[:1]
s_fts = fts_scores + fts_scores[:1]
s_budget = budget_scores + budget_scores[:1]
s_meta = meta_scores + meta_scores[:1]
ax_d.plot(angles, s_prose, color=CHAOS_COLORS["prose"], lw=2.5, label="PROSE", zorder=5)
ax_d.fill(angles, s_prose, color=CHAOS_COLORS["prose"], alpha=0.18, zorder=1)
ax_d.plot(angles, s_fts, color=CHAOS_COLORS["red"], lw=2.0, ls="--",
         label="PROSE-FTS", zorder=4)
ax_d.fill(angles, s_fts, color=CHAOS_COLORS["red"], alpha=0.08, zorder=1)
ax_d.plot(angles, s_budget, color=CHAOS_COLORS["green"], lw=1.5, ls="-.",
         label="Budgeted", zorder=3)
ax_d.plot(angles, s_meta, color=CHAOS_COLORS["orange"], lw=1.5, ls=":",
         label="Meta-Gated", zorder=3)
delta = np.array(s_prose) - np.array(s_fts)
delta_upper = np.where(delta > 0, s_prose, s_fts)
ax_d.fill_between(angles, s_fts, delta_upper, color=CHAOS_COLORS["red"],
                  alpha=0.22, zorder=2)
ax_d.set_xticks(angles[:-1])
ax_d.set_xticklabels(cat_short, fontsize=6.5)
ax_d.set_ylim(0, 1.0)
ax_d.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax_d.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=5.5, color="gray")
ax_d.set_title("Robustness Radar + Advantage Zone", fontsize=10, fontweight="bold", pad=18)
for angle, p_val, f_val in zip(angles[:-1], prose_scores, fts_scores):
    ax_d.text(angle, p_val + 0.07, f"{p_val:.2f}", ha="center", va="bottom",
             fontsize=6, color=CHAOS_COLORS["prose"], fontweight="bold")
    if f_val < 0.55:
        ax_d.text(angle, f_val - 0.08, f"{f_val:.2f}", ha="center", va="top",
                 fontsize=5.5, color=CHAOS_COLORS["red"])
area_prose = np.mean(prose_scores)
area_fts = np.mean(fts_scores)
ax_d.text(0.5, 0.02,
         f"Mean Robustness\nPROSE: {area_prose:.2f}  FTS: {area_fts:.2f}\nΔ = +{(area_prose-area_fts)/area_fts*100:.0f}%",
         transform=ax_d.transAxes, ha="center", va="bottom", fontsize=6.5,
         fontweight="bold", color=CHAOS_COLORS["black"],
         bbox=dict(boxstyle="round,pad=0.2", facecolor=lighten(CHAOS_COLORS["prose"], 0.7),
                   edgecolor=CHAOS_COLORS["prose"], alpha=0.9, lw=1))
ax_d.legend(loc="upper right", bbox_to_anchor=(1.35, 1.18), fontsize=6.5,
           framealpha=0.92, edgecolor="gray")
print("Panel (d) done", flush=True)

# === Labels & Save ===
ax_a.text(-0.18, 1.10, "(a)", transform=ax_a.transAxes, fontsize=11,
         fontweight="bold", va="top", ha="right")
ax_b.text(-0.04, 1.10, "(b)", transform=ax_b.transAxes, fontsize=11,
         fontweight="bold", va="top", ha="right")
ax_c.text(-0.18, 1.10, "(c)", transform=ax_c.transAxes, fontsize=11,
         fontweight="bold", va="top", ha="right")
ax_d.text(-0.12, 1.08, "(d)", transform=ax_d.transAxes, fontsize=11,
         fontweight="bold", va="top", ha="right")
fig.suptitle("PROSE: Dominating the Latency-Recovery Pareto Space",
             fontsize=13, fontweight="bold", y=0.97)
print("Labels done", flush=True)

out = Path("D:/LLM/outputs/chaos_style_figures")
print("Saving PNG...", flush=True)
fig.savefig(out / "fig5_pareto_sensitivity.png", format="png", bbox_inches="tight", dpi=200)
print("PNG OK", flush=True)
print("Saving PDF...", flush=True)
fig.savefig(out / "fig5_pareto_sensitivity.pdf", format="pdf", bbox_inches="tight", dpi=200)
print("PDF OK", flush=True)
print("SAVE OK", flush=True)
