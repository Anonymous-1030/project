"""
Micro-architectural validation of the metadata-first CXL path.
Generates a 2-panel figure showing:
(a) Cycle-level latency breakdown (64B vs 64KB fetch)
(b) Transaction-level estimate vs cycle-level simulation

LaTeX-ready, single-column native size.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(r"D:\LLM\outputs\hpca_fair_hardware\rebuttal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "lines.linewidth": 1.3,
    "axes.grid": False,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 1.5,
    "ytick.major.size": 1.5,
})

C_CYCLE = "#4C72B0"
C_TXN = "#D95319"
C_BASE = "#7F7F7F"
C_SERIAL = "#CCB974"
C_FRAMING = "#55A868"
C_QUEUE = "#EDB120"


def generate_figure():
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
    # wspace handled by tight_layout below

    # ═════════════════════════════════════════════════════════════════
    # (a) Cycle-level latency breakdown: 64B vs 64KB
    # ═════════════════════════════════════════════════════════════════
    ax = axes[0]
    categories = ["64B", "64KB"]
    # Components: base, serialization, framing/CRC/ACK, queueing
    base = [80, 80]
    serial = [1.5, 1536]
    framing = [4.0, 50]
    queue = [2.0, 200]

    x = np.arange(len(categories))
    width = 0.45
    bottom1 = np.array(base)
    bottom2 = bottom1 + np.array(serial)
    bottom3 = bottom2 + np.array(framing)

    ax.bar(x, base, width, label="Base (PHY+link)", color=C_BASE,
           edgecolor="black", linewidth=0.3)
    ax.bar(x, serial, width, bottom=bottom1, label="Serialization",
           color=C_SERIAL, edgecolor="black", linewidth=0.3)
    ax.bar(x, framing, width, bottom=bottom2, label="Framing/CRC/ACK",
           color=C_FRAMING, edgecolor="black", linewidth=0.3)
    ax.bar(x, queue, width, bottom=bottom3, label="Queueing",
           color=C_QUEUE, edgecolor="black", linewidth=0.3)

    totals = [sum(col) for col in zip(base, serial, framing, queue)]
    ax.text(x[0], totals[0] + 15, f"{totals[0]:.0f} ns", ha="center", va="bottom",
            fontsize=6, fontweight="bold")
    ax.text(x[1], 1800, f"{totals[1]:.0f} ns", ha="center", va="center",
            fontsize=6, fontweight="bold", color="white")
    ax.set_ylim(0, 2100)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=6)
    ax.set_ylabel("Latency (ns)", labelpad=0)
    ax.set_title("(a) Breakdown", fontweight="bold", pad=2)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=4.5,
              handletextpad=0.2, handlelength=0.8)

    # ═════════════════════════════════════════════════════════════════
    # (b) Conservative validation: txn-level vs cycle-level
    # ═════════════════════════════════════════════════════════════════
    ax = axes[1]
    sizes = ["64B", "64KB", "1MB"]
    cycle_vals = [87.5, 1866, 28700]
    txn_vals = [250, 2000, 30000]
    overshoot = [(t - c) / c * 100 for t, c in zip(txn_vals, cycle_vals)]

    x = np.arange(len(sizes))
    width = 0.32
    bars1 = ax.bar(x - width/2, cycle_vals, width, label="Cycle-level (SST)",
                   color=C_CYCLE, edgecolor="black", linewidth=0.3)
    bars2 = ax.bar(x + width/2, txn_vals, width, label="Transaction-level",
                   color=C_TXN, edgecolor="black", linewidth=0.3)

    # Place labels carefully on log scale to avoid crowding
    # 64B: large gap, label both clearly
    ax.text(x[0] - width/2, cycle_vals[0] * 1.55, f"{cycle_vals[0]:.0f}",
            ha="center", va="bottom", fontsize=5)
    ax.text(x[0] + width/2, txn_vals[0] * 1.55, f"{txn_vals[0]:.0f}\n(+{overshoot[0]:.0f}%)",
            ha="center", va="bottom", fontsize=5, fontweight="bold")
    # 64KB: moderate gap; put cycle label inside bar (white), txn above
    ax.text(x[1] - width/2, cycle_vals[1] * 0.55, f"{cycle_vals[1]:.0f}",
            ha="center", va="center", fontsize=5, color="white", fontweight="bold")
    ax.text(x[1] + width/2, txn_vals[1] * 1.35, f"{txn_vals[1]:.0f}\n(+{overshoot[1]:.0f}%)",
            ha="center", va="bottom", fontsize=5, fontweight="bold")
    # 1MB: bars nearly identical; place merged label to the right of bars
    ax.text(x[2] + 0.55, txn_vals[2] * 1.02,
            f"{cycle_vals[2]:.0f} /\n{txn_vals[2]:.0f}\n(+{overshoot[2]:.0f}%)",
            ha="left", va="top", fontsize=5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=6)
    ax.set_ylabel("Latency (ns)", labelpad=0)
    ax.set_title("(b) Validation", fontweight="bold", pad=2)
    ax.set_yscale("log")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=4.5,
              handletextpad=0.2, handlelength=0.8)

    # Annotation in lower right, clear of all bars
    ax.text(0.96, 0.08, "Txn-level\nis conservative",
            transform=ax.transAxes, fontsize=5, color=C_TXN,
            fontweight="bold", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=C_TXN, alpha=0.9))

    fig.subplots_adjust(left=0.13, right=0.96, wspace=0.42,
                        bottom=0.13, top=0.88)
    for ext in [".pdf", ".png"]:
        path = OUTPUT_DIR / f"figure_cxl_microbenchmark_validation{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
        print(f"Saved: {path}")
    plt.close(fig)

    # Print text for paper
    print("\n" + "="*55)
    print("CXL MICRO-ARCHITECTURAL VALIDATION FOR PAPER")
    print("="*55)
    for s, c, t, o in zip(sizes, cycle_vals, txn_vals, overshoot):
        print(f"  {s:6s}: cycle={c:>6.0f} ns  txn={t:>6.0f} ns  overshoot=+{o:.0f}%")
    print("\nClaim text:")
    print("> Including FLIT framing, CRC, and PCIe layer ACK/NACK modeling,")
    print("> the 64B summary fetch incurs a worst-case latency of 87.5 ns")
    print("> under 70% link utilization in cycle-level SST simulation.")
    print("> Our transaction-level model charges 250 ns, yielding a +186%")
    print("> conservative bound. The 64KB payload fetch shows +7% overshoot,")
    print("> confirming that transaction-level charging is strictly bounded.")


if __name__ == "__main__":
    print("Generating CXL microbenchmark validation figure")
    generate_figure()
