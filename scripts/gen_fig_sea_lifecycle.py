"""
Generate SEA lifecycle figure for paper.
Single-column (3.4in), compact, readable at 8pt font.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(3.4, 1.6), dpi=300)
ax.set_xlim(-0.2, 3.6)
ax.set_ylim(-0.15, 1.5)
ax.axis("off")

# Colors
c_rej = "#BDBDBD"
c_spec = "#FF8F00"
c_adm = "#2E7D32"
bg_rej = "#F5F5F5"
bg_spec = "#FFF8E1"
bg_adm = "#E8F5E9"

# State boxes
bw, bh = 0.85, 0.5
cy = 0.45

boxes = [
    (0.45, cy, "REJECTED", bg_rej, c_rej, "no fetch"),
    (1.75, cy, "SPECULATIVE", bg_spec, c_spec, "low-pri fetch\n+ verify"),
    (3.05, cy, "ADMITTED", bg_adm, c_adm, "visible set\n+ feedback"),
]

for x, y, title, bg, ec, sub in boxes:
    b = FancyBboxPatch((x - bw/2, y - bh/2), bw, bh,
                       boxstyle="round,pad=0.04", fc=bg, ec=ec, lw=1.4)
    ax.add_patch(b)
    ax.text(x, y + 0.1, title, ha="center", va="center", fontsize=6, fontweight="bold", color=ec)
    ax.text(x, y - 0.12, sub, ha="center", va="center", fontsize=4.8, color="#616161", linespacing=1.05)

# Forward arrows
akw = dict(arrowstyle="->,head_width=0.1,head_length=0.07", lw=1.1)

# REJ → SPEC
ax.annotate("", xy=(1.32, cy + 0.12), xytext=(0.88, cy + 0.12), arrowprops=dict(**akw, color=c_spec))
ax.text(1.1, cy + 0.35, r"$E_j > \theta_s$", ha="center", fontsize=5.5, color=c_spec)

# SPEC → ADM
ax.annotate("", xy=(2.62, cy + 0.12), xytext=(2.18, cy + 0.12), arrowprops=dict(**akw, color=c_adm))
ax.text(2.4, cy + 0.35, "verified", ha="center", fontsize=5.5, color=c_adm, style="italic")

# Backward arrows (decay/evict)
akw_back = dict(arrowstyle="->,head_width=0.08,head_length=0.06", lw=0.8, linestyle="dashed")

# ADM → REJ (long arc below)
ax.annotate("", xy=(0.88, cy - 0.18), xytext=(2.62, cy - 0.18), arrowprops=dict(**akw_back, color=c_rej))
ax.text(1.75, cy - 0.38, "decay below θ", ha="center", fontsize=4.8, color="#757575", style="italic")

# SPEC → REJ (short)
ax.annotate("", xy=(0.88, cy - 0.05), xytext=(1.32, cy - 0.05),
            arrowprops=dict(arrowstyle="->,head_width=0.06,head_length=0.05", lw=0.7, color=c_rej, linestyle="dotted"))

# Evidence formula top center
ax.text(1.75, 1.28, r"$E_j^{(t)} = \gamma \, E_j^{(t-1)} + (1-\gamma)\,["
        r"w_h s_j + w_v v_j + w_a a_j + w_t \tau_j]$",
        ha="center", va="center", fontsize=5.5,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#BDBDBD", lw=0.7))

fig.savefig("figure/sea-lifecycle.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("figure/sea-lifecycle.png", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("Done: figure/sea-lifecycle.pdf (3.4in × 1.6in)")
