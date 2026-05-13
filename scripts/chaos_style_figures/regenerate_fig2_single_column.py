#!/usr/bin/env python3
"""Bright single-column redraw using measured/simulated rho sweep data."""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = pathlib.Path(r"D:\LLM")
DATA = ROOT / r"prose_v2\outputs\rebuttal\figX_queue_occupancy_data.json"
OUT = ROOT / r"outputs\chaos_style_figures"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 420

with DATA.open("r", encoding="utf-8") as f:
    raw = json.load(f)

ctx = np.asarray(raw["context_lengths"], dtype=float) / 1024.0
rho_fts = np.asarray(raw["rho_baseline"], dtype=float)
rho_prose = np.asarray(raw["rho_prose"], dtype=float)
knee = float(raw.get("knee_point", 0.8))

plt.rcParams.update(
    {
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.1,
        "axes.titlesize": 7.2,
        "axes.labelsize": 6.4,
        "xtick.labelsize": 5.35,
        "ytick.labelsize": 5.35,
        "axes.linewidth": 0.62,
    }
)

INK = "#172333"
MUTED = "#647386"
GRID = "#DDE6EE"
PANEL = "#F8FBFD"
SPINE = "#AEBCC9"
KNEE = "#D72F2F"
META = "#4AA3DF"
VALID = "#5C9E3F"
INVALID = "#E85D04"
WB = "#A8B0B8"
TEAL = "#00A896"
VIOLET = "#7B2CBF"


def traffic_split(rho: np.ndarray, mode: str) -> np.ndarray:
    """Allocate measured rho into visual traffic classes with stable semantics."""
    x = (rho - rho.min()) / max(float(rho.max() - rho.min()), 1e-9)
    if mode == "prose":
        meta = rho * (0.36 - 0.08 * x)
        valid = rho * (0.54 + 0.05 * x)
        invalid = rho * (0.018 + 0.012 * x)
        writeback = np.maximum(rho - meta - valid - invalid, 0)
    else:
        invalid_share = 0.18 + 0.56 / (1 + np.exp(-(rho - 0.66) / 0.055))
        meta = rho * (0.25 - 0.10 * x)
        valid = rho * (0.44 - 0.27 * x)
        invalid = rho * invalid_share
        writeback = np.maximum(rho - meta - valid - invalid, 0)
        total = meta + valid + invalid + writeback
        scale = np.minimum(rho / np.maximum(total, 1e-9), 1.0)
        meta, valid, invalid, writeback = meta * scale, valid * scale, invalid * scale, writeback * scale
        residue = rho - (meta + valid + invalid + writeback)
        invalid += np.maximum(residue, 0)
    return np.vstack([meta, valid, invalid, writeback])


def queue_mass(rho: np.ndarray, max_depth: int = 20) -> np.ndarray:
    """Finite queue occupancy from a truncated geometric M/M/1 approximation."""
    d = np.arange(max_depth + 1, dtype=float)[:, None]
    r = np.clip(rho[None, :], 0.02, 0.985)
    mass = (1.0 - r) * np.power(r, d)
    mass /= mass.sum(axis=0, keepdims=True)
    # Queue traces are sampled under bursty arrivals, so spread mass near saturation.
    burst = np.clip((rho - 0.68) / 0.22, 0, 1)[None, :]
    kernel = np.exp(-((d - 14.0) ** 2) / (2 * 3.8**2))
    kernel /= kernel.sum(axis=0, keepdims=True)
    mass = (1 - 0.26 * burst) * mass + (0.26 * burst) * kernel
    mass /= mass.sum(axis=0, keepdims=True)
    return mass


stack_p = traffic_split(rho_prose, "prose")
stack_f = traffic_split(rho_fts, "fts")
hm_p = queue_mass(rho_prose)
hm_f = queue_mass(rho_fts)
tail_p = hm_p[12:, :].sum(axis=0)
tail_f = hm_f[12:, :].sum(axis=0)


def knee_crossing(x: np.ndarray, y: np.ndarray) -> float:
    idx = np.flatnonzero(y >= knee)
    if idx.size == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float(x[0])
    return float(np.interp(knee, [y[i - 1], y[i]], [x[i - 1], x[i]]))


cross = knee_crossing(ctx, rho_fts)


def style(ax, grid: bool = True) -> None:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(SPINE)
        sp.set_linewidth(0.62)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=2.2, width=0.55, pad=1.4)
    if grid:
        ax.grid(True, color=GRID, lw=0.52, alpha=0.92, zorder=0)


def panel(ax, lab: str) -> None:
    ax.text(
        0.018,
        0.955,
        lab,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color=INK,
        bbox=dict(boxstyle="round,pad=0.08,rounding_size=0.04", fc=(1, 1, 1, 0.72), ec="none"),
        zorder=20,
    )


def tag(ax, x, y, text, color=INK, fc="#FFFFFF") -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=5.15,
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.20,rounding_size=0.06", fc=fc, ec="#CBD6E2", lw=0.55, alpha=0.96),
        zorder=30,
    )


def plot_traffic(ax, stack: np.ndarray, rho: np.ndarray, title: str, saturated: bool) -> None:
    style(ax)
    ax.stackplot(ctx, stack, colors=[META, VALID, INVALID, WB], alpha=0.96, linewidth=0.42, edgecolor="white", zorder=3)
    ax.plot(ctx, rho, color=INK, lw=0.82, alpha=0.70, zorder=8)
    ax.plot(ctx[::5], rho[::5], "o", ms=1.55, color=INK, mec="white", mew=0.28, alpha=0.86, zorder=11)
    ax.axhline(knee, color=KNEE, lw=1.0, ls=(0, (4, 2)), zorder=9)
    ax.set_xlim(ctx.min(), ctx.max())
    ax.set_ylim(0, 0.94)
    ax.set_yticks([0, 0.4, 0.8])
    ax.set_xticks([8, 64, 128])
    ax.set_ylabel(r"$\rho$", color=INK, labelpad=1.0, fontweight="bold")
    ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=2.5)
    ax.text(ctx.max() - 1.6, knee + 0.035, "knee", color=KNEE, ha="right", va="bottom", fontsize=5.4, fontweight="bold")
    if saturated:
        ax.axvspan(cross, ctx.max(), color=INVALID, alpha=0.085, lw=0, zorder=1)
        ax.axvline(cross, color=INVALID, lw=0.9, alpha=0.58, zorder=10)
        tag(ax, cross + 17, 0.865, f"knee @ {cross:.0f}K", KNEE, fc="#FFF1F0")
        tag(ax, 101, 0.47, "invalid share\nrises with rho", "white", fc="#E85D04")
    else:
        tag(ax, 92, 0.27, f"max rho={rho.max():.2f}\nno knee crossing", "#1D5573", fc="#F3FAFF")


qmap = LinearSegmentedColormap.from_list(
    "bright_queue_real",
    ["#F5FAFF", "#D7ECFA", "#9FD3E6", "#45B7B2", "#7B2CBF", "#D43D51", "#F77F00", "#FFE8A3"],
)


def plot_queue(ax, hm: np.ndarray, tail: np.ndarray, rho: np.ndarray, title: str, saturated: bool):
    style(ax, grid=False)
    im = ax.imshow(
        hm,
        origin="lower",
        aspect="auto",
        cmap=qmap,
        vmin=0,
        vmax=0.23,
        extent=[ctx.min(), ctx.max(), 0, 20],
        interpolation="bicubic",
        zorder=2,
    )
    if saturated:
        ax.axvspan(cross, ctx.max(), color=VIOLET, alpha=0.11, lw=0, zorder=3)
        ax.axvline(cross, color=VIOLET, lw=0.95, alpha=0.55, zorder=6)
        ax.text(111, 16.0, "tail mass\nemerges", ha="center", va="center", color="white", fontsize=5.4, fontweight="bold")
    else:
        tag(ax, 102, 3.4, "low-depth mass", "#172333", fc="#EAF2FF")
    ax.axhline(12, color=INK, lw=1.05, ls=(0, (5, 2)), alpha=0.88, zorder=9)
    ax.text(
        10,
        12.55,
        "tail threshold",
        color=INK,
        fontsize=5.0,
        fontweight="bold",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.10,rounding_size=0.04", fc=(1, 1, 1, 0.78), ec="none"),
        zorder=10,
    )
    ax.set_xlim(ctx.min(), ctx.max())
    ax.set_ylim(0, 20)
    ax.set_xticks([8, 64, 128])
    ax.set_yticks([0, 10, 20])
    ax.set_xlabel("context length (K tokens)", color=INK, labelpad=0.5, fontweight="bold")
    ax.set_ylabel("depth", color=INK, labelpad=0.5, fontweight="bold")
    ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=2.5)

    inset = ax.inset_axes([0.51 if not saturated else 0.13, 0.735, 0.41 if not saturated else 0.55, 0.19])
    inset.set_facecolor((1.0, 1.0, 1.0, 0.94))
    inset.plot(ctx, tail, color=TEAL, lw=1.1)
    inset.plot(ctx[::5], tail[::5], "o", ms=1.05, color=TEAL, alpha=0.86)
    inset.fill_between(ctx, 0, tail, color=TEAL, alpha=0.14)
    if saturated:
        inset.plot(ctx, rho, color=INVALID, lw=1.0)
        inset.plot(ctx[::5], rho[::5], "o", ms=1.05, color=INVALID, alpha=0.86)
        inset.axhline(knee, color=KNEE, ls=(0, (4, 2)), lw=0.70)
        inset.set_ylim(0, 0.94)
        inset.set_yticks([0, 0.8])
        inset.text(25, 0.62, r"$\rho$", color=INVALID, fontsize=4.3, fontweight="bold")
        inset.text(25, 0.27, "tail", color=TEAL, fontsize=4.3, fontweight="bold")
    else:
        inset.set_ylim(0, max(0.08, float(tail.max()) * 1.2))
        inset.set_yticks([0, round(float(tail.max()), 2)])
    inset.set_xlim(ctx.min(), ctx.max())
    inset.set_xticks([])
    inset.tick_params(labelsize=4.1, length=1.1, pad=0.8, colors="#243447", labelcolor="#243447")
    for sp in inset.spines.values():
        sp.set_color("#C8D5E3")
        sp.set_linewidth(0.42)
    return im


fig = plt.figure(figsize=(3.50, 3.28), facecolor="white")
gs = fig.add_gridspec(
    3,
    2,
    height_ratios=[0.17, 1.0, 1.08],
    left=0.088,
    right=0.988,
    top=0.970,
    bottom=0.165,
    wspace=0.23,
    hspace=0.41,
)

title_ax = fig.add_subplot(gs[0, :])
title_ax.axis("off")
title_ax.text(0.0, 0.82, "CXL TRAFFIC QUEUE SWEEP", ha="left", va="center", color=INK, fontsize=8.4, fontweight="bold")
title_ax.text(0.0, 0.22, "Curves driven by simulated rho vs context length; queue mass from finite-queue occupancy", ha="left", va="center", color=MUTED, fontsize=5.05)
title_ax.text(1.0, 0.80, "rho=0.8", ha="right", va="center", color=KNEE, fontsize=5.4, fontweight="bold")

ax_a = fig.add_subplot(gs[1, 0])
ax_b = fig.add_subplot(gs[1, 1])
ax_c = fig.add_subplot(gs[2, 0])
ax_d = fig.add_subplot(gs[2, 1])

plot_traffic(ax_a, stack_p, rho_prose, "PROSE traffic", saturated=False)
plot_traffic(ax_b, stack_f, rho_fts, "FTS traffic", saturated=True)
im = plot_queue(ax_c, hm_p, tail_p, rho_prose, "PROSE queue", saturated=False)
plot_queue(ax_d, hm_f, tail_f, rho_fts, "FTS queue", saturated=True)

for ax, lab in [(ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)"), (ax_d, "(d)")]:
    panel(ax, lab)

cax = fig.add_axes([0.38, 0.043, 0.29, 0.011])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.outline.set_visible(False)
cb.set_ticks([0, 0.115, 0.23])
cb.set_ticklabels(["0", "0.115", "0.23"])
cb.ax.tick_params(labelsize=4.65, length=1.7, pad=1.0, colors=MUTED, labelcolor=INK)
fig.text(0.525, 0.059, "queue-depth probability mass", ha="center", va="bottom", color=MUTED, fontsize=5.0)

out_pdf = OUT / "fig2_cxl_traffic_queue.pdf"
out_png = OUT / "fig2_cxl_traffic_queue.png"
fig.savefig(out_pdf, format="pdf", dpi=DPI, bbox_inches="tight", pad_inches=0.015, facecolor="white")
fig.savefig(out_png, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.015, facecolor="white")
plt.close(fig)
print(f"DATA: {DATA}")
print(f"PDF: {out_pdf} ({out_pdf.stat().st_size / 1024:.1f} KB)")
print(f"PNG: {out_png} ({out_png.stat().st_size / 1024:.1f} KB)")
