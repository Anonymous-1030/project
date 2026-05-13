"""
Chaos-style plotting utilities for ProSE v2 figures.

Inspired by chaos.pdf visual language:
- High information density (3-4 panels per figure)
- Unified cool-warm palette (deep blue -> teal -> orange -> red)
- Heavy use of heatmaps for correlation / distribution
- Clear data-flow architecture diagrams
- Sans-serif fonts, minimum 8pt, vector PDF output
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Chaos Color Palette
# ---------------------------------------------------------------------------
CHAOS_COLORS = {
    # Core semantic colors
    "prose": "#1f4e79",          # Deep blue - PROSE / metadata path
    "prose_light": "#5b9bd5",    # Light blue - PROSE auxiliary
    "teal": "#2e9da6",           # Teal - cue / intermediate
    "green": "#548235",          # Deep green - committed / valid
    "green_light": "#a9d18e",    # Light green - success background
    "orange": "#c55a11",         # Orange-red - FTS / invalid payload
    "orange_light": "#f4b084",   # Light orange - warning
    "red": "#c00000",            # Deep red - reject / abort / stall
    "purple": "#6b4c8a",         # Purple - oracle / highlight
    "purple_light": "#b4a7d6",   # Light purple
    "gray": "#7f7f7f",           # Gray - baseline / disabled
    "gray_light": "#d9d9d9",     # Light gray - background
    "black": "#1a1a1a",          # Near-black - text / borders
    "white": "#ffffff",
    # Traffic categories
    "metadata_read": "#5b9bd5",
    "valid_payload": "#548235",
    "invalid_payload": "#c55a11",
    "writeback": "#a6a6a6",
    # Decision categories
    "promote": "#548235",
    "defer": "#ffc000",
    "reject": "#c00000",
}

# Categorical palette for bars/lines
CATEGORICAL = [
    CHAOS_COLORS["prose"],
    CHAOS_COLORS["teal"],
    CHAOS_COLORS["green"],
    CHAOS_COLORS["orange"],
    CHAOS_COLORS["red"],
    CHAOS_COLORS["purple"],
    CHAOS_COLORS["gray"],
]

# Warm-cool diverging for heatmaps
DIVERGING_CMAP = "RdBu_r"
SEQUENTIAL_CMAP = "mako"  # Deep blue -> purple -> pink (dark background friendly)
SEQUENTIAL_CMAP_LIGHT = "rocket"  # For white-background heatmaps

# ---------------------------------------------------------------------------
# Matplotlib style configuration
# ---------------------------------------------------------------------------
CHAOS_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "axes.linewidth": 0.6,
    "axes.edgecolor": CHAOS_COLORS["black"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def setup_chaos_style():
    """Apply chaos-style matplotlib configuration."""
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    mpl.use("Agg")
    for key, val in CHAOS_RC.items():
        mpl.rcParams[key] = val
    return plt


def teardown_chaos_style():
    """Reset matplotlib defaults."""
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.close("all")


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def export_figure(fig, output_dir: Path, name: str, formats=None):
    if formats is None:
        formats = ["png", "pdf"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, bbox_inches="tight", dpi=300)
        paths[fmt] = path
    return paths


# ---------------------------------------------------------------------------
# Multi-panel helpers
# ---------------------------------------------------------------------------
def create_multi_panel(nrows=1, ncols=2, figsize_per_panel=(3.2, 2.4), wspace=0.35, hspace=0.40):
    """Create a high-density multi-panel figure.

    Default: 3.2x2.4 in per panel -> ~2-column paper friendly.
    """
    from matplotlib import pyplot as plt

    figsize = (ncols * figsize_per_panel[0], nrows * figsize_per_panel[1])
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    return fig, axes


def label_panels(axes, labels=None, fontsize=11, fontweight="bold", x=-0.18, y=1.06):
    """Add (a), (b), (c)... labels to panels."""
    if labels is None:
        labels = [f"({chr(97 + i)})" for i in range(axes.size)]
    for ax, lab in zip(axes.flat, labels):
        ax.text(x, y, lab, transform=ax.transAxes, fontsize=fontsize,
                fontweight=fontweight, va="top", ha="right")


# ---------------------------------------------------------------------------
# Heatmap helpers (seaborn)
# ---------------------------------------------------------------------------
def plot_heatmap(ax, data, row_labels, col_labels, cmap=SEQUENTIAL_CMAP,
                 vmin=None, vmax=None, cbar_label="", annotate=True,
                 fmt=".2f", fontsize=7, linewidths=0.5):
    """Plot a polished heatmap on the given axes."""
    import seaborn as sns

    sns.heatmap(data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                annot=annotate, fmt=fmt, annot_kws={"size": fontsize},
                linewidths=linewidths, linecolor="white",
                cbar_kws={"label": cbar_label, "shrink": 0.8},
                xticklabels=col_labels, yticklabels=row_labels,
                square=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Rotate x labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------
def add_arrow_annotation(ax, text, xy, xytext, color="#333333", fontsize=8, arrowprops=None):
    if arrowprops is None:
        arrowprops = dict(arrowstyle="->", color=color, lw=1.0)
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fontsize,
                color=color, fontweight="bold",
                arrowprops=arrowprops,
                ha="center", va="center")


def add_horizontal_brace(ax, x1, x2, y, text, color="#333333", fontsize=8):
    """Draw a horizontal brace with text above it."""
    ax.plot([x1, x1, x2, x2], [y - 0.01, y, y, y - 0.01], color=color, lw=1.0, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.text((x1 + x2) / 2, y + 0.015, text, ha="center", va="bottom",
            fontsize=fontsize, color=color, fontweight="bold", transform=ax.get_xaxis_transform(), clip_on=False)


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------
def lighten(hex_color, amount=0.3):
    """Lighten a hex color by mixing with white."""
    from matplotlib.colors import to_rgb
    rgb = np.array(to_rgb(hex_color))
    rgb = rgb + (1 - rgb) * amount
    return "#" + "".join(f"{int(c * 255):02x}" for c in rgb)


def darken(hex_color, amount=0.3):
    """Darken a hex color by mixing with black."""
    from matplotlib.colors import to_rgb
    rgb = np.array(to_rgb(hex_color))
    rgb = rgb * (1 - amount)
    return "#" + "".join(f"{int(c * 255):02x}" for c in rgb)
