"""Unified figure-style helper for all eval_*.pdf/.png figures.

All figures are landscape (double-column width). Two supported shapes:

    1x2   →   two panels side-by-side       (7.00" x  2.30")
    2x2   →   four panels in a 2x2 grid     (7.00" x  4.55")

Every panel is an IDENTICAL 2.80" x 1.50" rectangle (absolute inches via
ax.set_position), so the plotting area of every panel across every
figure in the paper has pixel-identical width, height, and x-offset.

The 2x2 layout aligns panels by column (same x0 in each column) and by
row (same y0 in each row), so the figure reads as a clean grid.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Figure-level dimensions (large landscape canvas).
# We render at a larger physical size than the paper's target column width
# so that when the embedded figure is scaled down, fonts and lines remain
# legible without the reader having to zoom in. Each sub-panel is itself a
# clean, self-contained mini-figure.
FIG_W_IN    = 10.00
PANEL_W_IN  = 4.00
PANEL_H_IN  = 2.20

# Margins — tuned so titles/xlabels never clip and panel rects line up.
LEFT_IN     = 0.78
RIGHT_IN    = 0.22       # FIG_W - LEFT - 2*PANEL_W - GAP_X_IN
GAP_X_IN    = 1.00
TOP_IN      = 0.32
GAP_Y_IN    = 0.88       # between top and bottom row in 2x2
BOTTOM_IN   = 0.70
HEADER_LEGEND_IN = 0.46  # extra top reserved for a shared figure legend


PALETTE = dict(
    # Hero policies
    prose      = "#1F3A5A",   # deep navy (our HW architecture)
    oracle     = "#16A085",   # teal (best-in-class oracle)
    # Contrast baselines
    sw_host    = "#E67E22",   # warm orange (SW on host)
    sw_gpu     = "#D35400",   # burnt orange (SW on GPU)
    fts        = "#C0392B",   # brick red (FTS family)
    vllm       = "#7F8C8D",   # slate grey (naive CXL)
    # Secondary categories
    accent_r   = "#B03030",   # red accent for highlights
    accent_g   = "#1E8449",   # green accent for "good" zone
    accent_v   = "#8E44AD",   # purple for secondary metric
    baseline_g = "#7F8C8D",   # grey for reference baseline
    # Zone shading
    shade_ok   = "#E8F3EC",
    shade_bad  = "#FCE4E1",
    shade_neutral = "#ECEFF4",
)


def setup_style():
    """One-shot rcParams for the whole figure suite.

    Fonts are sized for the 10" canvas — they render large and crisp here,
    and remain comfortably readable after the paper's typesetter scales
    the figure to column width.

    Weight is set to 'black' (heaviest) so every text element in the
    figure — titles, axis labels, tick labels, legend entries, and
    per-panel annotations — is visually heavy.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Black", "Arial", "DejaVu Sans"],
        "font.size": 11.0,
        "font.weight": "black",
        "axes.labelsize": 11.5,
        "axes.labelweight": "black",
        "axes.titlesize": 12.5,
        "axes.titleweight": "black",
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 9.5,
        "axes.linewidth": 1.4,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 3.6,
        "ytick.major.size": 3.6,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle":  "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#000000",
        "grid.linestyle": ":",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.25,
        "figure.dpi": 120,
    })


def _figure_size(shape: str, header_legend: bool = False):
    extra = HEADER_LEGEND_IN if header_legend else 0.0
    if shape == "1x2":
        h = TOP_IN + PANEL_H_IN + BOTTOM_IN + extra
    elif shape == "2x2":
        h = TOP_IN + PANEL_H_IN + GAP_Y_IN + PANEL_H_IN + BOTTOM_IN + extra
    else:
        raise ValueError(f"shape must be '1x2' or '2x2', got {shape!r}")
    return (FIG_W_IN, h)


def make_figure(shape: str = "1x2", header_legend: bool = False):
    """Create a figure and return (fig, axes).

    shape='1x2'  ->  axes = [ax_L, ax_R]
    shape='2x2'  ->  axes = [ax_TL, ax_TR, ax_BL, ax_BR]   (row-major)

    If `header_legend=True`, an extra HEADER_LEGEND_IN-inch band is
    reserved above the top row for a shared figure.legend().
    """
    fig_w, fig_h = _figure_size(shape, header_legend)
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Column x-positions are identical in every figure of any shape.
    x0_col = [
        LEFT_IN / fig_w,
        (LEFT_IN + PANEL_W_IN + GAP_X_IN) / fig_w,
    ]
    w_frac = PANEL_W_IN / fig_w
    h_frac = PANEL_H_IN / fig_h

    axes = []
    if shape == "1x2":
        y0 = BOTTOM_IN / fig_h
        for c in (0, 1):
            ax = fig.add_axes((x0_col[c], y0, w_frac, h_frac))
            axes.append(ax)
    else:  # 2x2
        y0_bot = BOTTOM_IN / fig_h
        y0_top = (BOTTOM_IN + PANEL_H_IN + GAP_Y_IN) / fig_h
        for y0 in (y0_top, y0_bot):
            for c in (0, 1):
                ax = fig.add_axes((x0_col[c], y0, w_frac, h_frac))
                axes.append(ax)
    return fig, axes


def style_axis(ax, *, title=None, xlabel=None, ylabel=None,
               ylabel_color=None, grid=True):
    """Apply the standard axis cosmetics uniformly."""
    if title is not None:
        ax.set_title(title, fontweight="black", loc="left", pad=2.8)
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=1.8, fontweight="black")
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=2.2, fontweight="black",
                      color=ylabel_color if ylabel_color else "black")
    if grid:
        ax.grid(True, which="major", linestyle=":", linewidth=0.42, alpha=0.55)
        ax.set_axisbelow(True)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("black")


def style_twin_axis(ax_r, *, ylabel=None, ylabel_color=None):
    """Style a twin-y axis the same way the primary axes are styled."""
    ax_r.spines["top"].set_visible(False)
    if ylabel is not None:
        ax_r.set_ylabel(ylabel, color=ylabel_color or "black",
                        labelpad=4, fontweight="black",
                        rotation=270, va="bottom")
    if ylabel_color is not None:
        ax_r.tick_params(axis="y", colors=ylabel_color)
    for lbl in ax_r.get_yticklabels():
        lbl.set_fontweight("black")


def callout(ax, x, y, text, *, color="#1F3A5A", fc="#ECF2F8",
            ha="left", va="top", fontsize=5.8, mono=False,
            transform=None):
    """Standard callout-box style used in all panels."""
    kw = dict(
        transform=transform if transform is not None else ax.transAxes,
        ha=ha, va=va,
        fontsize=fontsize, color=color, fontweight="black",
        bbox=dict(boxstyle="round,pad=0.25", fc=fc,
                  ec=color, lw=0.5, alpha=0.94),
    )
    if mono:
        kw["family"] = "monospace"
    return ax.text(x, y, text, **kw)


def save(fig, out_path_stem):
    """Save identical PDF + PNG pair at paper-grade DPI."""
    pdf = str(out_path_stem) + ".pdf"
    png = str(out_path_stem) + ".png"
    fig.savefig(pdf, bbox_inches=None, pad_inches=0.02)
    fig.savefig(png, dpi=360, bbox_inches=None, pad_inches=0.02)
    plt.close(fig)
    return pdf, png
