"""Shared plotting/IO helpers for rebuttal experiments."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "out" / "data"
FIG_DIR = ROOT / "out" / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":         150,
    "savefig.dpi":        600,
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          25,
    "font.weight":        "bold",
    "axes.titlesize":     27,
    "axes.titleweight":   "bold",
    "axes.labelsize":     25,
    "axes.labelweight":   "bold",
    "legend.fontsize":    20,
    "legend.frameon":     False,
    "xtick.labelsize":    22,
    "ytick.labelsize":    22,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     2.0,
    "lines.linewidth":    3.5,
    "lines.markersize":   12,
    "figure.constrained_layout.use": True,
})

# Colour palette — CB-friendly
C = {
    "fts":     "#d62728",
    "sw_host": "#ff7f0e",
    "sw_gpu":  "#bcbd22",
    "iommu":   "#17becf",
    "cefe":    "#2ca02c",
    "oracle":  "#7f7f7f",
    "accent1": "#1f77b4",
    "accent2": "#9467bd",
}


def save_json(name: str, obj: Dict[str, Any]) -> Path:
    path = DATA_DIR / f"{name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def save_fig(fig, name: str) -> Path:
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png
