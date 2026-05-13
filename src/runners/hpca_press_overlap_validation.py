"""
PRESS Overlapping-Scale Validation.

Core idea: In the directly measurable 8K--32K regime, construct
pressure-equivalent subscale runs and verify that PRESS predicts the
natural-scale results. If the matched runs preserve policy ordering
and exposed-stall trends, PRESS is calibrated before use at 128K--1M.

Design:
  Natural run at scale S:      N_chunks = S/64,  budget_ratio = 0.10
  PRESS-match for S at S/2:    N_chunks = S/128, budget_ratio = 0.20
                               (same absolute budget_chunks -> same promotion pressure)
  Deliberately unmatched:      N_chunks = S/128, budget_ratio = 0.10
                               (half budget_chunks -> pressure mismatch)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.runners.hpca_rebuttal_experiments import (
    RebuttalSimulator,
    _generate_attention_sequence,
)

logger = logging.getLogger(__name__)


class PRESSOverlapValidator:
    """Validate PRESS by overlapping natural and pressure-matched subscale runs."""

    # Target scales and their configurations
    CONFIGS = {
        # 8K target
        "natural_8k": {"num_chunks": 128, "budget_chunks": 13, "seq_len": 8192},
        "press_8k_at_4k": {"num_chunks": 64, "budget_chunks": 13, "seq_len": 4096},
        "unmatched_8k_at_4k": {"num_chunks": 64, "budget_chunks": 6, "seq_len": 4096},
        # 16K target
        "natural_16k": {"num_chunks": 256, "budget_chunks": 26, "seq_len": 16384},
        "press_16k_at_8k": {"num_chunks": 128, "budget_chunks": 26, "seq_len": 8192},
        "unmatched_16k_at_8k": {"num_chunks": 128, "budget_chunks": 13, "seq_len": 8192},
        # 32K target
        "natural_32k": {"num_chunks": 512, "budget_chunks": 51, "seq_len": 32768},
        "press_32k_at_16k": {"num_chunks": 256, "budget_chunks": 51, "seq_len": 16384},
        "unmatched_32k_at_16k": {"num_chunks": 256, "budget_chunks": 26, "seq_len": 16384},
    }

    METHODS = ["prose", "prose_fts", "freqrec_prefetcher", "stream_prefetcher"]
    NUM_STEPS = 50
    ANCHOR_RATIO = 0.10

    def _make_trace(self, num_chunks: int, seed: int = 42) -> Dict[str, Any]:
        """Generate a trace with statistically consistent attention pattern."""
        rng = np.random.RandomState(seed)
        base_attn = rng.exponential(1.0, num_chunks)
        base_attn = base_attn / base_attn.sum()
        return {"num_chunks": num_chunks, "chunk_attention": base_attn}

    def run(self, output_dir: Path) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("[PRESS Overlap Validation]")
        logger.info("=" * 60)

        simulator = RebuttalSimulator()
        results: Dict[str, Dict[str, Any]] = {}

        for cfg_name, cfg in self.CONFIGS.items():
            results[cfg_name] = {}
            trace = self._make_trace(cfg["num_chunks"])
            budget_ratio = cfg["budget_chunks"] / cfg["num_chunks"]

            for method in self.METHODS:
                res = simulator.simulate_single(
                    method=method,
                    trace=trace,
                    budget_ratio=budget_ratio,
                    num_decode_steps=self.NUM_STEPS,
                    seq_len=cfg["seq_len"],
                )
                results[cfg_name][method] = {
                    "mean_recovery": round(res.mean_recovery, 4),
                    "latency_us": round(res.latency_us, 2),
                    "throughput_tps": round(res.throughput_tps, 1),
                    "queue_utilization": round(res.queue_utilization, 4),
                    "invalid_traffic_ratio": round(res.invalid_traffic_ratio, 4),
                    "p99_latency_us": round(res.p99_latency_us, 2),
                }

        # Compute prediction errors: |natural - press_match| / natural
        errors = {}
        for target in ("8k", "16k", "32k"):
            nat_key = f"natural_{target}"
            press_key = f"press_{target}_at_{int(target[:-1])//2}k"
            um_key = f"unmatched_{target}_at_{int(target[:-1])//2}k"

            errors[target] = {}
            for method in self.METHODS:
                nat = results[nat_key][method]
                press = results[press_key][method]
                um = results[um_key][method]

                def rel_err(n, p):
                    return abs(n - p) / max(abs(n), 1e-9)

                errors[target][method] = {
                    "recovery_press": round(rel_err(nat["mean_recovery"], press["mean_recovery"]), 4),
                    "latency_press": round(rel_err(nat["latency_us"], press["latency_us"]), 4),
                    "throughput_press": round(rel_err(nat["throughput_tps"], press["throughput_tps"]), 4),
                    "rho_press": round(rel_err(nat["queue_utilization"], press["queue_utilization"]), 4),
                    "recovery_um": round(rel_err(nat["mean_recovery"], um["mean_recovery"]), 4),
                    "latency_um": round(rel_err(nat["latency_us"], um["latency_us"]), 4),
                    "throughput_um": round(rel_err(nat["throughput_tps"], um["throughput_tps"]), 4),
                    "rho_um": round(rel_err(nat["queue_utilization"], um["queue_utilization"]), 4),
                }

        out = output_dir / "press_overlap_validation.json"
        with open(out, "w") as f:
            json.dump({
                "experiment": "PRESS_Overlap_Validation",
                "results": results,
                "prediction_errors": errors,
            }, f, indent=2)
        logger.info(f"  Saved to {out}")

        return {"results": results, "errors": errors}


def generate_figure(data: Dict[str, Any], output_dir: Path):
    """Generate the overlapping-scale validation figure."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    results = data["results"]
    targets = [
        ("8k", "8K Target"),
        ("16k", "16K Target"),
        ("32k", "32K Target"),
    ]
    methods = ["prose", "prose_fts", "freqrec_prefetcher", "stream_prefetcher"]
    method_labels = {
        "prose": "PROSE",
        "prose_fts": "PROSE-FTS",
        "freqrec_prefetcher": "FreqRec-PF",
        "stream_prefetcher": "StreamPF",
    }
    method_colors = {
        "prose": "#2E7D32",
        "prose_fts": "#C62828",
        "freqrec_prefetcher": "#1565C0",
        "stream_prefetcher": "#F57C00",
    }

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(
        "PRESS Overlapping-Scale Validation: Natural vs Pressure-Equivalent Subscale",
        fontsize=12,
    )

    metrics = [
        ("latency_us", "Latency (us)"),
        ("mean_recovery", "Recovery"),
        ("queue_utilization", "Queue Utilization (rho)"),
    ]

    for row_idx, (target, title) in enumerate(targets):
        nat_key = f"natural_{target}"
        press_key = f"press_{target}_at_{int(target[:-1])//2}k"
        um_key = f"unmatched_{target}_at_{int(target[:-1])//2}k"

        for col_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            x = np.arange(len(methods))
            width = 0.25

            nat_vals = [results[nat_key][m][metric] for m in methods]
            press_vals = [results[press_key][m][metric] for m in methods]
            um_vals = [results[um_key][m][metric] for m in methods]

            ax.bar(x - width, nat_vals, width, label="Natural", color="#37474F")
            ax.bar(x, press_vals, width, label="PRESS-matched", color="#00897B")
            ax.bar(x + width, um_vals, width, label="Delib. unmatched", color="#E53935")

            ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(ylabel)
            if col_idx == 0:
                ax.text(
                    -0.15, 0.5, title, transform=ax.transAxes,
                    fontsize=10, fontweight="bold", va="center", ha="right", rotation=90,
                )
            ax.set_xticks(x)
            ax.set_xticklabels([method_labels[m] for m in methods], rotation=15, ha="right")
            ax.grid(True, alpha=0.3, axis="y")
            if row_idx == 0 and col_idx == 2:
                ax.legend(frameon=True, loc="upper left")

    out = output_dir / "fig_press_overlap_validation.png"
    fig.savefig(out)
    logger.info(f"Saved {out}")
    plt.close(fig)

    # Second figure: prediction error summary
    errors = data["errors"]
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    fig2.suptitle("PRESS Prediction Error: Matched vs Unmatched", fontsize=12)

    for idx, target in enumerate(("8k", "16k", "32k")):
        ax = axes2[idx]
        x = np.arange(len(methods))
        width = 0.35

        press_errs = [errors[target][m]["latency_press"] for m in methods]
        um_errs = [errors[target][m]["latency_um"] for m in methods]

        ax.bar(x - width/2, press_errs, width, label="PRESS-matched", color="#00897B")
        ax.bar(x + width/2, um_errs, width, label="Delib. unmatched", color="#E53935")

        ax.set_ylabel("Relative Error (Latency)")
        ax.set_title(f"{target.upper()} Target")
        ax.set_xticks(x)
        ax.set_xticklabels([method_labels[m] for m in methods], rotation=15, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        if idx == 2:
            ax.legend(frameon=True)

    out2 = output_dir / "fig_press_prediction_error.png"
    fig2.savefig(out2)
    logger.info(f"Saved {out2}")
    plt.close(fig2)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/hpca_rebuttal")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    validator = PRESSOverlapValidator()
    data = validator.run(outdir)
    generate_figure(data, outdir)

    logger.info("PRESS overlap validation complete.")


if __name__ == "__main__":
    main()
