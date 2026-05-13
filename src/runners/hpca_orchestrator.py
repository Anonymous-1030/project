"""Unified HPCA Experiment Orchestrator.

One-click runner for the complete HPCA evaluation:
  1. ODUS training (trace → label → MLP → LUT distillation)
  2. End-to-end benchmark evaluation (Passkey / LongBench / RULER)
  3. Promotion miss characterization
  4. Hardware sensitivity analysis
  5. Figure & table generation (JSON + LaTeX)

Usage:
    python -m prosex.src.runners.hpca_orchestrator --mode all
    python -m prosex.src.runners.hpca_orchestrator --mode figures_only
    python -m prosex.src.runners.hpca_orchestrator --mode train_odus
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the full HPCA experiment suite."""
    # Output
    output_dir: str = "outputs/hpca"

    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "cuda"
    dtype: str = "float16"

    # ODUS training
    train_odus: bool = True
    odus_num_samples: int = 50
    odus_epochs: int = 100
    odus_chunk_size: int = 64

    # Benchmarks
    run_benchmarks: bool = True
    methods: List[str] = field(default_factory=lambda: [
        "full_kv", "h2o", "snapkv", "streaming", "prose",
    ])
    budget_ratios: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20, 0.40])
    passkey_lengths: List[int] = field(default_factory=lambda: [1024, 4096, 16384])
    longbench_tasks: List[str] = field(default_factory=lambda: [
        "hotpotqa", "narrativeqa", "qasper",
    ])
    samples_per_config: int = 5

    # Figures
    generate_figures: bool = True

    # Miss characterization
    run_miss_study: bool = True
    miss_num_chunks: int = 200
    miss_num_steps: int = 100


class HPCAOrchestrator:
    """Orchestrates the complete HPCA experiment pipeline."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.out = Path(config.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Any] = {}

    # ── 1. ODUS Training ─────────────────────────────────────────────

    def train_odus(self) -> Dict[str, Any]:
        """Train ODUS MLP and distill to PPU LUT."""
        logger.info("=" * 60)
        logger.info("[1/5] ODUS Training Pipeline")
        logger.info("=" * 60)

        odus_dir = self.out / "odus"
        odus_dir.mkdir(exist_ok=True)
        model_path = str(odus_dir / "odus_model.pt")

        try:
            import torch
            if not torch.cuda.is_available() and self.config.device == "cuda":
                logger.warning("CUDA not available, using synthetic ODUS training")
                return self._train_odus_synthetic(model_path)

            from src.training.pipeline.odus_trainer import run_training_pipeline

            # Load model wrapper
            from transformers import AutoModelForCausalLM, AutoTokenizer

            class SimpleModelWrapper:
                def __init__(self, model_name, device, dtype):
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name, trust_remote_code=True,
                    )
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name, torch_dtype=dtype_map.get(dtype, torch.float16),
                        device_map=device, trust_remote_code=True,
                        attn_implementation="eager", output_attentions=True,
                    )
                    self.model.eval()
                    self.device = device

            wrapper = SimpleModelWrapper(
                self.config.model_name, self.config.device, self.config.dtype,
            )

            result = run_training_pipeline(
                model_wrapper=wrapper,
                trace_output_dir=str(odus_dir / "traces"),
                labels_output_file=str(odus_dir / "labels.json"),
                model_output_file=model_path,
                chunk_size=self.config.odus_chunk_size,
                num_trace_samples=self.config.odus_num_samples,
            )

            # Distill to LUT
            if result.get("status") == "completed":
                lut_result = self._distill_to_lut(model_path, odus_dir)
                result["lut_distillation"] = lut_result

            del wrapper
            gc.collect()
            torch.cuda.empty_cache()

            self.results["odus_training"] = result
            return result

        except Exception as e:
            logger.error(f"ODUS training failed: {e}")
            return self._train_odus_synthetic(model_path)

    def _train_odus_synthetic(self, model_path: str) -> Dict[str, Any]:
        """Train ODUS with synthetic data (no GPU needed)."""
        import torch
        from src.training.pipeline.odus_trainer import (
            ODUSMLP, ODUSTrainer, TeacherLabel,
        )
        from src.promotion.scorer.odus import RuntimeFeatures

        logger.info("  Training ODUS with synthetic calibration data...")

        # Generate synthetic labels
        rng = np.random.RandomState(42)
        labels = []
        for i in range(2000):
            features = RuntimeFeatures()
            features.chunk_position = rng.random()
            features.chunk_recency = rng.random()
            features.query_chunk_similarity = rng.random() * 2 - 1
            features.lexical_overlap = rng.random() * 0.3
            features.distance_to_nearest_anchor = rng.random() * 10
            features.distance_to_promoted = rng.random() * 10
            features.past_promotion_count = rng.randint(0, 5)
            features.past_promotion_success_rate = rng.random()
            features.is_section_boundary = rng.random() > 0.8
            features.is_title_adjacent = rng.random() > 0.9

            # Synthetic utility: correlated with similarity and recency
            utility = (
                0.3 * max(0, features.query_chunk_similarity)
                + 0.2 * features.chunk_recency
                + 0.15 * (1 - features.chunk_position)
                + 0.1 * features.lexical_overlap
                + 0.1 * (1 / (1 + features.distance_to_nearest_anchor))
                + 0.05 * features.past_promotion_success_rate
                + 0.05 * float(features.is_section_boundary)
                + 0.05 * float(features.is_title_adjacent)
                + rng.normal(0, 0.05)
            )
            utility = max(0.0, min(1.0, utility))

            label = TeacherLabel(
                request_id=f"syn_{i}", step=0, chunk_id=i % 100,
                attention_mass=rng.random(), utility_score=utility,
            )
            label.runtime_features = features
            labels.append(label)

        trainer = ODUSTrainer(num_epochs=self.config.odus_epochs, device="cpu")
        train_result = trainer.train(labels)
        trainer.save_model(model_path)

        # Distill to LUT
        odus_dir = Path(model_path).parent
        lut_result = self._distill_to_lut(model_path, odus_dir)

        result = {
            "status": "completed_synthetic",
            "n_labels": len(labels),
            "best_val_loss": train_result["best_val_loss"],
            "model_path": model_path,
            "lut_distillation": lut_result,
        }
        self.results["odus_training"] = result
        return result

    def _distill_to_lut(self, model_path: str, output_dir: Path) -> Dict[str, Any]:
        """Distill trained ODUS MLP to PPU LUT."""
        from src.config import PPUConfig
        from src.hardware.ppu.lut_distill import LUTDistiller

        logger.info("  Distilling ODUS → PPU LUT...")
        config = PPUConfig()
        distiller = LUTDistiller(config)

        try:
            table, report = distiller.distill_from_odus_weights(model_path)
            lut_path = str(output_dir / "ppu_lut.npy")
            np.save(lut_path, table)
            logger.info(f"  LUT distilled: {report.occupied_entries}/{report.num_lut_entries} entries, MSE={report.mse:.4f}")
            return {
                "status": "completed",
                "lut_path": lut_path,
                **report.to_dict(),
            }
        except Exception as e:
            logger.warning(f"  LUT distillation failed: {e}")
            return {"status": "failed", "error": str(e)}

    # ── 2. Benchmark Evaluation ──────────────────────────────────────

    def run_benchmarks(self) -> Dict[str, Any]:
        """Run end-to-end benchmarks across methods and budgets."""
        logger.info("=" * 60)
        logger.info("[2/5] End-to-End Benchmark Evaluation")
        logger.info("=" * 60)

        try:
            import torch
            if not torch.cuda.is_available() and self.config.device == "cuda":
                logger.warning("CUDA not available, skipping real benchmarks")
                self.results["benchmarks"] = {"status": "skipped_no_gpu"}
                return self.results["benchmarks"]

            from src.runners.e2e_eval_runner import (
                ProSEEndToEndRunner, E2ERunConfig,
            )

            all_results = []
            for method in self.config.methods:
                for ratio in self.config.budget_ratios:
                    if method == "full_kv" and ratio != self.config.budget_ratios[0]:
                        continue

                    logger.info(f"  Running {method} @ {ratio:.0%}...")
                    cfg = E2ERunConfig(
                        model_name=self.config.model_name,
                        method=method,
                        budget_ratio=ratio,
                        passkey_lengths=self.config.passkey_lengths,
                        longbench_tasks=self.config.longbench_tasks,
                        samples_per_config=self.config.samples_per_config,
                        output_dir=str(self.out / "benchmarks"),
                    )
                    runner = ProSEEndToEndRunner(cfg)

                    try:
                        passkey_result = runner.evaluate_passkey()
                        all_results.append(passkey_result)
                    except Exception as e:
                        logger.error(f"  Failed {method}@{ratio}: {e}")
                        all_results.append({
                            "method": method, "budget_ratio": ratio, "error": str(e),
                        })
                    finally:
                        del runner
                        gc.collect()
                        torch.cuda.empty_cache()

            result = {"status": "completed", "results": all_results}
            self._save("benchmark_results.json", result)
            self.results["benchmarks"] = result
            return result

        except Exception as e:
            logger.error(f"Benchmark evaluation failed: {e}")
            self.results["benchmarks"] = {"status": "failed", "error": str(e)}
            return self.results["benchmarks"]

    # ── 3. Miss Characterization ─────────────────────────────────────

    def run_miss_characterization(self) -> Dict[str, Any]:
        """Run promotion miss characterization study."""
        logger.info("=" * 60)
        logger.info("[3/5] Promotion Miss Characterization")
        logger.info("=" * 60)

        from src.eval.promotion_miss_characterization import (
            CrossMethodComparison,
        )

        comp = CrossMethodComparison()
        study = comp.run_full_study(
            num_chunks=self.config.miss_num_chunks,
            num_steps=self.config.miss_num_steps,
        )

        self._save("miss_characterization.json", study)
        self.results["miss_characterization"] = study

        # Print key finding
        for method, stats in study["summary"].items():
            logger.info(
                f"  {method}: promotion_miss_share="
                f"{stats['promotion_miss_share_of_total_miss']:.1f}%"
            )

        return study

    # ── 4. Hardware Analysis ─────────────────────────────────────────

    def run_hardware_analysis(self) -> Dict[str, Any]:
        """Run PPU design space exploration and CXL sensitivity."""
        logger.info("=" * 60)
        logger.info("[4/5] Hardware Sensitivity Analysis")
        logger.info("=" * 60)

        from src.hardware.ppu.design_space_explorer import (
            PPUDesignSpaceExplorer,
        )

        explorer = PPUDesignSpaceExplorer()
        result = explorer.run_all()
        self._save("hardware_analysis.json", result)
        self.results["hardware"] = result

        overhead = result["gpu_overhead"]
        logger.info(
            f"  PPU overhead: area={overhead['area_overhead_pct']:.4f}%, "
            f"power={overhead['power_overhead_pct']:.4f}%"
        )
        return result

    # ── 5. Figure & Table Generation ─────────────────────────────────

    def generate_figures(self) -> Dict[str, str]:
        """Generate all HPCA figures and tables."""
        logger.info("=" * 60)
        logger.info("[5/5] Generating Figures & Tables")
        logger.info("=" * 60)

        from src.runners.hpca_figure_generator import HPCAFigureGenerator

        gen = HPCAFigureGenerator(output_dir=str(self.out / "figures"))
        paths = gen.generate_all()

        # Generate LaTeX tables
        latex_path = self._generate_latex_tables()
        paths["latex_tables"] = latex_path

        self.results["figures"] = paths
        return paths

    def _generate_latex_tables(self) -> str:
        """Generate LaTeX-ready tables for the paper."""
        latex_lines = []

        # Table 3: PPU configurations
        latex_lines.append(r"% Table 3: PPU Design Points")
        latex_lines.append(r"\begin{table}[t]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\caption{PPU design points at 7nm. Area and power overhead relative to H100 SXM (814\,mm$^2$, 700\,W).}")
        latex_lines.append(r"\label{tab:ppu-configs}")
        latex_lines.append(r"\small")
        latex_lines.append(r"\begin{tabular}{lrrrrrrr}")
        latex_lines.append(r"\toprule")
        latex_lines.append(r"Config & LUT & Counters & DMA & Area & Power & Freq & Area \\")
        latex_lines.append(r"       & entries &          & depth & (mm$^2$) & (mW) & (GHz) & OH (\%) \\")
        latex_lines.append(r"\midrule")

        try:
            from src.hardware.ppu.design_space_explorer import PPUDesignSpaceExplorer
            explorer = PPUDesignSpaceExplorer()
            for p in explorer.explore_design_space():
                oh = p.area_mm2 / 814.0 * 100
                latex_lines.append(
                    f"{p.label} & {p.lut_entries} & {p.counter_entries} & "
                    f"{p.dma_depth} & {p.area_mm2:.4f} & {p.power_mw:.1f} & "
                    f"{p.achievable_freq_ghz:.2f} & {oh:.4f} \\\\"
                )
        except Exception as e:
            latex_lines.append(f"% Error generating data: {e}")

        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\end{table}")
        latex_lines.append("")

        # Table 4: Area/Power overhead across GPUs
        latex_lines.append(r"% Table 4: KCMC Area/Power Overhead")
        latex_lines.append(r"\begin{table}[t]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\caption{KCMC area and power overhead across GPU targets.}")
        latex_lines.append(r"\label{tab:area-power}")
        latex_lines.append(r"\small")
        latex_lines.append(r"\begin{tabular}{lrrrr}")
        latex_lines.append(r"\toprule")
        latex_lines.append(r"GPU & Die Area & TDP & Area OH & Power OH \\")
        latex_lines.append(r"    & (mm$^2$) & (W) & (\%)    & (\%) \\")
        latex_lines.append(r"\midrule")

        try:
            from src.hardware_model.area_power import KCMCHardwareModel
            gpus = [
                ("H100-SXM", 814.0, 700.0),
                ("A100-80G", 826.0, 400.0),
                ("L40S", 609.0, 350.0),
            ]
            for name, die, tdp in gpus:
                m = KCMCHardwareModel(gpu_die_area_mm2=die, gpu_tdp_w=tdp)
                latex_lines.append(
                    f"{name} & {die:.0f} & {tdp:.0f} & "
                    f"{m.area_overhead_percent():.4f} & "
                    f"{m.power_overhead_percent():.4f} \\\\"
                )
        except Exception as e:
            latex_lines.append(f"% Error: {e}")

        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\end{table}")
        latex_lines.append("")

        # Table: Miss characterization summary
        latex_lines.append(r"% Table: Promotion Miss Breakdown at 10\% Budget")
        latex_lines.append(r"\begin{table}[t]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\caption{Miss category breakdown at 10\% KV budget. Promotion miss dominates across all methods.}")
        latex_lines.append(r"\label{tab:miss-breakdown}")
        latex_lines.append(r"\small")
        latex_lines.append(r"\begin{tabular}{lrrrr}")
        latex_lines.append(r"\toprule")
        latex_lines.append(r"Method & Retention & Promotion & Scoring & Recovered \\")
        latex_lines.append(r"       & Miss (\%) & Miss (\%) & Miss (\%) & (\%) \\")
        latex_lines.append(r"\midrule")

        if "miss_characterization" in self.results:
            fig1 = self.results["miss_characterization"].get("figure_1_stacked_bar", {})
            data = fig1.get("data", {})
            methods = fig1.get("methods", [])
            for i, method in enumerate(methods):
                ret = data.get("retention_miss", [0] * len(methods))[i]
                pro = data.get("promotion_miss", [0] * len(methods))[i]
                sco = data.get("scoring_miss", [0] * len(methods))[i]
                rec = data.get("recovered", [0] * len(methods))[i]
                latex_lines.append(
                    f"{method} & {ret:.1f} & {pro:.1f} & {sco:.1f} & {rec:.1f} \\\\"
                )

        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\end{table}")

        # Write LaTeX file
        latex_path = self.out / "tables.tex"
        with open(latex_path, "w") as f:
            f.write("\n".join(latex_lines))
        logger.info(f"  LaTeX tables saved to {latex_path}")
        return str(latex_path)

    # ── Orchestration ────────────────────────────────────────────────

    def run_all(self) -> Dict[str, Any]:
        """Run the complete HPCA experiment pipeline."""
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("HPCA Experiment Orchestrator — Full Pipeline")
        logger.info("=" * 60)

        if self.config.train_odus:
            self.train_odus()

        if self.config.run_benchmarks:
            self.run_benchmarks()

        if self.config.run_miss_study:
            self.run_miss_characterization()

        self.run_hardware_analysis()

        if self.config.generate_figures:
            self.generate_figures()

        elapsed = time.time() - t0
        self.results["total_time_seconds"] = elapsed
        self._save("orchestrator_results.json", self.results)

        logger.info("=" * 60)
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        logger.info(f"Results saved to {self.out}")
        logger.info("=" * 60)

        return self.results

    def run_figures_only(self) -> Dict[str, Any]:
        """Generate figures and tables without running experiments."""
        self.run_miss_characterization()
        self.run_hardware_analysis()
        return self.generate_figures()

    # ── Helpers ───────────────────────────────────────────────────────

    def _save(self, filename: str, data: Any) -> str:
        path = self.out / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return str(path)


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="HPCA Experiment Orchestrator")
    parser.add_argument("--mode", choices=["all", "figures_only", "train_odus", "benchmarks", "miss_study"],
                        default="all")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="outputs/hpca")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU mode")
    args = parser.parse_args()

    config = OrchestratorConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        device="cpu" if args.no_gpu else args.device,
    )

    if args.mode == "figures_only":
        config.train_odus = False
        config.run_benchmarks = False

    orchestrator = HPCAOrchestrator(config)

    if args.mode == "all":
        orchestrator.run_all()
    elif args.mode == "figures_only":
        orchestrator.run_figures_only()
    elif args.mode == "train_odus":
        orchestrator.train_odus()
    elif args.mode == "benchmarks":
        orchestrator.run_benchmarks()
    elif args.mode == "miss_study":
        orchestrator.run_miss_characterization()


if __name__ == "__main__":
    main()
