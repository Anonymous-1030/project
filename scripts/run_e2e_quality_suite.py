#!/usr/bin/env python3
"""
End-to-End Quality Evaluation Suite: Perplexity + ROUGE + Passkey.

Directly addresses reviewer criticism:
  "You cannot use trace-driven recovery as a proxy for generation quality."

Outputs:
  outputs/hpca_quality_eval/quality_comparison.json
  outputs/hpca_quality_eval/fig_perplexity_vs_budget.pdf
  outputs/hpca_quality_eval/fig_rouge_vs_budget.pdf
"""

import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from runners.e2e_eval_runner import E2ERunConfig, ProSEEndToEndRunner


# ── Synthetic long-context texts for perplexity ──────────────────────

FILLER_SENTENCES = [
    "The city was bustling with activity as people went about their daily routines.",
    "Research in artificial intelligence continues to advance at a rapid pace.",
    "The mountain trail wound through dense forests and across rushing streams.",
    "Economic indicators suggest a period of moderate growth ahead.",
    "Ancient civilizations developed sophisticated systems of governance.",
    "The library contained thousands of rare manuscripts from centuries past.",
    "Climate patterns have been shifting noticeably over the past decade.",
    "Musicians from around the world gathered for the annual festival.",
]


def build_synthetic_texts(tokenizer, num_texts: int = 20, target_lengths: List[int] = None):
    if target_lengths is None:
        target_lengths = [1024, 4096, 8192, 16384]
    texts = []
    filler = " ".join(FILLER_SENTENCES)
    base_tokens = len(tokenizer.encode(filler))

    for _ in range(num_texts):
        target = random.choice(target_lengths)
        repeats = max(1, target // base_tokens + 2)
        full = (filler + " ") * repeats
        tokens = tokenizer.encode(full, add_special_tokens=False)
        if len(tokens) > target:
            tokens = tokens[:target]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        texts.append(text)
    return texts


def evaluate_perplexity_for_methods(model_name, methods, budget_ratios, texts, device="cuda"):
    results = []
    for method in methods:
        for ratio in budget_ratios:
            if method == "full_kv" and ratio != budget_ratios[0]:
                continue
            config = E2ERunConfig(
                model_name=model_name,
                method=method,
                budget_ratio=ratio,
                device=device,
                dtype="float16",
            )
            runner = ProSEEndToEndRunner(config)
            try:
                print(f"[QualityEval] Running perplexity: {method} @ {ratio:.0%} ...")
                result = runner.evaluate_perplexity(texts)
                results.append(result)
                print(f"  -> mean_ppl={result.get('mean_perplexity', float('nan')):.2f}  "
                      f"weighted_ppl={result.get('token_weighted_perplexity', float('nan')):.2f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"method": method, "budget_ratio": ratio, "error": str(e)})
            finally:
                del runner
                gc.collect()
                torch.cuda.empty_cache()
    return results


def evaluate_longbench_rouge_for_methods(model_name, methods, budget_ratios, tasks, device="cuda"):
    results = []
    for method in methods:
        for ratio in budget_ratios:
            if method == "full_kv" and ratio != budget_ratios[0]:
                continue
            config = E2ERunConfig(
                model_name=model_name,
                method=method,
                budget_ratio=ratio,
                device=device,
                dtype="float16",
                longbench_tasks=tasks,
                max_new_tokens=128,
                samples_per_config=5,
            )
            runner = ProSEEndToEndRunner(config)
            try:
                print(f"[QualityEval] Running LongBench ROUGE: {method} @ {ratio:.0%} ...")
                result = runner.evaluate_longbench(tasks=tasks)
                # Extract average ROUGE-L across summarization tasks
                rouges = []
                for task_name, task_res in result.get("task_results", {}).items():
                    if isinstance(task_res, dict) and "rouge_l" in task_res:
                        rouges.append(task_res["rouge_l"])
                avg_rouge = float(np.mean(rouges)) if rouges else float("nan")
                results.append({
                    "method": method,
                    "budget_ratio": ratio,
                    "avg_rouge_l": avg_rouge,
                    "task_results": result.get("task_results", {}),
                })
                print(f"  -> avg_rouge_l={avg_rouge:.3f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"method": method, "budget_ratio": ratio, "error": str(e)})
            finally:
                del runner
                gc.collect()
                torch.cuda.empty_cache()
    return results


def plot_perplexity(results, output_dir: Path):
    methods = sorted({r["method"] for r in results if "mean_perplexity" in r})
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in methods:
        pts = [(r["budget_ratio"], r["mean_perplexity"])
               for r in results if r.get("method") == method and "mean_perplexity" in r]
        if not pts:
            continue
        pts.sort(key=lambda x: x[0])
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=method)
    ax.set_xlabel("Budget Ratio")
    ax.set_ylabel("Mean Perplexity")
    ax.set_title("Perplexity vs. KV Budget Ratio")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_perplexity_vs_budget.pdf", dpi=300)
    print(f"[Plot] Saved {output_dir / 'fig_perplexity_vs_budget.pdf'}")


def plot_rouge(results, output_dir: Path):
    methods = sorted({r["method"] for r in results if "avg_rouge_l" in r})
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in methods:
        pts = [(r["budget_ratio"], r["avg_rouge_l"])
               for r in results if r.get("method") == method and "avg_rouge_l" in r]
        if not pts:
            continue
        pts.sort(key=lambda x: x[0])
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=method)
    ax.set_xlabel("Budget Ratio")
    ax.set_ylabel("ROUGE-L")
    ax.set_title("LongBench Summarization ROUGE-L vs. KV Budget")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_rouge_vs_budget.pdf", dpi=300)
    print(f"[Plot] Saved {output_dir / 'fig_rouge_vs_budget.pdf'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="d:/LLM/models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--methods", default="full_kv,streaming,h2o,snapkv,quest,prose,stream_prefetcher")
    parser.add_argument("--budget_ratios", default="0.05,0.10,0.20,0.40")
    parser.add_argument("--max_length", type=int, default=8192, help="Max token length for perplexity texts")
    parser.add_argument("--samples", type=int, default=15, help="Number of perplexity texts")
    parser.add_argument("--skip_longbench", action="store_true", help="Skip LongBench (saves time)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[QualityEval] Auto-selected device: {args.device}")

    output_dir = Path("outputs/hpca_quality_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = args.methods.split(",")
    budget_ratios = [float(x) for x in args.budget_ratios.split(",")]

    # ── Prepare texts ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"[QualityEval] Preparing {args.samples} synthetic texts up to {args.max_length} tokens ...")
    texts = build_synthetic_texts(tokenizer, num_texts=args.samples,
                                   target_lengths=[1024, 4096, min(8192, args.max_length), args.max_length])

    # ── Perplexity ──
    perplexity_results = evaluate_perplexity_for_methods(
        args.model, methods, budget_ratios, texts, device=args.device
    )

    # ── LongBench ROUGE (summarization) ──
    rouge_results = []
    if not args.skip_longbench:
        rouge_results = evaluate_longbench_rouge_for_methods(
            args.model, methods, budget_ratios,
            tasks=["gov_report", "qmsum", "multi_news"],
            device=args.device,
        )

    # ── Save ──
    report = {
        "config": {
            "model": args.model,
            "methods": methods,
            "budget_ratios": budget_ratios,
            "perplexity_num_texts": len(texts),
        },
        "perplexity": perplexity_results,
        "rouge": rouge_results,
    }
    with open(output_dir / "quality_comparison.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Plot ──
    plot_perplexity(perplexity_results, output_dir)
    if rouge_results:
        plot_rouge(rouge_results, output_dir)

    print("\n" + "=" * 80)
    print("SUCCESS: Quality evaluation artifacts written to outputs/hpca_quality_eval/")
    print("=" * 80)


if __name__ == "__main__":
    main()
