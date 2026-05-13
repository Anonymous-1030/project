"""
Ablation runner for ProSE-X 2.0.

Runs a sweep of ablation experiments and generates comparison report.

Example usage:
    python -m prosex.src.runners.run_ablation \
        --base-config configs/prosex_v2/default.yaml \
        --ablation-dir configs/prosex_v2/ablations \
        --output outputs/ablations
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
import copy

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.runners.run_prosex_v2 import run_experiment, load_config
from src.config import ProSEXv2Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_dicts(base: Dict, override: Dict) -> Dict:
    """Deep merge override into base."""
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def run_ablation_sweep(
    base_config_path: str,
    ablation_configs: List[Path],
    output_dir: str,
    num_steps: int = 10,
    gold_chunk_id: str = None,
) -> Dict[str, Any]:
    """
    Run ablation sweep.
    
    Args:
        base_config_path: Path to base config
        ablation_configs: List of ablation config paths
        output_dir: Output directory
        num_steps: Steps per experiment
        gold_chunk_id: Gold chunk for evaluation
        
    Returns:
        Sweep results
    """
    logger.info("Starting ablation sweep")
    
    # Load base config
    base_config = load_config(base_config_path)
    
    results = {}
    
    # Run baseline
    logger.info("Running baseline (full system)")
    baseline_output = Path(output_dir) / "baseline"
    results["baseline"] = run_experiment(
        config=base_config,
        output_dir=str(baseline_output),
        num_steps=num_steps,
        gold_chunk_id=gold_chunk_id,
    )
    
    # Run each ablation
    for ablation_path in ablation_configs:
        name = ablation_path.stem
        logger.info(f"Running ablation: {name}")
        
        # Load ablation overrides
        with open(ablation_path, 'r') as f:
            ablation_overrides = yaml.safe_load(f)
        
        # Merge with base
        merged_dict = merge_dicts(base_config.to_dict(), ablation_overrides)
        ablation_config = ProSEXv2Config.from_dict(merged_dict)
        
        # Run experiment
        ablation_output = Path(output_dir) / name
        results[name] = run_experiment(
            config=ablation_config,
            output_dir=str(ablation_output),
            num_steps=num_steps,
            gold_chunk_id=gold_chunk_id,
        )
    
    # Generate comparison report
    report = generate_comparison_report(results)
    
    # Save report
    report_path = Path(output_dir) / "ablation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Ablation sweep complete. Report saved to {report_path}")
    
    return results


def generate_comparison_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comparison report from ablation results.
    
    Args:
        results: Dictionary mapping experiment name to results
        
    Returns:
        Comparison report
    """
    report = {
        "experiments": list(results.keys()),
        "comparisons": {},
    }
    
    # Extract key metrics from each experiment
    for name, result in results.items():
        step_results = result.get("step_results", [])
        
        # Aggregate metrics
        avg_candidate_recall = 0.0
        avg_scheduler_utilization = 0.0
        
        for step in step_results:
            metrics = step.get("metrics", {})
            
            if "candidate" in metrics:
                recall_at_10 = metrics["candidate"].get("candidate_recall_at_k", {}).get("10", 0)
                avg_candidate_recall += recall_at_10
            
            if "scheduler" in metrics:
                util = metrics["scheduler"].get("utilization", 0)
                avg_scheduler_utilization += util
        
        n_steps = len(step_results)
        if n_steps > 0:
            avg_candidate_recall /= n_steps
            avg_scheduler_utilization /= n_steps
        
        report["comparisons"][name] = {
            "avg_candidate_recall@10": avg_candidate_recall,
            "avg_scheduler_utilization": avg_scheduler_utilization,
        }
    
    # Compute relative to baseline
    if "baseline" in report["comparisons"]:
        baseline_metrics = report["comparisons"]["baseline"]
        
        for name, metrics in report["comparisons"].items():
            if name == "baseline":
                continue
            
            rel_recall = (
                metrics["avg_candidate_recall@10"] / baseline_metrics["avg_candidate_recall@10"]
                if baseline_metrics["avg_candidate_recall@10"] > 0 else 0
            )
            
            metrics["relative_recall"] = rel_recall
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Run ProSE-X 2.0 ablation sweep")
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/prosex_v2/default.yaml",
        help="Base configuration"
    )
    parser.add_argument(
        "--ablation-dir",
        type=str,
        default=None,
        help="Directory containing ablation configs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ablations",
        help="Output directory"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Steps per experiment"
    )
    
    args = parser.parse_args()
    
    # Collect ablation configs
    ablation_configs = []
    if args.ablation_dir:
        ablation_dir = Path(args.ablation_dir)
        if ablation_dir.exists():
            ablation_configs = sorted(ablation_dir.glob("*.yaml"))
    
    # Also check default ablation configs
    default_ablation_dir = Path("configs/prosex_v2")
    if default_ablation_dir.exists():
        for path in default_ablation_dir.glob("ablation_*.yaml"):
            if path not in ablation_configs:
                ablation_configs.append(path)
    
    logger.info(f"Found {len(ablation_configs)} ablation configs")
    
    # Run sweep
    results = run_ablation_sweep(
        base_config_path=args.base_config,
        ablation_configs=ablation_configs,
        output_dir=args.output,
        num_steps=args.steps,
    )
    
    print(f"Ablation sweep complete. Results in {args.output}")


if __name__ == "__main__":
    main()
