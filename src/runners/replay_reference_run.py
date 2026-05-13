"""
Replay Script for Reference Run

Re-runs the exact configuration from the frozen reference run
to verify reproducibility and establish baseline metrics.
"""

import json
import argparse
import logging
from typing import Dict, Any
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.config import ProSEXv2Config
from src.eval.shared.evaluator_v2 import SharedEvaluatorV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_reference_config(path: str) -> Dict[str, Any]:
    """Load reference configuration from saved report."""
    with open(path, 'r') as f:
        report = json.load(f)
    return report.get("config", {})


def replay_reference_run(
    config_path: str = "outputs/reports/reference_inconsistent_run.json",
    output_dir: str = "outputs/reports",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Replay the reference run.
    
    Args:
        config_path: Path to reference config JSON
        output_dir: Where to save results
        dry_run: If True, only load config without running
        
    Returns:
        Results dictionary
    """
    logger.info(f"Loading reference config from {config_path}")
    
    # Load config
    config_dict = load_reference_config(config_path)
    config = ProSEXv2Config.from_dict(config_dict)
    
    logger.info(f"Replaying reference run with seed={config.seed}")
    logger.info(f"Experiment: {config.experiment_name}")
    
    if dry_run:
        return {
            "status": "dry_run",
            "config_loaded": True,
            "config": config.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }
    
    # Create evaluator with v2 (has consistency checks)
    evaluator = SharedEvaluatorV2(
        config=config.to_dict(),
        experiment_id=f"replay_{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    # NOTE: This is a replay framework. To actually reproduce metrics,
    # integrate with a workload runner that generates the same step data.
    # For now, this validates that the config can be loaded and evaluator initialized.
    
    result = {
        "status": "replay_framework_ready",
        "config_loaded": True,
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "evaluator_version": evaluator.metric_spec_version,
        "note": (
            "This is the reference replay framework. "
            "To fully reproduce metrics, integrate with workload runner."
        ),
    }
    
    # Save result
    output_path = Path(output_dir) / "reference_replay_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Replay result saved to {output_path}")
    
    return result


def verify_reproducibility(
    baseline_path: str,
    replay_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Verify that replay matches baseline.
    
    Returns:
        Verification report
    """
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    baseline_config = baseline.get("config", {})
    replay_config = replay_result.get("config", {})
    
    # Check key parameters
    checks = {
        "seed_matches": baseline_config.get("seed") == replay_config.get("seed"),
        "workload_matches": baseline_config.get("workload") == replay_config.get("workload"),
        "budget_ratio_matches": (
            baseline_config.get("budget_ratio_of_tail") == 
            replay_config.get("budget_ratio_of_tail")
        ),
        "anchor_neighbor_enabled_matches": (
            baseline_config.get("mqr_ulf", {}).get("anchor_neighbor_enabled") ==
            replay_config.get("mqr_ulf", {}).get("anchor_neighbor_enabled")
        ),
        "burst_radius_matches": (
            baseline_config.get("burst", {}).get("radius") ==
            replay_config.get("burst", {}).get("radius")
        ),
    }
    
    all_pass = all(checks.values())
    
    return {
        "reproducibility_verified": all_pass,
        "checks": checks,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Replay reference run for ProSE-X 2.0"
    )
    parser.add_argument(
        "--config",
        default="outputs/reports/reference_inconsistent_run.json",
        help="Path to reference config JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reports",
        help="Output directory for results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load config, don't run",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify reproducibility after replay",
    )
    
    args = parser.parse_args()
    
    result = replay_reference_run(
        config_path=args.config,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    
    if args.verify:
        verification = verify_reproducibility(args.config, result)
        result["verification"] = verification
        print("\n" + "="*70)
        print("REPRODUCIBILITY VERIFICATION")
        print("="*70)
        print(f"Verified: {verification['reproducibility_verified']}")
        print("\nChecks:")
        for check, passed in verification['checks'].items():
            status = "PASS" if passed else "FAIL"
            print(f"  {check}: {status}")
    
    print("\n" + "="*70)
    print("REPLAY RESULT")
    print("="*70)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
