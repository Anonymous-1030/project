"""
Main runner for ProSE-X 2.0 experiments.

Example usage:
    python -m prosex.src.runners.run_prosex_v2 \
        --config configs/prosex_v2/default.yaml \
        --output outputs/experiments/exp001
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.config import ProSEXv2Config
from src.promotion.pipeline import PromotionPipeline, AblationPipeline
from src.core_types import ChunkMetadata, QueryContext, ChunkTier
from src.eval.metrics.recall_metrics import (
    CandidateMetricsCalculator, ScoringMetricsCalculator, SchedulerMetricsCalculator
)
from src.eval.failure_attribution.attributor import FailureAttributor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> ProSEXv2Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return ProSEXv2Config.from_dict(data)


def create_mock_chunks(
    request_id: str,
    num_chunks: int = 20,
    chunk_size: int = 512,
) -> List[ChunkMetadata]:
    """Create mock chunks for testing."""
    import numpy as np
    
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        
        # Create signature
        signature = np.random.randn(128).astype(np.float32)
        
        chunk = ChunkMetadata(
            chunk_id=f"{request_id}:{start}-{end}",
            request_id=request_id,
            token_start=start,
            token_end=end,
            position_ratio=i / num_chunks,
            num_tokens=chunk_size,
            logical_bytes=chunk_size * 128,
            signature=signature,
            signature_hex=f"sig_{i:04x}",
            tier=ChunkTier.TAIL,
            creation_step=0,
            is_section_boundary=(i % 5 == 0),
            is_title_adjacent=(i < 2),
        )
        chunks.append(chunk)
    
    return chunks


def run_experiment(
    config: ProSEXv2Config,
    output_dir: str,
    num_steps: int = 10,
    gold_chunk_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a complete experiment.
    
    Args:
        config: Experiment configuration
        output_dir: Output directory for logs
        num_steps: Number of decode steps to simulate
        gold_chunk_id: Optional gold chunk for evaluation
        
    Returns:
        Experiment results
    """
    logger.info(f"Starting experiment: {config.experiment_name}")
    
    # Initialize pipeline
    pipeline = PromotionPipeline(config)
    
    # Create mock data
    request_id = "test_request_001"
    all_chunks = create_mock_chunks(request_id, num_chunks=20)
    
    # Split into tiers
    anchor_chunks = all_chunks[:2]  # First 2 as anchors
    tail_chunks = all_chunks[2:]    # Rest as tail
    
    for c in anchor_chunks:
        c.tier = ChunkTier.ANCHOR
    
    # Track promoted
    promoted_chunks = []
    
    # Metrics calculators
    candidate_calc = CandidateMetricsCalculator()
    scoring_calc = ScoringMetricsCalculator()
    scheduler_calc = SchedulerMetricsCalculator()
    attributor = FailureAttributor()
    
    # Results
    step_results = []
    
    for step in range(num_steps):
        logger.debug(f"Step {step}")
        
        # Create query context
        query = QueryContext(
            request_id=request_id,
            step=step,
            query_signature=all_chunks[0].signature,  # Use first chunk's sig
            query_tokens=[100, 200, 300],
            active_anchor_ids=[c.chunk_id for c in anchor_chunks],
        )
        
        # Run pipeline
        result = pipeline.run(
            query=query,
            tail_chunks=tail_chunks,
            anchor_chunks=anchor_chunks,
            promoted_chunks=promoted_chunks,
        )
        
        # Compute metrics
        metrics = {}
        
        if result.ulf_result:
            gold_set = {gold_chunk_id} if gold_chunk_id else None
            metrics["candidate"] = candidate_calc.compute(
                result.ulf_result, gold_set
            ).to_dict()
        
        if result.scorer_result:
            metrics["scoring"] = scoring_calc.compute(
                result.scorer_result, gold_chunk_id
            ).to_dict()
        
        if result.scheduler_result:
            metrics["scheduler"] = scheduler_calc.compute(
                result.scheduler_result, gold_chunk_id
            ).to_dict()
        
        # Store result
        step_results.append({
            "step": step,
            "pipeline_result": result.to_dict(),
            "metrics": metrics,
        })
        
        # Update promoted chunks for next step
        promoted_chunks = [
            all_chunks[cid] for cid in result.sticky_result.promoted_ids
            if cid in [c.chunk_id for c in all_chunks]
        ]
    
    # Aggregate results
    summary = {
        "experiment_name": config.experiment_name,
        "num_steps": num_steps,
        "config": config.to_dict(),
        "step_results": step_results,
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "results.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run ProSE-X 2.0 experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/experiments/default",
        help="Output directory"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of steps to simulate"
    )
    parser.add_argument(
        "--gold-chunk",
        type=str,
        default=None,
        help="Gold chunk ID for evaluation"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run experiment
    results = run_experiment(
        config=config,
        output_dir=args.output,
        num_steps=args.steps,
        gold_chunk_id=args.gold_chunk,
    )
    
    print(f"Experiment complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
