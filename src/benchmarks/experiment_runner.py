"""
Unified Experiment Runner for ProSE-X 2.0.

Supports running experiments across:
- Multiple models (Qwen, Llama, Mistral)
- Multiple benchmarks (Passkey, LongBench, RULER)
- Multiple methods (ProSE, H2O, StreamingLLM, SnapKV, Full KV)
- Multiple configurations (budget ratios, chunk sizes, etc.)
"""

import torch
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.attention.attention_runner import RealAttentionRunner, RetentionMode
from src.baselines.h2o import H2ORunner
from src.baselines.streaming_llm import StreamingLLMRunner
from src.baselines.snapkv import SnapKVRunner

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    model_name: str
    method: str  # "prose", "h2o", "streaming_llm", "snapkv", "full"
    
    # Method-specific params
    budget_ratio: float = 0.1
    promote_ratio: float = 0.02
    
    # Benchmark
    benchmark: str = "passkey"
    
    # Misc
    device: str = "cuda"
    max_seq_len: int = 32768
    seed: int = 42


@dataclass
class ExperimentResult:
    """Result from a single experiment."""
    config: ExperimentConfig
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "config": asdict(self.config),
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
        }


class ModelRegistry:
    """
    Registry of supported models.
    
    Maps friendly names to HuggingFace model paths.
    """
    
    MODELS = {
        # Qwen 2.5 series
        "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
        
        # Llama 3 series
        "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        
        # Mistral series
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "mistral-7b-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    }
    
    @classmethod
    def get_model_path(cls, name: str) -> str:
        """Get HuggingFace path for model name."""
        if name in cls.MODELS:
            return cls.MODELS[name]
        # Assume it's already a path
        return name
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List available model names."""
        return list(cls.MODELS.keys())


class ExperimentRunner:
    """
    Main experiment runner.
    
    Handles model loading, benchmark setup, and result collection.
    """
    
    def __init__(self, output_dir: str = "outputs/experiments"):
        """
        Initialize runner.
        
        Args:
            output_dir: Where to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[ExperimentResult] = []
        self.current_model = None
        self.current_model_name = None
    
    def load_model(self, model_name: str, device: str = "cuda"):
        """Load a model by name."""
        from prose_stage1.prose.models.model_wrapper import ModelWrapper
        
        model_path = ModelRegistry.get_model_path(model_name)
        logger.info(f"Loading model: {model_name} ({model_path})")
        
        self.current_model = ModelWrapper(
            model_name=model_path,
            device=device,
        )
        self.current_model_name = model_name
    
    def create_runner(self, config: ExperimentConfig):
        """Create method-specific runner."""
        if config.method == "prose":
            return RealAttentionRunner(
                self.current_model,
                chunk_size=512,
            )
        elif config.method == "h2o":
            return H2ORunner(
                self.current_model,
                hh_ratio=config.budget_ratio * 0.5,
                recent_ratio=config.budget_ratio * 0.5,
            )
        elif config.method == "streaming_llm":
            window_size = int(config.max_seq_len * config.budget_ratio)
            return StreamingLLMRunner(
                self.current_model,
                num_sink_tokens=4,
                window_size=window_size,
            )
        elif config.method == "snapkv":
            return SnapKVRunner(
                self.current_model,
                observation_tokens=64,
                retention_ratio=config.budget_ratio,
            )
        elif config.method == "full":
            # Full KV - no compression
            return RealAttentionRunner(
                self.current_model,
                chunk_size=512,
            )
        else:
            raise ValueError(f"Unknown method: {config.method}")
    
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a single experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            ExperimentResult
        """
        import time
        
        start_time = time.time()
        
        # Load model if needed
        if self.current_model_name != config.model_name:
            self.load_model(config.model_name, config.device)
        
        # Create runner
        runner = self.create_runner(config)
        
        # Run benchmark
        if config.benchmark == "passkey":
            from src.benchmarks.passkey import PasskeyBenchmark

            benchmark = PasskeyBenchmark(
                tokenizer=self.current_model.tokenizer,
                context_lengths=[4096, 16384, 32768],
                num_samples_per_config=5,
            )
            examples = benchmark.generate_dataset()
            metrics = benchmark.evaluate(runner, examples)

        elif config.benchmark == "longbench":
            from src.benchmarks.longbench import LongBenchBenchmark

            benchmark = LongBenchBenchmark(
                tokenizer=self.current_model.tokenizer,
                tasks=config.longbench_tasks if hasattr(config, "longbench_tasks") else None,
                max_samples_per_task=config.max_samples if hasattr(config, "max_samples") else 50,
            )
            metrics = benchmark.evaluate(runner)

        elif config.benchmark == "ruler":
            from src.benchmarks.ruler import RULERBenchmark

            benchmark = RULERBenchmark(
                tokenizer=self.current_model.tokenizer,
                context_lengths=[4096, 8192, 16384, 32768, 65536],
                num_samples_per_config=config.num_samples if hasattr(config, "num_samples") else 10,
            )
            examples = benchmark.generate_dataset()
            metrics = benchmark.evaluate(runner, examples)

        else:
            raise ValueError(f"Unknown benchmark: {config.benchmark}")
        
        duration = time.time() - start_time
        
        result = ExperimentResult(
            config=config,
            metrics=metrics,
            duration_seconds=duration,
        )
        
        self.results.append(result)
        return result
    
    def run_sweep(
        self,
        models: List[str],
        methods: List[str],
        budget_ratios: List[float],
        benchmark: str = "passkey",
    ) -> List[ExperimentResult]:
        """
        Run a parameter sweep.
        
        Args:
            models: List of model names
            methods: List of methods to compare
            budget_ratios: List of budget ratios to test
            benchmark: Benchmark to use
            
        Returns:
            List of results
        """
        configs = []
        for model in models:
            for method in methods:
                for budget in budget_ratios:
                    configs.append(ExperimentConfig(
                        model_name=model,
                        method=method,
                        budget_ratio=budget,
                        benchmark=benchmark,
                    ))
        
        logger.info(f"Running sweep: {len(configs)} experiments")
        
        for i, config in enumerate(configs):
            logger.info(f"Experiment {i+1}/{len(configs)}: {config}")
            try:
                self.run_experiment(config)
            except Exception as e:
                logger.error(f"Experiment failed: {e}")
        
        return self.results
    
    def save_results(self, filename: Optional[str] = None):
        """Save results to file."""
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        data = [r.to_dict() for r in self.results]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.results)} results to {filepath}")
        return filepath
    
    def export_csv(self, filename: str = "results.csv"):
        """Export results to CSV for analysis."""
        rows = []
        for r in self.results:
            row = {
                "model": r.config.model_name,
                "method": r.config.method,
                "budget_ratio": r.config.budget_ratio,
                "benchmark": r.config.benchmark,
                "duration": r.duration_seconds,
            }
            # Flatten metrics
            for k, v in r.metrics.items():
                if isinstance(v, dict):
                    for subk, subv in v.items():
                        row[f"{k}_{subk}"] = subv
                else:
                    row[k] = v
            rows.append(row)
        
        df = pd.DataFrame(rows)
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        
        logger.info(f"Exported {len(rows)} rows to {filepath}")
        return filepath


def run_full_comparison(
    models: List[str] = ["qwen2.5-1.5b"],
    budget_ratios: List[float] = [0.05, 0.1, 0.2, 0.3],
    benchmarks: List[str] = ["passkey", "longbench", "ruler"],
    output_dir: str = "outputs/experiments",
):
    """
    Run full comparison of all methods across all benchmarks.

    This is the main entry point for running comprehensive experiments.
    """
    runner = ExperimentRunner(output_dir)

    methods = ["full", "h2o", "streaming_llm", "snapkv", "prose"]

    for benchmark in benchmarks:
        results = runner.run_sweep(
            models=models,
            methods=methods,
            budget_ratios=budget_ratios,
            benchmark=benchmark,
        )

    runner.save_results()
    runner.export_csv()

    # Print summary
    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)

    for result in runner.results:
        config = result.config
        metrics = result.metrics
        print(f"\n{config.model_name} | {config.method} | budget={config.budget_ratio} | {config.benchmark}")
        if "accuracy" in metrics:
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
        if "overall" in metrics:
            print(f"  Overall: {metrics['overall']:.4f}")
        print(f"  Duration: {result.duration_seconds:.1f}s")

    return runner.results


if __name__ == "__main__":
    # Run comparison
    results = run_full_comparison()
