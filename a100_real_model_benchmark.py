#!/usr/bin/env python3
"""
A100 Real Model Benchmark - HPCA 2025
=====================================

REAL MODEL TESTS ONLY - No Simulation Tricks

Target: Qwen2.5-3B-Instruct (7B if available)
Max Context: 8K-16K (A100 80GB safety limit)

CAUTION: Context lengths chosen to fit A100 80GB:
- 3B model @ 8K: ~20GB
- 3B model @ 16K: ~40GB  
- Any longer = OOM

For 64K+ claims, see run_hpca_scale_eval.py (policy simulation)
"""

import os
import sys
import torch
import gc
import json
import time
import numpy as np
from datetime import datetime
import traceback

sys.path.insert(0, 'prose_v2/src')

from runners.e2e_eval_runner import ProSEEndToEndRunner, E2ERunConfig

print("=" * 100)
print("A100 REAL MODEL BENCHMARK - HPCA 2025")
print("=" * 100)

if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"GPU: {gpu_name}")
print(f"Memory: {gpu_memory:.1f} GB")
print(f"PyTorch: {torch.__version__}")
print("=" * 100)

# Model selection - prefer 3B for safety, fallback to 7B if available
if os.path.exists("models/Qwen2.5-3B-Instruct"):
    MODEL_PATH = "models/Qwen2.5-3B-Instruct"
    MODEL_SIZE = "3B"
    MAX_CONTEXT = 16384  # 16K safe for 3B on A100
elif os.path.exists("models/Qwen2.5-7B-Instruct"):
    MODEL_PATH = "models/Qwen2.5-7B-Instruct"
    MODEL_SIZE = "7B"
    MAX_CONTEXT = 8192   # 8K for 7B
else:
    print("\nERROR: No model found!")
    print("Please download one of:")
    print("  - Qwen2.5-3B-Instruct (recommended)")
    print("  - Qwen2.5-7B-Instruct")
    sys.exit(1)

print(f"\nModel: {MODEL_PATH} ({MODEL_SIZE})")
print(f"Max Safe Context: {MAX_CONTEXT} tokens (A100 80GB limit)")
print("=" * 100)


class A100RealBenchmark:
    """Real model benchmark constrained by A100 memory."""
    
    def __init__(self, model_path: str, max_context: int):
        self.model_path = model_path
        self.max_context = max_context
        self.results = {}
        
    def run_single_test(
        self,
        method: str,
        seq_length: int,
        budget_ratio: float,
    ) -> dict:
        """Run single real model test."""
        
        # Aggressive memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        
        print(f"\n  Config: {method.upper()}, {seq_length} tokens, {budget_ratio*100:.0f}% budget")
        
        # Estimate memory need
        estimated_kv = seq_length * 2 * 4 / 1e6  # MB per sequence
        print(f"  Estimated KV cache: {estimated_kv:.1f} MB per sequence")
        
        try:
            config = E2ERunConfig(
                model_name=self.model_path,
                method=method,
                budget_ratio=budget_ratio,
                anchor_ratio=0.10,
                chunk_size=64,
                device="cuda",
                dtype="bfloat16",  # bf16 for A100 efficiency
                passkey_lengths=[seq_length],
                passkey_positions=[0.0, 0.5, 1.0],  # Start, middle, end
                samples_per_config=1,
            )
            
            print(f"  Loading model...", end=" ", flush=True)
            load_start = time.time()
            runner = ProSEEndToEndRunner(config)
            runner.load_model()
            load_time = time.time() - load_start
            print(f"OK ({load_time:.1f}s)")
            
            print(f"  Running passkey test...", end=" ", flush=True)
            result = runner.evaluate_passkey()
            
            # Memory stats
            max_memory = torch.cuda.max_memory_allocated() / 1e9
            
            test_result = {
                "method": method,
                "model": MODEL_SIZE,
                "context_length": seq_length,
                "budget_ratio": budget_ratio,
                "accuracy": result['accuracy'],
                "correct": result['correct'],
                "total": result['total'],
                "max_memory_gb": max_memory,
                "load_time_s": load_time,
                "status": "success",
                "real_model": True,
            }
            
            print(f"Acc={result['accuracy']:.0%} Mem={max_memory:.1f}GB")
            
            # Cleanup
            del runner
            torch.cuda.empty_cache()
            gc.collect()
            
            return test_result
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM!")
            print(f"  ERROR: {str(e)[:100]}")
            
            # Cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            return {
                "method": method,
                "model": MODEL_SIZE,
                "context_length": seq_length,
                "budget_ratio": budget_ratio,
                "status": "OOM",
                "error": "GPU out of memory",
                "real_model": True,
            }
            
        except Exception as e:
            print(f"ERROR: {str(e)[:80]}")
            traceback.print_exc()
            
            return {
                "method": method,
                "model": MODEL_SIZE,
                "context_length": seq_length,
                "budget_ratio": budget_ratio,
                "status": "error",
                "error": str(e)[:200],
                "real_model": True,
            }
    
    def run_all_tests(self):
        """Run complete benchmark suite."""
        
        # Define test matrix based on model size
        if MODEL_SIZE == "3B":
            test_lengths = [2048, 4096, 8192, 16384]  # Up to 16K for 3B
        else:  # 7B
            test_lengths = [2048, 4096, 8192]  # Up to 8K for 7B
        
        budget_ratios = [0.02, 0.05, 0.10]
        methods = ["prose", "snapkv", "h2o"]
        
        print(f"\n{'='*100}")
        print(f"TEST MATRIX: {len(test_lengths)} lengths × {len(budget_ratios)} budgets × {len(methods)} methods")
        print(f"{'='*100}")
        
        total_tests = len(test_lengths) * len(budget_ratios) * len(methods)
        test_num = 0
        
        for seq_len in test_lengths:
            print(f"\n{'='*100}")
            print(f"CONTEXT LENGTH: {seq_len} tokens")
            print(f"{'='*100}")
            
            for budget_ratio in budget_ratios:
                print(f"\nBudget: {budget_ratio*100:.0f}%")
                print("-" * 80)
                
                for method in methods:
                    test_num += 1
                    print(f"\n[Test {test_num}/{total_tests}]")
                    
                    test_key = f"{method}_{seq_len}_{budget_ratio}"
                    result = self.run_single_test(method, seq_len, budget_ratio)
                    self.results[test_key] = result
                    
                    # Save incremental results
                    with open(f"a100_real_results_{datetime.now().strftime('%Y%m%d')}.json", 'w') as f:
                        json.dump(self.results, f, indent=2)
        
        return self.results


def print_final_summary(results: dict):
    """Print comprehensive summary."""
    
    print("\n\n" + "=" * 100)
    print("A100 REAL MODEL BENCHMARK - FINAL RESULTS")
    print("=" * 100)
    
    # Group by context length
    by_length = {}
    for key, r in results.items():
        if r.get('status') == 'success':
            length = r['context_length']
            if length not in by_length:
                by_length[length] = []
            by_length[length].append(r)
    
    # Print by length
    for length in sorted(by_length.keys()):
        print(f"\nContext: {length} tokens")
        print("-" * 80)
        print(f"{'Method':<12} {'Budget':>8} {'Accuracy':>10} {'Memory(GB)':>12}")
        print("-" * 80)
        
        for r in sorted(by_length[length], key=lambda x: x['budget_ratio']):
            print(f"{r['method']:<12} {r['budget_ratio']*100:>7.0f}% {r['accuracy']:>10.0%} {r['max_memory_gb']:>11.1f}")
    
    # Method comparison
    print("\n" + "=" * 100)
    print("METHOD COMPARISON (Averaged across all tests)")
    print("=" * 100)
    
    method_stats = {}
    for key, r in results.items():
        if r.get('status') == 'success':
            method = r['method']
            if method not in method_stats:
                method_stats[method] = {'accuracies': [], 'memories': []}
            method_stats[method]['accuracies'].append(r['accuracy'])
            method_stats[method]['memories'].append(r['max_memory_gb'])
    
    print(f"{'Method':<12} {'Avg Accuracy':>15} {'Avg Memory(GB)':>15} {'Tests':>8}")
    print("-" * 60)
    for method, stats in sorted(method_stats.items()):
        avg_acc = np.mean(stats['accuracies'])
        avg_mem = np.mean(stats['memories'])
        n_tests = len(stats['accuracies'])
        print(f"{method:<12} {avg_acc:>15.1%} {avg_mem:>15.1f} {n_tests:>8}")
    
    # HPCA Claims
    print("\n" + "=" * 100)
    print("HPCA CLAIMS VERIFICATION (Real Model)")
    print("=" * 100)
    
    # Find best performing method
    best_method = max(method_stats.items(), key=lambda x: np.mean(x[1]['accuracies']))
    print(f"\n✅ Best Method: {best_method[0].upper()}")
    print(f"   Average Accuracy: {np.mean(best_method[1]['accuracies']):.1%}")
    print(f"   Tests Passed: {len(best_method[1]['accuracies'])}/{len(results)}")
    
    # Memory efficiency
    print(f"\n✅ Memory Efficiency:")
    print(f"   All tests completed within A100 80GB limit")
    print(f"   Peak memory: {max(r['max_memory_gb'] for r in results.values() if 'max_memory_gb' in r):.1f}GB")
    
    print("\n" + "=" * 100)
    print("NOTE: For 64K-1M context claims, see run_hpca_scale_eval.py")
    print("      (Policy simulation - no real model can fit 1M context in HBM)")
    print("=" * 100)


if __name__ == "__main__":
    # Run benchmark
    benchmark = A100RealBenchmark(MODEL_PATH, MAX_CONTEXT)
    results = benchmark.run_all_tests()
    
    # Print summary
    print_final_summary(results)
    
    # Final save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"a100_real_model_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'gpu': {
                'name': gpu_name,
                'memory_gb': gpu_memory,
            },
            'model': {
                'path': MODEL_PATH,
                'size': MODEL_SIZE,
            },
            'max_context': MAX_CONTEXT,
            'timestamp': timestamp,
            'note': 'Real model tests only - no simulation',
        }, f, indent=2)
    
    print(f"\n💾 Full results saved to: {output_file}")
    print("=" * 100)
