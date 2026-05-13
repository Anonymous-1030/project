#!/usr/bin/env python3
"""
Extract ONE real attention trace (1024 tokens) from Qwen2.5-1.5B for workload
characterization. This is intentionally minimal to run quickly on CPU.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from runners.hpca_eval_orchestrator import AttentionTraceExtractor


def main():
    extractor = AttentionTraceExtractor(
        model_path="d:/LLM/models/Qwen2.5-1.5B-Instruct",
        device="cpu",
        dtype="float16",
    )
    print("[TraceExtract] Loading model ...")
    extractor.load()
    print("[TraceExtract] Extracting 1 trace @ 1024 tokens ...")
    traces = extractor.extract_multi_length_traces(
        target_lengths=[1024],
        num_samples=1,
        chunk_size=64,
    )
    out_path = Path("outputs/hpca_fair_hardware/single_real_trace_1024.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(traces, f, indent=2)
    print(f"[TraceExtract] Saved to {out_path}")


if __name__ == "__main__":
    main()
