#!/usr/bin/env python3
"""
Analyze previously-collected real attention traces to prove needle-heavy
patterns exist in real LLM inference. Uses existing trace JSONs so no model
loading is required.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from eval.workload_characterizer import WorkloadCharacterizer


def load_traces(path: str):
    with open(path) as f:
        data = json.load(f)
    # The trace files may be a list of trace entries or a dict
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "traces" in data:
        return data["traces"]
    return [data]


def main():
    trace_dir = Path("outputs")
    trace_files = [
        "traces_prose_8192_0.1_20260401.json",
        "traces_h2o_8192_0.1_20260401.json",
        "traces_snapkv_8192_0.1_20260401.json",
    ]

    all_traces = []
    for tf in trace_files:
        p = trace_dir / tf
        if not p.exists():
            print(f"[WARN] {p} not found, skipping")
            continue
        traces = load_traces(str(p))
        print(f"[INFO] Loaded {len(traces)} traces from {tf}")
        all_traces.extend(traces)

    if not all_traces:
        print("[ERROR] No trace files found.")
        return

    # Convert traces to attention arrays
    attn_sequences = []
    for t in all_traces:
        if isinstance(t, dict) and "chunk_attention" in t:
            attn_sequences.append(np.array(t["chunk_attention"]))
        elif isinstance(t, list):
            # Maybe it's a list of per-step attention dicts
            for step in t:
                if isinstance(step, dict) and "chunk_attention" in step:
                    attn_sequences.append(np.array(step["chunk_attention"]))

    print(f"[INFO] Total attention snapshots: {len(attn_sequences)}")

    char = WorkloadCharacterizer(top_k_ratio=0.10)
    report = char.characterize_trace(attn_sequences)

    print("\n" + "=" * 60)
    print("REAL TRACE WORKLOAD CHARACTERIZATION")
    print("=" * 60)
    print(f"Total steps analyzed: {report.total_steps}")
    print(f"Pattern distribution:")
    for k, v in report.pattern_distribution.items():
        print(f"  {k:20s}: {v:6.1%}")
    print(f"Mean Gini:            {report.mean_gini:.3f}")
    print(f"Mean entropy (bits):  {report.mean_entropy_bits:.3f}")
    print(f"Mean sequential bias: {report.mean_sequential_bias:.3f}")
    print(f"Mean top-K drift:     {report.mean_top_k_drift:.3f}")
    print("=" * 60)

    # Save
    out_dir = Path("outputs/hpca_fair_hardware")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "source": "existing_real_traces",
        "num_snapshots": len(attn_sequences),
        "total_steps": report.total_steps,
        "pattern_distribution": report.pattern_distribution,
        "mean_gini": report.mean_gini,
        "mean_entropy_bits": report.mean_entropy_bits,
        "mean_sequential_bias": report.mean_sequential_bias,
        "mean_top_k_drift": report.mean_top_k_drift,
    }
    with open(out_dir / "workload_characterization_real_traces.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Saved to {out_dir / 'workload_characterization_real_traces.json'}")


if __name__ == "__main__":
    main()
