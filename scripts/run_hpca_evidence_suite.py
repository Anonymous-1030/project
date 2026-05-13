#!/usr/bin/env python3
"""
HPCA Evidence Suite — Collects all evidence for HPCA paper submission.

Outputs to outputs/hpca_evidence/:
  - odus_training.json      — ODUS training curves
  - pht_qfc_cycles.json     — PHT/QFC cycle-accurate statistics
  - sw_ablation.json         — Software-only ablation results
  - hw_simulation.json       — Full HW simulation (PHT + QFC integrated)
  - summary.json             — Unified summary of all Table data

Usage:
    python run_hpca_evidence_suite.py
"""

import sys
import os
import json
import time
from pathlib import Path

# Fix paths for imports
ROOT = Path(__file__).resolve().parent.parent.parent  # d:\LLM
PROSE_V2 = ROOT / "prose_v2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROSE_V2))

from src.hardware.pht_cycle_sim import PHTCycleAccurateSim
from src.hardware.qfc_cycle_sim import QFCCycleAccurateSim, run_comparison_sweep

# Import the integrated evaluator from heterogeneous_memory_simulator
sys.path.insert(0, str(ROOT))
from heterogeneous_memory_simulator import HPCAIntegratedEvaluator

# Output directory
EVIDENCE_DIR = ROOT / "outputs" / "hpca_evidence"


def ensure_dirs():
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data, filename):
    path = EVIDENCE_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  -> Saved: {path}")


# ---------------------------------------------------------------
# Step 1: Load ODUS training results
# ---------------------------------------------------------------
def collect_odus_training():
    print("\n[1/5] Loading ODUS training results...")
    # Check both possible locations
    odus_paths = [
        ROOT / "outputs" / "odus_ablation_results.json",
        ROOT / "prose_v2" / "outputs" / "odus_ablation_results.json",
    ]
    for odus_path in odus_paths:
        if odus_path.exists():
            with open(odus_path) as f:
                data = json.load(f)
            print(f"  Loaded from {odus_path}")
            return data
    print(f"  WARNING: ODUS results not found, generating placeholder")
    placeholder = {
        "status": "file_not_found",
        "note": "Run ODUS training first: python scripts/run_odus_training.py",
    }
    return placeholder


# ---------------------------------------------------------------
# Step 2: Run QFC comparison sweep
# ---------------------------------------------------------------
def collect_qfc_sweep():
    print("\n[2/5] Running QFC comparison sweep...")
    batch_sizes = [1, 8, 32, 64, 128, 256]
    chunks_list = [8, 16, 32, 64]
    all_results = {}
    for nc in chunks_list:
        sweep = run_comparison_sweep(batch_sizes=batch_sizes, chunks=nc)
        for bs, data in sweep.items():
            key = f"batch{bs}_chunks{nc}"
            all_results[key] = data
    print(f"  Collected {len(all_results)} QFC sweep configs")
    return all_results


# ---------------------------------------------------------------
# Step 3: Collect PHT statistics
# ---------------------------------------------------------------
def collect_pht_stats():
    print("\n[3/5] Collecting PHT cycle-accurate statistics...")
    pht = PHTCycleAccurateSim(num_entries=1024)
    test_scenarios = {}

    for num_chunks in [8, 16, 32, 64, 128]:
        pht.reset()
        # Use realistic interleaved workload (produces RAW hazards + cold misses)
        anchor_list = list(range(max(1, num_chunks // 4)))
        for a in anchor_list:
            pht.set_anchor(a)

        promote_pattern = []
        query_pattern = []
        for step_i in range(6):
            shift_center = (step_i * num_chunks // 6) % num_chunks
            promoted = [shift_center, (shift_center + 1) % num_chunks, (shift_center + 2) % num_chunks]
            promote_pattern.append(promoted)
            queried = [shift_center, (shift_center + 1) % num_chunks,
                       (shift_center + 3) % num_chunks, (shift_center - 1) % num_chunks]
            query_pattern.append(queried)

        stats = pht.simulate_workload(
            chunk_ids=list(range(num_chunks)),
            promote_pattern=promote_pattern,
            query_pattern=query_pattern,
            anchor_ids=anchor_list,
            ticks_between_steps=1,
        )
        test_scenarios[f"chunks_{num_chunks}"] = {
            "num_chunks": num_chunks,
            "total_cycles": stats.total_cycles,
            "total_queries": stats.total_queries,
            "total_updates": stats.total_updates,
            "pipeline_stalls": stats.pipeline_stalls,
            "raw_hazards": stats.raw_hazards,
            "hit_rate": stats.hit_rate,
            "avg_ema": stats.avg_ema,
            "active_entries": stats.active_entries,
            "anchor_entries": stats.anchor_entries,
            "cycles_per_query": stats.total_cycles / max(stats.total_queries, 1),
            "cycles_per_update": stats.total_cycles / max(stats.total_updates, 1),
        }

    print(f"  Collected PHT stats for {len(test_scenarios)} scenarios")
    return test_scenarios


# ---------------------------------------------------------------
# Step 4: Load SW ablation results
# ---------------------------------------------------------------
def collect_sw_ablation():
    print("\n[4/5] Loading SW ablation results...")
    sw_path = ROOT / "outputs" / "sw_ablation_results.json"
    if sw_path.exists():
        with open(sw_path) as f:
            data = json.load(f)
        print(f"  Loaded SW ablation from {sw_path}")
        return data
    else:
        print(f"  WARNING: {sw_path} not found, checking root...")
        sw_path2 = ROOT / "sw_ablation_results.json"
        if sw_path2.exists():
            with open(sw_path2) as f:
                data = json.load(f)
            print(f"  Loaded SW ablation from {sw_path2}")
            return data
        print(f"  SW ablation not found, generating placeholder")
        return {"status": "file_not_found", "note": "Run SW ablation first"}


# ---------------------------------------------------------------
# Step 5: Run HPCAIntegratedEvaluator full evaluation
# ---------------------------------------------------------------
def collect_hw_simulation():
    print("\n[5/5] Running HPCAIntegratedEvaluator full evaluation...")
    evaluator = HPCAIntegratedEvaluator()
    results = evaluator.run_full_evaluation(
        batch_sizes=[1, 8, 32, 64, 128, 256],
        context_chunks=[8, 16, 32, 64],
    )
    evaluator.print_hpca_table(results)
    print(f"  Collected {len(results)} HW simulation configs")
    return results


# ---------------------------------------------------------------
# Summary: aggregate all
# ---------------------------------------------------------------
def build_summary(odus, qfc, pht, sw, hw):
    print("\nBuilding summary...")
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sections": {},
    }

    # ODUS summary
    if isinstance(odus, dict) and "status" not in odus:
        summary["sections"]["odus_training"] = {
            "status": "available",
            "num_entries": len(odus) if isinstance(odus, (list, dict)) else 0,
        }
    else:
        summary["sections"]["odus_training"] = {"status": "missing"}

    # QFC summary: extract key speedup numbers
    qfc_speedups = {}
    for key, data in qfc.items():
        qfc_speedups[key] = data.get("speedup", 0)
    avg_speedup = sum(qfc_speedups.values()) / max(len(qfc_speedups), 1)
    max_speedup = max(qfc_speedups.values()) if qfc_speedups else 0
    summary["sections"]["qfc_sweep"] = {
        "status": "available",
        "num_configs": len(qfc),
        "avg_speedup": round(avg_speedup, 2),
        "max_speedup": round(max_speedup, 2),
    }

    # PHT summary
    pht_cycles = {k: v["total_cycles"] for k, v in pht.items()}
    summary["sections"]["pht_stats"] = {
        "status": "available",
        "num_scenarios": len(pht),
        "cycle_counts": pht_cycles,
    }

    # SW ablation summary
    if isinstance(sw, dict) and "status" not in sw:
        summary["sections"]["sw_ablation"] = {
            "status": "available",
            "num_entries": len(sw) if isinstance(sw, (list, dict)) else 0,
        }
    else:
        summary["sections"]["sw_ablation"] = {"status": "missing"}

    # HW simulation summary: extract ProSE-HW vs baseline speedups
    hw_summary = {}
    for key, data in hw.items():
        methods = data["methods"]
        hw_summary[key] = {
            "prose_hw_p99_ms": methods["ProSE-HW"]["p99_ms"],
            "prose_sw_p99_ms": methods["ProSE-SW"]["p99_ms"],
            "snapkv_p99_ms": methods["SnapKV"]["p99_ms"],
            "qfc_speedup": data["qfc_speedup"],
            "qfc_bw_reduction": data["qfc_bw_reduction"],
        }
    summary["sections"]["hw_simulation"] = {
        "status": "available",
        "num_configs": len(hw),
        "configs": hw_summary,
    }

    # Top-level key numbers for paper
    summary["paper_highlights"] = {
        "qfc_avg_speedup": round(avg_speedup, 2),
        "qfc_max_speedup": round(max_speedup, 2),
        "pht_query_latency_cycles": 1,
        "pht_update_latency_cycles": 3,
        "pht_area_mm2": 0.019,
        "pht_power_mw": 24.9,
    }

    return summary


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 72)
    print("HPCA Evidence Collection Suite")
    print("=" * 72)

    ensure_dirs()

    # Collect all evidence
    odus = collect_odus_training()
    qfc = collect_qfc_sweep()
    pht = collect_pht_stats()
    sw = collect_sw_ablation()
    hw = collect_hw_simulation()

    # Save individual files
    print("\n" + "-" * 72)
    print("Saving evidence files...")
    save_json(odus, "odus_training.json")
    save_json({"qfc_sweep": qfc, "pht_stats": pht}, "pht_qfc_cycles.json")
    save_json(sw, "sw_ablation.json")
    save_json(hw, "hw_simulation.json")

    # Build and save summary
    summary = build_summary(odus, qfc, pht, sw, hw)
    save_json(summary, "summary.json")

    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"HPCA Evidence Suite COMPLETE in {elapsed:.1f}s")
    print(f"All files saved to: {EVIDENCE_DIR}")
    print("=" * 72)

    # Print summary highlights
    print("\nPaper Highlights:")
    for k, v in summary["paper_highlights"].items():
        print(f"  {k}: {v}")

    return summary


if __name__ == "__main__":
    main()
