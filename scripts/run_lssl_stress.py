"""
LSSL stress-test runner (analytical model).

The PROSE query-sketch ablation (v2 results) measured that Recovery@K is
determined by the SCORER'S SKETCH STATE. From that data:

    fresh  sketch  : Recovery@K ≈ 0.820  (sd ≈ 0.06)
    aging  sketch  : Recovery@K ≈ 0.700  (sd ≈ 0.08, one-step extrap.)
    stale  sketch  : Recovery@K ≈ 0.420  (sd ≈ 0.09, BELOW no-sketch)
    no-sketch base : Recovery@K ≈ 0.540  (sd ≈ 0.05, endpoint-only)

Under failure injection, each policy transitions between states:

    lssl     : fresh when clean, aging for ≤ τ=2 steps after a miss,
               otherwise **hard-floored to no-sketch**.
    no-lssl  : fresh when clean, otherwise keeps re-using the stale
               cached sketch. No fallback.
    vanilla  : fresh when clean, drops the sketch the instant it misses
               (no aging, goes straight to no-sketch). No extrapolation.

This runner plays out 200 steps × 3 seeds × 5 scenarios × 3 intensities ×
3 policies and reports mean / p5 / steps-below-baseline.

The per-state distributions are calibrated from the v2 data so that
"fresh everywhere" reproduces the reported 0.82 mean recovery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


OUT = Path("d:/LLM/outputs/lssl_stress")
OUT.mkdir(parents=True, exist_ok=True)


# ─── Calibrated per-state Recovery@K distributions ──────────────────
STATE_MEAN = dict(fresh=0.820, aging=0.700, stale=0.420, nosketch=0.540)
STATE_SD   = dict(fresh=0.060, aging=0.080, stale=0.090, nosketch=0.050)
NO_SKETCH_BASELINE = STATE_MEAN["nosketch"]

SCENARIOS = [
    ("speculative",   [0.05, 0.15, 0.30], "Speculative divergence (per step)"),
    ("batched",       [0.04, 0.12, 0.22], "Batched refresh-miss (per step)"),
    ("preempt",       [0.05, 0.15, 0.28], "Preempt events (per step)"),
    ("kv_compress",   [0.04, 0.10, 0.20], "KV-compression events (per step)"),
    ("multitenant",   [8,    4,    2   ], "Starvation period (smaller = harder)"),
]


def sample_recovery(state: str, rng: np.random.Generator) -> float:
    mu = STATE_MEAN[state]
    sd = STATE_SD[state]
    v = rng.normal(mu, sd)
    return float(np.clip(v, 0.0, 1.0))


# ─── Policies: map (clean/miss) event stream → per-step state ───────

def policy_states_lssl(miss_mask: np.ndarray, tau: int = 2) -> List[str]:
    """LSSL: Fresh, Aging (for ≤τ steps after a miss), else Stale->no-sketch."""
    states = []
    age_since_fresh = 0     # steps since last confirmed-fresh sketch
    have_history = False
    for m in miss_mask:
        if not m:                    # clean step: install fresh sketch
            age_since_fresh = 0
            have_history = True
            states.append("fresh")
        else:
            age_since_fresh += 1
            if not have_history or age_since_fresh > tau:
                # LSSL's hard floor: fall back to summary-only scorer
                states.append("nosketch")
            else:
                # Aging: 1-step Kalman extrapolation from last 2 snapshots
                states.append("aging")
    return states


def policy_states_no_lssl(miss_mask: np.ndarray) -> List[str]:
    """no-LSSL: on clean steps use fresh sketch; on miss steps KEEP using
    the stale cached sketch indefinitely."""
    states = []
    have_history = False
    for m in miss_mask:
        if not m:
            have_history = True
            states.append("fresh")
        else:
            # Using the stale frozen sketch -- harmful
            states.append("stale" if have_history else "nosketch")
    return states


def policy_states_vanilla(miss_mask: np.ndarray) -> List[str]:
    """Vanilla: on miss, IMMEDIATELY drop to no-sketch (no aging,
    no floor guarantee -- mirrors a reactive fallback that throws away
    the sketch the instant coherence flags it)."""
    states = []
    for m in miss_mask:
        states.append("nosketch" if m else "fresh")
    return states


# ─── Miss-mask generator for each failure scenario ───────────────────

def miss_mask_for(scenario: str, intensity, num_steps: int,
                  rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(num_steps, dtype=bool)
    if scenario == "speculative":
        mask = rng.random(num_steps) < float(intensity)
    elif scenario == "batched":
        # Batched mode: misses come in bursts of 2-3 consecutive steps,
        # modelling shared-bandwidth contention during a batch flush.
        p = float(intensity)
        i = 0
        while i < num_steps:
            if rng.random() < p:
                burst = rng.integers(2, 4)
                for k in range(burst):
                    if i + k < num_steps:
                        mask[i + k] = True
                i += burst + rng.integers(2, 5)
            else:
                i += 1
    elif scenario == "preempt":
        # Preempts are instantaneous but tend to be followed by 1 step
        # of refresh lag.
        for i in range(num_steps):
            if rng.random() < float(intensity):
                mask[i] = True
                if i + 1 < num_steps:
                    mask[i + 1] = True
    elif scenario == "kv_compress":
        # Periodic compression: probabilistic epoch start, lag for the
        # next 2 steps while the sketch is rebuilt.
        i = 0
        while i < num_steps:
            if rng.random() < float(intensity):
                for k in range(3):
                    if i + k < num_steps:
                        mask[i + k] = True
                i += 3
            else:
                i += 1
    elif scenario == "multitenant":
        # Deterministic starvation: every N steps the tenant's refresh
        # budget is revoked, so that step's sketch is stale.
        period = int(intensity)
        for i in range(num_steps):
            if i > 0 and i % period == 0:
                mask[i] = True
    return mask


# ─── Main simulation ─────────────────────────────────────────────────

def run_scenario_cell(scenario, intensity, num_steps, seeds) -> Dict:
    cell = {}
    for policy_name, states_fn in [
        ("vanilla", policy_states_vanilla),
        ("no-lssl", policy_states_no_lssl),
        ("lssl",    policy_states_lssl),
    ]:
        runs = []
        for s in seeds:
            rng = np.random.default_rng(s)
            mask = miss_mask_for(scenario, intensity, num_steps, rng)
            states = states_fn(mask)
            per_step = np.array(
                [sample_recovery(st, rng) for st in states],
                dtype=np.float64,
            )
            runs.append(dict(
                mean_recovery=float(per_step.mean()),
                p5_recovery  =float(np.percentile(per_step, 5)),
                p50_recovery =float(np.percentile(per_step, 50)),
                min_recovery =float(per_step.min()),
                steps_below_baseline=int(
                    (per_step < NO_SKETCH_BASELINE - 0.01).sum()),
                fresh_frac  =float(sum(1 for st in states if st=="fresh"))/num_steps,
                aging_frac  =float(sum(1 for st in states if st=="aging"))/num_steps,
                stale_frac  =float(sum(1 for st in states if st=="stale"))/num_steps,
                nosketch_frac=float(sum(1 for st in states if st=="nosketch"))/num_steps,
                miss_rate   =float(mask.mean()),
                per_step    =per_step.tolist(),
            ))
        cell[policy_name] = dict(
            mean_recovery=float(np.mean([r["mean_recovery"] for r in runs])),
            p5_recovery  =float(np.mean([r["p5_recovery"]   for r in runs])),
            p50_recovery =float(np.mean([r["p50_recovery"]  for r in runs])),
            min_recovery =float(np.mean([r["min_recovery"]  for r in runs])),
            steps_below_baseline=float(np.mean(
                [r["steps_below_baseline"] for r in runs])),
            fresh_frac  =float(np.mean([r["fresh_frac"]   for r in runs])),
            aging_frac  =float(np.mean([r["aging_frac"]   for r in runs])),
            stale_frac  =float(np.mean([r["stale_frac"]   for r in runs])),
            nosketch_frac=float(np.mean([r["nosketch_frac"] for r in runs])),
            miss_rate   =float(np.mean([r["miss_rate"]    for r in runs])),
            per_seed=runs,
        )
    return cell


def main():
    num_steps = 200
    seeds = [17, 23, 41, 59, 73]
    results = dict(
        meta=dict(
            num_steps=num_steps,
            seeds=seeds,
            state_mean=STATE_MEAN,
            state_sd=STATE_SD,
            no_sketch_baseline=NO_SKETCH_BASELINE,
        ),
        scenarios={},
    )
    for scenario, intensities, label in SCENARIOS:
        results["scenarios"][scenario] = dict(
            label=label,
            intensities=intensities,
            cells={},
        )
        for intensity in intensities:
            cell = run_scenario_cell(scenario, intensity, num_steps, seeds)
            results["scenarios"][scenario]["cells"][f"{intensity}"] = cell
            print(
                f"[{scenario:<12s} @ {intensity}]  "
                f"lssl: mean={cell['lssl']['mean_recovery']:.3f} "
                f"p5={cell['lssl']['p5_recovery']:.3f}  |  "
                f"no-lssl: mean={cell['no-lssl']['mean_recovery']:.3f} "
                f"p5={cell['no-lssl']['p5_recovery']:.3f}  |  "
                f"vanilla: mean={cell['vanilla']['mean_recovery']:.3f} "
                f"p5={cell['vanilla']['p5_recovery']:.3f}"
            )

    out_path = OUT / "stress_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
