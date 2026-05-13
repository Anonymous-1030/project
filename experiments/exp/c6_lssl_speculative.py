"""
C6 — LSSL evaluated in a closed-loop speculative-decode trace.

Models speculative decoding with a target model running committed tokens and
a draft model proposing 4-token lookahead.  Each speculative step triggers
candidate-set drift, which stresses the Liveness-Safe Sketch Lifecycle (LSSL).

Per step, each chunk's sketch is in one of three states:
  * Fresh   — sketch updated within tau_fresh steps
  * Aging   — sketch updated within tau_aging steps (degraded but usable)
  * Missing — sketch never observed or TTL expired

LSSL policy:
  * Fresh  → admit if score > theta_fresh
  * Aging  → admit if score > theta_aging (> theta_fresh) and with
             confidence guard
  * Missing → drop (safer than keep-stale)

Compared policies:
  no_lssl      — treat all sketches as Fresh (stale-sketch poisoning possible)
  stale_keep   — accept aging/missing (reviewer's "keep-stale")
  lssl_drop    — PROSE's LSSL: drop-on-miss, confidence-guard on aging
  lssl_refresh — LSSL + precompute sketches for draft candidates at commit
                 (proposed fix in case Fresh-fraction falls below 0.70)

Produces:
  out/data/c6_lssl_speculative.json
  out/figures/c6_lssl_freshness.{png,pdf}
  out/figures/c6_lssl_policy_compare.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.io_utils import save_fig, save_json, C


def simulate_trace(
    n_steps: int,
    n_chunks: int,
    draft_lookahead: int,
    draft_acceptance: float,
    sketch_refresh_latency_steps: int,
    policy: str,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    last_refresh = -np.ones(n_chunks, dtype=int) * 1_000_000
    # Emulate a query-dependent drift: each speculative step rewrites ~8%
    # of chunks' useful-ness label (i.e. the scoring "ground truth" shifts).
    useful = rng.random(n_chunks) < 0.04
    recovery = []
    state_counts = {"fresh": 0, "aging": 0, "missing": 0}

    tau_fresh = 4
    tau_aging = 12
    theta_fresh = 0.45
    theta_aging = 0.55
    confidence_guard = 0.1

    admits_per_step = 64

    for t in range(n_steps):
        # Speculative step: propose `draft_lookahead` candidate tokens.
        # Each committed step refreshes sketches for the recently-visited
        # chunk set (bounded by credit budget).
        if t % max(1, draft_lookahead) == 0:
            # Batch-refresh top recent chunks each commit
            recent = rng.choice(n_chunks, size=min(256, n_chunks), replace=False)
            last_refresh[recent] = t

        # Some speculative steps are rejected: draft_acceptance controls rate
        rejected = rng.random() > draft_acceptance
        if rejected:
            # The sketches we *did* refresh on the draft path are now stale w.r.t.
            # the committed path — simulate by invalidating a random subset
            invalidated = rng.choice(n_chunks, size=int(0.15 * n_chunks), replace=False)
            last_refresh[invalidated] = -1_000_000

        ages = t - last_refresh
        fresh   = ages <= tau_fresh
        aging   = (ages > tau_fresh) & (ages <= tau_aging)
        missing = ages > tau_aging
        state_counts["fresh"]   += fresh.sum()
        state_counts["aging"]   += aging.sum()
        state_counts["missing"] += missing.sum()

        # Noisy scores (ground truth + noise; policy gates on them)
        scores = useful.astype(float) * 0.6 + rng.normal(0.5, 0.20, n_chunks)
        conf   = np.clip(scores, 0, 1)  # confidence proxy

        # Apply policy to decide candidate admission per chunk
        admit_gate = np.zeros(n_chunks, dtype=bool)
        if policy == "no_lssl":
            admit_gate = scores >= theta_fresh
        elif policy == "stale_keep":
            admit_gate = ((fresh | aging) & (scores >= theta_fresh)) | \
                         (missing & (scores >= theta_aging))
        elif policy == "lssl_drop":
            admit_gate = (fresh & (scores >= theta_fresh)) | \
                         (aging & (scores >= theta_aging) & (conf >= theta_aging + confidence_guard))
        elif policy == "lssl_refresh":
            admit_gate = (fresh & (scores >= theta_fresh)) | \
                         (aging & (scores >= theta_aging) & (conf >= theta_aging + confidence_guard))
            # Refresh draft-path candidates at commit (proactive).
            draft_cands = rng.choice(n_chunks, size=128, replace=False)
            last_refresh[draft_cands] = t
        else:
            raise ValueError(policy)

        # Top budget among admit_gate
        idx = np.where(admit_gate)[0]
        if len(idx) > admits_per_step:
            top = idx[np.argsort(scores[idx])[::-1][:admits_per_step]]
        else:
            top = idx

        top_useful_ids = np.where(useful)[0][:32]
        recall = len(set(top) & set(top_useful_ids)) / max(1, len(top_useful_ids))

        # LSSL safety event: did we admit a stale sketch?
        stale_admit_rate = float((aging | missing)[top].sum()) / max(1, len(top))
        recovery.append((recall, stale_admit_rate))

        # Useful-label drift each step
        drift = rng.random(n_chunks) < 0.02
        useful[drift] = ~useful[drift]

    recov = np.array([r[0] for r in recovery])
    stale = np.array([r[1] for r in recovery])
    total = sum(state_counts.values())
    return dict(
        policy=policy,
        mean_recovery=float(recov.mean()),
        p5_recovery=float(np.percentile(recov, 5)),
        mean_stale_admit=float(stale.mean()),
        fresh_frac=state_counts["fresh"] / total,
        aging_frac=state_counts["aging"] / total,
        missing_frac=state_counts["missing"] / total,
        recov_trace=recov.tolist(),
    )


def main():
    policies = ["no_lssl", "stale_keep", "lssl_drop", "lssl_refresh"]
    results = []
    for p in policies:
        r = simulate_trace(
            n_steps=2000, n_chunks=2048, draft_lookahead=4,
            draft_acceptance=0.65, sketch_refresh_latency_steps=1,
            policy=p, seed=11,
        )
        results.append(r)
        print(f"[C6] {p:<14s}  recovery={r['mean_recovery']:.3f}  "
              f"stale_admit={r['mean_stale_admit']:.3f}  "
              f"fresh/aging/missing = "
              f"{r['fresh_frac']:.2f}/{r['aging_frac']:.2f}/{r['missing_frac']:.2f}")

    save_json("c6_lssl_speculative", {"results": results})

    # --------- Figure: combined 3-panel (freshness + stale-admit + recovery)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.0))
    idx = np.arange(len(policies))
    colors_bar = [C["fts"], C["sw_host"], C["cefe"], C["accent2"]]

    # Panel 1: Stale-admit rate (the key safety metric — differentiates policies)
    stale_vals = [r["mean_stale_admit"] for r in results]
    axes[0].bar(idx, stale_vals, color=colors_bar, edgecolor="black", lw=0.6)
    axes[0].set_xticks(idx); axes[0].set_xticklabels(policies, rotation=15, ha="right")
    axes[0].set_ylabel("Stale-sketch admit rate")
    axes[0].set_title("Safety: fraction of admits\nusing stale/missing sketches")
    for i, v in enumerate(stale_vals):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=18)

    # Panel 2: Recovery@K (quality — should be similar across policies)
    recov_vals = [r["mean_recovery"] for r in results]
    axes[1].bar(idx, recov_vals, color=colors_bar, edgecolor="black", lw=0.6)
    axes[1].set_xticks(idx); axes[1].set_xticklabels(policies, rotation=15, ha="right")
    axes[1].set_ylabel("Mean Recovery@K")
    axes[1].set_title("Quality: Recovery@K\n(iso-quality across policies)")
    for i, v in enumerate(recov_vals):
        axes[1].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=18)

    # Panel 3: Freshness distribution (only lssl_refresh differs)
    fresh   = [r["fresh_frac"]   for r in results]
    aging   = [r["aging_frac"]   for r in results]
    missing = [r["missing_frac"] for r in results]
    axes[2].bar(idx, fresh,   color=C["cefe"], label="Fresh",   edgecolor="black", lw=0.4)
    axes[2].bar(idx, aging,   bottom=fresh, color=C["accent1"],
           label="Aging", edgecolor="black", lw=0.4)
    axes[2].bar(idx, missing, bottom=np.array(fresh)+np.array(aging),
           color=C["fts"], label="Missing", edgecolor="black", lw=0.4)
    axes[2].set_xticks(idx); axes[2].set_xticklabels(policies, rotation=15, ha="right")
    axes[2].set_ylabel("Fraction of observations")
    axes[2].set_title("Sketch freshness distribution\n(only lssl_refresh changes refresh)")
    axes[2].legend(loc="upper right", fontsize=16)

    fig.suptitle("")  # removed — subplot titles are self-explanatory
    save_fig(fig, "c6_lssl_freshness")
    save_fig(fig, "c6_lssl_freshness")

    # --------- Figure: time-series of stale-admit (shows dynamics) ----
    fig, ax = plt.subplots(figsize=(12, 5.5))
    # Re-run with trace output for two key policies
    for p, col, ls in [("no_lssl", C["fts"], "-"),
                       ("lssl_drop", C["cefe"], "--"),
                       ("lssl_refresh", C["accent2"], "-")]:
        r2 = simulate_trace(
            n_steps=500, n_chunks=2048, draft_lookahead=4,
            draft_acceptance=0.65, sketch_refresh_latency_steps=1,
            policy=p, seed=11,
        )
        # Smooth with rolling window
        trace = np.array(r2["recov_trace"])
        window = 20
        smoothed = np.convolve(trace, np.ones(window)/window, mode="valid")
        ax.plot(smoothed, color=col, ls=ls, label=p, lw=2.0)
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Recovery@K (smoothed)")
    ax.set_title("Recovery@K trace under speculative-decode drift")
    ax.legend(frameon=False)
    ax.grid(True, ls=":", lw=0.5, alpha=0.5)
    save_fig(fig, "c6_lssl_policy_compare")

    print("[C6] done.")


if __name__ == "__main__":
    main()
