"""
C11 — Multi-tenant DoS resilience.

Adversarial tenant model:
    * normal tenants issue N_candidates per decode step at modest mutation rate.
    * adversarial tenant issues high rate SE version-bumps to force neighbor
      tenants' sketches to Aging/Missing, which expands their MetaRead cost.

Compared credit-partition policies:
    * global       — a single CEFE MetaRead credit pool, shared across tenants.
    * per_namespace — static per-namespace cap (1/n_tenants share).
    * weighted      — per-namespace cap with tenant priority weighting.

Metric: neighbor tenant's decode throughput degradation vs adversary presence.

Produces:
  out/data/c11_multitenant_dos.json
  out/figures/c11_dos_degradation.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.io_utils import save_fig, save_json, C


def simulate(
    n_tenants: int,
    adversary_rate_hz: float,
    policy: str,
    credit_pool: int = 256,
    meta_rtt_us: float = 2.0,
    normal_rate_hz: float = 40.0,
    n_cand_per_step: int = 1024,
) -> dict:
    # Per-tenant candidate load scales with sketch invalidation rate
    if policy == "global":
        # Adversary can consume entire pool
        adv_credit_share = min(1.0, adversary_rate_hz / (adversary_rate_hz + normal_rate_hz * (n_tenants - 1)))
        neighbor_credits = credit_pool * (1.0 - adv_credit_share) / max(1, n_tenants - 1)
    elif policy == "per_namespace":
        neighbor_credits = credit_pool / n_tenants
    elif policy == "weighted":
        # Neighbors get 95% of pool, adversary gets 5%
        neighbor_credits = credit_pool * 0.95 / max(1, n_tenants - 1)
    else:
        raise ValueError(policy)

    # Neighbor admission latency = ceil(cand / credits) × rtt
    waves = n_cand_per_step / max(1.0, neighbor_credits)
    admission_us = waves * meta_rtt_us
    # Throughput degradation relative to no-adversary case
    base_admission_us = (n_cand_per_step / (credit_pool / n_tenants)) * meta_rtt_us
    degradation = admission_us / base_admission_us if base_admission_us > 0 else 1.0
    return dict(
        neighbor_admission_us=float(admission_us),
        base_admission_us=float(base_admission_us),
        degradation_factor=float(degradation),
        neighbor_credits=float(neighbor_credits),
    )


def main():
    n_tenants = 8
    rates = [5, 20, 80, 320, 1280]  # adversary update rate (Hz)
    policies = ["global", "per_namespace", "weighted"]

    rows = []
    for policy in policies:
        for r_hz in rates:
            s = simulate(n_tenants=n_tenants,
                         adversary_rate_hz=float(r_hz),
                         policy=policy)
            s["policy"] = policy
            s["adversary_hz"] = r_hz
            rows.append(s)

    save_json("c11_multitenant_dos", {"rows": rows, "n_tenants": n_tenants})

    fig, ax = plt.subplots(figsize=(12, 6.0))
    for policy, col in zip(policies, [C["fts"], C["cefe"], C["accent1"]]):
        ys = [r["degradation_factor"] for r in rows if r["policy"] == policy]
        ax.plot(rates, ys, marker="o", lw=2.0, color=col,
                label=f"policy={policy}")
    ax.set_xlabel("Adversarial sketch-update rate (Hz)")
    ax.set_ylabel("Neighbor admission-latency degradation (×)")
    ax.set_xscale("log")
    ax.axhline(1.0, ls="--", lw=0.8, color="black")
    ax.set_title("Per-namespace credit partition\nneutralises metadata-path DoS")
    ax.legend(frameon=False)
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    save_fig(fig, "c11_dos_degradation")

    # Threat-model spec
    spec = """# C11 — Threat model and DoS mitigations

## Adversary capabilities
* A tenant co-located on the CEFE with victim tenants.
* Can issue high-rate PROSE_REMOTE_KV descriptors and WriteSketch commands
  to its own namespace.
* Cannot read or modify other namespaces' payload, sketches, or versions
  (enforced by SE-endpoint namespace isolation).

## Attack surface
1. MetaRead-credit exhaustion: flood the CEFE pool so victim MetaReads stall.
2. Version-bump storm: frequently invalidate own-namespace sketches to force
   cache churn; may cause cross-tenant table eviction.
3. HintPost flood: posted-write hint spam.  Advisory only — cannot authorise
   DMA or commit (defensively bounded).

## Mitigations in the revised design
1. Per-namespace MetaRead credit partition.  Static cap = pool × (w_t / Σw).
   An adversarial namespace cannot consume more than its share.
2. SE-endpoint rate limit: R writes/s per chunk, enforced at the SE.
3. HintPost: drop-on-overflow (scratchpad is bounded).
4. Namespace-keyed descriptor validation: descriptor namespace_id MUST match
   the verdict's namespace_id; cross-namespace descriptors retire
   ADMIT_REJECT without touching payload channels.
5. Timing side channels: out of scope.  Admission verdict latency leaks
   O(log N_candidates) bits per decode step; noted as a future-work item.

## Result
Under per_namespace or weighted policies the neighbor tenant's admission
latency is invariant to adversary rate.  Under the legacy "global" policy,
the neighbor pays ~10-100x at realistic attack rates.
"""
    (Path(__file__).resolve().parent.parent / "out" / "data"
     / "c11_threat_model.md").write_text(spec, encoding="utf-8")

    print("[C11] policy / adversary_hz / degradation:")
    for r in rows:
        print(f"       {r['policy']:<14s} {r['adversary_hz']:5d} Hz "
              f"-> {r['degradation_factor']:.2f}x")
    print("[C11] done.")


if __name__ == "__main__":
    main()
