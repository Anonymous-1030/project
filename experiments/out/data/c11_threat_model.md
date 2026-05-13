# C11 — Threat model and DoS mitigations

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
