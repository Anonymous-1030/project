# C4 — PROSE_REMOTE_KV descriptor: stream/fence semantics

## Descriptor format
Every PROSE_REMOTE_KV descriptor carries:
  * tenant_id, namespace_id, chunk_id, version, payload_addr
  * ordering_attr ∈ {NON_STRICT, STRICT_SENTINEL}
  * completion_mask : 3-bit one-of {ADMIT_COMMITTED, ADMIT_REJECT, ADMIT_ABORT}

## Ordering rule
- A NON_STRICT descriptor is ordered against the preceding descriptor only for
  its own completion post; it does not stall subsequent descriptors on the
  same stream.
- Consumers that depend on the payload read the completion code via a
  stream-callback (CUDA graph conditional node or persistent kernel read-of-
  completion) and branch: if ADMIT_COMMITTED, consume; otherwise take the
  miss path.
- STRICT_SENTINEL mode is provided for compatibility with naive consumers.
  On reject, CEFE posts a zero-length DMA that advances the consumer's
  stream fence; offered load impact is sub-1%.

## Fence interaction
- PROSE_REMOTE_KV descriptors are NOT serialising.  A subsequent DMA on the
  same stream is not blocked by an in-flight admission decision.
- A cudaStreamSynchronize crosses all PROSE_REMOTE_KV completions: this makes
  batch-level validation straightforward (all admissions resolve before the
  next batch step reads).

## Backpressure boundary
- MetaRead credit pool M (default 256) throttles the DSQ only when all
  credits are in flight.  The DSQ holds up PROSE_REMOTE_KV descriptors but
  does NOT hold up ordinary DMA descriptors — CEFE's descriptor classifier
  forwards those unchanged on a separate bypass path.

## Commit boundary
- ADMIT_COMMITTED is emitted only after:
  (1) payload integrity check (CRC) passes
  (2) version matches the version recorded at MetaRead time (else ADMIT_ABORT)
  (3) namespace_id matches the descriptor namespace (else ADMIT_REJECT)
- Consumers see a stable, validated payload; the transient Promotion Buffer
  region is reused only after an ADMIT_COMMITTED or ADMIT_ABORT completion.

## Stream semantic corollary
With the above, no spurious stalls arise on correctly-written consumers.  A
legacy consumer that treats PROSE_REMOTE_KV as a normal DMA is covered by
STRICT_SENTINEL mode at an O(<1%) offered-load cost.
