| System | Verdict input | Binding boundary | Quantum (ns) | Payload quantum (B) | Pre-payload reorder? | Commit discipline | RPE risk | Schema instance | Note |
|---|---|---|---|---|---|---|---|---|---|
| Quest (ICML'24) | key min/max pages | HBM page-load | 120 | 4096 | yes | implicit (page level) | low | PCM-hbm | verdict binds inside HBM; CXL not in scope |
| InfiniGen (OSDI'24) | key proxy projection | host-runtime | 900 | 65536 | no | implicit (no abort) | high | non-PCM for CXL | advisory; reorderable by coalescer |
| SnapKV | self-attention mass | HBM page-load | 100 | 4096 | yes | implicit | low | PCM-hbm | operates on resident KV only |
| H2O | heavy-hitter history | HBM page-load | 120 | 4096 | yes | implicit | low | PCM-hbm | eviction-first, not a CXL filter |
| TinyLFU (generic) | frequency sketch | cache-insert | 50 | — | yes | implicit | low | PCM-generic | classical admission filter |
| SW-PCM-host (this) | compact summary | host runtime | 47000 | 65536 | yes | explicit | low | PCM-host | binds pre-doorbell; OS-crossing dominates |
| SW-PCM-GPU (this) | compact summary | GPU persistent kernel | 5200 | 65536 | yes | explicit | low | PCM-gpu | contends with compute kernels |
| PROSE (CEFE, this) | compact summary | on-CE (pre-DMA-dispatch) | 3900 | 65536 | yes | explicit (LSSL + commit) | minimal | PCM-cefe | tightest binding point for CXL-resident KV |