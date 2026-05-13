"""
Baseline methods for comparison.

Existing baselines (KV reduction):
- H2O: Heavy-Hitter Oracle (NeurIPS 2023)
- StreamingLLM: Attention Sink (ICLR 2024)
- SnapKV: Observation-window based (ICML 2024)
- Quest: Query-aware page retrieval (ICML 2024)
- RetrievalAttention: ANN-based token retrieval (NeurIPS 2024)
- InfiniGen: Layer-wise speculative prefetch (OSDI 2024)
- MagicPIG: LSH sampling (NeurIPS 2024)
- Full KV: Upper baseline (no compression)

NEW Tier 1 CXL-Aware Baselines (Must-Have):
- PROSE-FTS: PROSE with fetch-then-score ordering (core ablation)
- Oracle-Policy: Perfect utility oracle with CXL queue constraints
- vLLM-CXL: vLLM PagedAttention extended to CXL
- StreamPrefetcher: Hardware stream prefetcher (formalized)
- FreqRec-PF: Frequency-Recency hybrid prefetcher (formalized)

NEW Tier 2 CXL-Aware Baselines (Should-Have):
- FreqRec-PF+Meta: FreqRec with 64B metadata but fetch-then-score
- H2O-CXL: H2O heavy-hitter retention + blind CXL paging
- SnapKV-CXL: SnapKV local retention + blind CXL offloading
- InfLLM-CXL: InfLLM retrieval-based + fetch-then-decide on CXL
- CUDA-UM: CUDA Unified Memory + cudaMemPrefetchAsync
- Oracle-Candidate: Oracle exposure + ODUS-X ranking

NEW Tier 3 PROSE Ablations:
- NoPHT: PHT/PTB disabled
- SingleCue: Single-cue ODUS-X (5 variants)
- NoPBuffer: Direct commit, no Promotion Buffer
- NoVersionGate: Skip version validation
- FIFOVictim: FIFO eviction instead of utility-per-byte
"""

# Original baselines
from src.baselines.h2o import H2OPolicy, H2ORunner
from src.baselines.streaming_llm import StreamingLLMPolicy, StreamingLLMRunner
from src.baselines.snapkv import SnapKVPolicy, SnapKVRunner
from src.baselines.quest import QuestPolicy, QuestRunner
from src.baselines.retrieval_attention import RetrievalAttentionPolicy
from src.baselines.infinigen import InfiniGenPolicy
from src.baselines.magicpig import MagicPIGPolicy

# Tier 1: CXL-aware hardware baselines
from src.baselines.stream_prefetcher import StreamPrefetcherPolicy
from src.baselines.freqrec_pf import FreqRecPrefetcherPolicy
from src.baselines.prose_fts import PROSEFTSPolicy
from src.baselines.oracle_policy import OraclePolicy, OracleCandidateOnlyPolicy
from src.baselines.vllm_cxl import VLLMCXLPolicy

# Tier 2: KV reduction + CXL baselines
from src.baselines.freqrec_meta import FreqRecMetaPolicy
from src.baselines.h2o_cxl import H2OCXLPolicy
from src.baselines.snapkv_cxl import SnapKVCXLPolicy
from src.baselines.infllm_cxl import InfLLMCXLPolicy
from src.baselines.cuda_um import CUDAUnifiedMemoryPolicy

# Tier 3: PROSE ablations
from src.baselines.prose_ablations import (
    NoPHTPolicy,
    SingleCuePolicy,
    NoPBufferPolicy,
    NoVersionGatePolicy,
    FIFOVictimPolicy,
)

__all__ = [
    # Original
    "H2OPolicy", "H2ORunner",
    "StreamingLLMPolicy", "StreamingLLMRunner",
    "SnapKVPolicy", "SnapKVRunner",
    "QuestPolicy", "QuestRunner",
    "RetrievalAttentionPolicy",
    "InfiniGenPolicy",
    "MagicPIGPolicy",
    # Tier 1
    "StreamPrefetcherPolicy",
    "FreqRecPrefetcherPolicy",
    "PROSEFTSPolicy",
    "OraclePolicy",
    "OracleCandidateOnlyPolicy",
    "VLLMCXLPolicy",
    # Tier 2
    "FreqRecMetaPolicy",
    "H2OCXLPolicy",
    "SnapKVCXLPolicy",
    "InfLLMCXLPolicy",
    "CUDAUnifiedMemoryPolicy",
    # Tier 3
    "NoPHTPolicy",
    "SingleCuePolicy",
    "NoPBufferPolicy",
    "NoVersionGatePolicy",
    "FIFOVictimPolicy",
]
