"""
P0: Closed-Loop Quality Validation on Real GPU
================================================

True KV cache eviction with position-aware generation loop.

This script closes the gap identified in the base audit: instead of
simulating KV pruning via attention-mask zeroing, it performs actual
past_key_values eviction using SparseKVCacheManager.prune_by_chunks()
and feeds the pruned cache back into model.forward() with explicit
position_ids for RoPE correctness.

Experimental Design (from HPCA_REBUTTAL_EXPERIMENT_PLAN.md):
- Models: Qwen2.5-3B (stretch 7B)
- Lengths: 8K / 16K / 32K with artificial HBM capping
- Tasks: passkey retrieval, NIAH (RULER), LongBench retrieval-heavy
- Baselines: ProSE, ProSE-no-PHT, StreamPrefetcher, H2O, SnapKV, Full-KV
- Metrics: accuracy/EM, token-level latency, compression ratio

Usage:
    # GPU run (full experiment)
    python -m prose_v2.scripts.run_p0_closed_loop_eval

    # CPU smoke test (tiny sequences, validates logic)
    python -m prose_v2.scripts.run_p0_closed_loop_eval --smoke-test

    # Single method + length for quick validation
    python -m prose_v2.scripts.run_p0_closed_loop_eval --method prose --length 8192 --smoke-test
"""

import os
import sys
import gc
import json
import math
import time
import random
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

import torch
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.attention.kv_cache_manager import SparseKVCacheManager, PrunedKVCache
from src.runners.e2e_eval_runner import (
    AttentionHookExtractor,
    H2OPolicy,
    SnapKVPolicy,
    StreamingLLMPolicy,
    FullKVPolicy,
    ProSEPromotionPolicy,
    StreamPrefetcherPolicy,
    QuestPolicyE2E,
    BaselinePolicy,
)
from src.benchmarks.passkey import PasskeyBenchmark
from src.benchmarks.ruler import RULERBenchmark
from src.benchmarks.longbench import LongBenchBenchmark, LONGBENCH_TASKS

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# ── Logging ──────────────────────────────────────────────────────────
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("p0_eval")


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class P0Config:
    """P0 experiment configuration."""

    # Model
    model_path: str = str(PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct")
    output_dir: str = str(PROJECT_ROOT / "outputs" / "hpca_fair_hardware" / "p0")

    # Sequence lengths to test
    context_lengths: List[int] = field(default_factory=lambda: [8192, 16384, 32768])

    # Chunking
    chunk_size: int = 256
    anchor_ratio: float = 0.10

    # Generation
    max_new_tokens_passkey: int = 10
    max_new_tokens_ruler: int = 20
    max_new_tokens_longbench: int = 128

    # HBM cap (GB) — force spill-dominated states.
    # For Qwen2.5-3B (36 layers, 2 KV heads, 128 head_dim, fp16):
    #   bytes/token = 2 * 36 * 2 * 128 * 2 = 36,864 B = 36 KB
    #   8K  → 294 MB full KV
    #   16K → 589 MB full KV
    #   32K → 1.18 GB full KV
    # Caps are chosen to force aggressive pruning:
    hbm_caps_gb: Dict[int, float] = field(default_factory=lambda: {
        8192: 0.15,   # ~50% budget at 8K
        16384: 0.30,  # ~50% budget at 16K
        32768: 0.60,  # ~50% budget at 32K
    })

    # Methods
    methods: List[str] = field(default_factory=lambda: [
        "full_kv", "prose", "prose_no_pht", "stream_prefetcher",
        "h2o", "snapkv",
    ])

    # Fixed budget overrides (if set, ignore HBM cap)
    budget_ratios: Optional[Dict[str, float]] = None

    # Sampling
    passkey_samples_per_config: int = 10
    ruler_samples_per_config: int = 5
    longbench_max_samples: int = 20
    longbench_tasks: List[str] = field(default_factory=lambda: [
        "hotpotqa", "narrativeqa", "qasper",
    ])
    ruler_tasks: List[str] = field(default_factory=lambda: [
        "niah_single", "niah_multi",
    ])

    # Device / dtype
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"

    # Smoke test (tiny sequences for CPU validation)
    smoke_test: bool = False

    # Seed
    seed: int = 42

    def __post_init__(self):
        if self.smoke_test:
            self.context_lengths = [256, 512]
            self.passkey_samples_per_config = 2
            self.ruler_samples_per_config = 2
            self.longbench_max_samples = 3
            self.hbm_caps_gb = {256: 0.01, 512: 0.01}


# ═══════════════════════════════════════════════════════════════════════
# MODEL WRAPPER — True Sparse KV Generation
# ═══════════════════════════════════════════════════════════════════════

class TrueSparseKVGenerator:
    """
    End-to-end generator with TRUE KV eviction.

    Flow per sample:
      1. Prefill (full forward) → capture past_key_values + attention hooks
      2. Policy selects chunks
      3. SparseKVCacheManager.prune_by_chunks() → genuinely smaller KV
      4. Decode with explicit position_ids for RoPE correctness
    """

    def __init__(self, model, tokenizer, config: P0Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device

        # Attention hooks for chunk-level attention extraction
        self.hook_extractor = AttentionHookExtractor(model)
        self.hook_extractor.register()

        # Architecture params
        cfg = model.config
        self.num_layers = getattr(cfg, "num_hidden_layers", 32)
        self.num_kv_heads = getattr(
            cfg, "num_key_value_heads",
            getattr(cfg, "num_attention_heads", 32),
        )
        self.head_dim = getattr(
            cfg, "head_dim",
            cfg.hidden_size // cfg.num_attention_heads,
        )

    # ── Helpers ──────────────────────────────────────────────────────

    def _kv_bytes_per_token(self) -> int:
        """KV cache bytes per token (fp16)."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * 2

    def _full_kv_gb(self, seq_len: int) -> float:
        return self._kv_bytes_per_token() * seq_len / (1024 ** 3)

    def _effective_budget(self, seq_len: int, method: str,
                          hbm_cap_gb: Optional[float] = None) -> float:
        """Derive budget ratio from HBM cap or use full for baseline."""
        if method == "full_kv":
            return 1.0
        if self.config.budget_ratios and method in self.config.budget_ratios:
            return self.config.budget_ratios[method]
        if hbm_cap_gb is not None:
            full_gb = self._full_kv_gb(seq_len)
            ratio = hbm_cap_gb / full_gb
            return min(1.0, max(0.05, ratio))
        return 0.10  # default

    def _create_policy(self, method: str) -> BaselinePolicy:
        if method == "h2o":
            return H2OPolicy()
        elif method == "snapkv":
            return SnapKVPolicy()
        elif method == "streaming":
            return StreamingLLMPolicy()
        elif method == "full_kv":
            return FullKVPolicy()
        elif method == "quest":
            return QuestPolicyE2E()
        elif method == "prose":
            return ProSEPromotionPolicy(
                enable_qfc=True, enable_pht_hw=True, enable_lookahead=True,
            )
        elif method == "prose_no_pht":
            return ProSEPromotionPolicy(
                enable_qfc=True, enable_pht_hw=False, enable_lookahead=True,
            )
        elif method == "stream_prefetcher":
            return StreamPrefetcherPolicy(prefetch_depth=2, stream_threshold=2)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _dynamic_cache_to_tuple(
        self, cache: DynamicCache,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Convert DynamicCache → tuple of (K,V) per layer."""
        result = []
        for layer in getattr(cache, "layers", []):
            result.append((layer.keys, layer.values))
        return tuple(result)

    def _pruned_to_dynamic_cache(self, pruned: PrunedKVCache) -> DynamicCache:
        """Convert PrunedKVCache → DynamicCache for transformers 5.0+."""
        cache = DynamicCache()
        for layer_idx in range(pruned.num_layers):
            k, v = pruned.kv_per_layer[layer_idx]
            cache.update(k, v, layer_idx)
        return cache

    def _extract_chunk_attention(
        self, seq_len: int, chunk_size: int,
    ) -> Tuple[Dict[int, float], List[Tuple[int, int]]]:
        """Aggregate attention hooks into per-chunk masses."""
        boundaries = []
        for start in range(0, seq_len, chunk_size):
            boundaries.append((start, min(start + chunk_size, seq_len)))
        chunk_attn = self.hook_extractor.get_per_chunk_attention(boundaries, layer_idx=0)
        return chunk_attn, boundaries

    # ── Core: Prefill + Prune ───────────────────────────────────────

    @torch.no_grad()
    def prefill_and_prune(
        self,
        input_ids: torch.Tensor,
        method: str,
        hbm_cap_gb: Optional[float] = None,
    ) -> Tuple[DynamicCache, int, int, Dict[str, Any]]:
        """
        Run full prefill, select chunks, prune KV, return pruned DynamicCache.

        Returns:
            pruned_cache: DynamicCache with reduced sequence length
            original_seq_len: N (full context length)
            pruned_seq_len: M (retained tokens)
            stats: dict with timing, compression, etc.
        """
        t0 = time.time()
        seq_len = input_ids.shape[1]
        budget_ratio = self._effective_budget(seq_len, method, hbm_cap_gb)

        # ── Prefill ──
        self.hook_extractor.step_clear()
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            use_cache=True,
            output_attentions=True,
        )
        past_kv = outputs.past_key_values  # DynamicCache
        prefill_ms = (time.time() - t0) * 1000

        # ── Chunk attention ──
        chunk_attn, boundaries = self._extract_chunk_attention(
            seq_len, self.config.chunk_size,
        )
        num_chunks = len(boundaries)
        budget_chunks = max(1, int(num_chunks * budget_ratio))

        # Fallback: if hooks didn't capture attention, use uniform
        if not chunk_attn:
            logger.warning("Attention hooks captured no data; using uniform chunk attention")
            chunk_attn = {cid: 1.0 / num_chunks for cid in range(num_chunks)}

        # Anchor chunks (first few + last few)
        num_anchors = max(1, int(num_chunks * self.config.anchor_ratio))
        anchor_ids = sorted(set(
            list(range(min(num_anchors // 2, num_chunks))) +
            list(range(max(0, num_chunks - num_anchors // 2), num_chunks))
        ))

        # ── Policy selection ──
        policy = self._create_policy(method)
        active_ids = policy.select_active_chunks(
            num_chunks=num_chunks,
            budget_chunks=budget_chunks,
            chunk_attn=chunk_attn,
            anchor_ids=anchor_ids,
            step=0,
        )
        anchor_chunk_ids = [c for c in active_ids if c in anchor_ids]
        promoted_chunk_ids = [c for c in active_ids if c not in anchor_ids]

        # ── True KV pruning ──
        t1 = time.time()
        kv_manager = SparseKVCacheManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=torch.float16,
        )
        kv_manager.load_from_prefill(self._dynamic_cache_to_tuple(past_kv))
        kv_manager.set_chunk_boundaries(boundaries)

        pruned = kv_manager.prune_by_chunks(
            anchor_chunk_ids=anchor_chunk_ids,
            promoted_chunk_ids=promoted_chunk_ids,
        )
        prune_ms = (time.time() - t1) * 1000

        # Convert back to DynamicCache
        pruned_cache = self._pruned_to_dynamic_cache(pruned)

        stats = {
            "prefill_ms": round(prefill_ms, 2),
            "prune_ms": round(prune_ms, 2),
            "original_seq_len": seq_len,
            "pruned_seq_len": pruned.total_entries,
            "compression_ratio": round(pruned.compression_ratio, 4),
            "budget_ratio": round(budget_ratio, 4),
            "num_chunks": num_chunks,
            "active_chunks": len(active_ids),
            "anchor_chunks": len(anchor_chunk_ids),
            "promoted_chunks": len(promoted_chunk_ids),
            "full_kv_gb": round(self._full_kv_gb(seq_len), 4),
            "pruned_kv_gb": round(self._full_kv_gb(pruned.total_entries), 4),
        }

        return pruned_cache, seq_len, pruned.total_entries, stats

    # ── Core: Decode with explicit position IDs ─────────────────────

    @torch.no_grad()
    def generate_with_pruned_kv(
        self,
        pruned_cache: DynamicCache,
        original_seq_len: int,
        query_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate answer tokens using pruned KV + explicit position_ids.

        Position IDs are set to the original token positions so that RoPE
        embeddings are computed correctly even though the KV cache has
        been physically evicted (reduced length).
        """
        device = self.device
        query_ids = query_ids.to(device)
        q_len = query_ids.shape[1]

        # Query tokens start at position = original_seq_len
        query_positions = torch.arange(
            original_seq_len, original_seq_len + q_len,
            device=device,
        ).unsqueeze(0)

        t0 = time.time()
        out = self.model(
            input_ids=query_ids,
            past_key_values=pruned_cache,
            position_ids=query_positions,
            use_cache=True,
        )
        past_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = [next_token]

        for step in range(max_new_tokens - 1):
            # Next position = original_seq_len + q_len + step
            pos = original_seq_len + q_len + step
            position_ids = torch.tensor([[pos]], device=device)

            out = self.model(
                input_ids=next_token,
                past_key_values=past_kv,
                position_ids=position_ids,
                use_cache=True,
            )
            past_kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        decode_ms = (time.time() - t0) * 1000
        num_steps = len(generated_ids)

        all_ids = torch.cat(generated_ids, dim=-1)
        stats = {
            "decode_ms": round(decode_ms, 2),
            "tokens_generated": num_steps,
            "ms_per_token": round(decode_ms / max(num_steps, 1), 3),
        }
        return all_ids, stats


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class P0Evaluator:
    """Runs P0 benchmarks across methods, lengths, and HBM caps."""

    def __init__(self, config: P0Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.generator = None

    def load_model(self):
        """Load model and tokenizer."""
        if self.model is not None:
            return

        logger.info(f"Loading model from {self.config.model_path}")
        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtype_map.get(self.config.dtype, torch.float16)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            dtype=dtype,
            device_map=self.config.device,
            attn_implementation="eager",
        )
        self.model.eval()

        self.generator = TrueSparseKVGenerator(self.model, self.tokenizer, self.config)
        logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    def unload_model(self):
        """Free GPU memory."""
        if self.generator:
            self.generator.hook_extractor.clear()
        self.model = None
        self.tokenizer = None
        self.generator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Passkey ──────────────────────────────────────────────────────

    def evaluate_passkey(self, method: str, ctx_len: int) -> Dict[str, Any]:
        """Run passkey retrieval for one (method, length) config."""
        hbm_cap = self.config.hbm_caps_gb.get(ctx_len)

        benchmark = PasskeyBenchmark(
            tokenizer=self.tokenizer,
            context_lengths=[ctx_len],
            passkey_positions=[0.0, 0.25, 0.5, 0.75, 1.0],
            num_samples_per_config=self.config.passkey_samples_per_config,
        )
        examples = benchmark.generate_dataset()

        correct = 0
        total = 0
        details = []
        latencies = []

        for ex in examples:
            context_ids = self.tokenizer.encode(
                ex.context, return_tensors="pt",
            ).to(self.config.device)
            query_ids = self.tokenizer.encode(
                ex.query, return_tensors="pt",
            ).to(self.config.device)

            try:
                pruned_cache, orig_len, pruned_len, prune_stats = \
                    self.generator.prefill_and_prune(context_ids, method, hbm_cap)

                gen_ids, gen_stats = self.generator.generate_with_pruned_kv(
                    pruned_cache, orig_len, query_ids,
                    self.config.max_new_tokens_passkey,
                )

                gen_text = self.tokenizer.decode(
                    gen_ids[0], skip_special_tokens=True,
                )
                is_correct = ex.passkey in gen_text

                correct += int(is_correct)
                total += 1
                latencies.append(gen_stats["ms_per_token"])

                details.append({
                    "context_length": ex.context_length,
                    "position": ex.passkey_position,
                    "passkey": ex.passkey,
                    "correct": is_correct,
                    "generated": gen_text[:60],
                    **prune_stats,
                    **gen_stats,
                })
            except Exception as e:
                logger.error(f"Passkey sample failed: {e}")
                total += 1
                details.append({
                    "context_length": ex.context_length,
                    "position": ex.passkey_position,
                    "error": str(e),
                    "correct": False,
                })

            # Aggressive GC between samples
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        accuracy = correct / max(total, 1)
        return {
            "benchmark": "passkey",
            "method": method,
            "context_length": ctx_len,
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": total,
            "p99_latency_ms": round(np.percentile(latencies, 99), 3) if latencies else None,
            "mean_latency_ms": round(np.mean(latencies), 3) if latencies else None,
            "details": details,
        }

    # ── RULER (NIAH) ─────────────────────────────────────────────────

    def evaluate_ruler(self, method: str, ctx_len: int) -> Dict[str, Any]:
        """Run RULER NIAH tasks for one (method, length) config."""
        hbm_cap = self.config.hbm_caps_gb.get(ctx_len)

        benchmark = RULERBenchmark(
            tokenizer=self.tokenizer,
            context_lengths=[ctx_len],
            num_samples_per_config=self.config.ruler_samples_per_config,
        )
        examples = benchmark.generate_dataset(tasks=self.config.ruler_tasks)

        task_counts = {t: {"correct": 0, "total": 0} for t in self.config.ruler_tasks}
        details = []
        latencies = []

        for ex in examples:
            context_ids = self.tokenizer.encode(
                ex.context, return_tensors="pt",
                truncation=True, max_length=ctx_len,
            ).to(self.config.device)
            query_ids = self.tokenizer.encode(
                ex.query, return_tensors="pt",
            ).to(self.config.device)

            try:
                pruned_cache, orig_len, pruned_len, prune_stats = \
                    self.generator.prefill_and_prune(context_ids, method, hbm_cap)

                gen_ids, gen_stats = self.generator.generate_with_pruned_kv(
                    pruned_cache, orig_len, query_ids,
                    self.config.max_new_tokens_ruler,
                )

                pred = self.tokenizer.decode(
                    gen_ids[0], skip_special_tokens=True,
                ).strip()
                is_correct = any(ans in pred for ans in ex.all_answers)

                task_counts[ex.task]["correct"] += int(is_correct)
                task_counts[ex.task]["total"] += 1
                latencies.append(gen_stats["ms_per_token"])

                details.append({
                    "task": ex.task,
                    "context_length": ex.target_length,
                    "correct": is_correct,
                    "prediction": pred[:60],
                    **prune_stats,
                    **gen_stats,
                })
            except Exception as e:
                logger.error(f"RULER sample failed: {e}")
                task_counts[ex.task]["total"] += 1
                details.append({
                    "task": ex.task,
                    "error": str(e),
                    "correct": False,
                })

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        task_accs = {
            t: c["correct"] / max(c["total"], 1)
            for t, c in task_counts.items()
        }
        overall = sum(task_accs.values()) / max(len(task_accs), 1)

        return {
            "benchmark": "ruler",
            "method": method,
            "context_length": ctx_len,
            "accuracy": round(overall, 4),
            "by_task": {k: round(v, 4) for k, v in task_accs.items()},
            "p99_latency_ms": round(np.percentile(latencies, 99), 3) if latencies else None,
            "mean_latency_ms": round(np.mean(latencies), 3) if latencies else None,
            "details": details,
        }

    # ── LongBench ────────────────────────────────────────────────────

    def evaluate_longbench(self, method: str, ctx_len: int) -> Dict[str, Any]:
        """Run LongBench retrieval-heavy subsets."""
        hbm_cap = self.config.hbm_caps_gb.get(ctx_len)

        benchmark = LongBenchBenchmark(
            tokenizer=self.tokenizer,
            tasks=self.config.longbench_tasks,
            max_samples_per_task=self.config.longbench_max_samples,
            max_gen_tokens=self.config.max_new_tokens_longbench,
        )

        results_by_task = {}
        latencies = []

        for task in self.config.longbench_tasks:
            try:
                examples = benchmark.load_dataset(task)
            except Exception as e:
                logger.warning(f"LongBench dataset {task} unavailable: {e}")
                continue
            if not examples:
                continue

            scores = []
            for ex in examples:
                # Truncate context to target length
                context_ids = self.tokenizer.encode(
                    ex.context, return_tensors="pt",
                    truncation=True, max_length=ctx_len,
                ).to(self.config.device)
                query_ids = self.tokenizer.encode(
                    ex.query, return_tensors="pt",
                ).to(self.config.device)

                try:
                    pruned_cache, orig_len, pruned_len, prune_stats = \
                        self.generator.prefill_and_prune(context_ids, method, hbm_cap)

                    gen_ids, gen_stats = self.generator.generate_with_pruned_kv(
                        pruned_cache, orig_len, query_ids,
                        self.config.max_new_tokens_longbench,
                    )

                    pred = self.tokenizer.decode(
                        gen_ids[0], skip_special_tokens=True,
                    ).strip()
                    score = benchmark._score(pred, ex, LONGBENCH_TASKS[task]["metric"])
                    scores.append(score)
                    latencies.append(gen_stats["ms_per_token"])
                except Exception as e:
                    logger.error(f"LongBench {task} sample failed: {e}")
                    scores.append(0.0)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_score = sum(scores) / max(len(scores), 1)
            results_by_task[task] = {
                "score": round(avg_score, 4),
                "n_samples": len(scores),
                "metric": LONGBENCH_TASKS[task]["metric"],
            }

        overall = sum(r["score"] for r in results_by_task.values()) / max(len(results_by_task), 1)

        return {
            "benchmark": "longbench",
            "method": method,
            "context_length": ctx_len,
            "overall": round(overall, 4),
            "by_task": results_by_task,
            "p99_latency_ms": round(np.percentile(latencies, 99), 3) if latencies else None,
            "mean_latency_ms": round(np.mean(latencies), 3) if latencies else None,
        }

    # ── Sweep orchestrator ───────────────────────────────────────────

    def run_sweep(self) -> Dict[str, Any]:
        """Run full P0 sweep across methods, lengths, and benchmarks."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        all_results = []
        summary_rows = []

        for ctx_len in self.config.context_lengths:
            logger.info(f"\n{'='*60}")
            logger.info(f"Context Length: {ctx_len}")
            logger.info(f"{'='*60}")

            for method in self.config.methods:
                logger.info(f"\n--- Method: {method} ---")
                tag = f"{method}@{ctx_len}"

                try:
                    self.load_model()

                    # Passkey
                    pk_result = self.evaluate_passkey(method, ctx_len)
                    all_results.append(pk_result)
                    summary_rows.append({
                        "benchmark": "passkey",
                        "method": method,
                        "length": ctx_len,
                        "accuracy": pk_result["accuracy"],
                        "p99_lat_ms": pk_result.get("p99_latency_ms"),
                    })
                    logger.info(f"  Passkey: {pk_result['accuracy']:.2%} "
                                f"({pk_result['correct']}/{pk_result['total']})")

                    # RULER
                    ruler_result = self.evaluate_ruler(method, ctx_len)
                    all_results.append(ruler_result)
                    summary_rows.append({
                        "benchmark": "ruler",
                        "method": method,
                        "length": ctx_len,
                        "accuracy": ruler_result["accuracy"],
                        "p99_lat_ms": ruler_result.get("p99_latency_ms"),
                    })
                    logger.info(f"  RULER:   {ruler_result['accuracy']:.2%}")

                    # LongBench (only for ProSE and Full-KV to save time)
                    if method in ("prose", "full_kv"):
                        lb_result = self.evaluate_longbench(method, ctx_len)
                        all_results.append(lb_result)
                        summary_rows.append({
                            "benchmark": "longbench",
                            "method": method,
                            "length": ctx_len,
                            "accuracy": lb_result["overall"],
                            "p99_lat_ms": lb_result.get("p99_latency_ms"),
                        })
                        logger.info(f"  LongBench: {lb_result['overall']:.2%}")

                except Exception as e:
                    logger.error(f"FAILED {tag}: {e}")
                    traceback.print_exc()
                    all_results.append({
                        "method": method,
                        "context_length": ctx_len,
                        "error": str(e),
                    })

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = out_dir / f"p0_results_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump({
                "config": asdict(self.config),
                "results": all_results,
                "summary": summary_rows,
            }, f, indent=2, default=str)

        summary_path = out_dir / f"p0_summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary_rows, f, indent=2)

        logger.info(f"\nResults saved to {results_path}")
        self._print_summary(summary_rows)
        return {"results": all_results, "summary": summary_rows}

    def _print_summary(self, rows: List[Dict]):
        """Pretty-print summary table."""
        logger.info("\n" + "="*70)
        logger.info("P0 SUMMARY")
        logger.info("="*70)
        print(f"{'Benchmark':<12} {'Method':<18} {'Length':>8} {'Accuracy':>10} {'P99(ms)':>10}")
        print("-"*70)
        for r in rows:
            acc = r.get("accuracy")
            acc_str = f"{acc:.2%}" if acc is not None else "N/A"
            lat = r.get("p99_lat_ms")
            lat_str = f"{lat:.2f}" if lat is not None else "N/A"
            print(f"{r['benchmark']:<12} {r['method']:<18} {r['length']:>8} "
                  f"{acc_str:>10} {lat_str:>10}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="P0 Closed-Loop Quality Validation")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to local HF model")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny sequences for CPU validation")
    parser.add_argument("--method", type=str, default=None,
                        help="Run single method only")
    parser.add_argument("--length", type=int, default=None,
                        help="Run single length only")
    parser.add_argument("--passkey-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = P0Config(smoke_test=args.smoke_test, seed=args.seed)

    if args.model_path:
        config.model_path = args.model_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.device:
        config.device = args.device
    if args.method:
        config.methods = [args.method]
    if args.length:
        config.context_lengths = [args.length]

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    evaluator = P0Evaluator(config)
    evaluator.run_sweep()


if __name__ == "__main__":
    main()
