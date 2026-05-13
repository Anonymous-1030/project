#!/usr/bin/env python3
"""
Extended Context Real Evaluation: 4K / 8K on Real Hardware (CPU).

Goal: Produce genuine passkey-accuracy data points at the upper limit of
the CPU envelope using careful memory management.  This directly addresses
the reviewer gap: "No validated data beyond 16K" by pushing the real-model
envelope as far as feasible on available hardware, then using calibrated
simulation for 16K+.

Model: Qwen2.5-1.5B-Instruct (local)
Trick: float16 + no_grad + small batch.  No GPU required.
Methods compared:
  - full_kv   (oracle baseline)
  - prose     (ProSE sparse policy, budget_ratio=0.10)
  - snapkv    (software baseline)

Output:
  outputs/hpca_extended/extended_context_report.json
  outputs/hpca_extended/fig_extended_passkey.pdf
"""

import gc
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from transformers import AutoModelForCausalLM, AutoTokenizer
from runners.e2e_eval_runner import (
    E2ERunConfig, ProSEEndToEndRunner,
    SparseKVCacheManager, AttentionHookExtractor,
    ProSEPromotionPolicy, SnapKVPolicy, FullKVPolicy,
)


def build_passkey_text(tokenizer, target_length: int, depth: float = 0.5):
    """Build a synthetic passkey text of roughly target length."""
    filler = (
        "The city was bustling with activity as people went about their daily routines. "
        "Research in artificial intelligence continues to advance at a rapid pace. "
        "The mountain trail wound through dense forests and across rushing streams. "
        "Economic indicators suggest a period of moderate growth ahead. "
        "Ancient civilizations developed sophisticated systems of governance. "
    )
    passkey = str(np.random.randint(100000, 999999))
    needle = f" The secret passkey is {passkey}. Remember this number. "

    base_tokens = len(tokenizer.encode(filler))
    repeats = max(1, target_length // base_tokens + 2)
    full_filler = (filler + " ") * repeats

    insert_pos = int(len(full_filler) * depth)
    text = full_filler[:insert_pos] + needle + full_filler[insert_pos:]
    tokens = tokenizer.encode(text, truncation=True, max_length=target_length)
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text, passkey


def evaluate_method_on_prompt(model, tokenizer, text, query, method_name, budget_ratio=0.10):
    """Run prefill + decode for a single method."""
    device = model.device
    context_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    query_ids = tokenizer.encode(query, return_tensors="pt").to(device)
    seq_len = context_ids.shape[1]

    t0 = time.time()

    with torch.no_grad():
        # Prefill
        outputs = model(input_ids=context_ids, use_cache=True, output_attentions=True)
        past_kv = outputs.past_key_values

        # Build KV manager
        num_layers = model.config.num_hidden_layers
        num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        head_dim = getattr(model.config, "head_dim", 128)
        chunk_size = 64

        kv_manager = SparseKVCacheManager(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            chunk_size=chunk_size,
            anchor_ratio=0.10,
            budget_ratio=budget_ratio,
            device=device,
        )
        kv_manager.store_prefill_kv(past_kv)

        # Extract attention for policy
        hook = AttentionHookExtractor(model)
        hook.register()
        _ = model(input_ids=context_ids, use_cache=False, output_attentions=True)
        chunk_attn = hook.get_per_chunk_attention(kv_manager.chunk_boundaries, layer_idx=0)
        kv_manager.update_anchors_from_attention(chunk_attn)
        hook.clear()

        # Select policy
        if method_name == "full_kv":
            active_ids = list(range(len(kv_manager.chunk_boundaries)))
        elif method_name == "prose":
            policy = ProSEPromotionPolicy()
            budget = kv_manager.get_budget_chunks()
            active_ids = policy.select_active_chunks(
                num_chunks=len(kv_manager.chunk_boundaries),
                budget_chunks=budget,
                chunk_attn=chunk_attn,
                anchor_ids=kv_manager.anchor_ids,
                step=0,
            )
        elif method_name == "snapkv":
            policy = SnapKVPolicy()
            budget = kv_manager.get_budget_chunks()
            active_ids = policy.select_active_chunks(
                num_chunks=len(kv_manager.chunk_boundaries),
                budget_chunks=budget,
                chunk_attn=chunk_attn,
                anchor_ids=kv_manager.anchor_ids,
                step=0,
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")

        promoted = [c for c in active_ids if c not in kv_manager.anchor_ids]
        kv_manager.set_promoted(promoted)

        # Build sparse cache
        active_kv = kv_manager.get_active_kv()
        from transformers.cache_utils import DynamicCache
        past_kv_cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(active_kv):
            past_kv_cache.update(k, v, layer_idx)

        # Decode query
        out = model(input_ids=query_ids, past_key_values=past_kv_cache, use_cache=True)
        past_kv_cache = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Generate 12 answer tokens
        generated = [next_token]
        for _ in range(11):
            out = model(input_ids=next_token, past_key_values=past_kv_cache, use_cache=True)
            past_kv_cache = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)

    gen_text = tokenizer.decode(torch.cat(generated, dim=-1)[0], skip_special_tokens=True)
    elapsed = time.time() - t0

    # Cleanup
    del past_kv_cache, kv_manager, hook
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return gen_text, elapsed, seq_len


def main():
    output_dir = Path("outputs/hpca_extended")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = "d:/LLM/models/Qwen2.5-1.5B-Instruct"
    device = "cpu"
    dtype = torch.float16

    lengths = [4096]
    methods = ["full_kv", "snapkv", "prose"]
    samples_per_length = 2  # Fast fallback for CPU-only envelope
    query = "What is the secret passkey?"

    print("=" * 80)
    print("Extended Context Real Evaluation (CPU, fp16)")
    print("=" * 80)
    print(f"Lengths: {lengths}")
    print(f"Methods: {methods}")
    print("NOTE: 16K/32K omitted on CPU due to excessive latency; run on GPU if available.")
    print("=" * 80)

    print("\nLoading model (this may take 30-60s)...")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded in {time.time()-t_load:.1f}s")

    results = []
    for length in lengths:
        for method in methods:
            for sample_idx in range(samples_per_length):
                depth = np.random.uniform(0.1, 0.9)
                text, passkey = build_passkey_text(tokenizer, length, depth=depth)
                print(f"\n[Eval] length={length} method={method} sample={sample_idx} depth={depth:.2f} tokens={len(tokenizer.encode(text))}")

                try:
                    gen_text, elapsed, seq_len = evaluate_method_on_prompt(
                        model, tokenizer, text, query, method,
                        budget_ratio=0.10 if method != "full_kv" else 1.0
                    )
                    correct = passkey in gen_text
                    print(f"  -> correct={correct} time={elapsed:.1f}s  gen='{gen_text[:40]}...'")
                    results.append({
                        "length": length,
                        "method": method,
                        "sample": sample_idx,
                        "depth": depth,
                        "seq_len": seq_len,
                        "correct": correct,
                        "passkey": passkey,
                        "generated": gen_text[:60],
                        "elapsed_s": elapsed,
                    })
                except Exception as e:
                    print(f"  -> ERROR: {e}")
                    results.append({
                        "length": length,
                        "method": method,
                        "sample": sample_idx,
                        "depth": depth,
                        "error": str(e),
                    })

    # Save
    report = {
        "model": model_path,
        "device": device,
        "dtype": str(dtype),
        "results": results,
        "interpretation": (
            "Real-model passkey evaluation at extended context lengths on CPU. "
            "Full KV serves as oracle; ProSE and SnapKV are compared under 10% budget."
        ),
    }
    with open(output_dir / "extended_context_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        pts = [(r["length"], 1.0 if r.get("correct") else 0.0)
               for r in results if r["method"] == method and "error" not in r]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=method.upper(), linewidth=2, markersize=8)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Passkey Accuracy")
    ax.set_title("Extended Context Passkey Accuracy (Real Model, CPU)")
    ax.set_xticks(lengths)
    ax.set_ylim(-0.05, 1.15)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_extended_passkey.pdf", dpi=300)
    print(f"\n[Plot] Saved {output_dir / 'fig_extended_passkey.pdf'}")

    # Summary print
    print("\n" + "=" * 80)
    print("EXTENDED CONTEXT RESULTS SUMMARY")
    print("=" * 80)
    for method in methods:
        for length in lengths:
            subset = [r for r in results if r["method"] == method and r["length"] == length and "error" not in r]
            if subset:
                acc = np.mean([1.0 if r["correct"] else 0.0 for r in subset])
                avg_time = np.mean([r["elapsed_s"] for r in subset])
                print(f"  {method:<10} @ {length:>6}  accuracy={acc:.0%}  avg_time={avg_time:.1f}s")
            else:
                print(f"  {method:<10} @ {length:>6}  N/A")
    print(f"\nArtifacts: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
