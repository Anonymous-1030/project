#!/usr/bin/env python3
"""
ODUS-X A100 Validation Script — Memory-Optimized Version.

Critical fix: forward pass is performed ONCE per prompt.
Last-layer attention is captured via a single hook instead of
output_attentions=True (which stores all 36 layers and OOMs at 8K).
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from src.core_types import ChunkMetadata, QueryContext, ChunkState, ChunkTier
from src.promotion.scorer.odus_x import AdaptiveGatingScorer
from src.promotion.scorer.odus import RuntimeFeatures

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_passkey_prompt(tokenizer, target_length: int):
    haystack = (
        "The bustling city streets were filled with the sounds of daily life. "
        "Recent advances in artificial intelligence continue to reshape industries worldwide. "
        "Mountain trails often wind through dense forests and across rushing streams. "
        "Economic forecasts suggest a period of sustained moderate growth ahead. "
        "Ancient civilizations developed remarkably sophisticated governance systems. "
    )
    base_tok = len(tokenizer.encode(haystack))
    repeats = max(1, target_length // base_tok + 3)
    base_text = (haystack + " ") * repeats

    passkey = str(np.random.randint(100000, 999999))
    needle = f" The secret passkey is {passkey}. Remember this number. "
    depth = np.random.uniform(0.1, 0.9)
    pos = int(len(base_text) * depth)
    text = base_text[:pos] + needle + base_text[pos:]

    query = f"\nQuestion: What is the secret passkey?\nAnswer:"
    full_text = text + query

    tokens = tokenizer.encode(full_text, truncation=True, max_length=target_length)
    full_text = tokenizer.decode(tokens, skip_special_tokens=True)

    needle_text = f"secret passkey is {passkey}"
    needle_char_pos = full_text.find(needle_text)
    if needle_char_pos < 0:
        needle_char_pos = len(full_text) // 2

    pre_needle = full_text[:needle_char_pos]
    needle_tok_pos = len(tokenizer.encode(pre_needle))
    return full_text, passkey, needle_tok_pos


def extract_once(model, tokenizer, text, device):
    """
    Memory-safe extraction: use hooks to capture ONLY last-layer hidden state
    and attention, without output_hidden_states or output_attentions.
    This is critical for 7B @ 16K on A100 80GB.
    """
    tokens = tokenizer.encode(text, return_tensors="pt").to(device)
    seq_len = tokens.shape[1]
    hidden_container = {}
    attn_container = {}

    def hidden_hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        hidden_container["hidden"] = h.detach().cpu()

    def attn_hook_fn(module, input, output):
        if isinstance(output, tuple) and len(output) >= 2:
            attn = output[1]
            if attn is not None and isinstance(attn, torch.Tensor):
                attn_container["attn"] = attn.detach().cpu()

    handle_h = model.model.layers[-1].register_forward_hook(hidden_hook_fn)
    handle_a = model.model.layers[-1].self_attn.register_forward_hook(attn_hook_fn)

    with torch.no_grad():
        _ = model(
            input_ids=tokens,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )

    handle_h.remove()
    handle_a.remove()

    last_hidden = hidden_container["hidden"][0].float().cpu().numpy()
    last_attn = attn_container["attn"][0].float().cpu().numpy()
    return last_hidden, last_attn, seq_len


def simulate_step_selection(
    hidden_states,
    last_attn,
    seq_len,
    needle_tok_pos,
    chunk_size,
    scorer,
    policy_name,
    num_simulated_steps=3,
):
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    boundaries = [(i * chunk_size, min((i + 1) * chunk_size, seq_len)) for i in range(num_chunks)]
    budget_K = max(1, int(num_chunks * 0.1))

    all_chunks = {}
    for i, (s, e) in enumerate(boundaries):
        cm = ChunkMetadata(
            chunk_id=str(i),
            request_id="req_1",
            token_start=s,
            token_end=e,
            position_ratio=i / max(num_chunks - 1, 1),
            num_tokens=e - s,
            logical_bytes=(e - s) * 1024,
            state=ChunkState.ACTIVE,
        )
        all_chunks[cm.chunk_id] = cm

    step_recoveries = []
    start_query_idx = max(1, seq_len - num_simulated_steps)

    for step_offset in range(num_simulated_steps):
        query_token_idx = start_query_idx + step_offset
        query_embed = hidden_states[query_token_idx]

        attn_1d = last_attn[:, query_token_idx, :].mean(axis=0)
        chunk_attn = np.zeros(num_chunks)
        for i, (s, e) in enumerate(boundaries):
            chunk_attn[i] = attn_1d[s:e].sum()

        gold_topk = set(np.argsort(-chunk_attn)[:budget_K].tolist())

        query = QueryContext(
            request_id="req_1",
            step=step_offset,
            query_signature=query_embed.astype(np.float32),
        )

        scores = []
        for i in range(num_chunks):
            chunk = all_chunks[str(i)]
            s, e = boundaries[i]
            chunk_embed = hidden_states[s:e].mean(axis=0)
            sim = _cosine_sim(chunk_embed, query_embed)

            if policy_name == "fixed_similarity":
                scores.append((i, sim))
                continue
            if policy_name == "fixed_recency":
                recency = 1.0 / (1.0 + abs(i - (query_token_idx // chunk_size)) / 5.0)
                scores.append((i, recency))
                continue
            if policy_name == "fixed_random":
                scores.append((i, random.random()))
                continue

            recency = 1.0 / (1.0 + abs(i - (query_token_idx // chunk_size)) / 5.0)
            feat = RuntimeFeatures(
                chunk_position=chunk.position_ratio,
                chunk_recency=recency,
                query_chunk_similarity=sim,
                lexical_overlap=0.0,
                distance_to_nearest_anchor=10.0,
                distance_to_promoted=10.0,
                past_promotion_count=chunk.promoted_count,
                past_promotion_success_rate=(
                    min(chunk.access_count / max(chunk.promoted_count, 1), 1.0)
                    if chunk.promoted_count > 0 else 0.0
                ),
            )
            score, _, _ = scorer.score(feat, chunk, query)
            scores.append((i, score))

        selected = set([str(i) for i, _ in sorted(scores, key=lambda x: -x[1])[:budget_K]])
        recovery = len(selected & {str(g) for g in gold_topk}) / len(gold_topk)
        step_recoveries.append(recovery)

        for sid in selected:
            all_chunks[sid].promoted_count += 1
            all_chunks[sid].last_promotion_step = step_offset

        for gid in gold_topk:
            all_chunks[str(gid)].access_count += 1
            all_chunks[str(gid)].last_access_step = step_offset

    return float(np.mean(step_recoveries))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--contexts", type=str, default="4096,8192,16384")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="outputs/hpca_odus_v2")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    print("Model loaded.")

    lengths = [int(x) for x in args.contexts.split(",")]

    policies = {
        "fixed_similarity": AdaptiveGatingScorer(),
        "fixed_recency": AdaptiveGatingScorer(),
        "fixed_random": AdaptiveGatingScorer(),
        "adaptive_gating": AdaptiveGatingScorer(),
        "adaptive_no_drift": AdaptiveGatingScorer(force_mode="mixed"),
        "adaptive_no_similarity": AdaptiveGatingScorer(zero_similarity=True),
        "adaptive_no_pht": AdaptiveGatingScorer(zero_pht=True),
    }

    results = {name: {length: [] for length in lengths} for name in policies}
    drift_records = {length: [] for length in lengths}

    for length in lengths:
        print(f"\n=== Context length {length} ===")
        for sample_idx in range(args.num_samples):
            text, passkey, needle_tok_pos = build_passkey_prompt(tokenizer, length)

            # ONE forward pass per prompt
            hidden_states, last_attn, seq_len = extract_once(model, tokenizer, text, device)

            for policy_name, scorer in policies.items():
                rec = simulate_step_selection(
                    hidden_states, last_attn, seq_len, needle_tok_pos,
                    args.chunk_size, scorer, policy_name,
                )
                results[policy_name][length].append(rec)

            drift_records[length].append(
                round(policies["adaptive_gating"].drift_level, 3)
            )

            # Clear cache between samples
            if device == "cuda":
                torch.cuda.empty_cache()

    agg = {}
    for policy_name, per_length in results.items():
        agg[policy_name] = {}
        for length, recs in per_length.items():
            agg[policy_name][length] = {
                "mean_recovery": round(float(np.mean(recs)), 3),
                "std": round(float(np.std(recs)), 3),
            }
        print(f"\n{policy_name}:")
        print(json.dumps(agg[policy_name], indent=2))

    report = {
        "config": vars(args),
        "aggregate": agg,
        "drift_levels": {k: round(float(np.mean(v)), 3) for k, v in drift_records.items()},
    }

    out_file = output_dir / "odus_x_validation.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Saved] {out_file}")


if __name__ == "__main__":
    main()
