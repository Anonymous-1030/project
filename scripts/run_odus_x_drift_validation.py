#!/usr/bin/env python3
"""
ODUS-X Drift-Heavy Validation.

Tests reactive gating by using a multi-hop prompt:
  Step 1: query about early section (stable mode warms up)
  Step 2: query about middle section (mixed transition)
  Step 3: query about needle at random depth (reactive mode)

Measures whether adaptive_gating outperforms fixed_stable and
fixed_reactive baselines when drift is high.
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

from src.core_types import ChunkMetadata, QueryContext, ChunkState
from src.promotion.scorer.odus_x import AdaptiveGatingScorer
from src.promotion.scorer.odus import RuntimeFeatures

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def _cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def extract_once(model, tokenizer, text, device):
    """Memory-safe: hooks only, no output_hidden_states."""
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


def build_drift_prompt(tokenizer, target_length):
    """Build a 3-hop prompt with needle at random depth."""
    sections = [
        "The early Renaissance was marked by a revival of classical learning and values. "
        "Artists like Leonardo da Vinci and Michelangelo transformed European culture. "
        "The printing press accelerated the spread of knowledge across the continent. ",
        "The Industrial Revolution brought mechanization and urbanization. "
        "Steam engines and factories restructured society and labor patterns. "
        "Economic theories evolved to explain capital accumulation and market dynamics. ",
        "The digital age began with the invention of the transistor and integrated circuit. "
        "Computers and the internet created a globally connected information society. "
        "Artificial intelligence now promises to reshape work and creativity once again. ",
    ]
    base_text = " ".join(sections)
    base_tok = len(tokenizer.encode(base_text))
    repeats = max(1, target_length // base_tok + 3)
    full_base = (base_text + " ") * repeats

    passkey = str(np.random.randint(100000, 999999))
    needle = f" The secret passkey is {passkey}. Remember this number. "
    depth = np.random.uniform(0.2, 0.8)
    pos = int(len(full_base) * depth)
    text = full_base[:pos] + needle + full_base[pos:]

    # 3-hop query sequence
    q1 = "\nQ1: What invention accelerated knowledge spread in the early Renaissance?\nA1:"
    q2 = "\nQ2: What technology restructured labor patterns during the Industrial Revolution?\nA2:"
    q3 = f"\nQ3: What is the secret passkey mentioned above?\nA3:"
    full_text = text + q1 + q2 + q3

    tokens = tokenizer.encode(full_text, truncation=True, max_length=target_length)
    full_text = tokenizer.decode(tokens, skip_special_tokens=True)

    # Locate needle
    needle_text = f"secret passkey is {passkey}"
    nc = full_text.find(needle_text)
    if nc < 0:
        nc = len(full_text) // 2
    needle_tok_pos = len(tokenizer.encode(full_text[:nc]))

    # Locate query positions (approximate by text search)
    q1_pos = full_text.find("Q1:")
    q2_pos = full_text.find("Q2:")
    q3_pos = full_text.find("Q3:")
    if q1_pos < 0: q1_pos = len(full_text) - 300
    if q2_pos < 0: q2_pos = len(full_text) - 200
    if q3_pos < 0: q3_pos = len(full_text) - 100

    query_positions = [
        len(tokenizer.encode(full_text[:q1_pos])),
        len(tokenizer.encode(full_text[:q2_pos])),
        len(tokenizer.encode(full_text[:q3_pos])),
    ]
    return full_text, needle_tok_pos, query_positions


def simulate_multi_step(hidden_states, last_attn, seq_len, needle_tok_pos, query_positions,
                        chunk_size, scorer, policy_name):
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    boundaries = [(i * chunk_size, min((i + 1) * chunk_size, seq_len)) for i in range(num_chunks)]
    budget_K = max(1, int(num_chunks * 0.1))
    needle_chunk_id = min(needle_tok_pos // chunk_size, num_chunks - 1)

    all_chunks = {}
    for i, (s, e) in enumerate(boundaries):
        cm = ChunkMetadata(
            chunk_id=str(i), request_id="req_1",
            token_start=s, token_end=e,
            position_ratio=i / max(num_chunks - 1, 1),
            num_tokens=e - s, logical_bytes=(e - s) * 1024,
            state=ChunkState.ACTIVE,
        )
        all_chunks[cm.chunk_id] = cm

    recoveries = []
    drift_levels = []

    for step_idx, q_idx in enumerate(query_positions):
        q_idx = min(q_idx, seq_len - 1)
        query_embed = hidden_states[q_idx]

        attn_1d = last_attn[:, q_idx, :].mean(axis=0)
        chunk_attn = np.zeros(num_chunks)
        for i, (s, e) in enumerate(boundaries):
            chunk_attn[i] = attn_1d[s:e].sum()

        gold_topk = set(np.argsort(-chunk_attn)[:budget_K].tolist())

        query = QueryContext(
            request_id="req_1", step=step_idx,
            query_signature=query_embed.astype(np.float32),
        )

        scores = []
        for i in range(num_chunks):
            chunk = all_chunks[str(i)]
            s, e = boundaries[i]
            chunk_embed = hidden_states[s:e].mean(axis=0)
            sim = _cosine_sim(chunk_embed, query_embed)

            if policy_name == "fixed_similarity":
                scores.append((i, sim)); continue
            if policy_name == "fixed_recency":
                recency = 1.0 / (1.0 + abs(i - (q_idx // chunk_size)) / 5.0)
                scores.append((i, recency)); continue

            recency = 1.0 / (1.0 + abs(i - (q_idx // chunk_size)) / 5.0)
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

        drift_levels.append(round(scorer.drift_level, 3))
        selected = set([str(i) for i, _ in sorted(scores, key=lambda x: -x[1])[:budget_K]])
        recovery = len(selected & {str(g) for g in gold_topk}) / len(gold_topk)
        recoveries.append(recovery)

        # Update metadata
        for sid in selected:
            all_chunks[sid].promoted_count += 1
            all_chunks[sid].last_promotion_step = step_idx
        for gid in gold_topk:
            all_chunks[str(gid)].access_count += 1
            all_chunks[str(gid)].last_access_step = step_idx

    return recoveries, drift_levels, needle_chunk_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="outputs/hpca_odus_v2")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch_dtype,
        device_map=device, attn_implementation="eager", local_files_only=True,
    )
    model.eval()
    print("Model loaded.")

    policies = {
        "fixed_similarity": AdaptiveGatingScorer(),
        "fixed_recency": AdaptiveGatingScorer(),
        "fixed_stable": AdaptiveGatingScorer(force_mode="stable"),
        "fixed_reactive": AdaptiveGatingScorer(force_mode="reactive"),
        "adaptive_gating": AdaptiveGatingScorer(),
    }

    results = {name: {"step0": [], "step1": [], "step2": []} for name in policies}
    drift_agg = []

    print(f"\n=== Drift-heavy validation @ {args.length} ===")
    for sample_idx in range(args.num_samples):
        text, needle_tok_pos, query_positions = build_drift_prompt(tokenizer, args.length)
        hidden_states, last_attn, seq_len = extract_once(model, tokenizer, text, device)

        for policy_name, scorer in policies.items():
            recs, drifts, nid = simulate_multi_step(
                hidden_states, last_attn, seq_len, needle_tok_pos,
                query_positions, args.chunk_size, scorer, policy_name,
            )
            results[policy_name]["step0"].append(recs[0])
            results[policy_name]["step1"].append(recs[1])
            results[policy_name]["step2"].append(recs[2])
            if policy_name == "adaptive_gating":
                drift_agg.append(drifts)

        if device == "cuda":
            torch.cuda.empty_cache()

    agg = {}
    for policy_name, per_step in results.items():
        agg[policy_name] = {
            step: {
                "mean_recovery": round(float(np.mean(vals)), 3),
                "std": round(float(np.std(vals)), 3),
            }
            for step, vals in per_step.items()
        }
        print(f"\n{policy_name}:")
        print(json.dumps(agg[policy_name], indent=2))

    # Drift summary
    mean_drift = [np.mean([d[i] for d in drift_agg]) for i in range(3)] if drift_agg else [0, 0, 0]
    print(f"\nMean drift levels: step0={mean_drift[0]:.3f}, step1={mean_drift[1]:.3f}, step2={mean_drift[2]:.3f}")

    report = {
        "config": vars(args),
        "aggregate": agg,
        "drift_levels": [round(float(x), 3) for x in mean_drift],
    }
    out_file = output_dir / "odus_x_drift_validation.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Saved] {out_file}")


if __name__ == "__main__":
    main()
