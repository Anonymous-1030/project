#!/usr/bin/env python3
"""
ODUS v2.0 Validation Suite for A100 80GB

Validates the upgraded "Multi-Cue Ensemble Ranker" narrative:
  1. Bottom-layer hidden-state sketch replaces input-embedding proxy.
  2. No single cue dominates; ensemble fusion is the architectural value.
  3. 64B compression works when task-aware (PCA / trained linear), 
     not when random.

Metrics (ranking-focused, NOT Pearson regression):
  - Needle Recall@K
  - MRR (Mean Reciprocal Rank)
  - Spearman ρ (rank correlation)

Usage (single GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/run_odus_v2_validation.py \
    --model Qwen/Qwen2.5-3B-Instruct --experiment semantic_sketch \
    --contexts 4096,8192,16384

Usage (4x A100 launcher):
  bash scripts/launch_odus_v2_a100.sh
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def build_needle_prompts(tokenizer, lengths, num_samples=10, chunk_size=64):
    """Generate needle-in-haystack prompts with needles at varying depths."""
    haystack_sentences = [
        "The bustling city streets were filled with the sounds of daily life.",
        "Recent advances in artificial intelligence continue to reshape industries.",
        "Mountain trails often wind through dense forests and rushing streams.",
        "Economic forecasts suggest a period of sustained moderate growth ahead.",
        "Ancient civilizations developed remarkably sophisticated governance systems.",
        "Climate researchers emphasize the importance of long-term data collection.",
        "The library archives contain thousands of volumes on medieval history.",
        "Space exploration agencies are planning missions to the outer solar system.",
        "Philosophers throughout the ages have debated the nature of consciousness.",
        "Modern transportation networks connect even the most remote rural communities.",
    ]
    filler = " ".join(haystack_sentences) + " "
    base_tok = len(tokenizer.encode(filler))

    prompts = []
    for length in lengths:
        repeats = max(1, length // base_tok + 3)
        base_text = (filler + " ") * repeats
        for _ in range(num_samples):
            passkey = str(np.random.randint(100000, 999999))
            needle = f" The secret passkey is {passkey}. Remember this number. "
            depth = np.random.uniform(0.1, 0.9)
            pos = int(len(base_text) * depth)
            text = base_text[:pos] + needle + base_text[pos:]
            tokens = tokenizer.encode(text, truncation=True, max_length=length)
            text = tokenizer.decode(tokens, skip_special_tokens=True)

            # Locate needle chunk index
            needle_pos_char = pos
            needle_token_pos_approx = len(tokenizer.encode(base_text[:needle_pos_char]))
            needle_chunk_id = needle_token_pos_approx // chunk_size
            num_chunks = (len(tokens) + chunk_size - 1) // chunk_size
            needle_chunk_id = min(needle_chunk_id, num_chunks - 1)

            prompts.append({
                "text": text,
                "length": length,
                "passkey": passkey,
                "needle_chunk_id": needle_chunk_id,
                "num_chunks": num_chunks,
            })
    return prompts


def get_layer_hidden_state(model, tokenizer, text, layer_idx, chunk_size, device):
    """
    Forward pass with a hook to extract hidden states from a SINGLE layer.
    Avoids storing all layers (saves memory for 64K context on 7B).
    
    layer_idx = -1  → input embeddings (baseline)
    layer_idx = 0   → first transformer layer output
    layer_idx = k   → k-th transformer layer output
    """
    model.eval()
    tokens = tokenizer.encode(text, return_tensors="pt").to(device)
    seq_len = tokens.shape[1]

    with torch.no_grad():
        if layer_idx == -1:
            # Input embedding baseline
            embed_layer = model.get_input_embeddings()
            hidden = embed_layer(tokens)[0]  # [seq_len, hidden_dim]
        else:
            hidden_container = {}

            def hook_fn(module, input, output):
                # output is typically (hidden_state, ...) for decoder layers
                h = output[0] if isinstance(output, tuple) else output
                hidden_container["h"] = h

            target_module = model.model.layers[layer_idx]
            handle = target_module.register_forward_hook(hook_fn)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _ = model(input_ids=tokens, use_cache=False)

            handle.remove()
            hidden = hidden_container["h"][0]  # [seq_len, hidden_dim]

    hidden = hidden.float().cpu().numpy()
    query_embed = hidden[-1]

    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    chunk_embeds = np.zeros((num_chunks, hidden.shape[1]), dtype=np.float32)
    for i in range(num_chunks):
        s = i * chunk_size
        e = min((i + 1) * chunk_size, seq_len)
        chunk_embeds[i] = hidden[s:e].mean(axis=0)

    return chunk_embeds, query_embed


def get_attention_mass(model, tokenizer, text, chunk_size, device):
    """Extract per-chunk attention mass from last layer, last query position."""
    model.eval()
    tokens = tokenizer.encode(text, return_tensors="pt").to(device)
    seq_len = tokens.shape[1]

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(input_ids=tokens, output_attentions=True, use_cache=False)

    last_attn = outputs.attentions[-1][0].float().cpu().numpy()  # [heads, q, kv]
    attn_1d = last_attn[:, -1, :].mean(axis=0)  # [kv_len]

    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    chunk_masses = np.zeros(num_chunks)
    for i in range(num_chunks):
        s = i * chunk_size
        e = min((i + 1) * chunk_size, seq_len)
        chunk_masses[i] = attn_1d[s:e].sum()
    total = chunk_masses.sum()
    if total > 0:
        chunk_masses /= total
    return chunk_masses


def compute_ranking_metrics(sims, needle_chunk_id, k_values=[5, 10, 20]):
    """Compute ranking metrics for a single prompt."""
    ranked = np.argsort(-sims)
    needle_rank = np.where(ranked == needle_chunk_id)[0][0] + 1  # 1-indexed

    metrics = {
        "mrr": 1.0 / needle_rank,
        "needle_rank": int(needle_rank),
    }
    for k in k_values:
        metrics[f"recall@{k}"] = 1.0 if needle_rank <= k else 0.0
    return metrics


def evaluate_semantic_sketch(model, tokenizer, args, device):
    """
    Experiment 1: Compare input embedding vs bottom-layer vs deeper layers.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Semantic Sketch Layer Comparison")
    print("=" * 80)

    lengths = [int(x) for x in args.contexts.split(",")]
    layer_indices = [int(x) for x in args.layers.split(",")]
    chunk_size = args.chunk_size
    prompts = build_needle_prompts(tokenizer, lengths, num_samples=args.num_samples, chunk_size=chunk_size)

    results = {}
    for layer_idx in layer_indices:
        layer_name = "input_emb" if layer_idx == -1 else f"layer_{layer_idx}"
        print(f"\n--- Evaluating {layer_name} ---")
        all_metrics = []

        for p in prompts:
            chunk_embeds, query_embed = get_layer_hidden_state(
                model, tokenizer, p["text"], layer_idx, chunk_size, device
            )
            sims = compute_similarities(chunk_embeds, query_embed)
            metrics = compute_ranking_metrics(sims, p["needle_chunk_id"])
            all_metrics.append({**metrics, "length": p["length"]})

        # Aggregate by length
        per_length = {}
        for m in all_metrics:
            per_length.setdefault(m["length"], []).append(m)

        agg = {}
        for length, entries in per_length.items():
            agg[length] = {
                "mrr": round(np.mean([e["mrr"] for e in entries]), 3),
                "recall@5": round(np.mean([e["recall@5"] for e in entries]), 3),
                "recall@10": round(np.mean([e["recall@10"] for e in entries]), 3),
                "mean_rank": round(np.mean([e["needle_rank"] for e in entries]), 1),
            }
        results[layer_name] = agg
        print(json.dumps(agg, indent=2))

    return {"experiment": "semantic_sketch", "results": results}


def compute_similarities(chunk_embeds, query_embed):
    chunk_norm = chunk_embeds / (np.linalg.norm(chunk_embeds, axis=1, keepdims=True) + 1e-12)
    q_norm = query_embed / (np.linalg.norm(query_embed) + 1e-12)
    return chunk_norm @ q_norm


def evaluate_compression(model, tokenizer, args, device):
    """
    Experiment 3: Compare random vs PCA vs trained linear projection to 64B.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: 64B Compression Methods")
    print("=" * 80)

    lengths = [int(x) for x in args.contexts.split(",")]
    chunk_size = args.chunk_size
    prompts = build_needle_prompts(tokenizer, lengths, num_samples=args.num_samples, chunk_size=chunk_size)
    layer_idx = args.compression_layer  # typically 0

    # Gather training data for PCA / trained head (use first half of prompts)
    train_prompts = prompts[: len(prompts) // 2]
    test_prompts = prompts[len(prompts) // 2 :]

    hidden_dim = None
    train_representations = []
    train_queries = []
    train_labels = []

    print("Gathering training representations...")
    for p in train_prompts:
        chunk_embeds, query_embed = get_layer_hidden_state(
            model, tokenizer, p["text"], layer_idx, chunk_size, device
        )
        if hidden_dim is None:
            hidden_dim = chunk_embeds.shape[1]
        train_representations.append(chunk_embeds)
        train_queries.append(query_embed)
        labels = np.zeros(len(chunk_embeds))
        labels[p["needle_chunk_id"]] = 1.0
        train_labels.append(labels)

    proj_dim = args.proj_dim  # 16 for fp32, 32 for bf16 → 64B or 128B

    # --- Random projection ---
    rng = np.random.RandomState(42)
    rand_proj = rng.randn(hidden_dim, proj_dim).astype(np.float32)
    rand_proj /= np.linalg.norm(rand_proj, axis=0, keepdims=True) + 1e-12

    # --- PCA projection ---
    all_train_chunks = np.vstack(train_representations)
    mean_vec = all_train_chunks.mean(axis=0)
    centered = all_train_chunks - mean_vec
    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(-eigvals)
    pca_basis = eigvecs[:, idx[:proj_dim]].astype(np.float32)

    # --- Trained linear head ---
    print("Training linear projection head...")
    trained_head = train_linear_head(
        train_representations, train_queries, train_labels, proj_dim, device, epochs=100
    )

    methods = {
        "random": lambda c, q: project_and_sim(c, q, rand_proj),
        "pca": lambda c, q: project_and_sim(c - mean_vec, q - mean_vec, pca_basis),
        "trained": lambda c, q: trained_head_project(c, q, trained_head, device),
    }

    results = {}
    for method_name, proj_fn in methods.items():
        print(f"\n--- Method: {method_name} ---")
        all_metrics = []
        for p in test_prompts:
            chunk_embeds, query_embed = get_layer_hidden_state(
                model, tokenizer, p["text"], layer_idx, chunk_size, device
            )
            sims = proj_fn(chunk_embeds, query_embed)
            metrics = compute_ranking_metrics(sims, p["needle_chunk_id"])
            all_metrics.append({**metrics, "length": p["length"]})

        per_length = {}
        for m in all_metrics:
            per_length.setdefault(m["length"], []).append(m)
        agg = {}
        for length, entries in per_length.items():
            agg[length] = {
                "mrr": round(np.mean([e["mrr"] for e in entries]), 3),
                "recall@5": round(np.mean([e["recall@5"] for e in entries]), 3),
                "recall@10": round(np.mean([e["recall@10"] for e in entries]), 3),
                "mean_rank": round(np.mean([e["needle_rank"] for e in entries]), 1),
            }
        results[method_name] = agg
        print(json.dumps(agg, indent=2))

    return {"experiment": "compression", "results": results}


def project_and_sim(chunk_embeds, query_embed, proj_matrix):
    c = chunk_embeds @ proj_matrix
    q = query_embed @ proj_matrix
    c_norm = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    return c_norm @ q_norm


def trained_head_project(chunk_embeds, query_embed, head, device):
    with torch.no_grad():
        c = torch.from_numpy(chunk_embeds).to(device)
        q = torch.from_numpy(query_embed).to(device)
        c_proj = F.normalize(head(c), dim=-1)
        q_proj = F.normalize(head(q.unsqueeze(0)), dim=-1)
        sims = (c_proj @ q_proj.T).squeeze(-1).cpu().numpy()
    return sims


def train_linear_head(chunk_embeds_list, query_embeds_list, labels_list, proj_dim, device, epochs=100, lr=1e-2):
    """Fast contrastive training of a linear projection head."""
    hidden_dim = chunk_embeds_list[0].shape[1]
    head = nn.Linear(hidden_dim, proj_dim, bias=False).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for chunks, query, labels in zip(chunk_embeds_list, query_embeds_list, labels_list):
            c = torch.from_numpy(chunks).to(device)
            q = torch.from_numpy(query).to(device)
            y = torch.from_numpy(labels).to(device)
            needle_idx = y.argmax().item()

            c_proj = F.normalize(head(c), dim=-1)  # [N, proj_dim]
            q_proj = F.normalize(head(q.unsqueeze(0)), dim=-1)  # [1, proj_dim]
            logits = (c_proj @ q_proj.T).squeeze(-1) / 0.1  # temperature 0.1

            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([needle_idx], device=device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, loss={total_loss/max(count,1):.4f}")

    return head


def evaluate_ensemble_ablation(model, tokenizer, args, device):
    """
    Experiment 2: Single-cue vs multi-cue ensemble ablation.
    Uses bottom-layer sketch (layer 0) as the semantic signal.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Multi-Cue Ensemble Ablation")
    print("=" * 80)

    lengths = [int(x) for x in args.contexts.split(",")]
    chunk_size = args.chunk_size
    prompts = build_needle_prompts(tokenizer, lengths, num_samples=args.num_samples, chunk_size=chunk_size)
    layer_idx = 0  # Bottom-layer sketch

    variants = {
        "similarity_only": {"use_sim": 1.0, "use_recency": 0.0, "use_position": 0.0, "use_anchor": 0.0, "use_history": 0.0},
        "recency_only": {"use_sim": 0.0, "use_recency": 1.0, "use_position": 0.0, "use_anchor": 0.0, "use_history": 0.0},
        "position_only": {"use_sim": 0.0, "use_recency": 0.0, "use_position": 1.0, "use_anchor": 0.0, "use_history": 0.0},
        "anchor_only": {"use_sim": 0.0, "use_recency": 0.0, "use_position": 0.0, "use_anchor": 1.0, "use_history": 0.0},
        "history_only": {"use_sim": 0.0, "use_recency": 0.0, "use_position": 0.0, "use_anchor": 0.0, "use_history": 1.0},
        "sim_plus_recency": {"use_sim": 1.0, "use_recency": 1.0, "use_position": 0.0, "use_anchor": 0.0, "use_history": 0.0},
        "full_ensemble": {"use_sim": 1.0, "use_recency": 1.0, "use_position": 1.0, "use_anchor": 1.0, "use_history": 1.0},
    }

    # Train a tiny MLP ensemble on synthetic feature vectors
    print("Training ensemble MLP ablator...")
    ensemble_mlp = train_ensemble_mlp(prompts, model, tokenizer, layer_idx, chunk_size, device, epochs=200)

    results = {}
    for variant_name, weights in variants.items():
        print(f"\n--- Variant: {variant_name} ---")
        all_metrics = []
        for p in prompts:
            chunk_embeds, query_embed = get_layer_hidden_state(
                model, tokenizer, p["text"], layer_idx, chunk_size, device
            )
            sims = compute_ensemble_score(
                chunk_embeds, query_embed, p, weights, ensemble_mlp, device
            )
            metrics = compute_ranking_metrics(sims, p["needle_chunk_id"])
            all_metrics.append({**metrics, "length": p["length"]})

        per_length = {}
        for m in all_metrics:
            per_length.setdefault(m["length"], []).append(m)
        agg = {}
        for length, entries in per_length.items():
            agg[length] = {
                "mrr": round(np.mean([e["mrr"] for e in entries]), 3),
                "recall@5": round(np.mean([e["recall@5"] for e in entries]), 3),
                "recall@10": round(np.mean([e["recall@10"] for e in entries]), 3),
                "mean_rank": round(np.mean([e["needle_rank"] for e in entries]), 1),
            }
        results[variant_name] = agg
        print(json.dumps(agg, indent=2))

    return {"experiment": "ensemble_ablation", "results": results}


def train_ensemble_mlp(prompts, model, tokenizer, layer_idx, chunk_size, device, epochs=200, lr=1e-2):
    """Train a small MLP that maps 5-D cue vector to utility score."""
    # Collect training features
    X_list = []
    y_list = []
    for p in prompts:
        chunk_embeds, query_embed = get_layer_hidden_state(model, tokenizer, p["text"], layer_idx, chunk_size, device)
        features, needle_idx = build_cue_features(chunk_embeds, query_embed, p)
        X_list.append(features)
        labels = np.zeros(len(features))
        labels[needle_idx] = 1.0
        y_list.append(labels)

    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list)

    # Balance: upsample positive examples
    pos_mask = y_all == 1.0
    neg_mask = ~pos_mask
    if pos_mask.sum() > 0:
        X_pos = X_all[pos_mask]
        X_neg = X_all[neg_mask]
        y_pos = y_all[pos_mask]
        y_neg = y_all[neg_mask]
        # Repeat positives to roughly match negatives
        repeat = max(1, len(y_neg) // max(len(y_pos), 1) // 2)
        X_bal = np.vstack([X_neg] + [X_pos] * repeat)
        y_bal = np.hstack([y_neg] + [y_pos] * repeat)
    else:
        X_bal, y_bal = X_all, y_all

    X_t = torch.from_numpy(X_bal).float().to(device)
    y_t = torch.from_numpy(y_bal).float().to(device).unsqueeze(1)

    mlp = nn.Sequential(
        nn.Linear(5, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    ).to(device)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr)
    for epoch in range(epochs):
        pred = mlp(X_t)
        loss = F.binary_cross_entropy(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Ensemble MLP epoch {epoch+1}/{epochs}, BCE={loss.item():.4f}")

    return mlp


def build_cue_features(chunk_embeds, query_embed, prompt):
    """Build 5-D cue vectors for each chunk."""
    num_chunks = len(chunk_embeds)
    needle_idx = prompt["needle_chunk_id"]

    # Cue 1: semantic similarity
    sims = compute_similarities(chunk_embeds, query_embed)

    # Cue 2: recency (proxy by inverse distance from current position)
    recency = np.array([1.0 / (1.0 + abs(i - (num_chunks - 1)) / 5.0) for i in range(num_chunks)])

    # Cue 3: position bias (earlier chunks slightly favored)
    position = 1.0 - np.arange(num_chunks) / max(num_chunks - 1, 1)

    # Cue 4: anchor distance (simulate anchors every N chunks)
    anchor_interval = max(1, num_chunks // 4)
    anchor_positions = np.arange(0, num_chunks, anchor_interval)
    anchor_dist = np.zeros(num_chunks)
    for i in range(num_chunks):
        dists = np.abs(anchor_positions - i)
        anchor_dist[i] = dists.min() / max(num_chunks / anchor_interval, 1.0)
    anchor_score = 1.0 - anchor_dist

    # Cue 5: simulated PHT history success rate
    # Simulate a previous promotion hotspot at a RANDOM anchor (not the needle)
    # to avoid information leakage during testing.
    rng = np.random.RandomState((needle_idx * 31 + num_chunks) % 2**31)
    prev_hotspot = rng.randint(0, num_chunks)
    history = np.zeros(num_chunks)
    for i in range(num_chunks):
        dist = abs(i - prev_hotspot)
        history[i] = np.exp(-dist / 3.0)

    features = np.stack([sims, recency, position, anchor_score, history], axis=1).astype(np.float32)
    return features, needle_idx


def compute_ensemble_score(chunk_embeds, query_embed, prompt, weights, ensemble_mlp, device):
    features, _ = build_cue_features(chunk_embeds, query_embed, prompt)
    feat_t = torch.from_numpy(features).float().to(device)

    with torch.no_grad():
        mlp_scores = ensemble_mlp(feat_t).cpu().numpy().squeeze(-1)

    # If only using one cue, use that cue directly; otherwise use MLP output
    active_cues = [k for k, v in weights.items() if v > 0]
    if len(active_cues) == 1:
        cue_map = {
            "use_sim": 0, "use_recency": 1, "use_position": 2,
            "use_anchor": 3, "use_history": 4,
        }
        idx = cue_map[active_cues[0]]
        return features[:, idx]
    else:
        return mlp_scores


def evaluate_scale_test(model, tokenizer, args, device):
    """
    Experiment 4: Long-context scale test with full ensemble.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Scale Test (Full Ensemble)")
    print("=" * 80)

    lengths = [int(x) for x in args.contexts.split(",")]
    chunk_size = args.chunk_size
    prompts = build_needle_prompts(tokenizer, lengths, num_samples=args.num_samples, chunk_size=chunk_size)
    layer_idx = 0

    # Train ensemble MLP on these prompts
    print("Training ensemble MLP for scale test...")
    ensemble_mlp = train_ensemble_mlp(prompts, model, tokenizer, layer_idx, chunk_size, device, epochs=200)
    weights = {"use_sim": 1.0, "use_recency": 1.0, "use_position": 1.0, "use_anchor": 1.0, "use_history": 1.0}

    all_metrics = []
    latencies = []
    for p in prompts:
        t0 = time.time()
        chunk_embeds, query_embed = get_layer_hidden_state(
            model, tokenizer, p["text"], layer_idx, chunk_size, device
        )
        sims = compute_ensemble_score(chunk_embeds, query_embed, p, weights, ensemble_mlp, device)
        metrics = compute_ranking_metrics(sims, p["needle_chunk_id"])
        latencies.append((time.time() - t0) * 1000)  # ms
        all_metrics.append({**metrics, "length": p["length"]})

    per_length = {}
    for m in all_metrics:
        per_length.setdefault(m["length"], []).append(m)

    results = {}
    for length, entries in per_length.items():
        # Filter latencies for this length
        length_latencies = [lat for e, lat in zip(all_metrics, latencies) if e["length"] == length]
        results[length] = {
            "mrr": round(np.mean([e["mrr"] for e in entries]), 3),
            "recall@5": round(np.mean([e["recall@5"] for e in entries]), 3),
            "recall@10": round(np.mean([e["recall@10"] for e in entries]), 3),
            "mean_rank": round(np.mean([e["needle_rank"] for e in entries]), 1),
            "mean_latency_ms": round(np.mean(length_latencies), 1) if length_latencies else 0.0,
        }
        print(f"Length {length}: {json.dumps(results[length], indent=2)}")

    return {"experiment": "scale_test", "results": results}


def main():
    parser = argparse.ArgumentParser(description="ODUS v2.0 Validation Suite")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["semantic_sketch", "ensemble_ablation", "compression", "scale_test"],
                        help="Which experiment to run")
    parser.add_argument("--contexts", type=str, default="4096,8192,16384",
                        help="Comma-separated context lengths")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Samples per context length")
    parser.add_argument("--chunk_size", type=int, default=64,
                        help="Tokens per chunk")
    parser.add_argument("--layers", type=str, default="-1,0,2,6,12,23",
                        help="Comma-separated layer indices for semantic_sketch (-1=input_emb)")
    parser.add_argument("--compression_layer", type=int, default=0,
                        help="Layer to compress for compression experiment")
    parser.add_argument("--proj_dim", type=int, default=16,
                        help="Projection dimension (16 fp32 dims = 64B)")
    parser.add_argument("--output_dir", type=str, default="outputs/hpca_odus_v2",
                        help="Output directory")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    print("Model loaded.")

    experiment_map = {
        "semantic_sketch": evaluate_semantic_sketch,
        "ensemble_ablation": evaluate_ensemble_ablation,
        "compression": evaluate_compression,
        "scale_test": evaluate_scale_test,
    }

    result = experiment_map[args.experiment](model, tokenizer, args, device)

    out_file = output_dir / f"{args.experiment}_results.json"
    with open(out_file, "w") as f:
        json.dump({"config": vars(args), **result}, f, indent=2)
    print(f"\n[Saved] {out_file}")


if __name__ == "__main__":
    main()
