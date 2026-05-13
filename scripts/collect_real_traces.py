"""
Collect Real Attention Traces from Qwen2.5-1.5B on Passkey Tasks.

Runs the model on passkey retrieval tasks and records per-chunk attention
distributions at each generation step.  Output format is compatible with
the existing traces_*.json files used by the ProSE-X evaluation pipeline.

Usage:
    python collect_real_traces.py
"""

import os
os.environ["KMP_AFFINITY"] = "disabled"
os.environ["OMP_NUM_THREADS"] = "4"

import gc
import json
import math
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

# ── Configuration ────────────────────────────────────────────────────

MODEL_PATH = r"d:\LLM\models\Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = Path(r"d:\LLM\outputs\real_traces")

CONTEXT_LENGTHS = [2048, 4096]
CHUNK_SIZE = 256
PASSKEY_POSITIONS = [0.25, 0.5, 0.75]
SAMPLES_PER_CONFIG = 3
MAX_GEN_STEPS = 10
BUDGET_RATIO = 0.1

# ── Filler text (matches PasskeyBenchmark) ───────────────────────────

FILLER_TEXT = (
    "The quick brown fox jumps over the lazy dog. This is filler text used to create "
    "long context lengths for testing. The grass is green and the sky is blue. "
    "Mountains rise in the distance and rivers flow through valleys. "
)


# ── Helpers ──────────────────────────────────────────────────────────

def generate_passkey() -> str:
    return "".join([str(random.randint(0, 9)) for _ in range(5)])


def build_passkey_input(tokenizer, target_length: int, position: float):
    """Build a passkey-in-haystack input and return (input_ids, passkey, passkey_chunk)."""
    passkey = generate_passkey()
    passkey_sentence = f"The pass key is {passkey}. Remember it."

    # Build filler of roughly the right character length
    repeats = (target_length * 6) // len(FILLER_TEXT) + 2  # rough chars-per-token ~4-6
    all_filler = (FILLER_TEXT + " ") * repeats

    insert_pos = int(len(all_filler) * position)
    context = all_filler[:insert_pos] + " " + passkey_sentence + " " + all_filler[insert_pos:]

    # Tokenize and trim
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    if len(context_ids) > target_length:
        context_ids = context_ids[:target_length]

    # Figure out which chunk the passkey lands in
    passkey_token_ids = tokenizer.encode(passkey_sentence, add_special_tokens=False)
    # Search for sub-sequence in context_ids
    passkey_start_tok = None
    for i in range(len(context_ids) - len(passkey_token_ids) + 1):
        if context_ids[i : i + len(passkey_token_ids)] == passkey_token_ids:
            passkey_start_tok = i
            break
    if passkey_start_tok is None:
        passkey_start_tok = int(len(context_ids) * position)
    passkey_chunk = passkey_start_tok // CHUNK_SIZE

    # Append query
    query = "What is the pass key? The pass key is"
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    full_ids = context_ids + query_ids

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = torch.tensor([full_ids], dtype=torch.long).to(device)
    return input_ids, passkey, passkey_chunk, len(context_ids)


def aggregate_chunk_attention(attentions, seq_len: int, chunk_size: int):
    """
    Aggregate raw attention tensors to per-chunk masses.

    attentions: tuple of (batch, heads, q_len, kv_len)  -- one per layer.
    Returns dict {chunk_idx: float} averaged across layers.
    """
    num_chunks = math.ceil(seq_len / chunk_size)
    chunk_masses = {c: 0.0 for c in range(num_chunks)}

    num_layers = len(attentions)
    for attn in attentions:
        # attn: [1, heads, q_len, kv_len]
        layer_avg = attn[0].mean(dim=0)  # [q_len, kv_len]
        query_attn = layer_avg[-1].detach().float()  # last query position -> [kv_len]
        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, query_attn.shape[0])
            chunk_masses[c] += float(query_attn[start:end].sum())

    # Average across layers
    for c in chunk_masses:
        chunk_masses[c] /= num_layers

    return chunk_masses


# ── Main ─────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Real Trace Collector  —  Qwen2.5-1.5B on Passkey")
    print("=" * 60)

    # ── Load model & tokenizer ───────────────────────────────────────
    print(f"\n[1/3] Loading model from {MODEL_PATH} ...")
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    print(f"    Model loaded in {time.time() - t0:.1f}s  "
          f"(params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M)")

    # ── Collect traces ───────────────────────────────────────────────
    print(f"\n[2/3] Collecting traces "
          f"(lengths={CONTEXT_LENGTHS}, positions={PASSKEY_POSITIONS}, "
          f"samples={SAMPLES_PER_CONFIG}, steps={MAX_GEN_STEPS})")

    all_output_paths = []
    total_configs = len(CONTEXT_LENGTHS) * len(PASSKEY_POSITIONS) * SAMPLES_PER_CONFIG
    config_idx = 0

    for ctx_len in CONTEXT_LENGTHS:
        for pos in PASSKEY_POSITIONS:
            for sample_id in range(SAMPLES_PER_CONFIG):
                config_idx += 1
                tag = f"ctx={ctx_len} pos={pos} sample={sample_id}"
                print(f"\n  [{config_idx}/{total_configs}] {tag}")

                try:
                    result = collect_one_trace(
                        model, tokenizer, ctx_len, pos, sample_id
                    )
                except Exception:
                    traceback.print_exc()
                    print(f"    !! FAILED — skipping")
                    continue

                # Save individual file
                fname = (f"trace_real_Qwen2.5-1.5B_{ctx_len}_pos{pos}"
                         f"_s{sample_id}.json")
                out_path = OUTPUT_DIR / fname
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                all_output_paths.append(str(out_path))
                print(f"    -> saved {out_path.name}")

                # Free memory between samples
                gc.collect()

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n[3/3] Done — {len(all_output_paths)} trace files saved to {OUTPUT_DIR}")
    for p in all_output_paths:
        print(f"    {p}")


def collect_one_trace(model, tokenizer, ctx_len, position, sample_id):
    """Collect traces for a single (ctx_len, position) configuration."""

    input_ids, passkey, passkey_chunk, context_token_len = build_passkey_input(
        tokenizer, ctx_len, position
    )
    seq_len = input_ids.shape[1]
    num_chunks = math.ceil(seq_len / CHUNK_SIZE)
    budget = max(1, int(num_chunks * BUDGET_RATIO))

    print(f"    seq_len={seq_len}  chunks={num_chunks}  passkey='{passkey}'  "
          f"passkey_chunk={passkey_chunk}  budget={budget}")

    traces = []
    generated_tokens = []
    past_key_values = None

    with torch.no_grad():
        current_ids = input_ids

        for step in range(MAX_GEN_STEPS):
            t_step = time.time()

            if past_key_values is not None:
                # Incremental decode: only feed last token
                outputs = model(
                    input_ids=current_ids[:, -1:],
                    past_key_values=past_key_values,
                    output_attentions=True,
                    use_cache=True,
                )
            else:
                # First step: full prefill
                outputs = model(
                    input_ids=current_ids,
                    output_attentions=True,
                    use_cache=True,
                )

            logits = outputs.logits[:, -1, :]
            next_token = int(logits.argmax(dim=-1).item())
            generated_tokens.append(next_token)

            # Attention aggregation
            attentions = outputs.attentions  # tuple of (1, heads, q, kv)
            kv_len = attentions[0].shape[-1]
            chunk_masses = aggregate_chunk_attention(
                attentions, kv_len, CHUNK_SIZE
            )

            # Gold chunks = top-budget by attention mass
            sorted_chunks = sorted(
                chunk_masses.items(), key=lambda x: x[1], reverse=True
            )
            gold_chunks = sorted([c for c, _ in sorted_chunks[:budget]])

            traces.append({
                "token_id": step,
                "step": step,
                "chunk_attention_masses": {str(k): v for k, v in chunk_masses.items()},
                "gold_chunks": gold_chunks,
                "generated_token": next_token,
                "passkey_chunk": passkey_chunk,
            })

            # Update for next step
            past_key_values = outputs.past_key_values
            next_ids = torch.tensor([[next_token]], dtype=torch.long, device=current_ids.device)
            current_ids = torch.cat([current_ids, next_ids], dim=1)

            # Free attention tensors immediately
            del attentions, outputs
            gc.collect()

            elapsed = time.time() - t_step
            decoded = tokenizer.decode([next_token])
            print(f"      step {step}: tok={next_token} '{decoded.strip()}'  "
                  f"({elapsed:.1f}s)")

            # Early stop on EOS
            if next_token == tokenizer.eos_token_id:
                print(f"      (EOS reached at step {step})")
                break

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    passkey_found = passkey in generated_text

    result = {
        "method": "real_model",
        "model": "Qwen2.5-1.5B-Instruct",
        "context_length": ctx_len,
        "budget_ratio": BUDGET_RATIO,
        "total_chunks": num_chunks,
        "selected_chunks": budget,
        "synthetic": False,
        "passkey": passkey,
        "passkey_position": position,
        "passkey_chunk": passkey_chunk,
        "passkey_found_in_generation": passkey_found,
        "generated_text": generated_text,
        "num_gen_steps": len(traces),
        "timestamp": datetime.now().isoformat(),
        "traces": traces,
    }
    return result


if __name__ == "__main__":
    main()
