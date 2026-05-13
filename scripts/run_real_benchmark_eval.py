"""
Real Benchmark Evaluation — Passkey Retrieval with KV Cache Policies.

Evaluates ProSE-SW, H2O, SnapKV, StreamingLLM, and Full-KV on passkey
retrieval using Qwen2.5-1.5B.  Uses attention-mask simulation to
approximate KV-cache pruning without deep integration.

Also attempts LongBench evaluation; skips gracefully if data unavailable.

Usage:
    python run_real_benchmark_eval.py
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
from typing import Dict, List, Optional, Tuple

import torch

# ── Configuration ────────────────────────────────────────────────────

MODEL_PATH = r"d:\LLM\models\Qwen2.5-1.5B-Instruct"
OUTPUT_PATH = Path(r"d:\LLM\outputs\real_benchmark_results.json")

CONTEXT_LENGTHS = [1024, 2048]
PASSKEY_POSITIONS = [0.25, 0.5, 0.75]
BUDGET_RATIO = 0.1
SAMPLES_PER_CONFIG = 2
CHUNK_SIZE = 256
MAX_GEN_TOKENS = 20

# ── Filler text ──────────────────────────────────────────────────────

FILLER_TEXT = (
    "The quick brown fox jumps over the lazy dog. This is filler text used to create "
    "long context lengths for testing. The grass is green and the sky is blue. "
    "Mountains rise in the distance and rivers flow through valleys. "
)


# ── Passkey construction ─────────────────────────────────────────────

def generate_passkey() -> str:
    return "".join([str(random.randint(0, 9)) for _ in range(5)])


def build_passkey_sample(tokenizer, target_length: int, position: float):
    """Build passkey sample using chat template.  Returns (context_ids, query_ids, passkey_str)."""
    passkey = generate_passkey()
    passkey_sentence = f"The pass key is {passkey}. Remember it."

    repeats = (target_length * 6) // len(FILLER_TEXT) + 2
    all_filler = (FILLER_TEXT + " ") * repeats

    insert_pos = int(len(all_filler) * position)
    context_text = all_filler[:insert_pos] + " " + passkey_sentence + " " + all_filler[insert_pos:]

    # Trim filler to fit within target_length tokens (roughly)
    filler_ids = tokenizer.encode(context_text, add_special_tokens=False)
    if len(filler_ids) > target_length:
        filler_ids = filler_ids[:target_length]
        context_text = tokenizer.decode(filler_ids, skip_special_tokens=True)

    # Use chat template for Instruct model
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Read the text carefully and answer the question."},
        {"role": "user", "content": f"{context_text}\n\nWhat is the pass key mentioned in the text above? Reply with only the 5-digit number."},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # Split: everything before the assistant turn is "context", the assistant prompt is "query"
    # For simplicity, treat the whole thing as context_ids and use empty query_ids
    context_ids = full_ids
    query_ids = []  # already included in chat template

    return context_ids, query_ids, passkey


# ── KV retention policies (attention-mask simulation) ────────────────

def select_full_kv(seq_len: int, _attn_weights, _chunk_size: int) -> List[int]:
    """Full KV — retain everything."""
    return list(range(seq_len))


def select_prose_sw(seq_len: int, attn_weights: torch.Tensor,
                    chunk_size: int) -> List[int]:
    """ProSE-SW: Promotion-based selection.

    Simulates the sliding-window promotion policy:
    - Always keep sink tokens (first chunk) and recent tokens (last chunk)
    - Promote chunks with highest attention mass
    """
    num_chunks = math.ceil(seq_len / chunk_size)
    budget = max(2, int(num_chunks * BUDGET_RATIO))

    # Compute per-chunk attention
    chunk_scores = {}
    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, seq_len)
        chunk_scores[c] = float(attn_weights[start:end].sum())

    # Always include sink (0) and recent (last)
    retained_chunks = {0, num_chunks - 1}

    # Fill remaining budget with top-scoring chunks
    remaining_budget = budget - len(retained_chunks)
    if remaining_budget > 0:
        sorted_chunks = sorted(
            [(c, s) for c, s in chunk_scores.items() if c not in retained_chunks],
            key=lambda x: x[1], reverse=True,
        )
        for c, _ in sorted_chunks[:remaining_budget]:
            retained_chunks.add(c)

    # Expand chunks to positions
    positions = []
    for c in sorted(retained_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, seq_len)
        positions.extend(range(start, end))
    return positions


def select_h2o(seq_len: int, attn_weights: torch.Tensor,
               chunk_size: int) -> List[int]:
    """H2O: Heavy-Hitter Oracle — top-K positions by cumulative attention."""
    budget = max(1, int(seq_len * BUDGET_RATIO))
    hh_budget = budget // 2
    recent_budget = budget - hh_budget

    # Heavy hitters
    top_indices = attn_weights.argsort(descending=True)[:hh_budget].tolist()
    # Recent
    recent = list(range(max(0, seq_len - recent_budget), seq_len))

    return sorted(set(top_indices + recent))


def select_snapkv(seq_len: int, attn_weights: torch.Tensor,
                  chunk_size: int) -> List[int]:
    """SnapKV: Observation-window selection.

    Uses the last observation_tokens query positions' attention as scores.
    Here we approximate with the aggregated attention already available.
    """
    budget = max(1, int(seq_len * BUDGET_RATIO))
    top_indices = attn_weights.argsort(descending=True)[:budget].tolist()
    return sorted(top_indices)


def select_streaming_llm(seq_len: int, _attn_weights,
                         chunk_size: int) -> List[int]:
    """StreamingLLM: Keep first sink_size + last recent_size tokens."""
    budget = max(1, int(seq_len * BUDGET_RATIO))
    sink_size = max(4, budget // 4)
    recent_size = budget - sink_size

    sink = list(range(min(sink_size, seq_len)))
    recent = list(range(max(0, seq_len - recent_size), seq_len))
    return sorted(set(sink + recent))


METHODS = {
    "full_kv": select_full_kv,
    "prose_sw": select_prose_sw,
    "h2o": select_h2o,
    "snapkv": select_snapkv,
    "streaming_llm": select_streaming_llm,
}


# ── Evaluation core ─────────────────────────────────────────────────

def evaluate_sample(
    model, tokenizer,
    context_ids: List[int],
    query_ids: List[int],
    passkey: str,
    method_fn,
) -> Tuple[bool, str]:
    """
    Evaluate one sample with a given retention policy.

    1) Full forward to get attention weights.
    2) Select retained positions using the policy.
    3) Create attention mask zeroing out non-retained positions.
    4) Greedy decode with the masked attention.

    Returns (is_correct, generated_text).
    """
    full_ids = context_ids + query_ids
    device = next(model.parameters()).device
    input_tensor = torch.tensor([full_ids], dtype=torch.long).to(device)
    seq_len = input_tensor.shape[1]

    with torch.no_grad():
        # Step 1: Get attention weights from full forward
        outputs = model(
            input_ids=input_tensor,
            output_attentions=True,
            use_cache=False,
        )
        # Aggregate attention: average across layers & heads, last query pos
        attn_agg = None
        for attn in outputs.attentions:
            layer_avg = attn[0].mean(dim=0)[-1].float()  # [kv_len]
            if attn_agg is None:
                attn_agg = layer_avg
            else:
                attn_agg = attn_agg + layer_avg
        attn_agg = attn_agg / len(outputs.attentions)

        del outputs
        gc.collect()
        torch.cuda.empty_cache()

        # Step 2: Select retained positions
        retained = method_fn(seq_len, attn_agg, CHUNK_SIZE)
        retained_set = set(retained)

        # Step 3: Greedy decode with attention mask
        generated_tokens = []
        past_kv = None
        current_input = input_tensor

        for step in range(MAX_GEN_TOKENS):
            # Build causal attention mask with zeroed-out non-retained positions
            cur_len = current_input.shape[1]

            if past_kv is not None:
                # Incremental: feed only last token
                step_input = current_input[:, -1:]
                out = model(
                    input_ids=step_input,
                    past_key_values=past_kv,
                    use_cache=True,
                )
            else:
                # First step: apply mask via custom attention_mask
                # Create 2D mask: [1, seq_len] — 1 for retained, 0 for masked
                mask = torch.zeros(1, cur_len, dtype=torch.long, device=current_input.device)
                for p in retained_set:
                    if p < cur_len:
                        mask[0, p] = 1
                # Always keep query positions visible
                query_start = len(context_ids)
                for p in range(query_start, cur_len):
                    mask[0, p] = 1

                out = model(
                    input_ids=current_input,
                    attention_mask=mask,
                    use_cache=True,
                )

            logits = out.logits[:, -1, :]
            next_token = int(logits.argmax(dim=-1).item())
            generated_tokens.append(next_token)
            past_kv = out.past_key_values

            next_ids = torch.tensor([[next_token]], dtype=torch.long, device=current_input.device)
            current_input = torch.cat([current_input, next_ids], dim=1)

            del out
            gc.collect()
            torch.cuda.empty_cache()

            if next_token == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    is_correct = passkey in generated_text
    return is_correct, generated_text


# ── LongBench attempt ────────────────────────────────────────────────

def try_longbench_eval(model, tokenizer) -> Optional[Dict]:
    """Try to run a minimal LongBench evaluation.  Returns None on failure."""
    print("\n  Attempting LongBench evaluation ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("THUDM/LongBench", "qasper", split="test",
                          trust_remote_code=True)
        print(f"    Loaded {len(ds)} LongBench/qasper examples")

        # Only evaluate 2 examples to stay within time budget
        correct = 0
        total = min(2, len(ds))
        for i in range(total):
            item = ds[i]
            context = item.get("context", item.get("input", ""))[:4000]
            query = item.get("input", item.get("question", ""))
            answers = item.get("answers", [item.get("answer", "")])
            if isinstance(answers, str):
                answers = [answers]

            prompt = context + "\n\nQuestion: " + query + "\nAnswer:"
            messages_lb = [
                {"role": "user", "content": prompt},
            ]
            prompt_chat = tokenizer.apply_chat_template(messages_lb, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(prompt_chat, return_tensors="pt",
                                         truncation=True, max_length=4096).to(model.device)
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=64,
                                     do_sample=False)
            gen = tokenizer.decode(out[0][input_ids.shape[1]:],
                                   skip_special_tokens=True).strip()
            # Simple F1-style check
            gen_lower = gen.lower()
            hit = any(a.lower() in gen_lower for a in answers if a)
            correct += int(hit)
            print(f"    LongBench sample {i}: {'CORRECT' if hit else 'WRONG'}")

        return {
            "benchmark": "longbench_qasper",
            "accuracy": correct / total if total else 0,
            "correct": correct,
            "total": total,
            "note": "minimal_eval_2_samples",
        }
    except Exception as e:
        print(f"    LongBench unavailable: {e}")
        return {"benchmark": "longbench", "status": "skipped", "reason": str(e)}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Real Benchmark Evaluation — Passkey + Baselines")
    print("=" * 60)

    # ── Load model ───────────────────────────────────────────────────
    print(f"\n[1/4] Loading model from {MODEL_PATH} ...")
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
    print(f"    Loaded in {time.time() - t0:.1f}s")

    # ── Build samples ────────────────────────────────────────────────
    print(f"\n[2/4] Building passkey samples "
          f"(lengths={CONTEXT_LENGTHS}, positions={PASSKEY_POSITIONS}, "
          f"samples={SAMPLES_PER_CONFIG})")

    samples = []
    for ctx_len in CONTEXT_LENGTHS:
        for pos in PASSKEY_POSITIONS:
            for sid in range(SAMPLES_PER_CONFIG):
                ctx_ids, q_ids, pk = build_passkey_sample(tokenizer, ctx_len, pos)
                samples.append({
                    "context_ids": ctx_ids,
                    "query_ids": q_ids,
                    "passkey": pk,
                    "context_length": ctx_len,
                    "position": pos,
                    "sample_id": sid,
                })
    print(f"    {len(samples)} samples prepared")

    # ── Evaluate each method ─────────────────────────────────────────
    print(f"\n[3/4] Evaluating methods: {list(METHODS.keys())}")

    method_results = {}
    configs_list = []

    for method_name, method_fn in METHODS.items():
        print(f"\n  --- {method_name} ---")
        correct = 0
        total = 0
        per_sample_results = []

        for i, sample in enumerate(samples):
            tag = (f"ctx={sample['context_length']} pos={sample['position']} "
                   f"s={sample['sample_id']}")
            print(f"    [{i + 1}/{len(samples)}] {tag} ... ", end="", flush=True)
            t_sample = time.time()

            try:
                is_correct, gen_text = evaluate_sample(
                    model, tokenizer,
                    sample["context_ids"],
                    sample["query_ids"],
                    sample["passkey"],
                    method_fn,
                )
                elapsed = time.time() - t_sample
                status = "CORRECT" if is_correct else "WRONG"
                print(f"{status} ({elapsed:.1f}s)  gen='{gen_text[:40]}'")

                correct += int(is_correct)
                total += 1
                per_sample_results.append({
                    "context_length": sample["context_length"],
                    "position": sample["position"],
                    "passkey": sample["passkey"],
                    "correct": is_correct,
                    "generated": gen_text[:100],
                })
            except Exception:
                traceback.print_exc()
                print("FAILED")
                total += 1
                per_sample_results.append({
                    "context_length": sample["context_length"],
                    "position": sample["position"],
                    "correct": False,
                    "error": True,
                })

            gc.collect()
            torch.cuda.empty_cache()

        accuracy = correct / total if total > 0 else 0.0
        method_results[method_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": per_sample_results,
        }
        print(f"  => {method_name}: {correct}/{total} = {accuracy:.2%}")

    # ── LongBench ────────────────────────────────────────────────────
    longbench_result = try_longbench_eval(model, tokenizer)

    # ── Assemble and save results ────────────────────────────────────
    for ctx_len in CONTEXT_LENGTHS:
        for pos in PASSKEY_POSITIONS:
            configs_list.append({
                "context_length": ctx_len,
                "position": pos,
                "budget": BUDGET_RATIO,
            })

    results = {
        "model": "Qwen2.5-1.5B-Instruct",
        "benchmark": "passkey",
        "timestamp": datetime.now().isoformat(),
        "configs": configs_list,
        "methods": method_results,
        "longbench": longbench_result,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[4/4] Results saved to {OUTPUT_PATH}")
    print("\n  === Summary ===")
    for m, r in method_results.items():
        print(f"    {m:20s}  {r['accuracy']:.2%}  ({r['correct']}/{r['total']})")
    if longbench_result:
        lb_status = longbench_result.get("status", "")
        if lb_status == "skipped":
            print(f"    {'longbench':20s}  SKIPPED ({longbench_result.get('reason', '')})")
        else:
            print(f"    {'longbench':20s}  {longbench_result.get('accuracy', 0):.2%}")


if __name__ == "__main__":
    main()
