"""
Lightweight logic validation for P0 closed-loop KV pruning.

Uses synthetic tensors to verify:
1. prefill_and_prune reduces KV cache size correctly
2. generate_with_pruned_kv passes explicit position_ids
3. Benchmark adapters work correctly

No real model loading — runs in <10 seconds on CPU.
"""

import sys
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.attention.kv_cache_manager import SparseKVCacheManager, PrunedKVCache
from transformers.cache_utils import DynamicCache


class MockModel:
    """Minimal mock that mimics HF causal LM forward behavior."""

    def __init__(self, num_layers=4, num_kv_heads=2, head_dim=8):
        self.config = type("C", (), {
            "num_hidden_layers": num_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": num_kv_heads,
            "hidden_size": 32,
            "head_dim": head_dim,
            "vocab_size": 100,
        })()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._past_kv = None
        self._recorded_position_ids = []

    def __call__(self, input_ids, past_key_values=None, position_ids=None,
                 use_cache=False, output_attentions=False, **kwargs):
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Simulate attention output
        hidden = torch.randn(batch, seq_len, 32, device=device)

        # Simulate past_key_values update
        if past_key_values is not None:
            past_len = past_key_values.get_seq_length()
        else:
            past_len = 0

        new_cache = DynamicCache()
        if past_key_values is not None:
            for layer in past_key_values.layers:
                new_cache.update(layer.keys, layer.values, len(new_cache.layers))

        # Add new KV for each layer
        for layer_idx in range(self.num_layers):
            new_k = torch.randn(batch, self.num_kv_heads, seq_len, self.head_dim, device=device)
            new_v = torch.randn(batch, self.num_kv_heads, seq_len, self.head_dim, device=device)
            new_cache.update(new_k, new_v, layer_idx)

        # Record position_ids for verification
        if position_ids is not None:
            self._recorded_position_ids.append(position_ids.detach().cpu().tolist())

        # Simulate logits
        logits = torch.randn(batch, seq_len, self.config.vocab_size, device=device)

        # Mock attention weights if requested
        attentions = None
        if output_attentions:
            # [batch, heads, q_len, kv_len]
            kv_len = new_cache.get_seq_length()
            attentions = tuple(
                torch.rand(batch, 4, seq_len, kv_len, device=device)
                for _ in range(self.num_layers)
            )

        class MockOutput:
            def __init__(self, logits, past_key_values, attentions):
                self.logits = logits
                self.past_key_values = past_key_values
                self.attentions = attentions

        return MockOutput(logits, new_cache, attentions)

    def eval(self):
        pass

    def named_modules(self):
        # Return minimal modules for AttentionHookExtractor
        return [("model.layers.0.self_attn", object())]

    def parameters(self):
        yield torch.tensor(0.0)


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 2
        self.eos_token = "<|endoftext|>"
        self.pad_token = self.eos_token

    def encode(self, text, return_tensors=None, **kwargs):
        # Deterministic encoding based on text length
        tokens = list(range(10, 10 + len(text) // 3))
        if return_tensors == "pt":
            return torch.tensor([tokens])
        return tokens

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return " ".join(str(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "\n".join(m["content"] for m in messages)


# ── Tests ──────────────────────────────────────────────────────────

def test_pruned_to_dynamic_cache():
    """Verify PrunedKVCache → DynamicCache conversion."""
    kv_manager = SparseKVCacheManager(num_layers=3, num_kv_heads=2, head_dim=8)

    # Create fake prefill KV
    fake_kv = tuple(
        (torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8))
        for _ in range(3)
    )
    kv_manager.load_from_prefill(fake_kv)
    kv_manager.set_chunk_boundaries([(0, 4), (4, 7), (7, 10)])

    pruned = kv_manager.prune_by_chunks(anchor_chunk_ids=[0, 2], promoted_chunk_ids=[1])
    assert pruned.total_entries == 10  # all chunks retained
    assert pruned.original_entries == 10

    pruned2 = kv_manager.prune_by_chunks(anchor_chunk_ids=[0], promoted_chunk_ids=[])
    assert pruned2.total_entries == 4  # only first chunk
    assert pruned2.compression_ratio == 0.4

    # Convert to DynamicCache
    cache = DynamicCache()
    for layer_idx in range(pruned2.num_layers):
        k, v = pruned2.kv_per_layer[layer_idx]
        cache.update(k, v, layer_idx)

    assert cache.get_seq_length() == 4
    print("[PASS] test_pruned_to_dynamic_cache passed")


def test_prefill_and_prune():
    """Test full prefill → prune pipeline with mock model."""
    from prose_v2.scripts.run_p0_closed_loop_eval import TrueSparseKVGenerator, P0Config

    config = P0Config(smoke_test=True)
    model = MockModel(num_layers=4, num_kv_heads=2, head_dim=8)
    tokenizer = MockTokenizer()
    config.device = "cpu"
    config.chunk_size = 4
    config.anchor_ratio = 0.25

    gen = TrueSparseKVGenerator(model, tokenizer, config)

    # Prefill with 12 tokens (3 chunks of 4)
    input_ids = torch.arange(12).unsqueeze(0)
    pruned_cache, orig_len, pruned_len, stats = gen.prefill_and_prune(
        input_ids, method="prose", hbm_cap_gb=None,
    )

    assert orig_len == 12
    assert pruned_cache.get_seq_length() < 12  # should have pruned
    assert stats["num_chunks"] == 3
    assert stats["compression_ratio"] < 1.0
    print(f"[PASS] test_prefill_and_prune passed (orig={orig_len}, pruned={pruned_cache.get_seq_length()})")


def test_generate_with_pruned_kv():
    """Test generation with explicit position_ids."""
    from prose_v2.scripts.run_p0_closed_loop_eval import TrueSparseKVGenerator, P0Config

    config = P0Config(smoke_test=True)
    model = MockModel(num_layers=4, num_kv_heads=2, head_dim=8)
    tokenizer = MockTokenizer()
    config.device = "cpu"

    gen = TrueSparseKVGenerator(model, tokenizer, config)

    # Create a pruned cache with 5 tokens
    cache = DynamicCache()
    for layer_idx in range(4):
        cache.update(
            torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8), layer_idx
        )

    query_ids = torch.tensor([[10, 11, 12]])
    gen_ids, stats = gen.generate_with_pruned_kv(
        cache, original_seq_len=20, query_ids=query_ids, max_new_tokens=3,
    )

    assert gen_ids.shape[1] == 3
    assert stats["tokens_generated"] == 3

    # Verify position_ids were passed correctly
    recorded = model._recorded_position_ids
    assert len(recorded) == 3  # query + 2 generation steps

    # First forward: query tokens at positions [20, 21, 22]
    assert recorded[0] == [[20, 21, 22]]
    # Second forward: next token at position 23
    assert recorded[1] == [[23]]
    # Third forward: next token at position 24
    assert recorded[2] == [[24]]

    print("[PASS] test_generate_with_pruned_kv passed")
    print(f"  Recorded position_ids: {recorded}")


def test_full_kv_no_pruning():
    """Verify full_kv baseline retains everything."""
    from prose_v2.scripts.run_p0_closed_loop_eval import TrueSparseKVGenerator, P0Config

    config = P0Config(smoke_test=True)
    model = MockModel(num_layers=4, num_kv_heads=2, head_dim=8)
    tokenizer = MockTokenizer()
    config.device = "cpu"
    config.chunk_size = 4

    gen = TrueSparseKVGenerator(model, tokenizer, config)
    input_ids = torch.arange(12).unsqueeze(0)

    pruned_cache, orig_len, pruned_len, stats = gen.prefill_and_prune(
        input_ids, method="full_kv", hbm_cap_gb=None,
    )

    assert pruned_cache.get_seq_length() == 12
    assert stats["compression_ratio"] == 1.0
    print("[PASS] test_full_kv_no_pruning passed")


def test_hbm_cap_budget():
    """Verify HBM cap correctly forces smaller budgets."""
    from prose_v2.scripts.run_p0_closed_loop_eval import TrueSparseKVGenerator, P0Config

    config = P0Config(smoke_test=True)
    model = MockModel(num_layers=4, num_kv_heads=2, head_dim=8)
    tokenizer = MockTokenizer()
    config.device = "cpu"
    config.chunk_size = 4

    gen = TrueSparseKVGenerator(model, tokenizer, config)
    input_ids = torch.arange(12).unsqueeze(0)

    # Very small HBM cap → aggressive pruning
    # Mock model: 4 layers, 2 KV heads, 8 head_dim => 256 bytes/token
    # 12 tokens => 3072 bytes = 2.9e-6 GB full KV
    pruned_cache, orig_len, pruned_len, stats = gen.prefill_and_prune(
        input_ids, method="prose", hbm_cap_gb=1e-7,
    )

    # Cap is ~3% of full KV, so budget should be ~3%
    assert stats["budget_ratio"] < 0.5
    assert pruned_cache.get_seq_length() < 12
    print(f"[PASS] test_hbm_cap_budget passed (budget={stats['budget_ratio']:.2%})")


if __name__ == "__main__":
    print("=" * 60)
    print("P0 Logic Validation (No Real Model)")
    print("=" * 60)
    test_pruned_to_dynamic_cache()
    test_prefill_and_prune()
    test_generate_with_pruned_kv()
    test_full_kv_no_pruning()
    test_hbm_cap_budget()
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
