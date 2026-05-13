#!/usr/bin/env python3
"""
Run vLLM with ProSE-X v2 promotion-based sparse KV cache.

This script demonstrates how to integrate prose_v2's promotion pipeline
with vLLM for long-context LLM inference with sparse attention.

Usage:
    python scripts/run_prose_vllm.py \
        --model Qwen/Qwen2-7B-Instruct \
        --bridge-config configs/bridge/default.yaml \
        --max-model-len 32768 \
        --prompt "Your long prompt here..."

Requirements:
    - vLLM installed and working
    - prose_v2 installed (pip install -e .)
    - GPU with sufficient memory
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Optional


def setup_logging(level: str = "INFO"):
    """Configure logging for the bridge."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run vLLM with ProSE-X v2 promotion-based sparse KV cache"
    )

    # Model args
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name or path (e.g., Qwen/Qwen2-7B-Instruct)"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=32768,
        help="Maximum model context length"
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.90,
        help="GPU memory utilization fraction"
    )

    # Bridge args
    parser.add_argument(
        "--bridge-config", type=str, default="configs/bridge/default.yaml",
        help="Path to bridge YAML configuration"
    )
    parser.add_argument(
        "--disable-promotion", action="store_true",
        help="Disable promotion (full attention baseline)"
    )

    # Generation args
    parser.add_argument(
        "--prompt", type=str,
        help="Prompt text (or use --prompt-file)"
    )
    parser.add_argument(
        "--prompt-file", type=str,
        help="File containing the prompt text"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0.0 = greedy)"
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1,
        help="Number of prompts to process concurrently"
    )

    # Logging
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--stats-file", type=str,
        help="Save bridge statistics to JSON file"
    )

    return parser.parse_args()


def load_prompt(args) -> str:
    """Load prompt from args or file."""
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    # Default: a medium-length test prompt
    return (
        "The following is a passage from a research paper about attention mechanisms "
        "in large language models.\n\n"
        "Attention mechanisms have become a fundamental building block of modern "
        "neural network architectures. The transformer model, introduced by Vaswani "
        "et al. in 2017, relies entirely on self-attention to compute representations "
        "of input sequences. Unlike recurrent neural networks, which process tokens "
        "sequentially, the transformer processes all tokens in parallel, enabling "
        "efficient training on long sequences.\n\n"
        "The key innovation of the transformer is the scaled dot-product attention "
        "mechanism, which computes attention weights as:\n\n"
        "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V\n\n"
        "where Q, K, and V are the query, key, and value matrices respectively, "
        "and d_k is the dimension of the key vectors. This formulation allows each "
        "token to attend to all other tokens in the sequence, with the attention "
        "weights determined by the compatibility between query and key vectors.\n\n"
        "Multi-head attention extends this by computing multiple attention functions "
        "in parallel, allowing the model to jointly attend to information from "
        "different representation subspaces. The outputs of all heads are concatenated "
        "and projected to produce the final output.\n\n"
        "While powerful, the quadratic complexity of self-attention with respect to "
        "sequence length (O(n^2) in both computation and memory) poses challenges "
        "for long-context applications. This has motivated extensive research into "
        "efficient attention mechanisms, including sparse attention, linear attention, "
        "and memory-efficient implementations like FlashAttention.\n\n"
        "In this work, we propose ProSE-X, a promotion-centric sparse KV cache "
        "architecture that reduces memory requirements for long-context inference "
        "while maintaining output quality through a learned promotion policy."
    )


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Step 1: Validate environment
    # ------------------------------------------------------------------
    logger.info("Checking environment...")

    try:
        import torch
        logger.info(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.error("PyTorch not found. Please install PyTorch.")
        sys.exit(1)

    try:
        import vllm
        logger.info(f"vLLM version: {vllm.__version__}")
    except ImportError:
        logger.error("vLLM not found. Please install vLLM.")
        sys.exit(1)

    bridge_config_path = args.bridge_config
    if not os.path.exists(bridge_config_path):
        logger.error(f"Bridge config not found: {bridge_config_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Load prompt
    # ------------------------------------------------------------------
    prompt = load_prompt(args)
    logger.info(f"Prompt length: {len(prompt)} chars, ~{len(prompt.split())} words")

    prompts = [prompt] * args.num_prompts
    logger.info(f"Processing {len(prompts)} prompt(s)")

    # ------------------------------------------------------------------
    # Step 3: Create vLLM engine with prose_v2
    # ------------------------------------------------------------------
    logger.info("Initializing vLLM engine with ProSE-X v2 bridge...")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,            # Required for prose_v2
        enable_prefix_caching=False,   # Conflicts with sparse attention
        trust_remote_code=True,
    )

    # Install prose_v2 bridge
    from src.bridge import integrate_with_vllm

    hook = integrate_with_vllm(llm, bridge_config_path)

    if args.disable_promotion:
        logger.info("Promotion DISABLED (--disable-promotion flag)")
        hook.config.enable_promotion = False
    else:
        logger.info("Promotion ENABLED")

    # ------------------------------------------------------------------
    # Step 4: Run generation
    # ------------------------------------------------------------------
    logger.info("Starting generation...")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Step 5: Print results
    # ------------------------------------------------------------------
    logger.info(f"Generation complete in {elapsed:.2f}s")
    logger.info("=" * 60)

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        logger.info(f"--- Output {i+1} ({num_tokens} tokens) ---")
        logger.info(generated_text[:500] + ("..." if len(generated_text) > 500 else ""))
        logger.info("")

    # ------------------------------------------------------------------
    # Step 6: Report stats
    # ------------------------------------------------------------------
    stats = hook.get_stats()
    logger.info("=" * 60)
    logger.info("Bridge Statistics:")
    logger.info(f"  Total hook calls:       {stats.get('total_hook_calls', 0)}")
    logger.info(f"  Total filtered steps:   {stats.get('total_filtered_steps', 0)}")
    logger.info(f"  Total errors:           {stats.get('total_errors', 0)}")
    logger.info(f"  Total promotion calls:  {stats.get('total_promotion_calls', 0)}")
    logger.info(f"  Pipeline errors:        {stats.get('total_pipeline_errors', 0)}")
    logger.info(f"  Avg filter ratio:       {stats.get('avg_filter_ratio', 0):.2%}")
    logger.info(f"  Known requests:         {stats.get('known_requests', 0)}")
    logger.info(f"  Elapsed time:           {elapsed:.2f}s")

    if args.stats_file:
        import json
        with open(args.stats_file, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"Stats saved to {args.stats_file}")

    # ------------------------------------------------------------------
    # Step 7: Cleanup
    # ------------------------------------------------------------------
    hook.uninstall()
    logger.info("Done.")


if __name__ == "__main__":
    main()
