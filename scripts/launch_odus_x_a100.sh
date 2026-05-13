#!/bin/bash
# ODUS-X Validation Launcher (single-GPU sequential)
#
# Runs the critical real-model validation for the upgraded ODUS-X scorer.
# Uses GPU 0 sequentially to avoid multi-process memory contention.

set -e

export HF_HUB_OFFLINE=1
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

SCRIPT="prose_v2/scripts/run_odus_x_validation.py"
OUTDIR="outputs/hpca_odus_v2"
MODEL_3B="/home/Administrator/LLM_Project/LLM/models/Qwen2.5-3B-Instruct"
MODEL_7B="/home/Administrator/LLM_Project/LLM/models/Qwen2.5-7B-Instruct"

mkdir -p "$OUTDIR"

echo "================================================================================"
echo "ODUS-X Validation on A100 (Sequential Single-GPU)"
echo "================================================================================"

# Job 1: 3B at 4K-16K (fast, core validation)
echo "[Job 1] 3B @ 4K-16K ..."
CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
  --model "$MODEL_3B" \
  --contexts "4096,8192,16384" \
  --num_samples 12 \
  --chunk_size 64 \
  --output_dir "$OUTDIR" \
  > "$OUTDIR/odus_x_3b.log" 2>&1

echo "  -> 3B results saved."

# Job 2: 7B at 16K (scale test, slower)
echo "[Job 2] 7B @ 16K ..."
CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
  --model "$MODEL_7B" \
  --contexts "16384" \
  --num_samples 8 \
  --chunk_size 64 \
  --output_dir "$OUTDIR" \
  --dtype bfloat16 \
  > "$OUTDIR/odus_x_7b.log" 2>&1

echo "  -> 7B results saved."

echo ""
echo "================================================================================"
echo "All ODUS-X validation jobs completed."
echo "Output: $OUTDIR/odus_x_validation.json"
echo "Logs:   $OUTDIR/odus_x_3b.log"
echo "        $OUTDIR/odus_x_7b.log"
echo "================================================================================"
