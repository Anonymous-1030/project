#!/bin/bash
# Launch remaining ODUS-X validation jobs on A100 (sequential, safe)

set -e
export HF_HUB_OFFLINE=1
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

MODEL_3B="/home/Administrator/LLM_Project/LLM/models/Qwen2.5-3B-Instruct"
MODEL_7B="/home/Administrator/LLM_Project/LLM/models/Qwen2.5-7B-Instruct"
OUTDIR="outputs/hpca_odus_v2"
mkdir -p "$OUTDIR"

echo "================================================================================"
echo "ODUS-X Remaining Validation Jobs"
echo "================================================================================"

# Job 1: Drift-heavy validation on 3B @ 8K
echo "[Job 1] Drift-heavy validation (3B @ 8192) ..."
CUDA_VISIBLE_DEVICES=0 python prose_v2/scripts/run_odus_x_drift_validation.py \
  --model "$MODEL_3B" \
  --length 8192 \
  --num_samples 12 \
  --chunk_size 64 \
  --output_dir "$OUTDIR" \
  > "$OUTDIR/odus_x_drift.log" 2>&1

echo "  -> Drift results saved."

# Job 2: Scale test on 7B @ 16K/32K
echo "[Job 2] Scale test (7B @ 16384,32768) ..."
CUDA_VISIBLE_DEVICES=0 python prose_v2/scripts/run_odus_x_validation.py \
  --model "$MODEL_7B" \
  --contexts "16384,32768" \
  --num_samples 8 \
  --chunk_size 64 \
  --output_dir "${OUTDIR}_7b" \
  --dtype bfloat16 \
  > "$OUTDIR/odus_x_7b.log" 2>&1

echo "  -> 7B results saved."

echo ""
echo "================================================================================"
echo "All remaining jobs completed."
echo "================================================================================"
