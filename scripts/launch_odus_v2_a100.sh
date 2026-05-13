#!/bin/bash
# Launch ODUS v2.0 validation suite across 4x A100 80GB GPUs
#
# Recommended experiment plan:
#   GPU 0: Semantic Sketch Layer Comparison (1.5B/3B, 1K-16K)
#   GPU 1: Ensemble Ablation (3B, 4K-16K)
#   GPU 2: Compression Methods (3B, 4K-16K)
#   GPU 3: Scale Test (7B, 16K-64K)
#
# Usage:
#   bash prose_v2/scripts/launch_odus_v2_a100.sh

set -e

# Force offline mode and disable SSL cert verification for HuggingFace hub
export HF_HUB_OFFLINE=1
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

SCRIPT="prose_v2/scripts/run_odus_v2_validation.py"
OUTDIR="outputs/hpca_odus_v2"

# Adjust model paths to your A100 environment
MODEL_BASE="/home/Administrator/LLM_Project/LLM/models"
MODEL_1_5B="${MODEL_BASE}/Qwen2.5-1.5B-Instruct"
MODEL_3B="${MODEL_BASE}/Qwen2.5-3B-Instruct"
MODEL_7B="${MODEL_BASE}/Qwen2.5-7B-Instruct"

mkdir -p "$OUTDIR"

echo "================================================================================"
echo "ODUS v2.0 Validation Launch: 4x A100 Parallel Jobs"
echo "================================================================================"

# -----------------------------------------------------------------------------
# Job 1: Semantic Sketch Layer Comparison (GPU 0)
# Compare input embedding vs layer 0 vs deeper layers
# -----------------------------------------------------------------------------
echo "[Job 1] Launching semantic_sketch on GPU 0 ..."
CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
  --model "$MODEL_3B" \
  --experiment semantic_sketch \
  --contexts=1024,4096,8192,16384 \
  --num_samples=15 \
  --layers=-1,0,2,6,12,23 \
  --output_dir "$OUTDIR" \
  > "$OUTDIR/semantic_sketch.log" 2>&1 &
PID1=$!

# -----------------------------------------------------------------------------
# Job 2: Ensemble Ablation (GPU 1)
# Single-cue vs multi-cue ensemble
# -----------------------------------------------------------------------------
echo "[Job 2] Launching ensemble_ablation on GPU 1 ..."
CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" \
  --model "$MODEL_3B" \
  --experiment ensemble_ablation \
  --contexts=4096,8192,16384 \
  --num_samples=15 \
  --output_dir "$OUTDIR" \
  > "$OUTDIR/ensemble_ablation.log" 2>&1 &
PID2=$!

# -----------------------------------------------------------------------------
# Job 3: Compression Methods (GPU 2)
# Random vs PCA vs trained linear projection
# -----------------------------------------------------------------------------
echo "[Job 3] Launching compression on GPU 2 ..."
CUDA_VISIBLE_DEVICES=2 python "$SCRIPT" \
  --model "$MODEL_3B" \
  --experiment compression \
  --contexts=4096,8192,16384 \
  --num_samples=15 \
  --compression_layer=0 \
  --proj_dim=16 \
  --output_dir "$OUTDIR" \
  > "$OUTDIR/compression.log" 2>&1 &
PID3=$!

# -----------------------------------------------------------------------------
# Job 4: Scale Test with 7B (GPU 3)
# Full ensemble at 16K-64K to prove A100-scale viability
# -----------------------------------------------------------------------------
echo "[Job 4] Launching scale_test on GPU 3 ..."
CUDA_VISIBLE_DEVICES=3 python "$SCRIPT" \
  --model "$MODEL_7B" \
  --experiment scale_test \
  --contexts=16384,32768,65536 \
  --num_samples=10 \
  --output_dir "$OUTDIR" \
  --dtype bfloat16 \
  > "$OUTDIR/scale_test.log" 2>&1 &
PID4=$!

# -----------------------------------------------------------------------------
echo ""
echo "All 4 jobs launched in background."
echo "  Job 1 (GPU 0) PID=$PID1  ->  $OUTDIR/semantic_sketch.log"
echo "  Job 2 (GPU 1) PID=$PID2  ->  $OUTDIR/ensemble_ablation.log"
echo "  Job 3 (GPU 2) PID=$PID3  ->  $OUTDIR/compression.log"
echo "  Job 4 (GPU 3) PID=$PID4  ->  $OUTDIR/scale_test.log"
echo ""
echo "Monitor progress:"
echo "  tail -f $OUTDIR/*.log"
echo ""
echo "Wait for completion:"
echo "  wait $PID1 $PID2 $PID3 $PID4"
echo ""
