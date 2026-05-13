#!/bin/bash
#
# Run vLLM with ProSE-X 2.0 metrics in WSL
#
# Usage:
#   ./run_vllm_wsl.sh --model microsoft/Phi-3-mini-4k-instruct --context-length 4096
#

set -e

# Default values
MODEL="microsoft/Phi-3-mini-4k-instruct"
CONTEXT_LENGTH=4096
BUDGET_RATIO=0.1
WORKLOAD_TYPE="passkey"
OUTPUT_DIR="outputs/vllm_wsl"
USE_PROSE_V2=true
DETAILED_METRICS=true
NUM_STEPS=20

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --context-length)
            CONTEXT_LENGTH="$2"
            shift 2
            ;;
        --budget-ratio)
            BUDGET_RATIO="$2"
            shift 2
            ;;
        --workload-type)
            WORKLOAD_TYPE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-prose-v2)
            USE_PROSE_V2=false
            shift
            ;;
        --no-detailed-metrics)
            DETAILED_METRICS=false
            shift
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL              Model name (default: $MODEL)"
            echo "  --context-length N         Context length in tokens (default: $CONTEXT_LENGTH)"
            echo "  --budget-ratio RATIO       Budget ratio 0.0-1.0 (default: $BUDGET_RATIO)"
            echo "  --workload-type TYPE       passkey|multihop|needle (default: $WORKLOAD_TYPE)"
            echo "  --output-dir DIR           Output directory (default: $OUTPUT_DIR)"
            echo "  --no-prose-v2              Disable ProSE-X v2 features"
            echo "  --no-detailed-metrics      Disable detailed per-step metrics"
            echo "  --num-steps N              Number of decode steps (default: $NUM_STEPS)"
            echo "  --help                     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}ProSE-X 2.0 vLLM Runner for WSL${NC}"
echo -e "${CYAN}==========================================${NC}"
echo ""

# Check if we're in WSL
if ! grep -q Microsoft /proc/version 2>/dev/null && ! grep -q microsoft /proc/version 2>/dev/null; then
    echo -e "${YELLOW}Warning: Not running in WSL. This script is designed for WSL.${NC}"
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${GREEN}Configuration:${NC}"
echo "  Model: $MODEL"
echo "  Context Length: $CONTEXT_LENGTH"
echo "  Budget Ratio: $BUDGET_RATIO"
echo "  Workload Type: $WORKLOAD_TYPE"
echo "  Use ProSE-X v2: $USE_PROSE_V2"
echo "  Detailed Metrics: $DETAILED_METRICS"
echo "  Num Steps: $NUM_STEPS"
echo "  Project Root: $PROJECT_ROOT"
echo ""

# Check for conda environment
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install conda first.${NC}"
    exit 1
fi

# Activate vllm environment
echo -e "${YELLOW}Activating conda environment 'vllm'...${NC}"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate vllm

# Verify Python environment
echo -e "${YELLOW}Python: $(which python)${NC}"
echo -e "${YELLOW}Version: $(python --version)${NC}"
echo ""

# Build Python command
PYTHON_SCRIPT="$PROJECT_ROOT/prose_v2/src/runners/run_vllm_metrics.py"

CMD_ARGS=(
    --model "$MODEL"
    --context-length "$CONTEXT_LENGTH"
    --budget-ratio "$BUDGET_RATIO"
    --workload-type "$WORKLOAD_TYPE"
    --output-dir "$OUTPUT_DIR"
    --num-steps "$NUM_STEPS"
)

if [ "$USE_PROSE_V2" = true ]; then
    CMD_ARGS+=(--use-prose-v2)
fi

if [ "$DETAILED_METRICS" = true ]; then
    CMD_ARGS+=(--detailed-metrics)
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the Python script
echo -e "${CYAN}Running vLLM simulation with ProSE-X 2.0...${NC}"
echo "Command: python $PYTHON_SCRIPT ${CMD_ARGS[*]}"
echo ""

cd "$PROJECT_ROOT"
python "$PYTHON_SCRIPT" "${CMD_ARGS[@]}"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}==========================================${NC}"
    echo -e "${GREEN}Run completed successfully!${NC}"
    echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
    echo -e "${GREEN}==========================================${NC}"
else
    echo -e "${RED}==========================================${NC}"
    echo -e "${RED}Run failed with exit code: $EXIT_CODE${NC}"
    echo -e "${RED}==========================================${NC}"
    exit $EXIT_CODE
fi
