#!/bin/bash
# FireTrend: Evaluation Script


# 1. Environment settings
export CUDA_VISIBLE_DEVICES=0

# 2. Path setup
PROJECT_DIR="$(dirname $(dirname $(realpath $0)))"
CONFIG_FILE="$PROJECT_DIR/config.yaml"
LOG_DIR="$PROJECT_DIR/outputs/logs"
CKPT_DIR="$PROJECT_DIR/outputs/checkpoints"
RESULTS_DIR="$PROJECT_DIR/outputs/predictions"

mkdir -p $RESULTS_DIR

# 3. Locate latest checkpoint
LATEST_CKPT=$(ls -t $CKPT_DIR/*.pth 2>/dev/null | head -n 1)

if [ -z "$LATEST_CKPT" ]; then
    echo "No checkpoint found in $CKPT_DIR"
    exit 1
else
    echo "Found checkpoint: $LATEST_CKPT"
fi

# 4. Run evaluation
EVAL_MODE="test"   # can be "val" or "test"

echo "🚀 Starting FireTrend evaluation..."
python "$PROJECT_DIR/main.py" \
    --config "$CONFIG_FILE" \
    --checkpoint "$LATEST_CKPT" \
    --mode $EVAL_MODE \
    --results_dir "$RESULTS_DIR" \
    --eval \
    2>&1 | tee "$LOG_DIR/eval_$(date +%Y%m%d_%H%M%S).log"

echo "Evaluation completed. Results saved to $RESULTS_DIR"
