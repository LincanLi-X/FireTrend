#!/bin/bash
# FireTrend: Training Script


# 1. Basic environment settings
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# 2. Path setup
PROJECT_DIR="$(dirname $(dirname $(realpath $0)))"
CONFIG_FILE="$PROJECT_DIR/config.yaml"
LOG_DIR="$PROJECT_DIR/outputs/logs"
CKPT_DIR="$PROJECT_DIR/outputs/checkpoints"

mkdir -p $LOG_DIR
mkdir -p $CKPT_DIR

# 3. Experiment name
EXP_NAME="FireTrend_train_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

# 4. Training parameters (override yaml if needed)
EPOCHS=80
BATCH_SIZE=4
LR=1e-4

# 5. Start training
echo "🚀 Starting FireTrend training..."
echo "Project Directory: $PROJECT_DIR"
echo "Config: $CONFIG_FILE"
echo "Logs: $LOG_FILE"
echo "Checkpoints: $CKPT_DIR"

python "$PROJECT_DIR/main.py" \
    --config "$CONFIG_FILE" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --save_dir "$CKPT_DIR" \
    --train \
    2>&1 | tee "$LOG_FILE"

echo "Training finished. Logs saved to $LOG_FILE"
