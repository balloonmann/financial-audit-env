#!/usr/bin/env bash
set -euo pipefail

echo "Applying low-VRAM training profile"
export TRAIN_MAX_SEQ_LENGTH=1536
export TRAIN_BATCH_SIZE=1
export TRAIN_NUM_GENERATIONS=2
export TRAIN_MAX_COMPLETION_LENGTH=512
export TRAIN_EPOCHS=1
export TRAIN_LOGGING_STEPS=5
export TRAIN_SAVE_STEPS=50
export TRAIN_LEARNING_RATE=5e-6

python training/train_grpo.py
