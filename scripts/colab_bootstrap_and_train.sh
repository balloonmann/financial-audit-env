#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Verifying GPU"
nvidia-smi

echo "[2/6] Upgrading pip"
python -m pip install --upgrade pip

echo "[3/6] Installing training dependencies"
pip install unsloth trl datasets peft accelerate bitsandbytes pytest httpx

echo "[4/6] Installing project package"
pip install -e .

echo "[5/6] Running training pipeline test"
python -m pytest tests/test_training_pipeline.py -q

echo "[6/6] Running training"
python training/train_grpo.py

echo "Done. If successful, adapter is in ./grpo-financial-audit-adapter"
