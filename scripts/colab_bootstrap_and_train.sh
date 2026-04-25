#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Verifying GPU"
nvidia-smi

echo "[2/6] Upgrading pip"
python -m pip install --upgrade pip

echo "[3/6] Installing training dependencies"
pip install --extra-index-url https://download.pytorch.org/whl/cu124 --upgrade --force-reinstall torch==2.6.0+cu124 torchvision==0.21.0+cu124
pip install -r requirements-training.txt
pip install --no-deps unsloth==2025.11.1 unsloth_zoo==2025.11.2
pip install --no-deps mergekit==0.1.4
pip install immutables scipy
pip uninstall -y torchaudio || true
python -m pip check || true

echo "[4/6] Installing project package"
pip install -e .

echo "[5/6] Running training pipeline test"
python -m pytest tests/test_training_pipeline.py -q

echo "[6/6] Running training"
python training/train_grpo.py

echo "Done. If successful, adapter is in ./grpo-financial-audit-adapter"
