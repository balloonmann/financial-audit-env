#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Verifying GPU"
nvidia-smi

echo "[2/6] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[3/6] Installing project package"
python -m pip install -e .

echo "[4/6] Installing pinned training dependencies"
python -m pip install -r requirements-training.txt
python -m pip uninstall -y torchaudio || true
python -m pip check || true

echo "[5/6] Verifying runtime imports"
python - <<'PY'
import torch
import transformers
import trl
from trl import GRPOConfig, GRPOTrainer

print("versions", torch.__version__, transformers.__version__, trl.__version__)
print("imports_ok", GRPOConfig.__name__, GRPOTrainer.__name__)
PY

echo "[6/6] Running HF Jobs training"
python -u hf_jobs_train.py

echo "Done. Artifacts are in ./artifacts and adapter in ./grpo-financial-audit-adapter"