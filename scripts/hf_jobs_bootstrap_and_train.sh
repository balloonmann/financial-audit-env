#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Verifying GPU"
nvidia-smi

echo "[2/6] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[3/6] Installing project package"
python -m pip install -e .

echo "[4/6] Installing pinned training dependencies"
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 --upgrade --force-reinstall torch==2.6.0+cu124 torchvision==0.21.0+cu124
python -m pip install -r requirements-training.txt
python -m pip install --no-deps unsloth==2025.11.1 unsloth_zoo==2025.11.2
python -m pip install --no-deps mergekit==0.1.4
python -m pip install immutables scipy
python -m pip uninstall -y torchaudio || true
python -m pip check || true

export BNB_CUDA_VERSION=124

echo "[5/6] Verifying runtime imports"
python - <<'PY'
import torch
import transformers
import trl
import bitsandbytes as bnb
from trl import GRPOConfig, GRPOTrainer

print("versions", torch.__version__, transformers.__version__, trl.__version__)
print("imports_ok", GRPOConfig.__name__, GRPOTrainer.__name__)
print("bnb", bnb.__version__)

# Fail fast if bitsandbytes did not load CUDA quantization symbols.
from bitsandbytes.cextension import lib
if not hasattr(lib, "cquantize_blockwise_fp16_nf4"):
	raise RuntimeError("bitsandbytes CUDA kernels not loaded (nf4 quantization symbol missing)")
PY

echo "[6/6] Running HF Jobs training"
python -u hf_jobs_train.py

echo "Done. Artifacts are in ./artifacts and adapter in ./grpo-financial-audit-adapter"