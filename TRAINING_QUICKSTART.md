# Training Quickstart

This guide answers three things:
1. What to install now on a Windows laptop.
2. What training means in this repository.
3. How Unsloth plus TRL GRPO are used by the competition workflow.

---

## 1) Install Now on Windows (Immediate Local Setup)

Run from repository root:

```powershell
python -m pip install --upgrade pip
pip install -e .
pip install pytest httpx
pip install -r requirements-training.txt
```

Optional local attempt (may fail on native Windows, expected):

```powershell
pip install unsloth bitsandbytes
```

Why this is enough right now:
- It validates evaluator, reward parser, and training script flow.
- It lets you run dry-run training before GPU time.

Validation commands:

```powershell
python -m pytest tests/test_training_pipeline.py -q
python training/train_grpo.py
```

If Unsloth is installed but you are on CPU-only laptop, force dry-run mode:

```powershell
$env:FORCE_DRY_RUN = "1"
python training/train_grpo.py
```

Expected local behavior:
- If Unsloth is unavailable, script runs dry-run checks and confirms pipeline readiness.

---

## 2) Exact Colab Setup (Recommended for Real Training)

Use this when you want actual model updates, not dry-run.

### Colab Runtime
1. Runtime -> Change runtime type -> GPU (T4 is sufficient).

### Cell 1: Install dependencies

```python
!pip -q install --upgrade pip
!pip -q install -r requirements-training.txt
```

### Cell 2: Get repository

```python
%cd /content
!git clone https://github.com/balloonmann/financial-audit-env.git
%cd /content/financial-audit-env
```

### Cell 3: Install project package

```python
!pip -q install -e .
```

### Cell 4: Run training

```python
!python training/train_grpo.py
```

Alternative one-command path:

```python
!bash scripts/colab_bootstrap_and_train.sh
```

HF Jobs one-command path:

```bash
hf jobs run --flavor a10g-large --timeout 6h --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -lc "set -euo pipefail; apt-get update -qq; apt-get install -y -qq git; git clone https://github.com/balloonmann/financial-audit-env.git; cd financial-audit-env; bash scripts/hf_jobs_bootstrap_and_train.sh"
```

Low-VRAM profile (recommended on T4 if you hit OOM):

```python
!bash scripts/colab_train_low_vram.sh
```

### Cell 5: Save artifacts

```python
!ls -la
!ls -la grpo-financial-audit-adapter
```

---

## 3) What Training Means Here

In this project, training means:
1. Model generates findings for seeded tasks.
2. Environment computes deterministic score/reward.
3. GRPO updates LoRA adapter to increase reward on train seeds.
4. Candidate behavior is checked on held-out seeds.

Key files:
- `training/train_grpo.py`: Unsloth model load + GRPO loop.
- `training/evaluator.py`: in-process deterministic evaluation.
- `training/reward.py`: output parsing + reward shaping.
- `financial_audit_env/server/graders.py`: scoring primitives.

---

## 4) Will Judges Use Unsloth/GRPO to Test Environment?

Practical expectation:
1. Environment quality is judged by reproducibility, API correctness, scoring logic, and demo reliability.
2. Unsloth plus GRPO are judged as your training pipeline capability and evidence of improvement.
3. Best scoring signal comes from clear before/after held-out metrics, not from claiming a single run.

---

## 5) What Unsloth and GRPO Do Right Now

Right now in this repo:
1. Unsloth provides efficient 4-bit model loading and LoRA adaptation path.
2. TRL GRPOTrainer performs reward-driven optimization using your evaluator/reward function.
3. Without GPU dependencies, script falls back to dry-run verification so you can still validate pipeline integrity.

---

## 6) Brownie-Point Checklist

Before competition, make sure you can show:
1. Fixed train and held-out seed split.
2. Before versus after held-out table.
3. One ablation (for example overseer on vs off).
4. Reproducible command path and logged hyperparameters.
5. Passing focused tests and training pipeline tests.

---

## 7) Common Failure Fixes

1. Module import error from direct script run:
- Use `python training/train_grpo.py` from repo root (already supported).

2. CUDA/Unsloth issues on Windows:
- Use Colab for actual training.

3. Error: "Unsloth cannot find any torch accelerator":
- This means no supported GPU is available.
- Use `FORCE_DRY_RUN=1` locally for pipeline checks.
- Run real training in Colab GPU runtime.

4. Out-of-memory in Colab:
- Reduce sequence length.
- Reduce batch size.
- Reduce number of generations.

5. Easy Colab command set:
- `bash scripts/colab_bootstrap_and_train.sh` for full setup + train.
- `bash scripts/colab_train_low_vram.sh` for safer memory profile.

6. HF Jobs error `No module named 'mergekit'`:
- Use the pinned dependency path (`requirements-training.txt`) or run `bash scripts/hf_jobs_bootstrap_and_train.sh`.
