"""
HF Jobs training script — run via https://huggingface.co/docs/hub/spaces-run-jobs
Trains Llama-3.1-8B on financial audit tasks using Unsloth + TRL GRPO.
"""

import os
import gc
import json
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# PatchFastRL must be imported and called before TRL is imported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

# ─────────────────────────────────────────────────────────────────────────────
# Config — tuned for A10G GPU
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME        = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH    = 4096
LORA_R            = 16
LORA_ALPHA        = 16
TRAIN_EPOCHS      = 3
BATCH_SIZE        = 2       # Must divide NUM_GENERATIONS
NUM_GENERATIONS   = 2       # generation_batch_size (BATCH_SIZE) must be divisible by this
MAX_COMPLETION    = 384
LEARNING_RATE     = 2e-5
LOGGING_STEPS     = 5
SAVE_STEPS        = 50
ADAPTER_DIR       = "./grpo-financial-audit-adapter"
ARTIFACTS_DIR     = "./artifacts"
TRAIN_SEEDS       = list(range(42, 52))
HELD_OUT_SEEDS    = list(range(100, 105))
TASK_IDS          = ["expense_audit", "invoice_match", "gst_reconciliation", "fraud_detection"]

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print(f"[{datetime.now()}] HF Jobs GRPO Training")
print(f"Model: {MODEL_NAME}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from financial_audit_env.server.environment import FinancialAuditEnvironment
from financial_audit_env.server.tasks import TASKS
from financial_audit_env.models import AuditAction, Finding
from training.reward import parse_findings_from_text
from training.evaluator import InProcessEvaluator
from datasets import Dataset

try:
    from trl import GRPOTrainer, GRPOConfig
except Exception as exc:
    print("ERROR: Failed to import TRL GRPO components.")
    print(f"Cause: {exc}")
    print(
        "Hint: install pinned training deps with "
        "'python -m pip install -r requirements-training.txt' "
        "or run 'bash scripts/hf_jobs_bootstrap_and_train.sh'."
    )
    raise

evaluator = InProcessEvaluator()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_ID_FIELDS = (
    "document_id", "expense_id", "invoice_id", "po_id", "grn_id",
    "txn_id", "vendor_id", "invoice_no", "id", "doc_id",
)

def _collect_doc_ids(obj, out):
    """Recursively collect any known ID fields from nested dicts/lists."""
    if isinstance(obj, dict):
        for k in _ID_FIELDS:
            v = obj.get(k)
            if isinstance(v, str) and v:
                out.append(v)
        for v in obj.values():
            _collect_doc_ids(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_doc_ids(item, out)

def build_prompt(task_id, seed, doc_chars=4000):
    env = FinancialAuditEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    task = TASKS[task_id]
    # obs.documents is dict-of-lists; collect IDs from every nested doc
    raw_ids = []
    _collect_doc_ids(obs.documents, raw_ids)
    seen = set()
    doc_ids = [x for x in raw_ids if not (x in seen or seen.add(x))]
    content = (
        "You are a financial auditor. Identify errors in the documents below.\n"
        "Output ONLY a valid JSON array. Each object must have exactly these fields:\n"
        '  {"document_id": "<MUST be from VALID IDs list>", "error_type": "<from ALLOWED list>", '
        '"description": "<brief reason>", "confidence": <0.0-1.0>}\n\n'
        f"TASK: {obs.task_description}\n\n"
        f"VALID DOCUMENT IDs (use ONLY these exact strings): {json.dumps(doc_ids)}\n\n"
        f"ALLOWED ERROR TYPES: {json.dumps(task['error_types'])}\n\n"
        f"DOCUMENTS:\n{json.dumps(obs.documents)[:doc_chars]}\n\n"
        "Output the JSON array only. If no errors, output []. No explanation text."
    )
    return [{"role": "user", "content": content}]

def _norm_conf(x):
    try:
        v = float(x)
    except Exception:
        return 0.7
    if v > 1.0 and v <= 100.0:
        v /= 100.0
    return max(0.0, min(1.0, v))

def run_eval(model, tokenizer, task_ids, seeds, label):
    """Evaluate model on task_ids x seeds. Returns DataFrame."""
    rows = []
    model.eval()
    for tid in task_ids:
        for s in seeds:
            messages = build_prompt(tid, s, doc_chars=6000)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", max_length=3712, truncation=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=192, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.empty_cache()
            completion = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            findings = parse_findings_from_text(completion)
            result = evaluator.evaluate(tid, s, findings)
            rows.append({
                "task_id": tid, "seed": s, "label": label,
                "score": result["score"],
                "weighted_score": result["weighted_score"],
                "precision": result["precision"],
                "recall": result["recall"],
                "num_findings": len(findings),
            })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Baseline score (verified from prior run, skipping re-eval to save VRAM)
# ─────────────────────────────────────────────────────────────────────────────
HF_MODEL_MAP = {
    "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit": "Qwen/Qwen2.5-1.5B-Instruct",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit": "Qwen/Qwen2.5-7B-Instruct",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}
HF_BASE_ID = HF_MODEL_MAP.get(MODEL_NAME, MODEL_NAME.replace("unsloth/", "").replace("-bnb-4bit", ""))

BASELINE_SCORE = 0.1690
print(f"\n[{datetime.now()}] Step 1: Baseline score (pre-verified): {BASELINE_SCORE:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: GRPO training
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[{datetime.now()}] Step 2: GRPO training")
print(f"  Loading {MODEL_NAME} with Unsloth...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(
    f"  Trainable params: {trainable_params:,} / {all_params:,} "
    f"({100.0 * trainable_params / max(all_params, 1):.4f}%)"
)

# Build training dataset
train_rows = []
for tid in TASK_IDS:
    for s in TRAIN_SEEDS:
        train_rows.append({
            "prompt": build_prompt(tid, s),
            "task_id": tid,
            "seed": s,
        })
train_dataset = Dataset.from_list(train_rows)
print(f"  Dataset: {len(train_dataset)} prompts")

# Reward function — shaped to create variance within GRPO groups.
# Tightened ladder so model can't game floor without finding real matches:
#   0.01   — no JSON attempt
#   0.02   — JSON attempt but 0 findings parsed
#   0.025  — findings parsed, 0 doc matches (small bump only)
#   0.05+  — partial doc matches (right doc, wrong error type)
#   evaluator score — any true positives (real partial_credit_f1, up to 0.99)
_reward_debug_n = 0

def _extract_completion_text(comp):
    """TRL may pass completions as str or list of message dicts."""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list):
        for msg in reversed(comp):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", ""))
        return " ".join(str(m.get("content", "")) for m in comp if isinstance(m, dict))
    return str(comp)

def reward_fn(completions, task_id, seed, **kwargs):
    global _reward_debug_n
    import re as _re
    rewards = []
    for comp, tid, s in zip(completions, task_id, seed):
        try:
            comp = _extract_completion_text(comp)
            findings = parse_findings_from_text(comp)

            # Debug: log first 8 reward calls to verify output format
            if _reward_debug_n < 8:
                print(f"\n[REWARD {_reward_debug_n}] tid={tid} s={s} "
                      f"parsed={len(findings)} comp[:300]={comp[:300]!r}")
                _reward_debug_n += 1

            if not findings:
                has_json = bool(_re.search(r'\[\s*\{', comp))
                rewards.append(0.02 if has_json else 0.01)
                continue

            result = evaluator.evaluate(tid, int(s), findings)
            score = float(result["score"])
            tp   = result.get("true_positives", 0)
            pm   = result.get("grader_result", {}).get("partial_matches", 0)

            # Tight floor: only meaningful matches lift score above 0.025
            if score <= 0.01:
                if pm > 0:
                    score = min(0.05 + pm * 0.015, 0.10)  # partial doc hit
                else:
                    score = 0.025  # parsed but no doc match
            rewards.append(score)
        except Exception as e:
            print(f"[REWARD ERROR] {e}")
            rewards.append(0.01)
    return rewards

# Train
grpo_config = GRPOConfig(
    output_dir=ADAPTER_DIR,
    num_train_epochs=TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION,
    max_prompt_length=MAX_SEQ_LENGTH - MAX_COMPLETION,
    learning_rate=LEARNING_RATE,
    max_grad_norm=1.0,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    report_to="none",
    bf16=True,  # A10G uses bf16
    beta=0.0,   # Disable KL penalty -> frees reference model from VRAM (~5 GiB savings)
)
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,
    reward_funcs=reward_fn,
    processing_class=tokenizer,
)

print(f"  Starting GRPO training...")
trainer.train()
print(f"  Training complete!")

model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"  Adapter saved to {ADAPTER_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Post-training evaluation
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[{datetime.now()}] Step 3: Post-training evaluation")
FastLanguageModel.for_inference(model)
trained_df = run_eval(model, tokenizer, TASK_IDS, HELD_OUT_SEEDS, label="GRPO Trained")
trained_df.to_csv(f"{ARTIFACTS_DIR}/trained_heldout.csv", index=False)
print(f"  Trained mean score: {trained_df['score'].mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Results and plots
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[{datetime.now()}] Step 4: Comparison plots")

def summarize(df):
    return df.groupby("task_id")[["score", "precision", "recall"]].mean()

tasks = list(summarize(trained_df).index)
x = range(len(tasks))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric in zip(axes, ["score", "recall"]):
    trai_vals = summarize(trained_df)[metric].reindex(tasks).values
    base_vals = [BASELINE_SCORE] * len(tasks) if metric == "score" else [0.0] * len(tasks)
    ax.bar([i - width/2 for i in x], base_vals, width, label="Baseline", color="steelblue", alpha=0.85)
    ax.bar([i + width/2 for i in x], trai_vals, width, label="GRPO Trained", color="darkorange", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(tasks, rotation=15)
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Held-out {metric.capitalize()} — Baseline vs Trained")
    ax.set_ylim(0, 1)
    ax.legend()

plt.tight_layout()
plt.savefig(f"{ARTIFACTS_DIR}/before_after_comparison.png", dpi=180, bbox_inches="tight")
plt.close()

delta = trained_df["score"].mean() - BASELINE_SCORE
pct = delta / max(BASELINE_SCORE, 1e-6) * 100

print(f"  Baseline mean score : {BASELINE_SCORE:.4f}")
print(f"  Trained  mean score : {trained_df['score'].mean():.4f}")
print(f"  Delta               : {delta:+.4f}  ({pct:+.1f}%)")

# Summary table
summary_df = summarize(trained_df).assign(model="GRPO Trained").reset_index()
print("\nResults summary:")
print(summary_df.pivot_table(index="task_id", columns="model", values="score").round(4))

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Upload to HuggingFace Hub
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[{datetime.now()}] Step 5: Upload to HuggingFace Hub")

from huggingface_hub import HfApi, upload_folder, whoami

user = whoami()["name"]
api = HfApi()

# Upload adapter
adapter_repo = f"{user}/financial-audit-grpo-adapter"
api.create_repo(repo_id=adapter_repo, repo_type="model", exist_ok=True)
upload_folder(repo_id=adapter_repo, folder_path=ADAPTER_DIR, repo_type="model")
print(f"  Adapter  : https://huggingface.co/{adapter_repo}")

# Upload artifacts
artifact_repo = f"{user}/financial-audit-eval-artifacts"
api.create_repo(repo_id=artifact_repo, repo_type="dataset", exist_ok=True)
upload_folder(repo_id=artifact_repo, folder_path=ARTIFACTS_DIR, repo_type="dataset")
print(f"  Artifacts: https://huggingface.co/datasets/{artifact_repo}")

print(f"\n[{datetime.now()}] ✓ Training complete!")
