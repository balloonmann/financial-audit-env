"""
GRPO Training Script — designed to run in Google Colab.
Uses Unsloth + TRL GRPOTrainer for memory-efficient training on T4.

Usage (Colab):
    1. Upload this file + financial_audit_env/ + training/ to Colab
    2. Run setup cell: !pip install unsloth trl datasets peft
    3. Execute: !python training/train_grpo.py

Pre-onsite: Verify it runs without OOM on free T4.
Onsite: Run with HF compute credits for actual training.
"""

import json
import os
import sys
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Colab setup — uncomment these in Colab
# ---------------------------------------------------------------------------
# !pip install unsloth trl datasets peft

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 4096
LORA_R = 16
LORA_ALPHA = 16
TRAIN_SEEDS = list(range(42, 52))       # 10 training seeds
HELD_OUT_SEEDS = list(range(100, 105))  # 5 held-out seeds
TASKS = ["expense_audit", "invoice_match", "gst_reconciliation", "fraud_detection"]


def setup_model():
    """Load model with Unsloth for memory efficiency on free T4."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Unsloth not installed. Install with: pip install unsloth")
        print("Falling back to dry-run mode for verification.")
        return None, None

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
    )
    return model, tokenizer


def create_dataset():
    """Create training dataset of audit prompts with task/seed metadata."""
    try:
        from datasets import Dataset
    except ImportError:
        print("datasets not installed. Install with: pip install datasets")
        return None

    from financial_audit_env.server.tasks import TASKS as TASK_DEFS

    prompts = []
    for task_id, task in TASK_DEFS.items():
        for seed in TRAIN_SEEDS:
            prompts.append({
                "prompt": (
                    f"You are a financial auditor performing: {task['name']}.\n\n"
                    f"{task['description']}\n\n"
                    f"Analyze the financial documents and report your findings as a JSON array.\n"
                    f"Each finding must have: document_id, error_type, description, and confidence (0.0-1.0).\n\n"
                    f"Valid error types: {', '.join(task['error_types'])}\n\n"
                    f"Output format:\n"
                    f'[{{"document_id": "...", "error_type": "...", "description": "...", "confidence": 0.85}}]\n\n'
                    f"Be precise — false positives will lower your score. "
                    f"Be thorough — missed errors will also lower your score."
                ),
                "task_id": task_id,
                "seed": seed,
            })

    return Dataset.from_list(prompts)


def create_reward_fn():
    """Create reward function using in-process evaluator (no HTTP overhead)."""
    from training.evaluator import InProcessEvaluator
    from training.reward import parse_findings_from_text

    evaluator = InProcessEvaluator()

    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        """
        GRPO reward function.
        Parses each completion, evaluates against ground truth.
        Returns F1-based scores in [0.01, 0.99].
        """
        task_ids = kwargs.get("task_id", ["expense_audit"] * len(completions))
        seeds = kwargs.get("seed", [42] * len(completions))

        # Handle single values
        if isinstance(task_ids, str):
            task_ids = [task_ids] * len(completions)
        if isinstance(seeds, int):
            seeds = [seeds] * len(completions)

        rewards = []
        for comp, tid, s in zip(completions, task_ids, seeds):
            try:
                findings = parse_findings_from_text(comp)
                result = evaluator.evaluate(tid, s, findings)
                rewards.append(result["score"])
            except Exception:
                rewards.append(0.01)
        return rewards

    return reward_fn


def evaluate_held_out(model=None, tokenizer=None):
    """Evaluate on held-out seeds to measure actual improvement."""
    from training.evaluator import InProcessEvaluator

    evaluator = InProcessEvaluator()
    results = {}

    for task_id in TASKS:
        task_results = []
        for seed in HELD_OUT_SEEDS:
            # Get baseline (empty submission) score
            baseline = evaluator.run_empty_episode(task_id, seed)
            task_results.append({
                "seed": seed,
                "baseline_score": baseline["score"],
            })
            # TODO: When model is available, generate actual findings and evaluate
        results[task_id] = task_results

    return results


def train():
    """Main training loop."""
    print("=" * 60)
    print("Financial Audit GRPO Training")
    print("=" * 60)

    # Step 1: Load model
    print("\n[1/5] Loading model...")
    model, tokenizer = setup_model()

    if model is None:
        print("\n[DRY RUN] Unsloth not available. Verifying pipeline components...\n")
        # Verify all components work without actual model
        print("  Checking evaluator...", end=" ")
        from training.evaluator import InProcessEvaluator
        ev = InProcessEvaluator()
        result = ev.evaluate("expense_audit", 42, [
            {"document_id": "EXP-001", "error_type": "over_limit", "description": "test"}
        ])
        print(f"OK (score={result['score']})")

        print("  Checking reward parser...", end=" ")
        from training.reward import parse_findings_from_text
        parsed = parse_findings_from_text('[{"document_id": "EXP-001", "error_type": "over_limit", "description": "test"}]')
        print(f"OK (parsed {len(parsed)} findings)")

        print("  Checking dataset creation...", end=" ")
        ds = create_dataset()
        if ds:
            print(f"OK ({len(ds)} prompts)")
        else:
            print("SKIP (datasets not installed)")

        print("  Checking reward function...", end=" ")
        reward_fn = create_reward_fn()
        rewards = reward_fn(["No findings found."], task_id=["expense_audit"], seed=[42])
        print(f"OK (reward={rewards[0]})")

        print("  Checking held-out evaluation...", end=" ")
        held_out = evaluate_held_out()
        total_seeds = sum(len(v) for v in held_out.values())
        print(f"OK ({total_seeds} seed evaluations)")

        print("\n[OK] All pipeline components verified. Ready for Colab with Unsloth.")
        return {"status": "dry_run_passed"}

    # Step 2: Create dataset
    print("\n[2/5] Creating dataset...")
    dataset = create_dataset()
    if dataset is None:
        print("ERROR: Could not create dataset")
        return {"status": "failed", "reason": "dataset_creation_failed"}
    print(f"  Created {len(dataset)} training prompts")

    # Step 3: Create reward function
    print("\n[3/5] Setting up reward function...")
    reward_fn = create_reward_fn()

    # Step 4: Train with GRPO
    print("\n[4/5] Starting GRPO training...")
    try:
        from trl import GRPOTrainer, GRPOConfig

        config = GRPOConfig(
            output_dir="./grpo-financial-audit",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            num_generations=4,            # Generate 4 completions per prompt
            max_completion_length=1024,
            logging_steps=5,
            save_steps=50,
            learning_rate=5e-6,
            report_to="none",             # Disable wandb for now
        )

        trainer = GRPOTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            reward_funcs=reward_fn,
            tokenizer=tokenizer,
        )

        trainer.train()
        print("  Training complete!")

    except ImportError:
        print("  TRL not installed. Install with: pip install trl")
        return {"status": "failed", "reason": "trl_not_installed"}

    # Step 5: Save and evaluate
    print("\n[5/5] Saving adapter and evaluating...")
    output_dir = "./grpo-financial-audit-adapter"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Adapter saved to {output_dir}")

    held_out_results = evaluate_held_out(model, tokenizer)
    print(f"  Held-out evaluation: {json.dumps(held_out_results, indent=2)}")

    print("\n" + "=" * 60)
    print("[OK] Training complete. Push adapter to HF Hub:")
    print(f"  model.push_to_hub('your-username/financial-audit-grpo')")
    print("=" * 60)

    return {
        "status": "completed",
        "adapter_path": output_dir,
        "held_out_results": held_out_results,
    }


if __name__ == "__main__":
    train()
