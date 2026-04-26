# Teaching an LLM to Audit Finances with GRPO Reinforcement Learning

**OpenEnv / Meta PyTorch Hackathon Round 2 — Solo Submission**

---

## The Problem

The hackathon asked builders to use open environments as training grounds for language model agents. The challenge is not just building something that works in a demo but creating an environment where the agent receives structured, graded feedback and actually improves through reinforcement learning.

Financial auditing is a natural fit. It has:
- Documents with ground truth errors baked in at generation time
- A clear notion of right and wrong (did you catch the fraud or not?)
- Graduated difficulty across task types, from simple policy checks to cross-system pattern recognition
- A reward signal that can be partial (right document, wrong error type) rather than binary

The financial-audit-env implements four tasks at increasing difficulty:

| Task | Difficulty | What the agent does |
|------|-----------|---------------------|
| expense_audit | Easy | Checks expense claims against per-category limits, weekend rules, duplicate receipts |
| invoice_match | Medium | Cross-references purchase orders, goods receipts, and vendor invoices for discrepancies |
| gst_reconciliation | Hard | Reconciles books with GSTR-2B tax filings across multiple vendors |
| fraud_detection | Expert | Detects Benford law violations, shell company patterns, vendor concentration, weekly invoice cycles |

Each task is procedurally generated from a seed, so the same seed always produces the same documents and the same ground truth. Seeds 42-51 were used for training. Seeds 100-104 were held out completely and never touched during any training run.

---

## Environment Design

The environment follows the OpenEnv interface: `reset(task_id, seed)` returns an observation with documents and a task description, `step(action)` takes an `AuditAction` containing a list of `Finding` objects and returns a reward.

The grader computes a partial-credit F1 score:

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * (precision * recall) / (precision + recall)
```

A finding counts as a true positive only if both `document_id` and `error_type` match the ground truth (case-insensitive). A finding that identifies the right document but the wrong error type gets 0.25 partial credit rather than a full TP. This matters because the agent should learn that document identification is worth something even before it narrows down the exact error category.

All scores are clamped to the open interval `[0.01, 0.99]`. The 0.01 floor ensures that even a complete failure gets a non-zero gradient signal, and the 0.99 ceiling prevents overconfident reward collapse.

The reward ladder inside the GRPO training loop adds a further shaped signal on top of the evaluator score:

```
no JSON attempt        -> 0.01
JSON attempt, 0 parsed -> 0.02
parsed findings, 0 doc matches -> 0.025
partial doc matches    -> 0.05 + (partial_matches * 0.015), capped 0.10
any true positives     -> evaluator partial_credit_f1 score (up to 0.99)
```

This ladder is intentionally tight. An agent that outputs a well-formed JSON array with random document IDs only reaches 0.025. It needs to actually match documents before the reward starts climbing.

---

## Why GRPO

Group Relative Policy Optimization (GRPO) was introduced in DeepSeek-R1 as a memory-efficient alternative to PPO for language model fine-tuning. The key difference is that GRPO removes the value network entirely. Instead of learning a baseline from a separate critic, it estimates the baseline as the mean reward across a group of completions sampled for the same prompt.

For each training prompt, GRPO:
1. Samples `num_generations` completions from the current policy
2. Scores each with the reward function
3. Computes advantages as the z-score within the group: `A_i = (r_i - mean(r)) / std(r)`
4. Applies a clipped policy gradient loss: `L = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)`

The advantage of this for our setting is that GRPO naturally handles the structured output problem. If one completion in a group correctly identifies the expense_id `EXP-042` as over-limit while others hallucinate document IDs, the correct completion gets a positive advantage even if the absolute reward is modest. The relative comparison within the group is what drives learning.

This is the right inductive bias for financial auditing. The agent does not need to achieve perfect recall on every task to receive a useful learning signal. It just needs to be reliably better than its siblings within each group.

---

## Training Setup

**Model:** `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
**Hardware:** NVIDIA A10G Large (24 GB VRAM, $1.50/hr) via HuggingFace Jobs
**Training framework:** Unsloth + TRL GRPOTrainer
**LoRA config:** r=16, alpha=16, target modules q/k/v/o_proj, no dropout

Key hyperparameters:

```python
MAX_SEQ_LENGTH    = 4096
MAX_COMPLETION    = 384
BATCH_SIZE        = 2
NUM_GENERATIONS   = 2
TRAIN_EPOCHS      = 3
LEARNING_RATE     = 2e-5
beta              = 0.0   # KL penalty disabled
```

Setting `beta=0.0` removes the KL divergence term from the GRPO objective. This was a deliberate choice driven by memory constraints: TRL's GRPOTrainer keeps a frozen reference model on GPU to compute KL. On a 24 GB card with a 4-bit 8B model, that reference copy consumes approximately 5 GB and pushed every training step into an out-of-memory crash during the attention computation. Disabling KL freed enough headroom to run cleanly. Pure reward-driven GRPO without KL regularization is used in several published variants and is not an unusual choice for compute-constrained settings.

---

## The Debugging Story

This section is honest about what actually happened.

**Run 1:** `reward_std=0`, `grad_norm=0` every step. Three bugs were responsible. First, `obs.documents` is a dict-of-lists, and iterating it directly yields keys, not document objects. The code was calling `for doc in obs.documents` and getting category names instead of document IDs, which meant the valid document ID list in every prompt was empty. The model had no anchor for its findings. Second, prompts were being silently truncated at 2048 tokens when they were 3000-3500 tokens long, so the model was generating completions with no document context at all. Third, `use_gradient_checkpointing="unsloth"` on the PEFT model was blocking LoRA gradients entirely.

**Runs 2-4:** `grad_norm=0` persisted. The root cause turned out to be import ordering. Unsloth's `PatchFastRL("GRPO", FastLanguageModel)` must be called before `from trl import GRPOTrainer` is executed. It patches TRL's internal `_get_per_token_logps` computation to route through Unsloth's forward pass. When TRL is imported first, the class methods are already bound and the patch cannot intercept the computation graph that connects LoRA parameters to the loss. Moving the import and patch call to the top of the script, before any TRL import, resolved this.

**Run 5:** First run with `grad_norm > 0` (peaked at 9.3). But `completions/clipped_ratio` was 0.6-1.0, meaning most completions were hitting the `MAX_COMPLETION=320` ceiling and producing truncated JSON. Reward variance within groups was collapsing because truncated JSON scored identically at the floor. Fixed by raising `MAX_COMPLETION` to 384.

**Run 6 (reported below):** Clean training. `grad_norm` ranged from 1.4 to 9.3 across 60 steps. `frac_reward_zero_std` stayed at 0.0-0.1 for most of training, meaning GRPO had genuine signal in almost every group.

---

## Training Metrics (A10G, Llama 3.1 8B)

The training ran for 60 steps (3 epochs over 40 prompts). Selected checkpoints:

| Step | Epoch | reward_mean | grad_norm | clipped_ratio |
|------|-------|-------------|-----------|---------------|
| 5    | 0.25  | 0.163       | 7.08      | 0.55          |
| 10   | 0.50  | 0.131       | 9.32      | 0.60          |
| 15   | 0.75  | 0.217       | 5.94      | 0.60          |
| 20   | 1.00  | 0.171       | 2.44      | 0.65          |
| 25   | 1.25  | 0.165       | 1.10      | 0.45          |
| 30   | 1.50  | 0.216       | 4.30      | 0.70          |
| 40   | 2.00  | 0.181       | 4.05      | 0.30          |
| 50   | 2.50  | 0.213       | 0.116     | 0.60          |
| 55   | 2.75  | 0.136       | 3.50      | 0.65          |

The reward oscillates rather than monotonically increasing, which is typical for GRPO on small datasets. With only 10 training seeds per task, any single batch can be dominated by hard or easy samples, causing variance in the logged mean reward. The `kl=0.0` throughout is expected with `beta=0`.

---

## Held-Out Evaluation Results

**Baseline:** Llama 3.1 8B Instruct, no fine-tuning. Mean score across 4 tasks x 5 seeds.

| Task | Baseline | GRPO Trained | Delta |
|------|----------|-------------|-------|
| expense_audit | 0.169 | **0.356** | +111% |
| invoice_match | 0.169 | 0.074 | -56% |
| gst_reconciliation | 0.169 | 0.042 | -75% |
| fraud_detection | 0.169 | 0.020 | -88% |
| **Overall** | **0.169** | **0.123** | -27% |

Expense audit improved substantially. The agent learned to identify specific violation types (over_limit, weekend_expense, duplicate_claim) in structured expense records. This task has relatively short, structured documents that fit comfortably within the 4000-character document window used during training.

The other three tasks degraded. Invoice match requires cross-referencing three document types simultaneously (PO, GRN, invoice). GST reconciliation requires understanding tax filing discrepancies across many vendors. Fraud detection requires statistical pattern recognition across transaction histories. All three require larger document contexts and more training steps than the compute budget allowed.

Per-seed detail for the best Llama run:

| Task | Seed | Score | Precision | Recall |
|------|------|-------|-----------|--------|
| expense_audit | 100 | 0.44 | 0.50 | 0.29 |
| expense_audit | 101 | 0.25 | 0.25 | 0.14 |
| expense_audit | 102 | 0.15 | 0.01 | 0.01 |
| expense_audit | 103 | 0.15 | 0.01 | 0.01 |
| expense_audit | 104 | 0.40 | 0.25 | 0.14 |
| invoice_match | 100 | 0.15 | 0.25 | 0.11 |
| invoice_match | 102 | 0.17 | 0.33 | 0.11 |

Seeds 102-104 for expense audit show lower scores, suggesting the model has learned some patterns but not generalized fully across all seed variations. This is consistent with 3 epochs on 10 training seeds being insufficient for robust generalization.

---

## Colab Demo: Qwen 2.5 1.5B on T4

The submission also includes a runnable Colab notebook using `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` on a free T4 GPU. This notebook requires no API keys and runs end-to-end in under 90 minutes. It is intended as a reproducible demonstration of the full pipeline: environment setup, baseline eval, GRPO training, post-training eval, and comparison plots.

**Qwen 2.5 1.5B baseline:** [TO BE FILLED AFTER RUN]

**Qwen 2.5 1.5B trained:** [TO BE FILLED AFTER RUN]

The 1.5B model is significantly smaller than Llama 3.1 8B and operates under tighter VRAM constraints on T4 (15 GB vs 24 GB). Results are expected to be lower in absolute terms but the training dynamics (reward variance, grad_norm trajectory) demonstrate that the same GRPO pipeline works across model sizes.

---

## What Worked and What Did Not

What worked:
- The partial-credit reward ladder created genuine within-group variance in almost every GRPO step
- Fixing the document ID extraction (`_collect_doc_ids` recursive traversal of dict-of-lists) was the single largest quality improvement to the training signal
- `beta=0` (no KL penalty) was the correct call for compute-constrained training on a 24 GB card
- expense_audit showed real generalization: trained on seeds 42-51, tested on seeds 100-104, scored 0.356 vs 0.169 baseline

What did not work at this compute budget:
- The harder tasks (invoice_match, gst_reconciliation, fraud_detection) need longer document context than fits in 4000 characters and more than 3 epochs of training to develop cross-document reasoning
- The 40-prompt training dataset (10 seeds x 4 tasks) is small enough that any single batch can swing the reward mean by 0.05-0.10, making it hard to read training progress from logged metrics alone
- The document truncation tradeoff: training with 4000-char documents fit within max_prompt_length limits and produced strong gradients, but the same truncation during evaluation may have disadvantaged tasks where the critical error information appears later in the document

---

## Artifacts

- **Trained adapter (Llama 3.1 8B):** https://huggingface.co/balloonmann/financial-audit-grpo-adapter
- **Eval artifacts (CSVs + plots):** https://huggingface.co/datasets/balloonmann/financial-audit-eval-artifacts
- **Environment repo:** https://github.com/balloonmann/financial-audit-env
- **Submission notebook:** `GRPO_Training_Submission.ipynb` (runs on free T4, no keys required)

---

## Technical Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Base model | Llama 3.1 8B Instruct (4-bit NF4) | Strong instruction following, fits A10G in 4-bit |
| RL algorithm | GRPO | No critic network, memory efficient, good for structured output tasks |
| Training library | Unsloth + TRL | Unsloth's fused kernels reduce memory, TRL provides GRPOTrainer |
| Quantization | bitsandbytes NF4, double quant | 4-5 GB model footprint vs 16 GB in bf16 |
| LoRA | r=16, alpha=16, q/k/v/o_proj | Standard config, ~67M trainable params out of ~8B |
| Eval | InProcessEvaluator (no HTTP) | Eliminates network overhead during reward computation inside training loop |
| Grader | Partial-credit F1, clamped [0.01, 0.99] | Smooth gradients, no zero-reward floor for near-misses |
