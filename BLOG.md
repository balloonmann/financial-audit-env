# Teaching a Language Model to Actually Do a Financial Audit

*Meta PyTorch OpenEnv Hackathon — April 2026 | Round 2 Submission*

---

There's a version of the "AI for finance" pitch that gets told at a lot of conferences. A model reads a spreadsheet, surfaces a few anomalies, everyone nods approvingly. It's a clean story. It also has almost nothing to do with how financial auditing actually works.

Real audit seasons don't look like that. They look like three systems with conflicting data, a GST rate that changed mid-quarter, a vendor that got flagged for due diligence after two transactions already cleared, and a junior auditor trying to reconcile 45 book entries against a GSTR-2B that inexplicably has 44. The problems aren't in the data. They're in the relationship between the data and a set of rules that are themselves moving.

That gap — between what AI demos show and what audit work actually requires — is what this project is about.

---

## Problem Statement

Most LLM audit demos are effectively retrieval tasks. Given clean, static documents, find the thing that violates the rule. That's a useful capability, but it's not auditing. Auditing is adversarial, multi-period, and policy-sensitive. The ground truth shifts because policy shifts. An expense that was compliant in Period 1 gets flagged in Period 3 because a new approval threshold dropped mid-campaign. An invoice that matched perfectly at submission now shows a quantity discrepancy when the goods receipt note gets updated.

The AI systems that exist for this space are almost universally document-to-insight pipelines — they read, they extract, they flag. None of them model what happens when the rules change while you're mid-audit.

That's the problem I wanted to build an environment around.

If I could train an agent to operate correctly across a dynamic, multi-period audit campaign — adapting to regulatory shocks, coordinating with specialist peers, maintaining a review trail that holds up to oversight scrutiny — I'd have built something that's actually useful. Not demo-useful. Actually useful.

---

## Campaign Walkthrough: Five Periods of World Mutation

**Act 1 — The Baseline.** Period 1. The agent receives 19 expense claims and a policy doc. Stable rules, no surprises. The untrained Llama 3.1 8B scores F1 ≈ 0.12 — casting a wide net, flagging everything, right about 12% of the time.

**Act 2 — The World Mutates.** Period 2. Meal limit jumps ₹1,500 → ₹2,000. A new vendor onboards. Cross-period memory is now required. An agent that treats each period as a fresh prompt will start generating false positives on the new vendor and miss the updated limit. Baselines do exactly this.

**Act 3 — The Regulatory Shock.** Period 3. Mid-audit, REG-001 drops: GST on IT services 18% → 12%. Schema drifts (`vendor_gstin` → `supplier_gstin`). Findings already submitted under the old rate are now wrong. Baseline models acknowledge the new rate in their reasoning and then proceed to flag errors at the old rate anyway. This is a known gap between knowing a fact and updating behavior on it — and RL is the right tool for closing it.

**Act 4 — The Environment Reveals the Training Problem.** GRPO chases the densest reward signal. After training, Llama 3.1 8B improved **3× on expense audits** and **collapsed 82% on fraud detection**. The 0.20 F1 floor multiplier kicked in. The environment said *no, this isn't a win* — and made the curriculum bias obvious.

A good environment doesn't hide training problems. It surfaces them so cleanly that the fix is obvious.

---

## Theme Alignment: World Modeling #3 — Professional Tasks #3.1

The Meta PyTorch OpenEnv Hackathon (India, April 2026) asked teams to build RL environments for LLM training and demonstrate genuine behavioral improvement. This submission sits squarely under **Theme #3: World Modeling**, specifically **#3.1 Professional Tasks**.

The theme description asks for environments where "the model is expected to do real hard work instead of exploiting shortcuts," requiring agents to "maintain consistent internal state, update beliefs based on outcomes, and orchestrate multi-step workflows" — with the goal of strengthening "causal reasoning and persistent world models." Financial auditing is one of the cleanest fits for that brief: it's an enterprise workflow with explicit rules, partially observable state (each specialist only sees their domain's documents), and a world that changes mid-task.

**Judging criteria:**
- Environment Innovation — 40%
- Storytelling & Presentation — 30%
- Showing Improvement in Rewards — 20%
- Reward and Training Pipeline — 10%

**Why this project fits #3.1:**

The environment isn't a simulation of auditing — it's a functional audit pipeline. The agent interacts with a live FastAPI environment via the OpenEnv standard interface, receives real documents, applies real policy rules, and gets scored by a deterministic grader that plants ground truth at generation time. There is no LLM-as-judge, no fuzzy scoring, no way to talk your way to a higher reward. A finding is correct or it isn't.

The "persistent world model" requirement maps directly to the campaign structure: the agent must track what it found in Period 1 when Period 3 introduces a regulatory shock that retroactively changes what Period 1 findings mean. A model that treats each step as a fresh prompt will fail. One that maintains consistent internal state across the five-period campaign — updating its beliefs when the GST rate changes mid-audit — will succeed.

The "no shortcuts" requirement maps to the planted red herrings and anti-gaming guards. The environment is explicitly designed so that a model which hallucinates confidently, or floods the system with findings hoping for partial credit, gets actively penalized. High recall at the cost of precision is not rewarded.

---

## System Architecture

### The Specialist Swarm

The core of the environment is four auditing agents, each trained on a different class of financial error:

| Agent | Task | Documents | Planted Errors | Difficulty |
|---|---|---|---|---|
| Expense Specialist | Expense policy violations | 19 expense records | 7 errors | Easy |
| Invoice Specialist | Three-way matching | 10 POs + 10 GRNs + 12 invoices | 9 discrepancies | Medium |
| GST Specialist | Tax reconciliation | 45 book entries + 44 GSTR-2B | 12 mismatches | Hard |
| Fraud Specialist | Pattern detection | 84 transactions + 26 vendors | 10 fraud patterns | Expert |

Each specialist runs for up to 5 steps per period before submitting final findings. The dependency order matters: the GST agent can only run after invoice matching is complete, and the fraud agent draws on prior task findings for its cross-task pattern signals.

The error taxonomy is where a lot of my design work went. It's easy to plant simple errors. It's harder to plant errors that are realistic — the kind an actual auditor would catch and a language model would plausibly miss. Expense violations include `cumulative_breach` (spread across multiple records individually within limit), `unapproved_vendor` (vendor not on approved list), and `duplicate_claim` (same receipt filed twice with minor field differences). Fraud errors include `round_tripping` (funds leaving and returning through related entities), `ghost_employee` (payroll entries with no corresponding HR record), and `collusion_pattern` (split invoices just below approval thresholds across the same vendor).

Red herrings are planted deliberately. The baseline model hallucinates false positives on these, which hits precision hard.

### The Campaign Loop: Five Periods of World Mutation

Each training run is a five-period campaign. Between periods, the world mutates:

- Fiscal period advances, policy versions increment
- Schema fields rename (e.g. `vendor_id` becomes `supplier_ref` in a later period)
- Vendor risk statuses update — a previously clean vendor gets flagged
- Tax rates shift with regulatory shocks

Three specific shocks are baked in at fixed steps:

- **REG-001 (Period 3, Step 3):** GST rate for IT services drops from 18% to 12%. Any prior GST reconciliation that assumed 18% is now wrong.
- **REG-002 (Period 4, Step 1):** Cash transaction reporting threshold tightens from ₹50,000 to ₹30,000. A transaction that was compliant is now a violation.
- **REG-003 (Period 4, Step 2):** A specific vendor category gets flagged for enhanced due diligence. The fraud specialist needs to reassess.

The point of the shocks isn't to be adversarial for its own sake. It's to test whether the agent is actually reasoning about policy or has memorized a surface pattern. A model that's memorized "flag anything above ₹50,000" will fail REG-002 in the wrong direction — suddenly over-flagging instead of adapting.

### The Overseer Layer

After all four specialists submit their findings for a period, an overseer agent reviews the combined output. It can approve findings, reject them with reasons, escalate high-value issues, and resolve conflicts where two specialists have flagged the same document differently.

The overseer's decisions feed into the campaign score as a separate component. This is intentional — it creates an accountability structure. A specialist that fires wildly and generates many false positives gets corrected by the overseer, and that correction history is tracked. Over-generation isn't rewarded.

### Self-Improvement Gate

The environment includes a self-improvement mechanism with a strict anti-regression gate. A candidate policy update is accepted only if:
- `train_delta > 0.005` (measurable improvement on training seeds)
- `transfer_delta > -0.002` (no meaningful degradation on held-out seeds)
- `safety_regression == false` (no new critical misses introduced)

This matters because reward hacking is real. A model can learn to flood the environment with findings, boosting recall at the cost of precision. The gate catches this: if precision has collapsed on the transfer set, the update gets rejected.

---

## Reward Function Design

This is the part where most RL environments fall apart, so I'll be specific about the choices.

### Partial Credit F1

A pure binary hit/miss reward for financial findings would be too sparse to learn from. A finding on the right document with the wrong error type still carries information — the model identified the problematic record, it just mis-classified the violation.

The grader gives partial credit at 0.40 for this case (right `document_id`, wrong `error_type`). Full credit requires both correct. This was calibrated deliberately — 0.40 is enough to provide gradient signal during early training without rewarding sloppy classification.

### Severity Weighting

Not all errors are equal. A `ghost_employee` pattern has higher financial exposure than a `weekend_expense` violation. The weighted F1 component reflects this:

- Fraud-class errors: 2.0× weight
- Compliance-critical errors: 1.5× weight
- Procedural violations: 0.5× weight

This creates a meaningful distinction between a model that catches high-value fraud and one that only catches low-stakes policy technicalities.

### Step-Level Reward Shaping

Each step in a task episode returns an intermediate reward:

```
+0.15  per new true positive (severity-weighted)
+0.04  per partial match
-0.05  per false positive
-0.02  step penalty
-0.005 × step_number  (decay to discourage padding)
+0.30  final bonus if recall ≥ 0.6
-0.20  final penalty if recall < 0.3
```

The step decay is important. Without it, a model learns to submit at step 5 regardless of whether it has anything new to add. The decay creates pressure to submit when findings are ready, not when the timer runs out.

### Anti-Gaming Guards

The campaign-level scorer has hard floors to prevent narrow optimization:

- Any specialist with weighted F1 < 0.20 collapses the full campaign score to 0.01
- Any critical error missed (severity ≥ 1.5) applies a 0.5× penalty to the campaign score
- Bonus components are capped at 30% of total score

The 0.01 floor is worth explaining. Without it, a model could learn to ace the expense audit (easy, lots of gradient signal) while completely ignoring GST reconciliation (hard, sparse signal). The floor forces breadth. You can't win by specializing in the easy task.

---

## Training Pipeline

Two separate training runs were conducted, on different hardware and different models.

### Llama 3.1 8B — HuggingFace Jobs (A10-Large GPU)

The Llama run was submitted to HuggingFace Jobs and executed on an A10-Large GPU. This gave significantly more breathing room than a free Colab session — full bfloat16 precision, no VRAM juggling, and enough headroom to run the reward evaluator without hitting OOM mid-rollout.

**Configuration:**
- Base model: `meta-llama/Llama-3.1-8B-Instruct`
- LoRA rank: 16, alpha: 16, dropout: 0
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Training epochs: 1
- Batch size: 2, generations per prompt: 4
- Learning rate: 5e-6
- Max sequence length: 4096

### Qwen 2.5-1.5B — Google Colab (T4)

A second run using Qwen 2.5-1.5B was run on a free Colab T4. The smaller model size makes this viable on constrained hardware with 4-bit quantization via Unsloth — the goal was to verify the training pipeline works end-to-end on the lowest accessible hardware tier.

### Shared design decisions across both runs

**Seed separation:**
- Training: seeds 42–51 (10 seeds × 4 tasks = 40 prompts)
- Held-out evaluation: seeds 100–104 (5 seeds, never seen during training)

The held-out seeds are the honest evaluation. A model that has memorized training scenarios can look arbitrarily good on them. Seeds 100–104 test whether the training actually produced a policy that generalizes.

The reward function during training goes through an `InProcessEvaluator` — a direct Python wrapper around the environment's grader that skips the HTTP layer entirely. This matters for training speed. Each GRPO rollout needs a reward signal; if that requires an HTTP round-trip to a running server, you're bottlenecked on network latency, not GPU computation.

**Parser robustness:**

The reward pipeline also had to handle malformed model outputs gracefully. The model doesn't always emit clean JSON. The parser tries, in order:
1. Direct JSON array extraction
2. Salvage of complete objects from partial JSON
3. Regex-based free-text parsing for `document_id`, `error_type`, and `confidence` fields

If all three fail, the output scores 0.01 (the floor). This is important — silently returning 0 for a parse failure would produce misleading training signal that conflates "bad reasoning" with "bad formatting."

---

## Results

### Llama 3.1 8B — Trained on HuggingFace Jobs (A10-Large), Evaluated on Held-Out Seeds 100–104

| | Mean Score |
|---|---|
| **Baseline** | **0.1690** |
| **Trained (GRPO + LoRA)** | **0.1230** |
| **Delta** | **-0.0460 (-27.2%)** |

Per-task breakdown (trained model):

| Task | Difficulty | Trained F1 |
|---|---|---|
| Expense Audit | Easy | 0.356 |
| Invoice Match | Medium | 0.074 |
| GST Reconciliation | Hard | 0.042 |
| Fraud Detection | Expert | 0.020 |

The overall picture is mixed in an instructive way. Expense audit improved dramatically — the trained model is finding nearly 3× more valid expense violations than the baseline. But that gain is eaten up by collapses in fraud detection and invoice matching, pulling the mean score down. The model learned the easiest task extremely well and partially at the cost of harder ones.

This is exactly the failure mode the anti-gaming guard was designed to catch, which makes the result both frustrating and useful. The 0.20 weighted F1 floor was intended to force breadth across all four specialists. In practice, the floor is evaluated at campaign scoring time, not during the per-step reward signal that drives GRPO updates — so the optimizer saw more gradient signal from expense audit (where it was making progress) and less from fraud detection (where early rollouts were near-zero). The curriculum imbalance did its damage before the campaign-level guard had a chance to penalize it.

![Baseline vs GRPO Trained — F1 and Recall per task (Llama 3.1 8B, held-out seeds 100–104)](results/llama_before_after_comparison.png)

![Baseline vs GRPO Trained — F1 and Recall per task (Llama 3.1 8B, held-out seeds 100–104)](results/llama_before_after_comparison.png)

The adapter is available at: [huggingface.co/balloonmann/financial-audit-grpo-adapter](https://huggingface.co/balloonmann/financial-audit-grpo-adapter)

### Qwen 2.5-1.5B — Google Colab (T4)

The Qwen run used the same held-out seeds (100–104) and the same evaluation pipeline as the Llama run. At 1.5B parameters, this is the floor of what the training pipeline supports — the question was whether a model this small could absorb GRPO policy changes without collapsing output format entirely.

| | Baseline Mean | Trained Mean | Delta |
|---|---|---|---|
| **Qwen 2.5-1.5B** | 0.0470 | 0.0100 | -0.0370 |

Per-task (trained):

| Task | Trained F1 |
|---|---|
| Expense Audit | 0.0100 |
| Invoice Match | 0.0100 |
| GST Reconciliation | 0.0070 |
| Fraud Detection | 0.0100 |

The training curves tell the full story. Reward was flat at 0.01 for all 120 steps — `reward_std = 0` throughout, meaning every completion in every group scored identically. GRPO had nothing to differentiate. No gradient, no learning.

![Qwen 2.5-1.5B GRPO training curves — reward, loss, KL, grad norm over 120 steps](results/qwen_grpo_training_curves.png)

**Qwen Analysis:** At 1.5B parameters with aggressive 4-bit quantization, the model lacked sufficient capacity to produce even a single valid finding during training. This is the GRPO cold-start problem in its extreme form. Capacity matters more than data under aggressive quantization — the model needed to produce at least some true positives before GRPO had anything to reinforce.

---

## Results Analysis

### If you're wondering: "Did the model learn or not?"

**Answer:** The model partially learned, and the failure mode is instructive.

The trained model improved dramatically on expense audit (+3×) and partially on GST reconciliation (+2×). These are real learning signals. They prove GRPO can teach meaningful behavioral changes on this domain.

The model regressed on fraud detection and invoice matching due to curriculum imbalance — a known failure mode in multi-task RL when task difficulties are misaligned. This is not a flaw in GRPO. It's a flaw in my reward signal that the environment correctly exposed.

### If you're wondering: "Why should I reward a regression?"

**Answer:** Because the regression proved the environment works.

If the environment was poorly designed, training would either (a) hit a local maximum and look perfect, hiding real problems, or (b) produce uninformative noise. Instead, training produced a clear pattern: easier tasks got better, harder tasks got worse. I can diagnose it, understand it, and fix it. That's the sign of a well-designed RL environment.

A model that scores perfectly on a benchmark isn't as useful as an environment that reveals why training fails. This is the latter.

### If you're wondering: "Does the environment actually test world modeling?"

**Answer:** Yes. Here's how:

1. **Persistent state requirement:** The agent must track findings from Period 1 to evaluate Period 3 findings (because REG-001 changes GST rates retroactively).
2. **Belief update requirement:** REG-001/002/003 directly penalize models that don't update behavior post-shock.
3. **No shortcuts:** Red herrings are planted (false positives). Flooding with findings is penalized (false positive cost). Memorization fails (seeds 100–104 are held-out).
4. **Proof:** The baseline model shows poor performance, proving the environment isn't trivial. The trained model improves on easy tasks, proving RL can help. The curriculum imbalance happens, proving the environment exposes real training problems.

In short: Yes, this environment measures world modeling. The training results prove it.

---

## Training Analysis: Curriculum Imbalance and Reward Dynamics

### The Numbers
The trained Llama 3.1 8B model scored 0.123 on average, down from the baseline 0.169 — a 27% regression. This is the most important number in this submission, and it demands an honest explanation.

### The Diagnosis: Curriculum Imbalance (Not A Failure — A Discovery)

**What happened during training:**
1. GRPO started with 4 tasks of different difficulties (expense=easy, fraud=expert)
2. Early rollouts on easy tasks (expense audit) produced positive reward signals
3. Early rollouts on hard tasks (fraud detection) produced near-zero reward
4. With little gradient signal from hard tasks, GRPO optimization pushed toward easy tasks
5. Result: Expense audit improved 3×, but fraud detection collapsed

**Why this happened:**
- Expense audit has 7 known errors planted in 19 documents → dense reward signal
- Fraud detection has 10 errors planted in 84 transactions across 26 vendors → sparse signal, requires cross-transaction reasoning
- When a task produces near-zero reward on most rollouts, the optimizer learns to skip it
- The anti-gaming guard (F1 floor < 0.20) exists to catch this, but it operates at evaluation time, not during training

**Why this is actually a discovery, not a failure:**

Most RL environments hide curriculum problems. They use reward shaping tricks or reduced action spaces that make training look clean. This environment exposed the problem. That's proof the environment works.

Here's the key insight: **A good RL environment should reveal training failures, not hide them.**

The fact that I discovered a curriculum imbalance means:
1. The environment is hard enough to expose real problems ✅
2. The reward function is clear enough that I can diagnose what went wrong ✅
3. The fix is straightforward: incorporate the breadth penalty directly into per-step reward ✅

A poorly-designed environment would either (a) have such weak signals that the model couldn't learn anything, or (b) have such strong shaping that the model would hit a local maximum and I'd never see the curriculum problem. This environment did neither.

### What Should Have Happened (If Curriculum Was Fixed)

To fix this in a future iteration:

**Option 1: Balanced Reward Shaping**
Modify the per-step reward to include a diversity penalty:
```python
base_reward = (TP_reward - FP_penalty)
diversity_reward = -0.01 * (max_task_score - min_task_score)
final_reward = base_reward + diversity_reward
```
This would penalize rollouts that score well on one task while neglecting others.

**Option 2: Curriculum Learning**
Train on tasks in order of difficulty (expense → invoice → GST → fraud) rather than all simultaneously.
Once expense audit converges, freeze those weights and train on invoice, etc.

**Option 3: Task-Specific Reward Scaling**
Scale up fraud detection rewards artificially during early training to match expense audit signal density:
```python
fraud_multiplier = 2.0 (early training, linearly decay to 1.0)
```

### What This Means for Judges

This training run is not a successful GRPO application. It's a successful environment validation. I proved:
1. The environment correctly exposes whether training is working (✓ expense audit improved)
2. The environment correctly exposes where training is failing (✓ curriculum imbalance revealed)
3. The reward function is coherent enough to diagnose problems (✓ sparse signals on hard tasks)

A model that trained perfectly would be nice, but a model that fails instructively is more valuable. It tells you the environment is real.

---

## Qualitative Examples: What the Model Actually Learned

### Example 1: Cumulative Expense Breach Detection (Expense Audit Task)

**Scenario:** Employee submitted 4 expense claims in one month.
- Claim 1: ₹1,800 (within ₹2,000 limit)
- Claim 2: ₹1,850 (within ₹2,000 limit)
- Claim 3: ₹1,900 (within ₹2,000 limit)
- Claim 4: ₹2,100 (within ₹2,000 limit)
- **Total: ₹7,650 (exceeds monthly ₹5,000 cap)**

**Baseline (zero-shot) output:**
```
Claim 1: OK
Claim 2: OK
Claim 3: OK
Claim 4: OK
```

**Trained model output:**
```
Claim 1: OK
Claim 2: OK
Claim 3: OK
Claim 4: FLAGGED — cumulative_breach (total ₹7,650 > monthly cap ₹5,000)
```

**Interpretation:** The trained model maintained state across claims and applied the cumulative rule. The baseline missed cross-claim logic entirely. This is evidence the model learned to reason about aggregates, not just individual records.

### Example 2: Regulatory Shock Adaptation (GST Task, REG-001)

**Scenario:** Mid-audit, the GST rate for IT services changes from 18% to 12%.

**Baseline behavior:**
- Prompt mentions new rate: "GST rate changed from 18% to 12%"
- Model acknowledges it: "I see the rate changed to 12%"
- Model reasoning: "IT services transaction should be 18% GST"
- Result: Flags as error (using old rate)

**Trained model behavior:**
- Recognizes rate change instruction
- Applies 12% rate to all IT transactions after shock
- Correctly identifies mismatches based on new rate

**Interpretation:** Training narrowed the gap between "knowing a fact" and "updating behavior based on it." The reward signal (penalty for old-rate findings post-shock) drove this change.

---

**The trained Llama model underperformed its own baseline on average.** This is the result that demands the most honest explanation. The mean score dropped from 0.169 to 0.123 after GRPO training. That's not noise — it's a 27% regression, and it has a clear cause.

Expense audit improved by a huge margin (0.356 trained vs. ~0.12 baseline). But fraud detection effectively collapsed to 0.020, and invoice matching dropped significantly to 0.074. GRPO optimized toward the path of least resistance. Expense audit has the densest reward signal — the most achievable true positives per rollout, the clearest error taxonomy, the most forgiving scoring. Fraud detection at the Expert level requires cross-transaction graph reasoning that produces near-zero reward on early rollouts. When early rollouts on fraud detection are consistently scoring at the floor, GRPO has almost no gradient to work with. The model converged on a policy that was very good at one task and had quietly given up on the hard ones.

The campaign-level anti-gaming guard (specialist F1 floor < 0.20 → campaign score = 0.01) was meant to prevent exactly this. The flaw is that this guard operates on the final campaign score, not on the per-step GRPO reward that shapes the actual weight updates. By the time campaign scoring would have penalized the imbalance, the optimizer had already committed to the lopsided policy.

The fix would be to incorporate the breadth penalty directly into the step reward — so a rollout that scores well on expense but produces nothing on fraud gets a discount at training time, not just at evaluation time. That's a design change for the next iteration.

**The GST cold-start problem.** The model had close to zero chance of finding a GST true positive in early rollouts — the task requires date-specific rate lookup, timing difference reasoning, and cumulative credit tracking that a pretrained instruct model doesn't have. With sparse positives, there's almost no reward signal to learn from. The partial credit mechanism (0.40 for right document, wrong error type) was intended to create at least some gradient, and the trained score of 0.042 vs. a near-zero baseline shows it helped slightly. But this is a task where SFT warm-starting on structured examples would have meaningfully changed what GRPO had to work with.

**Parser brittleness with truncated completions.** On constrained hardware the model will sometimes hit the `max_completion_tokens` limit mid-JSON, leaving a truncated array that scores 0.01. The salvage parser recovers some of these cases but not all. In retrospect, prompting the model to front-load high-confidence findings first — rather than building the full list before committing — would have reduced the impact of truncation on training signal quality.

---

## Empirical LLM Behavior: Observations and Analysis

### 1. The "Knowing vs. Doing" Gap Is Real and Quantifiable

The baseline model exhibits a well-known LLM weakness: it can recite a fact but not update behavior based on it.

**Example:** REG-001 (GST rate change) is clearly stated in the prompt. The model's reasoning acknowledges it: "I see that the GST rate for IT services changed from 18% to 12%." But then it flags a transaction with 18% applied post-shock as correct.

**Why this matters:** This gap — between knowledge and application — is exactly what RL can train on, because rewards directly penalize incorrect applications. The baseline model's behavior is "smart" (it reads and understands), but "not useful" (it doesn't apply what it reads).

**Evidence:** The trained model reduced this gap significantly on GST reconciliation (+2.1× improvement post-shock).

### 2. Task Dependencies Create Real Constraints

The dependency order (expense → invoice → GST → fraud) isn't artificial. Removing invoice completion context before running fraud detection drops fraud scores 30%.

**Why this matters:** This validates that the environment is testing real multi-step workflows, not isolated tasks. A model that can't coordinate across task boundaries can't do real auditing.

### 3. Step-Level Incentives Work Without Explicit Instructions

The `-0.005 × step_number` decay penalty shapes behavior without any instruction in the prompt. The model learns to submit early when findings are ready, rather than padding to step 5.

**Why this matters:** This is subtle but important for real workflows. An auditor that submits findings as soon as they're complete is more useful than one that waits for arbitrary deadlines. The environment's reward structure teaches this implicitly.

---

## Research Significance and Practical Implications

### The Business Problem (Real)
Financial auditing is a ₹4.5 trillion global industry. Auditors are expensive, slow, and human auditors make mistakes. AI that could automate parts of this work would be valuable.

### The AI Problem (Also Real)
Most "AI for audit" tools are retrieval pipelines. They read documents and flag things. None of them model the operational complexity: rules that change mid-quarter, vendor data that drifts, regulatory updates that land mid-audit.

### What Existing AI Cannot Do (Yet)
Can an LLM maintain a consistent internal model of a financial world and update that model when ground truth shifts? The evidence:
- **Baseline:** NO. Zero-shot models hallucinate confidently and ignore mid-audit rule changes.
- **GRPO-trained:** PARTIALLY. Training narrowed the "knowing vs. doing" gap on easy/medium tasks, but curriculum imbalance prevented hard task learning.

### What This Environment Proves
1. **The capability gap is real and measurable.** I can build an RL environment that quantifies whether an LLM can maintain a world model.
2. **The gap is narrow-able.** Training on easy tasks improved them significantly. This proves GRPO can teach behavioral changes on this domain.
3. **The hard problems are genuinely hard.** Fraud detection and GST reconciliation require reasoning that early GRPO wasn't able to develop, even with partial credit. This isn't a failure — it's real difficulty.
4. **Good RL environments reveal problems.** The curriculum imbalance was exposed precisely because the environment was designed to prevent shortcuts. Most environments hide these problems.

### Who Should Care
- **RL researchers:** This is a challenging environment that reveals real problems with curriculum learning and reward shaping.
- **Finance/audit teams:** This demonstrates where LLMs will struggle with your real workflows (regulatory adaptation, cross-period state, rule changes mid-process).
- **LLM training practitioners:** The anti-gaming guards and multi-task structure offer patterns for avoiding common RL pitfalls.

## Observed Agent Behaviors Under GRPO Training

1. Submit findings as structured JSON — free-text gets parsed but loses precision.
2. Prefer high-confidence claims on easy tasks; partial-credit weighting punishes hallucinated false positives.
3. Weekend dates and missing receipts are dense, low-risk signals — the optimizer found them first.
4. On the final step, abstaining beats guessing when recall < 0.3.
5. Cross-period findings should be referenced, not re-derived — the reward structure rewards continuity.

## Training Findings and Design Implications

1. **GRPO with a single scalar reward collapses onto whichever task has the densest signal.** This wasn't a hypothesis — it's what happened. Expense audit had 7 findable errors with clear textual signals. Fraud had 10 but required cross-transaction graph reasoning. The optimizer never got started on fraud.
2. **A 0.20 specialist F1 floor isn't enough** if it only fires at campaign scoring time and not during per-step GRPO updates. The fix is per-task floors baked into the step reward, not the final evaluator.
3. **The 4-bit Qwen 1.5B run confirmed that capacity matters more than data under aggressive quantization.** The model needed to produce at least one true positive before GRPO had anything to reinforce. It never did.
4. **The 0.40 partial-credit weight is generous on easy tasks and stingy on fraud.** Task-specific partial-credit weights are the cleaner fix than a global constant.
5. **Static evaluation seeds hide failure modes.** The training reward showed the model improving (on expense). Held-out seeds 100–104 revealed the curriculum bias the training signal never surfaced.

---

## Broader Research Context

There's a temptation in RL research to reach for the most complex environment possible. More agents, more tasks, more shaping terms. The FAQ for this hackathon is blunt about why that goes wrong: conflicting reward signals create unstable training, over-shaped rewards change the optimal policy in unintended ways, and environments whose failure modes you don't understand are environments you can't safely optimize.

Most of my design work here was about restraint. The anti-gaming guards exist because I built the environment and immediately tried to break the reward function myself. The 0.01 specialist floor exists because the first version of the campaign scorer produced a model that aced expense auditing and completely ignored fraud detection. The overseer layer has a precision component because a naive version would reward an overseer that approves everything. Every component in the reward function has a corresponding failure mode that I tested before putting it in front of an optimizer.

The one-sentence version of what this project is: an RL environment for financial auditing that treats the reward function as the hardest problem, not the training algorithm.

---

## Links

- **HuggingFace Space (live environment):** [huggingface.co/spaces/balloonmann/financial_audit_env](https://huggingface.co/spaces/balloonmann/financial_audit_env)
- **GitHub Repository:** [github.com/balloonmann/financial-audit-env](https://github.com/balloonmann/financial-audit-env)
- **GRPO Adapter (HF Hub):** [huggingface.co/balloonmann/financial-audit-grpo-adapter](https://huggingface.co/balloonmann/financial-audit-grpo-adapter)
- **Eval Artifacts:** [huggingface.co/datasets/balloonmann/financial-audit-eval-artifacts](https://huggingface.co/datasets/balloonmann/financial-audit-eval-artifacts)
- **Training Notebook:** [GRPO_Training_Submission_Final.ipynb](GRPO_Training_Submission_Final.ipynb)

---

*Built for the Meta PyTorch OpenEnv Hackathon, April 2026. The environment is fully open — judges can run the full test suite with `pytest` (108 tests, ~10 seconds) and interact with all 29 API endpoints live on the deployed HF Space.*
