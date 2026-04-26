# Teaching a Language Model to Actually Do a Financial Audit

*Meta PyTorch OpenEnv Hackathon — April 2026 | Round 2 Submission*

---

There's a version of the "AI for finance" pitch that gets told at a lot of conferences. A model reads a spreadsheet, surfaces a few anomalies, everyone nods approvingly. It's a clean story. It also has almost nothing to do with how financial auditing actually works.

Real audit seasons don't look like that. They look like three systems with conflicting data, a GST rate that changed mid-quarter, a vendor that got flagged for due diligence after two transactions already cleared, and a junior auditor trying to reconcile 45 book entries against a GSTR-2B that inexplicably has 44. The problems aren't in the data. They're in the relationship between the data and a set of rules that are themselves moving.

That gap — between what AI demos show and what audit work actually requires — is what this project is about.

---

## The Problem Worth Solving

Most LLM audit demos are effectively retrieval tasks. Given clean, static documents, find the thing that violates the rule. That's a useful capability, but it's not auditing. Auditing is adversarial, multi-period, and policy-sensitive. The ground truth shifts because policy shifts. An expense that was compliant in Period 1 gets flagged in Period 3 because a new approval threshold dropped mid-campaign. An invoice that matched perfectly at submission now shows a quantity discrepancy when the goods receipt note gets updated.

The AI systems that exist for this space are almost universally document-to-insight pipelines — they read, they extract, they flag. None of them model what happens when the rules change while you're mid-audit.

That's the problem I wanted to build an environment around.

If you can train an agent to operate correctly across a dynamic, multi-period audit campaign — adapting to regulatory shocks, coordinating with specialist peers, maintaining a review trail that holds up to oversight scrutiny — you've built something that's actually useful. Not demo-useful. Actually useful.

---

## Hackathon Theme: #3 World Modeling — #3.1 Professional Tasks

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

## Architecture: A Multi-Agent Financial Audit Control Tower

### The Specialist Swarm

The core of the environment is four auditing agents, each trained on a different class of financial error:

| Agent | Task | Documents | Planted Errors | Difficulty |
|---|---|---|---|---|
| Expense Specialist | Expense policy violations | 19 expense records | 7 errors | Easy |
| Invoice Specialist | Three-way matching | 10 POs + 10 GRNs + 12 invoices | 9 discrepancies | Medium |
| GST Specialist | Tax reconciliation | 45 book entries + 44 GSTR-2B | 12 mismatches | Hard |
| Fraud Specialist | Pattern detection | 84 transactions + 26 vendors | 10 fraud patterns | Expert |

Each specialist runs for up to 5 steps per period before submitting final findings. The dependency order matters: the GST agent can only run after invoice matching is complete, and the fraud agent draws on prior task findings for its cross-task pattern signals.

The error taxonomy is where a lot of the design work went. It's easy to plant simple errors. It's harder to plant errors that are realistic — the kind an actual auditor would catch and a language model would plausibly miss. Expense violations include `cumulative_breach` (spread across multiple records individually within limit), `unapproved_vendor` (vendor not on approved list), and `duplicate_claim` (same receipt filed twice with minor field differences). Fraud errors include `round_tripping` (funds leaving and returning through related entities), `ghost_employee` (payroll entries with no corresponding HR record), and `collusion_pattern` (split invoices just below approval thresholds across the same vendor).

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

## Reward Engineering: Getting the Signal Right

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

A second run using Qwen 2.5-1.5B is currently in progress on a free Colab T4. The smaller model size makes this viable on constrained hardware with 4-bit quantization via Unsloth. Results pending — see the results section below.

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

The adapter is available at: [huggingface.co/balloonmann/financial-audit-grpo-adapter](https://huggingface.co/balloonmann/financial-audit-grpo-adapter)

### Qwen 2.5-1.5B — Training in Progress (Google Colab, T4)

A second run using Qwen 2.5-1.5B is currently in progress on Colab. The smaller model is a useful comparison point — it runs comfortably on a free T4 with 4-bit quantization, which makes the training pipeline accessible without any paid compute. The question is whether the smaller parameter count costs meaningfully in task performance, or whether the financial audit tasks are constrained enough that a 1.5B model can learn the relevant policy patterns.

Results will be filled in here when the run completes. The same held-out seeds (100–104) will be used.

| | Baseline Mean | Trained Mean | Delta |
|---|---|---|---|
| **Qwen 2.5-1.5B** | *pending* | *pending* | *pending* |

Per-task (trained):

| Task | Trained F1 |
|---|---|
| Expense Audit | *pending* |
| Invoice Match | *pending* |
| GST Reconciliation | *pending* |
| Fraud Detection | *pending* |

---

## What Went Wrong (And Why That's Useful)

**The trained Llama model underperformed its own baseline on average.** This is the result that demands the most honest explanation. The mean score dropped from 0.169 to 0.123 after GRPO training. That's not noise — it's a 27% regression, and it has a clear cause.

Expense audit improved by a huge margin (0.356 trained vs. ~0.12 baseline). But fraud detection effectively collapsed to 0.020, and invoice matching dropped significantly to 0.074. GRPO optimized toward the path of least resistance. Expense audit has the densest reward signal — the most achievable true positives per rollout, the clearest error taxonomy, the most forgiving scoring. Fraud detection at the Expert level requires cross-transaction graph reasoning that produces near-zero reward on early rollouts. When early rollouts on fraud detection are consistently scoring at the floor, GRPO has almost no gradient to work with. The model converged on a policy that was very good at one task and had quietly given up on the hard ones.

The campaign-level anti-gaming guard (specialist F1 floor < 0.20 → campaign score = 0.01) was meant to prevent exactly this. The flaw is that this guard operates on the final campaign score, not on the per-step GRPO reward that shapes the actual weight updates. By the time campaign scoring would have penalized the imbalance, the optimizer had already committed to the lopsided policy.

The fix would be to incorporate the breadth penalty directly into the step reward — so a rollout that scores well on expense but produces nothing on fraud gets a discount at training time, not just at evaluation time. That's a design change for the next iteration.

**The GST cold-start problem.** The model had close to zero chance of finding a GST true positive in early rollouts — the task requires date-specific rate lookup, timing difference reasoning, and cumulative credit tracking that a pretrained instruct model doesn't have. With sparse positives, there's almost no reward signal to learn from. The partial credit mechanism (0.40 for right document, wrong error type) was intended to create at least some gradient, and the trained score of 0.042 vs. a near-zero baseline shows it helped slightly. But this is a task where SFT warm-starting on structured examples would have meaningfully changed what GRPO had to work with.

**Parser brittleness with truncated completions.** On constrained hardware the model will sometimes hit the `max_completion_tokens` limit mid-JSON, leaving a truncated array that scores 0.01. The salvage parser recovers some of these cases but not all. In retrospect, prompting the model to front-load high-confidence findings first — rather than building the full list before committing — would have reduced the impact of truncation on training signal quality.

---

## What Surprised Me

The overseer component turned out to be more useful as a training signal than I expected. Initially I built it as a governance layer — something to make the submission more architecturally complete. But the overseer's precision on conflict resolution turns out to be a meaningful differentiator between a model that understands the audit rules and one that's just pattern-matching. Two specialists flagging the same document with different error types is a signal worth distinguishing from two specialists independently flagging different errors on the same vendor. The overseer layer forces the model to handle that distinction explicitly.

The regulatory shock timing also revealed something interesting about instruct model behavior. REG-001 (the GST rate change) drops at Period 3, Step 3 — mid-audit. The baseline model mostly ignores it. The prompt includes the updated rate, the model acknowledges it in its reasoning, and then continues flagging errors at the old rate. There's a well-known gap between "knowing a fact" and "updating behavior based on that fact" in instruct models. The RL loop should narrow that gap, because the reward directly penalizes any finding that assumes the old rate post-shock.

The step-decay penalty worked better than expected. Early in testing without it, the model would almost always wait until step 5 to submit, regardless of when it had complete findings. Adding the `-0.005 × step_number` decay pushed submission timing earlier without requiring any explicit instruction. The model learned to submit when it was done, not when the clock ran out. That's a subtle but important behavior for a real auditing workflow.

---

## Why It Matters

Financial auditing is a ₹4.5 trillion industry globally, with a chronic shortage of qualified professionals and an equally chronic problem with errors that slip through. The AI tools that exist for this space are retrieval pipelines — they read documents and flag things. None of them model the operational reality: rules that change mid-quarter, vendor data that drifts, regulatory updates that land while an audit is in flight.

The capability gap this environment targets is real: can a language model maintain a consistent internal model of a financial world and update that model correctly when the ground truth shifts? The answer from the baseline run is no, not without training. The answer from the GRPO run is: partially, and in a very specific direction that reveals exactly where the training signal needs to be strengthened.

That's what a good RL environment is supposed to produce — not a model that scores well on a benchmark, but a clear picture of what works, what doesn't, and why. Judges looking for evidence of genuine learning will find it here: a real capability improvement on the easiest task, a documented collapse on the hardest ones, and an honest analysis of the curriculum imbalance that caused it.

## The Bigger Picture

There's a temptation in RL research to reach for the most complex environment possible. More agents, more tasks, more shaping terms. The FAQ for this hackathon is blunt about why that goes wrong: conflicting reward signals create unstable training, over-shaped rewards change the optimal policy in unintended ways, and environments whose failure modes you don't understand are environments you can't safely optimize.

Most of the design work here was about restraint. The anti-gaming guards exist because I built the environment and immediately tried to break the reward function myself. The 0.01 specialist floor exists because the first version of the campaign scorer produced a model that aced expense auditing and completely ignored fraud detection. The overseer layer has a precision component because a naive version would reward an overseer that approves everything. Every component in the reward function has a corresponding failure mode that I tested before putting it in front of an optimizer.

The one-sentence version of what this project is: an RL environment for financial auditing that treats the reward function as the hardest problem, not the training algorithm.

---

## Links

- **HuggingFace Space (live environment):** [balloonmann-financial-audit-env.hf.space](https://balloonmann-financial-audit-env.hf.space)
- **GitHub Repository:** [github.com/balloonmann/financial-audit-env](https://github.com/balloonmann/financial-audit-env)
- **GRPO Adapter (HF Hub):** [huggingface.co/balloonmann/financial-audit-grpo-adapter](https://huggingface.co/balloonmann/financial-audit-grpo-adapter)
- **Eval Artifacts:** [huggingface.co/datasets/balloonmann/financial-audit-eval-artifacts](https://huggingface.co/datasets/balloonmann/financial-audit-eval-artifacts)
- **Training Notebook:** *(Colab link — to be added on competition day)*

---

*Built for the Meta PyTorch OpenEnv Hackathon, April 2026. The environment is fully open — judges can run the full test suite with `pytest` (108 tests, ~10 seconds) and interact with all 29 API endpoints live on the deployed HF Space.*
