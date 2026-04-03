---
title: Financial Audit Env
emoji: 💰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# Financial Audit Environment

An OpenEnv-compatible reinforcement learning environment that evaluates how well AI agents can audit financial documents. Built for the [Meta PyTorch Hackathon](https://pytorch.devpost.com/).

**[Live API](https://balloonmann-financial-audit-env.hf.space/docs)** · **[GitHub](https://github.com/balloonmann/financial-audit-env)** · **78 tests passing**

---

## Why This Exists

### The Problem

Financial auditing is one of the most labour-intensive activities in any business. Auditors manually check thousands of invoices, expense claims, and tax filings to find errors, policy violations, and fraud. The work is largely pattern matching — cross-referencing one document against another, verifying amounts, checking dates, spotting duplicates — but it demands deep domain knowledge and sustained attention. It's slow, expensive, and error-prone when done at scale.

There are two parallel gaps here:

1. **No standardised benchmark for auditing agents.** LLM agents are increasingly pitched as tools for financial review. But there's no reproducible, deterministic environment to evaluate how well they actually perform on auditing tasks — across multiple difficulty tiers, with realistic noise, and with grading that goes beyond binary pass/fail.
2. **Static datasets don't test understanding.** If you hand a model the same 50 invoices every time, it can memorise the answers. You don't know if it learned to *audit*, or if it learned to *recall*. Real auditing requires reasoning over unseen data.

### What This Environment Fixes

This project addresses both problems:

- **Procedurally generated data.** Every seed produces a unique set of financial documents with unique planted errors. Same seed = identical data (fully reproducible), but different seed = fresh challenge. The agent can't memorise answers — it has to actually reason.
- **Graduated difficulty.** Tasks move from single-document policy checks (can a human do this in 5 minutes?) all the way to multi-transaction forensic fraud detection (would a real auditor need a week?). This lets you test where a model's auditing ability breaks down.
- **Deterministic, multi-dimensional grading.** No LLM-as-judge, no vibe checks. Scoring uses F1 on exact `(document_id, error_type)` matches, weighted by severity, with partial credit, confusion matrices, and monetary risk assessments. Same findings always get the same score.
- **Dense reward signals.** Instead of a single score at the end of an episode, the environment returns per-step rewards, making it usable for RL training — not just evaluation.
- **Red herrings.** Entries that look suspicious but are compliant (expense at exactly the limit, Friday-evening dinner, ₹1 rounding difference). This forces precision — without them, the optimal strategy is "flag everything."

---

## Tasks

The environment ships with **4 auditing tasks** across a clear difficulty progression. The competition originally specified three difficulty tiers (easy, medium, hard); we added a fourth **Expert-level** task — Fraud Detection — to demonstrate that the architecture scales to open-ended forensic pattern recognition. The first three tasks are used for baseline scoring; the Expert task is available as an extension for more capable agents.

| # | Task | Difficulty | Documents Given | Errors Planted | What the Agent Must Do |
|---|------|-----------|----------------|---------------|----------------------|
| 1 | **Expense Policy Audit** | Easy | 19 expense claims + company policy | 7 violations | Check individual claims against a policy document. Single-document reasoning. |
| 2 | **Invoice Three-Way Match** | Medium | 10 POs + 10 GRNs + 12 invoices | 9 discrepancies | Cross-reference purchase orders, goods receipts, and vendor invoices. Multi-document matching. |
| 3 | **GST Reconciliation** | Hard | 45 book entries + 44 GSTR-2B entries | 12 mismatches | Reconcile internal purchase register against government GST portal data. Cross-system, regulation-aware. |
| 4 | **Fraud Detection** | Expert | 84 transactions + 26 vendor records | 10 fraud patterns | Detect statistical anomalies, relationship graphs, and behavioural red flags across many transactions. Forensic-level. |

### What Makes Each Task Hard

**Expense Audit (Easy)** — Straightforward policy checks (over-limit amounts, duplicate receipts, weekend expenses), but includes red herrings: an expense *at exactly* the daily limit is legal, a Friday evening expense is not a weekend expense, and two claims with the same amount but different receipts are not duplicates.

**Invoice Match (Medium)** — Requires cross-referencing three document types simultaneously. Includes cascading errors (a price mismatch in a line item also causes the total to be wrong — the agent must flag both). Partial deliveries are normal and should not be flagged.

**GST Reconciliation (Hard)** — Involves Indian GST rules: intra-state (CGST + SGST) vs inter-state (IGST), GSTIN format validation, 180-day ITC claim limits, blocked ITC categories. The agent must compare two independently-maintained datasets (books vs GSTR-2B) and identify entries missing from either side.

**Fraud Detection (Expert)** — Pattern recognition over dozens of transactions: circular invoicing chains, shell company indicators (shared bank accounts, invoices before incorporation date), Benford's law violations, split invoices to avoid approval thresholds, sudden vendor volume spikes. This task cannot be solved by row-by-row checks — it requires statistical and relational reasoning.

---

Per-step reward: **+0.15** per true positive, **-0.05** per false positive.

The environment runs as a REST API following the OpenEnv specification. An episode looks like this:

```
1. RESET  →  Agent chooses a task and seed. Environment generates data.
2. READ   →  Agent receives documents, task description, and valid error types.
3. SUBMIT →  Agent sends findings. Environment returns per-step reward.
4. SCORE  →  Agent calls the grader for full breakdown (F1, confusion matrix, risk).
```

### API Flow

```bash
# Start an episode
POST /reset  {"task_id": "expense_audit", "seed": 42}
             → Returns documents + task description

# Submit findings (one or more steps)
POST /step   {"action": {"findings": [...], "submit_final": true}}
             → Returns reward + feedback + running stats

# Get full grading breakdown
GET /grader  → F1 score, weighted F1, confusion matrix, risk assessment
```

### Grading System

The grading goes well beyond binary pass/fail:

| Metric | What It Measures |
|--------|-----------------|
| **F1 Score** (0.0–1.0) | Primary metric. Based on exact `(document_id, error_type)` matches |
| **Weighted F1** | Critical errors (fraud, duplicate invoices) count more than minor ones (weekend expenses) |
| **Partial Credit** | Right document but wrong error type gets 0.25 credit instead of 0 |
| **Confusion Matrix** | Per-error-type breakdown: which kinds of errors the agent catches vs misses |
| **Risk Score** | Monetary value in ₹ — how much the agent's findings would have saved the company |

### Reward Signal

Each step returns a dense reward for RL training:

- **+0.15** per new true positive (weighted by error severity)
- **+0.04** per partial match (right document, wrong error type)
- **−0.05** per false positive
- **−0.02** step penalty (discourages unnecessary steps)
- **+0.30** bonus on final step if recall ≥ 0.8
- **−0.20** penalty on final step if recall < 0.3

---

## Getting Started

### Run Locally

```bash
git clone https://github.com/balloonmann/financial-audit-env.git
cd financial-audit-env
pip install -e .
python -m financial_audit_env.server.app
```

Full interactive docs at `/docs`.

---

```bash
docker build -t financial-audit-env .
docker run -p 8000:8000 financial-audit-env
```

### Use the Hosted Version

The environment is deployed on HuggingFace Spaces. No setup required.

```bash
# Health check
curl https://balloonmann-financial-audit-env.hf.space/health

# Start an audit
curl -X POST https://balloonmann-financial-audit-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "expense_audit", "seed": 42}'
```

---

## Action Space

What the agent sends:

```json
{
  "findings": [
    {
      "document_id": "EXP-010",
      "error_type": "over_limit",
      "description": "Meal expense of ₹4500 exceeds the ₹1500 daily limit",
      "suggested_fix": "Reject and request revised claim"
    }
  ],
  "submit_final": true
}
```

Each finding requires `document_id` (the ID of the flagged document), `error_type` (from the task's allowed list), and `description`. The `suggested_fix` field is optional.

## Observation Space

What the agent receives:

```json
{
  "task_id": "expense_audit",
  "task_description": "Review employee expense claims against policy...",
  "documents": { "expenses": [...], "policy": {...} },
  "findings_so_far": [],
  "feedback": "Step 1/5: Accepted 3 findings. TP: 2, FP: 1",
  "step_number": 1,
  "max_steps": 5,
  "done": false,
  "reward": 0.13
}
```

---

## Investigation Mode

An optional mode that tests whether the agent knows *where to look*, not just what to find.

1. Reset with `"investigation_mode": true`
2. The agent receives a data summary (document counts, available categories) — but no actual data
3. The agent requests specific categories to inspect (each request costs a step)
4. Then submits findings as usual

This mimics how real auditors work: they don't read every document cover to cover. They start by identifying high-risk areas and focusing their review.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message with links |
| `/health` | GET | Status, version, task count |
| `/docs` | GET | Interactive Swagger UI |
| `/tasks` | GET | All 4 tasks with descriptions and allowed error types |
| `/reset` | POST | Start a new episode (send `task_id` + `seed`) |
| `/step` | POST | Submit findings (set `submit_final: true` to end) |
| `/state` | GET | Current episode progress |
| `/grader` | GET | Full scoring breakdown after episode ends |
| `/session` | POST | Create an isolated session (for concurrent agents) |
| `/leaderboard` | GET | Best scores per model |
| `/metrics` | GET | Usage statistics (total resets, steps, uptime) |
| `/adaptive-difficulty` | GET | Difficulty suggestions based on score history |

---

## Baseline Results

The baseline agent uses **Llama 3.1 8B** through the free HuggingFace Inference API. It reads all documents in a single prompt and submits its findings in one step.

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1/
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
python inference.py --env-url http://localhost:8000
```

### Scores

| Task | Difficulty | F1 Score | Precision | Recall |
|------|-----------|----------|-----------|--------|
| Expense Audit | Easy | 0.3158 | 0.23 | 0.50 |
| Invoice Match | Medium | 0.2667 | 0.17 | 0.67 |
| GST Reconciliation | Hard | 0.0000 | 0.00 | 0.00 |
| Fraud Detection | Expert | 0.0000 | 0.00 | 0.00 |

**Why did it fail?**
The model frequently drops to 0.00 on the Easy task because it struggles with abstract rules (like date math for weekend expenses, or tracking cumulative limits). It actively gets tricked by "red herrings"—perfectly legal expenses that it hallucinates as errors—which entirely destroys its precision score.

The low scores are intentional as they define the baseline. There is substantial room for improvement through better prompting strategies (chain-of-thought, multi-step analysis), larger models, tool use, or RL fine-tuning.

---

## Setup

- **Rate limiting**: 30 requests/min per IP
- **Request size limits**: 1 MB maximum
- **Secure headers**: HSTS, X-Frame-Options, X-Content-Type-Options
- **Input validation**: Pydantic models for all requests
- **Answer isolation**: Error messages never leak ground truth data
- **Request tracing**: Every response includes an `X-Request-ID` header

---

## Project Structure

```
financial_audit_env/
  server/
    app.py              # FastAPI application with all endpoints
    environment.py      # Core RL environment (reset, step, state)
    data_generator.py   # Synthetic data generation with planted errors
    graders.py          # F1, weighted F1, partial credit, risk scoring
    security.py         # Rate limiting, security headers, auth
    tasks.py            # Task definitions (4 tasks, easy → expert)
  models.py             # Pydantic models (Action, Observation, State)
  baseline.py           # Baseline agent using Llama 3.1
tests/                  # 78 pytest tests (security, grading, data gen)
inference.py            # Hackathon inference script ([START]/[STEP]/[END] logs)
openenv.yaml            # OpenEnv specification config
Dockerfile              # HuggingFace Spaces deployment
```

---

## Design Decisions

**Why synthetic data?** Static datasets let models memorise answers. Procedural generation with a seed gives reproducibility *and* novelty — same seed = same test, new seed = new test.

**Why F1 scoring instead of LLM-as-judge?** Deterministic, reproducible, free. No API calls needed to grade. Same findings always produce the same score. This also means the environment can run entirely offline.

**Why dense rewards?** A single score at the end of an episode gives the agent nothing to learn from during training. Per-step rewards with severity weighting create a gradient the agent can follow.

**Why red herrings?** Without them, the dominant strategy is "flag everything." Red herrings force the agent to balance precision against recall. This is also how real auditing works — most transactions are legitimate.

**Why a fixed reference date?** Using `datetime.now()` means the same seed generates different data on different days. A fixed reference date (Jan 15, 2026) makes every run with the same seed produce byte-identical output.

**Why 4 tasks when the spec asks for 3?** The first three tasks (Easy, Medium, Hard) satisfy the competition requirements. We added a fourth Expert-level task — Fraud Detection — to stress-test the architecture on open-ended pattern recognition. It's excluded from the default baseline run to keep scores comparable, but available for agents that want a harder challenge.

---

## Future Work

- **Async data generation** for parallel training across multiple episodes
- **International tax systems** — EU VAT, US sales tax — beyond Indian GST
- **Multi-agent mode** where agents specialise in different error types
- **Export agent interactions** as fine-tuning datasets for smaller models
- **Curriculum learning** — automatic task progression based on agent performance
