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

An OpenEnv-compatible environment for training AI agents on financial document auditing. Built for the Meta PyTorch Hackathon.

**[Try the API](https://balloonmann-financial-audit-env.hf.space/docs)** | **78 tests passing**

## Problem

Financial auditing is repetitive and expensive. Auditors check thousands of invoices, expense claims, and tax returns by hand. Most of the work is pattern matching: does this invoice match the purchase order? Did this employee exceed their expense limit? Is this vendor's GST number valid?

This environment generates realistic Indian financial documents, plants errors in them, and scores an agent on how many it finds.

## What's Inside

Four tasks, each harder than the last:

| Task | Difficulty | What you get | What you find |
|------|-----------|-------------|--------------|
| **Expense Policy Audit** | Easy | 19 expense claims + company policy | 7 policy violations (over-limit meals, duplicate receipts, weekend expenses) |
| **Invoice Three-Way Match** | Medium | 10 purchase orders + 10 goods receipts + 12 invoices | 9 discrepancies (price mismatches, duplicate invoices, wrong tax rates) |
| **GST Reconciliation** | Hard | 45 book entries + 44 GSTR-2B entries | 12 mismatches (missing entries, invalid GSTINs, excess claims) |
| **Fraud Detection** | Expert | 84 transactions + 26 vendors | 10 fraud patterns (circular invoicing, shell companies, Benford violations) |

Each task includes red herrings: entries that look suspicious but are valid. An expense at exactly the daily limit is legal. A ₹1 rounding difference on an invoice is normal. The agent has to tell the difference.

## How It Works

The environment runs as a REST API.

1. **Reset** — specify the task. The environment generates fresh data with hidden errors.
2. **Read** — examine the documents in the response.
3. **Submit** — send findings (which document, what error type, why).
4. **Score** — findings are checked against ground truth and returned as an F1 score.

```
POST /reset  {"task_id": "expense_audit", "seed": 42}
             -> Returns documents + task description
POST /step   {"action": {"findings": [...], "submit_final": true}}
             -> Returns reward + feedback
GET /grader  -> Returns F1 score, confusion matrix, risk assessment
```

Grading breakdown:

- **F1 Score** (0.0 to 1.0) based on matching `(document_id, error_type)` pairs
- **Weighted F1** where critical errors (fraud) count more than minor ones (weekend expenses)
- **Partial credit** for flagging the right document with the wrong error type (0.25 instead of 0)
- **Confusion matrix** showing per-error-type performance
- **Risk score** in rupees representing financial exposure caught

Per-step reward signal: +0.15 per correct finding, -0.05 per false positive.

## Getting Started

### Run locally

```bash
git clone https://github.com/balloonmann/financial-audit-env.git
cd financial-audit-env
pip install -e .
python -m financial_audit_env.server.app
```

Server starts at `http://localhost:8000`. `/health` confirms it's running. `/docs` opens the Swagger UI.

### Run with Docker

```bash
docker build -t financial-audit-env .
docker run -p 8000:8000 financial-audit-env
```

### Use the hosted version

```bash
curl https://balloonmann-financial-audit-env.hf.space/health

curl -X POST https://balloonmann-financial-audit-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "expense_audit", "seed": 42}'
```

## API Endpoints

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/` | GET | Welcome message with links |
| `/health` | GET | Returns status, version, task count |
| `/docs` | GET | Interactive Swagger UI |
| `/tasks` | GET | Lists all 4 tasks with descriptions and allowed error types |
| `/reset` | POST | Starts a new episode. Accepts `task_id` and `seed` |
| `/step` | POST | Submit findings. `submit_final: true` ends the episode |
| `/state` | GET | Current progress (step number, errors found so far) |
| `/grader` | GET | Full scoring breakdown after episode ends |
| `/session` | POST | Creates an isolated session for parallel agents |
| `/leaderboard` | GET | Best scores per model |
| `/metrics` | GET | Usage stats (total resets, steps, uptime) |
| `/adaptive-difficulty` | GET | Suggests difficulty adjustments based on past scores |

## Action Space

```json
{
  "findings": [
    {
      "document_id": "EXP-010",
      "error_type": "over_limit",
      "description": "Meal expense of 4500 exceeds the 1500 daily limit",
      "suggested_fix": "Reject and request revised claim"
    }
  ],
  "submit_final": true
}
```

## Observation Space

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

## Investigation Mode

An optional mode where the agent receives a document summary on reset rather than the full data.

1. Reset with `"investigation_mode": true`
2. Receive a summary (document counts, categories) without the underlying data
3. Request specific categories (costs a step)
4. Submit findings as usual

## Running the Baseline

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1/
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
python inference.py --env-url http://localhost:8000
```

### Baseline Results (Llama 3.1 8B)

| Task | Difficulty | F1 Score |
|------|-----------|----------|
| Expense Audit | Easy | 0.3158 |
| Invoice Match | Medium | 0.0000 |
| GST Reconciliation | Hard | 0.0000 |
| Fraud Detection | Expert | not yet tested |

## Security

Rate limiting at 30 req/min per IP, 1MB request size limit, HSTS and X-Frame-Options headers, Pydantic input validation. Error messages are sanitized and never expose ground truth. Every response includes an `X-Request-ID` header.

## Project Layout

```
financial_audit_env/
  server/
    app.py              # FastAPI app
    environment.py      # RL environment (reset, step, state)
    data_generator.py   # Synthetic data generation with planted errors
    graders.py          # F1 scoring, weighted F1, risk scoring
    security.py         # Rate limiting, headers, auth
    tasks.py            # Task definitions
  models.py             # Pydantic models
  baseline.py           # Baseline agent
tests/                  # 78 pytest tests
inference.py            # Hackathon inference script
openenv.yaml            # OpenEnv spec config
Dockerfile
```

## Design Choices

**Synthetic data** — static datasets allow memorization. Each seed generates unique data.

**F1 scoring** — deterministic, reproducible, requires no API calls to grade.

**Dense rewards** — per-step signals (+0.15 correct, -0.05 wrong) support RL training.

**Red herrings** — without them, flagging everything is optimal. Red herrings enforce precision.

**Fixed reference date** — `datetime.now()` breaks seed reproducibility across days. A fixed date (Jan 15, 2026) keeps generation stable.

## What's Next

- Async data generation for parallel training
- International tax system support (EU VAT, US sales tax)
- Multi-agent mode with specialization by error type
- Export of agent interactions as fine-tuning datasets
