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

An OpenEnv-compatible environment that teaches AI agents how to audit financial documents. Built for the Meta PyTorch Hackathon.

**[Try the API](https://balloonmann-financial-audit-env.hf.space/docs)** | **78 tests passing**

## The Problem

Financial auditing is boring, repetitive, and expensive. Companies pay auditors to check thousands of invoices, expense claims, and tax returns by hand. Most of the work is pattern matching: does this invoice match the purchase order? Did this employee exceed their expense limit? Is this vendor's GST number valid?

This environment turns that process into something an AI agent can learn. It generates fake (but realistic) Indian financial documents, hides errors in them, and scores the agent on how many it finds.

## What's Inside

There are 4 tasks, each harder than the last:

| Task | Difficulty | What you get | What you find |
|------|-----------|-------------|--------------|
| **Expense Policy Audit** | Easy | 19 expense claims + company policy | 7 policy violations (over-limit meals, duplicate receipts, weekend expenses) |
| **Invoice Three-Way Match** | Medium | 10 purchase orders + 10 goods receipts + 12 invoices | 9 discrepancies (price mismatches, duplicate invoices, wrong tax rates) |
| **GST Reconciliation** | Hard | 45 book entries + 44 GSTR-2B entries | 12 mismatches (missing entries, invalid GSTINs, excess claims) |
| **Fraud Detection** | Expert | 84 transactions + 26 vendors | 10 fraud patterns (circular invoicing, shell companies, Benford violations) |

Every task also includes red herrings: entries that look suspicious but are actually fine. An expense at exactly the daily limit? That's legal. A ₹1 rounding difference on an invoice? That's normal. The agent has to tell the difference.

## How It Works

The environment runs as a REST API. Here's the basic flow:

1. **Reset** - Tell the environment which task you want. It generates fresh data with hidden errors.
2. **Read** - Look at the documents in the response. Figure out what's wrong.
3. **Submit** - Send your findings (which document, what type of error, why).
4. **Score** - The environment checks your findings against the ground truth and gives you an F1 score.

```
POST /reset  {"task_id": "expense_audit", "seed": 42}
             -> Returns documents + task description

POST /step   {"action": {"findings": [...], "submit_final": true}}
             -> Returns reward + feedback

GET /grader  -> Returns F1 score, confusion matrix, risk assessment
```

The grading goes beyond a simple pass/fail:

- **F1 Score** (0.0 to 1.0) based on matching `(document_id, error_type)` pairs
- **Weighted F1** where critical errors (fraud) count more than minor ones (weekend expenses)
- **Partial credit** if you flag the right document but guess the wrong error type (0.25 instead of 0)
- **Confusion matrix** showing which error types you nailed and which you missed
- **Risk score** in rupees: how much money your findings would have saved

Each step also gives a reward signal (+0.15 per correct finding, -0.05 per false positive) so you can use this for RL training, not just evaluation.

## Getting Started

### Run locally

```bash
git clone https://github.com/balloonmann/financial-audit-env.git
cd financial-audit-env
pip install -e .
python -m financial_audit_env.server.app
```

Server starts at `http://localhost:8000`. Hit `/health` to check it's running, or `/docs` for the interactive Swagger UI.

### Run with Docker

```bash
docker build -t financial-audit-env .
docker run -p 8000:8000 financial-audit-env
```

### Use the hosted version

The environment is live on HuggingFace Spaces. No setup needed.

```bash
# Check it's up
curl https://balloonmann-financial-audit-env.hf.space/health

# Start an audit
curl -X POST https://balloonmann-financial-audit-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "expense_audit", "seed": 42}'
```

## API Endpoints

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/` | GET | Welcome message with links to everything else |
| `/health` | GET | Returns status, version, task count |
| `/docs` | GET | Interactive Swagger UI (try the API in your browser) |
| `/tasks` | GET | Lists all 4 tasks with descriptions and allowed error types |
| `/reset` | POST | Starts a new episode. Send `task_id` and `seed` |
| `/step` | POST | Submit findings. Set `submit_final: true` to end the episode |
| `/state` | GET | Shows current progress (step number, errors found so far) |
| `/grader` | GET | Full scoring breakdown after episode ends |
| `/session` | POST | Creates an isolated session (for running multiple agents at once) |
| `/leaderboard` | GET | Best scores per model |
| `/metrics` | GET | Usage stats (total resets, steps, uptime) |
| `/adaptive-difficulty` | GET | Suggests harder/easier settings based on past scores |

## What the Agent Sends (Action Space)

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

## What the Agent Gets Back (Observation Space)

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

There's an optional mode where the agent doesn't get all the documents upfront. Instead:

1. Reset with `"investigation_mode": true`
2. You get a summary (how many documents, what categories exist) but no actual data
3. Request specific categories to look at (costs a step)
4. Then submit findings as usual

This tests whether an agent can figure out where to look first, which is what real auditors actually do.

## Running the Baseline

The baseline uses Llama 3.1 8B through the free HuggingFace Inference API:

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1/
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct

python inference.py --env-url http://localhost:8000
```

### Baseline Results

| Task | Difficulty | F1 Score |
|------|-----------|----------|
| Expense Audit | Easy | 0.3158 |
| Invoice Match | Medium | 0.0000 |
| GST Reconciliation | Hard | 0.0000 |
| Fraud Detection | Expert | not yet tested |

Yes, the scores are low. That's the point. There's a lot of room for a better agent or a bigger model to improve on this.

## Security

The API has rate limiting (30 req/min per IP), request size limits (1MB), secure headers (HSTS, X-Frame-Options), and input validation through Pydantic. Error messages are sanitized so they never leak the ground truth answers. Every response has an `X-Request-ID` header for debugging.

## Project Layout

```
financial_audit_env/
  server/
    app.py              # FastAPI app with all endpoints
    environment.py      # The actual RL environment (reset, step, state)
    data_generator.py   # Generates fake financial data with planted errors
    graders.py          # F1 scoring, weighted F1, risk scoring
    security.py         # Rate limiting, headers, auth
    tasks.py            # Task definitions (4 tasks, easy to expert)
  models.py             # Pydantic models (Action, Observation, State)
  baseline.py           # Baseline agent using Llama 3.1
tests/                  # 78 pytest tests
inference.py            # Hackathon inference script with [START]/[STEP]/[END] logs
openenv.yaml            # OpenEnv spec config
Dockerfile              # For HF Spaces deployment
```

## Design Choices

**Why synthetic data?** Static datasets let models memorize answers. Every seed generates unique data, so the agent has to actually understand the task.

**Why F1 scoring instead of LLM-as-judge?** Deterministic, reproducible, free. No API calls needed to grade. Same findings always get the same score.

**Why dense rewards?** A single score at the end of an episode isn't useful for training. Per-step rewards (+0.15 per correct, -0.05 per wrong) give the agent something to learn from.

**Why red herrings?** Without them, the optimal strategy is "flag everything." Red herrings force precision.

**Why a fixed reference date?** Using `datetime.now()` means the same seed generates different data on different days. A fixed date (Jan 15, 2026) keeps everything reproducible.

## What's Next

- Async data generation for parallel training
- Support for international tax systems (EU VAT, US sales tax) beyond Indian GST
- Multi-agent mode where agents specialize in different error types
- Export agent interactions as fine-tuning datasets
