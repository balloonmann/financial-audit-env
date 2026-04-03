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

An **OpenEnv-compatible reinforcement learning environment** that trains AI agents on real-world financial auditing tasks — from simple expense policy checks to complex fraud pattern detection.

[![Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://balloonmann-financial-audit-env.hf.space)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/balloonmann/financial-audit-env)
[![Tests](https://img.shields.io/badge/tests-78%20passed-brightgreen)]()
[![Version](https://img.shields.io/badge/version-2.0-blue)]()

---

## What This Does

Financial auditing is the process of checking whether a company's financial records are accurate — invoices match purchase orders, expenses follow policy, GST returns balance, and transactions aren't fraudulent. It's a task that costs businesses **thousands of manual hours** every year.

This environment generates **synthetic but realistic Indian financial documents** with mathematically verifiable planted errors. AI agents interact through a REST API: they receive documents, analyze them, submit findings, and get scored on accuracy.

**Why it matters:** Instead of training agents on toy games, this environment teaches them to solve a genuine business problem — one where mistakes have real monetary consequences.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **Validation** | Pydantic v2 (strict mode) |
| **AI Baseline** | OpenAI Client → HuggingFace Inference API → Meta Llama 3.1 8B |
| **Infrastructure** | Docker, HuggingFace Spaces |
| **Framework** | OpenEnv (reinforcement learning environment spec) |
| **Testing** | Pytest (78 tests) |

---

## Tasks

The environment includes **4 tasks** with progressive difficulty:

| # | Task | Difficulty | Documents | Planted Errors | What the Agent Must Do |
|---|------|-----------|-----------|----------------|----------------------|
| 1 | **Expense Policy Audit** | Easy | 19 expense claims + policy rules | 7 violations | Check if expenses follow company policy (limits, categories, receipts, vendors) |
| 2 | **Invoice Three-Way Match** | Medium | 10 POs + 10 GRNs + 12 invoices | 9 discrepancies | Cross-match purchase orders ↔ goods receipts ↔ invoices for price/quantity/tax errors |
| 3 | **GST Return Reconciliation** | Hard | 45 book entries + 44 GSTR-2B entries | 12 mismatches | Reconcile internal books against government GST portal filings |
| 4 | **Fraud Pattern Detection** | Expert | 84 transactions + 26 vendors | 10 fraud patterns | Detect circular invoicing, shell companies, Benford violations, split invoices, etc. |

Each task includes **red herring entries** — items that look suspicious but are actually correct (e.g., an expense at exactly the limit, a minor ₹1 rounding difference). This forces agents to be precise, not just paranoid.

---

## How It Works

```
Agent                          Environment
  │                                │
  │── POST /reset ────────────────▶│  Generate fresh data with planted errors
  │◀─────────── observation ───────│  Return documents + task description
  │                                │
  │── POST /step (findings) ──────▶│  Grade each finding against ground truth
  │◀─────── reward + feedback ─────│  +0.15 per correct, -0.05 per false positive
  │                                │
  │── POST /step (submit_final) ──▶│  Compute final F1 score
  │◀────── F1 score + details ─────│  Weighted F1, confusion matrix, risk score
```

### Scoring

The primary metric is **F1 score** (0.0–1.0) based on `(document_id, error_type)` tuple matching. The grading system also provides:

- **Weighted F1** — critical errors (fraud: 2.0×) count more than minor ones (weekend expense: 0.5×)
- **Partial credit** — right document + wrong error type gets 0.25 credit instead of 0
- **Confusion matrix** — which error types the agent found vs missed
- **Risk scoring** — rupee value of caught vs missed errors (e.g., "₹3.5L risk mitigated")
- **Dense rewards** — per-step feedback with decay (earlier findings worth more)

---

## Quick Start

### Option 1: Local Development

```bash
git clone https://github.com/balloonmann/financial-audit-env.git
cd financial-audit-env
python -m venv venv && venv\Scripts\activate  # Windows
pip install -e .
python -m financial_audit_env.server.app
```

Server starts at `http://localhost:8000`. Check health: `GET /health`.

### Option 2: Docker

```bash
git clone https://github.com/balloonmann/financial-audit-env.git && cd financial-audit-env
docker build -t financial-audit-env .
docker run -p 8000:8000 financial-audit-env
```

### Option 3: Use the Live API

The environment is deployed on HuggingFace Spaces:

```
https://balloonmann-financial-audit-env.hf.space
```

Try it:
```bash
curl https://balloonmann-financial-audit-env.hf.space/health
curl -X POST https://balloonmann-financial-audit-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "expense_audit", "seed": 42}'
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns status and version |
| `/tasks` | GET | List all tasks with descriptions and allowed error types |
| `/reset` | POST | Start a new episode — `{task_id, seed, investigation_mode}` |
| `/step` | POST | Submit findings — `{action: {findings: [...], submit_final: bool}}` |
| `/state` | GET | Current episode state (step count, found errors) |
| `/grader` | GET | Final score with F1, weighted F1, confusion matrix, risk |
| `/session` | POST | Create an isolated session for multi-tenancy |
| `/leaderboard` | GET | Best scores per model |
| `/metrics` | GET | Usage statistics (resets, steps, uptime) |
| `/adaptive-difficulty` | GET | Difficulty recommendation based on score history |

### Action Space (what the agent submits)
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

### Observation Space (what the agent receives)
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

For a more realistic multi-step experience, reset with `investigation_mode: true`:

1. **Step 1:** Agent receives a data summary (counts, categories) — not the full documents
2. **Step 2:** Agent requests specific document categories to investigate
3. **Step 3:** Agent receives detailed data for those categories
4. **Step 4+:** Agent submits findings based on what it's seen

This tests whether an agent can **triage and prioritize** — a core real-world audit skill.

---

## Running the Baseline Agent

The baseline uses Meta Llama 3.1 8B via the free HuggingFace Inference API:

```bash
export HF_TOKEN=your_huggingface_token
export API_BASE_URL=https://router.huggingface.co/v1/
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct

python inference.py --env-url http://localhost:8000
```

### Baseline Scores (Llama 3.1 8B)

| Task | Difficulty | F1 Score | Notes |
|------|-----------|----------|-------|
| Expense Audit | Easy | 0.3158 | Finds some violations, misses cumulative breach |
| Invoice Match | Medium | 0.0000 | Struggles with cross-document matching |
| GST Reconciliation | Hard | 0.0000 | Cannot reconcile at scale |
| Fraud Detection | Expert | — | Not yet benchmarked |

These scores show there's significant room for improvement — a better agent or model should be able to score much higher.

---

## Security

- **Rate limiting** — 30 requests/minute per IP with TTL-based cleanup
- **Input validation** — Pydantic strict mode, max 50 findings per step
- **Secure headers** — OWASP recommended (HSTS, X-Frame-Options, nosniff)
- **Ground truth protection** — error messages never leak planted error details
- **Request tracking** — every response includes `X-Request-ID` for debugging
- **Body size limits** — 1MB maximum request size
- **CORS** — configurable via `CORS_ORIGINS` environment variable

---

## Project Structure

```
├── financial_audit_env/
│   ├── server/
│   │   ├── app.py              ← FastAPI application + all endpoints
│   │   ├── environment.py      ← Core RL environment (reset/step/state)
│   │   ├── data_generator.py   ← Synthetic data with planted errors
│   │   ├── graders.py          ← F1 scoring, weighted F1, risk scoring
│   │   ├── security.py         ← Rate limiting, headers, auth
│   │   └── tasks.py            ← Task definitions (easy → expert)
│   ├── models.py               ← Pydantic models (Action, Observation, State)
│   ├── baseline.py             ← Baseline agent using Llama 3.1
│   └── client.py               ← Python client for the environment
├── tests/                      ← 78 pytest tests
├── inference.py                ← Hackathon inference script
├── openenv.yaml                ← OpenEnv spec configuration
├── Dockerfile                  ← Container for HF Spaces deployment
└── pyproject.toml              ← Dependencies and project config
```

---

## Architecture & Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Synthetic data (not static datasets)** | Prevents AI memorization — every seed generates unique data |
| **Deterministic F1 grading (not LLM-as-judge)** | Reproducible, fair, zero cost — no API calls needed for scoring |
| **Dense rewards (not sparse)** | +0.15 per correct finding enables RL training, not just evaluation |
| **Red herrings in data** | Forces precision — a naive "flag everything" strategy gets penalized |
| **Pydantic strict validation** | Safely handles raw LLM outputs — rejects malformed JSON immediately |
| **Session-based multi-tenancy** | Multiple agents can run concurrently without state conflicts |
| **REFERENCE_DATE (not datetime.now())** | Same seed = identical data regardless of when you run it |

---

## Future Scope

- **Scalability:** Async data generation workers for parallel RL training
- **International:** Expand beyond Indian GST to support EU VAT, US sales tax
- **Multi-agent:** Collaborative auditing where agents specialize in different error types
- **Adversarial mode:** Agent tries to plant errors that fool other agents
- **Fine-tuning dataset:** Export agent interactions as training data for smaller models
