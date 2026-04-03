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

An OpenEnv-compatible RL environment for financial auditing tasks. Agents operate over synthetic Indian financial documents — expense claims, invoices, GST returns, vendor transactions — and are scored on their ability to identify planted errors via deterministic F1 grading.

**[Live API](https://balloonmann-financial-audit-env.hf.space/docs)** | **78 tests passing**

---

## Environment Overview

Tasks with increasing complexity. Each task generates unique documents per seed, plants a fixed number of verifiable errors, and includes red herrings to penalize indiscriminate flagging.

| Task | Difficulty | Documents | Planted Errors |
|------|-----------|-----------|---------------|
| `expense_audit` | Easy | 19 expense claims + policy definition | 7 violations |
| `invoice_match` | Medium | 10 POs + 10 GRNs + 12 invoices | 9 discrepancies |
| `gst_reconciliation` | Hard | 45 book entries + 44 GSTR-2B entries | 12 mismatches |

Red herrings are deliberate. An expense at exactly the daily limit is valid. A ₹1 rounding difference on an invoice is within tolerance. The grader only awards credit for correctly identified violations, not suspicious-looking entries that happen to be clean.

---

## Reward Structure

Per-step reward: **+0.15** per true positive, **-0.05** per false positive.

Terminal grading via F1 score on matched `(document_id, error_type)` pairs:

- **Standard F1** — equal weight across all error types
- **Weighted F1** — fraud and GST violations weighted higher than minor policy infractions
- **Partial credit** — correct document, wrong error type scores 0.25
- **Risk score** — rupee value of financial exposure covered by correct findings
- **Confusion matrix** — per-error-type breakdown of TP, FP, FN

Scores are fully deterministic. Same seed, same findings, same score every time.

---

## API

```
POST /reset     {"task_id": "expense_audit", "seed": 42}
POST /step      {"action": {"findings": [...], "submit_final": true}}
GET  /state     current step, errors found, false positives
GET  /grader    full scoring breakdown after episode completion
GET  /tasks     all tasks with allowed error types and schemas
GET  /health    status, version, task count
```

Additional endpoints:

```
POST /session            isolated session for parallel agent runs
GET  /leaderboard        best scores per model
GET  /metrics            uptime, total resets, total steps
GET  /adaptive-difficulty   difficulty recommendation based on score history
```

Full interactive docs at `/docs`.

---

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

`submit_final: false` accumulates findings across steps (max 5). `submit_final: true` triggers terminal grading.

---

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

---

## Investigation Mode

Reset with `"investigation_mode": true` to withhold raw document data on episode start. The agent receives document counts and category summaries, then requests specific categories (at step cost) before submitting findings. Tests prioritisation behaviour rather than exhaustive search.

---

## Baseline

Llama 3.1 8B Instruct via HuggingFace Inference API. Uses OpenAI-compatible client as required by contest spec.

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1/
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
python inference.py --env-url http://localhost:8000
```

| Task | Difficulty | F1 Score |
|------|-----------|----------|
| `expense_audit` | Easy | 0.3158 |
| `invoice_match` | Medium | 0.0000 |
| `gst_reconciliation` | Hard | 0.0000 |

---

## Setup

**Local:**
```bash
git clone https://github.com/balloonmann/financial-audit-env.git
cd financial-audit-env
pip install -e .
python -m financial_audit_env.server.app
```

**Docker:**
```bash
docker build -t financial-audit-env .
docker run -p 8000:8000 financial-audit-env
```

**Hosted:** `https://balloonmann-financial-audit-env.hf.space`

---

## Architecture

```
financial_audit_env/
  server/
    app.py              # FastAPI — all endpoints
    environment.py      # reset(), step(), state() — core RL loop
    data_generator.py   # seed-based synthetic document generation
    graders.py          # F1, weighted F1, risk scoring
    security.py         # rate limiting, OWASP headers, input validation
    tasks.py            # task registry (4 tasks)
  models.py             # Pydantic — AuditAction, AuditObservation, EpisodeState
  baseline.py           # Llama 3.1 8B baseline agent
tests/                  # 78 pytest tests
inference.py            # contest inference script
openenv.yaml            # OpenEnv spec config
Dockerfile
```

---

## Design Decisions

**Seed-based generation** prevents memorisation. Every unique seed produces a structurally valid but distinct document set with freshly planted errors at randomised positions and values.

**Deterministic grading** removes LLM-as-judge variability. Ground truth is computed at generation time and stored server-side. The grader matches submitted `(document_id, error_type)` pairs against it mathematically.

**Dense per-step rewards** make the environment usable for RL training, not just evaluation. Terminal-only rewards provide insufficient signal for policy gradient methods on tasks with sparse correct answers.

**Red herrings enforce precision.** Without plausible-but-valid entries, recall-maximising strategies (flag everything) score perfectly. Red herrings make precision a load-bearing part of the F1 score.

**Fixed reference date (Jan 15, 2026)** keeps generation reproducible across calendar days. `datetime.now()` breaks seed determinism.

**Rate limiting at 30 req/min per IP**, 1MB request cap, sanitised error messages (ground truth never exposed), `X-Request-ID` on every response.

---

## Roadmap

- Async data generation for high-throughput parallel RL training
- International tax structures (EU VAT, US sales tax) beyond Indian GST
- Multi-agent collaborative auditing with specialisation by error type
- Agent interaction export as supervised fine-tuning datasets
