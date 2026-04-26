<img width="2501" height="879" alt="llama_before_after_comparison" src="https://github.com/user-attachments/assets/a6cc2715-5ff3-415a-b20b-4716f045fa21" />
---
title: Financial Audit Env
emoji: 💰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
tags: [openenv]
---

# Financial Audit Environment — Multi-Agent Oversight Platform

An OpenEnv-compatible reinforcement learning environment for training AI agents to audit financial documents through **multi-agent cooperation**, **regulatory adaptation**, and **self-improvement**. Built for the [Meta PyTorch Hackathon — Round 2](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon).

**[Live API](https://balloonmann-financial-audit-env.hf.space/docs)** · **108 tests passing** · **Theme: #3 World Modeling — #3.1 Professional Tasks**

- Blog post: [BLOG.md](BLOG.md)
- Training notebook: [GRPO_Training_Submission_Final.ipynb](GRPO_Training_Submission_Final.ipynb)

---

## Submission Hub (Competition)

This section is the single source of truth for all links judges need.

### Required Links

- GitHub Repository: https://github.com/balloonmann/financial-audit-env
- Hugging Face Space (Environment URL): https://balloonmann-financial-audit-env.hf.space
- Training Notebook: [GRPO_Training_Submission_Final.ipynb](GRPO_Training_Submission_Final.ipynb)
- Blog Post: [BLOG.md](BLOG.md)
- GRPO Adapter (HF Hub): https://huggingface.co/balloonmann/financial-audit-grpo-adapter
- Eval Artifacts: https://huggingface.co/datasets/balloonmann/financial-audit-eval-artifacts

### Submission Checklist

- [x] OpenEnv-compatible environment hosted on HF Space
- [x] `openenv.yaml` present and valid
- [x] Training notebook (Unsloth + TRL GRPO) — `training/GRPO_Training_Submission.ipynb`
- [x] Blog post with results — `BLOG.md`
- [x] Adapter artifact uploaded to HF Hub
- [x] Eval artifacts (baseline + trained CSVs) on HF datasets
- [x] Training results embedded in README with plots

### How This README Maps to Judging Criteria

- Environment Innovation (40%): multi-agent campaign + shocks + schema drift + self-improvement.
- Storytelling (30%): domain framing, architecture, flow, and problem relevance are documented.
- Showing Improvement in Rewards (20%): scaffold present, final before/after evidence to be added tomorrow.
- Reward & Training Pipeline (10%): deterministic graders + reward parser + GRPO training path implemented.

---

## Round 2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Campaign Controller                          │
│  5-period campaigns with world mutation, schema drift,          │
│  regulatory shocks, and cross-period memory                     │
├────────────┬────────────┬────────────┬────────────┬─────────────┤
│  Expense   │  Invoice   │    GST     │   Fraud    │  Overseer   │
│ Specialist │ Specialist │ Specialist │ Specialist │   (Review)  │
├────────────┴────────────┴────────────┴────────────┴─────────────┤
│                    Instruction Registry                         │
│  22 base instructions + 3 regulatory shocks = 25 total          │
├─────────────────────────────────────────────────────────────────┤
│  Self-Improvement Engine  │  Adversarial Red/Blue  │  Training  │
│  (Critic + Regression     │  (Difficulty control   │  (GRPO +   │
│   Gate + Seed Separation) │   + Arms race)         │   Unsloth) │
└─────────────────────────────────────────────────────────────────┘
```

### What's New in Round 2

| Feature | Description |
|---------|-------------|
| **Multi-Agent Campaign** | 4 specialist agents + 1 overseer per period, task dependency order enforced |
| **5-Period Campaigns** | World mutates each period (policy changes, schema drift, vendor status) |
| **Regulatory Shocks** | 3 mid-period rule drops — agent must adapt without restart |
| **Schema Drift** | Field renames in period 3+ (e.g., `vendor_gstin` → `supplier_gstin`) |
| **Overseer Review** | Approves/rejects specialist findings, resolves conflicts |
| **Self-Improvement** | Critic analysis, regression gate, held-out seed separation |
| **Adversarial Red/Blue** | Tunable fraud difficulty (5 levels), arms race tracking |
| **22 Frozen Instructions** | Binary-checkable rules across 5 buckets (policy, dependency, deadline, escalation, schema) |
| **GRPO Training** | Colab-ready script using Unsloth + TRL GRPOTrainer |
| **Confidence Calibration** | ECE scoring when agents provide confidence scores |

---

## Campaign Flow

A Round 2 campaign runs **5 periods**, each with these phases:

```
Period N:
  1. Expense Specialist → audits expenses, submits findings
  2. Invoice Specialist → audits invoices (depends on expense done)
  3. GST Specialist     → reconciles GST (depends on invoice done)
  4. Fraud Specialist   → detects fraud (depends on all above + prior findings)
  5. Overseer           → reviews all findings, approves/rejects/escalates
  6. Advance Period     → world mutates (policy updates, schema drift, new alerts)

  ⚠️ REGULATORY SHOCKS may drop mid-period after specific steps!
     Agent receives new rule and must apply it to remaining work.
```

### World Mutation Across Periods

| Period | Changes |
|--------|---------|
| 1 | Baseline — all 17 active instructions apply |
| 2 | Meal limit increases ₹1,500→₹2,000, new vendor onboarded, cross-period memory required |
| 3 | GST rate change (IT services 18%→12%), schema drift (field renames), REG-001 shock |
| 4 | Vendor under investigation, REG-002 + REG-003 shocks (cash/UPI threshold, new vendor risk) |
| 5 | Annual reconciliation — full portfolio review, all 22 instructions active |

---

## Tasks

| # | Task | Difficulty | Documents | Errors | What the Agent Must Do |
|---|------|-----------|-----------|--------|----------------------|
| 1 | **Expense Policy Audit** | Easy | 19 expense claims + policy | 7 violations | Check claims against policy |
| 2 | **Invoice Three-Way Match** | Medium | 10 POs + 10 GRNs + 12 invoices | 9 discrepancies | Cross-reference 3 document types |
| 3 | **GST Reconciliation** | Hard | 45 book entries + 44 GSTR-2B | 12 mismatches | Reconcile books vs government data |
| 4 | **Fraud Detection** | Expert | 84 transactions + 26 vendors | 10 fraud patterns | Forensic pattern recognition |

---

## API Reference

### Standard Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome page with API docs link |
| `/health` | GET | Status, version, task count |
| `/docs` | GET | Interactive Swagger UI |
| `/tasks` | GET | All 4 tasks with descriptions and error types |
| `/reset` | POST | Start a new episode (`task_id` + `seed`) |
| `/step` | POST | Submit findings (`submit_final: true` to end) |
| `/state` | GET | Current episode progress |
| `/grader` | GET | Full scoring breakdown |
| `/session` | POST | Create isolated session |
| `/leaderboard` | GET/POST | Best scores per model |
| `/metrics` | GET | Usage statistics |
| `/adaptive-difficulty` | GET | Difficulty suggestions |

### Round 2 Campaign Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/campaign/start` | POST | Start 5-period campaign (`seed`, `total_periods`) |
| `/campaign/task/start` | POST | Start specialist task in current period (`campaign_id`, `role`) |
| `/campaign/task/submit` | POST | Submit specialist findings (returns regulatory shocks if triggered) |
| `/campaign/period/advance` | POST | Advance period (world mutation) |
| `/campaign/state/{id}` | GET | Full campaign state |
| `/overseer/review` | POST | Overseer reviews specialist findings |
| `/self-improve` | POST | Run one improvement iteration (strict seed separation) |
| `/self-improve/history` | GET | Iteration history for reward curves |

---

## Grading System

### Per-Task Scoring

| Metric | What It Measures |
|--------|-----------------| 
| **Partial Credit F1** (0.01–0.99) | Primary metric — rewards right document even with wrong error type (0.40 credit) |
| **Strict F1** | Exact `(document_id, error_type)` matches only |
| **Weighted F1** | Critical errors count more (fraud=2.0×, weekend expense=0.5×) |
| **Partial Credit** | Right document, wrong error type = 0.40 credit (false positive weight = 0.60) |
| **Confusion Matrix** | Per-error-type breakdown |
| **Risk Score** | Monetary value of caught vs missed errors |
| **ECE** | Expected Calibration Error (when confidence scores provided) |

### Campaign-Level Scoring

Formula: **35%** specialist F1 + **25%** overseer quality + **10%** instruction compliance + **10%** memory + **8%** schema/policy + **7%** improvement + **5%** efficiency.

**Anti-gaming guards:**
- Any specialist weighted F1 < 0.20 → multiplier = 0.0
- Any critical error missed (severity ≥ 1.5) → multiplier = 0.5
- Bonus components capped at 30% of total

### Reward Signal

- **+0.15** per new true positive (severity-weighted)
- **+0.04** per partial match (right doc, wrong error type)
- **−0.05** per false positive
- **−0.02** step penalty + **−0.005** × step_number decay
- **+0.30** bonus on final step if recall ≥ 0.6
- **−0.20** penalty on final step if recall < 0.3

---

## Training

### GRPO Training with Unsloth + TRL

The `training/train_grpo.py` script is designed for Google Colab with free T4 GPU:

```bash
# In Colab:
!pip install -r requirements-training.txt

# Upload financial_audit_env/ and training/ directories, then:
!python training/train_grpo.py
```

HF Jobs path (A10G):

```bash
hf jobs run --flavor a10g-large --timeout 6h --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -lc "set -euo pipefail; apt-get update -qq; apt-get install -y -qq git; git clone https://github.com/balloonmann/financial-audit-env.git; cd financial-audit-env; bash scripts/hf_jobs_bootstrap_and_train.sh"
```

**Key components:**
- **InProcessEvaluator** (`training/evaluator.py`) — Direct Python evaluation, no HTTP overhead
- **Reward Function** (`training/reward.py`) — Parses JSON/free-text model output, returns F1-based score
- **GRPO Config** — 4bit quantized Llama 3.1 8B, LoRA r=16, 10 training seeds × 4 tasks = 40 prompts

### Seed Separation

| Set | Seeds | Purpose |
|-----|-------|---------|
| Training | 42–51 | GRPO optimization |
| Held-out | 100–104 | Regression gate evaluation |

The self-improvement engine enforces **strict disjoint seed sets** — overlapping seeds are rejected.

### Dry-Run Verification

```bash
# Verify pipeline locally (no GPU needed):
python training/train_grpo.py
# Output: "✅ All pipeline components verified. Ready for Colab with Unsloth."
```

---

### Training Results — Held-Out Seeds 100–104

#### Llama 3.1 8B — HuggingFace Jobs (A10-Large GPU)

| | Mean Score |
|---|---|
| Baseline | 0.1690 |
| GRPO Trained | 0.1230 |
| Delta | -0.0460 (-27.2%) |

| Task | Difficulty | Baseline F1 | Trained F1 | Delta |
|---|---|---|---|---|
| expense_audit | Easy | ~0.12 | **0.356** | +196% |
| invoice_match | Medium | ~0.18 | 0.074 | -59% |
| gst_reconciliation | Hard | ~0.01 | 0.042 | +320% |
| fraud_detection | Expert | ~0.11 | 0.020 | -82% |

Expense audit improved dramatically after GRPO. Fraud detection collapsed — the optimizer chased the densest reward signal. See [BLOG.md](BLOG.md) for the full analysis.

#### Qwen 2.5-1.5B — Google Colab (T4)

| | Mean Score |
|---|---|
| Baseline | 0.0470 |
| GRPO Trained | 0.0100 |
| Delta | -0.0370 (-78.7%) |

| Task | Baseline F1 | Trained F1 |
|---|---|---|
| expense_audit | 0.026 | 0.010 |
| invoice_match | 0.122 | 0.010 |
| gst_reconciliation | 0.018 | 0.007 |
| fraud_detection | 0.022 | 0.010 |

At 1.5B parameters quantized to 4-bit, the model collapses to floor on all tasks after training — insufficient capacity to absorb the GRPO policy changes while maintaining valid output format.

#### Comparison Plot

![Held-out Score and Recall — Baseline vs GRPO Trained (Llama 3.1 8B)](<img width="2501" height="879" alt="llama_before_after_comparison" src="https://github.com/user-attachments/assets/957c3c91-87fa-4971-ae36-5d5b79552944" />
)

*Left: F1 score per task. Right: Recall per task. Orange = GRPO trained, Blue = baseline. Evaluated on held-out seeds 100–104.*

Artifact links:
- Eval CSVs: https://huggingface.co/datasets/balloonmann/financial-audit-eval-artifacts
- Adapter: https://huggingface.co/balloonmann/financial-audit-grpo-adapter

---

## Inference

### Round 1 (Single-Task)

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1/
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
python inference.py --env-url http://localhost:8000
```

**Baseline Scores**

| Task | Difficulty | F1 Score | Precision | Recall |
|------|-----------|----------|-----------|--------|
| Expense Audit | Easy | 0.1200 | 0.07 | 0.43 |
| Invoice Match | Medium | 0.1800 | 0.11 | 0.44 |
| GST Reconciliation | Hard | 0.0100 | 0.01 | 0.01 |
| Fraud Detection | Expert | 0.1100 | 0.11 | 0.10 |

Scoring notes:

- Seed: `42`
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- API provider: HuggingFace Router (`https://router.huggingface.co/v1/`)
- Baseline parser includes malformed-JSON recovery and compact-prompt fallback for strict context windows
- Results may vary slightly by provider-side model revision and transient inference behavior

**Why did it fail?**
The model frequently drops to 0.00 on the tasks because it struggles with abstract rules (like date math for weekend expenses, or tracking cumulative limits). It actively gets tricked by "red herrings"—perfectly legal expenses that it hallucinates as errors—which entirely destroys its precision score.


### Round 2 (Campaign)

```bash
python inference.py --env-url http://localhost:8000 --campaign --seed 42
```

The campaign flow:
1. Starts a 5-period campaign via `/campaign/start`
2. Runs all 4 specialists per period in dependency order
3. Handles regulatory shocks returned in step responses
4. Submits overseer review after specialists complete
5. Advances period (world mutation)
6. Logs with `[START]/[STEP]/[END]` format

### Automated Hackathon Demo (No-LLM Path)

Use a deterministic no-LLM flow to generate a reproducible demo artifact:

```bash
python scripts/run_hackathon_demo.py --env-url http://localhost:8000 --seed 42 --periods 5 --output artifacts/hackathon_demo_summary.json
```

If protected endpoints are enabled, include API key:

```bash
python scripts/run_hackathon_demo.py --env-url http://localhost:8000 --api-key "$ADMIN_API_KEY"
```

---

## Getting Started

### Run Locally

```bash
git clone https://github.com/balloonmann/financial-audit-env.git
cd financial-audit-env
pip install -e .
python -m financial_audit_env.server.app
```

### Docker

```bash
docker build -t financial-audit-env .
docker run -p 8000:8000 financial-audit-env
```

### Run Tests

```bash
python -m pytest tests/ -x -v
```

### Judge Runbook (Fast Repro)

Use this exact sequence for a quick end-to-end verification:

```bash
# 1) Install package
pip install -e .

# 2) Start environment server
python -m financial_audit_env.server.app

# 3) In a new terminal: run all tests
python -m pytest tests -q

# 4) Run campaign inference flow
python inference.py --env-url http://localhost:8000 --campaign --seed 42

# 5) Hit the live Space directly
curl https://balloonmann-financial-audit-env.hf.space/health
```

---

## Project Structure

```
financial_audit_env/
  server/
    app.py              # FastAPI — all endpoints (standard + Round 2 campaign)
    environment.py      # Core RL environment (reset, step, state)
    data_generator.py   # Synthetic data with planted errors + schema drift helpers
    graders.py          # Partial Credit F1 (primary), weighted F1, ECE, campaign score, cross-agent agreement
    campaign.py         # Campaign controller — 5-period orchestration
    instructions.py     # 22 frozen instructions + 3 regulatory shocks
    regulatory.py       # Mid-period regulatory shock engine
    adversarial.py      # Red/Blue adversarial fraud difficulty control
    self_improve.py     # Self-improvement with regression gate
    security.py         # Rate limiting, security headers
    tasks.py            # Task definitions (4 tasks, easy → expert)
  models.py             # Pydantic models (Round 2: AgentRole, WorldState, CampaignState, etc.)
  baseline.py           # Baseline agent using Llama 3.1
training/
  evaluator.py          # InProcessEvaluator (no HTTP overhead)
  reward.py             # GRPO reward function (JSON + free-text parsing)
  train_grpo.py         # Colab training script (Unsloth + TRL)
tests/                  # 100+ pytest tests
inference.py            # Hackathon inference script (Round 1 + Round 2 campaign)
openenv.yaml            # OpenEnv specification config
Dockerfile              # HuggingFace Spaces deployment
```

---

## Design Decisions

**Why multi-agent?** Real auditing is team-based. Expense specialists focus on policy, invoice specialists on three-way matching, GST specialists on regulatory compliance. An overseer coordinates and resolves conflicts. This creates natural task dependencies and collaboration dynamics.

**Why 5 periods?** Long-horizon campaigns test memory, adaptation, and planning. Period 1 is baseline, period 3 introduces schema drift, period 4 brings regulatory shocks — the agent can't just memorize period 1 patterns.

**Why regulatory shocks?** Mid-episode rule changes are realistic (tax law changes mid-quarter) and test the agent's ability to adapt without restarting. This goes beyond static environments.

**Why strict seed separation?** If training and evaluation use the same seeds, the agent overfits. Separate seed pools (42–51 train, 100–104 held-out) ensure genuine improvement.

**Why deterministic scoring?** Same findings → same score, always. No LLM-as-judge, no randomness. This makes GRPO training signals clean and reproducible.

---

## Hackathon Alignment

| Judging Criteria | How We Address It |
|-----------------|-------------------|
| **Environment Innovation (40%)** | Multi-agent oversight with regulatory shocks, schema drift, self-improvement — goes well beyond static eval |
| **Storytelling (30%)** | Clear campaign flow: specialists → overseer → advance → adapt. Real-world financial auditing domain |
| **Showing Improvement in Rewards (20%)** | GRPO training script + self-improvement engine with before/after comparison on held-out seeds |
| **Reward and Training Pipeline (10%)** | InProcessEvaluator + GRPO reward function + Colab-ready training script |

### Minimum Submission Requirements Coverage

| Requirement from Guidelines | Current Status | Notes |
|-----------------------------|----------------|-------|
| Use OpenEnv (latest release) | Covered | Dependency floor set to `openenv-core>=0.2.3` |
| Minimal training script via Unsloth or HF TRL (Colab-rerunnable) | Covered | `training/train_grpo.py` + Colab scripts present |
| Evidence of real training (loss/reward plots) | Pending | To be added after final run tomorrow |
| Short write-up artifact (HF blog / <2 min video / slides) | Pending | To be added tomorrow |
| Environment hosted on Hugging Face Space | Covered | Live Space link provided |
| README includes all links and results | Partially covered | Link scaffold added; pending final artifact URLs and plots |

---

## Round 2 Implementation Scorecard

> Verified on 2026-04-24 with `python verify_r2_score.py` and `python -m pytest tests -q` - campaign flow wired and checks passing.

### Implementation Status

| Step | Component | Status | Verified |
|------|-----------|--------|----------|
| 1 | **Core Models** — `AgentRole`, `WorldState`, `CampaignState`, `OverseerAction`, `CriticReport`, `CampaignObservation` | ✅ Complete | All models import, `Finding` has `confidence`, `evidence_refs`, `rationale` |
| 2 | **Instructions Registry** — 22 frozen instructions + 3 regulatory shocks across 5 buckets | ✅ Complete | Period 1: 16 active, Period 2: 19, Period 3+: 22. Shock timing verified |
| 3 | **Campaign Controller** — 5-period orchestration with composition over inheritance | ✅ Complete | Start, task dependency, submit, advance, world mutation all work |
| 4 | **Extended Grading** — ECE, campaign score, cross-agent agreement | ✅ Complete | Anti-gaming guards fire correctly (see below) |
| 5 | **Self-Improvement + API** — Critic analysis, regression gate, 12 Round 2 endpoints | ✅ Complete | Seed overlap rejected, iteration history tracked |
| 6 | **Regulatory Shock Engine** — Mid-period rule injection with ground truth modification | ✅ Complete | REG-001 at P3/S3, REG-002+003 at P4. GT extends correctly |
| 7 | **Adversarial Red/Blue** — 5-level fraud difficulty with arms race tracking | ✅ Complete | Difficulty adapts on F1 > 0.70, deterministic per seed |
| 8 | **Training Infrastructure** — InProcessEvaluator, reward parser, GRPO script | ✅ Complete | JSON + free-text parsing, Colab dry-run ready |
| 9 | **Inference (Campaign)** - Multi-period campaign inference flow | Complete | Campaign runner, endpoints, and CLI flag are present and validated |
| 10 | **Tests + README** - pytest suite + documentation | Complete | Full test suite currently passes locally |

**Overall: 10/10 steps complete on implementation readiness.**

### Verified Scoring Metrics

#### Campaign-Level Score Composition

| Component | Weight | Description |
|-----------|--------|-------------|
| Specialist F1 (avg weighted) | 35% | Average severity-weighted F1 across 4 specialists |
| Overseer Quality | 25% | Correct approvals + rejections / total decisions |
| Instruction Compliance | 10% | Binary check against 22 frozen instructions |
| Cross-Period Memory | 10% | Findings carried forward, prior-period context used |
| Schema/Policy Adaptation | 8% | Handling schema drift (P3+) and policy updates (P2+) |
| Self-Improvement Delta | 7% | Score gain on held-out seeds after improvement iteration |
| Efficiency | 5% | Budget usage and step economy |

#### Anti-Gaming Guards (Verified)

| Guard | Trigger | Effect | Verified Value |
|-------|---------|--------|----------------|
| **Specialist Floor** | Any specialist weighted F1 < 0.20 | Score → **0.01** (multiplier = 0.0) | ✅ Fires correctly |
| **Safety Gate** | Critical error missed (severity ≥ 1.5) | Score × **0.50** | ✅ 0.62 → 0.31 |
| **Bonus Cap** | Non-core bonuses > 30% of raw total | Bonuses clamped | ✅ Enforced |

#### Verified Campaign Score Examples

| Scenario | Specialist Avg | Overseer | Compliance | Memory | Score |
|----------|---------------|----------|------------|--------|-------|
| Strong all-around | 0.6375 | 0.80 | 0.90 | 0.70 | **0.62** |
| One weak specialist (F1 < 0.20) | 0.5125 | 0.80 | 0.90 | 0.70 | **0.01** |
| Critical error missed | 0.6375 | 0.80 | 0.90 | 0.70 | **0.31** |

### Test Coverage

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_campaign_round2.py` | 10 | Campaign start, task submit, advance, reproducibility, full 5-period, overseer, self-improve, regulatory, state endpoints |
| `test_data_generators.py` | 22 | All 4 data generators, error types, reproducibility, red herrings, reference dates |
| `test_environment.py` | 20 | Reset/step, all 4 tasks, investigation mode, adaptive difficulty, edge cases |
| `test_graders.py` | 15 | F1, weighted F1, partial credit, confusion matrix, risk scoring, step reward |
| `test_regulatory.py` | 7 | Shock delivery, ground truth modification, timing, schema drift, tax recalculation |
| `test_security.py` | 5 | Rate limiting, IP independence, memory leak prevention, error sanitization |
| `test_self_improve.py` | 6 | Iteration tracking, seed overlap rejection, multiple iterations, candidate scoring |
| `test_adversarial.py` | 4 | Difficulty levels, adaptation, arms race data, determinism |
| **Total** | **108** | **All passing in ~10s** |

### API Surface

| Category | Endpoints | Count |
|----------|-----------|-------|
| Standard OpenEnv | `/`, `/health`, `/docs`, `/tasks`, `/reset`, `/step`, `/state`, `/grader`, `/session`, `/leaderboard`, `/metrics`, `/adaptive-difficulty`, `/baseline` | 13 |
| Campaign | `/campaign/start`, `/campaign/state`, `/campaign/state/{id}`, `/campaign/action`, `/campaign/task/start`, `/campaign/task/submit`, `/campaign/period`, `/campaign/period/advance` | 8 |
| Oversight | `/overseer/review`, `/overseer/report` | 2 |
| Self-Improvement | `/self-improve`, `/self-improve/history` | 2 |
| **Total** | | **29 routes** |

### Key Quantities

| Metric | Value |
|--------|-------|
| Instructions (frozen) | 22 base + 3 regulatory shocks = **25** |
| Instruction buckets | 5 (policy, dependency, deadline, escalation, schema) |
| Agent roles | 4 specialists + 1 overseer = **5** |
| Campaign periods | **5** with deterministic world mutation |
| Fraud difficulty levels | **5** (obvious → adversarial) |
| Training seeds | 42–51 (**10**) |
| Held-out seeds | 100–104 (**5**) |
| Score range | **(0.01, 0.99)** — strictly bounded, no 0.0 or 1.0 |
| Grading functions | F1, weighted F1, partial credit, ECE, campaign score, cross-agent agreement, step reward = **7** |
