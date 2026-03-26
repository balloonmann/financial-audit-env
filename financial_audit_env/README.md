# Financial Audit Environment

An OpenEnv-compatible reinforcement learning environment for training AI models on real-world financial auditing tasks.

**Tech Stack:** Python | FastAPI | Pydantic | HuggingFace Inference API

---

## Demo / Visual Proof
[Insert Demo Link or GIF Here]

Demonstrates an AI agent querying the environment, receiving synthetic invoices, and being graded on its findings.

---

## Tech Stack
- **Frontend:** Headless REST API
- **Backend:** Python, FastAPI, Uvicorn, Pydantic
- **Data / AI:** HuggingFace Inference API, Meta Llama 3.1 8B (Baseline Agent)
- **Infra / Tools:** Docker, Pytest, OpenEnv Framework

---

## Problem & Why This Wins
Financial auditing requires thousands of manual hours for invoice and GST reconciliation. Existing RL environments focus on toy games rather than complex, real-world business logic.

This environment bridges that gap by programmatically generating realistic Indian financial documents with mathematically verifiable planted errors, providing a deterministic validation framework for autonomous agents.

---

## Key Features
- **Synthetic Data Generation:** Dynamically constructs GST returns, purchase orders, and expense claims using reproducible random seeds.
- **Deterministic F1-Scoring:** Mathematically evaluates agent findings (True/False Positives) against hidden ground-truth errors instead of relying on subjective LLM-as-a-judge.
- **OpenEnv API Architecture:** Implements standard reinforcement learning endpoints (`/reset`, `/step`) over REST for universal agent compatibility.
- **Integrated Baseline Agent:** Provides a production-ready evaluation script leveraging the HuggingFace Inference API to test Meta Llama 3.1 against the environment.

---

## Quick Start
```bash
git clone https://github.com/YourUsername/financial-audit-env.git
cd financial-audit-env
python -m venv venv
.\venv\Scripts\activate
pip install -e .
python -m financial_audit_env.server.app
```

---

<details>
<summary><strong>Architecture, Data Flow & Design Decisions</strong></summary>

- **System Architecture:** Client (`POST /reset`) -> FastAPI Server -> Data Generator (Plants Errors) -> Returns JSON Observation -> Client (`POST /step`) -> Grader Module -> Returns F1 Reward.
- **Design Decision - Synthetic Data:** Relies on programmatic string and math manipulation rather than static datasets to prevent AI memorization.
- **Design Decision - Pydantic Validation:** Enforces strict structural typing on AI inputs to handle raw LLM outputs safely.
- **Trade-offs:** CPU-bound data generation causes minor latency spikes, favored over pre-baked datasets to maximize environmental variation.
- **Failure Handling:** Invalid JSON schemas from LLM agents are immediately rejected with descriptive 400 errors to guide model regeneration.

</details>

---

<details>
<summary><strong>Challenges & Key Learnings</strong></summary>

- **Hardest Technical Issue:** Resolving complex type-checker inference failures (Pyre/Pylance) while manipulating deeply nested, dynamically generated dictionaries.
- **What Broke:** Early implementation Pydantic schema mismatches dropped valid AI findings.
- **Key Learning:** Explicit type hinting of complex data structures is critical for stable static analysis, and deterministic scoring requires mathematically sound, seed-based ground-truth generation logic.

</details>

---

<details>
<summary><strong>Tasks, Action Space & Observation Space (Mandatory Spec)</strong></summary>

### Tasks & Expected Difficulty
| Task | Difficulty | Data Provided | Planted Errors |
|------|-----------|---------------|----------------|
| **Expense Policy Audits** | 🟢 Easy | 15 claims + policy definition | 6 violations |
| **Invoice 3-Way Math** | 🟡 Medium | 10 POs + 10 GRNs + 12 Invoices | 8 discrepancies |
| **GST Reconciliation** | 🔴 Hard | 30 book entries + 30 GSTR-2B entries | 12 mismatches |

### Action Space (`Pydantic AuditAction`)
```json
{
  "findings": [
    {
      "document_id": "EXP-010",
      "error_type": "over_limit",
      "description": "Meal expense of 4500 exceeds the 1500 daily limit",
      "suggested_fix": "Reject"
    }
  ],
  "submit_final": true
}
```

### Observation Space (`Pydantic AuditObservation`)
```json
{
  "task_id": "expense_audit",
  "task_description": "Review employee expense claims against policy.",
  "documents": { "expenses": [...], "policy": {...} },
  "findings_so_far": [],
  "feedback": "Step 1/5...",
  "step_number": 1,
  "max_steps": 5,
  "done": false,
  "reward": 0.15
}
```

### Baseline Scores (Meta Llama 3.1 8B)
*Scores generated via HuggingFace Inference API.*

| Task | Difficulty | Model | F1 Grader Score |
|------|------------|-------|-----------------|
| **Expense Audits** | Easy | Llama-3.1-8B-Instruct | `0.3158` |
| **Invoice Match** | Medium | Llama-3.1-8B-Instruct | `0.0000` |
| **GST Returns** | Hard | Llama-3.1-8B-Instruct | `0.0000` |

</details>

---

## Future Scope
- **Scalability:** Migrate data generation to asynchronous background workers to support high-throughput parallel reinforcement learning.
- **Production Considerations:** Expand GST formats to support international tax structures beyond the regional framework.
- **Feature Roadmap:** Introduce multi-agent collaborative auditing tasks.

---

## License
MIT License.
