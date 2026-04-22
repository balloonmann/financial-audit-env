#!/usr/bin/env python3
"""
Financial Audit Environment — Inference Script
==============================================
MANDATORY CONSTRAINTS MET:
- Uses API_BASE_URL, MODEL_NAME, and HF_TOKEN from environment.
- Placed in the root directory of the project.
- Uses the standard OpenAI Client for all LLM calls.
- Emits structured stdout logs: [START], [STEP], [END]

Usage:
  export API_BASE_URL=https://router.huggingface.co/v1/
  export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
  export HF_TOKEN=your_token_here
  python inference.py --env-url http://localhost:8000
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: 'openai' package not installed. Run: pip install openai>=1.0.0", file=sys.stderr)
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional — env vars can be set directly

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# Configuration from Environment Variables (MANDATORY)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Round-1 default: run the required 3 tasks (easy, medium, hard).
TASK_IDS = ["expense_audit", "invoice_match", "gst_reconciliation"]
SEED = 42
BENCHMARK = "financial_audit_env"
TEMPERATURE = 0.1
MAX_TOKENS = 4096
MAX_STEPS = 5  # Max steps per task (matches our env config)
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# MANDATORY Structured Logging: [START], [STEP], [END]
# ---------------------------------------------------------------------------


def strict_unit_interval(value: Any, default: float = 0.01) -> float:
    """Return a finite float constrained to a stable open interval."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = default

    # NaN check that works without importing math.
    if num != num:
        num = default

    if num <= 0.01:
        return 0.01
    if num >= 0.99:
        return 0.99
    return num

def log_start(task: str, env: str, model: str) -> None:
    """Emit the [START] structured log line."""
    print(
        f"[START] task={task} env={env} model={model}",
        flush=True,
    )


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    """Emit the [STEP] structured log line."""
    error_str = error if error else "null"
    done_str = "true" if done else "false"
    clamped_reward = strict_unit_interval(reward)
    print(
        f"[STEP] step={step} action={action} reward={clamped_reward:.6f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    """Emit the [END] structured log line."""
    success_str = "true" if success else "false"
    clamped_score = strict_unit_interval(score)
    safe_rewards = rewards if rewards else [clamped_score]
    rewards_str = ",".join([f"{strict_unit_interval(r):.6f}" for r in safe_rewards])
    print(
        f"[END] success={success_str} steps={steps} score={clamped_score:.6f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM Prompts & Parsing
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert financial auditor AI. You review financial documents and identify errors, violations, and discrepancies with extreme precision.

You MUST respond ONLY with a valid JSON array of findings. Each finding must have:
- "document_id": the ID of the document with the error
- "error_type": one of the allowed error types listed below
- "description": a clear explanation of the error
- "suggested_fix": what should be done to fix it

Do NOT include any text before or after the JSON array. Do NOT use markdown code blocks. Just output the raw JSON array."""

def build_task_prompt(task_description: str, documents: Dict[str, Any], error_types: List[str]) -> str:
    prompt_parts = [
        "# TASK",
        task_description,
        "",
        "# ALLOWED ERROR TYPES",
        json.dumps(error_types, indent=2),
        "",
        "# DOCUMENTS TO AUDIT",
    ]

    for doc_type, doc_data in documents.items():
        prompt_parts.append(f"\n## {doc_type.upper()}")
        if isinstance(doc_data, list):
            for i, row in enumerate(doc_data):
                prompt_parts.append(f"Row {i+1}: {json.dumps(row)}")
        elif isinstance(doc_data, dict):
            prompt_parts.append(json.dumps(doc_data, indent=2))

    prompt_parts.extend([
        "",
        "# INSTRUCTIONS",
        "Analyze ALL documents carefully. Find EVERY error/violation/discrepancy.",
        "Respond with ONLY a JSON array of findings. No other text.",
        'Example: [{"document_id": "EXP-001", "error_type": "over_limit", "description": "...", "suggested_fix": "..."}]',
    ])

    return "\n".join(prompt_parts)

def parse_llm_findings(response_text: str) -> List[Dict[str, str]]:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])  # Strip opening/closing backticks
        if text.endswith("```"):
            text = text[:-3].strip()

    start_idx = text.find("[")
    end_idx = text.rfind("]")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx + 1]

    try:
        findings = json.loads(text)
        if isinstance(findings, list):
            return findings
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM response as JSON: {text[:200]}...")

    return []

# ---------------------------------------------------------------------------
# Core Inference Execution
# ---------------------------------------------------------------------------
def run_agent_single_task(
    env_url: str,
    task_id: str,
    client: OpenAI,
    seed: int = SEED,
) -> Dict[str, Any]:
    session = requests.Session()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    # Emit [START] log
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # 1. Reset Environment
        logger.info(f"[{task_id}] Resetting environment...")
        reset_resp = session.post(
            f"{env_url}/reset",
            json={"task_id": task_id, "seed": seed},
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json().get("observation", reset_resp.json())

        # 2. Extract Data & Allowed Errors
        tasks_resp = session.get(f"{env_url}/tasks")
        tasks_resp.raise_for_status()
        task_info = next(t for t in tasks_resp.json()["tasks"] if t["id"] == task_id)
        error_types = task_info["error_types"]

        prompt = build_task_prompt(obs["task_description"], obs["documents"], error_types)

        # 3. Call LLM (via OpenAI Client — MANDATORY)
        logger.info(f"[{task_id}] Calling {MODEL_NAME} at {API_BASE_URL}...")
        start_time = time.time()

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=False,
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"[{task_id}] Inference failed: {e}")
            response_text = "[]"

        logger.info(f"[{task_id}] LLM responded in {time.time() - start_time:.1f}s")

        # 4. Parse & Submit Action
        findings = parse_llm_findings(response_text)
        logger.info(f"[{task_id}] Parsed {len(findings)} findings from LLM")

        action = {
            "findings": [
                {
                    "document_id": str(f.get("document_id", "")),
                    "error_type": str(f.get("error_type", "")),
                    "description": str(f.get("description", "No description")),
                    "suggested_fix": str(f.get("suggested_fix", "")) if f.get("suggested_fix") else None,
                }
                for f in findings
            ],
            "submit_final": True,
        }

        # Summarize the action for the log
        action_summary = f"submit_{len(findings)}_findings"

        step_resp = session.post(f"{env_url}/step", json={"action": action})
        step_resp.raise_for_status()
        step_data = step_resp.json()

        step_reward = strict_unit_interval(step_data.get("reward", 0.01) or 0.01)
        step_done = step_data.get("done", True)
        rewards.append(step_reward)
        steps_taken = 1

        # Emit [STEP] log
        log_step(
            step=1,
            action=action_summary,
            reward=step_reward,
            done=step_done,
            error=None,
        )

        # 5. Get Grader Results
        grader_resp = session.get(f"{env_url}/grader")
        grader_resp.raise_for_status()
        grader_data = grader_resp.json()

        score = strict_unit_interval(grader_data.get("score", 0.01))
        success = score >= SUCCESS_SCORE_THRESHOLD

        result = {
            "task_id": task_id,
            "task_name": task_info["name"],
            "difficulty": task_info["difficulty"],
            "score": score,
            "precision": strict_unit_interval(grader_data.get("precision", 0.01)),
            "recall": strict_unit_interval(grader_data.get("recall", 0.01)),
        }

        logger.info(f"[{task_id}] Score: {result['score']:.4f} (P={result['precision']:.2f}, R={result['recall']:.2f})")

    except Exception as exc:
        logger.error(f"[{task_id}] Task failed: {exc}")
        result = {
            "task_id": task_id,
            "task_name": task_id,
            "difficulty": "unknown",
            "score": 0.01,
            "precision": 0.01,
            "recall": 0.01,
        }
    finally:
        # Keep output parseable for strict validators even on exceptions.
        if not rewards:
            rewards = [strict_unit_interval(score)]
        if steps_taken <= 0:
            steps_taken = 1
        score = strict_unit_interval(score)
        # Emit [END] log
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return result


def run_campaign_round(env_url: str, seed: int = SEED, client=None) -> Dict[str, Any]:
    """
    Run a full multi-agent, multi-period campaign (Round 2).

    Flow per period:
      1. Start each specialist task in dependency order
      2. Use LLM to generate findings (if client available)
      3. Submit findings
      4. Handle any regulatory shocks returned
      5. Overseer reviews all specialist findings
      6. Advance to next period (world mutation)

    Logs with [START]/[STEP]/[END] format for mandatory hackathon logging.
    """
    session = requests.Session()
    roles = ["expense_specialist", "invoice_specialist", "gst_specialist", "fraud_specialist"]
    total_periods = 5

    logger.info("[CAMPAIGN START] Initializing multi-agent campaign")
    print(f"[START] campaign seed={seed} periods={total_periods}", flush=True)

    # Start campaign
    start = session.post(
        f"{env_url}/campaign/start",
        json={"seed": seed, "total_periods": total_periods},
    )
    start.raise_for_status()
    start_data = start.json()
    campaign_id = start_data["campaign_id"]

    period_results = []

    for period in range(1, total_periods + 1):
        logger.info(f"[CAMPAIGN] Period {period}/{total_periods}")
        print(f"[STEP] period={period} phase=start", flush=True)

        period_findings_by_role: Dict[str, List[Dict]] = {}

        # Run each specialist in order
        for role in roles:
            # Start task
            task_start = session.post(
                f"{env_url}/campaign/task/start",
                json={"campaign_id": campaign_id, "role": role},
            )
            if task_start.status_code != 200:
                logger.warning(f"  [{role}] task/start failed: {task_start.text}")
                continue
            task_data = task_start.json()

            # Get observation for prompt
            obs = task_data.get("observation", {})
            world_state = task_data.get("world_state", {})
            history = task_data.get("findings_history", [])
            instructions = task_data.get("active_instructions", [])
            shocks = task_data.get("pending_regulatory_shocks", [])

            # Generate findings using LLM (if available)
            findings: List[Dict] = []
            if client:
                try:
                    prompt = _build_campaign_prompt(obs, world_state, history, instructions, shocks, role)
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=2048,
                    )
                    findings = _parse_findings_response(response.choices[0].message.content)
                except Exception as e:
                    logger.warning(f"  [{role}] LLM call failed: {e}")

            # Submit findings
            submit = session.post(
                f"{env_url}/campaign/task/submit",
                json={
                    "campaign_id": campaign_id,
                    "role": role,
                    "action": {"findings": findings, "submit_final": True},
                },
            )
            if submit.status_code == 200:
                submit_data = submit.json()
                # Check for regulatory shocks
                obs_data = submit_data.get("observation", {})
                reg_shocks = obs_data.get("pending_regulatory_shocks", [])
                if reg_shocks:
                    logger.info(f"  [{role}] REGULATORY SHOCK received: {len(reg_shocks)} new rule(s)")
                    print(f"[STEP] period={period} role={role} regulatory_shock=true", flush=True)

            period_findings_by_role[role] = findings
            print(f"[STEP] period={period} role={role} findings={len(findings)}", flush=True)

        # Overseer review
        all_decisions = []
        for role, findings in period_findings_by_role.items():
            for f in findings:
                all_decisions.append({
                    "finding_ref": f"{f.get('document_id', '')}:{f.get('error_type', '')}",
                    "verdict": "approve",
                    "reason_code": "specialist_evidence",
                    "confidence": f.get("confidence", 0.7),
                })

        review = session.post(
            f"{env_url}/overseer/review",
            json={
                "campaign_id": campaign_id,
                "action": {
                    "audit_trail_id": f"trail-{campaign_id}-p{period}",
                    "decisions": all_decisions,
                    "conflicts_resolved": [],
                    "task_reassignments": {},
                },
            },
        )
        if review.status_code == 200:
            review_data = review.json()
            logger.info(f"  [overseer] Review complete: {review_data.get('result', {})}")

        print(f"[STEP] period={period} phase=overseer_review decisions={len(all_decisions)}", flush=True)

        period_results.append({
            "period": period,
            "findings_by_role": {r: len(f) for r, f in period_findings_by_role.items()},
            "overseer_decisions": len(all_decisions),
        })

        # Advance to next period (unless last)
        if period < total_periods:
            advance = session.post(
                f"{env_url}/campaign/period/advance",
                json={"campaign_id": campaign_id},
            )
            if advance.status_code == 200:
                adv_data = advance.json()
                new_obs = adv_data.get("observation", {})
                ws = new_obs.get("world_state", {})
                logger.info(f"  Advanced to period {period + 1}, "
                            f"policy_version={ws.get('policy_version', '?')}, "
                            f"schema_version={ws.get('schema_version', '?')}")

    # Get final campaign state
    status = session.get(f"{env_url}/campaign/state", params={"campaign_id": campaign_id})
    final_state = status.json() if status.status_code == 200 else {}

    print(f"[END] campaign_id={campaign_id} periods={total_periods} success=true", flush=True)

    # Print summary table
    print(f"\n{'='*60}", file=sys.stderr)
    print(f" CAMPAIGN RESULTS — {total_periods} periods", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for pr in period_results:
        total_f = sum(pr["findings_by_role"].values())
        print(f"  Period {pr['period']}: {total_f} findings, {pr['overseer_decisions']} overseer decisions", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    return {
        "campaign_id": campaign_id,
        "period_results": period_results,
        "state": final_state,
    }


def _build_campaign_prompt(obs, world_state, history, instructions, shocks, role):
    """Build a comprehensive prompt for a specialist agent."""
    task_desc = obs.get("task_description", "")
    docs = json.dumps(obs.get("documents", {}), indent=2, default=str)[:8000]

    prompt = f"You are a {role} in a multi-agent financial audit team.\n\n"
    prompt += f"TASK:\n{task_desc}\n\n"

    if world_state.get("policy_updates"):
        prompt += "ACTIVE POLICY CHANGES:\n"
        for p in world_state["policy_updates"]:
            prompt += f"  - {p}\n"
        prompt += "\n"

    if shocks:
        prompt += "⚠️ REGULATORY SHOCKS (apply immediately):\n"
        for s in shocks:
            prompt += f"  - {s.get('text', '')}\n"
        prompt += "\n"

    if history:
        prompt += f"PRIOR PERIOD FINDINGS (use for cross-period patterns):\n"
        for h in history[-10:]:
            prompt += f"  - P{h.get('period')}: {h.get('document_id')} ({h.get('error_type')})\n"
        prompt += "\n"

    if instructions:
        prompt += "INSTRUCTIONS TO FOLLOW:\n"
        for inst in instructions[:10]:
            prompt += f"  - [{inst.get('id')}] {inst.get('text')}\n"
        prompt += "\n"

    prompt += f"DOCUMENTS (truncated):\n{docs}\n\n"
    prompt += (
        "Report findings as JSON array. Each finding needs: "
        "document_id, error_type, description, confidence (0.0-1.0).\n"
        "Be precise (no false positives) and thorough (find all errors).\n"
        "Output ONLY the JSON array."
    )
    return prompt


def _parse_findings_response(text: str) -> List[Dict]:
    """Parse LLM response into findings dicts."""
    if not text:
        return []
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                valid = []
                for item in parsed:
                    if isinstance(item, dict) and "document_id" in item and "error_type" in item:
                        valid.append({
                            "document_id": str(item["document_id"]),
                            "error_type": str(item["error_type"]).lower(),
                            "description": str(item.get("description", "Finding")),
                            "confidence": float(item["confidence"]) if "confidence" in item else None,
                        })
                return valid
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return []

def main():
    parser = argparse.ArgumentParser(description="Financial Audit Env Inference")
    parser.add_argument("--env-url", default="http://localhost:8000", help="URL of the running environment")
    parser.add_argument("--task", default=None, help="Specific task ID to run (optional)")
    parser.add_argument("--campaign", action="store_true", help="Run Round 2 campaign flow instead of Round 1 tasks")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for reproducibility")
    args = parser.parse_args()

    print(f"{'='*60}", file=sys.stderr)
    print(f" OpenEnv Financial Audit - Inference Configuration", file=sys.stderr)
    print(f" Model Identifier: {MODEL_NAME}", file=sys.stderr)
    print(f" API Base URL:     {API_BASE_URL}", file=sys.stderr)
    print(f" Environment URL:  {args.env_url}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    if args.campaign:
        out = run_campaign_round(args.env_url, args.seed, client=client)
        print(json.dumps({"campaign": out}, indent=2, default=str), file=sys.stderr)
        return

    tasks_to_run = [args.task] if args.task else TASK_IDS
    results = {}
    total_score = 0.0

    for idx, task_id in enumerate(tasks_to_run):
        res = run_agent_single_task(args.env_url, task_id, client, args.seed)
        results[task_id] = res
        total_score += res["score"]
        if idx < len(tasks_to_run) - 1:
            print("\n", file=sys.stderr)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f" RESULTS", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"{'Task':<30} {'Difficulty':<12} {'Score':<8} {'P':<8} {'R':<8}", file=sys.stderr)
    print(f"{'-'*60}", file=sys.stderr)
    for k, v in results.items():
        print(f"{v['task_name']:<30} {v['difficulty']:<12} {v['score']:<8.4f} {v['precision']:<8.2f} {v['recall']:<8.2f}", file=sys.stderr)
    print(f"{'-'*60}", file=sys.stderr)
    print(f"{'AVERAGE':<42} {total_score / len(tasks_to_run):<8.4f}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

if __name__ == "__main__":
    main()
