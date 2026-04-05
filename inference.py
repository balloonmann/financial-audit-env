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
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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

TASK_IDS = ["expense_audit", "invoice_match", "gst_reconciliation", "fraud_detection"]
SEED = 42
BENCHMARK = "financial_audit_env"
TEMPERATURE = 0.1
MAX_TOKENS = 4096
MAX_STEPS = 5  # Max steps per task (matches our env config)
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# MANDATORY Structured Logging: [START], [STEP], [END]
# ---------------------------------------------------------------------------

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
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}",
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
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(
        f"[END] success={success_str} steps={steps} rewards={rewards_str}",
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
    score = 0.0
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

        step_reward = step_data.get("reward", 0.0) or 0.0
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

        score = grader_data.get("score", 0.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

        result = {
            "task_id": task_id,
            "task_name": task_info["name"],
            "difficulty": task_info["difficulty"],
            "score": score,
            "precision": grader_data.get("precision", 0.0),
            "recall": grader_data.get("recall", 0.0),
        }

        logger.info(f"[{task_id}] Score: {result['score']:.4f} (P={result['precision']:.2f}, R={result['recall']:.2f})")

    except Exception as exc:
        logger.error(f"[{task_id}] Task failed: {exc}")
        result = {
            "task_id": task_id,
            "task_name": task_id,
            "difficulty": "unknown",
            "score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
    finally:
        # Emit [END] log
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return result

def main():
    parser = argparse.ArgumentParser(description="Financial Audit Env Inference")
    parser.add_argument("--env-url", default="http://localhost:8000", help="URL of the running environment")
    parser.add_argument("--task", default=None, help="Specific task ID to run (optional)")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for reproducibility")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f" OpenEnv Financial Audit - Inference Configuration")
    print(f" Model Identifier: {MODEL_NAME}")
    print(f" API Base URL:     {API_BASE_URL}")
    print(f" Environment URL:  {args.env_url}")
    print(f"{'='*60}\n")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    tasks_to_run = [args.task] if args.task else TASK_IDS
    results = {}
    total_score = 0.0

    for idx, task_id in enumerate(tasks_to_run):
        res = run_agent_single_task(args.env_url, task_id, client, args.seed)
        results[task_id] = res
        total_score += res["score"]
        if idx < len(tasks_to_run) - 1:
            print("\n")

    print(f"\n{'='*60}")
    print(f" RESULTS")
    print(f"{'='*60}")
    print(f"{'Task':<30} {'Difficulty':<12} {'Score':<8} {'P':<8} {'R':<8}")
    print(f"{'-'*60}")
    for k, v in results.items():
        print(f"{v['task_name']:<30} {v['difficulty']:<12} {v['score']:<8.4f} {v['precision']:<8.2f} {v['recall']:<8.2f}")
    print(f"{'-'*60}")
    print(f"{'AVERAGE':<42} {total_score / len(tasks_to_run):<8.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
