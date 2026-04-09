#!/usr/bin/env python3
# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — Baseline inference script.
#
# Uses Meta's Llama model via the free HuggingFace Inference API
# (OpenAI-compatible endpoint) to audit financial data.
#
# This satisfies the contest requirement of:
# 1. Using an OpenAI-compatible API client
# 2. Producing reproducible baseline scores for all 3 tasks
# 3. Costing $0 (HF Inference API is free with HF_TOKEN)
#
# Usage:
#   export HF_TOKEN=your_huggingface_token
#   python baseline.py --base-url http://localhost:8000
#
# Or against a deployed HF Space:
#   python baseline.py --base-url https://your-space.hf.space

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("baseline")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model to use via HuggingFace Inference API (free tier)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# HF Inference API base URL (OpenAI-compatible)
HF_INFERENCE_URL = "https://router.huggingface.co/v1/"

# Tasks to run baseline on
TASK_IDS = ["expense_audit", "invoice_match", "gst_reconciliation"]

# Seed for reproducibility
SEED = 42

# Optional directory for raw model outputs (useful for debugging parser issues).
BASELINE_DEBUG_DIR = os.environ.get("BASELINE_DEBUG_DIR", "")


# ---------------------------------------------------------------------------
# Prompt templates — tailored per task for best Llama performance
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert financial auditor AI. You review financial documents and identify errors, violations, and discrepancies with extreme precision.

You MUST respond ONLY with a valid JSON array of findings. Each finding must have:
- "document_id": the ID of the document with the error
- "error_type": one of the allowed error types listed below
- "description": a clear explanation of the error
- "suggested_fix": what should be done to fix it

Do NOT include any text before or after the JSON array. Do NOT use markdown code blocks. Just output the raw JSON array."""


def build_task_prompt(task_description: str, documents: Dict[str, Any], error_types: List[str]) -> str:
    """
    Build the user prompt for the LLM with task context and data.

    Includes the task description, allowed error types, and a formatted
    view of the financial documents. Truncates large datasets to stay
    within context limits.
    """
    prompt_parts = [
        "# TASK",
        task_description,
        "",
        "# ALLOWED ERROR TYPES",
        json.dumps(error_types, indent=2),
        "",
        "# DOCUMENTS TO AUDIT",
    ]

    # Format each document type
    for doc_type, doc_data in documents.items():
        prompt_parts.append(f"\n## {doc_type.upper()}")
        if isinstance(doc_data, list):
            # Tabular data — show all rows
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


def build_compact_task_prompt(task_description: str, documents: Dict[str, Any], error_types: List[str]) -> str:
    """Build a compact prompt variant for context-limited providers."""
    return (
        "You are an expert financial auditor. Return ONLY a JSON array of findings.\n"
        "Each finding keys: document_id, error_type, description, suggested_fix.\n"
        f"Task: {task_description}\n"
        f"Allowed error types: {json.dumps(error_types, separators=(',', ':'), sort_keys=True)}\n"
        f"Documents: {json.dumps(documents, separators=(',', ':'), sort_keys=True)}"
    )


def _compact_documents(documents: Dict[str, Any], ratio: float = 0.65, min_items: int = 8) -> Dict[str, Any]:
    """Reduce list-heavy document payloads to fit strict context windows."""
    reduced: Dict[str, Any] = {}
    for key, value in documents.items():
        if isinstance(value, list):
            keep = max(min_items, int(len(value) * ratio))
            reduced[key] = value[:keep]
        elif isinstance(value, dict):
            nested: Dict[str, Any] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                    keep = max(min_items, int(len(sub_value) * ratio))
                    nested[sub_key] = sub_value[:keep]
                else:
                    nested[sub_key] = sub_value
            reduced[key] = nested
        else:
            reduced[key] = value
    return reduced


def _save_raw_response(task_id: str, response_text: str, attempt: int) -> None:
    """Persist raw model output for post-mortem debugging when enabled."""
    if not BASELINE_DEBUG_DIR:
        return
    debug_dir = Path(BASELINE_DEBUG_DIR)
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe_task = re.sub(r"[^a-zA-Z0-9_-]", "_", task_id)
    out_file = debug_dir / f"{safe_task}_attempt_{attempt}.txt"
    out_file.write_text(response_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM inference via HuggingFace Inference API
# ---------------------------------------------------------------------------

def call_llama(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    hf_token: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    """
    Call Meta's Llama model via HuggingFace Inference API.

    Uses the OpenAI-compatible chat completions endpoint.

    Args:
        prompt: User message content
        system_prompt: System message for role/format instructions
        hf_token: HuggingFace API token
        max_retries: Number of retries on failure

    Returns:
        Model's response text
    """
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set. Get one at https://huggingface.co/settings/tokens")

    # Use the openai client library for compatibility
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=HF_INFERENCE_URL,
            api_key=hf_token,
        )

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4096,
                    temperature=0.1,  # Low temperature for consistent outputs
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    except ImportError:
        # Fallback to raw requests if openai package not installed
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{HF_INFERENCE_URL}chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    return ""


def parse_llm_findings(response_text: str) -> List[Dict[str, str]]:
    """
    Parse the LLM's JSON response into a list of finding dicts.

    Handles common issues:
    - Response wrapped in markdown code blocks
    - Extra text before/after JSON
    - Malformed JSON (best effort)
    """
    text = response_text.strip()

    # Handle fenced code blocks while preserving only the first fenced payload.
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Primary path: parse the broadest JSON array found in the output.
    start_idx = text.find("[")
    end_idx = text.rfind("]")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        chunk = text[start_idx:end_idx + 1]
        try:
            parsed = json.loads(chunk)
            if isinstance(parsed, list):
                return [f for f in parsed if isinstance(f, dict)]
        except json.JSONDecodeError:
            pass

    # Fallback path: recover individual object literals when the array is malformed.
    recovered: List[Dict[str, str]] = []
    for obj in re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL):
        if '"document_id"' not in obj or '"error_type"' not in obj:
            continue
        try:
            value = json.loads(obj.replace("\r", " ").replace("\n", " "))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            recovered.append(value)

    if recovered:
        logger.info("Recovered %d findings from malformed JSON output", len(recovered))
    else:
        logger.warning("Failed to parse LLM response as findings JSON")
    return recovered


def _generate_findings_with_fallback(
    task_id: str,
    task_description: str,
    documents: Dict[str, Any],
    error_types: List[str],
    hf_token: str,
    max_attempts: int = 4,
) -> Tuple[List[Dict[str, str]], int, int]:
    """
    Generate findings with adaptive prompt fallback for context-constrained providers.

    Returns:
        findings, final_prompt_length, attempts_used
    """
    docs_variant = documents
    last_prompt_len = 0

    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            prompt = build_task_prompt(task_description, docs_variant, error_types)
        else:
            prompt = build_compact_task_prompt(task_description, docs_variant, error_types)
        last_prompt_len = len(prompt)

        try:
            response_text = call_llama(prompt, hf_token=hf_token)
            _save_raw_response(task_id, response_text, attempt)
            findings = parse_llm_findings(response_text)
            if findings:
                return findings, last_prompt_len, attempt

            # Empty parse: shrink payload and retry with compact prompt.
            if attempt < max_attempts:
                docs_variant = _compact_documents(docs_variant)
                continue
            return [], last_prompt_len, attempt
        except Exception as exc:
            message = str(exc).lower()
            context_error = "context_length_exceeded" in message or "reduce the length" in message
            if context_error and attempt < max_attempts:
                logger.warning("[%s] Context limit reached, retrying with compact prompt", task_id)
                docs_variant = _compact_documents(docs_variant)
                continue
            raise

    return [], last_prompt_len, max_attempts


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------

def run_baseline_single_task(
    env_url: str,
    task_id: str,
    hf_token: str,
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    Run the baseline agent on a single task.

    Steps:
    1. Reset environment with the task
    2. Build prompt from observation
    3. Call Llama for analysis
    4. Parse response into findings
    5. Submit findings and get grader score

    Args:
        env_url: Base URL of the environment server
        task_id: Task to run
        hf_token: HuggingFace API token
        seed: Random seed for reproducibility

    Returns:
        Dict with task_id, score, precision, recall, and details
    """
    session = requests.Session()

    # Step 1: Reset
    logger.info(f"[{task_id}] Resetting environment...")
    reset_resp = session.post(
        f"{env_url}/reset",
        json={"task_id": task_id, "seed": seed},
    )
    reset_resp.raise_for_status()
    reset_data = reset_resp.json()
    obs = reset_data.get("observation", reset_data)

    # Step 2: Build prompt
    logger.info(f"[{task_id}] Building prompt from observation...")
    task_desc = obs["task_description"]
    documents = obs["documents"]

    # Get task error types from /tasks endpoint
    tasks_resp = session.get(f"{env_url}/tasks")
    tasks_resp.raise_for_status()
    tasks_data = tasks_resp.json()
    task_info = next(t for t in tasks_data["tasks"] if t["id"] == task_id)
    error_types = task_info["error_types"]

    # Step 3: Call Llama + parse findings with context fallback
    logger.info(f"[{task_id}] Calling {MODEL}...")
    start_time = time.time()
    findings, prompt_length, attempts_used = _generate_findings_with_fallback(
        task_id=task_id,
        task_description=task_desc,
        documents=documents,
        error_types=error_types,
        hf_token=hf_token,
    )
    elapsed = time.time() - start_time
    logger.info(f"[{task_id}] Prompt length: {prompt_length} chars (attempt {attempts_used})")
    logger.info(f"[{task_id}] LLM responded in {elapsed:.1f}s")

    # Step 4: Parsed findings
    logger.info(f"[{task_id}] Parsed {len(findings)} findings from LLM")

    # Step 5: Submit findings
    action = {
        "findings": [
            {
                "document_id": f.get("document_id", ""),
                "error_type": f.get("error_type", ""),
                "description": f.get("description", "No description"),
                "suggested_fix": f.get("suggested_fix", None),
            }
            for f in findings
        ],
        "submit_final": True,
    }

    step_resp = session.post(
        f"{env_url}/step",
        json={"action": action},
    )
    step_resp.raise_for_status()
    step_data = step_resp.json()

    # Get grader score
    grader_resp = session.get(f"{env_url}/grader")
    grader_resp.raise_for_status()
    grader_data = grader_resp.json()

    result = {
        "task_id": task_id,
        "task_name": task_info["name"],
        "difficulty": task_info["difficulty"],
        "score": grader_data.get("score", 0.01),
        "precision": grader_data.get("precision", 0.01),
        "recall": grader_data.get("recall", 0.01),
        "true_positives": grader_data.get("true_positives", 0),
        "false_positives": grader_data.get("false_positives", 0),
        "total_errors": grader_data.get("total_errors", 0),
        "findings_submitted": len(findings),
        "inference_time_s": round(elapsed, 2),
        "prompt_length": prompt_length,
        "llm_attempts": attempts_used,
        "model": MODEL,
    }

    logger.info(
        f"[{task_id}] Score: {result['score']:.4f} "
        f"(P={result['precision']:.2f}, R={result['recall']:.2f})"
    )
    return result


def run_baseline_all_tasks(
    env_url: Optional[str] = None,
    env: Any = None,
    hf_token: Optional[str] = None,
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    Run the baseline agent on all 3 tasks and return scores.

    Can work with either a remote env_url or a local env instance.

    Args:
        env_url: Base URL of the environment server
        env: Local environment instance (alternative to env_url)
        hf_token: HuggingFace API token
        seed: Random seed

    Returns:
        Dict with task scores and overall summary
    """
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN", "")

    results = {}
    total_score = 0.01

    for task_id in TASK_IDS:
        if env_url:
            result = run_baseline_single_task(env_url, task_id, hf_token, seed)
        else:
            # Use local environment directly (for /baseline endpoint)
            result = _run_baseline_local(env, task_id, hf_token, seed)
        results[task_id] = result
        total_score += result["score"]

    avg_score = total_score / len(TASK_IDS)
    clamped_avg = 0.01 if avg_score <= 0.01 else (0.99 if avg_score >= 0.99 else avg_score)

    return {
        "tasks": results,
        "average_score": round(clamped_avg, 2),
        "model": MODEL,
        "seed": seed,
    }


def _run_baseline_local(env: Any, task_id: str, hf_token: str, seed: int) -> Dict[str, Any]:
    """Run baseline against a local environment instance."""
    from financial_audit_env.models import AuditAction, Finding
    from financial_audit_env.server.tasks import get_task

    task = get_task(task_id)
    obs = env.reset(task_id=task_id, seed=seed)

    start_time = time.time()
    raw_findings, prompt_length, attempts_used = _generate_findings_with_fallback(
        task_id=task_id,
        task_description=obs.task_description,
        documents=obs.documents,
        error_types=task["error_types"],
        hf_token=hf_token,
    )
    elapsed = time.time() - start_time

    findings = [
        Finding(
            document_id=f.get("document_id", "UNKNOWN"),
            error_type=f.get("error_type", "unknown"),
            description=f.get("description", "No description"),
            suggested_fix=f.get("suggested_fix"),
        )
        for f in raw_findings
    ]

    action = AuditAction(findings=findings, submit_final=True)
    result_obs = env.step(action)

    grader = env.last_grader_result or {}

    return {
        "task_id": task_id,
        "task_name": task["name"],
        "difficulty": task["difficulty"],
        "score": grader.get("score", 0.01),
        "precision": grader.get("precision", 0.01),
        "recall": grader.get("recall", 0.01),
        "true_positives": grader.get("true_positives", 0),
        "false_positives": grader.get("false_positives", 0),
        "total_errors": grader.get("total_errors", 0),
        "findings_submitted": len(findings),
        "inference_time_s": round(elapsed, 2),
        "prompt_length": prompt_length,
        "llm_attempts": attempts_used,
        "model": MODEL,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """
    Run the baseline inference script from command line.

    Usage:
        python baseline.py --base-url http://localhost:8000
    """
    parser = argparse.ArgumentParser(
        description="Run baseline agent on the Financial Audit Environment"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the environment server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Run a single task only (expense_audit, invoice_match, gst_reconciliation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducibility (default: {SEED})",
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.")
        print("Get a free token at: https://huggingface.co/settings/tokens")
        print("Then: export HF_TOKEN=your_token_here")
        sys.exit(1)

    print(f"{'='*60}")
    print(f" Financial Audit Environment — Baseline Agent")
    print(f" Model: {MODEL}")
    print(f" Server: {args.base_url}")
    print(f" Seed: {args.seed}")
    print(f"{'='*60}\n")

    if args.task:
        result = run_baseline_single_task(
            args.base_url, args.task, hf_token, args.seed
        )
        results = {args.task: result}
        avg_score = result["score"]
    else:
        all_results = run_baseline_all_tasks(
            env_url=args.base_url, hf_token=hf_token, seed=args.seed
        )
        results = all_results["tasks"]
        avg_score = all_results["average_score"]

    # Print results table
    print(f"\n{'='*60}")
    print(f" RESULTS")
    print(f"{'='*60}")
    print(f"{'Task':<30} {'Difficulty':<12} {'Score':<8} {'P':<8} {'R':<8}")
    print(f"{'-'*60}")
    for task_id, result in results.items():
        print(
            f"{result['task_name']:<30} "
            f"{result['difficulty']:<12} "
            f"{result['score']:<8.4f} "
            f"{result['precision']:<8.2f} "
            f"{result['recall']:<8.2f}"
        )
    print(f"{'-'*60}")
    print(f"{'AVERAGE':<42} {avg_score:<8.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
