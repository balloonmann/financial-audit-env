"""
Reward function for GRPO training.
Parses model output (JSON or free-text) and evaluates via InProcessEvaluator.

Used in the Colab notebook with Unsloth + TRL GRPOTrainer.
"""

import json
import re
from typing import Any, Dict, List

from .evaluator import InProcessEvaluator

_evaluator = InProcessEvaluator()


def parse_findings_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse model-generated findings from free text.

    Supports two formats:
    1. JSON array: [{"document_id": "EXP-010", "error_type": "over_limit", ...}]
    2. Free-text with labeled fields:
       - document_id: EXP-010
         error_type: over_limit
         description: Amount exceeds limit
    """
    if not text or not isinstance(text, str):
        return []

    # Try JSON first — look for array in the text
    try:
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list) and len(parsed) > 0:
                # Validate minimum fields
                valid = []
                for item in parsed:
                    if isinstance(item, dict) and "document_id" in item and "error_type" in item:
                        valid.append({
                            "document_id": str(item["document_id"]).strip(),
                            "error_type": str(item["error_type"]).strip().lower(),
                            "description": str(item.get("description", "Finding")).strip(),
                            "confidence": float(item["confidence"]) if "confidence" in item else None,
                        })
                if valid:
                    return valid
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: regex-based parsing for free-text output
    findings = []
    # Split on double newlines or lines starting with - or *
    blocks = re.split(r"\n\s*\n|\n(?=[-*]\s)", text)

    for block in blocks:
        doc_match = re.search(r"document_id[:\s]+([A-Z0-9][\w\-]*)", block, re.I)
        type_match = re.search(r"error_type[:\s]+([a-z_]+)", block, re.I)
        desc_match = re.search(r"description[:\s]+(.+?)(?:\n|$)", block, re.I)
        conf_match = re.search(r"confidence[:\s]+([\d.]+)", block, re.I)

        if doc_match and type_match:
            finding: Dict[str, Any] = {
                "document_id": doc_match.group(1).strip(),
                "error_type": type_match.group(1).strip().lower(),
                "description": desc_match.group(1).strip() if desc_match else "Finding",
            }
            if conf_match:
                try:
                    finding["confidence"] = float(conf_match.group(1))
                except ValueError:
                    pass
            findings.append(finding)

    return findings


def financial_audit_reward(
    completions: List[str],
    task_id: str = "expense_audit",
    seed: int = 42,
    **kwargs: Any,
) -> List[float]:
    """
    Reward function for GRPOTrainer.

    For each completion (model's text output):
    1. Parse findings from model output
    2. Evaluate against ground truth via InProcessEvaluator (no HTTP)
    3. Return F1-based score as reward

    Args:
        completions: List of model-generated text outputs
        task_id: Audit task to evaluate against
        seed: Random seed for deterministic data

    Returns:
        List of reward floats, one per completion, in [0.01, 0.99]
    """
    rewards = []
    for completion in completions:
        try:
            findings = parse_findings_from_text(completion)
            result = _evaluator.evaluate(task_id, seed, findings)
            reward = result["score"]  # Already clamped [0.01, 0.99]
        except Exception:
            reward = 0.01  # Minimum reward on any error
        rewards.append(reward)
    return rewards


def make_reward(metrics: Dict[str, Any]) -> float:
    """
    Build a scalar reward from pre-computed metrics.
    Used when metrics are already available (not from text parsing).

    Keys: f1_score, ece, agreement, exploit_flag
    """
    f1 = float(metrics.get("f1_score", 0.0))
    ece = float(metrics.get("ece", 1.0))
    agreement = float(metrics.get("agreement", 0.0))
    exploit_flag = float(metrics.get("exploit_flag", 0.0))

    reward = (0.65 * f1) + (0.20 * (1.0 - ece)) + (0.15 * agreement) - (0.50 * exploit_flag)
    return float(round(max(0.01, min(0.99, reward)), 6))
