# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — Deterministic graders.
#
# Each grader computes an F1 score (0.0–1.0) by comparing the agent's
# submitted findings against the ground truth errors.
#
# A finding "matches" a ground truth error if BOTH:
#   1. document_id matches (case-insensitive, stripped)
#   2. error_type matches (case-insensitive, stripped)
#
# The description and suggested_fix fields are informational only —
# they don't affect the score. This keeps grading fully deterministic.

from typing import Any, Dict, List


def compute_f1_score(
    findings: List[Dict[str, Any]],
    ground_truth: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Compute F1 score from agent findings vs ground truth.

    Args:
        findings: List of dicts with at least 'document_id' and 'error_type'
        ground_truth: List of dicts with 'document_id' and 'error_type'

    Returns:
        Dict with keys:
        - score: float in [0.0, 1.0] (the F1 score)
        - precision: float in [0.0, 1.0]
        - recall: float in [0.0, 1.0]
        - true_positives: int
        - false_positives: int
        - false_negatives: int
        - total_findings: int (what agent submitted)
        - total_errors: int (ground truth count)
        - matched_errors: list of matched (document_id, error_type) pairs
        - missed_errors: list of ground truth errors not found
        - false_positive_list: list of findings that didn't match
    """
    # Normalize ground truth into a set of (doc_id, error_type) tuples
    gt_set = set()
    for gt in ground_truth:
        key = (
            gt["document_id"].strip().upper(),
            gt["error_type"].strip().lower(),
        )
        gt_set.add(key)

    # Normalize findings and check matches
    matched = set()
    false_positive_list = []

    for finding in findings:
        doc_id = finding.get("document_id", "").strip().upper()
        error_type = finding.get("error_type", "").strip().lower()
        key = (doc_id, error_type)

        if key in gt_set and key not in matched:
            matched.add(key)
        else:
            false_positive_list.append({
                "document_id": finding.get("document_id", ""),
                "error_type": finding.get("error_type", ""),
            })

    # Compute metrics
    true_positives = len(matched)
    false_positives = len(false_positive_list)
    false_negatives = len(gt_set) - true_positives
    total_findings = len(findings)
    total_errors = len(gt_set)

    # Precision: of what we flagged, how many were correct?
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0

    # Recall: of all errors, how many did we find?
    if total_errors > 0:
        recall = true_positives / total_errors
    else:
        recall = 1.0  # No errors to find = perfect recall

    # F1: harmonic mean of precision and recall
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    # Compute missed errors for detailed feedback
    missed_errors = []
    for gt in ground_truth:
        key = (
            gt["document_id"].strip().upper(),
            gt["error_type"].strip().lower(),
        )
        if key not in matched:
            missed_errors.append({
                "document_id": gt["document_id"],
                "error_type": gt["error_type"],
            })

    return {
        "score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_findings": total_findings,
        "total_errors": total_errors,
        "matched_errors": [
            {"document_id": d, "error_type": e} for d, e in matched
        ],
        "missed_errors": missed_errors,
        "false_positive_list": false_positive_list,
    }


def compute_step_reward(
    new_findings: List[Dict[str, Any]],
    all_findings_so_far: List[Dict[str, Any]],
    ground_truth: List[Dict[str, str]],
    step_number: int,
    is_final: bool,
) -> float:
    """
    Compute reward for a single step. Provides dense signal
    rather than sparse end-of-episode-only reward.

    Reward components:
        +0.15  per NEW true positive in this step
        -0.05  per false positive in this step
        -0.02  step penalty (discourages unnecessary steps)
        +0.30  bonus if final submission and recall ≥ 0.8
        +0.10  bonus if final submission and precision ≥ 0.9
        -0.20  penalty if final submission and recall < 0.3

    Args:
        new_findings: Findings submitted in THIS step only
        all_findings_so_far: ALL findings accumulated across all steps
        ground_truth: Ground truth errors
        step_number: Current step number
        is_final: Whether this is the final submission

    Returns:
        Float reward value
    """
    reward = 0.0

    # Step penalty — small cost per step to discourage stalling
    reward -= 0.02

    # Evaluate just the new findings in this step
    gt_set = set()
    for gt in ground_truth:
        key = (gt["document_id"].strip().upper(), gt["error_type"].strip().lower())
        gt_set.add(key)

    # What was already matched before this step?
    prev_matched = set()
    prev_findings = all_findings_so_far[: len(all_findings_so_far) - len(new_findings)]
    for f in prev_findings:
        key = (f.get("document_id", "").strip().upper(), f.get("error_type", "").strip().lower())
        if key in gt_set:
            prev_matched.add(key)

    # Score new findings
    for finding in new_findings:
        key = (
            finding.get("document_id", "").strip().upper(),
            finding.get("error_type", "").strip().lower(),
        )
        if key in gt_set and key not in prev_matched:
            reward += 0.15  # New true positive
            prev_matched.add(key)
        else:
            reward -= 0.05  # False positive or duplicate

    # Final submission bonuses/penalties
    if is_final:
        result = compute_f1_score(all_findings_so_far, ground_truth)
        if result["recall"] >= 0.8:
            reward += 0.30
        if result["precision"] >= 0.9:
            reward += 0.10
        if result["recall"] < 0.3:
            reward -= 0.20

    return round(reward, 4)
