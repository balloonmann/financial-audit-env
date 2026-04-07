# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — Deterministic graders.
#
# Each grader computes scores by comparing the agent's submitted findings
# against the ground truth errors.
#
# Scoring methods:
#   1. F1 score (0.0–1.0) — primary metric, unchanged for compatibility
#   2. Weighted F1 — errors weighted by severity (critical errors worth more)
#   3. Partial credit — right document but wrong error_type gets 0.25 credit
#   4. Risk score — monetary value of caught vs missed errors
#   5. Confusion matrix — which error_types the agent is best/worst at
#
# A finding "matches" a ground truth error if BOTH:
#   1. document_id matches (case-insensitive, stripped)
#   2. error_type matches (case-insensitive, stripped)

from typing import Any, Dict, List, Optional

from .data_generator import ERROR_MONETARY_VALUES, ERROR_SEVERITY_WEIGHTS


# ---------------------------------------------------------------------------
# Phase-2 validator requires every task score to be strictly in (0, 1).
# We enforce: final_score = clamp(round(raw_score, N))
# ---------------------------------------------------------------------------
_SCORE_EPSILON = 0.01

def _clamp_score(score: float) -> float:
    """Clamp a score to be strictly within (0, 1) — never 0.0 or 1.0."""
    if score <= 0.0:
        return _SCORE_EPSILON
    elif score >= 1.0:
        return 1.0 - _SCORE_EPSILON
    return score

def strict_round_clamp(raw_score: float, n_digits: int = 4) -> float:
    """Safely round then clamp to guarantee the result is strictly in (0, 1)."""
    return _clamp_score(round(raw_score, n_digits))


def compute_f1_score(
    findings: List[Dict[str, Any]],
    ground_truth: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Compute F1 score from agent findings vs ground truth.
    Includes weighted F1, partial credit, confusion matrix, and risk scoring.

    Returns:
        Dict with keys:
        - score: float (0, 1) exclusive (unweighted F1 — primary metric)
        - weighted_score: float (0, 1) exclusive (severity-weighted F1)
        - partial_credit_score: float (0, 1) exclusive (with partial credit)
        - precision/recall: standard metrics, clamped to (0, 1) exclusive
        - true_positives/false_positives/false_negatives: counts
        - total_findings/total_errors: counts
        - matched_errors/missed_errors/false_positive_list: details
        - confusion_matrix: per-error_type breakdown
        - risk_score: monetary risk assessment
    """
    # Normalize ground truth into a set of (doc_id, error_type) tuples
    gt_set = set()
    gt_by_doc: Dict[str, set] = {}  # doc_id → set of error_types
    gt_list = []
    for gt in ground_truth:
        doc_id = gt["document_id"].strip().upper()
        error_type = gt["error_type"].strip().lower()
        key = (doc_id, error_type)
        gt_set.add(key)
        gt_list.append(key)
        if doc_id not in gt_by_doc:
            gt_by_doc[doc_id] = set()
        gt_by_doc[doc_id].add(error_type)

    # Normalize findings and check matches
    matched = set()
    partial_matches = []  # Right doc, wrong error_type
    false_positive_list = []

    for finding in findings:
        doc_id = finding.get("document_id", "").strip().upper()
        error_type = finding.get("error_type", "").strip().lower()
        key = (doc_id, error_type)

        if key in gt_set and key not in matched:
            matched.add(key)
        elif doc_id in gt_by_doc and error_type not in gt_by_doc[doc_id]:
            # Right document, wrong error type → partial credit
            partial_matches.append({
                "document_id": finding.get("document_id", ""),
                "error_type": finding.get("error_type", ""),
                "expected_types": list(gt_by_doc[doc_id]),
            })
        else:
            false_positive_list.append({
                "document_id": finding.get("document_id", ""),
                "error_type": finding.get("error_type", ""),
            })

    # --- Standard F1 ---
    true_positives = len(matched)
    false_positives = len(false_positive_list) + len(partial_matches)
    false_negatives = len(gt_set) - true_positives
    total_findings = len(findings)
    total_errors = len(gt_set)

    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0

    if total_errors > 0:
        recall = true_positives / total_errors
    else:
        recall = 1

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    # --- Weighted F1 (severity-weighted) ---
    weighted_tp = sum(
        ERROR_SEVERITY_WEIGHTS.get(et, 1) for (_, et) in matched
    )
    weighted_total = sum(
        ERROR_SEVERITY_WEIGHTS.get(gt[1], 1) for gt in gt_set
    )
    weighted_fp = sum(
        ERROR_SEVERITY_WEIGHTS.get(
            fp.get("error_type", "").strip().lower(), 1
        )
        for fp in false_positive_list
    )

    if weighted_tp + weighted_fp > 0:
        w_precision = weighted_tp / (weighted_tp + weighted_fp)
    else:
        w_precision = 0

    if weighted_total > 0:
        w_recall = weighted_tp / weighted_total
    else:
        w_recall = 1

    if w_precision + w_recall > 0:
        weighted_f1 = 2 * (w_precision * w_recall) / (w_precision + w_recall)
    else:
        weighted_f1 = 0

    # --- Partial Credit Score ---
    partial_credit_value = len(partial_matches) * 0.25
    effective_tp = true_positives + partial_credit_value
    effective_fp = len(false_positive_list) + len(partial_matches) * 0.75

    if effective_tp + effective_fp > 0:
        pc_precision = effective_tp / (effective_tp + effective_fp)
    else:
        pc_precision = 0    

    if total_errors > 0:
        pc_recall = min(effective_tp / total_errors, 1.0)
    else:
        pc_recall = 1

    if pc_precision + pc_recall > 0:
        partial_credit_f1 = 2 * (pc_precision * pc_recall) / (pc_precision + pc_recall)
    else:
        partial_credit_f1 = 0

    # --- Confusion Matrix ---
    # Track per-error_type: found, missed, false_flagged
    all_error_types = set()
    for _, et in gt_set:
        all_error_types.add(et)
    for f in findings:
        all_error_types.add(f.get("error_type", "").strip().lower())

    confusion_matrix: Dict[str, Dict[str, int]] = {}
    for et in sorted(all_error_types):
        gt_count = sum(1 for _, etype in gt_set if etype == et)
        found_count = sum(1 for _, etype in matched if etype == et)
        false_flagged = sum(
            1 for fp in false_positive_list
            if fp.get("error_type", "").strip().lower() == et
        )
        confusion_matrix[et] = {
            "ground_truth": gt_count,
            "correctly_found": found_count,
            "missed": gt_count - found_count,
            "false_positives": false_flagged,
            "severity_weight": ERROR_SEVERITY_WEIGHTS.get(et, 1),
        }

    # --- Risk Score (monetary) ---
    caught_value = sum(
        ERROR_MONETARY_VALUES.get(et, 0) for (_, et) in matched
    )
    missed_value = sum(
        ERROR_MONETARY_VALUES.get(et, 0)
        for (_, et) in gt_set
        if (_, et) not in matched
    )
    total_risk_value = caught_value + missed_value
    risk_mitigation_pct = (caught_value / total_risk_value * 100) if total_risk_value > 0 else 0.0

    # --- Missed errors list ---
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
        # All numeric scores clamped to (0, 1) exclusive — Phase-2 validator requirement
        "score": strict_round_clamp(f1, 4),
        "precision": strict_round_clamp(precision, 4),
        "recall": strict_round_clamp(recall, 4),
        "weighted_score": strict_round_clamp(weighted_f1, 4),
        "partial_credit_score": strict_round_clamp(partial_credit_f1, 4),
        # Counts
        "true_positives": true_positives,
        "false_positives": len(false_positive_list),
        "false_negatives": false_negatives,
        "partial_matches": len(partial_matches),
        "total_findings": total_findings,
        "total_errors": total_errors,
        # Details
        "matched_errors": [
            {"document_id": d, "error_type": e} for d, e in matched
        ],
        "missed_errors": missed_errors,
        "false_positive_list": false_positive_list,
        "partial_match_list": partial_matches,
        # Confusion matrix
        "confusion_matrix": confusion_matrix,
        # Risk scoring
        "risk_score": {
            "caught_value": caught_value,
            "missed_value": missed_value,
            "total_risk_value": total_risk_value,
            "risk_mitigation_pct": round(risk_mitigation_pct, 1),
        },
    }

# ---------------------------------------------------------------------------
# GRADING & REWARD DESIGN RATIONALE
# ---------------------------------------------------------------------------
#
# WHY F1 AS THE PRIMARY METRIC
#
# Precision and recall pull in opposite directions for an auditor. Flag
# everything and recall hits 1.0 while precision craters. Flag nothing and
# precision is perfect (vacuously) while recall is zero. F1 forces a balance.
# It's also standard in information retrieval, which auditing resembles: the
# agent is retrieving a specific set of errors from a document collection.
#
# WHY SEVERITY WEIGHTING
#
# Not all errors cost the same. A ₹200 rounding difference and a fabricated
# vendor invoice both show up as one row in the ground truth, but missing the
# fraud pattern is orders of magnitude more damaging than missing the rounding
# error. ERROR_SEVERITY_WEIGHTS from data_generator.py encode this: critical
# errors (fraud, duplicate invoices) score higher in weighted F1, so an agent
# that catches high-severity issues ranks above one that only catches the easy
# ones — even if their raw F1 scores are identical.
#
# WHY PARTIAL CREDIT (0.25)
#
# A finding on the right document with the wrong error_type is not nothing.
# The agent located the problem area — it just mis-labelled the root cause.
# 0.25 credit (one quarter of a full match) reflects this: meaningful signal,
# but a clear penalty for imprecision. The 0.75 remaining weight goes to
# effective_fp in the partial-credit precision calculation, so partial matches
# still hurt — they just don't hurt as much as a completely wrong document.
#
# WHY THE REWARD VALUES ARE WHAT THEY ARE
#
#   +0.15 per true positive (severity-weighted)
#   The base TP reward is the anchor. At max severity weight ~2.0, a critical
#   find pays +0.30. At default weight 1.0, a routine find pays +0.15. This
#   keeps the expected return per correct finding positive enough to be worth
#   the effort, while remaining bounded so a single lucky guess doesn't dominate
#   the episode score.
#
#   +0.04 per partial match
#   Roughly 27% of the TP reward. Enough to give the agent a gradient signal
#   toward the right document, not enough to make "right doc, wrong label" a
#   viable strategy.
#
#   -0.05 per false positive
#   This is intentionally softer than the TP reward. The asymmetry is
#   deliberate: in auditing, missing a real error (false negative) is generally
#   more costly than a false alarm. A -0.05 FP penalty means the break-even
#   point for guessing is a ~33% hit rate per guess (0.15 × 0.33 ≈ 0.05).
#   Below that hit rate, guessing costs more than it gains. Red herrings in the
#   data push the realistic hit rate for undiscriminating agents well below 33%,
#   making "flag everything" a losing strategy without requiring a punitive FP
#   penalty that would make cautious agents too conservative.
#
#   -0.02 step penalty + -0.005 × step_number decay
#   Discourages agents from submitting redundant steps or spreading findings
#   across many small batches. The decay makes early, confident submissions
#   worth more than late, hedging ones — which mirrors real audit practice where
#   efficiency matters.
#
#   +0.30 recall bonus (recall >= 0.8) on final step
#   A substantial bonus for catching most of what's there. 0.80 recall was
#   chosen as the threshold because it's achievable with competent reasoning but
#   not trivially so — the Llama 3.1 8B baseline lands around 0.33–0.68 recall
#   across tasks. The +0.30 value is large enough to be the deciding factor in
#   close episodes, creating a real incentive to push coverage.
#
#   +0.10 precision bonus (precision >= 0.9) on final step
#   Smaller than the recall bonus because precision is easier to game (just
#   submit fewer findings). The bonus rewards agents that are both comprehensive
#   AND accurate, but doesn't overweight precision at recall's expense.
#
#   -0.20 recall penalty (recall < 0.3) on final step
#   A floor penalty for agents that barely try. Without this, an agent could
#   accumulate small per-step gains while submitting almost nothing. 0.30 recall
#   is the minimum bar for a submission to be considered a genuine attempt.
#
# WHY NO LLM-AS-JUDGE
#
# Every score here is computed from exact (document_id, error_type) matches.
# Same findings always produce the same score, regardless of when or where the
# environment runs. This matters for RL training: a stochastic reward signal
# adds noise that makes it harder for the agent to learn which behaviors
# actually improve performance.
# ---------------------------------------------------------------------------
def compute_step_reward(
    new_findings: List[Dict[str, Any]],
    all_findings_so_far: List[Dict[str, Any]],
    ground_truth: List[Dict[str, str]],
    step_number: int,
    is_final: bool,
    max_steps: int = 10,
) -> float:
    """
    Compute reward for a single step with decay for later steps.

    Reward components:
        +0.15  per NEW true positive in this step
        +0.04  per partial match (right doc, wrong error_type)
        -0.05  per false positive in this step
        -0.02  step penalty (discourages unnecessary steps)
        -0.005 × step_number  decay (earlier findings worth more)
        +0.30  bonus if final submission and recall ≥ 0.8
        +0.10  bonus if final submission and precision ≥ 0.9
        -0.20  penalty if final submission and recall < 0.3
    """
    reward = 0

    # Step penalty + decay
    reward -= 0.02
    reward -= 0.005 * step_number  # Later steps worth slightly less

    # Build ground truth set
    gt_set = set()
    gt_by_doc: Dict[str, set] = {}
    for gt in ground_truth:
        doc_id = gt["document_id"].strip().upper()
        error_type = gt["error_type"].strip().lower()
        key = (doc_id, error_type)
        gt_set.add(key)
        if doc_id not in gt_by_doc:
            gt_by_doc[doc_id] = set()
        gt_by_doc[doc_id].add(error_type)

    # What was already matched before this step?
    prev_matched = set()
    prev_findings = all_findings_so_far[: len(all_findings_so_far) - len(new_findings)]
    for f in prev_findings:
        key = (f.get("document_id", "").strip().upper(), f.get("error_type", "").strip().lower())
        if key in gt_set:
            prev_matched.add(key)

    # Score new findings
    for finding in new_findings:
        doc_id = finding.get("document_id", "").strip().upper()
        error_type = finding.get("error_type", "").strip().lower()
        key = (doc_id, error_type)

        if key in gt_set and key not in prev_matched:
            # Severity-weighted reward
            weight = ERROR_SEVERITY_WEIGHTS.get(error_type, 1)
            reward += 0.15 * weight  # New true positive
            prev_matched.add(key)
        elif doc_id in gt_by_doc and error_type not in gt_by_doc[doc_id]:
            # Partial match — right document, wrong error type
            reward += 0.04
        else:
            reward -= 0.05  # False positive or duplicate

    # Final submission bonuses/penalties
    # Note: uses raw precision/recall (not clamped) for threshold comparisons
    if is_final:
        result = compute_f1_score(all_findings_so_far, ground_truth)
        if result["recall"] >= 0.8:
            reward += 0.30
        if result["precision"] >= 0.9:
            reward += 0.10
        if result["recall"] < 0.3:
            reward -= 0.20

    return round(reward, 4)