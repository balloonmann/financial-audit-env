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
# Numerical stability: expose score-like fields inside an open interval.
# We apply: final_score = clamp(round(raw_score, N)).
# ---------------------------------------------------------------------------
_SCORE_EPSILON = 0.01

def _clamp_score(score: float) -> float:
    """Keep a score inside the open interval [0.01, 0.99]."""
    if score <= 0.01:
        return 0.01
    elif score >= 0.99:
        return 0.99
    return score

def strict_round_clamp(raw_score: float, n_digits: int = 2) -> float:
    """Round first, then keep the result inside [0.01, 0.99]."""
    rounded = round(raw_score, n_digits)
    if rounded <= 0.01:
        return 0.01
    elif rounded >= 0.99:
        return 0.99
    return rounded


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
        - precision/recall: standard metrics, bounded to (0, 1)
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
    duplicates_list = []

    for finding in findings:
        doc_id = finding.get("document_id", "").strip().upper()
        error_type = finding.get("error_type", "").strip().lower()
        key = (doc_id, error_type)

        if key in gt_set:
            if key not in matched:
                matched.add(key)
            else:
                # STRONGER DUPLICATE HANDLING:
                # Explicitly track redundant findings separately from false positives.
                duplicates_list.append({
                    "document_id": finding.get("document_id", ""),
                    "error_type": finding.get("error_type", ""),
                })
        elif doc_id in gt_by_doc and error_type not in gt_by_doc[doc_id]:
            # EXPLICIT PARTIAL-CREDIT POLICY:
            # We award partial credit because the agent successfully localized the issue
            # to the correct document, even though the specific error classification was wrong.
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
        precision = 0.01

    if total_errors > 0:
        recall = true_positives / total_errors
    else:
        recall = 0.99

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.01

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
        w_precision = 0.01

    if weighted_total > 0:
        w_recall = weighted_tp / weighted_total
    else:
        w_recall = 0.99

    if w_precision + w_recall > 0:
        weighted_f1 = 2 * (w_precision * w_recall) / (w_precision + w_recall)
    else:
        weighted_f1 = 0.01

    # --- Partial Credit Score ---
    partial_credit_value = len(partial_matches) * 0.25
    effective_tp = true_positives + partial_credit_value
    effective_fp = len(false_positive_list) + len(partial_matches) * 0.75

    if effective_tp + effective_fp > 0:
        pc_precision = effective_tp / (effective_tp + effective_fp)
    else:
        pc_precision = 0.01    

    if total_errors > 0:
        pc_recall = min(effective_tp / total_errors, 0.99)
    else:
        pc_recall = 0.99

    if pc_precision + pc_recall > 0:
        partial_credit_f1 = 2 * (pc_precision * pc_recall) / (pc_precision + pc_recall)
    else:
        partial_credit_f1 = 0.01

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
    risk_mitigation_pct = (caught_value / total_risk_value * 100) if total_risk_value > 0 else 0.01

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
        # Keep public score-like outputs in a bounded open interval.
        "score": strict_round_clamp(f1, 2),
        "precision": strict_round_clamp(precision, 2),
        "recall": strict_round_clamp(recall, 2),
        "weighted_score": strict_round_clamp(weighted_f1, 2),
        "partial_credit_score": strict_round_clamp(partial_credit_f1, 2),
        # Counts
        "true_positives": true_positives,
        "false_positives": len(false_positive_list),
        "false_negatives": false_negatives,
        "weighted_false_negatives": round(weighted_total - weighted_tp, 2),
        "duplicates": len(duplicates_list),
        "partial_matches": len(partial_matches),
        "total_findings": total_findings,
        "total_errors": total_errors,
        # Details
        "matched_errors": [
            {"document_id": d, "error_type": e} for d, e in matched
        ],
        "missed_errors": missed_errors,
        "false_positive_list": false_positive_list,
        "duplicate_list": duplicates_list,
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

        if key in gt_set:
            if key not in prev_matched:
                # Severity-weighted reward
                weight = ERROR_SEVERITY_WEIGHTS.get(error_type, 1)
                reward += 0.15 * weight  # New true positive
                prev_matched.add(key)
            else:
                # STRONGER DUPLICATE HANDLING: explicitly punish redundant assertions
                reward -= 0.10  # Duplicate penalty
        elif doc_id in gt_by_doc and error_type not in gt_by_doc[doc_id]:
            # EXPLICIT PARTIAL-CREDIT POLICY:
            # We compensate the agent for successfully isolating a damaged document (+0.04),
            # providing a gradient signal toward the vicinity of the error.
            reward += 0.04
        else:
            reward -= 0.05  # Standard false positive penalty

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

    return strict_round_clamp(reward, 2)


def compute_ece(
    findings: List[Dict[str, Any]],
    ground_truth: List[Dict[str, str]],
    n_bins: int = 10,
    min_total: int = 10,
    min_bin_count: int = 3,
) -> Dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE) for finding confidence.

    Calibration activates only with sufficient data:
    - at least `min_total` findings with confidence
    - at least 2 bins with >= `min_bin_count` samples
    """
    scored = []
    gt_set = {
        (gt["document_id"].strip().upper(), gt["error_type"].strip().lower())
        for gt in ground_truth
    }

    for f in findings:
        conf = f.get("confidence")
        if conf is None:
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        c = max(0.0, min(1.0, c))
        key = (
            str(f.get("document_id", "")).strip().upper(),
            str(f.get("error_type", "")).strip().lower(),
        )
        label = 1.0 if key in gt_set else 0.0
        scored.append((c, label))

    if len(scored) < min_total:
        return {
            "available": False,
            "reason": "insufficient_total_samples",
            "samples": len(scored),
        }

    bins: List[List[Any]] = [[] for _ in range(n_bins)]
    for conf, label in scored:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, label))

    valid_bins = [b for b in bins if len(b) >= min_bin_count]
    if len(valid_bins) < 2:
        return {
            "available": False,
            "reason": "insufficient_valid_bins",
            "samples": len(scored),
            "valid_bins": len(valid_bins),
        }

    total = float(len(scored))
    ece = 0.0
    brier_sum = 0.0
    sharpness = 0.0
    bin_details: List[Dict[str, Any]] = []

    for i, b in enumerate(bins):
        if not b:
            continue
        conf_mean = sum(x[0] for x in b) / len(b)
        acc_mean = sum(x[1] for x in b) / len(b)
        w = len(b) / total
        ece += w * abs(acc_mean - conf_mean)
        sharpness += w * abs(conf_mean - 0.5)
        brier_sum += sum((x[0] - x[1]) ** 2 for x in b)
        bin_details.append(
            {
                "bin": i,
                "count": len(b),
                "confidence_mean": round(conf_mean, 4),
                "accuracy_mean": round(acc_mean, 4),
                "gap": round(abs(acc_mean - conf_mean), 4),
            }
        )

    brier = brier_sum / total if total > 0 else 1.0
    return {
        "available": True,
        "samples": int(total),
        "valid_bins": len(valid_bins),
        "ece": round(ece, 4),
        "brier": round(brier, 4),
        "sharpness": round(sharpness, 4),
        "bins": bin_details,
    }


def compute_cross_agent_agreement(
    findings_by_agent: Dict[str, List[Dict[str, Any]]],
    ground_truth: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Score cross-agent agreement on identical (document_id, error_type) claims.

    Agreement reward:
    - +0.10 per agreed correct finding (2+ agents)
    - -0.05 per agreed incorrect finding (2+ agents)
    """
    gt_set = {
        (gt["document_id"].strip().upper(), gt["error_type"].strip().lower())
        for gt in ground_truth
    }

    claimants: Dict[Any, List[str]] = {}
    for agent, findings in findings_by_agent.items():
        seen = set()
        for f in findings:
            key = (
                str(f.get("document_id", "")).strip().upper(),
                str(f.get("error_type", "")).strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            claimants.setdefault(key, []).append(agent)

    agreed_correct = 0
    agreed_incorrect = 0
    details: List[Dict[str, Any]] = []
    for key, agents in claimants.items():
        if len(agents) < 2:
            continue
        is_correct = key in gt_set
        if is_correct:
            agreed_correct += 1
        else:
            agreed_incorrect += 1
        details.append(
            {
                "document_id": key[0],
                "error_type": key[1],
                "agents": agents,
                "is_correct": is_correct,
            }
        )

    score = (agreed_correct * 0.10) - (agreed_incorrect * 0.05)
    return {
        "score": round(score, 4),
        "agreed_correct": agreed_correct,
        "agreed_incorrect": agreed_incorrect,
        "agreements": details,
    }


def compute_campaign_score(
    specialist_results: Dict[str, Dict[str, Any]],
    overseer_quality_score: float,
    instruction_compliance_rate: float,
    memory_score: float,
    schema_adaptation_score: float,
    self_improvement_delta: float,
    efficiency_score: float,
    critical_missed: bool,
) -> Dict[str, Any]:
    """
    Compute guarded campaign score with anti-gaming constraints.

    Guards:
    - Any specialist weighted_F1 < 0.20 => multiplier 0.0
    - Any critical miss => multiplier 0.5
    - Bonus components capped to <= 30% of raw total
    """
    def _norm(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    specialist_weighted = [
        _norm(r.get("weighted_score", r.get("score", 0.0)))
        for r in specialist_results.values()
    ]
    avg_specialist = sum(specialist_weighted) / len(specialist_weighted) if specialist_weighted else 0.0

    core = (
        0.35 * avg_specialist
        + 0.25 * _norm(overseer_quality_score)
    )

    bonus = (
        0.10 * _norm(instruction_compliance_rate)
        + 0.10 * _norm(memory_score)
        + 0.08 * _norm(schema_adaptation_score)
        + 0.07 * _norm(self_improvement_delta)
        + 0.05 * _norm(efficiency_score)
    )

    raw = core + bonus
    bonus_cap = raw * 0.30
    capped_bonus = min(bonus, bonus_cap)
    raw_capped = core + capped_bonus

    if specialist_weighted and any(x < 0.20 for x in specialist_weighted):
        quality_multiplier = 0.0
    elif critical_missed:
        quality_multiplier = 0.5
    else:
        quality_multiplier = 1.0

    final_score = strict_round_clamp(raw_capped * quality_multiplier, 2)

    return {
        "score": final_score,
        "raw_score": round(raw, 4),
        "core_component": round(core, 4),
        "bonus_component": round(bonus, 4),
        "bonus_component_capped": round(capped_bonus, 4),
        "quality_multiplier": quality_multiplier,
        "avg_specialist_weighted_f1": round(avg_specialist, 4),
        "specialist_weighted_f1": {
            k: round(_norm(v.get("weighted_score", v.get("score", 0.0))), 4)
            for k, v in specialist_results.items()
        },
        "critical_missed": bool(critical_missed),
    }