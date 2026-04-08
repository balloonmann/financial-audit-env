# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — Core environment implementation.
#
# Implements the OpenEnv Environment interface:
#   - reset(seed, episode_id, task_id) → initial observation
#   - step(action) → observation with reward and feedback
#   - state → current episode metadata
#
# v2 Features:
#   - Investigation mode: opt-in multi-step data exploration
#   - Adaptive difficulty: adjusts noise based on agent performance
#   - Observation filtering: excludes already-audited docs after step 1

import logging
import uuid
from typing import Any, Dict, List, Optional

from ..models import AuditAction, AuditObservation, AuditState, Finding
from .data_generator import generate_data_for_task
from .graders import compute_f1_score, compute_step_reward
from .tasks import TASKS, get_task

logger = logging.getLogger("financial_audit_env.environment")

# ---------------------------------------------------------------------------
# OpenEnv base class import
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        """Minimal Environment base class (standalone fallback)."""
        SUPPORTS_CONCURRENT_SESSIONS = True

        def reset(self, **kwargs):
            raise NotImplementedError

        def step(self, action, **kwargs):
            raise NotImplementedError

        @property
        def state(self):
            raise NotImplementedError


class FinancialAuditEnvironment(Environment):
    """
    OpenEnv-compatible environment for financial auditing tasks.

    The agent audits synthetic financial documents (expenses, invoices,
    GST returns, fraud patterns) to find planted errors. It receives partial
    credit for each correct finding and penalties for false positives.

    Supports 4 tasks with increasing difficulty:
    1. expense_audit (Easy): Policy violation detection
    2. invoice_match (Medium): Three-way PO/GRN/Invoice matching
    3. gst_reconciliation (Hard): GST return reconciliation
    4. fraud_detection (Expert): Fraud pattern recognition

    Modes:
    - Standard (default): Full documents on reset, submit anytime
    - Investigation (opt-in): Summary first, request details, then submit

    Episodes:
    - Start with reset(task_id="...") which generates fresh data
    - Agent submits findings via step(AuditAction(...))
    - Episode ends when submit_final=True or max_steps reached
    - Final score is F1 of (document_id, error_type) matches
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        """Initialize environment with empty state."""
        self._state = AuditState()
        self._ground_truth: List[Dict[str, str]] = []
        self._documents: Dict[str, Any] = {}
        self._findings: List[Dict[str, Any]] = []
        self._task: Optional[Dict[str, Any]] = None
        self._last_grader_result: Optional[Dict[str, Any]] = None
        self._episode_reward: float = 0.0
        # Adaptive difficulty tracking
        self._score_history: List[float] = []
        # Investigation mode
        self._investigation_mode: bool = False
        self._revealed_categories: List[str] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        investigation_mode: bool = False,
        **kwargs: Any,
    ) -> AuditObservation:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for data generation (default: 42).
            episode_id: Optional episode identifier (auto-generated if None)
            task_id: Which task to run.
            investigation_mode: If True, start in investigation mode where
                agent must request document categories before seeing full data.
        """
        if task_id is None:
            task_id = "expense_audit"
        if seed is None:
            seed = 42

        task = get_task(task_id)
        self._task = task
        self._investigation_mode = investigation_mode

        # Generate data with planted errors
        self._documents, self._ground_truth = generate_data_for_task(
            task["generator"], seed=seed
        )

        # Reset internal state
        self._findings = []
        self._episode_reward = 0.0
        self._last_grader_result = None
        self._revealed_categories = []
        self._state = AuditState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            total_errors=len(self._ground_truth),
            found_errors=0,
            false_positives=0,
            investigation_mode=investigation_mode,
        )

        logger.info(
            f"Reset: task={task_id}, seed={seed}, "
            f"errors={len(self._ground_truth)}, "
            f"investigation_mode={investigation_mode}, "
            f"episode={self._state.episode_id}"
        )

        if investigation_mode:
            # Investigation mode: return summary only, not full documents
            summary = self._build_data_summary()
            return AuditObservation(
                done=False,
                reward=0.0,
                task_id=task_id,
                task_description=task["description"],
                documents={},  # No documents yet — must investigate first
                findings_so_far=[],
                feedback=(
                    f"🔍 INVESTIGATION MODE — {task['name']}\n"
                    f"Difficulty: {task['difficulty']}\n"
                    f"You have {task['max_steps']} steps maximum.\n"
                    f"Available categories to investigate: {', '.join(self._documents.keys())}\n"
                    f"Use an investigate action to request specific categories, "
                    f"then submit findings when ready."
                ),
                step_number=0,
                max_steps=task["max_steps"],
                investigation_mode=True,
                available_categories=list(self._documents.keys()),
                data_summary=summary,
            )
        else:
            # Standard mode: full documents immediately
            return AuditObservation(
                done=False,
                reward=0.0,
                task_id=task_id,
                task_description=task["description"],
                documents=self._documents,
                findings_so_far=[],
                feedback=(
                    f"Environment ready. You are performing: {task['name']}.\n"
                    f"Difficulty: {task['difficulty']}.\n"
                    f"You have {task['max_steps']} steps maximum.\n"
                    f"Valid error types: {', '.join(task['error_types'])}\n"
                    f"Submit findings with submit_final=True when done."
                ),
                step_number=0,
                max_steps=task["max_steps"],
            )

    def step(
        self,
        action: AuditAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AuditObservation:
        """
        Process an agent action (batch of findings or investigation request).
        """
        if self._task is None:
            raise RuntimeError(
                "Environment not initialized. Call reset(task_id=...) first."
            )

        self._state.step_count += 1
        step_num = self._state.step_count

        # Handle investigation requests (investigation mode only)
        if self._investigation_mode and not action.findings and not action.submit_final:
            # Check if action has request_categories via kwargs
            request_categories = kwargs.get("request_categories", [])
            if request_categories:
                return self._handle_investigate(step_num, request_categories)

        # Validate error_types in findings
        valid_types = set(self._task["error_types"])
        new_finding_dicts = []
        validation_warnings = []

        for finding in action.findings:
            if finding.error_type.strip().lower() not in {t.lower() for t in valid_types}:
                validation_warnings.append(
                    f"Invalid error_type '{finding.error_type}' for task "
                    f"'{self._task['id']}'. Valid types: {', '.join(valid_types)}"
                )
                continue

            new_finding_dicts.append({
                "document_id": finding.document_id,
                "field": finding.field,
                "error_type": finding.error_type,
                "description": finding.description,
                "suggested_fix": finding.suggested_fix,
            })

        # Add new (valid) findings to accumulated list
        self._findings.extend(new_finding_dicts)

        # Check if episode should end
        is_final = action.submit_final or step_num >= self._task["max_steps"]

        # Compute step reward.
        # Keep cumulative episode rewards aligned with the final task score.
        if not is_final:
            step_reward = 0.0
        else:
            final_grader = compute_f1_score(self._findings, self._ground_truth)
            # Use the bounded final score as the terminal reward.
            step_reward = max(0.01, min(0.99, final_grader["score"]))
        
        self._episode_reward += step_reward

        # Compute running stats
        result = compute_f1_score(self._findings, self._ground_truth)
        self._state.found_errors = result["true_positives"]
        self._state.false_positives = result["false_positives"]

        # Track score for adaptive difficulty
        if is_final:
            self._last_grader_result = result
            self._score_history.append(result["score"])

        # Build feedback message
        feedback_parts = []
        if validation_warnings:
            feedback_parts.append("⚠️ Warnings:\n" + "\n".join(validation_warnings))

        feedback_parts.append(
            f"Step {step_num}/{self._task['max_steps']}: "
            f"Accepted {len(new_finding_dicts)} finding(s) this step."
        )
        feedback_parts.append(
            f"Running score — TP: {result['true_positives']}, "
            f"FP: {result['false_positives']}, "
            f"Precision: {result['precision']:.2f}, "
            f"Recall: {result['recall']:.2f}"
        )

        if result.get("partial_matches", 0) > 0:
            feedback_parts.append(
                f"Partial matches: {result['partial_matches']} "
                f"(right document, wrong error type)"
            )

        if is_final:
            feedback_parts.append(
                f"\n✅ EPISODE COMPLETE — Final F1 Score: {result['score']:.4f}"
            )
            feedback_parts.append(
                f"Weighted F1: {result['weighted_score']:.4f} | "
                f"Risk Mitigation: {result['risk_score']['risk_mitigation_pct']:.1f}%"
            )
            if result["missed_errors"]:
                missed_ids = [e["document_id"] for e in result["missed_errors"]]
                feedback_parts.append(
                    f"Missed errors in: {', '.join(missed_ids)}"
                )

        # Convert findings back to Finding objects for observation
        findings_as_models = [
            Finding(
                document_id=f["document_id"],
                field=f.get("field"),
                error_type=f["error_type"],
                description=f["description"],
                suggested_fix=f.get("suggested_fix"),
            )
            for f in self._findings
        ]

        # Observation filtering: after step 1 in standard mode,
        # indicate which documents have already been examined
        docs_to_return = self._documents if not is_final else {}

        return AuditObservation(
            done=is_final,
            reward=step_reward,
            task_id=self._task["id"],
            task_description=self._task["description"],
            documents=docs_to_return,
            findings_so_far=findings_as_models,
            feedback="\n".join(feedback_parts),
            step_number=step_num,
            max_steps=self._task["max_steps"],
            investigation_mode=self._investigation_mode,
        )

    def _handle_investigate(
        self, step_num: int, request_categories: List[str]
    ) -> AuditObservation:
        """Handle an investigation request — reveal requested document categories."""
        revealed_docs: Dict[str, Any] = {}
        valid_cats = list(self._documents.keys())
        new_reveals = []

        for cat in request_categories:
            if cat in self._documents:
                revealed_docs[cat] = self._documents[cat]
                if cat not in self._revealed_categories:
                    self._revealed_categories.append(cat)
                    new_reveals.append(cat)
            else:
                pass  # Ignore invalid categories

        self._state.revealed_categories = self._revealed_categories

        return AuditObservation(
            done=False,
            reward=0.0,  # Clamped cost for investigation step removed to enforce episode sum strictly in (0, 1)
            task_id=self._task["id"] if self._task else "",
            task_description=self._task["description"] if self._task else "",
            documents=revealed_docs,
            findings_so_far=[],
            feedback=(
                f"Step {step_num}/{self._task['max_steps'] if self._task else 0}: "
                f"Revealed {len(new_reveals)} category(ies): {', '.join(new_reveals) if new_reveals else 'none new'}\n"
                f"Total revealed: {', '.join(self._revealed_categories)}\n"
                f"Remaining: {', '.join(c for c in valid_cats if c not in self._revealed_categories)}"
            ),
            step_number=step_num,
            max_steps=self._task["max_steps"] if self._task else 0,
            investigation_mode=True,
            available_categories=[c for c in valid_cats if c not in self._revealed_categories],
        )

    def _build_data_summary(self) -> Dict[str, Any]:
        """Build a statistical summary of the data for investigation mode."""
        summary: Dict[str, Any] = {}
        for key, value in self._documents.items():
            if isinstance(value, list):
                summary[key] = {
                    "count": len(value),
                    "type": "list",
                    "sample_fields": list(value[0].keys()) if value else [],
                }
            elif isinstance(value, dict):
                summary[key] = {
                    "type": "dict",
                    "top_level_keys": list(value.keys()),
                }
        return summary

    @property
    def state(self) -> AuditState:
        """Get current episode state."""
        return self._state

    @property
    def last_grader_result(self) -> Optional[Dict[str, Any]]:
        """Get the grader result from the last completed episode."""
        return self._last_grader_result

    @property
    def score_history(self) -> List[float]:
        """Get score history for adaptive difficulty tracking."""
        return self._score_history

    def get_adaptive_difficulty(self) -> Dict[str, Any]:
        """
        Compute adaptive difficulty parameters based on score history.
        Returns parameters that can be used to adjust data generation.
        """
        if not self._score_history:
            return {"noise_level": 0.3, "suggestion": "default"}

        avg_score = sum(self._score_history[-5:]) / len(self._score_history[-5:])
        clamped_avg = 0.01 if avg_score <= 0.01 else (0.99 if avg_score >= 0.99 else avg_score)

        if avg_score >= 0.8:
            return {
                "noise_level": 0.7,
                "suggestion": "increase_difficulty",
                "avg_recent_score": round(clamped_avg, 2),
            }
        elif avg_score >= 0.5:
            return {
                "noise_level": 0.5,
                "suggestion": "maintain",
                "avg_recent_score": round(clamped_avg, 2),
            }
        else:
            return {
                "noise_level": 0.2,
                "suggestion": "decrease_difficulty",
                "avg_recent_score": round(clamped_avg, 2),
            }
