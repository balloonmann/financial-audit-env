# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — Core environment implementation.
#
# Implements the OpenEnv Environment interface:
#   - reset(seed, episode_id, task_id) → initial observation
#   - step(action) → observation with reward and feedback
#   - state → current episode metadata
#
# The environment generates synthetic financial data with planted errors,
# and scores the agent's ability to find them using F1-based grading.

import logging
import uuid
from typing import Any, Dict, List, Optional

from ..models import AuditAction, AuditObservation, AuditState, Finding
from .data_generator import generate_data_for_task
from .graders import compute_f1_score, compute_step_reward
from .tasks import TASKS, get_task

logger = logging.getLogger("financial_audit_env.environment")

# ---------------------------------------------------------------------------
# Attempt to import OpenEnv base class. Falls back to a compatible
# standalone base if openenv-core is not installed.
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server import Environment
except ImportError:
    # Standalone fallback for development/GitHub usage
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
    GST returns) to find planted errors. It receives partial credit
    for each correct finding and penalties for false positives.

    Supports 3 tasks with increasing difficulty:
    1. expense_audit (Easy): Policy violation detection
    2. invoice_match (Medium): Three-way PO/GRN/Invoice matching
    3. gst_reconciliation (Hard): GST return reconciliation

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

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AuditObservation:
        """
        Reset the environment for a new episode.

        Generates fresh financial data with planted errors for the
        specified task. All internal state is cleared.

        Args:
            seed: Random seed for data generation (default: 42).
                  Same seed → identical data → reproducible scores.
            episode_id: Optional episode identifier (auto-generated if None)
            task_id: Which task to run. One of:
                     'expense_audit', 'invoice_match', 'gst_reconciliation'
                     Defaults to 'expense_audit' if not specified.

        Returns:
            AuditObservation with the financial documents to audit

        Raises:
            ValueError: If task_id is not recognized
        """
        # Default task and seed
        if task_id is None:
            task_id = "expense_audit"
        if seed is None:
            seed = 42

        # Validate and load task
        task = get_task(task_id)
        self._task = task

        # Generate data with planted errors
        self._documents, self._ground_truth = generate_data_for_task(
            task["generator"], seed=seed
        )

        # Reset internal state
        self._findings = []
        self._episode_reward = 0.0
        self._last_grader_result = None
        self._state = AuditState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            total_errors=len(self._ground_truth),
            found_errors=0,
            false_positives=0,
        )

        logger.info(
            f"Reset: task={task_id}, seed={seed}, "
            f"errors={len(self._ground_truth)}, "
            f"episode={self._state.episode_id}"
        )

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
        Process an agent action (batch of findings).

        The agent submits findings which are matched against ground truth.
        Reward is computed per-step with bonuses/penalties on final submission.

        Args:
            action: AuditAction with findings and submit_final flag
            timeout_s: Optional timeout (unused, for interface compatibility)

        Returns:
            AuditObservation with feedback and updated reward

        Raises:
            RuntimeError: If called before reset()
            ValueError: If action contains invalid error_types
        """
        if self._task is None:
            raise RuntimeError(
                "Environment not initialized. Call reset(task_id=...) first."
            )

        # Increment step count
        self._state.step_count += 1
        step_num = self._state.step_count

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
                continue  # Skip invalid findings

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

        # Compute step reward (dense signal)
        step_reward = compute_step_reward(
            new_findings=new_finding_dicts,
            all_findings_so_far=self._findings,
            ground_truth=self._ground_truth,
            step_number=step_num,
            is_final=is_final,
        )
        self._episode_reward += step_reward

        # Compute running stats
        result = compute_f1_score(self._findings, self._ground_truth)
        self._state.found_errors = result["true_positives"]
        self._state.false_positives = result["false_positives"]

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

        if is_final:
            self._last_grader_result = result
            feedback_parts.append(
                f"\n✅ EPISODE COMPLETE — Final F1 Score: {result['score']:.4f}"
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

        return AuditObservation(
            done=is_final,
            reward=step_reward,
            task_id=self._task["id"],
            task_description=self._task["description"],
            documents=self._documents if not is_final else {},  # Clear docs on final
            findings_so_far=findings_as_models,
            feedback="\n".join(feedback_parts),
            step_number=step_num,
            max_steps=self._task["max_steps"],
        )

    @property
    def state(self) -> AuditState:
        """
        Get current episode state.

        Returns:
            AuditState with episode_id, step_count, task_id, and audit stats
        """
        return self._state

    @property
    def last_grader_result(self) -> Optional[Dict[str, Any]]:
        """
        Get the grader result from the last completed episode.

        Returns None if no episode has been completed yet.
        Used by the /grader endpoint.
        """
        return self._last_grader_result
