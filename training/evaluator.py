"""
In-process evaluator for GRPO training.
Uses direct Python calls — no HTTP overhead.
Produces identical scores to the API endpoints.

Usage in Colab:
    from training.evaluator import InProcessEvaluator
    evaluator = InProcessEvaluator()
    result = evaluator.evaluate("expense_audit", 42, [
        {"document_id": "EXP-010", "error_type": "over_limit", "description": "test"}
    ])
"""

from typing import Any, Dict, List, Optional

from financial_audit_env.server.environment import FinancialAuditEnvironment
from financial_audit_env.server.graders import compute_ece, compute_f1_score
from financial_audit_env.models import AuditAction, Finding


class InProcessEvaluator:
    """Direct Python evaluator for training — no HTTP overhead."""

    def __init__(self) -> None:
        self._env = FinancialAuditEnvironment()

    def evaluate(
        self,
        task_id: str,
        seed: int,
        findings_dicts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate findings against ground truth for a given task/seed.

        Args:
            task_id: One of expense_audit, invoice_match, gst_reconciliation, fraud_detection
            seed: Random seed for deterministic data generation
            findings_dicts: List of dicts with at minimum document_id, error_type, description

        Returns:
            Dict with reward, score, weighted_score, done, grader_result, ece
        """
        self._env.reset(task_id=task_id, seed=seed)

        findings = []
        for f in findings_dicts:
            findings.append(
                Finding(
                    document_id=f.get("document_id", "UNKNOWN"),
                    error_type=f.get("error_type", "unknown"),
                    description=f.get("description", "Finding"),
                    confidence=f.get("confidence"),
                    evidence_refs=f.get("evidence_refs", []),
                    rationale=f.get("rationale"),
                )
            )

        action = AuditAction(findings=findings, submit_final=True)
        obs = self._env.step(action)

        grader = self._env.last_grader_result or {}

        # Compute ECE if confidence scores are present
        ece_result: Optional[Dict[str, Any]] = None
        scored_findings = [fd for fd in findings_dicts if fd.get("confidence") is not None]
        if len(scored_findings) >= 10:
            ece_result = compute_ece(findings_dicts, self._env._ground_truth)

        return {
            "reward": obs.reward,
            "score": grader.get("score", 0.01),
            "weighted_score": grader.get("weighted_score", 0.01),
            "precision": grader.get("precision", 0.0),
            "recall": grader.get("recall", 0.0),
            "true_positives": grader.get("true_positives", 0),
            "false_positives": grader.get("false_positives", 0),
            "false_negatives": grader.get("false_negatives", 0),
            "done": obs.done,
            "grader_result": grader,
            "ece": ece_result,
        }

    def get_ground_truth(self, task_id: str, seed: int) -> List[Dict[str, str]]:
        """Get ground truth for a task/seed (for analysis/debugging only)."""
        self._env.reset(task_id=task_id, seed=seed)
        return self._env._ground_truth

    def get_documents(self, task_id: str, seed: int) -> Dict[str, Any]:
        """Get raw documents for a task/seed (for prompt construction)."""
        obs = self._env.reset(task_id=task_id, seed=seed)
        return obs.documents

    def get_task_info(self, task_id: str) -> Dict[str, Any]:
        """Get task description and error types."""
        from financial_audit_env.server.tasks import get_task
        return get_task(task_id)

    def run_empty_episode(self, task_id: str, seed: int) -> Dict[str, Any]:
        """Run an episode with no findings (baseline score)."""
        return self.evaluate(task_id, seed, [])

    def run_batch(self, task_id: str, seeds: List[int], findings_per_seed: Dict[int, List[Dict]]) -> List[Dict[str, Any]]:
        """Evaluate multiple seeds in batch."""
        results = []
        for seed in seeds:
            findings = findings_per_seed.get(seed, [])
            results.append(self.evaluate(task_id, seed, findings))
        return results
