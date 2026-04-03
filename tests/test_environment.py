"""Tests for environment — lifecycle, all tasks, investigation mode, adaptive difficulty."""

import pytest
from financial_audit_env.server.environment import FinancialAuditEnvironment
from financial_audit_env.models import AuditAction, Finding


class TestResetStep:
    """Tests for basic reset/step/state lifecycle."""

    def test_reset_returns_observation(self, env):
        obs = env.reset(task_id="expense_audit", seed=42)
        assert obs.task_id == "expense_audit"
        assert obs.done is False
        assert obs.step_number == 0
        assert len(obs.documents) > 0

    def test_step_without_reset_raises(self, env):
        with pytest.raises(RuntimeError, match="not initialized"):
            env.step(AuditAction(findings=[], submit_final=True))

    def test_step_with_findings(self, env):
        obs = env.reset(task_id="expense_audit", seed=42)
        action = AuditAction(
            findings=[
                Finding(
                    document_id="EXP-010",
                    error_type="over_limit",
                    description="Test",
                )
            ],
            submit_final=False,
        )
        obs = env.step(action)
        assert obs.done is False
        assert obs.step_number == 1
        assert len(obs.findings_so_far) == 1

    def test_submit_final_ends_episode(self, env):
        obs = env.reset(task_id="expense_audit", seed=42)
        action = AuditAction(
            findings=[
                Finding(
                    document_id="EXP-010",
                    error_type="over_limit",
                    description="Test",
                )
            ],
            submit_final=True,
        )
        obs = env.step(action)
        assert obs.done is True

    def test_max_steps_ends_episode(self, env):
        obs = env.reset(task_id="expense_audit", seed=42)
        for i in range(obs.max_steps):
            action = AuditAction(findings=[], submit_final=False)
            obs = env.step(action)
        assert obs.done is True

    def test_state_tracks_progress(self, env):
        env.reset(task_id="expense_audit", seed=42)
        state = env.state
        assert state.task_id == "expense_audit"
        assert state.step_count == 0
        assert state.total_errors > 0


class TestAllTasks:
    """Test that all 4 tasks run without errors."""

    @pytest.mark.parametrize("task_id", [
        "expense_audit", "invoice_match", "gst_reconciliation", "fraud_detection",
    ])
    def test_task_reset_and_step(self, env, task_id):
        obs = env.reset(task_id=task_id, seed=42)
        assert obs.task_id == task_id
        assert not obs.done
        assert len(obs.documents) > 0

        # Submit empty findings to end episode
        action = AuditAction(findings=[], submit_final=True)
        obs = env.step(action)
        assert obs.done is True

    @pytest.mark.parametrize("task_id", [
        "expense_audit", "invoice_match", "gst_reconciliation", "fraud_detection",
    ])
    def test_grader_after_completion(self, env, task_id):
        env.reset(task_id=task_id, seed=42)
        env.step(AuditAction(findings=[], submit_final=True))
        result = env.last_grader_result
        assert result is not None
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_invalid_task_raises(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="nonexistent_task", seed=42)

    def test_grader_result_has_enhanced_fields(self, env):
        from financial_audit_env.server.data_generator import generate_expense_data
        _, gt = generate_expense_data(42)
        env.reset(task_id="expense_audit", seed=42)
        # Submit perfect findings
        action = AuditAction(
            findings=[
                Finding(document_id=g["document_id"], error_type=g["error_type"], description="Test")
                for g in gt
            ],
            submit_final=True,
        )
        env.step(action)
        result = env.last_grader_result
        assert "weighted_score" in result
        assert "partial_credit_score" in result
        assert "confusion_matrix" in result
        assert "risk_score" in result


class TestInvestigationMode:
    """Tests for investigation mode."""

    def test_investigation_mode_no_docs_on_reset(self, env):
        obs = env.reset(task_id="expense_audit", seed=42, investigation_mode=True)
        assert obs.investigation_mode is True
        assert len(obs.documents) == 0  # No docs until investigated
        assert len(obs.available_categories) > 0
        assert obs.data_summary is not None

    def test_standard_mode_has_docs_on_reset(self, env):
        obs = env.reset(task_id="expense_audit", seed=42, investigation_mode=False)
        assert obs.investigation_mode is False
        assert len(obs.documents) > 0

    def test_investigation_mode_submit_still_works(self, env):
        env.reset(task_id="expense_audit", seed=42, investigation_mode=True)
        action = AuditAction(
            findings=[
                Finding(document_id="EXP-010", error_type="over_limit", description="Test")
            ],
            submit_final=True,
        )
        obs = env.step(action)
        assert obs.done is True


class TestAdaptiveDifficulty:
    """Tests for adaptive difficulty tracking."""

    def test_no_history_returns_default(self, env):
        result = env.get_adaptive_difficulty()
        assert result["suggestion"] == "default"

    def test_high_scores_increase_difficulty(self, env):
        # Simulate high scores
        env._score_history = [0.9, 0.85, 0.88]
        result = env.get_adaptive_difficulty()
        assert result["suggestion"] == "increase_difficulty"
        assert result["noise_level"] >= 0.7

    def test_low_scores_decrease_difficulty(self, env):
        env._score_history = [0.1, 0.2, 0.15]
        result = env.get_adaptive_difficulty()
        assert result["suggestion"] == "decrease_difficulty"


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_findings_submission(self, env):
        env.reset(task_id="expense_audit", seed=42)
        action = AuditAction(findings=[], submit_final=True)
        obs = env.step(action)
        assert obs.done is True
        assert env.last_grader_result["score"] == 0.0

    def test_invalid_error_type_filtered(self, env):
        env.reset(task_id="expense_audit", seed=42)
        action = AuditAction(
            findings=[
                Finding(
                    document_id="EXP-010",
                    error_type="nonexistent_type",
                    description="Test",
                )
            ],
            submit_final=False,
        )
        obs = env.step(action)
        # Invalid error type should be filtered out
        assert "Invalid error_type" in obs.feedback

    def test_many_findings_accepted(self, env):
        env.reset(task_id="expense_audit", seed=42)
        findings = [
            Finding(
                document_id=f"DOC-{i}",
                error_type="over_limit",
                description=f"Test {i}",
            )
            for i in range(50)
        ]
        action = AuditAction(findings=findings, submit_final=True)
        obs = env.step(action)
        assert obs.done is True

    def test_reset_clears_previous_state(self, env):
        env.reset(task_id="expense_audit", seed=42)
        env.step(AuditAction(
            findings=[Finding(document_id="EXP-010", error_type="over_limit", description="T")],
            submit_final=True,
        ))
        # Reset and run a new episode
        obs = env.reset(task_id="invoice_match", seed=42)
        assert obs.task_id == "invoice_match"
        assert len(obs.findings_so_far) == 0
        assert env.state.step_count == 0
