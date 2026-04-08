"""Tests for grading system — F1, weighted F1, partial credit, risk scoring."""

import pytest
from financial_audit_env.server.graders import compute_f1_score, compute_step_reward
from financial_audit_env.server.data_generator import generate_expense_data


class TestF1Score:
    """Tests for the core F1 scoring function."""

    @pytest.fixture
    def ground_truth(self):
        _, gt = generate_expense_data(42)
        return gt

    def test_perfect_score(self, ground_truth):
        result = compute_f1_score(ground_truth, ground_truth)
        assert 0 < result["score"] < 1
        assert 0 < result["precision"] < 1
        assert 0 < result["recall"] < 1
        assert result["true_positives"] == len(ground_truth)
        assert result["false_positives"] == 0

    def test_empty_findings(self, ground_truth):
        result = compute_f1_score([], ground_truth)
        assert 0 < result["score"] < 1
        assert 0 < result["precision"] < 1
        assert 0 < result["recall"] < 1

    def test_partial_findings(self, ground_truth):
        half = ground_truth[:3]
        result = compute_f1_score(half, ground_truth)
        assert 0 < result["score"] < 1
        assert result["true_positives"] == 3
        assert result["recall"] < 1

    def test_all_false_positives(self, ground_truth):
        fake = [{"document_id": "FAKE-001", "error_type": "fake_error"}]
        result = compute_f1_score(fake, ground_truth)
        assert 0 < result["score"] < 1
        assert result["false_positives"] == 1
        assert result["true_positives"] == 0

    def test_case_insensitive_matching(self, ground_truth):
        # Uppercase the document IDs and error types
        upper = [
            {"document_id": g["document_id"].upper(), "error_type": g["error_type"].upper()}
            for g in ground_truth
        ]
        result = compute_f1_score(upper, ground_truth)
        assert 0 < result["score"] < 1

    def test_duplicate_findings_not_double_counted(self, ground_truth):
        # Submit the same finding twice
        doubled = ground_truth + ground_truth
        result = compute_f1_score(doubled, ground_truth)
        assert result["true_positives"] == len(ground_truth)
        assert result["duplicates"] >= len(ground_truth)  # Duplicates are now tracked separately

    def test_score_changes_with_different_findings(self, ground_truth):
        """Verify grader isn't returning constant scores."""
        result1 = compute_f1_score(ground_truth[:1], ground_truth)
        result2 = compute_f1_score(ground_truth[:4], ground_truth)
        result3 = compute_f1_score(ground_truth, ground_truth)
        assert result1["score"] < result2["score"] < result3["score"]


class TestWeightedF1:
    """Tests for severity-weighted F1 scoring."""

    @pytest.fixture
    def ground_truth(self):
        _, gt = generate_expense_data(42)
        return gt

    def test_weighted_score_exists(self, ground_truth):
        result = compute_f1_score(ground_truth, ground_truth)
        assert "weighted_score" in result
        assert result["weighted_score"] > 0

    def test_perfect_weighted_score(self, ground_truth):
        result = compute_f1_score(ground_truth, ground_truth)
        assert 0 < result["weighted_score"] < 1


class TestPartialCredit:
    """Tests for partial credit (right doc, wrong error_type)."""

    def test_partial_credit_for_right_doc_wrong_type(self):
        gt = [{"document_id": "EXP-010", "error_type": "over_limit"}]
        findings = [{"document_id": "EXP-010", "error_type": "wrong_category"}]
        result = compute_f1_score(findings, gt)
        # Should have 0 true positives but 1 partial match
        assert result["true_positives"] == 0
        assert result["partial_matches"] == 1
        assert result["partial_credit_score"] > 0

    def test_no_partial_credit_for_wrong_doc(self):
        gt = [{"document_id": "EXP-010", "error_type": "over_limit"}]
        findings = [{"document_id": "WRONG-ID", "error_type": "over_limit"}]
        result = compute_f1_score(findings, gt)
        assert result["partial_matches"] == 0


class TestConfusionMatrix:
    """Tests for per-error_type confusion matrix."""

    @pytest.fixture
    def ground_truth(self):
        _, gt = generate_expense_data(42)
        return gt

    def test_confusion_matrix_exists(self, ground_truth):
        result = compute_f1_score(ground_truth[:3], ground_truth)
        assert "confusion_matrix" in result
        assert len(result["confusion_matrix"]) > 0

    def test_confusion_matrix_structure(self, ground_truth):
        result = compute_f1_score(ground_truth[:3], ground_truth)
        for error_type, stats in result["confusion_matrix"].items():
            assert "ground_truth" in stats
            assert "correctly_found" in stats
            assert "missed" in stats
            assert "false_positives" in stats
            assert "severity_weight" in stats

    def test_perfect_confusion_matrix(self, ground_truth):
        result = compute_f1_score(ground_truth, ground_truth)
        for error_type, stats in result["confusion_matrix"].items():
            assert stats["missed"] == 0
            assert stats["false_positives"] == 0


class TestRiskScoring:
    """Tests for monetary risk assessment."""

    @pytest.fixture
    def ground_truth(self):
        _, gt = generate_expense_data(42)
        return gt

    def test_risk_score_exists(self, ground_truth):
        result = compute_f1_score(ground_truth, ground_truth)
        assert "risk_score" in result
        rs = result["risk_score"]
        assert "caught_value" in rs
        assert "missed_value" in rs
        assert "total_risk_value" in rs
        assert "risk_mitigation_pct" in rs

    def test_perfect_risk_mitigation(self, ground_truth):
        result = compute_f1_score(ground_truth, ground_truth)
        assert result["risk_score"]["risk_mitigation_pct"] == 100.0
        assert result["risk_score"]["missed_value"] == 0

    def test_zero_risk_mitigation(self, ground_truth):
        result = compute_f1_score([], ground_truth)
        assert result["risk_score"]["risk_mitigation_pct"] == 0
        assert result["risk_score"]["caught_value"] == 0


class TestStepReward:
    """Tests for per-step reward computation."""

    @pytest.fixture
    def ground_truth(self):
        _, gt = generate_expense_data(42)
        return gt

    def test_correct_finding_positive_reward(self, ground_truth):
        findings = [ground_truth[0]]
        reward = compute_step_reward(findings, findings, ground_truth, 1, False)
        assert reward > 0  # True positive should give positive reward

    def test_false_positive_negative_reward(self, ground_truth):
        fake = [{"document_id": "FAKE", "error_type": "fake"}]
        reward = compute_step_reward(fake, fake, ground_truth, 1, False)
        assert reward <= 0.01  # False positive + step penalty (clamped to epsilon)

    def test_reward_decay_over_steps(self, ground_truth):
        findings = [ground_truth[0]]
        r1 = compute_step_reward(findings, findings, ground_truth, 1, False)
        r5 = compute_step_reward(findings, findings, ground_truth, 5, False)
        # Later steps should yield less reward due to decay
        assert r5 < r1

    def test_final_bonus_high_recall(self, ground_truth):
        # Submit all findings as final
        reward = compute_step_reward(ground_truth, ground_truth, ground_truth, 1, True)
        # Should get the high-recall bonus
        assert reward > 0.3

    def test_final_penalty_low_recall(self, ground_truth):
        fake = [{"document_id": "FAKE", "error_type": "fake"}]
        reward = compute_step_reward(fake, fake, ground_truth, 1, True)
        # Should get the low-recall penalty (clamped to epsilon)
        assert reward <= 0.01
