"""Training pipeline correctness tests (parser, reward, evaluator)."""

from training.evaluator import InProcessEvaluator
from training.reward import make_reward, parse_findings_from_text


def test_parse_findings_from_json_array():
    text = (
        '[{"document_id": "EXP-010", "error_type": "over_limit", '
        '"description": "Exceeded policy", "confidence": 0.88}]'
    )
    parsed = parse_findings_from_text(text)

    assert len(parsed) == 1
    assert parsed[0]["document_id"] == "EXP-010"
    assert parsed[0]["error_type"] == "over_limit"
    assert parsed[0]["confidence"] == 0.88


def test_parse_findings_from_truncated_json_array_salvages_complete_objects():
    text = (
        '[{"document_id": "INV-001", "error_type": "price_mismatch", '
        '"description": "Unit price mismatch", "confidence": 1.0}, '
        '{"document_id": "INV-002", "error_type": "quantity_mismatch", '
        '"description": "Quantity mismatch", "confidence": 0.8}, '
        '{"document_id": "INV-003", "error_type": "cascading_total", '
        '"description": "Total mismatch"'
    )

    parsed = parse_findings_from_text(text)

    assert len(parsed) == 2
    assert parsed[0]["document_id"] == "INV-001"
    assert parsed[1]["document_id"] == "INV-002"


def test_parse_findings_from_free_text_block():
    text = (
        "document_id: INV-101\n"
        "error_type: mismatch_amount\n"
        "description: Invoice amount differs from PO\n"
        "confidence: 0.73\n"
    )
    parsed = parse_findings_from_text(text)

    assert len(parsed) == 1
    assert parsed[0]["document_id"] == "INV-101"
    assert parsed[0]["error_type"] == "mismatch_amount"
    assert parsed[0]["description"].startswith("Invoice amount")


def test_parse_findings_invalid_input_returns_empty():
    assert parse_findings_from_text("") == []
    assert parse_findings_from_text("no structured finding here") == []


def test_make_reward_is_clamped_and_penalizes_exploit():
    high_quality = make_reward(
        {
            "f1_score": 0.92,
            "ece": 0.10,
            "agreement": 0.80,
            "exploit_flag": 0.0,
        }
    )
    exploited = make_reward(
        {
            "f1_score": 0.92,
            "ece": 0.10,
            "agreement": 0.80,
            "exploit_flag": 1.0,
        }
    )

    assert 0.01 <= high_quality <= 0.99
    assert 0.01 <= exploited <= 0.99
    assert exploited < high_quality


def test_inprocess_evaluator_smoke_returns_expected_keys():
    evaluator = InProcessEvaluator()
    result = evaluator.evaluate(
        task_id="expense_audit",
        seed=42,
        findings_dicts=[
            {
                "document_id": "EXP-001",
                "error_type": "over_limit",
                "description": "Policy breach",
                "confidence": 0.7,
            }
        ],
    )

    assert "score" in result
    assert "weighted_score" in result
    assert "precision" in result
    assert "recall" in result
    assert "grader_result" in result
    assert 0.01 <= result["score"] <= 0.99
