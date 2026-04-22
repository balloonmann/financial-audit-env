"""Adversarial red-vs-blue controller tests."""

from financial_audit_env.server.adversarial import FraudDifficultyController


def test_generate_adversarial_data_is_deterministic():
    controller = FraudDifficultyController()
    docs1, gt1 = controller.generate_adversarial_data(seed=42, difficulty=3)
    docs2, gt2 = controller.generate_adversarial_data(seed=42, difficulty=3)

    assert docs1 == docs2
    assert gt1 == gt2


def test_adapt_difficulty_increases_with_high_blue_f1():
    controller = FraudDifficultyController()
    controller.record_blue_result(seed=1, f1_score=0.9, caught=["a"], missed=[])
    controller.record_blue_result(seed=2, f1_score=0.8, caught=["b"], missed=[])
    controller.record_blue_result(seed=3, f1_score=0.85, caught=["c"], missed=[])

    out = controller.adapt_difficulty()
    assert out["new_level"] >= 2
    assert out["action"] in {"increased", "maintained"}


def test_adapt_difficulty_decreases_with_low_blue_f1():
    controller = FraudDifficultyController()
    controller.red_level = 3
    controller.record_blue_result(seed=1, f1_score=0.1, caught=[], missed=["a"])
    controller.record_blue_result(seed=2, f1_score=0.2, caught=[], missed=["b"])
    controller.record_blue_result(seed=3, f1_score=0.25, caught=[], missed=["c"])

    out = controller.adapt_difficulty()
    assert out["new_level"] <= 2
    assert out["action"] in {"decreased", "maintained"}


def test_get_arms_race_data_shape():
    controller = FraudDifficultyController()
    data = controller.get_arms_race_data()
    assert "current_red_level" in data
    assert "detection_history" in data
    assert "arms_race_log" in data
