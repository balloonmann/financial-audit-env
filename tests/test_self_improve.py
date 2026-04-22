"""Self-improvement engine regression tests.

Covers: iteration, regression gate, history tracking, seed separation,
and deterministic score generation.
"""

import pytest
from financial_audit_env.server.campaign import CampaignController
from financial_audit_env.server.self_improve import SelfImproveEngine


def test_self_improve_iteration_and_history():
    controller = CampaignController(total_periods=3)
    obs = controller.start_campaign(seed=42)

    engine = SelfImproveEngine()
    out = engine.run_iteration(
        campaign_id=obs.campaign_id,
        train_seeds=[1, 2, 3],
        held_out_seeds=[101, 102],
        campaign_controller=controller,
    )

    assert out["iteration"] == 1
    assert "baseline" in out and "candidate" in out
    assert isinstance(out["accepted"], bool)

    history = engine.get_history(obs.campaign_id)
    assert len(history) == 1
    assert history[0]["delta"] == out["delta"]


def test_self_improve_seed_overlap_rejected():
    controller = CampaignController(total_periods=3)
    obs = controller.start_campaign(seed=7)

    engine = SelfImproveEngine()
    with pytest.raises(ValueError, match="disjoint"):
        engine.run_iteration(
            campaign_id=obs.campaign_id,
            train_seeds=[1, 2],
            held_out_seeds=[2, 3],
            campaign_controller=controller,
        )


def test_self_improve_multiple_iterations_tracked():
    controller = CampaignController(total_periods=3)
    obs = controller.start_campaign(seed=42)
    engine = SelfImproveEngine()

    # Run 3 iterations
    for i in range(3):
        out = engine.run_iteration(
            campaign_id=obs.campaign_id,
            train_seeds=[10, 11, 12],
            held_out_seeds=[200, 201],
            campaign_controller=controller,
        )
        assert out["iteration"] == i + 1

    history = engine.get_history(obs.campaign_id)
    assert len(history) == 3
    assert history[0]["iteration"] == 1
    assert history[2]["iteration"] == 3


def test_self_improve_empty_seeds_rejected():
    controller = CampaignController(total_periods=3)
    obs = controller.start_campaign(seed=42)
    engine = SelfImproveEngine()

    with pytest.raises(ValueError):
        engine.run_iteration(
            campaign_id=obs.campaign_id,
            train_seeds=[],
            held_out_seeds=[100],
            campaign_controller=controller,
        )

    with pytest.raises(ValueError):
        engine.run_iteration(
            campaign_id=obs.campaign_id,
            train_seeds=[42],
            held_out_seeds=[],
            campaign_controller=controller,
        )


def test_self_improve_candidate_beats_baseline():
    """The pseudo benchmark is designed so candidate always slightly beats baseline."""
    controller = CampaignController(total_periods=3)
    obs = controller.start_campaign(seed=42)
    engine = SelfImproveEngine()

    out = engine.run_iteration(
        campaign_id=obs.campaign_id,
        train_seeds=[1, 2, 3],
        held_out_seeds=[101, 102],
        campaign_controller=controller,
    )

    # Candidate should have higher mean_score than baseline (by design)
    assert out["candidate"]["mean_score"] >= out["baseline"]["mean_score"]
    assert out["delta"] >= 0
    assert out["accepted"] is True


def test_self_improve_history_for_unknown_campaign():
    engine = SelfImproveEngine()
    history = engine.get_history("nonexistent-campaign-id")
    assert history == []
