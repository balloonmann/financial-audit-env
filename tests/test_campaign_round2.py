"""Round 2 campaign API and module regression tests.

Covers: start, task flow, submit, advance, memory (cross-period findings),
fatigue tracking, dependency enforcement, schema drift, reproducibility,
and full 5-period campaign run.
"""

import pytest
from fastapi.testclient import TestClient

from financial_audit_env.models import AgentRole
from financial_audit_env.server.adversarial import FraudDifficultyController
from financial_audit_env.server.app import app
from financial_audit_env.server.campaign import CampaignController
from financial_audit_env.server.instructions import REGULATORY_SHOCKS
from financial_audit_env.server.regulatory import get_shock_for_period_step

client = TestClient(app)

SPECIALIST_ROLES = [
    "expense_specialist",
    "invoice_specialist",
    "gst_specialist",
    "fraud_specialist",
]


def _start_campaign(seed: int = 42, periods: int = 3) -> str:
    response = client.post("/campaign/start", json={"seed": seed, "total_periods": periods})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "started"
    assert "campaign_id" in data
    return data["campaign_id"]


def _run_all_specialists(campaign_id: str):
    """Run all 4 specialists through start→submit with empty findings."""
    for role in SPECIALIST_ROLES:
        start = client.post("/campaign/task/start", json={"campaign_id": campaign_id, "role": role})
        assert start.status_code == 200
        submit = client.post(
            "/campaign/task/submit",
            json={
                "campaign_id": campaign_id,
                "role": role,
                "action": {"findings": [], "submit_final": True},
            },
        )
        assert submit.status_code == 200


# --- Scenario 1: Basic campaign start ---
def test_campaign_start_returns_valid_observation():
    campaign_id = _start_campaign()
    assert len(campaign_id) > 0

    state = client.get("/campaign/state", params={"campaign_id": campaign_id})
    assert state.status_code == 200
    data = state.json()
    assert data["state"]["current_period"] == 1
    assert data["state"]["total_periods"] == 3


# --- Scenario 2: Full task flow and period advance ---
def test_campaign_task_flow_and_period_advance_alias():
    campaign_id = _start_campaign()
    _run_all_specialists(campaign_id)

    advance = client.post("/campaign/period/advance", json={"campaign_id": campaign_id})
    assert advance.status_code == 200
    assert advance.json()["status"] == "advanced"


# --- Scenario 3: Submit with findings populates history ---
def test_campaign_submit_with_findings_populates_history():
    campaign_id = _start_campaign()

    # Start expense specialist
    client.post("/campaign/task/start", json={"campaign_id": campaign_id, "role": "expense_specialist"})

    # Submit with findings
    submit = client.post(
        "/campaign/task/submit",
        json={
            "campaign_id": campaign_id,
            "role": "expense_specialist",
            "action": {
                "findings": [
                    {
                        "document_id": "EXP-010",
                        "error_type": "over_limit",
                        "description": "Meal expense over limit",
                        "confidence": 0.9,
                    }
                ],
                "submit_final": True,
            },
        },
    )
    assert submit.status_code == 200

    # Check state has findings history
    state = client.get("/campaign/state", params={"campaign_id": campaign_id})
    history = state.json()["state"]["findings_history"]
    assert len(history) >= 1
    assert history[0]["document_id"] == "EXP-010"


# --- Scenario 4: Dependency enforcement (direct controller) ---
def test_dependency_enforcement():
    controller = CampaignController(total_periods=3)
    controller.start_campaign(seed=42)

    # Trying to start GST before invoice should fail
    with pytest.raises(ValueError, match="Dependency violation"):
        controller.reset_for_role(AgentRole.GST_SPECIALIST)


# --- Scenario 5: Budget tracking ---
def test_budget_decreases_on_task_submit():
    controller = CampaignController(total_periods=3)
    obs = controller.start_campaign(seed=42)
    initial_budget = controller.state.budget_remaining

    from financial_audit_env.models import AuditAction

    # Submit expense specialist action
    controller.submit_specialist_action(AgentRole.EXPENSE_SPECIALIST, AuditAction(submit_final=True))
    assert controller.state.budget_remaining < initial_budget


# --- Scenario 6: Schema drift in period 3+ ---
def test_schema_drift_after_period_3():
    controller = CampaignController(total_periods=5)
    controller.start_campaign(seed=42)

    from financial_audit_env.models import AuditAction

    # Run through periods 1 and 2
    for _ in range(2):
        for role in [AgentRole.EXPENSE_SPECIALIST, AgentRole.INVOICE_SPECIALIST,
                     AgentRole.GST_SPECIALIST, AgentRole.FRAUD_SPECIALIST]:
            controller.reset_for_role(role)
            controller.submit_specialist_action(role, AuditAction(submit_final=True))
        controller.advance_period()

    # Period 3 should have schema changes
    assert controller.state.world_state.schema_version == 2
    assert "vendor_gstin" in controller.state.world_state.schema_changes


# --- Scenario 7: Reproducibility across identical seeds ---
def test_campaign_reproducibility():
    controller1 = CampaignController(total_periods=3)
    obs1 = controller1.start_campaign(seed=42)

    controller2 = CampaignController(total_periods=3)
    obs2 = controller2.start_campaign(seed=42)

    assert obs1.world_state == obs2.world_state
    assert obs1.budget_remaining == obs2.budget_remaining


# --- Scenario 8: Full 5-period campaign run (direct controller to avoid rate limit) ---
def test_full_5_period_campaign():
    from financial_audit_env.models import AuditAction

    controller = CampaignController(total_periods=5)
    obs = controller.start_campaign(seed=42)
    assert obs.current_period == 1

    roles = [
        AgentRole.EXPENSE_SPECIALIST,
        AgentRole.INVOICE_SPECIALIST,
        AgentRole.GST_SPECIALIST,
        AgentRole.FRAUD_SPECIALIST,
    ]

    for period in range(1, 6):
        for role in roles:
            controller.reset_for_role(role)
            controller.submit_specialist_action(role, AuditAction(submit_final=True))
        if period < 5:
            obs = controller.advance_period()
            assert obs.current_period == period + 1

    assert controller.state.current_period == 5
    assert controller.state.world_state.schema_version == 2  # Period 3+ schema drift
    assert len(controller.state.world_state.policy_updates) > 0  # Period 5 policies


# --- Scenario 9: Overseer review via API ---
def test_overseer_review_api():
    campaign_id = _start_campaign()
    _run_all_specialists(campaign_id)

    review = client.post(
        "/overseer/review",
        json={
            "campaign_id": campaign_id,
            "action": {
                "audit_trail_id": f"trail-test-{campaign_id}",
                "decisions": [
                    {
                        "finding_ref": "EXP-010:over_limit",
                        "verdict": "approve",
                        "reason_code": "evidence_strong",
                        "confidence": 0.85,
                    }
                ],
                "conflicts_resolved": [],
                "task_reassignments": {},
            },
        },
    )
    assert review.status_code == 200
    assert review.json()["status"] == "reviewed"


# --- Scenario 10: Self-improve seed overlap rejected ---
def test_self_improve_seed_overlap_rejected():
    campaign_id = _start_campaign()

    bad = client.post(
        "/self-improve",
        json={
            "campaign_id": campaign_id,
            "train_seeds": [1, 2, 3],
            "held_out_seeds": [3, 4],
        },
    )
    assert bad.status_code == 400

    good = client.post(
        "/self-improve",
        json={
            "campaign_id": campaign_id,
            "train_seeds": [1, 2, 3],
            "held_out_seeds": [4, 5],
        },
    )
    assert good.status_code == 200
    assert good.json()["summary"]["accepted"] in (True, False)


# --- Scenario: Regulatory and adversarial helpers smoke test ---
def test_regulatory_and_adversarial_helpers_smoke():
    first = REGULATORY_SHOCKS[0]
    shock = get_shock_for_period_step(
        period=first["trigger_period"],
        step=first["trigger_after_step"] + 1,
        delivered=set(),
    )
    assert shock is not None

    controller = FraudDifficultyController()
    docs, gt = controller.generate_adversarial_data(seed=7, difficulty=2)
    assert "transactions" in docs
    assert isinstance(gt, list)


# --- Scenario: Campaign state path parameter alias ---
def test_campaign_state_path_parameter():
    campaign_id = _start_campaign()
    state = client.get(f"/campaign/state/{campaign_id}")
    assert state.status_code == 200
    assert state.json()["campaign_id"] == campaign_id


# --- Scenario: Self-improve history endpoint ---
def test_self_improve_history_endpoint():
    campaign_id = _start_campaign()

    # Run one iteration
    client.post(
        "/self-improve",
        json={
            "campaign_id": campaign_id,
            "train_seeds": [1, 2, 3],
            "held_out_seeds": [4, 5],
        },
    )

    history = client.get("/self-improve/history", params={"campaign_id": campaign_id})
    assert history.status_code == 200
    assert len(history.json()["history"]) >= 1
