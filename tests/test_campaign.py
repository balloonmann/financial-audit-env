"""Dedicated campaign controller behavior tests."""

import pytest

from financial_audit_env.models import AgentRole, AuditAction
from financial_audit_env.server.campaign import CampaignController


def _empty_final_action() -> AuditAction:
    return AuditAction(findings=[], submit_final=True)


def test_campaign_dependency_enforcement():
    controller = CampaignController(total_periods=3)
    controller.start_campaign(seed=42)

    # Invoice cannot run before expense in same period.
    with pytest.raises(ValueError, match="Dependency violation"):
        controller.reset_for_role(AgentRole.INVOICE_SPECIALIST)


def test_campaign_flow_advance_and_schema_drift():
    controller = CampaignController(total_periods=4)
    obs = controller.start_campaign(seed=42)
    assert obs.current_period == 1

    roles = [
        AgentRole.EXPENSE_SPECIALIST,
        AgentRole.INVOICE_SPECIALIST,
        AgentRole.GST_SPECIALIST,
        AgentRole.FRAUD_SPECIALIST,
    ]

    for role in roles:
        controller.reset_for_role(role)
        controller.submit_specialist_action(role, _empty_final_action())

    obs2 = controller.advance_period()
    assert obs2.current_period == 2

    # Advance to period 3 and ensure schema drift metadata appears.
    for role in roles:
        controller.reset_for_role(role)
        controller.submit_specialist_action(role, _empty_final_action())
    controller.advance_period()

    for role in roles:
        controller.reset_for_role(role)
        controller.submit_specialist_action(role, _empty_final_action())
    obs3 = controller.advance_period()

    assert obs3.current_period == 4
    assert controller.state.world_state.schema_version >= 2
    assert controller.state.world_state.schema_changes.get("vendor_gstin") == "supplier_gstin"


def test_campaign_reproducibility_by_seed():
    c1 = CampaignController(total_periods=3)
    c2 = CampaignController(total_periods=3)

    o1 = c1.start_campaign(seed=99)
    o2 = c2.start_campaign(seed=99)

    assert o1.world_state.model_dump() == o2.world_state.model_dump()


def test_shock_defers_finalization_when_triggered():
    controller = CampaignController(total_periods=5)
    controller.start_campaign(seed=42)

    # Move to period 3 quickly with empty submissions in dependency order.
    roles = [
        AgentRole.EXPENSE_SPECIALIST,
        AgentRole.INVOICE_SPECIALIST,
        AgentRole.GST_SPECIALIST,
        AgentRole.FRAUD_SPECIALIST,
    ]
    for _ in range(2):
        for role in roles:
            controller.reset_for_role(role)
            controller.submit_specialist_action(role, _empty_final_action())
        controller.advance_period()

    # Period 3 GST: use final action repeatedly until shock drop step reached.
    controller.reset_for_role(AgentRole.EXPENSE_SPECIALIST)
    controller.submit_specialist_action(AgentRole.EXPENSE_SPECIALIST, _empty_final_action())
    controller.reset_for_role(AgentRole.INVOICE_SPECIALIST)
    controller.submit_specialist_action(AgentRole.INVOICE_SPECIALIST, _empty_final_action())

    controller.reset_for_role(AgentRole.GST_SPECIALIST)
    first = controller.submit_specialist_action(AgentRole.GST_SPECIALIST, _empty_final_action())
    second = controller.submit_specialist_action(AgentRole.GST_SPECIALIST, _empty_final_action())
    third = controller.submit_specialist_action(AgentRole.GST_SPECIALIST, _empty_final_action())

    feedback = "\n".join(
        [
            first.period_observation.feedback if first.period_observation else "",
            second.period_observation.feedback if second.period_observation else "",
            third.period_observation.feedback if third.period_observation else "",
        ]
    )
    assert "Final submission was deferred" in feedback
    assert len(controller.state.regulatory_shocks_applied) >= 1
