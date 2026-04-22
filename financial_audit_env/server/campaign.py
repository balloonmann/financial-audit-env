"""Campaign controller for multi-period, multi-agent audit orchestration."""

import logging
import uuid
from typing import Any, Dict, List, Optional

from ..models import (
    AgentRole,
    AuditAction,
    CampaignObservation,
    CampaignState,
    OverseerAction,
)
from .environment import FinancialAuditEnvironment
from .instructions import get_active_instructions, get_pending_regulatory_shocks
from .regulatory import format_shock_notification, get_shock_for_period_step

logger = logging.getLogger("financial_audit_env.campaign")

# Budget and fatigue constants
STEP_COST = 2.5         # Each specialist step costs 2.5 audit hours
FATIGUE_PENALTY = 0.02  # Penalty per consecutive task without rotation
RUSH_PENALTY = 0.03     # Penalty when submitting at max_steps

TASK_BY_ROLE: Dict[AgentRole, str] = {
    AgentRole.EXPENSE_SPECIALIST: "expense_audit",
    AgentRole.INVOICE_SPECIALIST: "invoice_match",
    AgentRole.GST_SPECIALIST: "gst_reconciliation",
    AgentRole.FRAUD_SPECIALIST: "fraud_detection",
}

DEFAULT_TASK_ORDER = [
    AgentRole.EXPENSE_SPECIALIST,
    AgentRole.INVOICE_SPECIALIST,
    AgentRole.GST_SPECIALIST,
    AgentRole.FRAUD_SPECIALIST,
]


class CampaignController:
    """Composition-based wrapper around FinancialAuditEnvironment for campaigns."""

    def __init__(self, total_periods: int = 5):
        self._env = FinancialAuditEnvironment()
        self._state = CampaignState(total_periods=total_periods)
        self._task_order = list(DEFAULT_TASK_ORDER)
        self._completed_roles: Dict[int, List[str]] = {}
        self._delivered_shock_ids: set[str] = set()

    @property
    def state(self) -> CampaignState:
        return self._state

    def start_campaign(self, seed: int = 42, campaign_id: Optional[str] = None) -> CampaignObservation:
        campaign_id = campaign_id or str(uuid.uuid4())
        self._state = CampaignState(
            campaign_id=campaign_id,
            seed=seed,
            total_periods=self._state.total_periods,
            current_period=1,
            agent_roles={
                AgentRole.EXPENSE_SPECIALIST.value: AgentRole.EXPENSE_SPECIALIST,
                AgentRole.INVOICE_SPECIALIST.value: AgentRole.INVOICE_SPECIALIST,
                AgentRole.GST_SPECIALIST.value: AgentRole.GST_SPECIALIST,
                AgentRole.FRAUD_SPECIALIST.value: AgentRole.FRAUD_SPECIALIST,
                AgentRole.OVERSEER.value: AgentRole.OVERSEER,
            },
            instruction_compliance={},
        )
        self._completed_roles = {1: []}
        self._delivered_shock_ids = set()

        # Prime first specialist task.
        first_role = self._task_order[0]
        period_obs = self._env.reset(task_id=TASK_BY_ROLE[first_role], seed=seed)
        return self._build_campaign_observation(period_obs)

    def reset_for_role(self, role: AgentRole) -> CampaignObservation:
        self._validate_campaign_started()
        self._validate_dependency(role)

        task_id = TASK_BY_ROLE[role]
        period_seed = self._state.seed + (self._state.current_period - 1)
        period_obs = self._env.reset(task_id=task_id, seed=period_seed)
        return self._build_campaign_observation(period_obs)

    def submit_specialist_action(self, role: AgentRole, action: AuditAction) -> CampaignObservation:
        self._validate_campaign_started()
        self._validate_dependency(role)

        # Budget tracking
        self._state.budget_remaining = max(0, self._state.budget_remaining - STEP_COST)

        # Fatigue tracking
        consec = self._state.specialist_consecutive_tasks.get(role.value, 0) + 1
        self._state.specialist_consecutive_tasks[role.value] = consec

        obs = self._env.step(action)

        if action.findings:
            self._state.findings_history.extend(
                [
                    {
                        "period": self._state.current_period,
                        "role": role.value,
                        "document_id": f.document_id,
                        "error_type": f.error_type,
                        "confidence": f.confidence,
                    }
                    for f in action.findings
                ]
            )

        if obs.done:
            done_roles = self._completed_roles.setdefault(self._state.current_period, [])
            if role.value not in done_roles:
                done_roles.append(role.value)

        # Deliver at most one new shock per step to keep progression deterministic.
        delivered = False
        while not delivered:
            shock = get_shock_for_period_step(
                period=self._state.current_period,
                step=obs.step_number,
                delivered=self._delivered_shock_ids,
            )
            if shock is None:
                break
            self._delivered_shock_ids.add(str(shock["id"]))
            self._state.regulatory_shocks_applied.append(
                {
                    "id": shock["id"],
                    "period": self._state.current_period,
                    "step": obs.step_number,
                    "severity": shock.get("severity", 0),
                }
            )
            self._state.world_state.mid_period_rule_drops.append(shock)
            obs.feedback = f"{obs.feedback}\n\n{format_shock_notification(shock)}"
            delivered = True

        return self._build_campaign_observation(obs)

    def submit_overseer_action(self, action: OverseerAction) -> Dict[str, Any]:
        self._validate_campaign_started()

        accepted = sum(1 for d in action.decisions if d.verdict == "approve")
        rejected = sum(1 for d in action.decisions if d.verdict == "reject")
        escalated = sum(1 for d in action.decisions if d.verdict == "escalate")

        return {
            "campaign_id": self._state.campaign_id,
            "period": self._state.current_period,
            "audit_trail_id": action.audit_trail_id,
            "accepted": accepted,
            "rejected": rejected,
            "escalated": escalated,
            "conflicts_resolved": len(action.conflicts_resolved),
            "reassignments": action.task_reassignments,
        }

    def advance_period(self) -> CampaignObservation:
        self._validate_campaign_started()

        if self._state.current_period >= self._state.total_periods:
            return self._build_campaign_observation(None)

        self._state.current_period += 1
        self._state.world_state.fiscal_period = self._state.current_period
        self._state.world_state.policy_version += 1
        self._state.world_state.schema_version = 2 if self._state.current_period >= 3 else 1

        self._state.world_state.policy_updates = [
            inst["text"]
            for inst in get_active_instructions(self._state.current_period)
            if inst["bucket"] == "policy"
        ]

        # Simple deterministic mutation for schema drift.
        if self._state.current_period >= 3:
            self._state.world_state.schema_changes = {
                "vendor_gstin": "supplier_gstin",
            }

        # Reset periodic shock queue.
        self._state.world_state.mid_period_rule_drops = []
        self._completed_roles.setdefault(self._state.current_period, [])

        # Keep instruction helper imported and validated during campaign transitions.
        _ = get_pending_regulatory_shocks(self._state.current_period, step=0)

        # Prime first role for this period.
        first_role = self._task_order[0]
        period_seed = self._state.seed + (self._state.current_period - 1)
        period_obs = self._env.reset(task_id=TASK_BY_ROLE[first_role], seed=period_seed)
        return self._build_campaign_observation(period_obs)

    def _build_campaign_observation(self, period_obs: Optional[Any]) -> CampaignObservation:
        assignments = {role.value: TASK_BY_ROLE.get(role, "overseer") for role in self._task_order}
        assignments[AgentRole.OVERSEER.value] = "overseer_review"

        return CampaignObservation(
            campaign_id=self._state.campaign_id,
            current_period=self._state.current_period,
            total_periods=self._state.total_periods,
            world_state=self._state.world_state,
            period_observation=period_obs,
            findings_history_summary=self._state.findings_history[-25:],
            active_alerts=self._state.world_state.active_alerts,
            budget_remaining=self._state.budget_remaining,
            specialist_assignments=assignments,
            pending_regulatory_shocks=self._state.world_state.mid_period_rule_drops,
        )

    def _validate_campaign_started(self) -> None:
        if not self._state.campaign_id:
            raise RuntimeError("Campaign not started. Call start_campaign first.")

    def _validate_dependency(self, role: AgentRole) -> None:
        if role not in TASK_BY_ROLE:
            raise ValueError(f"Role '{role}' is not a specialist role.")

        done_roles = self._completed_roles.setdefault(self._state.current_period, [])
        required_before = self._task_order[: self._task_order.index(role)]
        for dep in required_before:
            if dep.value not in done_roles:
                raise ValueError(
                    f"Dependency violation: '{role.value}' requires '{dep.value}' completed first."
                )
