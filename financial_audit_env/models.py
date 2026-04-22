# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — Pydantic models for the OpenEnv interface.
#
# These models define the typed contracts between the agent and the environment:
#   - Finding: A single audit issue identified by the agent
#   - AuditAction: What the agent submits each step
#   - InvestigateAction: Request to drill into specific document categories
#   - AuditObservation: What the agent sees after each step
#   - AuditState: Internal episode metadata
#
# All models use Pydantic strict mode for input validation (OWASP best practice).

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Attempt to import OpenEnv base classes. Falls back to standalone definitions
# if openenv-core is not installed (for local development / GitHub usage).
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    class Action(BaseModel):
        """Base action model (standalone fallback)."""
        pass

    class Observation(BaseModel):
        """Base observation model (standalone fallback)."""
        done: bool = False
        reward: Optional[float] = None
        metadata: Optional[Dict[str, Any]] = None

    class State(BaseModel):
        """Base state model (standalone fallback)."""
        episode_id: Optional[str] = None
        step_count: int = 0


# ---------------------------------------------------------------------------
# Input sanitization constants (OWASP)
# ---------------------------------------------------------------------------
MAX_STRING_LENGTH = 500
MAX_DOCUMENT_ID_LENGTH = 100
MAX_FINDINGS_PER_STEP = 50


class AgentRole(str, Enum):
    """Roles in the multi-agent audit team."""

    EXPENSE_SPECIALIST = "expense_specialist"
    INVOICE_SPECIALIST = "invoice_specialist"
    GST_SPECIALIST = "gst_specialist"
    FRAUD_SPECIALIST = "fraud_specialist"
    OVERSEER = "overseer"


def _sanitize_string(value: str, max_length: int = MAX_STRING_LENGTH) -> str:
    """
    Sanitize a user-provided string:
    - Strip leading/trailing whitespace
    - Truncate to max_length
    - Remove null bytes and control characters (except newlines/tabs)
    """
    if not isinstance(value, str):
        return value
    value = value.replace("\x00", "")
    value = "".join(
        ch for ch in value
        if ch in ("\n", "\r", "\t") or (ord(ch) >= 32)
    )
    return value.strip()[:max_length]


# ---------------------------------------------------------------------------
# Finding — a single audit issue reported by the agent
# ---------------------------------------------------------------------------
class Finding(BaseModel):
    """
    Represents a single audit finding submitted by the agent.

    Attributes:
        document_id: The identifier of the document/row with the issue.
        field:       Optional — which specific field has the error.
        error_type:  The category of error found. Must match allowed types.
        description: Human-readable explanation of the finding.
        suggested_fix: Optional recommended corrective action.
        severity:    Optional severity level (auto-populated by grader).
    """
    model_config = ConfigDict(strict=True)

    document_id: str = Field(
        ...,
        min_length=1,
        max_length=MAX_DOCUMENT_ID_LENGTH,
        description="ID of the document/row with the issue",
    )
    field: Optional[str] = Field(
        default=None,
        max_length=MAX_DOCUMENT_ID_LENGTH,
        description="Specific field with the error",
    )
    error_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Category of error (must match task's allowed types)",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=MAX_STRING_LENGTH,
        description="Human-readable explanation",
    )
    suggested_fix: Optional[str] = Field(
        default=None,
        max_length=MAX_STRING_LENGTH,
        description="Recommended corrective action",
    )
    severity: Optional[float] = Field(
        default=None,
        description="Severity weight (auto-populated by grader, 0.0-2.0)",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in this finding (0.0-1.0). Used for calibration scoring.",
    )
    evidence_refs: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Evidence references, e.g. ['EXP-010.amount=4500', 'policy.meals.limit=1500']",
    )
    rationale: Optional[str] = Field(
        default=None,
        max_length=MAX_STRING_LENGTH,
        description="Reasoning behind this finding",
    )

    @field_validator("document_id", "error_type", mode="before")
    @classmethod
    def sanitize_ids(cls, v: str) -> str:
        """Sanitize identifier fields."""
        return _sanitize_string(v, MAX_DOCUMENT_ID_LENGTH)

    @field_validator("description", "suggested_fix", mode="before")
    @classmethod
    def sanitize_text(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize free-text fields."""
        if v is None:
            return None
        return _sanitize_string(v, MAX_STRING_LENGTH)

    @field_validator("rationale", mode="before")
    @classmethod
    def sanitize_rationale(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize rationale text."""
        if v is None:
            return None
        return _sanitize_string(v, MAX_STRING_LENGTH)

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def sanitize_evidence_refs(cls, v: Any) -> Any:
        """Sanitize and bound evidence refs list values."""
        if not isinstance(v, list):
            return v
        return [_sanitize_string(str(item), MAX_DOCUMENT_ID_LENGTH) for item in v]


# ---------------------------------------------------------------------------
# AuditAction — what the agent submits each step
# ---------------------------------------------------------------------------
class AuditAction(Action):
    """
    Action submitted by the agent at each step.

    The agent can submit findings incrementally across multiple steps,
    or all at once with submit_final=True.
    """
    findings: List[Finding] = Field(
        default_factory=list,
        max_length=MAX_FINDINGS_PER_STEP,
        description="Audit findings for this step (max 50)",
    )
    submit_final: bool = Field(
        default=False,
        description="Set True to end the episode and trigger final grading",
    )

    @field_validator("findings", mode="before")
    @classmethod
    def validate_findings_count(cls, v: list) -> list:
        """Enforce maximum findings per step to prevent DoS."""
        if len(v) > MAX_FINDINGS_PER_STEP:
            raise ValueError(
                f"Maximum {MAX_FINDINGS_PER_STEP} findings per step. "
                f"Got {len(v)}."
            )
        return v


# ---------------------------------------------------------------------------
# InvestigateAction — request to drill into specific data (investigation mode)
# ---------------------------------------------------------------------------
class InvestigateAction(Action):
    """
    Action for investigation mode: request to view specific document categories.

    Instead of submitting findings, the agent requests more detailed views
    of specific document types. Only available when investigation_mode=True.
    """
    request_categories: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Document categories to investigate (e.g., 'expenses', 'invoices')",
    )
    request_summary: bool = Field(
        default=False,
        description="Request a statistical summary of the data",
    )


# ---------------------------------------------------------------------------
# AuditObservation — what the agent sees after each step
# ---------------------------------------------------------------------------
class AuditObservation(Observation):
    """
    Observation returned to the agent after reset() or step().
    """
    task_id: str = ""
    task_description: str = ""
    documents: Dict[str, Any] = Field(default_factory=dict)
    findings_so_far: List[Finding] = Field(default_factory=list)
    feedback: str = ""
    step_number: int = 0
    max_steps: int = 0
    # Investigation mode fields (optional, backwards compatible)
    investigation_mode: bool = Field(
        default=False,
        description="Whether investigation mode is active",
    )
    available_categories: List[str] = Field(
        default_factory=list,
        description="Document categories available for investigation",
    )
    data_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Statistical summary of data (investigation mode)",
    )


# ---------------------------------------------------------------------------
# AuditState — internal episode metadata
# ---------------------------------------------------------------------------
class AuditState(State):
    """
    Internal state of the environment for the current episode.
    """
    task_id: str = ""
    total_errors: int = 0
    found_errors: int = 0
    false_positives: int = 0
    # Enhanced tracking
    investigation_mode: bool = False
    revealed_categories: List[str] = Field(default_factory=list)
    cumulative_score: float = 0.01


class WorldState(BaseModel):
    """Evolving state of the financial world across campaign periods."""

    fiscal_period: int = 1
    policy_version: int = 1
    tax_rates: Dict[str, float] = Field(
        default_factory=lambda: {
            "gst_standard": 0.18,
            "gst_reduced": 0.12,
            "gst_low": 0.05,
        }
    )
    schema_version: int = 1
    schema_changes: Dict[str, str] = Field(
        default_factory=dict,
        description="Field renames: {'old_name': 'new_name'}",
    )
    vendor_status_changes: Dict[str, str] = Field(
        default_factory=dict,
        description="Vendor compliance changes: {'V-001': 'blacklisted'}",
    )
    policy_updates: List[str] = Field(
        default_factory=list,
        description="Policy change descriptions active this period",
    )
    active_alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Unresolved alerts from prior periods",
    )
    mid_period_rule_drops: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="NEW rules injected mid-period (regulatory shock)",
    )


class CampaignState(BaseModel):
    """State of a multi-period audit campaign."""

    campaign_id: str = ""
    seed: int = 42
    total_periods: int = 5
    current_period: int = 1
    world_state: WorldState = Field(default_factory=WorldState)
    agent_roles: Dict[str, AgentRole] = Field(default_factory=dict)
    period_scores: List[Dict[str, Any]] = Field(default_factory=list)
    findings_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All findings from all prior periods",
    )
    instruction_compliance: Dict[str, bool] = Field(default_factory=dict)
    budget_remaining: float = Field(default=100.0)
    specialist_consecutive_tasks: Dict[str, int] = Field(default_factory=dict)
    regulatory_shocks_applied: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Track which regulatory shocks have been applied and when",
    )


class OverseerDecision(BaseModel):
    """Single overseer decision on a specialist finding."""

    finding_ref: str = Field(..., description="Reference: 'document_id:error_type'")
    verdict: str = Field(..., description="approve | reject | escalate")
    reason_code: str = Field(..., description="Reason for decision")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ConflictResolution(BaseModel):
    """Resolution of conflicting findings between specialists."""

    agents: List[str] = Field(..., description="Agent roles that conflicted")
    finding: str = Field(..., description="The disputed finding ref")
    resolution: str = Field(..., description="approve | reject")
    reason: str = Field(..., description="Justification")


class OverseerAction(Action):
    """Action submitted by the overseer agent."""

    audit_trail_id: str = Field(..., description="Unique review action ID")
    decisions: List[OverseerDecision] = Field(default_factory=list)
    conflicts_resolved: List[ConflictResolution] = Field(default_factory=list)
    task_reassignments: Dict[str, str] = Field(default_factory=dict)


class CriticReport(BaseModel):
    """Structured analysis of campaign failures."""

    missed_error_types: List[str] = Field(default_factory=list)
    high_fp_agents: List[str] = Field(default_factory=list)
    cross_period_gaps: List[str] = Field(default_factory=list)
    checklist_updates: List[str] = Field(default_factory=list)
    overall_diagnosis: str = ""


class CampaignObservation(BaseModel):
    """Campaign-level observation wrapping period-level data."""

    campaign_id: str = ""
    current_period: int = 1
    total_periods: int = 5
    world_state: WorldState = Field(default_factory=WorldState)
    period_observation: Optional[AuditObservation] = None
    findings_history_summary: List[Dict[str, Any]] = Field(default_factory=list)
    active_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    budget_remaining: float = 100.0
    specialist_assignments: Dict[str, str] = Field(default_factory=dict)
    pending_regulatory_shocks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="New rules dropped mid-period that agent must now apply",
    )
