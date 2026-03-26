# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — Pydantic models for the OpenEnv interface.
#
# These models define the typed contracts between the agent and the environment:
#   - Finding: A single audit issue identified by the agent
#   - AuditAction: What the agent submits each step
#   - AuditObservation: What the agent sees after each step
#   - AuditState: Internal episode metadata
#
# All models use Pydantic strict mode for input validation (OWASP best practice).

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Attempt to import OpenEnv base classes. Falls back to standalone definitions
# if openenv-core is not installed (for local development / GitHub usage).
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Standalone fallback — allows running without openenv-core installed
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


def _sanitize_string(value: str, max_length: int = MAX_STRING_LENGTH) -> str:
    """
    Sanitize a user-provided string:
    - Strip leading/trailing whitespace
    - Truncate to max_length
    - Remove null bytes and control characters (except newlines/tabs)

    This prevents injection attacks and excessive memory usage.
    """
    if not isinstance(value, str):
        return value
    # Remove null bytes
    value = value.replace("\x00", "")
    # Remove control chars except \n \r \t
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
                     e.g., "EXP-007", "INV-003", "BOOK-015"
        field:       Optional — which specific field has the error.
                     e.g., "amount", "gst_rate", "vendor_gstin"
        error_type:  The category of error found. Must match one of the
                     allowed error_types defined in the task.
                     e.g., "over_limit", "price_mismatch", "missing_in_gstr2b"
        description: Human-readable explanation of the finding.
        suggested_fix: Optional recommended corrective action.
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

    @field_validator("document_id", "error_type", mode="before")
    @classmethod
    def sanitize_ids(cls, v: str) -> str:
        """Sanitize identifier fields — strip whitespace, remove dangerous chars."""
        return _sanitize_string(v, MAX_DOCUMENT_ID_LENGTH)

    @field_validator("description", "suggested_fix", mode="before")
    @classmethod
    def sanitize_text(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize free-text fields."""
        if v is None:
            return None
        return _sanitize_string(v, MAX_STRING_LENGTH)


# ---------------------------------------------------------------------------
# AuditAction — what the agent submits each step
# ---------------------------------------------------------------------------
class AuditAction(Action):
    """
    Action submitted by the agent at each step.

    The agent can submit findings incrementally across multiple steps,
    or all at once with submit_final=True.

    Attributes:
        findings:     List of Finding objects for this step.
                      Maximum 50 findings per step to prevent abuse.
        submit_final: If True, the episode ends and the final grader
                      score is computed. If False, findings are accumulated
                      and the agent gets feedback before continuing.
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
# AuditObservation — what the agent sees after each step
# ---------------------------------------------------------------------------
class AuditObservation(Observation):
    """
    Observation returned to the agent after reset() or step().

    Contains the financial documents to audit, the task description,
    accumulated findings with feedback, and episode progress info.

    Attributes:
        task_id:          Which task is active (expense_audit, invoice_match, etc.)
        task_description: Natural language description of what to look for
        documents:        The financial data to audit — structure depends on task:
                          - expense_audit: {"expenses": [...], "policy": {...}}
                          - invoice_match: {"purchase_orders": [...], "grns": [...], "invoices": [...]}
                          - gst_reconciliation: {"books": [...], "gstr2b": [...]}
        findings_so_far:  Previously submitted findings (across all steps)
        feedback:         Environment's response to the last action
        step_number:      Current step (1-indexed)
        max_steps:        Maximum steps allowed for this task
    """
    task_id: str = ""
    task_description: str = ""
    documents: Dict[str, Any] = Field(default_factory=dict)
    findings_so_far: List[Finding] = Field(default_factory=list)
    feedback: str = ""
    step_number: int = 0
    max_steps: int = 0


# ---------------------------------------------------------------------------
# AuditState — internal episode metadata
# ---------------------------------------------------------------------------
class AuditState(State):
    """
    Internal state of the environment for the current episode.

    Extends the OpenEnv State base class (which provides episode_id
    and step_count) with audit-specific tracking fields.

    Note: total_errors is NOT exposed to the agent — it's used
    internally for grading only.
    """
    task_id: str = ""
    total_errors: int = 0
    found_errors: int = 0
    false_positives: int = 0
