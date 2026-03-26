# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — OpenEnv compatible RL environment
# for training AI agents on real-world financial auditing tasks.

"""
Financial Audit Environment

An OpenEnv-compatible environment that simulates real-world financial
auditing tasks: expense policy enforcement, invoice three-way matching,
and GST return reconciliation.

Usage:
    from financial_audit_env import AuditAction, AuditObservation, FinancialAuditEnv

    with FinancialAuditEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset(task_id="expense_audit")
        result = env.step(AuditAction(findings=[...], submit_final=True))
"""

from .models import AuditAction, AuditObservation, AuditState, Finding
from .client import FinancialAuditEnv

__all__ = [
    "AuditAction",
    "AuditObservation",
    "AuditState",
    "Finding",
    "FinancialAuditEnv",
]
