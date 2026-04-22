"""
Frozen instruction set for Round 2.
22 base instructions + 3 regulatory shock instructions = 25 total.
Each is binary-checkable. IDs are frozen.
"""

from typing import Any, Dict, List

INSTRUCTIONS = [
    # BUCKET 1: Policy Rules (8)
    {
        "id": "POL-001",
        "bucket": "policy",
        "text": "Meal expenses above Rs 1,500 require manager approval evidence.",
        "severity": 1.0,
        "active_from_period": 1,
    },
    {
        "id": "POL-002",
        "bucket": "policy",
        "text": "Travel expenses must have matching receipt_id. Missing receipt = violation.",
        "severity": 1.0,
        "active_from_period": 1,
    },
    {
        "id": "POL-003",
        "bucket": "policy",
        "text": "Weekend business expenses require pre-approval documentation.",
        "severity": 0.8,
        "active_from_period": 1,
    },
    {
        "id": "POL-004",
        "bucket": "policy",
        "text": "Cumulative monthly expense per employee must not exceed Rs 25,000.",
        "severity": 1.2,
        "active_from_period": 1,
    },
    {
        "id": "POL-005",
        "bucket": "policy",
        "text": "All invoices above Rs 50,000 require dual approval.",
        "severity": 1.5,
        "active_from_period": 1,
    },
    {
        "id": "POL-006",
        "bucket": "policy",
        "text": "Vendor must be on approved vendor list. Unapproved vendor = flag.",
        "severity": 1.0,
        "active_from_period": 1,
    },
    {
        "id": "POL-007",
        "bucket": "policy",
        "text": "In period 2+, meal expense limit increases to Rs 2,000 (policy update).",
        "severity": 1.0,
        "active_from_period": 2,
    },
    {
        "id": "POL-008",
        "bucket": "policy",
        "text": "Three-way match tolerance is Rs 1 for rounding. Differences above Rs 1 must be flagged.",
        "severity": 0.8,
        "active_from_period": 1,
    },
    # BUCKET 2: Dependencies (4)
    {
        "id": "DEP-001",
        "bucket": "dependency",
        "text": "GST reconciliation must not begin until invoice matching is complete.",
        "severity": 1.5,
        "active_from_period": 1,
    },
    {
        "id": "DEP-002",
        "bucket": "dependency",
        "text": "Fraud detection must consider findings from all prior completed tasks.",
        "severity": 1.2,
        "active_from_period": 1,
    },
    {
        "id": "DEP-003",
        "bucket": "dependency",
        "text": "Overseer review must occur after all specialists submit for the period.",
        "severity": 1.0,
        "active_from_period": 1,
    },
    {
        "id": "DEP-004",
        "bucket": "dependency",
        "text": "Findings from period N must be carried forward as context for period N+1.",
        "severity": 1.5,
        "active_from_period": 2,
    },
    # BUCKET 3: Deadlines (3)
    {
        "id": "DL-001",
        "bucket": "deadline",
        "text": "All specialist findings must be submitted within max_steps for that task.",
        "severity": 1.0,
        "active_from_period": 1,
    },
    {
        "id": "DL-002",
        "bucket": "deadline",
        "text": "Overseer review must be submitted within 2 steps of specialist completion.",
        "severity": 0.8,
        "active_from_period": 1,
    },
    {
        "id": "DL-003",
        "bucket": "deadline",
        "text": "Campaign must complete all periods. Incomplete = -0.20 penalty.",
        "severity": 1.5,
        "active_from_period": 1,
    },
    # BUCKET 4: Risk Escalation (4)
    {
        "id": "ESC-001",
        "bucket": "escalation",
        "text": "Any fraud finding must be escalated to overseer.",
        "severity": 2.0,
        "active_from_period": 1,
    },
    {
        "id": "ESC-002",
        "bucket": "escalation",
        "text": "Findings involving amounts above Rs 1,00,000 must be flagged high-severity.",
        "severity": 1.5,
        "active_from_period": 1,
    },
    {
        "id": "ESC-003",
        "bucket": "escalation",
        "text": "If vendor was flagged in prior period, all their current-period transactions must be reviewed.",
        "severity": 1.5,
        "active_from_period": 2,
    },
    {
        "id": "ESC-004",
        "bucket": "escalation",
        "text": "Cross-agent agreement (2+ agents flag same issue) auto-escalates to overseer.",
        "severity": 1.0,
        "active_from_period": 1,
    },
    # BUCKET 5: Schema Adaptation (3)
    {
        "id": "SCH-001",
        "bucket": "schema",
        "text": "In period 3+, field 'vendor_gstin' is renamed to 'supplier_gstin' in purchase register.",
        "severity": 1.0,
        "active_from_period": 3,
    },
    {
        "id": "SCH-002",
        "bucket": "schema",
        "text": "In period 3+, field 'hsn_description' is added to purchase register entries.",
        "severity": 0.5,
        "active_from_period": 3,
    },
    {
        "id": "SCH-003",
        "bucket": "schema",
        "text": "Agent must not reference old field names after schema change.",
        "severity": 0.8,
        "active_from_period": 3,
    },
]

REGULATORY_SHOCKS = [
    {
        "id": "REG-001",
        "text": "URGENT: GST rate for IT services (HSN 998314) changed from 18% to 12% effective immediately. Re-check all IT service invoices in current period.",
        "severity": 1.5,
        "trigger_period": 3,
        "trigger_after_step": 2,
        "affected_tasks": ["gst_reconciliation"],
        "rule_change": {"tax_rate": {"998314": 0.12}},
    },
    {
        "id": "REG-002",
        "text": "NEW POLICY: All cash/UPI payments above Rs 20,000 now require additional documentation. Flag any without it.",
        "severity": 1.2,
        "trigger_period": 4,
        "trigger_after_step": 1,
        "affected_tasks": ["expense_audit", "fraud_detection"],
        "rule_change": {"cash_upi_threshold": 20000},
    },
    {
        "id": "REG-003",
        "text": "COMPLIANCE ALERT: Vendors incorporated less than 6 months ago are now classified as high-risk. All transactions must be flagged for review.",
        "severity": 1.5,
        "trigger_period": 4,
        "trigger_after_step": 3,
        "affected_tasks": ["fraud_detection"],
        "rule_change": {"new_vendor_risk_months": 6},
    },
]

assert len(INSTRUCTIONS) == 22, f"Expected 22 instructions, got {len(INSTRUCTIONS)}"
assert len(REGULATORY_SHOCKS) == 3, f"Expected 3 regulatory shocks, got {len(REGULATORY_SHOCKS)}"

INSTRUCTION_BY_ID = {inst["id"]: inst for inst in INSTRUCTIONS + REGULATORY_SHOCKS}

INSTRUCTIONS_BY_BUCKET: Dict[str, List[Dict[str, Any]]] = {}
for inst in INSTRUCTIONS:
    INSTRUCTIONS_BY_BUCKET.setdefault(inst["bucket"], []).append(inst)


def get_active_instructions(period: int) -> List[Dict[str, Any]]:
    """Return base instructions active for the given period."""
    return [i for i in INSTRUCTIONS if i["active_from_period"] <= period]


def get_pending_regulatory_shocks(period: int, step: int) -> List[Dict[str, Any]]:
    """
    Return regulatory shocks that should drop at this point.

    A shock triggers when:
      - period matches trigger_period
      - step > trigger_after_step
    """
    return [
        shock
        for shock in REGULATORY_SHOCKS
        if shock["trigger_period"] == period and step > shock["trigger_after_step"]
    ]
