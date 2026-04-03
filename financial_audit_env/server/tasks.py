# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — Task definitions.
#
# Each task defines:
#   - A unique ID and human-readable name
#   - Difficulty level (easy / medium / hard / expert)
#   - A description that the agent reads to understand what to audit
#   - Maximum steps allowed before the episode auto-ends
#   - The set of valid error_types the agent can report
#   - The data generator function name to use
#
# Tasks are designed for a clear difficulty progression:
#   Easy   → single-document policy checks (expense claims)
#   Medium → cross-document matching (PO vs GRN vs Invoice)
#   Hard   → cross-system reconciliation (Books vs GSTR-2B)
#   Expert → pattern recognition across many transactions (fraud detection)

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Task 1: Expense Policy Violation Detection (Easy)
# ---------------------------------------------------------------------------
TASK_EXPENSE_AUDIT = {
    "id": "expense_audit",
    "name": "Expense Policy Violation Detection",
    "difficulty": "easy",
    "description": (
        "You are an internal auditor reviewing employee expense claims against "
        "the company's expense policy.\n\n"
        "Your job is to identify expenses that violate company policy. "
        "You will be given:\n"
        "1. A list of expense claims with fields: expense_id, date, employee, "
        "category, amount, description, receipt_id, vendor\n"
        "2. The company's expense policy with rules and limits\n\n"
        "For each violation found, report:\n"
        "- document_id: the expense_id of the violating entry\n"
        "- error_type: one of the allowed violation types\n"
        "- description: explain what the violation is\n"
        "- suggested_fix: what action should be taken\n\n"
        "IMPORTANT: Some entries may look suspicious but are actually compliant.\n"
        "For example, an expense exactly AT the limit is NOT a violation.\n"
        "A Friday evening expense is NOT a weekend expense.\n"
        "Same amount on different receipts is NOT a duplicate.\n\n"
        "Be thorough — find ALL violations. But be precise — false positives "
        "will lower your score."
    ),
    "max_steps": 5,
    "error_types": [
        "over_limit",         # Amount exceeds category limit
        "wrong_category",     # Expense miscategorized
        "duplicate_claim",    # Same receipt submitted twice
        "weekend_expense",    # Business expense on weekend without approval
        "missing_receipt",    # No receipt for expense over threshold
        "unapproved_vendor",  # Vendor not on approved list
        "cumulative_breach",  # Monthly total exceeds cumulative limit
    ],
    "generator": "generate_expense_data",
}

# ---------------------------------------------------------------------------
# Task 2: Invoice Three-Way Match Audit (Medium)
# ---------------------------------------------------------------------------
TASK_INVOICE_MATCH = {
    "id": "invoice_match",
    "name": "Invoice Three-Way Match Audit",
    "difficulty": "medium",
    "description": (
        "You are a procurement auditor performing a three-way match between "
        "Purchase Orders (POs), Goods Receipt Notes (GRNs), and Vendor Invoices.\n\n"
        "The three-way match ensures:\n"
        "- What was ordered (PO) matches what was received (GRN)\n"
        "- What was invoiced matches what was ordered and received\n"
        "- Prices, quantities, tax rates, and totals are consistent\n\n"
        "You will be given:\n"
        "1. Purchase Orders: po_id, vendor, items with unit_price and quantity\n"
        "2. Goods Receipt Notes: grn_id, po_id, items received with quantities\n"
        "3. Vendor Invoices: invoice_id, po_id, vendor, line items, tax, total\n\n"
        "For each discrepancy found, report:\n"
        "- document_id: the invoice_id (or po_id/grn_id if relevant)\n"
        "- error_type: one of the allowed discrepancy types\n"
        "- description: explain the discrepancy with specific numbers\n"
        "- suggested_fix: recommended resolution\n\n"
        "NOTE: Minor rounding differences (< ₹1) are acceptable and should NOT "
        "be flagged. GRN quantities may legitimately differ from PO if the "
        "invoice matches the GRN (partial delivery is normal).\n\n"
        "Watch for cascading errors: a price mismatch can cause the total to "
        "also be wrong — flag both the root cause AND the cascading effect."
    ),
    "max_steps": 8,
    "error_types": [
        "price_mismatch",      # Invoice price ≠ PO agreed price
        "quantity_mismatch",   # Invoiced qty ≠ received qty
        "duplicate_invoice",   # Two invoices for same PO/line
        "unmatched_invoice",   # Invoice with no corresponding PO
        "tax_error",           # Incorrect GST rate applied
        "total_mismatch",      # Line items don't sum to total
        "vendor_mismatch",     # Invoice vendor ≠ PO vendor
        "date_anomaly",        # Invoice dated before PO or GRN
        "cascading_total",     # Total wrong due to upstream price/qty error
    ],
    "generator": "generate_invoice_data",
}

# ---------------------------------------------------------------------------
# Task 3: GST Return Reconciliation (Hard)
# ---------------------------------------------------------------------------
TASK_GST_RECONCILIATION = {
    "id": "gst_reconciliation",
    "name": "GST Return Reconciliation",
    "difficulty": "hard",
    "description": (
        "You are a tax auditor reconciling a business's Purchase Register "
        "(internal books) against GSTR-2B data (auto-generated by the "
        "government GST portal based on suppliers' filings).\n\n"
        "Mismatches between books and GSTR-2B can cause:\n"
        "- Loss of Input Tax Credit (ITC) if claimed but not in GSTR-2B\n"
        "- Missed ITC if in GSTR-2B but not recorded in books\n"
        "- Penalties for incorrect claims\n\n"
        "You will be given:\n"
        "1. Purchase Register entries: invoice_no, vendor_name, vendor_gstin, "
        "date, taxable_value, cgst, sgst, igst, total, hsn_code, place_of_supply\n"
        "2. GSTR-2B entries: supplier_gstin, invoice_no, date, taxable_value, "
        "igst, cgst, sgst, total\n\n"
        "For each mismatch found, report:\n"
        "- document_id: the invoice_no from either dataset\n"
        "- error_type: one of the allowed mismatch types\n"
        "- description: explain the mismatch with specific values\n"
        "- suggested_fix: recommended action\n\n"
        "Key rules:\n"
        "- Intra-state supply uses CGST + SGST; inter-state uses IGST\n"
        "- GSTIN format: 2-digit state code + 10-char PAN + 1 entity + 1 check digit\n"
        "- ITC cannot be claimed for invoices older than 180 days\n"
        "- Certain categories (food & beverages, personal vehicles) have blocked ITC\n\n"
        "NOTE: Minor date differences (< 30 days) are normal and NOT errors.\n"
        "Taxable value differences of ₹10 or less are rounding and NOT errors."
    ),
    "max_steps": 10,
    "error_types": [
        "missing_in_gstr2b",
        "missing_in_books",
        "amount_mismatch",
        "tax_rate_mismatch",
        "gstin_mismatch",
        "date_mismatch",
        "duplicate_entry",
        "invalid_gstin",
        "wrong_tax_type",
        "itc_ineligible",
        "late_claim",
        "excess_claim",
    ],
    "generator": "generate_gst_data",
}

# ---------------------------------------------------------------------------
# Task 4: Fraud Pattern Detection (Expert)
# ---------------------------------------------------------------------------
TASK_FRAUD_DETECTION = {
    "id": "fraud_detection",
    "name": "Fraud Pattern Detection",
    "difficulty": "expert",
    "description": (
        "You are a forensic auditor analyzing transaction data to detect "
        "potential fraud patterns. This requires recognizing statistical "
        "anomalies, relationship patterns, and behavioral red flags across "
        "many transactions.\n\n"
        "You will be given:\n"
        "1. Transaction records: txn_id, date, vendor_id, vendor_name, amount, "
        "gst details, payment_mode, approved_by, bank_account\n"
        "2. Vendor registry: vendor_id, name, gstin, incorporation_date, "
        "bank_account, bank_name, category\n"
        "3. Audit context: period, thresholds, expected distributions\n\n"
        "Look for these fraud patterns:\n"
        "- Circular invoicing: A→B→C→A payment chains between related vendors\n"
        "- Split invoices: Large amounts split to stay below approval threshold\n"
        "- Shell companies: Newly incorporated vendors with shared PAN/bank details\n"
        "- Round number anomaly: Suspicious perfectly round invoice amounts\n"
        "- Benford's law violation: Leading digit distribution is statistically unlikely\n"
        "- Vendor concentration: Excessive payment % to a single vendor\n"
        "- Duplicate bank accounts: Different vendors sharing same bank account\n"
        "- Volume spikes: Sudden dramatic increase in vendor transaction frequency\n"
        "- Pre-incorporation invoices: Invoices dated before vendor was incorporated\n"
        "- Weekend patterns: Invoices consistently dated on non-business days\n\n"
        "For each fraud pattern found, report:\n"
        "- document_id: the txn_id of the most representative transaction\n"
        "- error_type: one of the allowed fraud types\n"
        "- description: explain the pattern with specific evidence\n"
        "- suggested_fix: recommended investigation action\n\n"
        "NOTE: Not every unusual pattern is fraud. Focus on clear, evidence-based "
        "findings. A single round number is not suspicious; a pattern of them is."
    ),
    "max_steps": 12,
    "error_types": [
        "circular_invoicing",
        "split_invoice",
        "shell_company",
        "round_number_anomaly",
        "benford_violation",
        "vendor_concentration",
        "duplicate_bank_account",
        "sudden_volume_spike",
        "invoice_before_incorporation",
        "weekend_pattern",
    ],
    "generator": "generate_fraud_data",
}


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
TASKS: Dict[str, Dict[str, Any]] = {
    "expense_audit": TASK_EXPENSE_AUDIT,
    "invoice_match": TASK_INVOICE_MATCH,
    "gst_reconciliation": TASK_GST_RECONCILIATION,
    "fraud_detection": TASK_FRAUD_DETECTION,
}

# Default tasks for baseline (excludes expert-level to avoid tanking average score)
DEFAULT_TASK_IDS = ["expense_audit", "invoice_match", "gst_reconciliation"]


def get_task(task_id: str) -> Dict[str, Any]:
    """Retrieve a task definition by ID."""
    if task_id not in TASKS:
        valid = ", ".join(TASKS.keys())
        raise ValueError(
            f"Unknown task_id: '{task_id}'. Valid options: {valid}"
        )
    return TASKS[task_id]


def get_all_tasks_summary() -> List[Dict[str, Any]]:
    """Return a summary of all tasks for the /tasks endpoint."""
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "difficulty": t["difficulty"],
            "description": t["description"],
            "max_steps": t["max_steps"],
            "error_types": t["error_types"],
            "action_schema": {
                "findings": [
                    {
                        "document_id": "string (required)",
                        "field": "string (optional)",
                        "error_type": f"string — one of {t['error_types']}",
                        "description": "string (required)",
                        "suggested_fix": "string (optional)",
                    }
                ],
                "submit_final": "boolean (default: false)",
            },
        }
        for t in TASKS.values()
    ]
