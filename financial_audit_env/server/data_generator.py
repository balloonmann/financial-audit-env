# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — Synthetic data generator.
#
# Generates realistic financial data with planted errors for each task.
# All generators accept a `seed` parameter for full reproducibility —
# same seed always produces identical data and identical planted errors.
#
# Security: Generated data never includes real PII or financial info.
# All names, GSTINs, and amounts are synthetic.
#
# v2 Changes:
#   - Fixed datetime.now() → REFERENCE_DATE for true reproducibility
#   - Added noise/red herring entries that look suspicious but are correct
#   - Scaled up GST task (40+12 entries)
#   - Added multi-document cross-referencing errors
#   - Added cascading errors to invoice task
#   - Added fraud detection generator (Task 4)

import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# REPRODUCIBILITY FIX: Fixed reference date instead of datetime.now()
# This ensures same seed = identical data regardless of when you run it.
# ---------------------------------------------------------------------------
REFERENCE_DATE = datetime(2026, 1, 15)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

# Realistic Indian vendor/employee names for synthetic data
VENDOR_NAMES = [
    "Reliance Office Supplies", "Tata Stationery Co.", "Infosys Catering",
    "Wipro Tech Solutions", "Mahindra Logistics", "HCL Furniture",
    "Bajaj Electronics", "Godrej Interiors", "Larsen & Toubro Services",
    "Adani Power Solutions", "Bharti Telecom", "Sun Pharma Supplies",
    "ITC Foods", "Asian Paints", "Hindustan Unilever Services",
    "Tech Mahindra Solutions", "JSW Steel Trading", "Vedanta Resources",
    "ONGC Services", "BPCL Petroleum",
]

EMPLOYEE_NAMES = [
    "Rajesh Kumar", "Priya Sharma", "Amit Patel", "Sunita Gupta",
    "Vikram Singh", "Neha Verma", "Arjun Reddy", "Kavita Iyer",
    "Rahul Mehta", "Deepa Nair", "Ananya Krishnan", "Rohit Malhotra",
    "Sanjay Deshmukh", "Meera Pillai", "Karthik Subramaniam",
]

EXPENSE_CATEGORIES = [
    "Travel", "Meals", "Office Supplies", "Software", "Equipment",
    "Training", "Communication", "Accommodation", "Transport", "Miscellaneous",
]

# Valid Indian state codes for GSTIN generation
STATE_CODES = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
    "32", "33", "34", "35", "36", "37",
]

HSN_CODES = [
    "8471", "3926", "8443", "9403", "8528",  # Electronics, plastic, printer, furniture, monitor
    "4820", "8523", "8504", "9405", "7326",  # Notebook, media, transformer, lamp, metal articles
]

# Monetary amounts for fraud severity tracking (used by graders)
ERROR_MONETARY_VALUES = {
    # Expense errors
    "over_limit": 3000,
    "wrong_category": 15000,
    "duplicate_claim": 5000,
    "weekend_expense": 1200,
    "missing_receipt": 3500,
    "unapproved_vendor": 2800,
    "cumulative_breach": 50000,
    # Invoice errors
    "price_mismatch": 25000,
    "quantity_mismatch": 15000,
    "duplicate_invoice": 50000,
    "unmatched_invoice": 59000,
    "tax_error": 10000,
    "total_mismatch": 7500,
    "vendor_mismatch": 30000,
    "date_anomaly": 20000,
    "cascading_total": 5000,
    # GST errors
    "missing_in_gstr2b": 4500,
    "missing_in_books": 3240,
    "amount_mismatch": 5000,
    "tax_rate_mismatch": 1000,
    "gstin_mismatch": 25000,
    "date_mismatch": 15000,
    "duplicate_entry": 20000,
    "invalid_gstin": 14160,
    "wrong_tax_type": 18000,
    "itc_ineligible": 400,
    "late_claim": 2700,
    "excess_claim": 4000,
    # Fraud errors
    "circular_invoicing": 150000,
    "split_invoice": 95000,
    "shell_company": 200000,
    "round_number_anomaly": 50000,
    "benford_violation": 75000,
    "vendor_concentration": 500000,
    "duplicate_bank_account": 100000,
    "sudden_volume_spike": 300000,
    "invoice_before_incorporation": 80000,
    "weekend_pattern": 60000,
}

# Severity weights for grading (1.0 = standard, >1.0 = critical, <1.0 = minor)
ERROR_SEVERITY_WEIGHTS = {
    # Expense (lower severity — internal policy issues)
    "over_limit": 0.8,
    "wrong_category": 0.9,
    "duplicate_claim": 1.2,
    "weekend_expense": 0.5,
    "missing_receipt": 0.7,
    "unapproved_vendor": 0.8,
    "cumulative_breach": 1.3,
    # Invoice (medium severity — financial discrepancies)
    "price_mismatch": 1.3,
    "quantity_mismatch": 1.2,
    "duplicate_invoice": 1.5,
    "unmatched_invoice": 1.4,
    "tax_error": 1.1,
    "total_mismatch": 1.0,
    "vendor_mismatch": 1.2,
    "date_anomaly": 0.9,
    "cascading_total": 0.8,
    # GST (high severity — regulatory compliance)
    "missing_in_gstr2b": 1.4,
    "missing_in_books": 1.3,
    "amount_mismatch": 1.2,
    "tax_rate_mismatch": 1.1,
    "gstin_mismatch": 1.5,
    "date_mismatch": 0.9,
    "duplicate_entry": 1.3,
    "invalid_gstin": 1.5,
    "wrong_tax_type": 1.2,
    "itc_ineligible": 1.0,
    "late_claim": 1.1,
    "excess_claim": 1.4,
    # Fraud (highest severity — potential criminal)
    "circular_invoicing": 2.0,
    "split_invoice": 1.8,
    "shell_company": 2.0,
    "round_number_anomaly": 1.3,
    "benford_violation": 1.4,
    "vendor_concentration": 1.5,
    "duplicate_bank_account": 1.8,
    "sudden_volume_spike": 1.6,
    "invoice_before_incorporation": 1.7,
    "weekend_pattern": 1.2,
}


def _generate_gstin(rng: random.Random, state_code: str = "27") -> str:
    """
    Generate a realistic-looking GSTIN (Goods and Services Tax ID Number).
    Format: SS-AAAA-A-0000-A-Z-A (15 chars)
    """
    pan_alpha = "".join(rng.choices(string.ascii_uppercase, k=5))
    pan_digits = "".join(rng.choices(string.digits, k=4))
    pan_check = rng.choice(string.ascii_uppercase)
    entity = str(rng.randint(1, 9))
    check = rng.choice(string.ascii_uppercase)
    return f"{state_code}{pan_alpha}{pan_digits}{pan_check}{entity}{check}"


def _random_date(rng: random.Random, start_days_ago: int = 90, end_days_ago: int = 1) -> str:
    """Generate a random date string (YYYY-MM-DD) within a range of days ago from REFERENCE_DATE."""
    days_ago = rng.randint(end_days_ago, start_days_ago)
    date = REFERENCE_DATE - timedelta(days=days_ago)
    return date.strftime("%Y-%m-%d")


def _random_old_date(rng: random.Random) -> str:
    """Generate a date older than 180 days from REFERENCE_DATE (for late_claim errors)."""
    days_ago = rng.randint(200, 365)
    date = REFERENCE_DATE - timedelta(days=days_ago)
    return date.strftime("%Y-%m-%d")


def _generate_bank_account(rng: random.Random) -> str:
    """Generate a realistic-looking Indian bank account number."""
    return "".join(rng.choices(string.digits, k=rng.choice([11, 12, 14, 16])))


# ---------------------------------------------------------------------------
# Task 1: Expense data generator
# ---------------------------------------------------------------------------

EXPENSE_POLICY = {
    "company": "Acme Financial Services Pvt. Ltd.",
    "effective_date": "2025-01-01",
    "rules": {
        "Travel": {"daily_limit": 5000, "receipt_required_above": 500},
        "Meals": {"daily_limit": 1500, "receipt_required_above": 500},
        "Office Supplies": {"daily_limit": 10000, "receipt_required_above": 500},
        "Software": {"daily_limit": 50000, "receipt_required_above": 0},
        "Equipment": {"daily_limit": 100000, "receipt_required_above": 0},
        "Training": {"daily_limit": 25000, "receipt_required_above": 500},
        "Communication": {"daily_limit": 2000, "receipt_required_above": 500},
        "Accommodation": {"daily_limit": 8000, "receipt_required_above": 0},
        "Transport": {"daily_limit": 3000, "receipt_required_above": 500},
        "Miscellaneous": {"daily_limit": 2000, "receipt_required_above": 500},
    },
    "general_rules": [
        "Weekend expenses require prior written approval from manager",
        "Maximum 1 expense claim per receipt",
        "All vendors must be from the approved vendor list",
        "Personal expenses cannot be claimed under any category",
        "Monthly cumulative claims per employee must not exceed ₹50,000",
    ],
    "approved_vendors": [
        "Reliance Office Supplies", "Tata Stationery Co.",
        "Wipro Tech Solutions", "Mahindra Logistics", "HCL Furniture",
        "Bajaj Electronics", "Godrej Interiors", "ITC Foods",
        "Sun Pharma Supplies", "Asian Paints",
    ],
}


def generate_expense_data(seed: int = 42) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """
    Generate expense claim data with 7 planted policy violations + red herrings.
    """
    rng = random.Random(seed)
    expenses: List[Dict[str, Any]] = []
    ground_truth: List[Dict[str, Any]] = []

    # Generate 9 clean expense entries
    for i in range(1, 10):
        category = rng.choice(EXPENSE_CATEGORIES[:6])
        limit = EXPENSE_POLICY["rules"][category]["daily_limit"]
        amount = rng.randint(200, int(limit * 0.8))
        date = REFERENCE_DATE - timedelta(days=rng.randint(5, 60))
        while date.weekday() >= 5:
            date -= timedelta(days=1)

        expenses.append({
            "expense_id": f"EXP-{i:03d}",
            "date": date.strftime("%Y-%m-%d"),
            "employee": rng.choice(EMPLOYEE_NAMES),
            "category": category,
            "amount": amount,
            "description": f"{category} expense for project work",
            "receipt_id": f"RCP-{rng.randint(10000, 99999)}",
            "vendor": rng.choice(EXPENSE_POLICY["approved_vendors"]),
        })

    # --- RED HERRINGS (suspicious but correct) ---

    # Herring 1: Expense at EXACTLY the limit (legal)
    expenses.append({
        "expense_id": "EXP-H01",
        "date": (REFERENCE_DATE - timedelta(days=rng.randint(5, 30))).strftime("%Y-%m-%d"),
        "employee": "Rahul Mehta",
        "category": "Meals",
        "amount": 1500,  # Exactly at limit — NOT a violation
        "description": "Client lunch meeting at hotel restaurant",
        "receipt_id": f"RCP-{rng.randint(10000, 99999)}",
        "vendor": "ITC Foods",
    })

    # Herring 2: Friday evening expense (close to weekend, but weekday)
    friday = REFERENCE_DATE - timedelta(days=rng.randint(7, 30))
    while friday.weekday() != 4:  # Friday
        friday -= timedelta(days=1)
    expenses.append({
        "expense_id": "EXP-H02",
        "date": friday.strftime("%Y-%m-%d"),
        "employee": "Arjun Reddy",
        "category": "Meals",
        "amount": 1400,
        "description": "Late Friday team dinner after sprint close",
        "receipt_id": f"RCP-{rng.randint(10000, 99999)}",
        "vendor": "ITC Foods",
    })

    # Herring 3: Same amount as another entry but different receipt (not duplicate)
    expenses.append({
        "expense_id": "EXP-H03",
        "date": expenses[4]["date"],
        "employee": expenses[4]["employee"],
        "category": expenses[4]["category"],
        "amount": expenses[4]["amount"],  # Same amount!
        "description": "Follow-up purchase, separate transaction",
        "receipt_id": f"RCP-{rng.randint(10000, 99999)}",  # DIFFERENT receipt
        "vendor": expenses[4]["vendor"],
    })

    # --- PLANT 7 VIOLATIONS ---

    # Violation 1: over_limit — meal expense way over ₹1500 limit
    exp_id = "EXP-010"
    expenses.append({
        "expense_id": exp_id,
        "date": _random_date(rng, 30, 5),
        "employee": "Rajesh Kumar",
        "category": "Meals",
        "amount": 4500,
        "description": "Team dinner at premium restaurant",
        "receipt_id": f"RCP-{rng.randint(10000, 99999)}",
        "vendor": "ITC Foods",
    })
    ground_truth.append({"document_id": exp_id, "error_type": "over_limit"})

    # Violation 2: wrong_category — personal electronics as Office Supplies
    exp_id = "EXP-011"
    expenses.append({
        "expense_id": exp_id,
        "date": _random_date(rng, 30, 5),
        "employee": "Priya Sharma",
        "category": "Office Supplies",
        "amount": 15000,
        "description": "Personal wireless earbuds - Sony WH-1000XM5",
        "receipt_id": f"RCP-{rng.randint(10000, 99999)}",
        "vendor": "Bajaj Electronics",
    })
    ground_truth.append({"document_id": exp_id, "error_type": "wrong_category"})

    # Violation 3: duplicate_claim — same receipt_id as EXP-003
    dup_receipt = expenses[2]["receipt_id"]
    exp_id = "EXP-012"
    expenses.append({
        "expense_id": exp_id,
        "date": expenses[2]["date"],
        "employee": expenses[2]["employee"],
        "category": expenses[2]["category"],
        "amount": expenses[2]["amount"],
        "description": expenses[2]["description"],
        "receipt_id": dup_receipt,
        "vendor": expenses[2]["vendor"],
    })
    ground_truth.append({"document_id": exp_id, "error_type": "duplicate_claim"})

    # Violation 4: weekend_expense — expense on a Saturday
    exp_id = "EXP-013"
    weekend_date = REFERENCE_DATE - timedelta(days=rng.randint(7, 30))
    while weekend_date.weekday() != 5:
        weekend_date -= timedelta(days=1)
    expenses.append({
        "expense_id": exp_id,
        "date": weekend_date.strftime("%Y-%m-%d"),
        "employee": "Vikram Singh",
        "category": "Meals",
        "amount": 1200,
        "description": "Working lunch during weekend shift",
        "receipt_id": f"RCP-{rng.randint(10000, 99999)}",
        "vendor": "ITC Foods",
    })
    ground_truth.append({"document_id": exp_id, "error_type": "weekend_expense"})

    # Violation 5: missing_receipt — high-value expense with no receipt
    exp_id = "EXP-014"
    expenses.append({
        "expense_id": exp_id,
        "date": _random_date(rng, 30, 5),
        "employee": "Amit Patel",
        "category": "Travel",
        "amount": 3500,
        "description": "Cab fare to client site",
        "receipt_id": "",
        "vendor": "Mahindra Logistics",
    })
    ground_truth.append({"document_id": exp_id, "error_type": "missing_receipt"})

    # Violation 6: unapproved_vendor
    exp_id = "EXP-015"
    expenses.append({
        "expense_id": exp_id,
        "date": _random_date(rng, 30, 5),
        "employee": "Neha Verma",
        "category": "Office Supplies",
        "amount": 2800,
        "description": "Printer cartridges",
        "receipt_id": f"RCP-{rng.randint(10000, 99999)}",
        "vendor": "QuickPrint Unofficial Store",
    })
    ground_truth.append({"document_id": exp_id, "error_type": "unapproved_vendor"})

    # Violation 7: cumulative_breach — employee total exceeds ₹50K monthly
    # Rajesh Kumar already has EXP-010 (₹4500). Add more to push over ₹50K.
    exp_id = "EXP-016"
    expenses.append({
        "expense_id": exp_id,
        "date": _random_date(rng, 30, 5),
        "employee": "Rajesh Kumar",
        "category": "Equipment",
        "amount": 48000,
        "description": "Ergonomic standing desk for home office",
        "receipt_id": f"RCP-{rng.randint(10000, 99999)}",
        "vendor": "HCL Furniture",
    })
    ground_truth.append({"document_id": exp_id, "error_type": "cumulative_breach"})

    # Shuffle so violations aren't bunched at the end
    rng.shuffle(expenses)

    documents = {
        "expenses": expenses,
        "policy": EXPENSE_POLICY,
    }
    return documents, ground_truth


# ---------------------------------------------------------------------------
# Task 2: Invoice three-way match data generator
# ---------------------------------------------------------------------------

def generate_invoice_data(seed: int = 42) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """
    Generate PO, GRN, and Invoice data with 9 planted discrepancies
    (including 1 cascading error) + red herrings.
    """
    rng = random.Random(seed)
    ground_truth = []

    # --- Generate 10 Purchase Orders ---
    purchase_orders: List[Dict[str, Any]] = []
    for i in range(1, 11):
        num_items = rng.randint(1, 3)
        items = []
        for j in range(num_items):
            items.append({
                "item_name": f"Item-{rng.choice(['A', 'B', 'C', 'D', 'E'])}{rng.randint(1,99)}",
                "unit_price": rng.choice([500, 1000, 1500, 2000, 2500, 3000, 5000]),
                "quantity": rng.randint(5, 50),
                "hsn_code": rng.choice(HSN_CODES),
            })
        vendor = rng.choice(VENDOR_NAMES[:8])
        purchase_orders.append({
            "po_id": f"PO-{i:03d}",
            "date": _random_date(rng, 90, 30),
            "vendor": vendor,
            "vendor_gstin": _generate_gstin(rng, "27"),
            "items": items,
            "total": sum(it["unit_price"] * it["quantity"] for it in items),
            "gst_rate": 18,
            "status": "approved",
        })

    # --- Generate 10 GRNs (one per PO, mostly matching) ---
    grns: List[Dict[str, Any]] = []
    for i, po in enumerate(purchase_orders, 1):
        received_items = []
        for item in po["items"]:
            received_items.append({
                "item_name": item["item_name"],
                "quantity_received": item["quantity"],
                "condition": "good",
            })
        grns.append({
            "grn_id": f"GRN-{i:03d}",
            "po_id": po["po_id"],
            "date": _random_date(rng, 29, 5),
            "items_received": received_items,
            "received_by": rng.choice(EMPLOYEE_NAMES),
        })

    # --- Generate 12 Invoices (10 matching POs + 2 extra) ---
    invoices: List[Dict[str, Any]] = []
    for i, po in enumerate(purchase_orders, 1):
        line_items = []
        for item in po["items"]:
            line_items.append({
                "item_name": item["item_name"],
                "unit_price": item["unit_price"],
                "quantity": item["quantity"],
                "amount": item["unit_price"] * item["quantity"],
            })
        subtotal = sum(li["amount"] for li in line_items)
        gst_amount = subtotal * po["gst_rate"] / 100
        invoices.append({
            "invoice_id": f"INV-{i:03d}",
            "po_id": po["po_id"],
            "date": _random_date(rng, 28, 3),
            "vendor": po["vendor"],
            "vendor_gstin": po["vendor_gstin"],
            "line_items": line_items,
            "subtotal": subtotal,
            "gst_rate": po["gst_rate"],
            "gst_amount": gst_amount,
            "total": subtotal + gst_amount,
        })

    # --- RED HERRINGS ---

    # Herring 1: Minor rounding difference (±₹0.50, acceptable in practice)
    invoices[0]["total"] = invoices[0]["total"] + 0.50

    # Herring 2: GRN received 1 less than PO, but invoice matches GRN (correct behavior)
    grns[2]["items_received"][0]["quantity_received"] = purchase_orders[2]["items"][0]["quantity"] - 1
    invoices[2]["line_items"][0]["quantity"] = grns[2]["items_received"][0]["quantity_received"]
    invoices[2]["line_items"][0]["amount"] = invoices[2]["line_items"][0]["unit_price"] * invoices[2]["line_items"][0]["quantity"]
    invoices[2]["subtotal"] = sum(li["amount"] for li in invoices[2]["line_items"])
    invoices[2]["gst_amount"] = invoices[2]["subtotal"] * 18 / 100
    invoices[2]["total"] = invoices[2]["subtotal"] + invoices[2]["gst_amount"]

    # --- PLANT 9 DISCREPANCIES ---

    # Discrepancy 1: price_mismatch — INV-002 has inflated unit price
    inv = invoices[1]
    inv["line_items"][0]["unit_price"] = inv["line_items"][0]["unit_price"] + 500
    inv["line_items"][0]["amount"] = inv["line_items"][0]["unit_price"] * inv["line_items"][0]["quantity"]
    ground_truth.append({"document_id": "INV-002", "error_type": "price_mismatch"})

    # Discrepancy 2: quantity_mismatch — INV-004 invoices more than GRN received
    inv = invoices[3]
    inv["line_items"][0]["quantity"] = inv["line_items"][0]["quantity"] + 10
    inv["line_items"][0]["amount"] = inv["line_items"][0]["unit_price"] * inv["line_items"][0]["quantity"]
    grns[3]["items_received"][0]["quantity_received"] = purchase_orders[3]["items"][0]["quantity"]
    ground_truth.append({"document_id": "INV-004", "error_type": "quantity_mismatch"})

    # Discrepancy 3: duplicate_invoice — INV-011 is duplicate of INV-005
    dup_inv = invoices[4].copy()
    dup_inv["invoice_id"] = "INV-011"
    dup_inv["date"] = _random_date(rng, 20, 3)
    invoices.append(dup_inv)
    ground_truth.append({"document_id": "INV-011", "error_type": "duplicate_invoice"})

    # Discrepancy 4: unmatched_invoice — INV-012 has no corresponding PO
    invoices.append({
        "invoice_id": "INV-012",
        "po_id": "PO-999",
        "date": _random_date(rng, 20, 3),
        "vendor": "Unknown Vendor Ltd.",
        "vendor_gstin": _generate_gstin(rng, "27"),
        "line_items": [{"item_name": "Mystery Item", "unit_price": 10000, "quantity": 5, "amount": 50000}],
        "subtotal": 50000,
        "gst_rate": 18,
        "gst_amount": 9000,
        "total": 59000,
    })
    ground_truth.append({"document_id": "INV-012", "error_type": "unmatched_invoice"})

    # Discrepancy 5: tax_error — INV-006 has wrong GST rate (12% instead of 18%)
    inv = invoices[5]
    inv["gst_rate"] = 12
    inv["gst_amount"] = inv["subtotal"] * 12 / 100
    inv["total"] = inv["subtotal"] + inv["gst_amount"]
    ground_truth.append({"document_id": "INV-006", "error_type": "tax_error"})

    # Discrepancy 6: total_mismatch — INV-007 total doesn't match line items
    inv = invoices[6]
    correct_total = inv["total"]
    inv["total"] = correct_total + 7500
    ground_truth.append({"document_id": "INV-007", "error_type": "total_mismatch"})

    # Discrepancy 7: vendor_mismatch — INV-008 vendor differs from PO vendor
    inv = invoices[7]
    inv["vendor"] = "Larsen & Toubro Services"
    ground_truth.append({"document_id": "INV-008", "error_type": "vendor_mismatch"})

    # Discrepancy 8: date_anomaly — INV-009 dated BEFORE its PO
    inv = invoices[8]
    po_date = datetime.strptime(purchase_orders[8]["date"], "%Y-%m-%d")
    inv["date"] = (po_date - timedelta(days=15)).strftime("%Y-%m-%d")
    ground_truth.append({"document_id": "INV-009", "error_type": "date_anomaly"})

    # Discrepancy 9: cascading_total — INV-002 already has price_mismatch,
    # its subtotal/total now don't match line items (cascading effect)
    # The root cause was price_mismatch, but the total is also wrong.
    inv = invoices[1]
    correct_subtotal = sum(li["amount"] for li in inv["line_items"])
    if inv["subtotal"] != correct_subtotal:
        ground_truth.append({"document_id": "INV-002", "error_type": "cascading_total"})

    documents = {
        "purchase_orders": purchase_orders,
        "grns": grns,
        "invoices": invoices,
    }
    return documents, ground_truth


# ---------------------------------------------------------------------------
# Task 3: GST reconciliation data generator (scaled up)
# ---------------------------------------------------------------------------

def generate_gst_data(seed: int = 42) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """
    Generate purchase register and GSTR-2B data with 12 planted mismatches.
    Scaled up to 40 clean entries + 12 error entries for lower error density.
    """
    rng = random.Random(seed)
    ground_truth = []

    vendors: List[Dict[str, str]] = []
    for i in range(20):
        state = rng.choice(["27", "29", "06", "33", "07"])
        gstin = _generate_gstin(rng, state)
        vendors.append({
            "name": VENDOR_NAMES[i] if i < len(VENDOR_NAMES) else f"Vendor-{i}",
            "gstin": gstin,
            "state": state,
        })

    our_state = "27"

    # --- Generate 40 clean book entries + matching GSTR-2B entries ---
    books: List[Dict[str, Any]] = []
    gstr2b: List[Dict[str, Any]] = []

    for i in range(1, 41):
        vendor = rng.choice(vendors)
        is_interstate = vendor["state"] != our_state
        taxable_value = rng.choice([5000, 10000, 15000, 20000, 25000, 30000, 50000])
        gst_rate = rng.choice([5, 12, 18, 28])

        if is_interstate:
            igst = round(taxable_value * gst_rate / 100, 2)
            cgst, sgst = 0, 0
        else:
            cgst = round(taxable_value * gst_rate / 200, 2)
            sgst = cgst
            igst = 0

        inv_date = _random_date(rng, 90, 10)
        inv_no = f"VINV-{i:04d}"

        books.append({
            "invoice_no": inv_no,
            "vendor_name": vendor["name"],
            "vendor_gstin": vendor["gstin"],
            "date": inv_date,
            "taxable_value": taxable_value,
            "gst_rate": gst_rate,
            "cgst": cgst,
            "sgst": sgst,
            "igst": igst,
            "total": taxable_value + cgst + sgst + igst,
            "hsn_code": rng.choice(HSN_CODES),
            "place_of_supply": vendor["state"],
        })

        gstr2b.append({
            "supplier_gstin": vendor["gstin"],
            "invoice_no": inv_no,
            "date": inv_date,
            "taxable_value": taxable_value,
            "cgst": cgst,
            "sgst": sgst,
            "igst": igst,
            "total": taxable_value + cgst + sgst + igst,
        })

    # --- RED HERRINGS ---

    # Herring 1: Date difference of 3 days (acceptable, < 30 days)
    gstr2b[5]["date"] = (datetime.strptime(books[5]["date"], "%Y-%m-%d") + timedelta(days=3)).strftime("%Y-%m-%d")

    # Herring 2: Taxable value off by ₹1 (rounding, acceptable)
    gstr2b[10]["taxable_value"] = books[10]["taxable_value"] + 1
    gstr2b[10]["total"] = gstr2b[10]["taxable_value"] + gstr2b[10]["cgst"] + gstr2b[10]["sgst"] + gstr2b[10]["igst"]

    # Herring 3: GSTIN that looks suspicious (repeated digits) but is valid
    # (already generated deterministically, just noting it's not an error)

    # --- PLANT 12 MISMATCHES ---

    # 1. missing_in_gstr2b
    inv_no = "VINV-0041"
    vendor = rng.choice(vendors)
    books.append({
        "invoice_no": inv_no,
        "vendor_name": vendor["name"],
        "vendor_gstin": vendor["gstin"],
        "date": _random_date(rng, 60, 10),
        "taxable_value": 25000,
        "gst_rate": 18,
        "cgst": 2250, "sgst": 2250, "igst": 0,
        "total": 29500,
        "hsn_code": "8471",
        "place_of_supply": "27",
    })
    ground_truth.append({"document_id": inv_no, "error_type": "missing_in_gstr2b"})

    # 2. missing_in_books
    inv_no = "VINV-0042"
    vendor = rng.choice(vendors)
    gstr2b.append({
        "supplier_gstin": vendor["gstin"],
        "invoice_no": inv_no,
        "date": _random_date(rng, 60, 10),
        "taxable_value": 18000,
        "cgst": 1620, "sgst": 1620, "igst": 0,
        "total": 21240,
    })
    ground_truth.append({"document_id": inv_no, "error_type": "missing_in_books"})

    # 3. amount_mismatch
    idx = 2
    gstr2b[idx]["taxable_value"] = books[idx]["taxable_value"] + 5000
    gstr2b[idx]["total"] = gstr2b[idx]["taxable_value"] + gstr2b[idx]["cgst"] + gstr2b[idx]["sgst"] + gstr2b[idx]["igst"]
    ground_truth.append({"document_id": books[idx]["invoice_no"], "error_type": "amount_mismatch"})

    # 4. tax_rate_mismatch
    idx = 4
    books[idx]["cgst"] = books[idx]["cgst"] + 500
    books[idx]["sgst"] = books[idx]["sgst"] + 500
    books[idx]["total"] = books[idx]["taxable_value"] + books[idx]["cgst"] + books[idx]["sgst"] + books[idx]["igst"]
    ground_truth.append({"document_id": books[idx]["invoice_no"], "error_type": "tax_rate_mismatch"})

    # 5. gstin_mismatch
    idx = 6
    gstr2b[idx]["supplier_gstin"] = _generate_gstin(rng, "27")
    ground_truth.append({"document_id": books[idx]["invoice_no"], "error_type": "gstin_mismatch"})

    # 6. date_mismatch (>30 days)
    idx = 8
    orig_date = datetime.strptime(books[idx]["date"], "%Y-%m-%d")
    gstr2b[idx]["date"] = (orig_date + timedelta(days=45)).strftime("%Y-%m-%d")
    ground_truth.append({"document_id": books[idx]["invoice_no"], "error_type": "date_mismatch"})

    # 7. duplicate_entry
    dup_entry = books[15].copy()
    dup_entry_inv = books[15]["invoice_no"]
    books.append(dup_entry)
    ground_truth.append({"document_id": dup_entry_inv, "error_type": "duplicate_entry"})

    # 8. invalid_gstin
    inv_no = "VINV-0043"
    books.append({
        "invoice_no": inv_no,
        "vendor_name": "Shady Corp",
        "vendor_gstin": "XXINVALID123",
        "date": _random_date(rng, 60, 10),
        "taxable_value": 12000,
        "gst_rate": 18,
        "cgst": 1080, "sgst": 1080, "igst": 0,
        "total": 14160,
        "hsn_code": "3926",
        "place_of_supply": "27",
    })
    gstr2b.append({
        "supplier_gstin": "XXINVALID123",
        "invoice_no": inv_no,
        "date": books[-1]["date"],
        "taxable_value": 12000,
        "cgst": 1080, "sgst": 1080, "igst": 0,
        "total": 14160,
    })
    ground_truth.append({"document_id": inv_no, "error_type": "invalid_gstin"})

    # 9. wrong_tax_type — IGST for intra-state
    idx = 18
    books[idx]["igst"] = books[idx]["cgst"] + books[idx]["sgst"]
    books[idx]["cgst"] = 0
    books[idx]["sgst"] = 0
    books[idx]["place_of_supply"] = our_state
    ground_truth.append({"document_id": books[idx]["invoice_no"], "error_type": "wrong_tax_type"})

    # 10. itc_ineligible — blocked category
    inv_no = "VINV-0044"
    vendor = rng.choice(vendors)
    books.append({
        "invoice_no": inv_no,
        "vendor_name": "ITC Foods",
        "vendor_gstin": vendor["gstin"],
        "date": _random_date(rng, 60, 10),
        "taxable_value": 8000,
        "gst_rate": 5,
        "cgst": 200, "sgst": 200, "igst": 0,
        "total": 8400,
        "hsn_code": "2106",
        "place_of_supply": "27",
        "description": "Catering services for office party",
    })
    gstr2b.append({
        "supplier_gstin": vendor["gstin"],
        "invoice_no": inv_no,
        "date": books[-1]["date"],
        "taxable_value": 8000,
        "cgst": 200, "sgst": 200, "igst": 0,
        "total": 8400,
    })
    ground_truth.append({"document_id": inv_no, "error_type": "itc_ineligible"})

    # 11. late_claim — invoice older than 180 days
    inv_no = "VINV-0045"
    vendor = rng.choice(vendors)
    old_date = _random_old_date(rng)
    books.append({
        "invoice_no": inv_no,
        "vendor_name": vendor["name"],
        "vendor_gstin": vendor["gstin"],
        "date": old_date,
        "taxable_value": 15000,
        "gst_rate": 18,
        "cgst": 1350, "sgst": 1350, "igst": 0,
        "total": 17700,
        "hsn_code": "8443",
        "place_of_supply": "27",
    })
    gstr2b.append({
        "supplier_gstin": vendor["gstin"],
        "invoice_no": inv_no,
        "date": old_date,
        "taxable_value": 15000,
        "cgst": 1350, "sgst": 1350, "igst": 0,
        "total": 17700,
    })
    ground_truth.append({"document_id": inv_no, "error_type": "late_claim"})

    # 12. excess_claim
    idx = 20
    books[idx]["cgst"] = books[idx]["cgst"] + 2000
    books[idx]["sgst"] = books[idx]["sgst"] + 2000
    books[idx]["total"] = books[idx]["taxable_value"] + books[idx]["cgst"] + books[idx]["sgst"] + books[idx]["igst"]
    ground_truth.append({"document_id": books[idx]["invoice_no"], "error_type": "excess_claim"})

    rng.shuffle(books)

    context = {
        "our_state_code": our_state,
        "our_gstin": _generate_gstin(rng, our_state),
        "current_date": REFERENCE_DATE.strftime("%Y-%m-%d"),
        "itc_blocked_categories": [
            "Food and beverages (except for restaurants/caterers)",
            "Membership of clubs, health/fitness centres",
            "Personal vehicles and related expenses",
            "Life/health insurance (unless mandatory)",
        ],
        "itc_time_limit_days": 180,
        "gstin_format": "2-digit state code + 10-char PAN + entity code + check digit (15 chars total)",
    }

    documents = {
        "purchase_register": books,
        "gstr2b": gstr2b,
        "context": context,
    }
    return documents, ground_truth


# ---------------------------------------------------------------------------
# Task 4: Fraud pattern detection data generator
# ---------------------------------------------------------------------------

def generate_fraud_data(seed: int = 42) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """
    Generate transaction data with 10 planted fraud patterns.
    Expert-level difficulty — requires pattern recognition across many transactions.
    """
    rng = random.Random(seed)
    ground_truth = []

    # Generate vendor registry
    vendor_registry: List[Dict[str, Any]] = []
    for i in range(25):
        state = rng.choice(["27", "29", "06", "33", "07", "09", "24"])
        incorporation_date = REFERENCE_DATE - timedelta(days=rng.randint(365, 3650))
        vendor_registry.append({
            "vendor_id": f"V-{i+1:03d}",
            "name": VENDOR_NAMES[i] if i < len(VENDOR_NAMES) else f"Enterprise Solutions {i}",
            "gstin": _generate_gstin(rng, state),
            "state": state,
            "incorporation_date": incorporation_date.strftime("%Y-%m-%d"),
            "bank_account": _generate_bank_account(rng),
            "bank_name": rng.choice(["SBI", "HDFC", "ICICI", "Axis", "PNB", "Kotak"]),
            "category": rng.choice(["IT Services", "Office Supplies", "Logistics",
                                     "Consulting", "Manufacturing", "Catering"]),
        })

    # Generate 50 clean transactions
    transactions: List[Dict[str, Any]] = []
    for i in range(1, 51):
        vendor = rng.choice(vendor_registry[:15])  # Use first 15 vendors for clean data
        amount = rng.choice([5000, 8000, 12000, 15000, 22000, 35000, 48000, 67000])
        # Add natural variation to amounts (not round numbers)
        amount += rng.randint(-499, 499)
        txn_date = REFERENCE_DATE - timedelta(days=rng.randint(5, 90))
        while txn_date.weekday() >= 5:
            txn_date -= timedelta(days=1)

        gst_rate = rng.choice([5, 12, 18])
        gst_amount = round(amount * gst_rate / 100, 2)

        transactions.append({
            "txn_id": f"TXN-{i:04d}",
            "date": txn_date.strftime("%Y-%m-%d"),
            "vendor_id": vendor["vendor_id"],
            "vendor_name": vendor["name"],
            "vendor_gstin": vendor["gstin"],
            "description": f"{vendor['category']} services",
            "amount": amount,
            "gst_rate": gst_rate,
            "gst_amount": gst_amount,
            "total": amount + gst_amount,
            "payment_mode": rng.choice(["NEFT", "RTGS", "UPI", "Cheque"]),
            "approved_by": rng.choice(EMPLOYEE_NAMES[:8]),
            "bank_account": vendor["bank_account"],
        })

    # --- PLANT 10 FRAUD PATTERNS ---

    # Fraud 1: circular_invoicing — A→B→C→A chain
    # V-016 invoices V-017, V-017 invoices V-018, V-018 invoices V-016
    circ_vendors = vendor_registry[15:18]
    for j, (src, dst) in enumerate([(0, 1), (1, 2), (2, 0)]):
        txn_id = f"TXN-F{j+1:02d}"
        transactions.append({
            "txn_id": txn_id,
            "date": _random_date(rng, 30, 5),
            "vendor_id": circ_vendors[src]["vendor_id"],
            "vendor_name": circ_vendors[src]["name"],
            "vendor_gstin": circ_vendors[src]["gstin"],
            "description": "Consulting services - project advisory",
            "amount": 75000 + rng.randint(-1000, 1000),
            "gst_rate": 18,
            "gst_amount": 13500,
            "total": 88500,
            "payment_mode": "NEFT",
            "approved_by": "Rajesh Kumar",
            "bank_account": circ_vendors[src]["bank_account"],
        })
    ground_truth.append({"document_id": "TXN-F01", "error_type": "circular_invoicing"})

    # Fraud 2: split_invoice — ₹95K split into two invoices just under ₹50K threshold
    split_vendor = vendor_registry[18]
    split_date = _random_date(rng, 20, 5)
    transactions.append({
        "txn_id": "TXN-F04",
        "date": split_date,
        "vendor_id": split_vendor["vendor_id"],
        "vendor_name": split_vendor["name"],
        "vendor_gstin": split_vendor["gstin"],
        "description": "IT infrastructure upgrade - Phase 1",
        "amount": 48000,
        "gst_rate": 18,
        "gst_amount": 8640,
        "total": 56640,
        "payment_mode": "NEFT",
        "approved_by": "Vikram Singh",
        "bank_account": split_vendor["bank_account"],
    })
    transactions.append({
        "txn_id": "TXN-F05",
        "date": split_date,
        "vendor_id": split_vendor["vendor_id"],
        "vendor_name": split_vendor["name"],
        "vendor_gstin": split_vendor["gstin"],
        "description": "IT infrastructure upgrade - Phase 2",
        "amount": 47000,
        "gst_rate": 18,
        "gst_amount": 8460,
        "total": 55460,
        "payment_mode": "NEFT",
        "approved_by": "Vikram Singh",
        "bank_account": split_vendor["bank_account"],
    })
    ground_truth.append({"document_id": "TXN-F04", "error_type": "split_invoice"})

    # Fraud 3: shell_company — two vendors with same PAN (similar GSTINs)
    base_gstin = vendor_registry[19]["gstin"]
    shell_gstin = base_gstin[:2] + base_gstin[2:12] + "2" + base_gstin[13:]  # Same PAN, different entity
    shell_vendor = {
        "vendor_id": "V-021",
        "name": "Pinnacle Business Solutions",
        "gstin": shell_gstin,
        "incorporation_date": (REFERENCE_DATE - timedelta(days=60)).strftime("%Y-%m-%d"),
    }
    vendor_registry.append(shell_vendor)
    transactions.append({
        "txn_id": "TXN-F06",
        "date": _random_date(rng, 30, 5),
        "vendor_id": "V-021",
        "vendor_name": shell_vendor["name"],
        "vendor_gstin": shell_gstin,
        "description": "Business consulting and advisory",
        "amount": 200000,
        "gst_rate": 18,
        "gst_amount": 36000,
        "total": 236000,
        "payment_mode": "RTGS",
        "approved_by": "Amit Patel",
        "bank_account": vendor_registry[19]["bank_account"],  # Same bank account as original!
    })
    ground_truth.append({"document_id": "TXN-F06", "error_type": "shell_company"})

    # Fraud 4: round_number_anomaly — multiple exactly round invoices
    round_vendor = vendor_registry[20]
    for j, amt in enumerate([10000, 25000, 50000]):
        transactions.append({
            "txn_id": f"TXN-F{7+j:02d}",
            "date": _random_date(rng, 60, 5),
            "vendor_id": round_vendor["vendor_id"],
            "vendor_name": round_vendor["name"],
            "vendor_gstin": round_vendor["gstin"],
            "description": f"Maintenance services batch {j+1}",
            "amount": amt,  # Perfectly round — suspicious
            "gst_rate": 18,
            "gst_amount": amt * 18 // 100,
            "total": amt + amt * 18 // 100,
            "payment_mode": "NEFT",
            "approved_by": rng.choice(EMPLOYEE_NAMES[:5]),
            "bank_account": round_vendor["bank_account"],
        })
    ground_truth.append({"document_id": "TXN-F07", "error_type": "round_number_anomaly"})

    # Fraud 5: benford_violation — amounts starting with 9 (statistically unlikely)
    benford_vendor = vendor_registry[21]
    for j in range(4):
        transactions.append({
            "txn_id": f"TXN-F{10+j:02d}",
            "date": _random_date(rng, 60, 5),
            "vendor_id": benford_vendor["vendor_id"],
            "vendor_name": benford_vendor["name"],
            "vendor_gstin": benford_vendor["gstin"],
            "description": f"Supply of materials order {j+1}",
            "amount": rng.choice([91000, 93500, 96000, 98500]),  # All start with 9
            "gst_rate": 18,
            "gst_amount": 17000,
            "total": 108000,
            "payment_mode": "RTGS",
            "approved_by": "Priya Sharma",
            "bank_account": benford_vendor["bank_account"],
        })
    ground_truth.append({"document_id": "TXN-F10", "error_type": "benford_violation"})

    # Fraud 6: vendor_concentration — 80%+ of monthly payments to one vendor
    conc_vendor = vendor_registry[22]
    for j in range(8):
        transactions.append({
            "txn_id": f"TXN-F{14+j:02d}",
            "date": _random_date(rng, 30, 5),
            "vendor_id": conc_vendor["vendor_id"],
            "vendor_name": conc_vendor["name"],
            "vendor_gstin": conc_vendor["gstin"],
            "description": f"Recurring service contract payment {j+1}",
            "amount": 60000 + rng.randint(-5000, 5000),
            "gst_rate": 18,
            "gst_amount": 10800,
            "total": 70800,
            "payment_mode": "NEFT",
            "approved_by": "Sunita Gupta",
            "bank_account": conc_vendor["bank_account"],
        })
    ground_truth.append({"document_id": "TXN-F14", "error_type": "vendor_concentration"})

    # Fraud 7: duplicate_bank_account — two different vendors share bank account
    dup_bank = vendor_registry[23]
    dup_bank["bank_account"] = vendor_registry[0]["bank_account"]  # Same as V-001
    transactions.append({
        "txn_id": "TXN-F22",
        "date": _random_date(rng, 30, 5),
        "vendor_id": dup_bank["vendor_id"],
        "vendor_name": dup_bank["name"],
        "vendor_gstin": dup_bank["gstin"],
        "description": "Professional services - audit support",
        "amount": 85000,
        "gst_rate": 18,
        "gst_amount": 15300,
        "total": 100300,
        "payment_mode": "NEFT",
        "approved_by": "Kavita Iyer",
        "bank_account": dup_bank["bank_account"],
    })
    ground_truth.append({"document_id": "TXN-F22", "error_type": "duplicate_bank_account"})

    # Fraud 8: sudden_volume_spike — vendor had 1 txn in prev quarter, 10 this month
    spike_vendor = vendor_registry[24]
    # One old transaction
    transactions.append({
        "txn_id": "TXN-F23",
        "date": (REFERENCE_DATE - timedelta(days=85)).strftime("%Y-%m-%d"),
        "vendor_id": spike_vendor["vendor_id"],
        "vendor_name": spike_vendor["name"],
        "vendor_gstin": spike_vendor["gstin"],
        "description": "One-time consulting",
        "amount": 15000,
        "gst_rate": 18,
        "gst_amount": 2700,
        "total": 17700,
        "payment_mode": "NEFT",
        "approved_by": "Rohit Malhotra",
        "bank_account": spike_vendor["bank_account"],
    })
    # 6 recent transactions (volume spike)
    for j in range(6):
        transactions.append({
            "txn_id": f"TXN-F{24+j:02d}",
            "date": _random_date(rng, 15, 1),
            "vendor_id": spike_vendor["vendor_id"],
            "vendor_name": spike_vendor["name"],
            "vendor_gstin": spike_vendor["gstin"],
            "description": f"Urgent procurement order {j+1}",
            "amount": 45000 + rng.randint(-3000, 3000),
            "gst_rate": 18,
            "gst_amount": 8100,
            "total": 53100,
            "payment_mode": "RTGS",
            "approved_by": "Rahul Mehta",
            "bank_account": spike_vendor["bank_account"],
        })
    ground_truth.append({"document_id": "TXN-F24", "error_type": "sudden_volume_spike"})

    # Fraud 9: invoice_before_incorporation — invoice dated before vendor was incorporated
    new_vendor = vendor_registry[19]  # Incorporation date is known
    incorp_date = datetime.strptime(new_vendor["incorporation_date"], "%Y-%m-%d")
    transactions.append({
        "txn_id": "TXN-F30",
        "date": (incorp_date - timedelta(days=30)).strftime("%Y-%m-%d"),  # Before incorporation!
        "vendor_id": new_vendor["vendor_id"],
        "vendor_name": new_vendor["name"],
        "vendor_gstin": new_vendor["gstin"],
        "description": "Pre-launch consulting services",
        "amount": 120000,
        "gst_rate": 18,
        "gst_amount": 21600,
        "total": 141600,
        "payment_mode": "RTGS",
        "approved_by": "Sanjay Deshmukh",
        "bank_account": new_vendor["bank_account"],
    })
    ground_truth.append({"document_id": "TXN-F30", "error_type": "invoice_before_incorporation"})

    # Fraud 10: weekend_pattern — invoices consistently on weekends
    wknd_vendor = vendor_registry[16]
    for j in range(4):
        wknd_date = REFERENCE_DATE - timedelta(days=7 * (j + 1))
        while wknd_date.weekday() not in (5, 6):
            wknd_date += timedelta(days=1)
        transactions.append({
            "txn_id": f"TXN-F{31+j:02d}",
            "date": wknd_date.strftime("%Y-%m-%d"),
            "vendor_id": wknd_vendor["vendor_id"],
            "vendor_name": wknd_vendor["name"],
            "vendor_gstin": wknd_vendor["gstin"],
            "description": f"Weekend delivery service batch {j+1}",
            "amount": 35000 + rng.randint(-2000, 2000),
            "gst_rate": 18,
            "gst_amount": 6300,
            "total": 41300,
            "payment_mode": "NEFT",
            "approved_by": "Meera Pillai",
            "bank_account": wknd_vendor["bank_account"],
        })
    ground_truth.append({"document_id": "TXN-F31", "error_type": "weekend_pattern"})

    rng.shuffle(transactions)

    documents = {
        "transactions": transactions,
        "vendor_registry": vendor_registry,
        "audit_context": {
            "audit_period_start": (REFERENCE_DATE - timedelta(days=90)).strftime("%Y-%m-%d"),
            "audit_period_end": REFERENCE_DATE.strftime("%Y-%m-%d"),
            "invoice_approval_threshold": 50000,
            "benford_expected_distribution": {
                "1": 0.301, "2": 0.176, "3": 0.125, "4": 0.097,
                "5": 0.079, "6": 0.067, "7": 0.058, "8": 0.051, "9": 0.046,
            },
            "max_vendor_concentration_pct": 30,
            "weekend_invoice_threshold": 2,
            "volume_spike_multiplier": 3,
        },
    }
    return documents, ground_truth


# ---------------------------------------------------------------------------
# Generator dispatcher
# ---------------------------------------------------------------------------

GENERATORS = {
    "generate_expense_data": generate_expense_data,
    "generate_invoice_data": generate_invoice_data,
    "generate_gst_data": generate_gst_data,
    "generate_fraud_data": generate_fraud_data,
}


def generate_data_for_task(generator_name: str, seed: int = 42) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Dispatch to the correct data generator."""
    if generator_name not in GENERATORS:
        raise ValueError(f"Unknown generator: {generator_name}")
    return GENERATORS[generator_name](seed)
