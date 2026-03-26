# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — Synthetic data generator.
#
# Generates realistic financial data with planted errors for each task.
# All generators accept a `seed` parameter for full reproducibility —
# same seed always produces identical data and identical planted errors.
#
# Security: Generated data never includes real PII or financial info.
# All names, GSTINs, and amounts are synthetic.

import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple


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
]

EMPLOYEE_NAMES = [
    "Rajesh Kumar", "Priya Sharma", "Amit Patel", "Sunita Gupta",
    "Vikram Singh", "Neha Verma", "Arjun Reddy", "Kavita Iyer",
    "Rahul Mehta", "Deepa Nair",
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


def _generate_gstin(rng: random.Random, state_code: str = "27") -> str:
    """
    Generate a realistic-looking GSTIN (Goods and Services Tax ID Number).

    Format: SS-AAAA-A-0000-A-Z-A (15 chars)
    SS = state code, next 10 = PAN-like, last 2 = entity + check digit
    """
    pan_alpha = "".join(rng.choices(string.ascii_uppercase, k=5))
    pan_digits = "".join(rng.choices(string.digits, k=4))
    pan_check = rng.choice(string.ascii_uppercase)
    entity = str(rng.randint(1, 9))
    check = rng.choice(string.ascii_uppercase)
    return f"{state_code}{pan_alpha}{pan_digits}{pan_check}{entity}{check}"


def _random_date(rng: random.Random, start_days_ago: int = 90, end_days_ago: int = 1) -> str:
    """Generate a random date string (YYYY-MM-DD) within a range of days ago."""
    days_ago = rng.randint(end_days_ago, start_days_ago)
    date = datetime.now() - timedelta(days=days_ago)
    return date.strftime("%Y-%m-%d")


def _random_old_date(rng: random.Random) -> str:
    """Generate a date older than 180 days (for late_claim errors)."""
    days_ago = rng.randint(200, 365)
    date = datetime.now() - timedelta(days=days_ago)
    return date.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Task 1: Expense data generator
# ---------------------------------------------------------------------------

# Company expense policy — provided to agent as context
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
    Generate expense claim data with 6 planted policy violations.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - documents: {"expenses": [...], "policy": {...}}
        - ground_truth: list of {"document_id": ..., "error_type": ...}
    """
    rng = random.Random(seed)
    expenses: List[Dict[str, Any]] = []
    ground_truth: List[Dict[str, Any]] = []

    # Generate 9 clean expense entries first
    for i in range(1, 10):
        category = rng.choice(EXPENSE_CATEGORIES[:6])  # Stick to common ones
        limit = EXPENSE_POLICY["rules"][category]["daily_limit"]
        amount = rng.randint(200, int(limit * 0.8))  # Under limit
        # Ensure weekday
        date = datetime.now() - timedelta(days=rng.randint(5, 60))
        while date.weekday() >= 5:  # Skip weekends
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

    # --- Plant 6 violations ---

    # Violation 1: over_limit — meal expense way over ₹1500 limit
    exp_id = "EXP-010"
    expenses.append({
        "expense_id": exp_id,
        "date": _random_date(rng, 30, 5),
        "employee": "Rajesh Kumar",
        "category": "Meals",
        "amount": 4500,  # Limit is 1500
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
    dup_receipt = expenses[2]["receipt_id"]  # Copy receipt from 3rd entry
    exp_id = "EXP-012"
    expenses.append({
        "expense_id": exp_id,
        "date": expenses[2]["date"],
        "employee": expenses[2]["employee"],
        "category": expenses[2]["category"],
        "amount": expenses[2]["amount"],
        "description": expenses[2]["description"],
        "receipt_id": dup_receipt,  # Same receipt!
        "vendor": expenses[2]["vendor"],
    })
    ground_truth.append({"document_id": exp_id, "error_type": "duplicate_claim"})

    # Violation 4: weekend_expense — expense on a Saturday
    exp_id = "EXP-013"
    # Find next Saturday from a reference date
    weekend_date = datetime.now() - timedelta(days=rng.randint(7, 30))
    while weekend_date.weekday() != 5:  # Saturday
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
        "amount": 3500,  # Above 500 threshold
        "description": "Cab fare to client site",
        "receipt_id": "",  # No receipt!
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
        "vendor": "QuickPrint Unofficial Store",  # Not on approved list
    })
    ground_truth.append({"document_id": exp_id, "error_type": "unapproved_vendor"})

    # Shuffle expenses so violations aren't bunched at the end
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
    Generate PO, GRN, and Invoice data with 8 planted discrepancies.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - documents: {"purchase_orders": [...], "grns": [...], "invoices": [...]}
        - ground_truth: list of {"document_id": ..., "error_type": ...}
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
            "gst_rate": 18,  # Standard GST rate
            "status": "approved",
        })

    # --- Generate 10 GRNs (one per PO, mostly matching) ---
    grns: List[Dict[str, Any]] = []
    for i, po in enumerate(purchase_orders, 1):
        received_items = []
        for item in po["items"]:
            received_items.append({
                "item_name": item["item_name"],
                "quantity_received": item["quantity"],  # Default: matches PO
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

    # --- Plant 8 discrepancies ---

    # Discrepancy 1: price_mismatch — INV-002 has inflated unit price
    inv = invoices[1]
    inv["line_items"][0]["unit_price"] = inv["line_items"][0]["unit_price"] + 500
    inv["line_items"][0]["amount"] = inv["line_items"][0]["unit_price"] * inv["line_items"][0]["quantity"]
    # Don't update subtotal/total — creates a cascade but the root cause is price
    ground_truth.append({"document_id": "INV-002", "error_type": "price_mismatch"})

    # Discrepancy 2: quantity_mismatch — INV-004 invoices more than GRN received
    inv = invoices[3]
    inv["line_items"][0]["quantity"] = inv["line_items"][0]["quantity"] + 10
    inv["line_items"][0]["amount"] = inv["line_items"][0]["unit_price"] * inv["line_items"][0]["quantity"]
    grns[3]["items_received"][0]["quantity_received"] = purchase_orders[3]["items"][0]["quantity"]  # GRN correct
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
        "po_id": "PO-999",  # Non-existent PO
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
    inv["gst_rate"] = 12  # Should be 18
    inv["gst_amount"] = inv["subtotal"] * 12 / 100
    inv["total"] = inv["subtotal"] + inv["gst_amount"]
    ground_truth.append({"document_id": "INV-006", "error_type": "tax_error"})

    # Discrepancy 6: total_mismatch — INV-007 total doesn't match line items
    inv = invoices[6]
    correct_total = inv["total"]
    inv["total"] = correct_total + 7500  # Inflated total
    ground_truth.append({"document_id": "INV-007", "error_type": "total_mismatch"})

    # Discrepancy 7: vendor_mismatch — INV-008 vendor differs from PO vendor
    inv = invoices[7]
    inv["vendor"] = "Larsen & Toubro Services"  # Different from PO vendor
    ground_truth.append({"document_id": "INV-008", "error_type": "vendor_mismatch"})

    # Discrepancy 8: date_anomaly — INV-009 dated BEFORE its PO
    inv = invoices[8]
    po_date = datetime.strptime(purchase_orders[8]["date"], "%Y-%m-%d")
    inv["date"] = (po_date - timedelta(days=15)).strftime("%Y-%m-%d")
    ground_truth.append({"document_id": "INV-009", "error_type": "date_anomaly"})

    documents = {
        "purchase_orders": purchase_orders,
        "grns": grns,
        "invoices": invoices,
    }
    return documents, ground_truth


# ---------------------------------------------------------------------------
# Task 3: GST reconciliation data generator
# ---------------------------------------------------------------------------

def generate_gst_data(seed: int = 42) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """
    Generate purchase register and GSTR-2B data with 12 planted mismatches.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - documents: {"purchase_register": [...], "gstr2b": [...], "context": {...}}
        - ground_truth: list of {"document_id": ..., "error_type": ...}
    """
    rng = random.Random(seed)
    ground_truth = []

    vendors: List[Dict[str, str]] = []
    for i in range(15):
        state = rng.choice(["27", "29", "06", "33", "07"])  # MH, KA, HR, TN, DL
        gstin = _generate_gstin(rng, state)
        vendors.append({
            "name": VENDOR_NAMES[i] if i < len(VENDOR_NAMES) else f"Vendor-{i}",
            "gstin": gstin,
            "state": state,
        })

    # Our company is in Maharashtra (state code 27)
    our_state = "27"

    # --- Generate 22 clean book entries + matching GSTR-2B entries ---
    books: List[Dict[str, Any]] = []
    gstr2b: List[Dict[str, Any]] = []

    for i in range(1, 23):
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

    # --- Plant 12 mismatches ---

    # 1. missing_in_gstr2b — book entry with no GSTR-2B match
    inv_no = "VINV-0023"
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
    # Intentionally not adding to gstr2b
    ground_truth.append({"document_id": inv_no, "error_type": "missing_in_gstr2b"})

    # 2. missing_in_books — GSTR-2B entry with no book match
    inv_no = "VINV-0024"
    vendor = rng.choice(vendors)
    gstr2b.append({
        "supplier_gstin": vendor["gstin"],
        "invoice_no": inv_no,
        "date": _random_date(rng, 60, 10),
        "taxable_value": 18000,
        "cgst": 1620, "sgst": 1620, "igst": 0,
        "total": 21240,
    })
    # Intentionally not adding to books
    ground_truth.append({"document_id": inv_no, "error_type": "missing_in_books"})

    # 3. amount_mismatch — taxable_value differs between books and GSTR-2B
    idx = 2  # Modify 3rd entry
    gstr2b[idx]["taxable_value"] = books[idx]["taxable_value"] + 5000
    gstr2b[idx]["total"] = gstr2b[idx]["taxable_value"] + gstr2b[idx]["cgst"] + gstr2b[idx]["sgst"] + gstr2b[idx]["igst"]
    ground_truth.append({"document_id": books[idx]["invoice_no"], "error_type": "amount_mismatch"})

    # 4. tax_rate_mismatch — CGST/SGST differ
    idx = 4
    books[idx]["cgst"] = books[idx]["cgst"] + 500
    books[idx]["sgst"] = books[idx]["sgst"] + 500
    books[idx]["total"] = books[idx]["taxable_value"] + books[idx]["cgst"] + books[idx]["sgst"] + books[idx]["igst"]
    ground_truth.append({"document_id": books[idx]["invoice_no"], "error_type": "tax_rate_mismatch"})

    # 5. gstin_mismatch — vendor GSTIN in books ≠ supplier GSTIN in GSTR-2B
    idx = 6
    gstr2b[idx]["supplier_gstin"] = _generate_gstin(rng, "27")  # Different GSTIN
    ground_truth.append({"document_id": books[idx]["invoice_no"], "error_type": "gstin_mismatch"})

    # 6. date_mismatch — dates differ by > 30 days
    idx = 8
    orig_date = datetime.strptime(books[idx]["date"], "%Y-%m-%d")
    gstr2b[idx]["date"] = (orig_date + timedelta(days=45)).strftime("%Y-%m-%d")
    ground_truth.append({"document_id": books[idx]["invoice_no"], "error_type": "date_mismatch"})

    # 7. duplicate_entry — same invoice appears twice in books
    dup_entry = books[10].copy()
    dup_entry_inv = books[10]["invoice_no"]
    books.append(dup_entry)
    ground_truth.append({"document_id": dup_entry_inv, "error_type": "duplicate_entry"})

    # 8. invalid_gstin — GSTIN with wrong format in books
    inv_no = "VINV-0025"
    books.append({
        "invoice_no": inv_no,
        "vendor_name": "Shady Corp",
        "vendor_gstin": "XXINVALID123",  # Invalid format
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

    # 9. wrong_tax_type — IGST used for intra-state (should be CGST+SGST)
    idx = 12
    # This entry is intra-state (same state) but uses IGST
    books[idx]["igst"] = books[idx]["cgst"] + books[idx]["sgst"]
    books[idx]["cgst"] = 0
    books[idx]["sgst"] = 0
    books[idx]["place_of_supply"] = our_state  # Same state = should be CGST+SGST
    ground_truth.append({"document_id": books[idx]["invoice_no"], "error_type": "wrong_tax_type"})

    # 10. itc_ineligible — blocked category (food & beverages)
    inv_no = "VINV-0026"
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
        "hsn_code": "2106",  # Food preparation HSN
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
    inv_no = "VINV-0027"
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

    # 12. excess_claim — ITC in books > GSTR-2B amount
    idx = 14
    books[idx]["cgst"] = books[idx]["cgst"] + 2000
    books[idx]["sgst"] = books[idx]["sgst"] + 2000
    books[idx]["total"] = books[idx]["taxable_value"] + books[idx]["cgst"] + books[idx]["sgst"] + books[idx]["igst"]
    ground_truth.append({"document_id": books[idx]["invoice_no"], "error_type": "excess_claim"})

    # Shuffle books (but keep gstr2b in order for realism — portal data is usually sorted)
    rng.shuffle(books)

    context = {
        "our_state_code": our_state,
        "our_gstin": _generate_gstin(rng, our_state),
        "current_date": datetime.now().strftime("%Y-%m-%d"),
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
# Generator dispatcher — called by environment.py
# ---------------------------------------------------------------------------

GENERATORS = {
    "generate_expense_data": generate_expense_data,
    "generate_invoice_data": generate_invoice_data,
    "generate_gst_data": generate_gst_data,
}


def generate_data_for_task(generator_name: str, seed: int = 42) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """
    Dispatch to the correct data generator.

    Args:
        generator_name: One of the generator function names from tasks.py
        seed: Random seed for reproducibility

    Returns:
        Tuple of (documents, ground_truth)

    Raises:
        ValueError: If generator_name is not recognized
    """
    if generator_name not in GENERATORS:
        raise ValueError(f"Unknown generator: {generator_name}")
    return GENERATORS[generator_name](seed)
