"""Regulatory shock helper tests.

Covers: shock delivery, delivery guard, ground truth modification for tax
and vendor rules, timing precision, notification formatting, and schema
drift data helpers.
"""

from datetime import date, timedelta

from financial_audit_env.server.data_generator import apply_schema_drift, apply_regulatory_shock
from financial_audit_env.server.instructions import REGULATORY_SHOCKS
from financial_audit_env.server.regulatory import (
    apply_shock_to_ground_truth,
    format_shock_notification,
    get_shock_for_period_step,
)


def test_get_shock_for_period_step_and_delivery_guard():
    shock = REGULATORY_SHOCKS[0]
    period = int(shock["trigger_period"])
    step = int(shock["trigger_after_step"]) + 1

    out = get_shock_for_period_step(period=period, step=step, delivered=set())
    assert out is not None
    assert out["id"] == shock["id"]

    delivered = {shock["id"]}
    skipped = get_shock_for_period_step(period=period, step=step, delivered=delivered)
    assert skipped is None


def test_apply_shock_to_ground_truth_tax_and_vendor_rules():
    tax_shock = {
        "rule_change": {"tax_rate": {"998314": 0.12}},
    }
    vendor_shock = {
        "rule_change": {"new_vendor_risk_months": 6},
    }

    documents = {
        "purchase_register": [
            {
                "invoice_no": "INV-1",
                "hsn_code": "998314",
                "taxable_value": 1000,
                "cgst": 90,
                "sgst": 90,
                "igst": 0,
            }
        ],
        "vendor_registry": [
            {
                "vendor_id": "V-NEW",
                "incorporation_date": (date.today() + timedelta(days=30)).strftime("%Y-%m-%d"),
            }
        ],
        "transactions": [
            {
                "txn_id": "TXN-1",
                "vendor_id": "V-NEW",
            }
        ],
    }

    gt = apply_shock_to_ground_truth([], documents, tax_shock)
    assert any(x["error_type"] == "tax_rate_mismatch" for x in gt)

    gt2 = apply_shock_to_ground_truth([], documents, vendor_shock)
    assert any(x["error_type"] == "high_risk_new_vendor" for x in gt2)


def test_format_shock_notification_contains_key_fields():
    shock = REGULATORY_SHOCKS[1]
    msg = format_shock_notification(shock)
    assert shock["id"] in msg
    assert "Affected tasks" in msg
    assert "Severity" in msg


def test_shock_timing_precision():
    """Shocks must NOT trigger at or before their trigger_after_step."""
    for shock in REGULATORY_SHOCKS:
        period = shock["trigger_period"]
        at_step = shock["trigger_after_step"]

        # Should NOT trigger before its step threshold — mark all earlier shocks as delivered
        earlier_ids = {
            s["id"] for s in REGULATORY_SHOCKS
            if s["trigger_period"] == period and s["trigger_after_step"] < at_step
        }
        result = get_shock_for_period_step(period=period, step=at_step, delivered=earlier_ids)
        assert result is None or result["id"] != shock["id"], (
            f"{shock['id']} should not trigger at step {at_step}"
        )

        # Should trigger one step later (with earlier shocks delivered)
        result = get_shock_for_period_step(period=period, step=at_step + 1, delivered=earlier_ids)
        assert result is not None, f"{shock['id']} should trigger at step {at_step + 1}"


def test_wrong_period_returns_none():
    """Shock for period 3 should not trigger in period 2."""
    shock = REGULATORY_SHOCKS[0]  # trigger_period = 3
    result = get_shock_for_period_step(
        period=shock["trigger_period"] - 1,
        step=shock["trigger_after_step"] + 1,
        delivered=set(),
    )
    assert result is None


def test_apply_schema_drift_renames_fields():
    """The apply_schema_drift helper should rename dict keys recursively."""
    documents = {
        "purchase_register": [
            {"vendor_gstin": "27AABCU9603R1ZM", "amount": 1000},
            {"vendor_gstin": "29AABCU9603R1ZM", "amount": 2000},
        ],
        "summary": {"vendor_gstin": "aggregate_field"},
    }
    schema_changes = {"vendor_gstin": "supplier_gstin"}

    result = apply_schema_drift(documents, schema_changes)

    # Original field should be renamed
    for entry in result["purchase_register"]:
        assert "supplier_gstin" in entry
        assert "vendor_gstin" not in entry
    assert "supplier_gstin" in result["summary"]


def test_apply_regulatory_shock_recalculates_tax():
    """The apply_regulatory_shock helper should modify tax amounts in place."""
    documents = {
        "purchase_register": [
            {
                "hsn_code": "998314",
                "taxable_value": 10000,
                "cgst": 900,
                "sgst": 900,
                "igst": 0,
                "total": 11800,
            }
        ]
    }
    shock = {"rule_change": {"tax_rate": {"998314": 0.12}}}

    result_docs, _ = apply_regulatory_shock(documents, [], shock)
    entry = result_docs["purchase_register"][0]

    # New rate 12% → cgst=600, sgst=600
    assert entry["cgst"] == 600.0
    assert entry["sgst"] == 600.0
    assert entry["total"] == 11200.0
