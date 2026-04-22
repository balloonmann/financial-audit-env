"""Regulatory shock helpers for mid-period rule injection."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .instructions import REGULATORY_SHOCKS


def get_shock_for_period_step(period: int, step: int, delivered: set) -> Optional[Dict[str, Any]]:
    """Return the first undelivered shock that triggers at this period/step."""
    for shock in REGULATORY_SHOCKS:
        if (
            shock["trigger_period"] == period
            and step > shock["trigger_after_step"]
            and shock["id"] not in delivered
        ):
            return shock
    return None


def apply_shock_to_ground_truth(
    ground_truth: List[Dict[str, str]],
    documents: Dict[str, Any],
    shock: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Extend ground truth with new rule-driven violations after a shock."""
    rule_change = shock.get("rule_change", {})
    new_gt = list(ground_truth)

    if "tax_rate" in rule_change:
        for hsn, new_rate in rule_change["tax_rate"].items():
            for entry in documents.get("purchase_register", []):
                if entry.get("hsn_code") != hsn:
                    continue
                taxable = float(entry.get("taxable_value", 0) or 0)
                expected_tax = round(taxable * float(new_rate), 2)
                actual_tax = (
                    float(entry.get("cgst", 0) or 0)
                    + float(entry.get("sgst", 0) or 0)
                    + float(entry.get("igst", 0) or 0)
                )
                if abs(actual_tax - expected_tax) > 1:
                    invoice_no = str(entry.get("invoice_no", "")).strip()
                    if invoice_no:
                        new_gt.append(
                            {
                                "document_id": invoice_no,
                                "error_type": "tax_rate_mismatch",
                            }
                        )

    if "new_vendor_risk_months" in rule_change:
        months = int(rule_change["new_vendor_risk_months"])
        cutoff = datetime.now() - timedelta(days=months * 30)
        for vendor in documents.get("vendor_registry", []):
            incorp = vendor.get("incorporation_date", "")
            try:
                incorp_date = datetime.strptime(str(incorp), "%Y-%m-%d")
            except (ValueError, TypeError):
                continue
            if incorp_date <= cutoff:
                continue
            vendor_id = vendor.get("vendor_id")
            if not vendor_id:
                continue
            for txn in documents.get("transactions", []):
                if txn.get("vendor_id") == vendor_id:
                    new_gt.append(
                        {
                            "document_id": str(txn.get("txn_id", "")).strip(),
                            "error_type": "high_risk_new_vendor",
                        }
                    )
                    break

    return new_gt


def format_shock_notification(shock: Dict[str, Any]) -> str:
    """Format human-readable shock text for agent-facing observation feedback."""
    return (
        f"REGULATORY CHANGE - {shock['id']}\n"
        f"{shock['text']}\n"
        f"Affected tasks: {', '.join(shock.get('affected_tasks', []))}\n"
        f"Severity: {shock.get('severity', 'n/a')}\n"
        "Apply this rule to remaining findings in this period."
    )
