"""Tests for data generators — reproducibility, noise, edge cases."""

import pytest
from financial_audit_env.server.data_generator import (
    generate_expense_data,
    generate_invoice_data,
    generate_gst_data,
    generate_fraud_data,
    REFERENCE_DATE,
)


class TestExpenseGenerator:
    """Tests for expense data generator."""

    def test_generates_expected_structure(self, expense_data):
        docs, gt = expense_data
        assert "expenses" in docs
        assert "policy" in docs
        assert isinstance(docs["expenses"], list)
        assert len(docs["expenses"]) > 0
        assert len(gt) == 7, f"Expected 7 ground truth errors, got {len(gt)}"

    def test_expense_has_required_fields(self, expense_data):
        docs, _ = expense_data
        required = {"expense_id", "date", "employee", "category", "amount", "description", "receipt_id", "vendor"}
        for exp in docs["expenses"]:
            assert required.issubset(set(exp.keys())), f"Missing fields in {exp['expense_id']}"

    def test_reproducibility(self):
        d1, g1 = generate_expense_data(42)
        d2, g2 = generate_expense_data(42)
        assert d1 == d2, "Same seed should produce identical data"
        assert g1 == g2, "Same seed should produce identical ground truth"

    def test_different_seeds_different_data(self):
        d1, _ = generate_expense_data(42)
        d2, _ = generate_expense_data(99)
        assert d1 != d2, "Different seeds should produce different data"

    def test_no_datetime_now_dependency(self):
        """Verify data doesn't change across calls (no datetime.now() usage)."""
        d1, g1 = generate_expense_data(42)
        d2, g2 = generate_expense_data(42)
        assert d1 == d2, "Data should be identical regardless of wall clock"

    def test_red_herrings_exist(self, expense_data):
        docs, gt = expense_data
        gt_ids = {g["document_id"] for g in gt}
        non_error_ids = {e["expense_id"] for e in docs["expenses"]} - gt_ids
        # Should have legitimate entries that aren't violations
        assert len(non_error_ids) > 5, "Should have clean entries as distractors"

    def test_cumulative_breach_error(self, expense_data):
        _, gt = expense_data
        error_types = {g["error_type"] for g in gt}
        assert "cumulative_breach" in error_types, "Should have cumulative breach error"

    def test_all_error_types_present(self, expense_data):
        _, gt = expense_data
        error_types = {g["error_type"] for g in gt}
        expected = {"over_limit", "wrong_category", "duplicate_claim", "weekend_expense",
                     "missing_receipt", "unapproved_vendor", "cumulative_breach"}
        assert error_types == expected, f"Expected {expected}, got {error_types}"


class TestInvoiceGenerator:
    """Tests for invoice three-way match generator."""

    def test_generates_expected_structure(self, invoice_data):
        docs, gt = invoice_data
        assert "purchase_orders" in docs
        assert "grns" in docs
        assert "invoices" in docs
        assert len(gt) == 9, f"Expected 9 errors, got {len(gt)}"

    def test_po_grn_invoice_counts(self, invoice_data):
        docs, _ = invoice_data
        assert len(docs["purchase_orders"]) == 10
        assert len(docs["grns"]) == 10
        assert len(docs["invoices"]) >= 12  # 10 + 2 extra (duplicate + unmatched)

    def test_cascading_error(self, invoice_data):
        _, gt = invoice_data
        error_types = [g["error_type"] for g in gt]
        assert "cascading_total" in error_types, "Should have cascading error"

    def test_reproducibility(self):
        d1, g1 = generate_invoice_data(42)
        d2, g2 = generate_invoice_data(42)
        assert d1 == d2
        assert g1 == g2


class TestGSTGenerator:
    """Tests for GST reconciliation generator."""

    def test_generates_expected_structure(self, gst_data):
        docs, gt = gst_data
        assert "purchase_register" in docs
        assert "gstr2b" in docs
        assert "context" in docs
        assert len(gt) == 12, f"Expected 12 errors, got {len(gt)}"

    def test_scaled_up_data(self, gst_data):
        docs, _ = gst_data
        # Should have 40+ book entries (40 clean + error entries)
        assert len(docs["purchase_register"]) >= 40, \
            f"Expected 40+ entries, got {len(docs['purchase_register'])}"

    def test_all_12_error_types(self, gst_data):
        _, gt = gst_data
        error_types = {g["error_type"] for g in gt}
        expected = {
            "missing_in_gstr2b", "missing_in_books", "amount_mismatch",
            "tax_rate_mismatch", "gstin_mismatch", "date_mismatch",
            "duplicate_entry", "invalid_gstin", "wrong_tax_type",
            "itc_ineligible", "late_claim", "excess_claim",
        }
        assert error_types == expected, f"Missing: {expected - error_types}"


class TestFraudGenerator:
    """Tests for fraud pattern detection generator."""

    def test_generates_expected_structure(self, fraud_data):
        docs, gt = fraud_data
        assert "transactions" in docs
        assert "vendor_registry" in docs
        assert "audit_context" in docs
        assert len(gt) == 10, f"Expected 10 fraud patterns, got {len(gt)}"

    def test_transaction_count(self, fraud_data):
        docs, _ = fraud_data
        assert len(docs["transactions"]) >= 50, \
            f"Expected 50+ transactions, got {len(docs['transactions'])}"

    def test_vendor_registry(self, fraud_data):
        docs, _ = fraud_data
        assert len(docs["vendor_registry"]) >= 20

    def test_all_fraud_types(self, fraud_data):
        _, gt = fraud_data
        error_types = {g["error_type"] for g in gt}
        expected = {
            "circular_invoicing", "split_invoice", "shell_company",
            "round_number_anomaly", "benford_violation", "vendor_concentration",
            "duplicate_bank_account", "sudden_volume_spike",
            "invoice_before_incorporation", "weekend_pattern",
        }
        assert error_types == expected, f"Missing: {expected - error_types}"

    def test_reproducibility(self):
        d1, g1 = generate_fraud_data(42)
        d2, g2 = generate_fraud_data(42)
        assert d1 == d2
        assert g1 == g2

    def test_audit_context_has_thresholds(self, fraud_data):
        docs, _ = fraud_data
        ctx = docs["audit_context"]
        assert "invoice_approval_threshold" in ctx
        assert "benford_expected_distribution" in ctx
        assert "max_vendor_concentration_pct" in ctx


class TestReferenceDate:
    """Verify REFERENCE_DATE is used instead of datetime.now()."""

    def test_reference_date_is_fixed(self):
        from datetime import datetime
        assert REFERENCE_DATE == datetime(2026, 1, 15)

    def test_all_dates_before_reference(self, expense_data):
        from datetime import datetime
        docs, _ = expense_data
        for exp in docs["expenses"]:
            date = datetime.strptime(exp["date"], "%Y-%m-%d")
            assert date <= REFERENCE_DATE, f"Date {exp['date']} is after reference date"
