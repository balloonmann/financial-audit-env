"""Shared fixtures for the Financial Audit Environment test suite."""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter before each test to prevent cross-test interference."""
    from financial_audit_env.server.security import rate_limiter
    rate_limiter._requests.clear()
    yield
    rate_limiter._requests.clear()


@pytest.fixture
def env():
    """Create a fresh environment instance for each test."""
    from financial_audit_env.server.environment import FinancialAuditEnvironment
    return FinancialAuditEnvironment()


@pytest.fixture
def expense_data():
    """Generate expense data with seed 42."""
    from financial_audit_env.server.data_generator import generate_expense_data
    return generate_expense_data(42)


@pytest.fixture
def invoice_data():
    """Generate invoice data with seed 42."""
    from financial_audit_env.server.data_generator import generate_invoice_data
    return generate_invoice_data(42)


@pytest.fixture
def gst_data():
    """Generate GST data with seed 42."""
    from financial_audit_env.server.data_generator import generate_gst_data
    return generate_gst_data(42)


@pytest.fixture
def fraud_data():
    """Generate fraud data with seed 42."""
    from financial_audit_env.server.data_generator import generate_fraud_data
    return generate_fraud_data(42)
