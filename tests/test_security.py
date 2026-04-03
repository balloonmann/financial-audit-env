"""Tests for security module — rate limiting, request IDs, ground truth leakage."""

import time
import pytest
from financial_audit_env.server.security import InMemoryRateLimiter


class TestRateLimiter:
    """Tests for in-memory rate limiter."""

    def test_allows_under_limit(self):
        limiter = InMemoryRateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert not limiter.is_rate_limited("127.0.0.1")

    def test_blocks_over_limit(self):
        limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter.is_rate_limited("127.0.0.1")
        assert limiter.is_rate_limited("127.0.0.1")

    def test_different_ips_independent(self):
        limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)
        limiter.is_rate_limited("1.1.1.1")
        limiter.is_rate_limited("1.1.1.1")
        assert limiter.is_rate_limited("1.1.1.1")
        assert not limiter.is_rate_limited("2.2.2.2")  # Different IP

    def test_cleanup_removes_stale_ips(self):
        limiter = InMemoryRateLimiter(max_requests=5, window_seconds=1)
        limiter.is_rate_limited("old_ip")
        # Force cleanup
        time.sleep(1.1)
        limiter._last_cleanup = 0  # Force cleanup on next call
        limiter.is_rate_limited("new_ip")
        assert "old_ip" not in limiter._requests

    def test_no_memory_leak(self):
        limiter = InMemoryRateLimiter(max_requests=100, window_seconds=1)
        # Simulate many unique IPs
        for i in range(50):
            limiter.is_rate_limited(f"ip_{i}")
        assert len(limiter._requests) == 50
        # After window expires, cleanup should remove them
        time.sleep(1.1)
        limiter._last_cleanup = 0
        limiter.is_rate_limited("trigger_cleanup")
        assert len(limiter._requests) < 50


class TestGroundTruthLeakage:
    """Verify error messages never expose ground truth."""

    def test_value_error_sanitized(self):
        """Ensure ground truth keywords are stripped from error messages."""
        from financial_audit_env.server.security import setup_security
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        setup_security(app)

        @app.get("/test-error")
        async def test_error():
            raise ValueError("Expected 12 ground_truth errors but got 5")

        client = TestClient(app)
        resp = client.get("/test-error")
        assert resp.status_code == 400
        # Should NOT contain ground truth info
        assert "ground_truth" not in resp.json()["detail"]
        assert "12" not in resp.json()["detail"]

    def test_generic_error_no_trace(self):
        from financial_audit_env.server.security import setup_security
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        setup_security(app)

        @app.get("/test-crash")
        async def test_crash():
            raise RuntimeError("Traceback with planted error details")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test-crash")
        assert resp.status_code == 500
        body = resp.json()
        # Key check: no internal details leaked to client
        assert "planted" not in body.get("detail", "")
        assert "Traceback" not in body.get("detail", "")
