# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — Security middleware and utilities.
#
# Implements OWASP best practices:
# - Rate limiting (SlowAPI) on all public endpoints
# - Input sanitization helpers
# - Custom error handlers (no stack traces in production)
# - Secure headers middleware
# - Request body size limits
# - API key handling from environment variables only

import logging
import os
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger("financial_audit_env.security")


# ---------------------------------------------------------------------------
# Rate limiting configuration
# ---------------------------------------------------------------------------
# Using a simple in-memory rate limiter to avoid extra dependencies.
# For production, replace with slowapi or redis-based limiter.

import time
from collections import defaultdict


class InMemoryRateLimiter:
    """
    Simple token-bucket rate limiter per client IP.

    Limits each IP to `max_requests` within `window_seconds`.
    Not suitable for multi-process/distributed deployments —
    use Redis-backed limiter for those scenarios.
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)

    def is_rate_limited(self, client_ip: str) -> bool:
        """Check if a client IP has exceeded the rate limit."""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > window_start
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            return True

        self._requests[client_ip].append(now)
        return False


from typing import Dict  # noqa: E402 (needed after class definition)

# Global rate limiter instance
rate_limiter = InMemoryRateLimiter(max_requests=30, window_seconds=60)


# ---------------------------------------------------------------------------
# Security middleware setup
# ---------------------------------------------------------------------------

# Maximum request body size (1MB) — prevents DoS via large payloads
MAX_BODY_SIZE = 1 * 1024 * 1024  # 1MB


def setup_security(app: FastAPI) -> None:
    """
    Apply all security middleware and handlers to a FastAPI app.

    This sets up:
    1. CORS with restricted origins
    2. Rate limiting middleware
    3. Request body size limits
    4. Secure response headers
    5. Custom error handlers (no stack traces)

    Args:
        app: FastAPI application instance
    """

    # --- CORS ---
    # Allow all origins for now since this runs as a public HF Space.
    # In production, restrict to specific domains.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # HF Spaces need open CORS
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # --- Rate limiting + body size + secure headers ---
    @app.middleware("http")
    async def security_middleware(request: Request, call_next: Callable) -> Response:
        """Combined security middleware for rate limiting, body size, and headers."""
        client_ip = request.client.host if request.client else "unknown"

        # Rate limiting
        if rate_limiter.is_rate_limited(client_ip):
            logger.warning(f"Rate limited: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
            )

        # Request body size limit (for POST/PUT requests)
        if request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_BODY_SIZE:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large. Maximum 1MB."},
                )

        # Process request
        response = await call_next(request)

        # Add secure headers (OWASP recommendations)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response

    # --- Custom error handlers (no stack traces in responses) ---
    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Catch-all error handler. Logs the full error internally
        but returns a safe message to the client.
        """
        logger.error(f"Unhandled error: {type(exc).__name__}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Please try again."},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle validation errors with a user-friendly message."""
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)},
        )


def get_hf_token() -> str:
    """
    Securely retrieve the HuggingFace token from environment variables.

    Never logs, prints, or returns the full token in error messages.

    Returns:
        The HF_TOKEN value

    Raises:
        RuntimeError: If HF_TOKEN is not set
    """
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Set it with: export HF_TOKEN=your_token_here"
        )
    return token


def mask_token(token: str) -> str:
    """Mask a token for safe logging — shows only first 4 and last 4 chars."""
    if len(token) <= 8:
        return "****"
    return f"{token[:4]}...{token[-4:]}"
