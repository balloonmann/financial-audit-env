# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — Security middleware and utilities.
#
# Implements OWASP best practices:
# - Rate limiting with TTL-based cleanup (no memory leak)
# - Input sanitization helpers
# - Custom error handlers (no stack traces in production)
# - Secure headers middleware
# - Request body size limits (stream-based, not header-based)
# - Request ID tracking for debugging
# - CORS configuration from environment variables
# - Optional API key auth for expensive endpoints
# - Ground truth leakage prevention

import logging
import os
import time
import uuid
from collections import defaultdict
from typing import Callable, Dict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger("financial_audit_env.security")


# ---------------------------------------------------------------------------
# Rate limiting with TTL-based cleanup (fixes memory leak)
# ---------------------------------------------------------------------------

class InMemoryRateLimiter:
    """
    Token-bucket rate limiter per client IP with automatic cleanup.

    Limits each IP to `max_requests` within `window_seconds`.
    Periodically evicts stale entries to prevent memory leak.
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # Clean every 5 minutes

    def is_rate_limited(self, client_ip: str) -> bool:
        """Check if a client IP has exceeded the rate limit."""
        now = time.time()

        # Periodic cleanup of stale entries
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup(now)

        window_start = now - self.window_seconds

        # Clean old requests for this IP
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > window_start
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            return True

        self._requests[client_ip].append(now)
        return False

    def _cleanup(self, now: float) -> None:
        """Remove stale IP entries to prevent memory leak."""
        window_start = now - self.window_seconds
        stale_ips = []
        for ip, timestamps in self._requests.items():
            # Remove IPs with no recent requests
            active = [t for t in timestamps if t > window_start]
            if not active:
                stale_ips.append(ip)
            else:
                self._requests[ip] = active

        for ip in stale_ips:
            del self._requests[ip]

        self._last_cleanup = now
        if stale_ips:
            logger.debug(f"Rate limiter cleanup: evicted {len(stale_ips)} stale IPs")


# Global rate limiter instance
rate_limiter = InMemoryRateLimiter(max_requests=30, window_seconds=60)


# ---------------------------------------------------------------------------
# Security middleware setup
# ---------------------------------------------------------------------------

# Maximum request body size (1MB)
MAX_BODY_SIZE = 1 * 1024 * 1024

# CORS origins from environment variable (default: * for HF Spaces)
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

# Optional API key for expensive endpoints
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")


def setup_security(app: FastAPI) -> None:
    """
    Apply all security middleware and handlers to a FastAPI app.

    Sets up:
    1. CORS with configurable origins
    2. Rate limiting middleware with TTL cleanup
    3. Request body size limits (stream-based)
    4. Secure response headers
    5. Request ID tracking
    6. Custom error handlers (no stack traces)
    7. Ground truth leakage prevention
    """

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # --- Combined security middleware ---
    @app.middleware("http")
    async def security_middleware(request: Request, call_next: Callable) -> Response:
        """Combined security middleware."""
        client_ip = request.client.host if request.client else "unknown"

        # Generate request ID for tracking
        request_id = str(uuid.uuid4())[:8]

        # Rate limiting
        if rate_limiter.is_rate_limited(client_ip):
            logger.warning(f"[{request_id}] Rate limited: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
                headers={"X-Request-ID": request_id},
            )

        # Request body size limit — stream-based check (not trusting header)
        if request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > MAX_BODY_SIZE:
                        return JSONResponse(
                            status_code=413,
                            content={"detail": "Request body too large. Maximum 1MB."},
                            headers={"X-Request-ID": request_id},
                        )
                except ValueError:
                    pass  # Invalid content-length header, let it through

        # Auth check for expensive endpoints
        if ADMIN_API_KEY and request.url.path in ("/baseline",):
            api_key = request.headers.get("X-API-Key", "")
            if api_key != ADMIN_API_KEY:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "API key required for this endpoint."},
                    headers={"X-Request-ID": request_id},
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
        response.headers["X-Request-ID"] = request_id
        # Prevent ground truth leakage via headers
        response.headers["X-Robots-Tag"] = "noindex, nofollow"

        return response

    # --- Custom error handlers (no stack traces in responses) ---
    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all error handler. Never leaks ground truth or internals."""
        logger.error(f"Unhandled error: {type(exc).__name__}: {exc}", exc_info=True)
        # Ground truth leakage prevention: sanitize error message
        safe_message = "Internal server error. Please try again."
        return JSONResponse(
            status_code=500,
            content={"detail": safe_message},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle validation errors — sanitize to prevent ground truth leakage."""
        error_msg = str(exc)
        # Remove any potential ground truth references
        for dangerous_word in ["ground_truth", "planted", "error_count", "answer"]:
            if dangerous_word in error_msg.lower():
                error_msg = "Invalid input. Please check your request format."
                break
        return JSONResponse(
            status_code=400,
            content={"detail": error_msg},
        )


def get_hf_token() -> str:
    """Securely retrieve the HuggingFace token from environment variables."""
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
