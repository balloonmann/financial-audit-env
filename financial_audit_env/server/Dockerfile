# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — Dockerfile
#
# Builds a minimal container that runs the FastAPI server.
# Compatible with Hugging Face Spaces deployment.
#
# Build:
#   docker build -f server/Dockerfile -t financial-audit-env .
#
# Run:
#   docker run -p 8000:8000 financial-audit-env
#
# With HF token (for baseline endpoint):
#   docker run -p 8000:8000 -e HF_TOKEN=your_token financial-audit-env

FROM python:3.11-slim

# Security: run as non-root user (OWASP best practice)
RUN useradd -m -s /bin/bash appuser

WORKDIR /app

# Install dependencies first (for Docker cache efficiency)
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire package
COPY . .

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app:$PYTHONPATH"

# Switch to non-root user
USER appuser

# Expose the server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "financial_audit_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
