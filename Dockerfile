# Financial Audit Environment — Dockerfile
#
# Builds a minimal container that runs the FastAPI server.
# Compatible with Hugging Face Spaces deployment.

FROM python:3.11-slim

# Security: run as non-root user (Hugging Face requirement)
RUN useradd -m -s /bin/bash appuser

WORKDIR /app

# 1. Install dependencies first (for Docker cache efficiency)
COPY financial_audit_env/server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the entire package
COPY . .

# 3. Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app:$PYTHONPATH"

# 4. Switch to non-root user
USER appuser

# 5. Expose the server port
EXPOSE 8000

# 6. Health check (monitors if the server actually starts)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 7. Start the server
CMD ["uvicorn", "financial_audit_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
