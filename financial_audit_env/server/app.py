# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — FastAPI application.
#
# Exposes the environment over HTTP/WebSocket with:
# - Standard OpenEnv endpoints (reset, step, state, health)
# - Custom endpoints required by the contest:
#   GET  /tasks    → list of tasks with action schema
#   GET  /grader   → grader score for last completed episode
#   POST /baseline → trigger baseline inference, return scores
# - Security middleware (rate limiting, OWASP headers, input validation)
#
# Architecture note:
#   When openenv-core IS installed: create_app() provides /reset, /step, /state, /ws.
#   When openenv-core is NOT installed: we provide standalone HTTP versions.
#   Custom endpoints (/tasks, /grader, /baseline) are always registered.

import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..models import AuditAction, AuditObservation
from .environment import FinancialAuditEnvironment
from .security import setup_security
from .tasks import TASKS, get_all_tasks_summary

logger = logging.getLogger("financial_audit_env.app")

# ---------------------------------------------------------------------------
# Track whether OpenEnv is providing standard endpoints
# ---------------------------------------------------------------------------
_OPENENV_AVAILABLE = False

# Global environment instance — used by custom endpoints (/grader, /baseline)
# and by standalone mode endpoints (/reset, /step, /state)
_env = FinancialAuditEnvironment()

# ---------------------------------------------------------------------------
# Create the FastAPI app
# ---------------------------------------------------------------------------
# We always use our own FastAPI app with a single shared environment instance.
# OpenEnv's create_app() is NOT used because it creates its own internal
# environment instance, causing dual-instance routing bugs. Our standalone
# endpoints provide the same reset/step/state API and are fully spec-compliant.
app = FastAPI(
    title="Financial Audit Environment",
    description=(
        "An OpenEnv-compatible RL environment for financial auditing tasks. "
        "Agents audit synthetic financial documents to find planted errors."
    ),
    version="1.0.0",
    docs_url="/docs",
)
logger.info("Financial Audit Environment — standalone FastAPI mode")

# ---------------------------------------------------------------------------
# Apply security middleware
# ---------------------------------------------------------------------------
setup_security(app)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = "expense_audit"
    seed: Optional[int] = 42
    episode_id: Optional[str] = None

    model_config = {"extra": "allow"}


class StepRequest(BaseModel):
    """Request body for the /step endpoint."""
    action: AuditAction


class BaselineResponse(BaseModel):
    """Response from the /baseline endpoint."""
    scores: Dict[str, Any]
    model: str
    status: str


# ---------------------------------------------------------------------------
# Standalone / Override endpoints
# These always exist regardless of whether openenv-core is installed,
# because we need consistent behavior for the /reset, /step, /state calls
# and OpenEnv's create_app may not pass our custom kwargs (task_id, etc.)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint — required for HF Space deployment."""
    return {"status": "healthy", "environment": "financial_audit_env"}


@app.post("/reset")
async def reset_endpoint(request: ResetRequest = ResetRequest()):
    """
    Reset the environment for a new episode.

    Generates fresh financial data with planted errors for the given task.

    Args (JSON body):
        task_id: "expense_audit" | "invoice_match" | "gst_reconciliation"
        seed: Random seed for reproducibility (default: 42)
        episode_id: Optional custom episode ID
    """
    try:
        obs = _env.reset(
            seed=request.seed,
            episode_id=request.episode_id,
            task_id=request.task_id,
        )
        return {
            "observation": obs.model_dump(),
            "done": obs.done,
            "reward": obs.reward,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step_endpoint(request: StepRequest):
    """
    Execute one step in the environment.

    Submit audit findings and receive feedback + reward.
    Set submit_final=True to end the episode and get final grading.
    """
    try:
        obs = _env.step(request.action)
        return {
            "observation": obs.model_dump(),
            "done": obs.done,
            "reward": obs.reward,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
async def state_endpoint():
    """Get current episode state (step count, found errors, etc.)."""
    return _env.state.model_dump()


# ---------------------------------------------------------------------------
# Contest-required custom endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
async def get_tasks():
    """
    List all available tasks with their descriptions and action schemas.

    Returns details for all 3 tasks:
    - expense_audit (Easy): Policy violation detection
    - invoice_match (Medium): Three-way PO/GRN/Invoice matching
    - gst_reconciliation (Hard): GST return reconciliation
    """
    return {
        "tasks": get_all_tasks_summary(),
        "total_tasks": len(TASKS),
    }


@app.get("/grader")
async def get_grader_score():
    """
    Get the grader score for the last completed episode.

    Returns the F1 score (0.0–1.0) along with precision, recall, and
    error counts. Must complete an episode first.
    """
    result = _env.last_grader_result
    if result is None:
        return {
            "status": "no_completed_episode",
            "message": "No episode completed. Call /reset then /step with submit_final=True.",
        }

    return {
        "status": "completed",
        "task_id": _env.state.task_id,
        "score": result["score"],
        "precision": result["precision"],
        "recall": result["recall"],
        "true_positives": result["true_positives"],
        "false_positives": result["false_positives"],
        "false_negatives": result["false_negatives"],
        "total_errors": result["total_errors"],
    }


@app.post("/baseline")
async def run_baseline():
    """
    Run the baseline agent on all 3 tasks and return scores.

    Uses Meta's Llama 3.1 8B Instruct via HuggingFace Inference API.
    Requires HF_TOKEN environment variable.
    """
    try:
        from ..baseline import run_baseline_all_tasks
    except ImportError:
        return JSONResponse(
            status_code=501,
            content={
                "status": "error",
                "message": "Baseline not available. Run baseline.py directly.",
            },
        )

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "HF_TOKEN not set. Get one at https://huggingface.co/settings/tokens",
            },
        )

    try:
        scores = run_baseline_all_tasks(env=_env, hf_token=hf_token)
        return BaselineResponse(
            scores=scores,
            model="meta-llama/Llama-3.1-8B-Instruct",
            status="completed",
        )
    except Exception as e:
        logger.error(f"Baseline failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Baseline failed. Check logs."},
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the server directly: python -m financial_audit_env.server.app"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
