# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — FastAPI application.
#
# Exposes the environment over HTTP with:
# - Standard OpenEnv endpoints (reset, step, state, health)
# - Custom endpoints:
#   GET  /tasks       → list of tasks with action schema
#   GET  /grader      → F1 score for last completed episode
#   POST /baseline    → trigger baseline inference
#   GET  /leaderboard → best scores per model
#   GET  /metrics     → basic usage statistics
# - Session-based multi-tenancy with global fallback
# - Security middleware

import logging
import os
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..models import AuditAction, AuditObservation
from .environment import FinancialAuditEnvironment
from .security import setup_security
from .tasks import TASKS, get_all_tasks_summary

logger = logging.getLogger("financial_audit_env.app")

# ---------------------------------------------------------------------------
# Global environment instance (fallback when no session_id provided)
# ---------------------------------------------------------------------------
_env = FinancialAuditEnvironment()

# Session-based environments for multi-tenancy
_sessions: Dict[str, Dict[str, Any]] = {}
_SESSION_TTL = 3600  # Sessions expire after 1 hour

# Leaderboard storage
_leaderboard: List[Dict[str, Any]] = []

# Metrics tracking
_metrics = {
    "total_resets": 0,
    "total_steps": 0,
    "total_episodes_completed": 0,
    "task_reset_counts": defaultdict(int),
    "start_time": time.time(),
}

# ---------------------------------------------------------------------------
# Create the FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Financial Audit Environment",
    description=(
        "An OpenEnv-compatible RL environment for financial auditing tasks. "
        "Agents audit synthetic financial documents to find planted errors. "
        "Supports 4 tasks (easy→expert), investigation mode, and adaptive difficulty."
    ),
    version="2.0.0",
    docs_url="/docs",
)
logger.info("Financial Audit Environment v2.0 — standalone FastAPI mode")

# ---------------------------------------------------------------------------
# Apply security middleware
# ---------------------------------------------------------------------------
setup_security(app)


# ---------------------------------------------------------------------------
# Session management helpers
# ---------------------------------------------------------------------------

def _get_env(session_id: Optional[str] = None) -> FinancialAuditEnvironment:
    """Get environment instance — session-based or global fallback."""
    if session_id and session_id in _sessions:
        _sessions[session_id]["last_access"] = time.time()
        return _sessions[session_id]["env"]
    return _env


def _create_session() -> str:
    """Create a new session with its own environment instance."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "env": FinancialAuditEnvironment(),
        "created_at": time.time(),
        "last_access": time.time(),
    }
    # Cleanup expired sessions
    _cleanup_sessions()
    return session_id


def _cleanup_sessions() -> None:
    """Remove expired sessions to prevent memory leak."""
    now = time.time()
    expired = [
        sid for sid, data in _sessions.items()
        if now - data["last_access"] > _SESSION_TTL
    ]
    for sid in expired:
        del _sessions[sid]
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired sessions")


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = "expense_audit"
    seed: Optional[int] = 42
    episode_id: Optional[str] = None
    session_id: Optional[str] = None
    investigation_mode: Optional[bool] = False

    model_config = {"extra": "allow"}


class StepRequest(BaseModel):
    """Request body for the /step endpoint."""
    action: AuditAction
    session_id: Optional[str] = None


class BaselineResponse(BaseModel):
    """Response from the /baseline endpoint."""
    scores: Dict[str, Any]
    model: str
    status: str


class LeaderboardEntry(BaseModel):
    """A single leaderboard entry."""
    model: str
    task_id: str
    score: float
    weighted_score: float
    risk_mitigation_pct: float
    timestamp: float


# ---------------------------------------------------------------------------
# Standard endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint — welcome page with API docs link."""
    return {
        "name": "Financial Audit Environment",
        "version": "2.0.0",
        "description": (
            "An OpenEnv-compatible RL environment for financial auditing tasks. "
            "Agents audit synthetic financial documents to find planted errors."
        ),
        "tasks": ["expense_audit", "invoice_match", "gst_reconciliation", "fraud_detection"],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "tasks": "/tasks",
            "reset": "POST /reset",
            "step": "POST /step",
            "grader": "/grader",
            "leaderboard": "/leaderboard",
            "metrics": "/metrics",
        },
        "quickstart": (
            "1. POST /reset with {\"task_id\": \"expense_audit\", \"seed\": 42}\n"
            "2. Read the documents in the observation\n"
            "3. POST /step with your findings\n"
            "4. GET /grader to see your score"
        ),
        "github": "https://github.com/balloonmann/financial-audit-env",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint — required for HF Space deployment."""
    return {
        "status": "healthy",
        "environment": "financial_audit_env",
        "version": "2.0.0",
        "tasks_available": len(TASKS),
        "active_sessions": len(_sessions),
    }


@app.post("/reset")
async def reset_endpoint(request: ResetRequest = ResetRequest()):
    """
    Reset the environment for a new episode.

    Args (JSON body):
        task_id: "expense_audit" | "invoice_match" | "gst_reconciliation" | "fraud_detection"
        seed: Random seed for reproducibility (default: 42)
        episode_id: Optional custom episode ID
        session_id: Optional session ID for multi-tenancy
        investigation_mode: If true, start in drill-down mode
    """
    try:
        env = _get_env(request.session_id)
        obs = env.reset(
            seed=request.seed,
            episode_id=request.episode_id,
            task_id=request.task_id,
            investigation_mode=request.investigation_mode or False,
        )
        # Track metrics
        _metrics["total_resets"] += 1
        _metrics["task_reset_counts"][request.task_id or "expense_audit"] += 1

        response = {
            "observation": obs.model_dump(),
            "done": obs.done,
            "reward": obs.reward,
        }
        if request.session_id:
            response["session_id"] = request.session_id
        return response
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
        env = _get_env(request.session_id)
        obs = env.step(request.action)
        _metrics["total_steps"] += 1
        if obs.done:
            _metrics["total_episodes_completed"] += 1
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
async def state_endpoint(session_id: Optional[str] = None):
    """Get current episode state (step count, found errors, etc.)."""
    env = _get_env(session_id)
    return env.state.model_dump()


# ---------------------------------------------------------------------------
# Session management endpoints
# ---------------------------------------------------------------------------

@app.post("/session")
async def create_session():
    """Create a new isolated session for multi-tenancy."""
    session_id = _create_session()
    return {
        "session_id": session_id,
        "ttl_seconds": _SESSION_TTL,
        "message": "Session created. Include session_id in /reset and /step requests.",
    }


# ---------------------------------------------------------------------------
# Contest-required custom endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
async def get_tasks():
    """List all available tasks with their descriptions and action schemas."""
    return {
        "tasks": get_all_tasks_summary(),
        "total_tasks": len(TASKS),
    }


@app.get("/grader")
async def get_grader_score(session_id: Optional[str] = None):
    """
    Get the grader score for the last completed episode.
    Includes F1, weighted F1, confusion matrix, and risk scoring.
    """
    env = _get_env(session_id)
    result = env.last_grader_result
    if result is None:
        return {
            "status": "no_completed_episode",
            "message": "No episode completed. Call /reset then /step with submit_final=True.",
        }

    def final_clamp(val: float) -> float:
        """Keep score-like fields within a stable open interval."""
        return max(0.01, min(0.99, val))

    return {
        "status": "completed",
        "task_id": env.state.task_id,
        # Primary score fields with a final stability clamp.
        "score": final_clamp(result["score"]),
        "precision": final_clamp(result["precision"]),
        "recall": final_clamp(result["recall"]),
        "true_positives": result["true_positives"],
        "false_positives": result["false_positives"],
        "false_negatives": result["false_negatives"],
        "total_errors": result["total_errors"],
        # Enhanced scoring fields with the same stability clamp.
        "weighted_score": final_clamp(result.get("weighted_score", result["score"])),
        "partial_credit_score": final_clamp(result.get("partial_credit_score", result["score"])),
        "partial_matches": result.get("partial_matches", 0),
        # Confusion matrix
        "confusion_matrix": result.get("confusion_matrix", {}),
        # Risk scoring
        "risk_score": result.get("risk_score", {}),
    }


# ---------------------------------------------------------------------------
# Leaderboard endpoint
# ---------------------------------------------------------------------------

@app.post("/leaderboard")
async def submit_to_leaderboard(
    model: str,
    task_id: str,
    score: float,
    weighted_score: float = 0.0,
    risk_mitigation_pct: float = 0.0,
):
    """Submit a score to the leaderboard."""
    entry = {
        "model": model,
        "task_id": task_id,
        "score": score,
        "weighted_score": weighted_score,
        "risk_mitigation_pct": risk_mitigation_pct,
        "timestamp": time.time(),
    }
    _leaderboard.append(entry)
    # Keep top 100 entries
    _leaderboard.sort(key=lambda x: x["score"], reverse=True)
    while len(_leaderboard) > 100:
        _leaderboard.pop()
    return {"status": "submitted", "rank": next(
        (i + 1 for i, e in enumerate(_leaderboard) if e == entry), -1
    )}


@app.get("/leaderboard")
async def get_leaderboard(task_id: Optional[str] = None, limit: int = 20):
    """Get the leaderboard — best scores per model."""
    entries = _leaderboard
    if task_id:
        entries = [e for e in entries if e["task_id"] == task_id]
    return {
        "leaderboard": entries[:limit],
        "total_entries": len(entries),
    }


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------

@app.get("/metrics")
async def get_metrics():
    """Get basic usage statistics."""
    uptime = time.time() - _metrics["start_time"]
    return {
        "uptime_seconds": round(uptime, 0),
        "total_resets": _metrics["total_resets"],
        "total_steps": _metrics["total_steps"],
        "total_episodes_completed": _metrics["total_episodes_completed"],
        "task_usage": dict(_metrics["task_reset_counts"]),
        "active_sessions": len(_sessions),
    }


# ---------------------------------------------------------------------------
# Adaptive difficulty endpoint
# ---------------------------------------------------------------------------

@app.get("/adaptive-difficulty")
async def get_adaptive_difficulty(session_id: Optional[str] = None):
    """Get adaptive difficulty recommendations based on score history."""
    env = _get_env(session_id)
    return env.get_adaptive_difficulty()


# ---------------------------------------------------------------------------
# Baseline endpoint
# ---------------------------------------------------------------------------

@app.post("/baseline")
async def run_baseline():
    """
    Run the baseline agent on all 3 tasks and return scores.
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
