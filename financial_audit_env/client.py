# Copyright (c) 2026. All rights reserved.
# Financial Audit Environment — Client for connecting to the environment.
#
# This client is used by agents (and the baseline script) to interact
# with the environment server over HTTP/WebSocket.

from typing import Any, Dict, Optional

from .models import AuditAction, AuditObservation, AuditState

# ---------------------------------------------------------------------------
# Try to import OpenEnv client base class.
# Falls back to a simple HTTP client if openenv-core is not installed.
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult

    class FinancialAuditEnv(EnvClient[AuditAction, AuditObservation, AuditState]):
        """
        OpenEnv client for the Financial Audit Environment.

        Usage (async):
            async with FinancialAuditEnv(base_url="http://localhost:8000") as env:
                result = await env.reset(task_id="expense_audit")
                result = await env.step(AuditAction(findings=[...]))

        Usage (sync):
            with FinancialAuditEnv(base_url="http://localhost:8000").sync() as env:
                result = env.reset(task_id="expense_audit")
                result = env.step(AuditAction(findings=[...]))
        """

        def _step_payload(self, action: AuditAction) -> dict:
            """Convert action to wire format for WebSocket."""
            return action.model_dump()

        def _parse_result(self, payload: dict) -> StepResult:
            """Parse step response from wire format."""
            obs_data = payload.get("observation", payload)
            obs = AuditObservation(**obs_data)
            return StepResult(
                observation=obs,
                reward=payload.get("reward", obs.reward),
                done=payload.get("done", obs.done),
            )

        def _parse_state(self, payload: dict) -> AuditState:
            """Parse state response from wire format."""
            return AuditState(**payload)

except ImportError:
    # Standalone fallback — simple HTTP client using requests
    import requests

    class FinancialAuditEnv:
        """
        Simple HTTP client for the Financial Audit Environment.

        Standalone fallback when openenv-core is not installed.
        Uses synchronous HTTP requests.

        Usage:
            env = FinancialAuditEnv(base_url="http://localhost:8000")
            result = env.reset(task_id="expense_audit")
            result = env.step(AuditAction(findings=[...]))
        """

        def __init__(self, base_url: str = "http://localhost:8000"):
            self.base_url = base_url.rstrip("/")
            self._session = requests.Session()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._session.close()

        def sync(self):
            """Return self for compatibility with OpenEnv .sync() pattern."""
            return self

        def reset(
            self,
            task_id: str = "expense_audit",
            seed: int = 42,
            episode_id: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Reset the environment via HTTP POST."""
            response = self._session.post(
                f"{self.base_url}/reset",
                json={
                    "task_id": task_id,
                    "seed": seed,
                    "episode_id": episode_id,
                },
            )
            response.raise_for_status()
            return response.json()

        def step(self, action: AuditAction) -> Dict[str, Any]:
            """Submit an action via HTTP POST."""
            response = self._session.post(
                f"{self.base_url}/step",
                json={"action": action.model_dump()},
            )
            response.raise_for_status()
            return response.json()

        def get_state(self) -> Dict[str, Any]:
            """Get current state via HTTP GET."""
            response = self._session.get(f"{self.base_url}/state")
            response.raise_for_status()
            return response.json()

        def get_tasks(self) -> Dict[str, Any]:
            """Get available tasks via HTTP GET."""
            response = self._session.get(f"{self.base_url}/tasks")
            response.raise_for_status()
            return response.json()

        def get_grader(self) -> Dict[str, Any]:
            """Get grader score for last episode via HTTP GET."""
            response = self._session.get(f"{self.base_url}/grader")
            response.raise_for_status()
            return response.json()
