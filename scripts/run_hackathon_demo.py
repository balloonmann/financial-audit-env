#!/usr/bin/env python3
"""Run a deterministic no-LLM campaign demo and write a summary artifact."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


ROLES = [
    "expense_specialist",
    "invoice_specialist",
    "gst_specialist",
    "fraud_specialist",
]


def _headers(api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json", "User-Agent": "hackathon-demo-runner/1.0"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _post(session: requests.Session, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    resp = session.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def run_demo(env_url: str, seed: int, total_periods: int, api_key: str) -> Dict[str, Any]:
    started_at = time.time()
    session = requests.Session()
    headers = _headers(api_key)

    start_payload = {"seed": seed, "total_periods": total_periods}
    start_data = _post(session, f"{env_url}/campaign/start", start_payload, headers)
    campaign_id = start_data["campaign_id"]

    periods: List[Dict[str, Any]] = []

    for period in range(1, total_periods + 1):
        period_row: Dict[str, Any] = {
            "period": period,
            "roles": {},
            "overseer_decisions": 0,
            "regulatory_shocks_seen": 0,
            "schema_version": None,
            "policy_version": None,
        }

        for role in ROLES:
            task_start = _post(
                session,
                f"{env_url}/campaign/task/start",
                {"campaign_id": campaign_id, "role": role},
                headers,
            )

            observation = task_start.get("observation", {})
            world_state = task_start.get("world_state", {})
            period_row["schema_version"] = world_state.get("schema_version")
            period_row["policy_version"] = world_state.get("policy_version")

            submit = _post(
                session,
                f"{env_url}/campaign/task/submit",
                {
                    "campaign_id": campaign_id,
                    "role": role,
                    "action": {"findings": [], "submit_final": True},
                },
                headers,
            )

            submit_obs = submit.get("observation", {})
            shocks = submit_obs.get("pending_regulatory_shocks", [])
            feedback = (submit_obs.get("period_observation") or {}).get("feedback", "")

            period_row["roles"][role] = {
                "task_id": observation.get("period_observation", {}).get("task_id"),
                "reward": observation.get("period_observation", {}).get("reward"),
                "shock_count": len(shocks),
                "final_deferred": "Final submission was deferred" in feedback,
            }
            period_row["regulatory_shocks_seen"] += len(shocks)

        review = _post(
            session,
            f"{env_url}/overseer/review",
            {
                "campaign_id": campaign_id,
                "action": {
                    "audit_trail_id": f"demo-{campaign_id}-p{period}",
                    "decisions": [],
                    "conflicts_resolved": [],
                    "task_reassignments": {},
                },
            },
            headers,
        )
        period_row["overseer_decisions"] = len(review.get("result", {}).get("decisions", []))

        if period < total_periods:
            _post(
                session,
                f"{env_url}/campaign/period/advance",
                {"campaign_id": campaign_id},
                headers,
            )

        periods.append(period_row)

    state_resp = session.get(
        f"{env_url}/campaign/state",
        params={"campaign_id": campaign_id},
        headers=headers,
        timeout=60,
    )
    state_resp.raise_for_status()
    final_state = state_resp.json()

    improve = _post(
        session,
        f"{env_url}/self-improve",
        {
            "campaign_id": campaign_id,
            "train_seeds": [42, 43, 44, 45, 46],
            "held_out_seeds": [100, 101, 102],
        },
        headers,
    )

    ended_at = time.time()
    return {
        "campaign_id": campaign_id,
        "seed": seed,
        "total_periods": total_periods,
        "duration_seconds": round(ended_at - started_at, 3),
        "periods": periods,
        "final_state": final_state,
        "self_improve": improve,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic hackathon campaign demo")
    parser.add_argument("--env-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--seed", type=int, default=42, help="Campaign seed")
    parser.add_argument("--periods", type=int, default=5, help="Campaign periods")
    parser.add_argument("--api-key", default="", help="Optional X-API-Key")
    parser.add_argument(
        "--output",
        default="artifacts/hackathon_demo_summary.json",
        help="Output artifact path",
    )
    args = parser.parse_args()

    summary = run_demo(args.env_url, args.seed, args.periods, args.api_key)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "campaign_id": summary["campaign_id"],
        "output": str(output_path),
        "duration_seconds": summary["duration_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
