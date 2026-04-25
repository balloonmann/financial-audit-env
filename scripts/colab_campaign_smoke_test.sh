#!/usr/bin/env bash
set -euo pipefail

# Colab/CI-friendly campaign smoke test for Round 2 endpoints.
#
# Configurable env vars:
#   ENV_URL=http://127.0.0.1:8000
#   SEED=42
#   PERIODS=3
#   OUTPUT=artifacts/campaign_smoke_summary.json
#   API_KEY=
#   START_SERVER=1
#   RUN_SCORE_VERIFY=0
#   RUN_SELF_IMPROVE=1
#   TRAIN_SEEDS=42,43,44,45,46
#   HELD_OUT_SEEDS=100,101,102

ENV_URL="${ENV_URL:-http://127.0.0.1:8000}"
SEED="${SEED:-42}"
PERIODS="${PERIODS:-3}"
OUTPUT="${OUTPUT:-artifacts/campaign_smoke_summary.json}"
API_KEY="${API_KEY:-}"
START_SERVER="${START_SERVER:-1}"
RUN_SCORE_VERIFY="${RUN_SCORE_VERIFY:-0}"
RUN_SELF_IMPROVE="${RUN_SELF_IMPROVE:-1}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42,43,44,45,46}"
HELD_OUT_SEEDS="${HELD_OUT_SEEDS:-100,101,102}"

mkdir -p "$(dirname "$OUTPUT")"

echo "[1/6] Installing local dependencies"
python -m pip install --upgrade pip
python -m pip install -e . requests httpx pytest

SERVER_PID=""
if [ "$START_SERVER" = "1" ]; then
  echo "[2/6] Starting local API server"
  python -m uvicorn financial_audit_env.server.app:app --host 127.0.0.1 --port 8000 >/tmp/campaign_smoke_server.log 2>&1 &
  SERVER_PID=$!
  trap 'if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" >/dev/null 2>&1 || true; fi' EXIT
else
  echo "[2/6] Reusing existing server at $ENV_URL"
fi

echo "[3/6] Waiting for /health"
python - <<'PY'
import os
import time
import requests

env_url = os.environ["ENV_URL"]
last_err = None
for _ in range(60):
    try:
        r = requests.get(f"{env_url}/health", timeout=5)
        if r.status_code == 200:
            print("health ok")
            break
    except Exception as exc:
        last_err = exc
    time.sleep(1)
else:
    raise SystemExit(f"health check failed: {last_err}")
PY

echo "[4/6] Running campaign smoke flow via run_hackathon_demo.py"
CMD=(python scripts/run_hackathon_demo.py --env-url "$ENV_URL" --seed "$SEED" --periods "$PERIODS" --output "$OUTPUT")
if [ -n "$API_KEY" ]; then
  CMD+=(--api-key "$API_KEY")
fi
"${CMD[@]}"

echo "[5/6] Validating output and extracting campaign_id"
CAMPAIGN_ID=$(python - <<'PY'
import json
import os

output = os.environ["OUTPUT"]
periods = int(os.environ["PERIODS"])

with open(output, "r", encoding="utf-8") as f:
    data = json.load(f)

assert data.get("campaign_id"), "missing campaign_id"
assert data.get("total_periods") == periods, f"expected total_periods={periods}"
assert len(data.get("periods", [])) == periods, "period count mismatch"
assert data.get("final_state"), "missing final_state"
assert data.get("self_improve"), "missing self_improve summary"
print(data["campaign_id"])
PY
)

echo "campaign_id=$CAMPAIGN_ID"

if [ "$RUN_SELF_IMPROVE" = "1" ]; then
  echo "[6/6] Running explicit self-improve call with custom seeds"
  python - <<'PY'
import json
import os
import requests


def parse_seed_csv(text: str):
    items = [x.strip() for x in text.split(",") if x.strip()]
    return [int(x) for x in items]

env_url = os.environ["ENV_URL"]
campaign_id = os.environ["CAMPAIGN_ID"]
api_key = os.environ.get("API_KEY", "")
train_seeds = parse_seed_csv(os.environ["TRAIN_SEEDS"])
held_out_seeds = parse_seed_csv(os.environ["HELD_OUT_SEEDS"])

headers = {"Content-Type": "application/json"}
if api_key:
    headers["X-API-Key"] = api_key

payload = {
    "campaign_id": campaign_id,
    "train_seeds": train_seeds,
    "held_out_seeds": held_out_seeds,
}

r = requests.post(f"{env_url}/self-improve", json=payload, headers=headers, timeout=60)
r.raise_for_status()
result = r.json()

state = requests.get(
    f"{env_url}/campaign/state", params={"campaign_id": campaign_id}, headers=headers, timeout=60
)
state.raise_for_status()

history = requests.get(
    f"{env_url}/self-improve/history", params={"campaign_id": campaign_id}, headers=headers, timeout=60
)
history.raise_for_status()

artifact = {
    "campaign_id": campaign_id,
    "self_improve_custom": result,
    "campaign_state": state.json(),
    "self_improve_history": history.json(),
}

os.makedirs("artifacts", exist_ok=True)
out = "artifacts/campaign_smoke_checks.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(artifact, f, indent=2)

summary = {
    "campaign_id": campaign_id,
    "accepted": result.get("summary", {}).get("accepted"),
    "train_delta": result.get("summary", {}).get("train_delta"),
    "transfer_delta": result.get("summary", {}).get("transfer_delta"),
    "history_count": len(history.json().get("history", [])),
    "artifact": out,
}
print(json.dumps(summary, indent=2))
PY
else
  echo "[6/6] Skipping custom self-improve call (RUN_SELF_IMPROVE=0)"
fi

if [ "$RUN_SCORE_VERIFY" = "1" ]; then
  echo "[extra] Running verify_r2_score.py"
  python verify_r2_score.py
fi

echo "Done. Smoke artifacts:"
echo "- $OUTPUT"
if [ "$RUN_SELF_IMPROVE" = "1" ]; then
  echo "- artifacts/campaign_smoke_checks.json"
fi
