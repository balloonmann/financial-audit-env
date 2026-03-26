"""Debug the /step endpoint."""
import requests, json

BASE = "http://127.0.0.1:8000"

# Reset first
r = requests.post(f"{BASE}/reset", json={"task_id": "expense_audit", "seed": 42})
print(f"RESET status: {r.status_code}")
obs = r.json()
print(f"RESET keys: {list(obs.keys())}")

# Step 
r = requests.post(f"{BASE}/step", json={
    "action": {
        "findings": [
            {
                "document_id": "EXP-010",
                "error_type": "over_limit",
                "description": "Meal 4500 exceeds 1500 limit",
            }
        ],
        "submit_final": True,
    }
})
print(f"\nSTEP status: {r.status_code}")
print(f"STEP body: {r.text[:500]}")
