"""Test HTTP endpoints of the running server."""
import json
import requests

BASE = "http://127.0.0.1:8000"

print("=" * 50)
print(" HTTP Endpoint Tests")
print("=" * 50)

# 1. Health
r = requests.get(f"{BASE}/health")
assert r.status_code == 200
print(f"GET /health → {r.status_code} {r.json()}")

# 2. Tasks
r = requests.get(f"{BASE}/tasks")
assert r.status_code == 200
data = r.json()
assert data["total_tasks"] == 4
print(f"GET /tasks → {r.status_code}, {data['total_tasks']} tasks")
for t in data["tasks"]:
    print(f"  - {t['id']}: {t['name']} ({t['difficulty']})")

# 3. Reset
r = requests.post(f"{BASE}/reset", json={"task_id": "expense_audit", "seed": 42})
assert r.status_code == 200
obs = r.json()["observation"]
print(f"POST /reset → task={obs['task_id']}, docs={len(obs['documents']['expenses'])} expenses")

# 4. Step with findings
r = requests.post(f"{BASE}/step", json={
    "action": {
        "findings": [
            {
                "document_id": "EXP-010",
                "error_type": "over_limit",
                "description": "Meal 4500 exceeds 1500 limit",
            },
            {
                "document_id": "EXP-011",
                "error_type": "wrong_category",
                "description": "Personal earbuds as Office Supplies",
            }
        ],
        "submit_final": True,
    }
})
assert r.status_code == 200
result = r.json()
print(f"POST /step → done={result['done']}, reward={result['reward']}")

# 5. Grader
r = requests.get(f"{BASE}/grader")
assert r.status_code == 200
grader = r.json()
print(f"GET /grader → score={grader['score']}, P={grader['precision']}, R={grader['recall']}")

# 6. Test all 4 tasks work
for task_id in ["expense_audit", "invoice_match", "gst_reconciliation", "fraud_detection"]:
    r = requests.post(f"{BASE}/reset", json={"task_id": task_id, "seed": 42})
    assert r.status_code == 200
    print(f"POST /reset({task_id}) → {r.status_code} ✓")

# 7. Rate limit test (should not be limited at 7 requests)
print("\nAll endpoint tests passed ✓")
