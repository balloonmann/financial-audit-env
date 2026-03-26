"""Quick verification script for the Financial Audit Environment."""
import sys
sys.path.insert(0, ".")

print("=" * 50)
print(" Financial Audit Environment — Verification")
print("=" * 50)

# Test 1: Data generators
print("\n[1/4] Testing data generators...")
from financial_audit_env.server.data_generator import (
    generate_expense_data,
    generate_invoice_data,
    generate_gst_data,
)

from typing import Dict, List, Any

d1: Dict[str, Any]; g1: List[Dict[str, Any]]
d1, g1 = generate_expense_data(42)
assert len(d1["expenses"]) == 15, f"Expected 15 expenses, got {len(d1['expenses'])}"
assert len(g1) == 6, f"Expected 6 errors, got {len(g1)}"
print(f"  Expense data: {len(d1['expenses'])} entries, {len(g1)} errors ✓")

d2: Dict[str, Any]; g2: List[Dict[str, Any]]
d2, g2 = generate_invoice_data(42)
assert len(g2) == 8, f"Expected 8 errors, got {len(g2)}"
print(f"  Invoice data: {len(d2['invoices'])} invoices, {len(g2)} errors ✓")

d3: Dict[str, Any]; g3: List[Dict[str, Any]]
d3, g3 = generate_gst_data(42)
assert len(g3) == 12, f"Expected 12 errors, got {len(g3)}"
print(f"  GST data: {len(d3['purchase_register'])} books + {len(d3['gstr2b'])} gstr2b, {len(g3)} errors ✓")

# Reproducibility check
d1b, g1b = generate_expense_data(42)
assert d1 == d1b, "Reproducibility check failed!"
print("  Reproducibility: same seed = same data ✓")

# Test 2: Graders
print("\n[2/4] Testing graders...")
from financial_audit_env.server.graders import compute_f1_score, compute_step_reward

# Perfect score
result = compute_f1_score(g1, g1)
assert result["score"] == 1.0, f"Expected 1.0, got {result['score']}"
print(f"  Perfect findings → score={result['score']} ✓")

# Empty findings
result = compute_f1_score([], g1)
assert result["score"] == 0.0, f"Expected 0.0, got {result['score']}"
print(f"  No findings → score={result['score']} ✓")

# Partial findings
result = compute_f1_score(g1[:3], g1)
assert 0.0 < result["score"] < 1.0
print(f"  Partial findings → score={result['score']:.4f} ✓")

# Test 3: Environment
print("\n[3/4] Testing environment reset/step/state cycle...")
from financial_audit_env.server.environment import FinancialAuditEnvironment
from financial_audit_env.models import AuditAction, Finding

env = FinancialAuditEnvironment()

# Reset
obs = env.reset(task_id="expense_audit", seed=42)
assert obs.done == False
assert obs.task_id == "expense_audit"
assert len(obs.documents["expenses"]) == 15
print(f"  reset() → task={obs.task_id}, {len(obs.documents['expenses'])} expenses ✓")

# Step with a correct finding
action = AuditAction(
    findings=[
        Finding(
            document_id=g1[0]["document_id"],
            error_type=g1[0]["error_type"],
            description="Test finding",
        )
    ],
    submit_final=False,
)
obs = env.step(action)
assert obs.done == False
assert obs.reward > 0  # Should get +0.15 - 0.02 = 0.13
print(f"  step(1 correct) → reward={obs.reward:.4f}, done={obs.done} ✓")

# Step with submit_final
action = AuditAction(
    findings=[
        Finding(
            document_id=g1[i]["document_id"],
            error_type=g1[i]["error_type"],
            description="Test finding",
        )
        for i in range(1, len(g1))
    ],
    submit_final=True,
)
obs = env.step(action)
assert obs.done == True
print(f"  step(final) → reward={obs.reward:.4f}, done={obs.done} ✓")

# State
state = env.state
assert state.task_id == "expense_audit"
assert state.found_errors == 6
print(f"  state → found={state.found_errors}/{state.total_errors}, FP={state.false_positives} ✓")

# Grader
grader = env.last_grader_result
assert grader is not None
assert grader["score"] == 1.0
print(f"  grader → F1={grader['score']:.4f} ✓")

# Test 4: All tasks
print("\n[4/4] Testing all tasks...")
for task_id in ["expense_audit", "invoice_match", "gst_reconciliation"]:
    obs = env.reset(task_id=task_id, seed=42)
    assert not obs.done
    assert obs.task_id == task_id
    assert len(obs.documents) > 0
    print(f"  {task_id}: reset OK, {obs.max_steps} max steps ✓")

print("\n" + "=" * 50)
print(" ALL TESTS PASSED ✓")
print("=" * 50)
