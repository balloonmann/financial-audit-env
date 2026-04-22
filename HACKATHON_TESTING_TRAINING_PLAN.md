# Testing and Training Execution Plan

## Goal
Turn the Round 2 architecture into a competition-ready pipeline with measurable reliability and repeatable training outcomes.

## Phase 1: Stability Gate (Local)
Run focused core suites before each training batch.

```bash
python -m pytest tests/test_campaign.py tests/test_campaign_round2.py tests/test_self_improve.py tests/test_regulatory.py tests/test_adversarial.py tests/test_security.py tests/test_graders.py tests/test_environment.py -q
```

Pass condition:
- 100% pass in focused suite.
- No auth/rate-limit regressions for local test clients.

## Phase 2: Training Pipeline Correctness
Run parser/reward/evaluator tests to ensure reward signals are stable.

```bash
python -m pytest tests/test_training_pipeline.py -q
```

Pass condition:
- JSON and free-text finding parsing remains robust.
- Reward function stays clamped and monotonic with exploit penalties.
- In-process evaluator returns bounded scores and complete metrics.

## Phase 3: GRPO Dry-Run
Use local dry-run to verify component wiring before GPU spend.

```bash
python training/train_grpo.py
```

Pass condition:
- Script reaches dry-run success line.
- Dataset, parser, reward, and held-out evaluation checks all succeed.

## Phase 4: Colab Training Run
In Colab with T4:
1. Install `unsloth trl datasets peft`.
2. Run `python training/train_grpo.py`.
3. Save LoRA adapter artifacts.

Pass condition:
- Trainer executes without OOM.
- Adapter saved.
- Held-out seed report generated.

## Phase 5: Self-Improve Gate Check
Run self-improvement endpoint with disjoint seed sets and verify:
- `train_delta > 0.005`
- `transfer_delta > -0.002`
- `safety_regression == false`

Only accept candidate policy updates when all three conditions hold.

## Logging and Artifacts
Capture per run:
- Focused test summary.
- Held-out results JSON.
- Self-improve iteration history.
- Final campaign demonstration trace.

## Suggested Daily Cadence
1. Run focused tests.
2. Run parser/reward/evaluator tests.
3. Make one training change.
4. Execute dry-run or Colab run.
5. Evaluate held-out behavior.
6. Promote only transfer-safe updates.
