# Hackathon Core Idea

## One-Line Thesis
Build a live multi-agent financial-audit control tower where specialist agents continuously audit evolving books, adapt to surprise regulation shocks, and are governed by an overseer plus self-improvement loop with strict anti-regression gates.

## Problem Being Solved
Most LLM audit demos solve one static snapshot. Real finance operations are dynamic:
- Policies change mid-cycle.
- Data schemas drift.
- New vendors and fraud patterns emerge.
- Teams need traceable oversight, not single-agent guesswork.

## Product Narrative
The platform is an "AI Internal Audit War Room" with five components:
1. Specialist swarm: Expense, Invoice, GST, and Fraud specialists collaborate in dependency order.
2. World mutation engine: Every period changes policy, schema, and risk posture.
3. Regulatory shock injector: New rules drop mid-period and force live adaptation.
4. Overseer governance: A reviewer agent approves/rejects/escalates findings and resolves conflicts.
5. Self-improve gate: Candidate policy updates are only accepted with transfer-safe gains.

## Why This Is Distinctive
- Beyond benchmark F1: captures operational resilience and adaptation quality.
- Governance-first: introduces a concrete review trail and escalation logic.
- Anti-gaming: campaign scoring punishes narrow optimization and missed critical issues.
- Practical training loop: reward, evaluator, and held-out gate are wired for fast iteration.

## Core Demo Story (6-8 minutes)
1. Start campaign with seeded reproducibility.
2. Run specialists through period 1 in normal mode.
3. Trigger period 3+ where schema drift appears.
4. Hit a regulatory shock on a GST step and show forced post-shock adaptation.
5. Run overseer review to accept/reject/escalate.
6. Execute one self-improvement iteration and show gate decision with train/transfer deltas.
7. Close with campaign score, risk coverage, and safety posture.

## Winning Criteria Mapping
- Innovation: multi-agent + live regulation adaptation + governance.
- Technical depth: deterministic environment, rich scoring, anti-regression controls.
- Reliability: reproducible seeds, targeted security hardening, extensive tests.
- Practicality: clear path to enterprise audit workflows.

## North-Star Metrics
- Campaign completion rate with no critical misses.
- Specialist weighted-F1 floor per period.
- Shock adaptation latency (steps until compliant post-shock behavior).
- Overseer precision on conflict resolution.
- Transfer-safe self-improvement acceptance rate.

## Immediate Build Priorities
1. Finalize deterministic campaign and shock semantics.
2. Harden authentication and write-path abuse resistance.
3. Add training-pipeline tests for parser/reward/evaluator correctness.
4. Run seed-split self-improvement experiments and capture transfer deltas.
5. Prepare a scripted demo runbook with expected outputs.
