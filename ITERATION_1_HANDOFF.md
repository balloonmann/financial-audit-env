# Iteration 1 Handoff

Date: 2026-04-25

## What Was Completed

- Low-VRAM training pipeline executed in Colab and adapter artifact uploaded.
- Interim held-out baseline evaluation executed with Qwen 1.5B (T4-safe path).
- Baseline artifacts exported to Drive and uploaded to HF dataset artifacts repo.

## Produced Artifact Links

- Adapter repo: https://huggingface.co/balloonmann/financial-audit-grpo-adapter
- Eval artifacts repo: https://huggingface.co/datasets/balloonmann/financial-audit-eval-artifacts

## Interim Baseline Metrics (Held-out 100-104)

- expense_audit: score 0.01, weighted_score 0.01, precision 0.01, recall 0.01
- invoice_match: score 0.01, weighted_score 0.01, precision 0.01, recall 0.01
- gst_reconciliation: score 0.01, weighted_score 0.01, precision 0.01, recall 0.01
- fraud_detection: score 0.01, weighted_score 0.01, precision 0.01, recall 0.01

## Important Interpretation

- This interim pass validates the evaluation/export pipeline and artifact handling.
- It is not the final benchmark comparison for judging.
- Final comparison should be same-family:
  - Llama baseline
  - Llama + uploaded adapter

## Competition-Day Finalization Checklist

- Add public Colab notebook link to README Submission Hub.
- Run final Llama baseline vs Llama+adapter on held-out seeds.
- Export final comparison table and plots, then replace interim placeholders.
- Publish HF blog (or video/slides) and add link in README.
- Verify all links in incognito before submission.
