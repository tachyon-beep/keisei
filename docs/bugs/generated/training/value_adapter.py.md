## Summary

No concrete bug found in /home/john/keisei/keisei/training/value_adapter.py

## Severity

- Severity: trivial
- Priority: P3

## Evidence

- [value_adapter.py](/home/john/keisei/keisei/training/value_adapter.py):51-53 correctly enforces `returns` for scalar-contract loss and computes `F.mse_loss(value_output.squeeze(-1), returns)`.
- [value_adapter.py](/home/john/keisei/keisei/training/value_adapter.py):76-81 validates required multi-head inputs (`value_cats`, `score_targets`, `score_pred`) before loss computation.
- [value_adapter.py](/home/john/keisei/keisei/training/value_adapter.py):83-90 includes the all-ignored-target guard (`value_output.sum() * 0.0`) to avoid `cross_entropy` NaN while preserving autograd connectivity.
- [value_adapter.py](/home/john/keisei/keisei/training/value_adapter.py):94-96 combines value and score losses consistently with adapter weights.
- [katago_ppo.py](/home/john/keisei/keisei/training/katago_ppo.py):611-618 passes correctly shaped/device-aligned tensors into adapter loss (`output.value_logits`, `batch_value_cats`, `batch_score_targets`, `output.score_lead`).
- [katago_ppo.py](/home/john/keisei/keisei/training/katago_ppo.py):624-641 inline fallback logic matches adapter semantics, reducing risk of contract divergence.
- [katago_loop.py](/home/john/keisei/keisei/training/katago_loop.py):97-109 computes rollout values under `torch.no_grad()` when using adapter scalar projection.
- [katago_ppo.py](/home/john/keisei/keisei/training/katago_ppo.py):136-166 buffer path validates `value_categories` domain and `score_targets` normalization/NaN before update-time loss use.

## Root Cause Hypothesis

No bug identified

## Suggested Fix

No code change recommended in /home/john/keisei/keisei/training/value_adapter.py based on current integration and call-site verification.
