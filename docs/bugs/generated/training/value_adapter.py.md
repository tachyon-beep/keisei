## Summary

No concrete bug found in /home/john/keisei/keisei/training/value_adapter.py

## Severity

- Severity: trivial
- Priority: P3

## Evidence

- `/home/john/keisei/keisei/training/value_adapter.py:98-118` validates required tensors (`value_cats`, `score_targets`, `score_pred`) and guards the all-ignored-label case to avoid `cross_entropy` NaN (`value_output.sum() * 0.0` fallback).
- `/home/john/keisei/keisei/training/value_adapter.py:71-89` scalar projection and blended projection are shape-consistent with callers (`(batch, 3)` logits and `(batch, 1)` score lead).
- `/home/john/keisei/keisei/training/katago_ppo.py:717-724` calls adapter with GPU-resident tensors that match expected dtypes/shapes (`value_cats` from long labels, score tensors as floats).
- `/home/john/keisei/keisei/training/katago_ppo.py:192-222` rollout buffer validation enforces value category domain and non-NaN/normalized score targets before training.
- `/home/john/keisei/keisei/training/katago_loop.py:427-431` ensures `katago_ppo` runs only with `se_resnet` multi-head architecture, matching adapter assumptions.
- `/home/john/keisei/tests/test_value_adapter.py:38-173` includes direct coverage for loss path, ignore-index behavior, all-ignored case, and score-blend behavior.

## Root Cause Hypothesis

No bug identified.

## Suggested Fix

Unknown
