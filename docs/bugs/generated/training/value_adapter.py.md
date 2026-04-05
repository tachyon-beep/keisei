## Summary

`MultiHeadValueAdapter` accepts unbounded `lambda_value`, `lambda_score`, and `score_blend_alpha`, so a bad config can silently invert or destabilize the training objective.

## Severity

- Severity: major
- Priority: P2

## Evidence

- Constructor stores raw hyperparameters with no validation: [value_adapter.py](/home/john/keisei/keisei/training/value_adapter.py#L65), [value_adapter.py](/home/john/keisei/keisei/training/value_adapter.py#L67), [value_adapter.py](/home/john/keisei/keisei/training/value_adapter.py#L68), [value_adapter.py](/home/john/keisei/keisei/training/value_adapter.py#L69).
- These values are used directly in loss/blending math:
  - `return self.lambda_value * value_loss + self.lambda_score * score_loss` at [value_adapter.py](/home/john/keisei/keisei/training/value_adapter.py#L118).
  - `return (1 - alpha) * wdl_value + alpha * score_value` at [value_adapter.py](/home/john/keisei/keisei/training/value_adapter.py#L88).
- Upstream PPO param validation does not constrain these fields either (only gamma/lambda/etc.): [katago_ppo.py](/home/john/keisei/keisei/training/katago_ppo.py#L102).
- Caller passes config values straight into adapter: [katago_loop.py](/home/john/keisei/keisei/training/katago_loop.py#L508), [katago_loop.py](/home/john/keisei/keisei/training/katago_loop.py#L510), [katago_loop.py](/home/john/keisei/keisei/training/katago_loop.py#L511), [katago_loop.py](/home/john/keisei/keisei/training/katago_loop.py#L512).

## Root Cause Hypothesis

The adapter assumes config sanitation is guaranteed elsewhere, but neither the adapter nor `KataGoPPOParams.__post_init__` enforces safe ranges. If `lambda_score < 0` or `lambda_value < 0`, optimization can push predictions away from targets; if `score_blend_alpha` is outside `[0, 1]`, value blending extrapolates and can produce out-of-range bootstrap values.

## Suggested Fix

Add explicit validation in `MultiHeadValueAdapter.__init__`:

- Require `lambda_value >= 0`
- Require `lambda_score >= 0`
- Require `0.0 <= score_blend_alpha <= 1.0`

Example change location: [value_adapter.py](/home/john/keisei/keisei/training/value_adapter.py#L65).  
Optionally mirror the same checks in `KataGoPPOParams.__post_init__` for earlier config-time failure.
