## Summary

`compile_mode` path can run `compiled_eval` in train mode, so rollout inference mutates BatchNorm statistics and violates eval/inference semantics.

## Severity

- Severity: major
- Priority: P1

## Evidence

- `/home/john/keisei/keisei/training/katago_ppo.py:242-244` correctly notes compile traces on first forward call.
- `/home/john/keisei/keisei/training/katago_ppo.py:258-264` sets `eval()` before creating `compiled_eval`, but immediately restores `train()`; no forward call happens here, so eval mode is not yet “captured”.
- `/home/john/keisei/keisei/training/katago_ppo.py:362-365` uses `compiled_eval` without calling `eval()`.
- `/home/john/keisei/keisei/training/katago_ppo.py:353-358` comment claims first `compiled_eval` call is eval-mode, but rollout calls happen from training loop while model is in train mode.
- `/home/john/keisei/keisei/training/katago_loop.py:524` calls `self.ppo.select_actions(...)` during rollout before `update()`, i.e., first compiled eval call can occur while model is in train mode.

## Root Cause Hypothesis

The code assumes `torch.compile()` captures module training/eval mode at compile-wrapper creation time; in reality, capture happens at first forward execution. Because `compiled_eval` is first executed without forcing `eval()`, it can compile/use train-mode behavior and update BN running stats during rollout.

## Suggested Fix

In `katago_ppo.py`, force eval mode around every `compiled_eval` execution (and restore prior mode afterward), e.g. in `select_actions()` and value-metrics eval path:
1. Save `prev_training = self.forward_model.training`
2. `self.forward_model.eval()`
3. Call `self.compiled_eval(...)`
4. Restore `self.forward_model.train(prev_training)` (or conditional restore)

Also update the stale comment at lines 353-358 to reflect true first-call semantics.
---
## Summary

`KataGoPPOParams` has no bounds validation, allowing invalid hyperparameters (for example `batch_size <= 0`) that crash or silently degenerate training in `update()`.

## Severity

- Severity: minor
- Priority: P2

## Evidence

- `/home/john/keisei/keisei/training/katago_ppo.py:54-70` defines `KataGoPPOParams` with no `__post_init__` validation.
- `/home/john/keisei/keisei/training/katago_ppo.py:534-550` uses `batch_size` directly in `range(0, total_samples, batch_size)`; `batch_size == 0` raises `ValueError: range() arg 3 must not be zero`.
- `/home/john/keisei/keisei/training/algorithm_registry.py:30-39` only checks constructor kwargs/types; it does not enforce numeric ranges.

## Root Cause Hypothesis

Parameter validation is limited to dataclass construction, but numeric invariants (positive batch size, valid clip/discount ranges, positive LR/grad clip, etc.) are not enforced at the boundary. Invalid configs can reach the hot training loop and fail late.

## Suggested Fix

Add `__post_init__` to `KataGoPPOParams` (in `katago_ppo.py`) to enforce constraints early, at minimum:
- `batch_size > 0`
- `epochs_per_batch > 0`
- `0.0 <= gamma <= 1.0`
- `0.0 <= gae_lambda <= 1.0`
- `clip_epsilon >= 0.0`
- `learning_rate > 0.0`
- `grad_clip > 0.0`
- non-negative loss coefficients (`lambda_*`) and positive `score_normalization`
