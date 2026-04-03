## Summary

`SLTrainer` accepts negative `grad_clip`, which makes `clip_grad_norm_` multiply gradients by a negative factor and silently reverse update direction.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Target code applies clipping directly from config without bounds checks: [`/home/john/keisei/keisei/sl/trainer.py:138`](file:///home/john/keisei/keisei/sl/trainer.py:138), [`/home/john/keisei/keisei/sl/trainer.py:139`](file:///home/john/keisei/keisei/sl/trainer.py:139).
- `SLConfig` has no validation on `grad_clip`: [`/home/john/keisei/keisei/sl/trainer.py:18`](file:///home/john/keisei/keisei/sl/trainer.py:18)–[`/home/john/keisei/keisei/sl/trainer.py:30`](file:///home/john/keisei/keisei/sl/trainer.py:30).
- Runtime check in this environment confirms negative clipping flips sign:
  - Before clip: grad `2.0`
  - After `torch.nn.utils.clip_grad_norm_([p], -0.5)`: grad `-0.49999976`

## Root Cause Hypothesis

`grad_clip` is treated as trusted input. PyTorch clipping does not reject negative `max_norm`; it computes a negative clip coefficient, so gradients are scaled and sign-inverted. Any run with `grad_clip <= 0` can produce unstable or adversarial optimization behavior without an explicit error.

## Suggested Fix

Add explicit validation in `SLTrainer.__init__` (or `SLConfig.__post_init__`) requiring `grad_clip > 0`, and fail fast with a clear `ValueError` (e.g., `"grad_clip must be > 0"`).
---
## Summary

`SLTrainer` allows `total_epochs <= 0`, but uses it as `CosineAnnealingLR(T_max=total_epochs)`, which can trigger scheduler divide-by-zero behavior once stepping starts.

## Severity

- Severity: minor
- Priority: P2

## Evidence

- Scheduler created directly from unvalidated config: [`/home/john/keisei/keisei/sl/trainer.py:66`](file:///home/john/keisei/keisei/sl/trainer.py:66), [`/home/john/keisei/keisei/sl/trainer.py:67`](file:///home/john/keisei/keisei/sl/trainer.py:67).
- Scheduler is stepped each non-empty epoch: [`/home/john/keisei/keisei/sl/trainer.py:151`](file:///home/john/keisei/keisei/sl/trainer.py:151), [`/home/john/keisei/keisei/sl/trainer.py:152`](file:///home/john/keisei/keisei/sl/trainer.py:152).
- No validation exists for `total_epochs` in `SLConfig`: [`/home/john/keisei/keisei/sl/trainer.py:23`](file:///home/john/keisei/keisei/sl/trainer.py:23).
- PyTorch scheduler implementation uses division/modulo by `self.T_max` in `get_lr`:
  - [`/home/john/keisei/.venv/lib/python3.13/site-packages/torch/optim/lr_scheduler.py` (CosineAnnealingLR `get_lr`, expressions with `/ self.T_max` and `% (2 * self.T_max)`)](file:///home/john/keisei/.venv/lib/python3.13/site-packages/torch/optim/lr_scheduler.py)

## Root Cause Hypothesis

`SLTrainer` assumes `total_epochs` is always a valid positive integer. When misconfigured to `0` (or negative), the scheduler math is invalid and can fail at runtime, creating a late, non-obvious configuration crash path.

## Suggested Fix

Validate `total_epochs >= 1` in `SLTrainer.__init__` (or `SLConfig.__post_init__`) before constructing the scheduler, and raise a clear `ValueError` on invalid input.
