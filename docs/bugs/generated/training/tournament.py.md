## Summary

Sequential match execution leaks GPU memory if loading the second opponent model fails, because `model_a` is allocated before the `try/finally` cleanup scope is entered.

## Severity

- Severity: major
- Priority: P1

## Evidence

- In [`/home/john/keisei/keisei/training/tournament.py:498`](file:///home/john/keisei/keisei/training/tournament.py#L498) and [`/home/john/keisei/keisei/training/tournament.py:499`](file:///home/john/keisei/keisei/training/tournament.py#L499), both models are loaded before entering cleanup.
- The cleanup `finally` starts only at [`/home/john/keisei/keisei/training/tournament.py:501`](file:///home/john/keisei/keisei/training/tournament.py#L501), so an exception from `load_opponent(entry_b, ...)` bypasses `release_models(...)` at [`/home/john/keisei/keisei/training/tournament.py:510`](file:///home/john/keisei/keisei/training/tournament.py#L510).
- This exact failure mode is explicitly handled in concurrent path code (`model_a` cleanup when `model_b` load fails) at [`/home/john/keisei/keisei/training/concurrent_matches.py:454`](file:///home/john/keisei/keisei/training/concurrent_matches.py#L454)-[`/home/john/keisei/keisei/training/concurrent_matches.py:467`](file:///home/john/keisei/keisei/training/concurrent_matches.py#L467), confirming expected behavior.

## Root Cause Hypothesis

The `try/finally` is scoped only around `play_match(...)`, not around model loading. If `entry_b` checkpoint loading fails (missing/corrupt checkpoint, deserialization error), `model_a` stays resident on GPU until GC, causing cumulative VRAM pressure across repeated failures.

## Suggested Fix

Refactor `_play_match` so cleanup covers partial-load failure:
- Initialize `model_a = model_b = None`.
- Load `model_a`, then `model_b` inside a `try`.
- In `finally`, call `release_models` only for non-`None` models and clear references.
- Optionally mirror concurrent path behavior by moving `model_a` to CPU before release when `model_b` load fails.
---
## Summary

Sequential path can misclassify and train on matches using stale pre-match role state, even though comments claim role changes between pairing and completion are respected.

## Severity

- Severity: minor
- Priority: P2

## Evidence

- `is_trainable` is computed once before match execution at [`/home/john/keisei/keisei/training/tournament.py:404`](file:///home/john/keisei/keisei/training/tournament.py#L404)-[`/home/john/keisei/keisei/training/tournament.py:407`](file:///home/john/keisei/keisei/training/tournament.py#L407).
- After match, fresh entries are re-read at [`/home/john/keisei/keisei/training/tournament.py:423`](file:///home/john/keisei/keisei/training/tournament.py#L423)-[`/home/john/keisei/keisei/training/tournament.py:424`](file:///home/john/keisei/keisei/training/tournament.py#L424), but recorded `match_type` still uses stale `is_trainable` at [`/home/john/keisei/keisei/training/tournament.py:438`](file:///home/john/keisei/keisei/training/tournament.py#L438).
- Training trigger comment says role changes are respected at [`/home/john/keisei/keisei/training/tournament.py:460`](file:///home/john/keisei/keisei/training/tournament.py#L460)-[`/home/john/keisei/keisei/training/tournament.py:461`](file:///home/john/keisei/keisei/training/tournament.py#L461), but gate still uses stale `is_trainable` at [`/home/john/keisei/keisei/training/tournament.py:462`](file:///home/john/keisei/keisei/training/tournament.py#L462).
- In contrast, concurrent path recomputes match trainability from refreshed entries at [`/home/john/keisei/keisei/training/tournament.py:321`](file:///home/john/keisei/keisei/training/tournament.py#L321).

## Root Cause Hypothesis

Role transitions can occur concurrently (main trainer/thread updates league roles). Sequential path partially refetches state but does not recompute trainability from refreshed entries, so DB labeling and dynamic-training gating can diverge from current role policy.

## Suggested Fix

In `_run_one_match`, after `current_a/current_b` are fetched:
- Recompute `is_train = is_training_match(current_a, current_b)`.
- Use `is_train` for `record_result(match_type=...)` instead of stale `is_trainable`.
- Gate `dynamic_trainer.record_match(...)` and update trigger with `is_train` (or recomputed policy), not stale pre-match boolean.
