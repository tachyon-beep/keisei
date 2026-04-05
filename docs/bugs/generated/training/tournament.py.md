## Summary

Concurrent tournament mode updates `PriorityScorer` per game instead of per match, which conflicts with sequential mode and collapses under-sampling priority after one high-game match.

## Severity

- Severity: major
- Priority: P1

## Evidence

- `/home/john/keisei/keisei/training/tournament.py:211` comment in sequential path explicitly states `record_result` should be called once per match (to avoid inflating `_under_sample_bonus` denominator).
- `/home/john/keisei/keisei/training/tournament.py:215` sequential path calls `record_result(...)` exactly once per played match.
- `/home/john/keisei/keisei/training/tournament.py:357` to `/home/john/keisei/keisei/training/tournament.py:360` concurrent path does:
  - `for _ in range(total): self.scheduler.priority_scorer.record_result(...)`
- `/home/john/keisei/keisei/training/priority_scorer.py:43` to `:47` uses `1.0 / (count + 1)` for under-sample bonus, so inflating `count` by `games_per_match` (often 64) massively suppresses pair priority after a single match.

## Root Cause Hypothesis

A behavior mismatch was introduced between sequential and concurrent tournament code paths: sequential code was updated to per-match scorer accounting, but concurrent code retained per-game accounting. This is triggered whenever `concurrent_pool` is enabled.

## Suggested Fix

In `/home/john/keisei/keisei/training/tournament.py` inside `_run_concurrent_round`, replace the per-game loop with a single `record_result(...)` call per non-empty match result (mirroring sequential path), then keep `record_round_result(...)` once and `advance_round()` once per round.
---
## Summary

In concurrent mode, one exception while processing a single match result can crash the entire tournament thread instead of isolating failure to that pairing.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Sequential path has per-match fault isolation:
  - `/home/john/keisei/keisei/training/tournament.py:195` to `:205` wraps `_run_one_match(...)` in `try/except` and continues.
- Concurrent path lacks equivalent per-result isolation:
  - `/home/john/keisei/keisei/training/tournament.py:295` to `:347` processes each result (DB writes, Elo update, role-elo update, dynamic trainer update) with no local `try/except`.
- `_run_loop` has a top-level catch that terminates the thread on any uncaught exception:
  - `/home/john/keisei/keisei/training/tournament.py:247` to `:248` logs `"Tournament thread crashed"`.

## Root Cause Hypothesis

Error handling granularity differs between sequential and concurrent branches. In concurrent mode, exceptions from `store.record_result`, `store.update_elo`, `role_elo_tracker.update_from_result`, or `dynamic_trainer.update` bubble out of `_run_concurrent_round` and trigger thread-level termination.

## Suggested Fix

Add per-result `try/except` inside `_run_concurrent_round` around result post-processing (Elo/result recording + dynamic update), log the failing pairing, and continue processing remaining results. Keep thread-level catch as a last resort, not primary control flow.
