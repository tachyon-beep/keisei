## Summary

Unhandled database write exceptions in `katago_loop.py` can crash the training loop during normal heartbeat/snapshot updates, turning observability failures into full training aborts.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Heartbeat path does not catch DB exceptions:
  - `/home/john/keisei/keisei/training/katago_loop.py:1704` (`_maybe_update_heartbeat`)
  - `/home/john/keisei/keisei/training/katago_loop.py:1710` directly calls `update_training_progress(...)` with no `try/except`.
- Snapshot path does not catch DB exceptions:
  - `/home/john/keisei/keisei/training/katago_loop.py:1714` (`_maybe_write_snapshots`)
  - `/home/john/keisei/keisei/training/katago_loop.py:1752` directly calls `write_game_snapshots(...)` with no `try/except`.
- These helpers are invoked in the rollout hot loop:
  - `/home/john/keisei/keisei/training/katago_loop.py:1307-1308`
- DB layer functions propagate sqlite errors (no internal exception suppression):
  - `/home/john/keisei/keisei/db.py:367` (`update_training_progress`)
  - `/home/john/keisei/keisei/db.py:241` (`write_game_snapshots`)
- In the same file, other DB writes are already treated as best-effort and guarded (showing intended resilience):
  - `/home/john/keisei/keisei/training/katago_loop.py:1594-1605`
  - `/home/john/keisei/keisei/training/katago_loop.py:1620-1637`

## Root Cause Hypothesis

`katago_loop.py` inconsistently handles DB failures: some writes are explicitly best-effort with logging, but heartbeat/snapshot helpers in the hot path are not guarded. Under transient SQLite contention or I/O errors, exceptions bubble up and terminate training even though these writes are non-critical telemetry/state updates.

## Suggested Fix

In `katago_loop.py`, make `_maybe_update_heartbeat` and `_maybe_write_snapshots` best-effort, matching the rest of the file:

- Wrap `update_training_progress(...)` in `_maybe_update_heartbeat` with `try/except Exception` and `logger.exception(...)` (or warning-level log).
- Wrap `write_game_snapshots(...)` in `_maybe_write_snapshots` with `try/except Exception` and log, then continue training.
- Keep existing timing/rate-limit behavior unchanged; only prevent non-critical DB failures from aborting RL training.
