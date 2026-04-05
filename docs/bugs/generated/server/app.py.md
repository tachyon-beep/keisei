## Summary

`training_status.episodes` is double-counted in the websocket loop because the same latest metrics row is re-added on every heartbeat/status update, inflating episode totals over time.

## Severity

- Severity: major
- Priority: P1

## Evidence

- In [`keisei/server/app.py`](/home/john/keisei/keisei/server/app.py:219), `total_episodes` is initialized from all historical metrics:
  - `total_episodes = sum(m.get("episodes_completed", 0) for m in metrics)`
- In the same file, on *any* training-state diff (including heartbeat changes), it re-reads one recent metric row and adds it again:
  - [`app.py:245-257`](/home/john/keisei/keisei/server/app.py:245)
  - `latest_metrics = ... read_metrics_since(... max(0, last_metrics_id - 1), 1)`
  - `total_episodes += episodes`
- The diff condition explicitly includes heartbeat changes:
  - [`app.py:250`](/home/john/keisei/keisei/server/app.py:250)
- Heartbeat is updated periodically even when no new metrics row is written:
  - [`keisei/training/katago_loop.py:1645-1649`](/home/john/keisei/keisei/training/katago_loop.py:1645)
- Metrics rows are written at epoch boundaries, and vecenv stats are reset afterward:
  - [`katago_loop.py:1498-1507`](/home/john/keisei/keisei/training/katago_loop.py:1498)
  - [`katago_loop.py:1532-1533`](/home/john/keisei/keisei/training/katago_loop.py:1532)

This means repeated heartbeat-only `training_status` pushes can keep adding the same epoch’s `episodes_completed` value multiple times.

## Root Cause Hypothesis

`total_episodes` is maintained in the wrong branch: it is incremented during training-state changes rather than only when truly new metric rows arrive. Because heartbeat changes are frequent and independent of metric insertion, the same row is counted repeatedly.

## Suggested Fix

In `/home/john/keisei/keisei/server/app.py`, move episode accumulation to the `new_metrics` branch and remove accumulation from the `training_status` branch.

Concrete change pattern:

1. Keep initialization, but null-safe:
- `total_episodes = sum((m.get("episodes_completed") or 0) for m in metrics)`

2. In `if new_metrics:` add:
- `total_episodes += sum((m.get("episodes_completed") or 0) for m in new_metrics)`

3. In the `if new_state and (...)` block:
- Delete `latest_metrics = ...`, `episodes = ...`, and `total_episodes += episodes`
- Continue sending `"episodes": total_episodes` as-is.

This ensures episode totals advance exactly once per newly observed metrics row.
