## Summary

`read_game_snapshots_since()` can permanently miss game snapshot updates because it uses a strict timestamp-only cursor (`updated_at > since`) even though `updated_at` is not guaranteed unique across writes.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Timestamp cursor is write-side generated in [`/home/john/keisei/keisei/db.py:241`](file:///home/john/keisei/keisei/db.py:241) via `strftime(...)` per row, but no uniqueness guarantee is added.
- Incremental read uses only strict timestamp comparison in [`/home/john/keisei/keisei/db.py:280`](file:///home/john/keisei/keisei/db.py:280):
  - `SELECT * FROM game_snapshots WHERE updated_at > ? ORDER BY game_id`
- Cursor advancement also tracks only max timestamp in [`/home/john/keisei/keisei/db.py:283`](file:///home/john/keisei/keisei/db.py:283)-[`285`](file:///home/john/keisei/keisei/db.py:285), so any later write with the same `updated_at` value is excluded forever.
- Consumer loop in [`/home/john/keisei/keisei/server/app.py:235`](file:///home/john/keisei/keisei/server/app.py:235)-[`240`](file:///home/john/keisei/keisei/server/app.py:240) advances `last_game_ts` directly from that returned max timestamp, so skipped rows are never retried.

## Root Cause Hypothesis

The DB API implements incremental sync with a single scalar time cursor and strict `>` filtering. Under close-together writes (or any equal timestamp collision), one batch can advance the cursor to `T`, and a subsequent write at the same `T` is then filtered out permanently.

## Suggested Fix

Use a stable composite cursor in `db.py` (timestamp + deterministic tie-breaker), for example `(updated_at, game_id)`:

- Query:
  - `WHERE updated_at > ? OR (updated_at = ? AND game_id > ?)`
  - `ORDER BY updated_at, game_id`
- Return both `max_updated_at` and `max_game_id_at_that_timestamp` from `read_game_snapshots_since`.
- Update `write_game_snapshots`/reader contract so incremental consumers can resume with that composite cursor and avoid loss on timestamp ties.
