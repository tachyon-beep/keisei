## Summary

`init_db()` does not actually migrate existing databases when schema evolves, even though runtime code assumes new columns/tables exist.

## Severity

- Severity: major
- Priority: P1

## Evidence

- `SCHEMA_VERSION` is fixed at `1` and only checked for equality, not schema shape: [db.py](/home/john/keisei/keisei/db.py:9), [db.py](/home/john/keisei/keisei/db.py:183), [db.py](/home/john/keisei/keisei/db.py:188)
- `init_db()` only uses `CREATE TABLE IF NOT EXISTS` and never `ALTER TABLE` for pre-existing tables: [db.py](/home/john/keisei/keisei/db.py:26)
- `read_league_data()` unconditionally selects newer columns (`optimizer_path`, `update_count`, `last_train_at`, role Elo columns). On an older existing `league_entries` table, this raises `sqlite3.OperationalError: no such column ...`: [db.py](/home/john/keisei/keisei/db.py:397)
- Server startup comment explicitly expects pending migrations to be applied via `init_db()`: [app.py](/home/john/keisei/keisei/server/app.py:130)

## Root Cause Hypothesis

Schema evolution was handled by appending columns to `CREATE TABLE IF NOT EXISTS` definitions, but without version bumps plus forward migrations (`ALTER TABLE ... ADD COLUMN`) for already-created tables. This is triggered when reusing an older DB file with the same `schema_version` value.

## Suggested Fix

Implement real incremental migrations in `init_db()`:
- Bump `SCHEMA_VERSION` when schema changes.
- Add migration steps per version (transactional) using `PRAGMA table_info(...)` + `ALTER TABLE ... ADD COLUMN` for missing columns, and `CREATE TABLE IF NOT EXISTS` for new tables.
- Update `schema_version` only after successful migration.
- Keep `read_league_data()` unchanged once migration guarantees those columns exist.
---
## Summary

`read_game_snapshots_since()` uses lexicographic string comparison for timestamps, which is incorrect when mixed ISO formats are present (with and without fractional seconds).

## Severity

- Severity: minor
- Priority: P2

## Evidence

- Writes `updated_at` in fractional format (`...%fZ`): [db.py](/home/john/keisei/keisei/db.py:254)
- Cursor filter compares raw TEXT values: [db.py](/home/john/keisei/keisei/db.py:301)
- Cursor advancement also uses lexicographic `max(...)` on timestamp strings: [db.py](/home/john/keisei/keisei/db.py:308)
- In the server loop this function drives incremental websocket updates, so missed rows become user-visible stale state: [app.py](/home/john/keisei/keisei/server/app.py:246)

## Root Cause Hypothesis

String ordering is not chronological across mixed formats:
- `"2026-01-01T12:00:00.123Z" < "2026-01-01T12:00:00Z"` lexicographically (because `"." < "Z"`).
If `since_ts` or existing rows are second-precision while new rows are fractional, valid newer snapshots can be skipped or cursored incorrectly.

## Suggested Fix

In `read_game_snapshots_since()`:
- Compare/order by parsed time, not raw text, e.g. `julianday(updated_at)`:
  - `WHERE julianday(updated_at) > julianday(?) OR (julianday(updated_at)=julianday(?) AND game_id > ?)`
  - `ORDER BY julianday(updated_at), game_id`
- Derive next cursor from the last ordered row (or normalized timestamp + game_id), not `max()` on raw timestamp strings.
- Optionally normalize incoming `since_ts` to one canonical format before query.
