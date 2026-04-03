## Summary

`read_game_snapshots_since()` can permanently miss valid game snapshot updates because it uses a second-resolution timestamp cursor with a strict `>` filter.

## Severity

- Severity: major
- Priority: P1

## Evidence

- [`/home/john/keisei/keisei/db.py:63`](/home/john/keisei/keisei/db.py:63): `updated_at` default uses `strftime('%Y-%m-%dT%H:%M:%SZ', 'now')` (second precision).
- [`/home/john/keisei/keisei/db.py:210`](/home/john/keisei/keisei/db.py:210): incremental read uses `WHERE updated_at > ?`.
- [`/home/john/keisei/keisei/db.py:215`](/home/john/keisei/keisei/db.py:215): next cursor is `max(updated_at)` from returned rows.
- [`/home/john/keisei/keisei/server/app.py:214`](/home/john/keisei/keisei/server/app.py:214) and [`/home/john/keisei/keisei/server/app.py:218`](/home/john/keisei/keisei/server/app.py:218): websocket poll loop advances `last_game_ts` to returned `max_ts`.
- [`/home/john/keisei/tests/test_db_gaps.py:167`](/home/john/keisei/tests/test_db_gaps.py:167): test explicitly codifies that `updated_at == since` is excluded.

If new rows are written later in the same second as `last_game_ts`, they are `== since`, not `> since`, and are skipped forever once time cursor advances.

## Root Cause Hypothesis

The incremental protocol uses a non-unique cursor (`updated_at` at second granularity) as if it were strictly monotonic. Under normal polling/write timing, multiple writes can share one timestamp bucket, so strict `>` drops late arrivals in that bucket.

## Suggested Fix

In `db.py`, change snapshot change-tracking to a stable, strictly increasing cursor, e.g.:
- Add `updated_seq INTEGER PRIMARY KEY AUTOINCREMENT` (or separate monotonic column) for snapshot writes, and query by `updated_seq > ?`.
- Or return/use a composite cursor `(updated_at, game_id)` and query with tie-break logic:
  `WHERE updated_at > ? OR (updated_at = ? AND game_id > ?)` with `ORDER BY updated_at, game_id`.

Also upgrade timestamp precision to fractional seconds (`%f`) for `updated_at` defaults to reduce collisions, but do not rely on precision alone as the sole fix.
---
## Summary

`init_db()` does not perform schema migrations for existing databases, so `SCHEMA_VERSION = 2` can coexist with v1 table layouts and cause runtime SQL failures in v2 write paths.

## Severity

- Severity: major
- Priority: P1

## Evidence

- [`/home/john/keisei/keisei/db.py:8`](/home/john/keisei/keisei/db.py:8): declares `SCHEMA_VERSION = 2`.
- [`/home/john/keisei/keisei/db.py:25`](/home/john/keisei/keisei/db.py:25): schema setup uses `CREATE TABLE IF NOT EXISTS ...` only; this does not add missing columns to existing tables.
- [`/home/john/keisei/keisei/db.py:110`](/home/john/keisei/keisei/db.py:110)-[`113`](/home/john/keisei/keisei/db.py:113): version logic only inserts version when missing; it never migrates older versions.
- [`/home/john/keisei/keisei/db.py:165`](/home/john/keisei/keisei/db.py:165)-[`171`](/home/john/keisei/keisei/db.py:171): `write_game_snapshots()` writes v2 columns (`game_type`, `demo_slot`, `opponent_id`) that are absent in older schemas.
- [`/home/john/keisei/keisei/db.py:278`](/home/john/keisei/keisei/db.py:278)-[`280`](/home/john/keisei/keisei/db.py:280): `update_training_progress(..., phase=...)` assumes `training_state.phase` exists.

## Root Cause Hypothesis

The code treats schema creation as migration. `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` are idempotent for new DBs but do not reconcile structural drift in existing DB files. Version metadata is not used to apply migration steps.

## Suggested Fix

Implement explicit versioned migrations in `db.py`:
- Read current schema version.
- For each step `< SCHEMA_VERSION`, apply `ALTER TABLE` / backfill / new-table creation guarded by column-existence checks (`PRAGMA table_info(...)`).
- Update `schema_version` after each successful migration inside one transaction.
- Add migration tests that start from a synthetic v1 DB and validate v2 writes (`write_game_snapshots`, `update_training_progress(phase=...)`) succeed without manual DB reset.
