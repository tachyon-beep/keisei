## Summary

`OpponentPool` is not safe to use from `DemonstratorRunner`’s background thread: it opens SQLite with default thread affinity and exposes unsynchronized mutable state, so threaded use can crash or race.

## Severity

- Severity: major
- Priority: P1

## Evidence

- [league.py#L89](/home/john/keisei/keisei/training/league.py#L89): `sqlite3.connect(self.db_path)` uses default `check_same_thread=True`.
- [league.py#L88](/home/john/keisei/keisei/training/league.py#L88): comment says “Do not share an OpponentPool instance across threads.”
- [demonstrator.py#L42](/home/john/keisei/keisei/training/demonstrator.py#L42): `DemonstratorRunner` is a `threading.Thread`.
- [demonstrator.py#L97](/home/john/keisei/keisei/training/demonstrator.py#L97): background thread calls `self.pool.list_entries()`.

## Root Cause Hypothesis

`OpponentPool` was implemented as single-threaded, but other training components are designed to call it from a background thread. Under real threaded use, SQLite can raise thread-affinity errors, and `_pinned`/DB operations can race because there is no lock.

## Suggested Fix

In `league.py`, make `OpponentPool` explicitly thread-safe:
1. Open SQLite with `check_same_thread=False`.
2. Add a `threading.RLock` and guard all `_conn` operations plus `_pinned` access.
3. Keep transaction boundaries intact inside the lock (especially `add_snapshot`, `_evict_if_needed`, `_delete_entry`, `record_result`, `update_elo`).
---
## Summary

`add_snapshot()` uses deterministic filenames (`{architecture}_ep{epoch}.pt`) without uniqueness checks, so repeated snapshots for the same architecture/epoch overwrite old checkpoint files while old DB rows remain, corrupting historical opponent identity.

## Severity

- Severity: major
- Priority: P1

## Evidence

- [league.py#L115](/home/john/keisei/keisei/training/league.py#L115): checkpoint path is only `architecture + epoch`.
- [league.py#L121](/home/john/keisei/keisei/training/league.py#L121): rename replaces existing file at that path.
- [league.py#L123](/home/john/keisei/keisei/training/league.py#L123): always inserts a new `league_entries` row, even if `checkpoint_path` already exists.
- [katago_loop.py#L334](/home/john/keisei/keisei/training/katago_loop.py#L334): league bootstrap always snapshots with `epoch=0`, so restarting league training reuses the same filename (`..._ep00000.pt`) and overwrites prior artifact.

## Root Cause Hypothesis

Filename generation assumes `(architecture, epoch)` is globally unique, but restart/resume and repeated runs can reuse epoch numbers. This causes multiple league entries to point to the same physical file (latest write), breaking replayability and Elo/history integrity.

## Suggested Fix

In `league.py`, make snapshot filenames unique per entry:
1. Include a unique suffix (UTC timestamp, UUID, or DB `entry_id`) in filename.
2. Or enforce unique `checkpoint_path` and fail fast on collision.
3. If keeping epoch in name, still append uniqueness (e.g., `{arch}_ep{epoch:05d}_{ts}.pt`).
