## Summary

Non-transactional checkpoint/optimizer file writes can leave orphaned files when an outer transaction rolls back, causing DB/filesystem divergence in `OpponentStore`.

## Severity

- Severity: major
- Priority: P1

## Evidence

- `OpponentStore.transaction()` only commits/rolls back at outermost depth (`keisei/training/opponent_store.py:271-283`), so inner operations may succeed on disk before final DB outcome.
- `add_entry()` writes checkpoint file inside the transaction before outermost commit (`keisei/training/opponent_store.py:300-321`), and only cleans up on immediate DB-update failure (`:322-331`), not on later outer rollback.
- `clone_entry()` copies checkpoint file inside transaction (`keisei/training/opponent_store.py:344-368`) with cleanup only for immediate DB-update failure (`:369-376`), not outer rollback.
- `save_optimizer()` writes optimizer file before commit (`keisei/training/opponent_store.py:764-773`) with no rollback cleanup.
- Existing test already exercises outer rollback after `add_entry()` (`tests/test_opponent_store.py:202-209`), proving this path is real; it asserts DB row removal but does not verify checkpoint cleanup (`:209`).

## Root Cause Hypothesis

File I/O side effects are executed inside nested transactions, but rollback semantics are DB-only and triggered at outermost scope; there is no deferred on-commit finalization or on-rollback cleanup for filesystem artifacts. Any exception after `add_entry()/clone_entry()/save_optimizer()` in an enclosing transaction triggers this mismatch.

## Suggested Fix

Implement transaction-scoped filesystem hooks in `OpponentStore`:

1. Track pending file operations per transaction depth (e.g., `self._pending_fs_ops` stack).
2. In `add_entry/clone_entry/save_optimizer`, write to temp paths and register:
   - `on_commit`: finalize/rename temp -> final.
   - `on_rollback`: delete temp/final artifacts created in this transaction.
3. In `transaction()`:
   - On outermost commit success, execute accumulated `on_commit` hooks.
   - On outermost rollback, execute accumulated `on_rollback` hooks.
4. Add regression tests that wrap these methods in an outer `with store.transaction(): ... raise RuntimeError(...)` and assert no checkpoint/optimizer file remains on disk.
