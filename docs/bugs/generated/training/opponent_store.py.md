## Summary

`save_optimizer()` does not register `metadata.json` for rollback, so a failed outer transaction can leave on-disk metadata inconsistent with rolled-back DB/file state.

## Severity

- Severity: major
- Priority: P2

## Evidence

- `save_optimizer()` only registers overwrite protection for `optimizer.pt`, not `metadata.json`:
  - `/home/john/keisei/keisei/training/opponent_store.py:938`
  - `/home/john/keisei/keisei/training/opponent_store.py:949`
- Rollback only restores paths tracked in `_pending_fs_ops`:
  - `/home/john/keisei/keisei/training/opponent_store.py:332`
- There is an existing nested-transaction rollback path that exercises this pattern:
  - `/home/john/keisei/tests/test_opponent_store.py:260`

## Root Cause Hypothesis

`save_optimizer()` updates DB + rewrites `metadata.json` inside a transaction, but only `optimizer.pt` is tracked for rollback. If `save_optimizer()` is called inside an outer transaction and that outer transaction aborts, DB and optimizer file roll back, but metadata content can remain at the uncommitted value.

## Suggested Fix

In `save_optimizer()`, register metadata overwrite before rewriting it, e.g. compute `meta_path = entry_dir / "metadata.json"` and call `_register_overwrite(meta_path)` before `_write_metadata(...)`, so rollback restores prior metadata content as well.
---
## Summary

`update_role()` changes `role` but does not update `training_enabled`, violating the store’s own role-to-trainability invariant.

## Severity

- Severity: minor
- Priority: P3

## Evidence

- Role-based trainability is explicitly set on insert/clone:
  - `/home/john/keisei/keisei/training/opponent_store.py:380`
  - `/home/john/keisei/keisei/training/opponent_store.py:444`
- `update_role()` updates only `role`:
  - `/home/john/keisei/keisei/training/opponent_store.py:514`
- `bootstrap_from_flat_pool()` relies on `update_role()` for mass role transitions (including to `DYNAMIC`):
  - `/home/john/keisei/keisei/training/tiered_pool.py:201`
  - `/home/john/keisei/keisei/training/tiered_pool.py:233`

## Root Cause Hypothesis

The implementation treats `training_enabled` as a role-derived field at creation time, but role transitions later do not keep that derived field synchronized. This can leave entries marked `role='dynamic'` with stale `training_enabled=0` (or vice versa).

## Suggested Fix

Update `update_role()` SQL to also set `training_enabled` consistently (e.g., `1` for `Role.DYNAMIC`, else `0`) in the same transaction.
