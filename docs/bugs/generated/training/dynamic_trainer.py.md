## Summary

`DynamicTrainer.update()` can report a failed update and trigger error fallback even after model weights were already committed to disk, creating inconsistent training/checkpoint state.

## Severity

- Severity: major
- Priority: P1

## Evidence

- In [`dynamic_trainer.py`](/home/john/keisei/keisei/training/dynamic_trainer.py#L313), checkpoint write is committed (`torch.save(...tmp)` then `rename` at [L314](/home/john/keisei/keisei/training/dynamic_trainer.py#L314)).
- After that commit, additional operations that can raise still run: optimizer flush at [L332-L334](/home/john/keisei/keisei/training/dynamic_trainer.py#L332), DB metadata update at [L336](/home/john/keisei/keisei/training/dynamic_trainer.py#L336), and cache bookkeeping through [L341](/home/john/keisei/keisei/training/dynamic_trainer.py#L341).
- Any exception in that post-checkpoint region is caught by the outer handler at [L194-L217](/home/john/keisei/keisei/training/dynamic_trainer.py#L194), which marks the update as failed (`return False`), increments error counters, and may disable the entry.
- Called methods can realistically raise:
  - [`OpponentStore.save_optimizer()`](/home/john/keisei/keisei/training/opponent_store.py#L759) performs file+DB writes.
  - [`OpponentStore.increment_update_count()`](/home/john/keisei/keisei/training/opponent_store.py#L802) performs DB writes.

## Root Cause Hypothesis

The method mixes irreversible checkpoint commit and fallible post-commit bookkeeping inside one broad `try/except`. If a late exception occurs, control flows into the “update failed” path even though weights were already changed on disk, causing false failure accounting and potential entry disablement on successful training steps.

## Suggested Fix

Split update flow into explicit phases in `dynamic_trainer.py`:

1. Training + prepare artifacts.
2. Commit checkpoint/optimizer/metadata in a controlled order with narrow exception handling.
3. Only treat as “failed update” if checkpoint was not committed.
4. If failure occurs after checkpoint commit, log as partial bookkeeping failure (do not increment consecutive training-error counter or disable entry).

At minimum, guard post-checkpoint operations separately so `update()` cannot return `False` after checkpoint rename already succeeded.
