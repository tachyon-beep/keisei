## Summary

`SLDataset.__getitem__` returns `policy_target`/`value_target` without bounds validation, so a malformed or corrupted shard record can crash SL training inside `cross_entropy` instead of failing fast at the data boundary.

## Severity

- Severity: major
- Priority: P2

## Evidence

- Target file returns raw decoded labels directly:
  - `/home/john/keisei/keisei/sl/dataset.py:181`
  - `/home/john/keisei/keisei/sl/dataset.py:183`
  - `/home/john/keisei/keisei/sl/dataset.py:189`
  - `/home/john/keisei/keisei/sl/dataset.py:190`
- The same file only warns on shard corruption (`trailing bytes`) and still ingests records:
  - `/home/john/keisei/keisei/sl/dataset.py:113`
  - `/home/john/keisei/keisei/sl/dataset.py:114`
  - `/home/john/keisei/keisei/sl/dataset.py:120`
- Downstream training assumes valid class indices and calls `cross_entropy` directly:
  - `/home/john/keisei/keisei/sl/trainer.py:133`
  - `/home/john/keisei/keisei/sl/trainer.py:136`
- `torch.nn.functional.cross_entropy` requires targets in `[0, C-1]`; out-of-range values cause runtime errors (e.g., “Target X is out of bounds”).

## Root Cause Hypothesis

The dataset layer treats shard bytes as trusted and does not enforce semantic invariants for categorical labels (`value_target` should be `{0,1,2}`, `policy_target` should be within policy head class count). Under shard corruption or bad upstream encoding, invalid labels propagate until the trainer’s loss call fails mid-epoch.

## Suggested Fix

Add explicit target validation in `/home/john/keisei/keisei/sl/dataset.py` before returning tensors, with a clear exception that includes shard path and index context. Concretely:

- Validate `value_target in {0,1,2}` in `__getitem__`.
- Validate `policy_target >= 0`; optionally also validate upper bound via a constructor arg (e.g., `max_policy_index`) or metadata from shard creation.
- On violation, raise `ValueError` (or a custom data-integrity exception) with `shard_path`, `idx`, and decoded values so failures are actionable and localized to data loading.
