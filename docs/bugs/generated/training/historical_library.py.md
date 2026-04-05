## Summary

`refresh()` incorrectly treats all candidates as “within proximity threshold” when a target’s neighbor distance is zero, causing distant checkpoints to be accepted in pass 1 as `log_spaced` instead of being deferred to fallback logic.

## Severity

- Severity: minor
- Priority: P2

## Evidence

- In [`historical_library.py`](/home/john/keisei/keisei/training/historical_library.py#L82), threshold is computed as `neighbor_dists[i] * 0.5`.
- In [`historical_library.py`](/home/john/keisei/keisei/training/historical_library.py#L83), filtering is gated by `if threshold > 0 and distance > threshold: continue`.
- When targets contain duplicates, `_neighbor_distances()` returns `0` for those slots (see [`historical_library.py`](/home/john/keisei/keisei/training/historical_library.py#L246) and [`historical_library.py`](/home/john/keisei/keisei/training/historical_library.py#L250)).
- Duplicate targets are known/expected from `_compute_targets()` at small epochs (see test assertion `targets == [1, 1, 1, 2, 2]` in [`test_historical_library.py`](/home/john/keisei/tests/test_historical_library.py#L53)).
- Therefore for `threshold == 0`, the current guard skips rejection entirely, so any `distance` is accepted in pass 1.

## Root Cause Hypothesis

The proximity check was written to avoid over-rejecting when thresholds are non-positive, but for duplicate targets a zero threshold should mean “exact match only.” Because zero is treated as a bypass case, distant candidates can be consumed in pass 1, which weakens the intended two-pass behavior and can misclassify selection as `log_spaced`.

## Suggested Fix

In [`historical_library.py`](/home/john/keisei/keisei/training/historical_library.py#L83), remove the `threshold > 0` gate and enforce the threshold unconditionally:

- Change:
  - `if threshold > 0 and distance > threshold:`
- To:
  - `if distance > threshold:`

This makes `threshold == 0` behave correctly (only `distance == 0` allowed in pass 1), preserving fallback behavior for distant candidates.
