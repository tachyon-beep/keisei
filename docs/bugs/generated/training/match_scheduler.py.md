## Summary

`MatchScheduler.effective_ratios()` can divide by zero (or produce invalid sampling weights) when configured role ratios are mathematically valid in aggregate but invalid for the currently non-empty tiers, causing runtime failure or biased opponent selection in league training.

## Severity

- Severity: major
- Priority: P1

## Evidence

- `keisei/training/match_scheduler.py:71-79`:
  - Builds `non_empty` from currently populated roles, then computes `total = sum(non_empty.values())`, then divides by `total` without guarding `total <= 0`.
- `keisei/training/match_scheduler.py:33-41`:
  - Uses only `w > 0` weights for `random.choices(...)`, which can hide invalid negative normalized weights rather than rejecting bad configuration for active tiers.
- `keisei/config.py:138-147`:
  - `MatchSchedulerConfig` only validates that all three ratios sum to `1.0`; it does not require each ratio to be non-negative.
- `keisei/training/katago_loop.py:770-774`:
  - `sample_for_learner(...)` is called in the epoch loop; failure here aborts training progress.
- `tests/test_config.py:259-321`:
  - Tests cover sum-to-one tolerance but do not cover negative per-tier ratios or active-tier cancellation cases.

## Root Cause Hypothesis

Scheduler normalization assumes that ratios for the currently available tiers always form a positive distribution. That assumption is false when one or more configured ratios are negative (still summing to 1 globally) and some tiers are empty at runtime; active-tier weights can sum to zero or become invalid, triggering division-by-zero or pathological sampling behavior.

## Suggested Fix

In `keisei/training/match_scheduler.py` (primary fix location), add defensive validation in `effective_ratios()` before normalization:

- Reject non-finite or negative active-tier weights with a clear `ValueError`.
- Reject `total <= 0` for active tiers with a clear `ValueError` (instead of dividing).
- Optionally re-normalize after filtering to strictly positive weights in `sample_for_learner()` and error if none remain.

Example fix shape (in `effective_ratios`):

- Validate each `w` in `non_empty.values()` is finite and `>= 0`.
- Compute `total`; if `total <= 0`, raise `ValueError("Active learner tier ratios must sum to > 0 ...")`.
- Then normalize as today.
