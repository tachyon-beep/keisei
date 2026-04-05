## Summary

`_weighted_sample()` can return more pairings than `weighted_round_size` (including the auto size), so weighted tournament rounds exceed their configured match budget.

## Severity

- Severity: major
- Priority: P1

## Evidence

- [`/home/john/keisei/keisei/training/match_scheduler.py:193`](/home/john/keisei/keisei/training/match_scheduler.py:193) sets `round_size` from config (`0` becomes `len(entries)` at line 195), implying a target size.
- [`/home/john/keisei/keisei/training/match_scheduler.py:218`](/home/john/keisei/keisei/training/match_scheduler.py:218) computes per-class allocation as `share = max(1, round(round_size * weights[mc] / total_weight))`.
- [`/home/john/keisei/keisei/training/match_scheduler.py:219`](/home/john/keisei/keisei/training/match_scheduler.py:219) extends `result` with each class share independently, with no global cap or remainder reconciliation.
- Because each present class gets at least 1, total allocated shares can exceed `round_size` whenever multiple classes are present and `round_size` is small (or due rounding overflow even at normal sizes).
- Integration impact: tournament executes every returned pairing in a loop (`for entry_a, entry_b in pairings`) at [`/home/john/keisei/keisei/training/tournament.py:210`](/home/john/keisei/keisei/training/tournament.py:210), so overshoot directly increases match workload and runtime.

## Root Cause Hypothesis

The allocator is class-local (`max(1, round(...))`) instead of budget-global. It guarantees minimum per-class representation but never enforces `sum(shares) <= round_size`. This is triggered when weighted mode is enabled and at least two weighted classes are present.

## Suggested Fix

Replace the current per-class `max(1, round(...))` allocation with a budget-constrained allocator:

1. Set `target = min(round_size, len(all_pairs))`.
2. Compute raw proportional quotas per class.
3. Take floor quotas first (bounded by each class pool size).
4. Distribute remaining quota by largest fractional remainder, only to classes with remaining capacity.
5. Remove `max(1, ...)` and enforce final `len(result) <= target`.

This keeps weighted proportions while respecting configured round-size limits.
