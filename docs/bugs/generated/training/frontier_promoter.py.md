## Summary

When `frontier_entries` is empty, `evaluate()` still restricts candidates to `topk_entries`, which contradicts the documented/implemented seeding rule that only `min_games_for_promotion` should matter.

## Severity

- Severity: major
- Priority: P1

## Evidence

- In [`/home/john/keisei/keisei/training/frontier_promoter.py:54`]( /home/john/keisei/keisei/training/frontier_promoter.py:54 ), `evaluate()` slices candidates to top-K:
  - `topk_entries = sorted_dynamics[: self.config.topk]`
- In [`/home/john/keisei/keisei/training/frontier_promoter.py:69`]( /home/john/keisei/keisei/training/frontier_promoter.py:69 ), only those top-K are checked:
  - `for entry in topk_entries: ...`
- In [`/home/john/keisei/keisei/training/frontier_promoter.py:90`]( /home/john/keisei/keisei/training/frontier_promoter.py:90 ), `should_promote()` explicitly says empty frontier bypasses criteria 2-5 and returns `True` after only min-games check.
- Therefore, any calibrated Dynamic entry outside top-K is never considered for seeding, despite the empty-frontier bypass rule.

## Root Cause Hypothesis

Promotion policy is split between `evaluate()` and `should_promote()`, but the empty-frontier special case was implemented only in `should_promote()`. The caller-side top-K prefilter in `evaluate()` unintentionally reintroduces criterion #2 during seeding.

## Suggested Fix

In `evaluate()`, handle empty-frontier seeding before top-K filtering, e.g. iterate all `sorted_dynamics` (highest Elo first) and return the first entry with `games_played >= min_games_for_promotion` (or equivalently call `should_promote()` across all dynamics when `frontier_entries` is empty). Keep top-K restriction only for non-empty frontier cases.
