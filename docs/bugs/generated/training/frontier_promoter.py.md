## Summary

`FrontierPromoter.should_promote()` skips lineage-overlap enforcement when the candidate has `lineage_group=None`, allowing repeated promotions from the same source lineage and defeating the configured diversity cap.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Target-file logic only counts overlap when `candidate.lineage_group` is truthy:
  - `/home/john/keisei/keisei/training/frontier_promoter.py:114`
  - `/home/john/keisei/keisei/training/frontier_promoter.py:118`
  - `/home/john/keisei/keisei/training/frontier_promoter.py:121`
- New entries are commonly created without `lineage_group`, so candidates can legitimately have `None`:
  - `/home/john/keisei/keisei/training/opponent_store.py:307`
  - `/home/john/keisei/keisei/training/opponent_store.py:309`
- Promotion clones are assigned a synthetic lineage (`source.lineage_group or f"lineage-{source_entry_id}"`), indicating lineage identity is expected even when source lineage is missing:
  - `/home/john/keisei/keisei/training/opponent_store.py:362`
- Existing test currently encodes the buggy behavior (candidate with `lineage=None` passes criterion 5):
  - `/home/john/keisei/tests/test_frontier_promoter.py:264`
  - `/home/john/keisei/tests/test_frontier_promoter.py:267`

## Root Cause Hypothesis

The lineage overlap check uses raw `candidate.lineage_group` and short-circuits when it is `None`. But the storage layer treats missing lineage as recoverable identity (`lineage-{source_id}` during clone). Because promoter does not apply the same normalization, lineage-cap logic is bypassed for many real candidates created via `add_entry()`. This triggers when dynamic candidates have null lineage (default path), especially across repeated review cycles.

## Suggested Fix

In `frontier_promoter.py`, normalize lineage before counting overlap, using the same fallback semantics as clone lineage.

Concrete change pattern:

- Compute canonical candidate lineage:
  - `candidate_lineage = candidate.lineage_group or f"lineage-{candidate.id}"`
- Compare frontier entries against this canonical value (and optionally canonicalize frontier entries too for robustness using `parent_entry_id`/`id` fallback).

Example adjustment in `should_promote()` criterion 5:

```python
candidate_lineage = candidate.lineage_group or f"lineage-{candidate.id}"
same_lineage_count = sum(
    1
    for e in frontier_entries
    if (e.lineage_group or f"lineage-{e.parent_entry_id or e.id}") == candidate_lineage
)
if same_lineage_count >= self.config.max_lineage_overlap:
    return False
```

Also update the test that currently expects `lineage=None` to always pass (`tests/test_frontier_promoter.py`) so it validates normalized-lineage enforcement instead.
