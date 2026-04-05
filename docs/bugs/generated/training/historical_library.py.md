## Summary

`HistoricalLibrary` does not prune or filter obsolete DB rows when `history.slots` is reduced, so stale historical slots remain active and can be consumed as if they were current.

## Severity

- Severity: major
- Priority: P2

## Evidence

- [`/home/john/keisei/keisei/training/historical_library.py:45`](/home/john/keisei/keisei/training/historical_library.py:45) `refresh()` only writes slot indices `0..self.config.slots-1` via `upsert_historical_slot(...)`; it never removes previously stored higher slot indices.
- [`/home/john/keisei/keisei/training/historical_library.py:114`](/home/john/keisei/keisei/training/historical_library.py:114) `get_slots()` returns every row from `store.get_historical_slots()` with no cap/filter by `self.config.slots`.
- [`/home/john/keisei/keisei/training/opponent_store.py:738`](/home/john/keisei/keisei/training/opponent_store.py:738) `get_historical_slots()` does `SELECT ... FROM historical_library ... ORDER BY h.slot_index` and returns all rows.
- [`/home/john/keisei/keisei/training/historical_gauntlet.py:78`](/home/john/keisei/keisei/training/historical_gauntlet.py:78) gauntlet runs over all non-empty provided slots, so stale slots can trigger unintended extra benchmark matches.

## Root Cause Hypothesis

Historical slot state is persisted in a table keyed by `slot_index`, but `HistoricalLibrary` assumes slot cardinality is static. When config changes (for example from 5 slots to 3), refresh updates only the first 3 records and leaves old rows untouched; subsequent reads include both new and stale rows.

## Suggested Fix

In `historical_library.py`, enforce configured slot cardinality during refresh/read:

1. At start of `refresh()`, remove rows with `slot_index >= self.config.slots` (either by a new `OpponentStore.delete_historical_slots_from(start_index)` method or an equivalent store API).
2. In `get_slots()`, defensively filter to `row["slot_index"] < self.config.slots` before building `HistoricalSlot` objects.

This keeps runtime behavior aligned with current config and prevents stale-slot gauntlet matches.
