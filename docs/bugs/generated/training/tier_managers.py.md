## Summary

`DynamicManager` treats only `protection_remaining == 0` as evictable, so a negative protection value (allowed by current config/store paths) makes entries permanently non-evictable and can deadlock Dynamic admissions when the tier is full.

## Severity

- Severity: major
- Priority: P2

## Evidence

- In target file, eviction eligibility is hard-coded to exact zero:
  - `/home/john/keisei/keisei/training/tier_managers.py:369-371` (`evict_weakest`)
  - `/home/john/keisei/keisei/training/tier_managers.py:405-407` (`weakest_elo`)
  - `/home/john/keisei/keisei/training/tier_managers.py:418-420` (`weakest_dynamic_elo`)
- New admissions write protection directly from config without clamping:
  - `/home/john/keisei/keisei/training/tier_managers.py:344`
- Config currently does not validate `protection_matches >= 0`:
  - `/home/john/keisei/keisei/config.py:108-129`
- Store accepts negative protection counts unchanged:
  - `/home/john/keisei/keisei/training/opponent_store.py:544-550`
- Failure mode path:
  1. `protection_matches = -1` configured.
  2. New Dynamic entries get `protection_remaining = -1`.
  3. Eligibility checks require `== 0`, so these entries are never eligible.
  4. `/home/john/keisei/keisei/training/tier_managers.py:330-338` can return `None` on full tier because no entry is evictable.

## Root Cause Hypothesis

The code assumes protection counters are always non-negative and eventually land exactly on `0`, but neither config nor persistence enforces that invariant. Once a negative value exists, equality checks (`== 0`) misclassify entries as indefinitely protected.

## Suggested Fix

In `keisei/training/tier_managers.py`, treat non-positive protection as unprotected:
- Change all three eligibility checks from `e.protection_remaining == 0` to `e.protection_remaining <= 0`.
- Optionally harden admission by clamping: `max(0, self._config.protection_matches)` when calling `set_protection(...)` in `DynamicManager.admit`.
