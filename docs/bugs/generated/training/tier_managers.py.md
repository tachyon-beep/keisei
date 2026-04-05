## Summary

`RecentFixedConfig.retire_if_below_dynamic_floor` is a dead configuration flag: `RecentFixedManager.review_oldest()` ignores it, so entries below the dynamic Elo floor are retired regardless of this setting.

## Severity

- Severity: minor
- Priority: P2

## Evidence

- Config defines and exposes the flag: `retire_if_below_dynamic_floor: bool = True` in [`/home/john/keisei/keisei/config.py:90`](/home/john/keisei/keisei/config.py:90).
- The review logic in [`/home/john/keisei/keisei/training/tier_managers.py`](/home/john/keisei/keisei/training/tier_managers.py) never references `retire_if_below_dynamic_floor`.
- In `review_oldest()`:
  - Elo floor check is computed at [`tier_managers.py:270-278`](/home/john/keisei/keisei/training/tier_managers.py:270).
  - Delay eligibility is computed from `under_calibrated = not games_ok or not opponents_ok or not stable_ok` at [`tier_managers.py:304`](/home/john/keisei/keisei/training/tier_managers.py:304), which explicitly excludes Elo-floor failure.
  - If Elo is the only failing criterion, code falls through to unconditional RETIRE at [`tier_managers.py:320-327`](/home/john/keisei/keisei/training/tier_managers.py:320).
- No other references to `retire_if_below_dynamic_floor` exist in repo (`rg` only finds config/tests/docs declaration sites).

## Root Cause Hypothesis

The config field was added to the schema/TOML surface but never wired into `RecentFixedManager.review_oldest()` decision branches. As a result, users can set `retire_if_below_dynamic_floor = false` with no behavioral effect, causing silent policy mismatch.

## Suggested Fix

In `RecentFixedManager.review_oldest()` (`tier_managers.py`), add an explicit branch for `elo_qualified` failure that honors `self._config.retire_if_below_dynamic_floor`:

- If `elo_qualified` is false and `retire_if_below_dynamic_floor` is `True`: keep current RETIRE behavior.
- If `elo_qualified` is false and `retire_if_below_dynamic_floor` is `False`: treat as DELAY-eligible (subject to soft overflow), or explicitly DELAY if that is intended policy.

Also add regression tests in `tests/test_tier_managers.py` for both flag values to prevent future silent no-op config fields.
