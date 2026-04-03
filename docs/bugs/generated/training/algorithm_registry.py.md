## Summary

`algorithm_registry.py` still registers legacy `"ppo"` as valid, but the active training loop only accepts `KataGoPPOParams`, so `"ppo"` passes config validation and then fails later with a type mismatch.

## Severity

- Severity: major
- Priority: P1

## Evidence

- `/home/john/keisei/keisei/training/algorithm_registry.py:22-25` registers both `"ppo"` and `"katago_ppo"`:
  - `_PARAM_SCHEMAS = {"ppo": PPOParams, "katago_ppo": KataGoPPOParams}`
- `/home/john/keisei/keisei/config.py:109-113` validates `training.algorithm` against `VALID_ALGORITHMS` imported from this registry, so `"ppo"` is accepted.
- `/home/john/keisei/keisei/training/katago_loop.py:241-247` requires the returned params to be `KataGoPPOParams` and raises otherwise:
  - `if not isinstance(ppo_params, KataGoPPOParams): raise TypeError(...)`
- Runtime verification:
  - `validate_algorithm_params("ppo", {})` returns `PPOParams` (not `KataGoPPOParams`), matching the above mismatch.

## Root Cause Hypothesis

The registry retains a stale legacy algorithm (`"ppo"`) from pre-KataGo training, while the current loop is KataGo-only. This creates a split contract: config validation says `"ppo"` is valid, but execution rejects it later.

## Suggested Fix

In `/home/john/keisei/keisei/training/algorithm_registry.py`, make the registry reflect currently supported algorithms only:

- Remove `"ppo": PPOParams` from `_PARAM_SCHEMAS` (or mark it deprecated and reject explicitly in `validate_algorithm_params` with a migration message).
- Keep only `"katago_ppo": KataGoPPOParams` in `VALID_ALGORITHMS`.

This makes failure happen early and correctly at config-validation time instead of late in loop initialization.
