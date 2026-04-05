## Summary

`sl_to_rl()` checks only architecture compatibility against `rl_config_path`, but it does not enforce or inherit compatible `model.params`, so SL can train/save with one shape and then fail during RL resume with a `load_state_dict` size-mismatch.

## Severity

- Severity: major
- Priority: P1

## Evidence

- [`/home/john/keisei/keisei/training/transition.py:67`](/home/john/keisei/keisei/training/transition.py:67) validates only `rl_config_early.model.architecture != architecture`.
- [`/home/john/keisei/keisei/training/transition.py:78`](/home/john/keisei/keisei/training/transition.py:78) builds SL model from `build_model(architecture, model_params or {})` (can differ from RL config params, or default unexpectedly when `model_params=None`).
- [`/home/john/keisei/keisei/training/transition.py:113`](/home/john/keisei/keisei/training/transition.py:113) later loads RL config and creates `KataGoTrainingLoop`, which builds model from RL config params.
- Resume path loads checkpoint into RL-built model at [`/home/john/keisei/keisei/training/checkpoint.py:114`](/home/john/keisei/keisei/training/checkpoint.py:114) via `model.load_state_dict(...)`; parameter-shape mismatch raises at this point.
- Existing tests only cover architecture mismatch, not params mismatch: [`/home/john/keisei/tests/test_sl_to_rl_error_paths.py:194`](/home/john/keisei/tests/test_sl_to_rl_error_paths.py:194).

## Root Cause Hypothesis

The transition logic treats “same architecture” as sufficient for checkpoint compatibility, but in this codebase tensor shapes also depend on architecture parameters (`channels`, `num_blocks`, head sizes, etc.). If `rl_config_path` params and `model_params` diverge (or `model_params` is omitted), SL checkpoint tensors no longer match the RL model shape.

## Suggested Fix

In `sl_to_rl()` Phase 0, when `rl_config_path` is provided:

- Resolve an effective SL param set:
  - If `model_params is None`, use `rl_config_early.model.params` for SL model construction.
  - If `model_params` is provided, require exact compatibility with `rl_config_early.model.params` (or canonicalized equality) and raise `ValueError` on mismatch before training.
- Use that effective param set consistently for `build_model(...)`.
- Add a regression test for `rl_config_path` + mismatched `model_params` to verify fail-fast behavior before any SL epoch runs.
