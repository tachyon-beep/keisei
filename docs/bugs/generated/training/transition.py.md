## Summary

`sl_to_rl()` can train/save an SL checkpoint with one model architecture/parameter set, then instantiate `KataGoTrainingLoop` with a different model from `rl_config_path`, causing resume-time checkpoint load failure.

## Severity

- Severity: major
- Priority: P1

## Evidence

- In [`/home/john/keisei/keisei/training/transition.py:63`](\/home\/john\/keisei\/keisei\/training\/transition.py:63), SL model is built from `architecture` + `model_params` function args, not from `rl_config_path` model settings.
- In [`/home/john/keisei/keisei/training/transition.py:98`](\/home\/john\/keisei\/keisei\/training\/transition.py:98)-[`100`](\/home\/john\/keisei\/keisei\/training\/transition.py:100), RL config is loaded only after SL checkpoint creation.
- In [`/home/john/keisei/keisei/training/transition.py:153`](\/home\/john\/keisei\/keisei\/training\/transition.py:153), loop is created from that RL config (which may define a different model).
- On resume, loop enforces checkpoint compatibility against RL config architecture at [`/home/john/keisei/keisei/training/katago_loop.py:691`](\/home\/john\/keisei\/keisei\/training\/katago_loop.py:691) via `expected_architecture=self.config.model.architecture`.
- `load_checkpoint()` raises on architecture mismatch at [`/home/john/keisei/keisei/training/checkpoint.py:97`](\/home\/john\/keisei\/keisei\/training\/checkpoint.py:97)-[`103`](\/home\/john\/keisei\/keisei\/training\/checkpoint.py:103), and can also fail `model.load_state_dict(...)` on shape mismatch at [`114`](\/home\/john\/keisei\/keisei\/training\/checkpoint.py:114).

## Root Cause Hypothesis

`sl_to_rl()` has no config-boundary validation tying SL model construction args to `rl_config_path` model config. If they diverge (common when caller relies on config file but leaves default `architecture="se_resnet"` / `model_params=None`), the mismatch is detected only later during RL checkpoint resume, producing a hard runtime failure.

## Suggested Fix

In `sl_to_rl()`:

1. Load `rl_config` first when `rl_config_path` is provided.
2. Validate SL model spec against `rl_config.model` before any training/checkpointing:
   - `architecture == rl_config.model.architecture`
   - `model_params` (normalized) matches `rl_config.model.params`
3. Either:
   - fail fast with a clear `ValueError` if mismatched, or
   - make `rl_config.model` the single source of truth for SL model build/checkpoint metadata when `rl_config_path` is supplied.

This fix belongs in `transition.py` because that file owns the SL->RL handoff contract.
