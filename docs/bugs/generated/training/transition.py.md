## Summary

`sl_to_rl()` can silently skip SL checkpoint resume when `rl_config_path` is provided, because it writes resume metadata to `db_path` but `KataGoTrainingLoop` reads from `rl_config.display.db_path`.

## Severity

- Severity: major
- Priority: P1

## Evidence

- In [`/home/john/keisei/keisei/training/transition.py:97`](/home/john/keisei/keisei/training/transition.py:97), the transition state is always written to the function argument `db_path`:
  - `init_db(db_path)`
  - `write_training_state(db_path, {... "checkpoint_path": str(ckpt_path) ...})`
- In [`/home/john/keisei/keisei/training/transition.py:113`](/home/john/keisei/keisei/training/transition.py:113), if `rl_config_path` is set, `rl_config = load_config(rl_config_path)` is used.
- `KataGoTrainingLoop` then reads resume state from its own `self.db_path` (loaded from config) in [`/home/john/keisei/keisei/training/katago_loop.py:363`](/home/john/keisei/keisei/training/katago_loop.py:363): `state = read_training_state(self.db_path)`.
- If those DB paths differ, `_check_resume()` sees no checkpoint and takes fresh-start path at [`/home/john/keisei/keisei/training/katago_loop.py:413`](/home/john/keisei/keisei/training/katago_loop.py:413), so SL weights are not loaded despite `resume_mode="sl"`.

## Root Cause Hypothesis

`sl_to_rl()` computes/writes resume metadata before resolving the effective RL config DB path, and it assumes `db_path` is the same DB that the RL loop will read. This assumption fails when `rl_config_path` specifies a different `[display].db_path` (or default path), causing a split-brain state store.

## Suggested Fix

In `transition.py`, resolve/build `rl_config` before writing training state, then write transition state to the same DB that `KataGoTrainingLoop` will read (`rl_config.display.db_path`). Optionally warn or raise if caller passed `db_path` that disagrees with config when `rl_config_path` is provided.
