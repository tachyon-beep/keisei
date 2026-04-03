## Summary

`evaluate.py` misattributes game outcomes because it treats terminal reward as color-based (`black`/`white`) while `VecEnv` defines reward from the perspective of the player who made the last move.

## Severity

- Severity: major
- Priority: P1

## Evidence

- In `/home/john/keisei/shogi-engine/crates/shogi-gym/src/vec_env.rs:94-99`, `compute_reward` is documented as: reward for “the player who just moved” (`last_mover`).
- In `/home/john/keisei/shogi-engine/crates/shogi-gym/src/vec_env.rs:99-120`, the sign is computed against `last_mover`, not fixed Black/White perspective.
- In `/home/john/keisei/keisei/training/evaluate.py:131-132`, terminal reward is converted to A’s perspective using only `a_is_black`:
  - `reward = float(step_result.rewards[0])`
  - `a_reward = reward if a_is_black else -reward`
- This ignores who actually made the terminal move in that game, so wins/losses can be flipped.

## Root Cause Hypothesis

The evaluator assumes `step_result.rewards[0]` is always from Black’s perspective. In reality it is from the **actor who just moved**. Because terminal ply parity varies, the last mover is sometimes A and sometimes B, so static sign conversion by `a_is_black` is incorrect and corrupts W/L/D and Elo estimates.

## Suggested Fix

In `evaluate.py`, track whether A was the actor on the terminal step, and convert reward using that fact:

- Before `env.step(...)`, compute `a_to_move` from `current_player` and `a_is_black`.
- After terminal step:
  - `a_reward = reward if a_to_move else -reward`

Concrete logic:

- `a_to_move = (current_player == 0 and a_is_black) or (current_player == 1 and not a_is_black)`
- Keep `model = models[current_player]` as-is.
- Use `a_to_move` from the final loop iteration when computing terminal `a_reward`.
