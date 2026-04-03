## Summary

In league split-merge mode, terminal outcomes that occur on opponent turns are silently dropped from PPO training because the loop records transitions only when `current_player == learner_side`, so many decisive rewards/dones never reach the learner buffer.

## Severity

- Severity: critical
- Priority: P0

## Evidence

- Learner-only storage is explicit:
  - `/home/john/keisei/keisei/training/katago_loop.py:497`
  - `/home/john/keisei/keisei/training/katago_loop.py:498-521`
- Opponent turns write no transition at all (`if li.numel() > 0:` gate), so rewards/dones on those steps are ignored for PPO:
  - `/home/john/keisei/keisei/training/katago_loop.py:499`
- Environment reward is for the **player who just moved** (`last_mover`), so terminal rewards can be emitted on opponent moves and must still influence learner training:
  - `/home/john/keisei/shogi-engine/crates/shogi-gym/src/vec_env.rs:94-99`
  - `/home/john/keisei/shogi-engine/crates/shogi-gym/src/vec_env.rs:344-357`

## Root Cause Hypothesis

The split-merge implementation treats each env step independently and only enqueues learner-turn samples, but rewards are generated every move from mover perspective. When the opponent ends a game, that terminal signal is produced on a step where `li` is empty, so the learner never gets the negative/positive terminal credit assignment for that game.

## Suggested Fix

In `katago_loop.py`, change split-merge collection to preserve learner-centric transition continuity across opponent turns. Concrete approach:

- Keep per-env pending learner transition state (obs/action/log_prob/value/legal_mask at learner move).
- On each subsequent env step:
  - Convert step reward to learner perspective (`+r` if last mover is learner, `-r` if last mover is opponent).
  - Accumulate reward into the pending learner transition for that env.
  - If episode terminates/truncates, finalize and `buffer.add(...)` that pending learner transition with terminal `done=True` and correct `value_cat`.
  - If turn returns to learner without terminal, finalize previous learner transition with `done=False` and bootstrap-ready next state linkage.
- This ensures all terminal outcomes (including opponent-turn mates) are learned.
---
## Summary

In split-merge league mode, bootstrap values are taken from side-to-move perspective but consumed as learner-perspective targets without sign correction, causing systematically wrong GAE targets when epoch ends on opponent-to-move states.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Bootstrap value is computed directly from final `obs` with no side correction:
  - `/home/john/keisei/keisei/training/katago_loop.py:559-564`
- In split-merge, learner is fixed to side `0` (black), and only learner transitions are stored:
  - `/home/john/keisei/keisei/training/katago_loop.py:470-477`
  - `/home/john/keisei/keisei/training/katago_loop.py:497-521`
- Env observations are generated from `game.position.current_player` perspective:
  - `/home/john/keisei/shogi-engine/crates/shogi-gym/src/vec_env.rs:410-421`
- GAE for split-merge uses `next_values` per env directly, without perspective transform:
  - `/home/john/keisei/keisei/training/katago_ppo.py:504-514`

## Root Cause Hypothesis

The loop assumes scalar value is directly usable for learner GAE regardless of who is to move. But the environment encodes observation from current-player perspective, so value sign flips with side-to-move. For envs ending an epoch on opponent turn, `next_values` should be negated before learner-centric GAE.

## Suggested Fix

In `katago_loop.py`, before `self.ppo.update(...)`, adjust bootstrap values in split-merge mode using `current_players`:

- Build a sign tensor: `+1` where `current_players == learner_side`, `-1` otherwise.
- Multiply `next_values` by this sign so GAE bootstrap is always learner-perspective.
- Keep no-league path unchanged.

Example logic location: right after `/home/john/keisei/keisei/training/katago_loop.py:563` and before `/home/john/keisei/keisei/training/katago_loop.py:575`.
