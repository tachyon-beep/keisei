## Summary

`ConcurrentMatchPool.run_round()` can send uninitialized/default action `0` to VecEnv environments that are outside partition coverage when `total_envs > parallel_matches * envs_per_match`, causing illegal-action runtime failures.

## Severity

- Severity: major
- Priority: P1

## Evidence

- In `/home/john/keisei/keisei/training/concurrent_matches.py:257`, `actions` is allocated for all `total_envs`.
- Active partitions only fill their own ranges (`/home/john/keisei/keisei/training/concurrent_matches.py:260-315`).
- Inactive fill only iterates `range(self.config.parallel_matches)` (`/home/john/keisei/keisei/training/concurrent_matches.py:325-336`), so it never covers indices `>= parallel_matches * envs_per_match`.
- `ConcurrencyConfig` allows `needed_envs < total_envs` (only rejects `needed_envs > total_envs`) in `/home/john/keisei/keisei/config.py:359-364`.
- VecEnv validates every action and raises if illegal (`/home/john/keisei/shogi-engine/crates/shogi-gym/src/vec_env.rs:654-659`, `:681-687`).

## Root Cause Hypothesis

The implementation assumes `total_envs == parallel_matches * envs_per_match`, but config validation permits larger `total_envs`; extra env indices never receive a legal fallback action and may carry default `0`, which can be illegal and trigger `VecEnv.step()` exceptions.

## Suggested Fix

In `run_round()` (`concurrent_matches.py`), add an upfront invariant check and/or full fallback coverage:
- Validate `self.config.total_envs == self.config.parallel_matches * self.config.envs_per_match` and fail fast with a clear error.
- Or additionally fill all uncovered env indices (`range(covered_envs, total_envs)`) with first legal action from `legal_masks.argmax(dim=-1)` before `vecenv.step()`.
---
## Summary

`ConcurrentMatchPool.run_round()` lacks exception-safe cleanup, so model resources may not be released if inference/step fails mid-round, and the tournament thread can die.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Model release in `run_round()` is only on normal completion paths (`/home/john/keisei/keisei/training/concurrent_matches.py:417-427`, `:454-463`).
- There is no `try/finally` guarding the game loop/body in `run_round()` (`/home/john/keisei/keisei/training/concurrent_matches.py:242-465`).
- `vecenv.step()` can raise runtime errors (e.g., illegal action) per engine contract (`/home/john/keisei/shogi-engine/crates/shogi-gym/src/vec_env.rs:681-687`).
- In tournament integration, `run_round()` is called directly (`/home/john/keisei/keisei/training/tournament.py:293-303`); if it raises, outer loop logs crash and stops thread (`/home/john/keisei/keisei/training/tournament.py:261-264`).
- Sequential path uses `try/finally` for release (`/home/john/keisei/keisei/training/tournament.py:501-510`), highlighting missing parity in concurrent path.

## Root Cause Hypothesis

`run_round()` was written for normal-path completion/stop-event cleanup but not hardened for exceptions from model forward or `vecenv.step()`, so cleanup logic is bypassed and the calling tournament thread aborts.

## Suggested Fix

Refactor `run_round()` in `concurrent_matches.py` to use exception-safe cleanup:
- Wrap assignment + main loop in `try`/`finally`.
- In `finally`, iterate all slots and call `release_fn` for any non-`None` loaded models, then null refs.
- Optionally catch per-slot/per-step exceptions, mark slot failed, and continue remaining pairings instead of aborting whole round.
