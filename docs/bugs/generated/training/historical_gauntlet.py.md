## Summary

`HistoricalGauntlet.run_gauntlet()` passes a potentially stale/configured `self.num_envs` into `play_match()` even when reusing a caller-provided `vecenv`, which can mismatch the actual VecEnv size and cause gauntlet matches to fail (or silently skip via exception handling).

## Severity

- Severity: major
- Priority: P1

## Evidence

- Target file passes configured `self.num_envs` to `play_match()` when `vecenv` may be externally provided:
  - `/home/john/keisei/keisei/training/historical_gauntlet.py:138-142`
- Tournament reuses one shared VecEnv and can size it from concurrency config (not tournament `num_envs`):
  - `/home/john/keisei/keisei/training/tournament.py:149-157`
- Gauntlet is constructed with `num_envs=config.league.tournament_num_envs`:
  - `/home/john/keisei/keisei/training/katago_loop.py:608-615`
- Concurrent mode is always wired into tournament construction:
  - `/home/john/keisei/keisei/training/katago_loop.py:617-634`
- Default configs differ (`tournament_num_envs=64` vs `concurrency.total_envs=32`), making mismatch likely by default:
  - `/home/john/keisei/keisei/config.py:235-239`
  - `/home/john/keisei/keisei/config.py:272`
- `play_batch()` assumes `num_envs` matches VecEnv outputs; mismatch can produce indexing/shape faults:
  - `/home/john/keisei/keisei/training/match_utils.py:125-127` (obs/masks from vecenv reset)
  - `/home/john/keisei/keisei/training/match_utils.py:128` (`current_players = np.zeros(num_envs, ...)`)
  - `/home/john/keisei/keisei/training/match_utils.py:166` (`obs[a_indices]` indexing based on mask built from `num_envs`)

## Root Cause Hypothesis

`HistoricalGauntlet` treats `self.num_envs` as authoritative even when it did not create the VecEnv. In tournament integration, gauntlet reuses a VecEnv whose size may be overridden by concurrency settings, so gauntlet’s configured env count can diverge from the VecEnv’s real batch dimension. This triggers runtime errors inside match execution; those are caught and logged per slot, resulting in gauntlet effectively doing no useful work.

## Suggested Fix

In `/home/john/keisei/keisei/training/historical_gauntlet.py`, determine effective env count from the actual `vecenv` when one is provided, and pass that value to `play_match()` instead of unconditional `self.num_envs`.

Concrete approach:
1. If gauntlet creates VecEnv locally, keep using `self.num_envs`.
2. If `vecenv` is provided, infer env count from `vecenv.num_envs` (or a safe probe of reset output shape if no attribute exists), validate it is positive, and use that for `play_match(...)`.
3. Optionally log a warning when inferred env count differs from configured `self.num_envs` to surface misconfiguration early.
