## Summary

`ConcurrentMatchPool` can retain strong references to released models after a slot completes if the next pairing load fails, causing GPU memory to remain occupied for the rest of `run_round`.

## Severity

- Severity: major
- Priority: P1

## Evidence

- In slot completion, models are “released” but references are not cleared: [concurrent_matches.py:417](/home/john/keisei/keisei/training/concurrent_matches.py:417), [concurrent_matches.py:425](/home/john/keisei/keisei/training/concurrent_matches.py:425).
- On swap-in load failure, `_assign_pairing` sets `slot.active = False` and returns without nulling `slot.model_a` / `slot.model_b`: [concurrent_matches.py:489](/home/john/keisei/keisei/training/concurrent_matches.py:489), [concurrent_matches.py:497](/home/john/keisei/keisei/training/concurrent_matches.py:497).
- Failed swap-ins are not re-appended to `active_slots`, so stale refs can sit in `slots` until function end while other slots continue running: [concurrent_matches.py:449](/home/john/keisei/keisei/training/concurrent_matches.py:449).
- Default tournament release path calls `release_models`, which only does `torch.cuda.empty_cache()` and does not free live tensors held by references: [tournament.py:281](/home/john/keisei/keisei/training/tournament.py:281), [match_utils.py:238](/home/john/keisei/keisei/training/match_utils.py:238).

## Root Cause Hypothesis

The code assumes `release_fn` is sufficient for cleanup, but slot objects continue holding model references. If a slot finishes, then next pairing load fails, those references are never overwritten and remain alive until `run_round` returns, delaying memory reclamation and risking OOM under long rounds.

## Suggested Fix

After releasing a completed slot (and in `_assign_pairing` exception paths), explicitly clear model references on the slot:
- Set `slot.model_a = None`, `slot.model_b = None` (optionally also `slot.entry_a/entry_b = None`) immediately after release.
- In `_assign_pairing` failure path, clear any stale slot refs before `return`.
- Keep `release_fn` call, but do not rely on it alone for object-lifetime cleanup.
---
## Summary

When a completed slot is reassigned, the new pairing can start from mid-game states left by the previous pairing, contaminating match fairness and rollout data attribution.

## Severity

- Severity: major
- Priority: P1

## Evidence

- The code explicitly swaps new pairings into a slot without per-partition reset: [concurrent_matches.py:427](/home/john/keisei/keisei/training/concurrent_matches.py:427).
- Comment acknowledges envs in the slot may be mid-game at reassignment time: [concurrent_matches.py:429](/home/john/keisei/keisei/training/concurrent_matches.py:429), [concurrent_matches.py:431](/home/john/keisei/keisei/training/concurrent_matches.py:431).
- Rollout collection immediately records pre-step observations/actions for the new pairing in that inherited state: [concurrent_matches.py:271](/home/john/keisei/keisei/training/concurrent_matches.py:271), [concurrent_matches.py:316](/home/john/keisei/keisei/training/concurrent_matches.py:316).
- Sequential path starts each batch with `vecenv.reset()`, avoiding cross-pair carry-over: [match_utils.py:125](/home/john/keisei/keisei/training/match_utils.py:125).

## Root Cause Hypothesis

`run_round` reuses partition envs as soon as game-count target is reached, but completion is tracked by cumulative done count, not “all envs in slot are at fresh episode start.” If some envs are still in-progress, the next pairing inherits those positions and outcomes are no longer purely from that pairing’s own game starts.

## Suggested Fix

Gate slot reassignment on clean boundary conditions:
- Only swap in a new pairing when all envs in that slot have just completed/reset in the same step, or
- Add/reset per-partition env state before assigning a new pairing (if VecEnv can expose partition reset), or
- If neither is possible, keep slot idle until it naturally reaches a full-boundary state, then assign.
