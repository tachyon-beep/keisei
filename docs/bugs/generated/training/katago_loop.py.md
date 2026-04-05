## Summary

`KataGoTrainingLoop.run()` starts the background tournament thread but does not guarantee `stop()` on exceptions, so failures in the main training loop can leak a live tournament thread and associated resources.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Tournament starts before rollout loop: [`/home/john/keisei/keisei/training/katago_loop.py:755`](​/home/john/keisei/keisei/training/katago_loop.py#L755)
- Tournament is only stopped at normal function end (no `try/finally` around training body): [`/home/john/keisei/keisei/training/katago_loop.py:1576`](​/home/john/keisei/keisei/training/katago_loop.py#L1576)
- `LeagueTournament.start()` creates a daemon thread and `stop()` is the graceful shutdown path: [`/home/john/keisei/keisei/training/tournament.py:110`](​/home/john/keisei/keisei/training/tournament.py#L110), [`/home/john/keisei/keisei/training/tournament.py:127`](​/home/john/keisei/keisei/training/tournament.py#L127)

## Root Cause Hypothesis

`run()` assumes normal control flow to its tail. Any raised exception between `start()` and the final `stop()` skips cleanup, leaving concurrent background work running against shared DB/device resources.

## Suggested Fix

Wrap the main body of `run()` (after optional `start()`) in `try/finally` and call `self._tournament.stop()` in the `finally` block when tournament is active.
---
## Summary

In split-merge (league/opponent) mode, `self.latest_values` is never updated, so spectator snapshots persist stale/default value estimates (typically zeros) instead of current policy value predictions.

## Severity

- Severity: minor
- Priority: P2

## Evidence

- `latest_values` initialized once: [`/home/john/keisei/keisei/training/katago_loop.py:552`](​/home/john/keisei/keisei/training/katago_loop.py#L552)
- It is updated only in the non-opponent branch: [`/home/john/keisei/keisei/training/katago_loop.py:1185`](​/home/john/keisei/keisei/training/katago_loop.py#L1185)
- In split-merge branch, learner values are produced (`sm_result.learner_values`) but never copied into `latest_values`: [`/home/john/keisei/keisei/training/katago_loop.py:1071`](​/home/john/keisei/keisei/training/katago_loop.py#L1071)
- Snapshot writer reads `latest_values` for UI payload: [`/home/john/keisei/keisei/training/katago_loop.py:1679`](​/home/john/keisei/keisei/training/katago_loop.py#L1679)

## Root Cause Hypothesis

When split-merge logic was added, rollout/value plumbing was migrated to pending transitions and scratch tensors, but snapshot-facing `latest_values` maintenance remained only in the legacy single-model path.

## Suggested Fix

Inside the split-merge branch, update `self.latest_values` each step by scattering learner predictions back to full-env shape (for learner envs) and optionally filling opponent envs from current `value_adapter`/forward pass or leaving prior values explicitly documented; then convert to list for snapshot use.
---
## Summary

`split_merge_step()` lacks the zero-legal-action safety check used elsewhere, so an all-false legal mask can propagate `-inf` logits into `softmax`/`Categorical`, producing NaNs or runtime failures during rollout.

## Severity

- Severity: major
- Priority: P1

## Evidence

- In `split_merge_step`, masked logits are softmaxed without guard:
  - Learner path: [`/home/john/keisei/keisei/training/katago_loop.py:308`](​/home/john/keisei/keisei/training/katago_loop.py#L308)
  - Opponent path: [`/home/john/keisei/keisei/training/katago_loop.py:357`](​/home/john/keisei/keisei/training/katago_loop.py#L357)
- Equivalent rollout path in PPO explicitly guards this condition and raises clear error:
  - [`/home/john/keisei/keisei/training/katago_ppo.py:461`](​/home/john/keisei/keisei/training/katago_ppo.py#L461)

## Root Cause Hypothesis

Defensive validation was implemented in `KataGoPPOAlgorithm.select_actions()` and `update()`, but not mirrored in split-merge inference path, leaving a brittle branch where malformed/terminal masks fail late and opaquely.

## Suggested Fix

Add explicit per-batch legal-count checks in `split_merge_step()` for both learner and opponent subsets (same pattern/message style as `katago_ppo.py`), raising before `softmax` if any row has zero legal actions.
