## Summary

`_play_evaluation_games` can crash at inference time when an environment state has zero legal actions, because it softmaxes an all-`-inf` logits row and then samples from invalid probabilities.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Target code masks logits and samples without validating legal-action count:
  - `/home/john/keisei/keisei/training/evaluate.py:126`
  - `/home/john/keisei/keisei/training/evaluate.py:127`
  - `/home/john/keisei/keisei/training/evaluate.py:128`
- In the same codebase, the demonstrator path explicitly guards this exact failure mode (`zero legal actions -> all-inf softmax -> NaN crash`), but evaluate does not:
  - `/home/john/keisei/keisei/training/demonstrator.py:182`
  - `/home/john/keisei/keisei/training/demonstrator.py:183`
  - `/home/john/keisei/keisei/training/demonstrator.py:184`
  - `/home/john/keisei/keisei/training/demonstrator.py:189`

## Root Cause Hypothesis

The evaluator assumes every state always has at least one legal action. If `legal_masks.sum(dim=-1) == 0` for any sampled state (engine edge case, corrupted state, or integration mismatch), `masked_fill(~legal_masks, -inf)` produces an all-`-inf` row; `softmax` then yields invalid probabilities, and `Categorical(probs).sample()` fails, aborting the evaluation run.

## Suggested Fix

In `/home/john/keisei/keisei/training/evaluate.py`, add a pre-sampling guard identical in spirit to demonstrator:

- Compute `legal_counts = legal_masks.sum(dim=-1)` before masking.
- If any count is zero, handle deterministically (for example: log warning, mark game as draw, and `continue` to next game) instead of calling softmax/sampling.
- Restructure the game loop so post-loop reward logic (`step_result` access at `/home/john/keisei/keisei/training/evaluate.py:137`) is skipped for this early-termination path.
