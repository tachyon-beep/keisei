## Summary

`play_batch` can crash with a `ValueError` when any environment reports a legal-action mask with zero valid moves, because it still applies `softmax` over all `-inf` logits and samples from an invalid `Categorical` distribution.

## Severity

- Severity: major
- Priority: P1

## Evidence

- [/home/john/keisei/keisei/training/match_utils.py:168](/home/john/keisei/keisei/training/match_utils.py:168) and [/home/john/keisei/keisei/training/match_utils.py:169](/home/john/keisei/keisei/training/match_utils.py:169): model A path masks illegal actions with `-inf` then does `softmax`.
- [/home/john/keisei/keisei/training/match_utils.py:178](/home/john/keisei/keisei/training/match_utils.py:178) and [/home/john/keisei/keisei/training/match_utils.py:179](/home/john/keisei/keisei/training/match_utils.py:179): same pattern for model B.
- Runtime confirmation (local REPL): `torch.softmax([[-inf, -inf]], dim=-1)` yields `tensor([[nan, nan]])`; `torch.distributions.Categorical(...)` then raises `ValueError ... constraint Simplex()`.
- Contrast: [/home/john/keisei/keisei/training/evaluate.py:125](/home/john/keisei/keisei/training/evaluate.py:125) guards this exact condition (`legal_counts == 0`) before sampling, but `match_utils.py` does not.

## Root Cause Hypothesis

The code assumes every state has at least one legal action. When that invariant is violated (engine edge case, corrupted mask, or transient env inconsistency), masking produces an all-`-inf` logit row, `softmax` becomes NaN, and sampling throws, aborting match execution.

## Suggested Fix

In `play_batch`, add a pre-sampling guard per actor subset (`a_indices`, `b_indices`) before `softmax`:

- Compute legal counts on `legal_masks[a_indices]` / `legal_masks[b_indices]`.
- If any row has zero legal actions, avoid `Categorical` on that row.
- Safe fallback options:
1. Mark those envs as draw/terminated for this step path (preferred for robustness), or
2. Select deterministic fallback action `0` (or `argmax` on mask cast) and log a warning with env indices.
- Keep existing sampling path for rows with at least one legal action.

This fix is localized to `keisei/training/match_utils.py` and prevents tournament/gauntlet crashes from invalid masks.
