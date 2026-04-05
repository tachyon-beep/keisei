## Summary

`_play_evaluation_games()` loads checkpoints directly onto the target device (`map_location=device`) before loading into CPU-resident models, causing unnecessary cross-device copies and potentially large transient CUDA memory spikes/OOM during evaluation.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Target file loads checkpoint tensors to the final device first:
  - `/home/john/keisei/keisei/training/evaluate.py:88`
  - `/home/john/keisei/keisei/training/evaluate.py:95`
- Then loads into models that are still on CPU:
  - `/home/john/keisei/keisei/training/evaluate.py:87`
  - `/home/john/keisei/keisei/training/evaluate.py:90`
  - `/home/john/keisei/keisei/training/evaluate.py:94`
  - `/home/john/keisei/keisei/training/evaluate.py:97`
- Only after that, models are moved to target device:
  - `/home/john/keisei/keisei/training/evaluate.py:91`
  - `/home/john/keisei/keisei/training/evaluate.py:98`

This creates a wasteful load path on CUDA:
1. checkpoint tensors allocated on GPU,
2. copied into CPU model params via `load_state_dict`,
3. model copied back to GPU via `.to(device)`.

- Related codebase pattern already avoids this by loading on CPU first:
  - `/home/john/keisei/keisei/training/opponent_store.py:726-733` (`map_location="cpu"` then `model.to(device)`).

## Root Cause Hypothesis

The evaluation loader uses `map_location=device` for convenience, but model construction/load ordering in this file is CPU-first. That mismatch produces redundant transfers and elevated peak memory. It is most likely to trigger when evaluating large checkpoints on constrained GPUs, especially since two models are loaded in one run.

## Suggested Fix

Load checkpoints on CPU, then move models once after `load_state_dict`:

- Change both loads in `/home/john/keisei/keisei/training/evaluate.py`:
  - `torch.load(checkpoint_a, map_location="cpu", weights_only=True)`
  - `torch.load(checkpoint_b, map_location="cpu", weights_only=True)`

Keep existing `model = model.to(device)` after state load. This matches existing project practice and removes unnecessary GPU peak usage.
