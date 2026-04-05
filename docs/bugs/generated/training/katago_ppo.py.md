## Summary

GAE is computed in reduced precision when AMP is enabled, because rollout value estimates are stored as fp16/bf16 and passed into the recursive GAE scan without upcasting to float32.

## Severity

- Severity: major
- Priority: P1

## Evidence

`/home/john/keisei/keisei/training/katago_ppo.py:485`  
`select_actions()` computes `scalar_values` from AMP-autocast model outputs and returns them without dtype normalization.

`/home/john/keisei/keisei/training/katago_ppo.py:180` and `/home/john/keisei/keisei/training/katago_ppo.py:229`  
`KataGoRolloutBuffer.add()` stores these values as-is (`values_cpu = values.detach().cpu()`), preserving fp16/bf16 if produced under AMP.

`/home/john/keisei/keisei/training/katago_ppo.py:529` and `/home/john/keisei/keisei/training/katago_ppo.py:543`  
`update()` feeds stored `values_2d` into `compute_gae_gpu()` with no cast.

`/home/john/keisei/keisei/training/gae.py:149`  
`compute_gae_gpu()` sets `compute_dtype = values.dtype`, so recursive GAE runs in the incoming dtype (fp16/bf16 here), not float32.

Same dtype coupling exists in CPU paths (`compute_gae`/`compute_gae_padded`) via `compute_dtype = values.dtype` (`/home/john/keisei/keisei/training/gae.py:35`, `/home/john/keisei/keisei/training/gae.py:83`).

## Root Cause Hypothesis

AMP was integrated into model forward paths, but the rollout-to-GAE path assumes value tensors are already safe for recursive temporal accumulation. Under `use_amp=True`, that assumption fails: value estimates can be fp16/bf16, and GAE inherits that low precision, increasing numerical error in long-horizon advantage recursion and introducing unstable policy updates.

## Suggested Fix

In `KataGoPPOAlgorithm.update()`, force GAE inputs to float32 before calling `compute_gae_gpu` / `compute_gae` / `compute_gae_padded` (and keep `next_values` float32 too), e.g.:

- Cast `next_values_cpu = next_values.detach().to(torch.float32).cpu()`.
- Cast `rewards_2d`/`values_2d` (and padded variants) to `torch.float32` before GAE.
- In GPU path, pass `next_values.detach().to(torch.float32)` to `compute_gae_gpu`.
- Optionally enforce `advantages = advantages.to(torch.float32)` before normalization and PPO loss use.

This keeps AMP for forward/backward hot paths while preserving numerically stable advantage computation.
