## Summary

No concrete bug found in /home/john/keisei/keisei/training/demonstrator.py

## Severity

- Severity: trivial
- Priority: P3

## Evidence

Reviewed `/home/john/keisei/keisei/training/demonstrator.py` end-to-end, focusing on:
- Inference guardrails and tensor flow in `_play_game()` (`/home/john/keisei/keisei/training/demonstrator.py:170-203`)
- `torch.no_grad()` use and masking/sampling path (`/home/john/keisei/keisei/training/demonstrator.py:181-196`)
- Pin/unpin lifecycle and exception safety (`/home/john/keisei/keisei/training/demonstrator.py:139-150`, `205-207`)
- Model output contract adapter `_get_policy_flat()` (`/home/john/keisei/keisei/training/demonstrator.py:22-33`)

Cross-checked integration assumptions:
- `OpponentStore.load_opponent()` sets `model.eval()` and device placement (`/home/john/keisei/keisei/training/opponent_store.py:607-624`)
- VecEnv legal mask dtype/shape expectations in tests (`/home/john/keisei/shogi-engine/crates/shogi-gym/tests/test_vec_env.py:138`, `/home/john/keisei/tests/test_demonstrator.py:259-270`)

## Root Cause Hypothesis

No bug identified.

## Suggested Fix

Unknown
