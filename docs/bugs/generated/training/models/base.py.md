## Summary

`BaseModel` encodes stale tensor contract constants (`OBS_CHANNELS=46`, `ACTION_SPACE=13527`) that are incompatible with the current spatial Shogi pipeline (`50` channels, `11259` actions), causing downstream shape/runtime failures when `BaseModel` architectures are used.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Target file defines old contract and constants:
  - `/home/john/keisei/keisei/training/models/base.py:15` documents input `(batch, 46, 9, 9)`.
  - `/home/john/keisei/keisei/training/models/base.py:17` documents policy logits `(batch, 13527)`.
  - `/home/john/keisei/keisei/training/models/base.py:21` sets `OBS_CHANNELS = 46`.
  - `/home/john/keisei/keisei/training/models/base.py:23` sets `ACTION_SPACE = 13527`.
- These constants are directly used to build tensor shapes in `BaseModel` subclasses:
  - `/home/john/keisei/keisei/training/models/resnet.py:39` uses `self.OBS_CHANNELS` for first conv input channels.
  - `/home/john/keisei/keisei/training/models/resnet.py:47` uses `self.ACTION_SPACE` for policy head output size.
- Active runtime paths are hardwired to KataGo spatial format:
  - `/home/john/keisei/keisei/training/evaluate.py:103-104` creates env with `observation_mode="katago", action_mode="spatial"`.
  - `/home/john/keisei/keisei/training/evaluate.py:121` applies `masked_fill(~legal_masks, ...)`, requiring logits and legal mask last-dim match.
  - `/home/john/keisei/keisei/training/katago_loop.py:291-294` asserts action space must be `11259`.
  - `/home/john/keisei/keisei/training/models/katago_base.py:43` defines spatial action space as `11,259`.
- Result: selecting a `BaseModel` architecture (`resnet`/`mlp`/`transformer`) can produce incompatible policy/logit or input-channel shapes against current env tensors.

## Root Cause Hypothesis

`BaseModel` appears to be left from an older non-spatial contract and was not updated during migration to KataGo-style observation/action conventions. Subclasses inherit these constants, so the mismatch is systematic and triggers whenever those architectures run in the current env/eval paths.

## Suggested Fix

Update the canonical contract in `/home/john/keisei/keisei/training/models/base.py` to current spatial values so all `BaseModel` subclasses build compatible tensor shapes:

- Change `OBS_CHANNELS` from `46` to `50`.
- Change `ACTION_SPACE` from `13527` to `11259`.
- Update the class docstring contract lines accordingly (`(batch, 50, 9, 9)` and `(batch, 11259)`).

This keeps the primary fix localized to the target file and propagates correctly to all existing `BaseModel` subclasses via inherited constants.
