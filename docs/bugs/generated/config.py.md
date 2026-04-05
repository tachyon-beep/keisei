## Summary

`DynamicConfig` does not validate `checkpoint_flush_every`, allowing `0`/negative values that trigger runtime failures during optimizer checkpoint cadence logic.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Target config lacks bounds check for `checkpoint_flush_every`:
  - [/home/john/keisei/keisei/config.py:103](/home/john/keisei/keisei/config.py:103)
  - [/home/john/keisei/keisei/config.py:108](/home/john/keisei/keisei/config.py:108) through [/home/john/keisei/keisei/config.py:128](/home/john/keisei/keisei/config.py:128) (no validation for this field)
- Value is used as a divisor/modulus in training path:
  - [/home/john/keisei/keisei/training/dynamic_trainer.py:332](/home/john/keisei/keisei/training/dynamic_trainer.py:332)
  - [/home/john/keisei/keisei/training/dynamic_trainer.py:334](/home/john/keisei/keisei/training/dynamic_trainer.py:334)
- If set to `0`, `% self.config.checkpoint_flush_every` raises `ZeroDivisionError`; this is caught and can repeatedly fail/disable entries rather than training cleanly:
  - [/home/john/keisei/keisei/training/dynamic_trainer.py:196](/home/john/keisei/keisei/training/dynamic_trainer.py:196) to [/home/john/keisei/keisei/training/dynamic_trainer.py:217](/home/john/keisei/keisei/training/dynamic_trainer.py:217)

## Root Cause Hypothesis

During Phase 3 config expansion, `checkpoint_flush_every` was added to `DynamicConfig` but omitted from `__post_init__` guards. Invalid TOML values can therefore pass config load and only fail deep in the update loop.

## Suggested Fix

Add explicit validation in `DynamicConfig.__post_init__`:

- `checkpoint_flush_every >= 1`
- (optionally) integer type check for this field, matching other strict boolean checks already done in `load_config`.

Also add regression tests in `tests/test_config.py` for `DynamicConfig(checkpoint_flush_every=0)` and negative values.
---
## Summary

`DynamicConfig` does not validate `max_buffer_depth`, so `0` is accepted and silently discards all rollout data, causing dynamic PPO updates to become persistent no-ops.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Target config defines but does not validate `max_buffer_depth`:
  - [/home/john/keisei/keisei/config.py:105](/home/john/keisei/keisei/config.py:105)
  - No corresponding guard in [/home/john/keisei/keisei/config.py:108](/home/john/keisei/keisei/config.py:108) through [/home/john/keisei/keisei/config.py:128](/home/john/keisei/keisei/config.py:128)
- Buffer construction uses `deque(maxlen=self.config.max_buffer_depth)`:
  - [/home/john/keisei/keisei/training/dynamic_trainer.py:76](/home/john/keisei/keisei/training/dynamic_trainer.py:76)
  - [/home/john/keisei/keisei/training/dynamic_trainer.py:201](/home/john/keisei/keisei/training/dynamic_trainer.py:201)
- With `max_buffer_depth=0`, appends are dropped; update then sees no data and returns `False` without training:
  - empty path in [/home/john/keisei/keisei/training/dynamic_trainer.py:129](/home/john/keisei/keisei/training/dynamic_trainer.py:129) to [/home/john/keisei/keisei/training/dynamic_trainer.py:133](/home/john/keisei/keisei/training/dynamic_trainer.py:133)
  - early no-op return at [/home/john/keisei/keisei/training/dynamic_trainer.py:231](/home/john/keisei/keisei/training/dynamic_trainer.py:231) to [/home/john/keisei/keisei/training/dynamic_trainer.py:232](/home/john/keisei/keisei/training/dynamic_trainer.py:232)

## Root Cause Hypothesis

`max_buffer_depth` was introduced as a memory/control knob but its lower bound was not constrained. Python `deque(maxlen=0)` is legal, so the failure mode is silent (data dropped) rather than an immediate config error.

## Suggested Fix

Add validation in `DynamicConfig.__post_init__`:

- `max_buffer_depth >= 1`

Consider also validating `max_consecutive_errors >= 1` in the same block to avoid immediate disable semantics from invalid values. Add regression tests for `DynamicConfig(max_buffer_depth=0)` and negative values.
