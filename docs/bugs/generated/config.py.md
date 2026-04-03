## Summary

`LeagueConfig` does not validate `epochs_per_seat`/`snapshot_interval`, so a value of `0` in TOML can crash training with modulo-by-zero in the RL loop.

## Severity

- Severity: major
- Priority: P1

## Evidence

- [`/home/john/keisei/keisei/config.py:52`](/home/john/keisei/keisei/config.py:52) only validates ratio sum in `LeagueConfig.__post_init__`; no bounds checks for `epochs_per_seat` or `snapshot_interval`.
- [`/home/john/keisei/keisei/config.py:44`](/home/john/keisei/keisei/config.py:44) and [`/home/john/keisei/keisei/config.py:45`](/home/john/keisei/keisei/config.py:45) define these fields.
- [`/home/john/keisei/keisei/training/katago_loop.py:651`](/home/john/keisei/keisei/training/katago_loop.py:651) computes `(epoch_i + 1) % self.config.league.epochs_per_seat`.
- [`/home/john/keisei/keisei/training/katago_loop.py:658`](/home/john/keisei/keisei/training/katago_loop.py:658) computes `(epoch_i + 1) % self.config.league.snapshot_interval`.

## Root Cause Hypothesis

Config boundary validation is incomplete in `LeagueConfig`: it checks ratio consistency but not positive integer constraints for interval fields that are later used as modulo divisors.

## Suggested Fix

In `LeagueConfig.__post_init__` in `/home/john/keisei/keisei/config.py`, add explicit validation:

- `epochs_per_seat >= 1`
- `snapshot_interval >= 1`
- optionally `max_pool_size >= 1` and `elo_k_factor > 0` for consistency

Raise `ValueError` with field-specific messages before constructing `AppConfig`.
---
## Summary

Boolean config values are not type-validated; quoted TOML strings like `"false"` can be interpreted as truthy and silently flip behavior (e.g., AMP/DDP flags).

## Severity

- Severity: major
- Priority: P2

## Evidence

- [`/home/john/keisei/keisei/config.py:125`](/home/john/keisei/keisei/config.py:125) uses `use_amp = bool(t.get("use_amp", False))`; `bool("false")` is `True`.
- [`/home/john/keisei/keisei/config.py:174`](/home/john/keisei/keisei/config.py:174) constructs `DistributedConfig(**dist_raw)` with no runtime type checks.
- [`/home/john/keisei/keisei/config.py:78`](/home/john/keisei/keisei/config.py:78)-[`80`](/home/john/keisei/keisei/config.py:80) annotate distributed fields as `bool`, but annotations are not enforced at runtime.
- [`/home/john/keisei/keisei/training/katago_loop.py:203`](/home/john/keisei/keisei/training/katago_loop.py:203) branches on `config.distributed.sync_batchnorm` truthiness.
- [`/home/john/keisei/keisei/training/katago_loop.py:210`](/home/john/keisei/keisei/training/katago_loop.py:210) and [`211`](/home/john/keisei/keisei/training/katago_loop.py:211) pass these values into DDP behavior flags.

## Root Cause Hypothesis

`load_config` relies on dataclass annotations without explicit type validation and also applies lossy boolean coercion (`bool(...)`) for `use_amp`, allowing malformed TOML types to pass and alter runtime behavior silently.

## Suggested Fix

In `/home/john/keisei/keisei/config.py`:

- Replace `use_amp = bool(...)` with strict type validation (`isinstance(value, bool)`), then assign directly.
- Add strict `bool` type checks for `[distributed]` fields before creating `DistributedConfig`.
- Raise `ValueError` with clear messages when non-boolean types are provided.
