## Summary

`load_config()` silently accepts unknown keys in core sections (`[training]`, `[display]`, `[model]`, and `[demonstrator]`), so typos are ignored and defaults are used, causing misconfigured training without an explicit error.

## Severity

- Severity: major
- Priority: P1

## Evidence

- `/home/john/keisei/keisei/config.py:511-572` reads core sections via `.get(...)` and never checks for unknown keys:
  - `t = raw.get("training", {})` then selected fields only (`num_games`, `max_ply`, `algorithm`, etc.)
  - `d = raw.get("display", {})` then only `moves_per_minute`, `db_path`
  - `m = raw.get("model", {})` then only `display_name`, `architecture`, `params`
- In contrast, explicit unknown-key rejection exists for:
  - `/home/john/keisei/keisei/config.py:628-641` (`[league]`)
  - `/home/john/keisei/keisei/config.py:664-671` (`[distributed]`)
- Runtime impact example: if user misspells `checkpoint_interval` under `[training]`, loader falls back to default `50` (`/home/john/keisei/keisei/config.py:526`) and checkpoint cadence changes silently; checkpoint write logic depends directly on this field at `/home/john/keisei/keisei/training/katago_loop.py:1613-1620`.

## Root Cause Hypothesis

Validation logic was tightened for newer sections (`league`, `distributed`) but not retrofitted for older top-level sections, leaving asymmetric config-boundary validation and allowing typo-driven silent fallback to defaults.

## Suggested Fix

In `load_config()` (`/home/john/keisei/keisei/config.py`), add explicit allowed-key checks for each core section before extracting values, mirroring existing `[league]`/`[distributed]` behavior. For example:

- `[training]` valid keys: `num_games`, `max_ply`, `algorithm`, `checkpoint_interval`, `checkpoint_dir`, `algorithm_params`, `use_amp`
- `[display]` valid keys: `moves_per_minute`, `db_path`
- `[model]` valid keys: `display_name`, `architecture`, `params`
- `[demonstrator]` valid keys from `fields(DemonstratorConfig)`

Raise `ValueError` on unknown keys with the valid-key list in the message. This prevents silent typo fallback and fails fast at config load time.
