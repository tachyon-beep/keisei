## Summary

`prepare_sl_data()` processes directory game files in filesystem-dependent order, producing non-deterministic shard contents across runs and breaking reproducible SL dataset generation.

## Severity

- Severity: major
- Priority: P1

## Evidence

- [`/home/john/keisei/keisei/sl/prepare.py:84`](#/home/john/keisei/keisei/sl/prepare.py:84) builds `game_files` by appending results from `Path.glob(...)` without sorting.
- [`/home/john/keisei/keisei/sl/prepare.py:113`](#/home/john/keisei/keisei/sl/prepare.py:113) iterates `for game_file in game_files:` directly, so shard order depends on unsorted discovery order.
- The resulting shard write sequence is order-sensitive (`_flush_shard` writes sequential shard indices), so any file ordering drift changes dataset byte layout and sample order.

## Root Cause Hypothesis

`Path.glob()` result ordering is not guaranteed across filesystems/platforms, and `prepare_sl_data()` never normalizes ordering before parsing/writing. Any multi-file source directory can trigger run-to-run nondeterminism.

## Suggested Fix

In `prepare_sl_data()`, sort (and ideally deduplicate) discovered paths before parsing, for example by normalized absolute path string:

- After collection: `game_files = sorted(set(game_files), key=lambda p: str(p.resolve()))`
- Then iterate in that stable order.
---
## Summary

Directory-source ingestion silently misses uppercase/mixed-case game extensions (for example `.SFEN`, `.CSA`) on case-sensitive filesystems, causing incomplete datasets.

## Severity

- Severity: minor
- Priority: P2

## Evidence

- [`/home/john/keisei/keisei/sl/prepare.py:90`](#/home/john/keisei/keisei/sl/prepare.py:90)-[`91`](#/home/john/keisei/keisei/sl/prepare.py:91): directory scan uses `source_path.glob(f"*{ext}")` where `ext` comes from parser registry (`.sfen`, `.csa` lowercase).
- [`/home/john/keisei/keisei/sl/prepare.py:114`](#/home/john/keisei/keisei/sl/prepare.py:114) later normalizes file suffix via `.lower()`, indicating extension matching is intended to be case-insensitive at parse time, but discovery is not.

## Root Cause Hypothesis

Extension filtering is split into two phases: case-sensitive glob at discovery time and case-insensitive parser lookup at processing time. Files excluded by the first phase never reach the second.

## Suggested Fix

Replace extension-based globbing with case-insensitive suffix filtering during directory enumeration, for example:

- Iterate `source_path.iterdir()` (or recursive `rglob("*")` if desired),
- Keep files where `p.suffix.lower() in parsers`,
- Then process those files.

This keeps discovery behavior consistent with parser selection logic.
