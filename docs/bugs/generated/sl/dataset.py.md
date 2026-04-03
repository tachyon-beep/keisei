## Summary

`SLDataset` loads shard files in lexicographic filename order, which misorders shards once indices exceed 3 digits (e.g., `shard_1000.bin` before `shard_999.bin`), silently remapping dataset indices to the wrong records.

## Severity

- Severity: major
- Priority: P1

## Evidence

- [`/home/john/keisei/keisei/sl/dataset.py:104`](/home/john/keisei/keisei/sl/dataset.py:104) uses `sorted(data_dir.glob("shard_*.bin"))`, which is string sort, not numeric shard-index sort.
- Writer naming allows indices beyond 3 digits: [`/home/john/keisei/keisei/sl/prepare.py:204`](/home/john/keisei/keisei/sl/prepare.py:204) uses `f"shard_{shard_idx:03d}.bin"` (minimum width, not fixed maximum).
- Verified ordering behavior:
  - `sort` output for sample names is: `shard_1000.bin`, `shard_1001.bin`, `shard_101.bin`, `shard_998.bin`, `shard_999.bin` (incorrect numeric order).

## Root Cause Hypothesis

String sorting was assumed to preserve shard sequence because early shard IDs are zero-padded to 3 digits. Once shard counts exceed 999, filename length increases and lexicographic ordering diverges from numeric ordering, causing deterministic but incorrect index-to-sample mapping.

## Suggested Fix

In `SLDataset.__init__`, sort shard files by parsed numeric suffix instead of raw path string, e.g. parse `shard_(\d+)\.bin` and sort by `int(index)`. Optionally reject nonconforming filenames instead of silently including them.
---
## Summary

`SLDataset` silently truncates corrupted/incomplete shard files with trailing partial records (`file_size % RECORD_SIZE != 0`) instead of surfacing data integrity errors.

## Severity

- Severity: minor
- Priority: P2

## Evidence

- [`/home/john/keisei/keisei/sl/dataset.py:107`](/home/john/keisei/keisei/sl/dataset.py:107)-[`108`](/home/john/keisei/keisei/sl/dataset.py:108): `n_positions = file_size // RECORD_SIZE` floors count.
- No remainder check is performed before accepting shard entries at [`/home/john/keisei/keisei/sl/dataset.py:109`](/home/john/keisei/keisei/sl/dataset.py:109)-[`113`](/home/john/keisei/keisei/sl/dataset.py:113).
- Existing test codifies silent drop for undersized shard: [`/home/john/keisei/tests/test_sl_pipeline.py:550`](/home/john/keisei/tests/test_sl_pipeline.py:550)-[`561`](/home/john/keisei/tests/test_sl_pipeline.py:561), indicating loader currently suppresses corruption signals.

## Root Cause Hypothesis

The loader is optimized for permissive ingestion and computes record count by integer division, but never validates shard byte-alignment. Interrupted writes or filesystem corruption therefore lose data silently, making training reproducibility/debugging harder.

## Suggested Fix

Add explicit integrity validation in `SLDataset.__init__`:
- compute `remainder = file_size % RECORD_SIZE`;
- if `remainder != 0`, raise `ValueError` (or at minimum log a high-severity warning and skip file via explicit policy);
- include shard path and remainder bytes in the message for triage.
