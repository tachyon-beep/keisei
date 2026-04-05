## Summary

`prepare_sl_data()` can silently mix stale shard files from previous runs in the same output directory, causing contaminated SL datasets and incorrect training data.

## Severity

- Severity: major
- Priority: P1

## Evidence

- `/home/john/keisei/keisei/sl/prepare.py:70` creates/uses `output_dir` with `exist_ok=True` but does not clean old `shard_*.bin`.
- `/home/john/keisei/keisei/sl/prepare.py:220` always writes new shards starting at `shard_000.bin` and increments from zero.
- `/home/john/keisei/keisei/sl/prepare.py:194` writes `num_shards` metadata, but nothing enforces it at read time.
- `/home/john/keisei/keisei/sl/dataset.py:109` loads **all** `shard_*.bin` via glob, not limited by `shard_meta.json["num_shards"]`.

This means if a prior run produced more shards than the current run, extra old shards remain on disk and are still loaded by `SLDataset`.

## Root Cause Hypothesis

The writer assumes overwrite-by-name is sufficient, but only shard indices produced in the current run are overwritten. Any higher-index shard files from older runs persist and are included by the loader’s glob scan, which has no pruning/consistency check against metadata.

## Suggested Fix

In `prepare_sl_data()` (target file), make output generation transactional or clean:

- Before writing new shards, delete existing `shard_*.bin` and stale temp/meta files in `output_dir`; or
- Write to a fresh temp subdirectory, then atomically replace/swap the final directory when complete.

Also add a post-write consistency guard in `prepare.py` (or companion test) to assert only `shard_000..shard_{num_shards-1}` exist after preparation.
