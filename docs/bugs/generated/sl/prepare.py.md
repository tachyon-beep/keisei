## Summary

`prepare_sl_data()` materializes each parser iterator with `list(...)`, which causes all-or-nothing loss of valid earlier games if a later game in the same file raises, and can also spike memory on large archives.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Target file eagerly consumes full iterator:
  - `/home/john/keisei/keisei/sl/prepare.py:93-100`
  - `file_records = list(parser.parse(game_file))` is wrapped in one `try`; any exception aborts the whole file and skips all records from that file.
- Parser is generator-based and can fail mid-file:
  - `/home/john/keisei/keisei/sl/parsers.py:229-237` (`CSAParser.parse` yields per block)
  - `/home/john/keisei/keisei/sl/parsers.py:294-296` integer parsing in move decoding can raise on malformed blocks.
- Current behavior path:
  - If later block raises while building `list(...)`, prior yielded records are never processed by `prepare_sl_data()`.

## Root Cause Hypothesis

The code converts a streaming parser API into a fully materialized list inside one exception boundary. That collapses partial progress semantics and ties correctness to every block in the file being parseable.

## Suggested Fix

Replace `file_records = list(parser.parse(game_file))` with streaming iteration so already-yielded records are kept, and handle parse exceptions without discarding prior work. For example:
- Iterate `for record in parser.parse(game_file): ...` directly.
- Keep exception handling around iteration (or per-record decode boundary) so malformed tail data does not erase valid earlier games.
- This also reduces peak memory usage for multi-game files.
---
## Summary

`shard_size` is not enforced as a true maximum per shard because flush happens only after finishing each game; a single long game can produce oversized shards and large in-memory buffers.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Flush condition is outside the per-move append path:
  - `/home/john/keisei/keisei/sl/prepare.py:108-145` appends all positions for a game
  - `/home/john/keisei/keisei/sl/prepare.py:147-160` checks `len(observations) >= shard_size` only after the whole game.
- Test suite documents this current behavior explicitly:
  - `/home/john/keisei/tests/test_prepare_sl.py:227-245` notes a 4-move game with `shard_size=2` still ends up flushed as one shard at game boundary.

## Root Cause Hypothesis

The flush trigger is placed at game granularity rather than position granularity, so `shard_size` acts like a post-game threshold check, not a strict cap.

## Suggested Fix

Move shard flush logic into the per-move loop (right after appending targets), and flush in chunks while `len(observations) >= shard_size` so each shard respects configured size. Keep final remainder flush at end.
