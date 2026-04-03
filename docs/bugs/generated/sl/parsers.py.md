## Summary

`SFENParser.parse()` fails to handle CRLF (`\r\n`) multi-game files, causing game boundaries to be lost and metadata lines from later games to be ingested as bogus USI moves.

## Severity

- Severity: major
- Priority: P1

## Evidence

- In [`/home/john/keisei/keisei/sl/parsers.py:74`](/home/john/keisei/keisei/sl/parsers.py:74), SFEN blocks are split with `text.strip().split("\n\n")`, which only matches LF-LF separators.
- In [`/home/john/keisei/keisei/sl/parsers.py:109`](/home/john/keisei/keisei/sl/parsers.py:109), every remaining line is blindly appended as a move (`ParsedMove(move_usi=line)`), so once splitting fails, lines like `result:...` and `startpos` become invalid moves.
- Repro (runtime probe):
  - Input: two SFEN games separated by CRLF blank line
  - Output observed: `num_records 1`, `moves ['7g7f', '3c3d', 'result:win_white', 'startpos', '2g2f', '8c8d']`

## Root Cause Hypothesis

The parser assumes Unix newlines and does not normalize line endings before block splitting. On CRLF data, game separators are not detected, so subsequent game headers are treated as move text.

## Suggested Fix

Normalize newlines before parsing, then split blocks on normalized blank lines; additionally reject non-USI move lines in the move loop.

Example direction in target file:
- Normalize once: `text = path.read_text().replace("\r\n", "\n").replace("\r", "\n")`
- Keep block split on `"\n\n"` after normalization (or use a regex for blank-line blocks)
- Add a move-format guard before appending `ParsedMove`
---
## Summary

`CSAParser.parse()` fails to split multi-game archives when separators use CRLF (`\r\n/\r\n`), merging multiple games into one `GameRecord`.

## Severity

- Severity: major
- Priority: P2

## Evidence

- In [`/home/john/keisei/keisei/sl/parsers.py:230`](/home/john/keisei/keisei/sl/parsers.py:230), game archives are split with `text.split("\n/\n")`, which misses `\r\n/\r\n`.
- When split fails, `_parse_single_game()` processes both games as one block ([`/home/john/keisei/keisei/sl/parsers.py:239`](/home/john/keisei/keisei/sl/parsers.py:239) onward), producing one combined move list and one outcome.
- Repro (runtime probe using a fake path object):
  - Input: two CSA games with `\r\n/\r\n` separator
  - Output observed: `num_records 1`, `num_moves_first_record 2`, `moves ['7g7f', '2g2f']` (should be two records)

## Root Cause Hypothesis

The parser hardcodes an LF-only archive delimiter and does not normalize newline style first. Windows-style CSA archives therefore bypass record splitting.

## Suggested Fix

Normalize newline endings before splitting and split separator lines robustly.

Example direction in target file:
- Normalize: `text = text.replace("\r\n", "\n").replace("\r", "\n")`
- Split archive with a line-based separator rule (e.g., regex for `^/$` lines) rather than literal `"\n/\n"` only.
