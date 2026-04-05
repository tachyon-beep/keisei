## Summary

A malformed CSA move line can raise an uncaught `IndexError` and abort parsing for the entire file, causing valid later games in the same archive to be dropped.

## Severity

- Severity: major
- Priority: P1

## Evidence

- In [`/home/john/keisei/keisei/sl/parsers.py:238`](\/home\/john\/keisei\/keisei\/sl\/parsers.py:238), `CSAParser.parse()` calls `_parse_single_game(block)` without a per-block `try/except`, so any exception escapes and terminates the generator.
- In [`/home/john/keisei/keisei/sl/parsers.py:297`](\/home\/john\/keisei\/keisei\/sl\/parsers.py:297), `_parse_single_game()` assumes fixed-width CSA move fields and indexes `body[0..3]` without validating move length.
- In [`/home/john/keisei/keisei/sl/parsers.py:155`](\/home\/john\/keisei\/keisei\/sl\/parsers.py:155), `_csa_move_to_usi()` also assumes those indices exist (`int(body[0])`, etc.), so malformed inputs trigger `IndexError`.
- Repro (executed in this session): calling `_parse_single_game("V2.2\n+\n+12\n%TORYO\n")` raises `IndexError: string index out of range`, while neighboring valid blocks parse successfully. Because `parse()` does not isolate block errors, that exception would stop subsequent blocks from being yielded.

## Root Cause Hypothesis

The parser trusts move-line structure too early and lacks error isolation at the game-block boundary. A single corrupted move in one block causes a hard exception that propagates out of the file iterator, so downstream valid games are never parsed.

## Suggested Fix

Add defensive validation and per-block exception isolation in `CSAParser`:

1. Validate CSA move lines before indexing:
- Require minimum length and numeric coordinate fields before calling `_csa_move_to_usi()`.
- If invalid, either skip the game (`return None`) or skip that move with warning (prefer skip-game for label integrity).

2. Isolate exceptions per block in `parse()`:
- Wrap `_parse_single_game(block)` in `try/except Exception`, log with file/block context, and `continue` so later blocks are still processed.

3. Add regression test:
- Multi-game `.csa` file with one malformed middle block should still yield records for valid blocks before/after it.
