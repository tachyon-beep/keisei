## Summary

`CSAParser` can emit illegal USI promotion suffixes (`+`) for already-promoted piece moves when a game uses `PI` (or otherwise lacks `P1..P9` board lines), because promotion inference runs with an empty/unknown board state.

## Severity

- Severity: minor
- Priority: P2

## Evidence

- Promotion inference depends on source-piece lookup in board state: [`/home/john/keisei/keisei/sl/parsers.py:176`](/home/john/keisei/keisei/sl/parsers.py:176), [`/home/john/keisei/keisei/sl/parsers.py:179`](/home/john/keisei/keisei/sl/parsers.py:179), [`/home/john/keisei/keisei/sl/parsers.py:180`](/home/john/keisei/keisei/sl/parsers.py:180).
- Board is only initialized from `P1..P9` lines; `PI` is not parsed into initial board state: [`/home/john/keisei/keisei/sl/parsers.py:260`](/home/john/keisei/keisei/sl/parsers.py:260), [`/home/john/keisei/keisei/sl/parsers.py:265`](/home/john/keisei/keisei/sl/parsers.py:265), [`/home/john/keisei/keisei/sl/parsers.py:270`](/home/john/keisei/keisei/sl/parsers.py:270), [`/home/john/keisei/keisei/sl/parsers.py:285`](/home/john/keisei/keisei/sl/parsers.py:285).
- Repro (executed): `_parse_single_game("V2.2\\nPI\\n+\\n+8283UM\\n%TORYO\\n")` returns move `8b8c+`, while equivalent explicit-board input returns `8b8c` (no promotion).
- Downstream integration expects parser `move_usi` to be replay/encoded later for policy targets: [`/home/john/keisei/keisei/sl/prepare.py:145`](/home/john/keisei/keisei/sl/prepare.py:145), [`/home/john/keisei/keisei/sl/prepare.py:152`](/home/john/keisei/keisei/sl/prepare.py:152).

## Root Cause Hypothesis

Promotion detection uses `piece in _PROMOTED and source_piece not in _PROMOTED`; when board state is unknown (`source_piece == ""`), any move ending as promoted piece is treated as a fresh promotion. This is triggered when initial position is provided as `PI` (currently ignored) or other formats that do not populate `board`.

## Suggested Fix

Add `PI` handling in `CSAParser` so board state is initialized to the standard starting position before move parsing, and make promotion inference robust to unknown source squares.

Concrete change in `parsers.py`:
1. Extend board initialization to recognize `PI` and create a full initial board map.
2. In `_csa_move_to_usi`, only append `+` when `source_piece` is known and unpromoted:
   - `if source_piece and piece in _PROMOTED and source_piece not in _PROMOTED: usi += "+"`
3. Add regression test: CSA block with `PI` and an already-promoted piece move should not get `+` suffix.
