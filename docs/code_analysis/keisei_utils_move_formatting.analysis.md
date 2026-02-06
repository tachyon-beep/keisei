# Code Analysis: keisei/utils/move_formatting.py

## 1. Purpose & Role

This module provides human-readable formatting of Shogi moves for logging and display purposes. It converts internal move tuple representations into strings combining USI notation with English-language descriptions that include Japanese piece names with translations. It is a pure display utility with no game logic or side effects.

## 2. Interface Contracts

### Public Functions

- **`format_move_with_description(selected_shogi_move, policy_output_mapper, game=None)`** -> `str` (lines 8-60): Formats a move using an optional `ShogiGame` instance to look up piece information at the source square. Returns USI notation followed by a description like `"7g7f - Fuhyo (Pawn) moving from 7g to 7f."`.

- **`format_move_with_description_enhanced(selected_shogi_move, policy_output_mapper, piece_info=None)`** -> `str` (lines 63-111): Variant that accepts a pre-fetched piece object instead of a game instance. Designed for cases where the game state may have already changed (e.g., after a move was applied), so piece info must be captured beforehand.

### Private Functions

- **`_get_piece_name(piece_type, is_promoting=False)`** -> `str` (lines 114-145): Maps `PieceType` enum values to bilingual names (e.g., "Fuhyo (Pawn)"). When `is_promoting=True`, shows the transformation arrow (e.g., "Fuhyo (Pawn) -> Tokin (Promoted Pawn)").

- **`_coords_to_square_name(row, col)`** -> `str` (lines 148-154): Converts 0-indexed board coordinates to standard Shogi square notation (e.g., row=0, col=0 -> "9a").

### Dependencies

- `keisei.shogi.shogi_core_definitions.PieceType`

## 3. Correctness Analysis

### Move Type Detection (line 28)

Drop moves are identified by checking `len(selected_shogi_move) == 5 and selected_shogi_move[0] is None`. Board moves also have length 5 (per the tuple definitions), so the distinguishing factor is whether `selected_shogi_move[0]` is `None`. This is correct for the project's `DropMoveTuple = Tuple[None, None, int, int, PieceType]` vs `BoardMoveTuple = Tuple[int, int, int, int, bool]` conventions.

### Promotion Name Mapping (lines 135-142)

The promotion mapping in `_get_piece_name` only includes pieces that can legally promote in Shogi: Pawn, Lance, Knight, Silver, Bishop, and Rook. Gold and King cannot promote and are correctly omitted. If `is_promoting=True` is passed for a Gold or King (which should never happen in valid play), the fallback `piece_names.get(piece_type, str(piece_type))` on line 143 is used, which returns the normal name. This is defensively correct.

### Coordinate Conversion (lines 148-154)

The formula `file = str(9 - col)` and `rank = chr(ord("a") + row)` maps (row=0, col=0) to "9a" and (row=8, col=8) to "1i". This matches standard Shogi notation where files run 9 to 1 (right to left from Black's perspective) and ranks run "a" to "i" (top to bottom). The conversion is correct.

### Near-Duplicate Code (lines 8-60 vs 63-111)

The two public functions are near-identical, differing only in how piece information is obtained (from a game object vs a pre-passed piece_info parameter). The control flow, USI conversion, coordinate conversion, and error handling are duplicated.

## 4. Robustness & Error Handling

- Both public functions handle `None` input by returning `"None"` (lines 20, 77).
- Both wrap their entire bodies in a `try/except Exception` block (lines 23/58, 80/109) that falls back to the raw string representation of the move tuple, including the exception message. This ensures formatting never crashes the caller.
- The game piece lookup on lines 43-48 has its own nested `try/except` for `AttributeError`, `KeyError`, and `TypeError`, with a fallback to the generic string "piece".
- The enhanced version (lines 99-103) has a similar nested try/except for the piece_info parameter.

## 5. Performance & Scalability

- All operations are O(1) dictionary lookups and string formatting. No computational concerns.
- The `piece_names` and `base_names` dictionaries are created on each call to `_get_piece_name` rather than being module-level constants. For a display utility, this is negligible.

## 6. Security & Safety

- Pure formatting utility with no I/O, no state mutation, no external access.
- No concerns.

## 7. Maintainability

**Strengths:**
- Clear docstrings on both public functions with argument descriptions and return value documentation.
- The bilingual naming approach is thoughtful for a Shogi application targeting international users.
- The separation into two functions (game-based vs pre-fetched piece info) serves a legitimate use case where move formatting happens after game state has changed.

**Weaknesses:**
- Significant code duplication between `format_move_with_description` and `format_move_with_description_enhanced`. The two functions share approximately 80% identical code.
- No type annotations on function parameters (all are untyped beyond docstrings).
- The module has no `__all__` export list.

## 8. Verdict

**SOUND**

This is a straightforward display utility that correctly handles Shogi move formatting with proper defensive error handling. The code duplication is a maintainability concern but does not affect correctness. All coordinate conversions and piece name mappings are accurate for standard Shogi notation.
