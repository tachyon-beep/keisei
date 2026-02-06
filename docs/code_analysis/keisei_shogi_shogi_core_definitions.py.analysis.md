# Analysis: keisei/shogi/shogi_core_definitions.py

**Lines:** 509
**Role:** Foundation type definitions for the entire Shogi engine. Defines `Color` enum, `PieceType` enum (14 piece types including promoted), `Piece` class, move tuple types (`BoardMoveTuple`, `DropMoveTuple`, `MoveTuple`), `TerminationReason` enum, `MoveApplicationResult` dataclass, observation plane constants (channel layout for neural network input), and various lookup tables (`BASE_TO_PROMOTED_TYPE`, `PROMOTED_TO_BASE_TYPE`, `PIECE_TYPE_TO_HAND_TYPE`, `SYMBOL_TO_PIECE_TYPE`, etc.). This is the most-imported module in the Shogi subsystem, serving as the single source of truth for core types.
**Key dependencies:** Imports only from Python stdlib (`dataclasses`, `enum`, `typing`). Imported by virtually every module in the `keisei.shogi` package, plus `keisei.training`, `keisei.evaluation`, `keisei.core`, `keisei.utils`, and many test files (38+ importers found).
**Analysis depth:** FULL

## Summary

This is a well-structured, carefully documented foundation module. The type definitions are consistent, the observation channel layout is clearly documented with channel maps, and the `Piece` class is properly implemented with equality, hashing, and deep copy support. The most significant concern is the duplication of observation constants with `keisei/constants.py` using different naming conventions, which creates a maintenance risk of divergence. There are a few minor issues worth noting but no critical bugs.

## Warnings

### [272-283 and constants.py:20-29] Observation plane constants are duplicated with different names in keisei/constants.py
**What:** The observation channel layout constants are defined here AND in `keisei/constants.py`, with different naming conventions:
- Here: `OBS_CURR_PLAYER_UNPROMOTED_START`, `OBS_OPP_PLAYER_UNPROMOTED_START`, etc.
- constants.py: `OBS_CURRENT_PLAYER_UNPROMOTED_START`, `OBS_OPPONENT_UNPROMOTED_START`, etc.

The values are currently identical (0, 8, 14, 22, 28, 35, 42, 43, 44, 45), but there is no mechanism to enforce they stay in sync.

**Why it matters:** If either copy is updated without updating the other, different parts of the codebase will use different channel layouts, leading to silent observation corruption. The `shogi_game_io.py` module imports from `shogi_core_definitions.py`, while other modules could import from `constants.py`. This is a classic "two sources of truth" problem.

**Evidence:**
```python
# shogi_core_definitions.py:
OBS_CURR_PLAYER_UNPROMOTED_START = 0
OBS_CURR_PLAYER_PROMOTED_START = 8

# constants.py:
OBS_CURRENT_PLAYER_UNPROMOTED_START = 0
OBS_CURRENT_PLAYER_PROMOTED_START = 8
```

### [352-361] OBS_UNPROMOTED_ORDER includes KING (8 elements) but hand channels only have 7 slots
**What:** `OBS_UNPROMOTED_ORDER` has 8 entries: `[PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK, KING]`. This is correct for board piece encoding (channels 0-7 and 14-21, which have 8 slots each). However, the hand channels only have 7 slots per player (channels 28-34 and 35-41), because kings cannot be in hand. Code that iterates `OBS_UNPROMOTED_ORDER` to populate hand channels (as `features.py` does) would write one element too many, overlapping into the next channel block.

The production code in `shogi_game_io.py` avoids this by using `get_unpromoted_types()` (7 elements, no King) for hand channels instead of `OBS_UNPROMOTED_ORDER`. But the constant naming and the comment on line 360 ("King is included for on-board representation") do not make this distinction obvious.

**Why it matters:** The constant is safe for its intended purpose (board piece channel indexing), but its name suggests it could be used for hand piece indexing too. The `features.py` file already makes this mistake. Future developers could easily repeat it.

### [424] Piece.is_promoted is a boolean attribute, not a method
**What:** `self.is_promoted: bool = piece_type in PROMOTED_TYPES_SET` sets `is_promoted` as a boolean attribute. However, `features.py` calls it as `piece.is_promoted()` (with parentheses, as a method call). This would raise `TypeError: 'bool' object is not callable` at runtime.

Additionally, `shogi_game_io.py` accesses it correctly as `p.is_promoted` (boolean attribute, line 478).

**Why it matters:** This confirms the incompatibility between `features.py` and the actual `Piece` class. From the perspective of `shogi_core_definitions.py` itself, the attribute is fine -- it's correctly set in `__init__`, updated in `promote()` and `unpromote()`, and used consistently within the module.

### [159-168] MoveTuple union type is not runtime-discriminable by type system alone
**What:** `MoveTuple = Union[BoardMoveTuple, DropMoveTuple]` defines the union as:
- `BoardMoveTuple = Tuple[int, int, int, int, bool]`
- `DropMoveTuple = Tuple[Optional[int], Optional[int], int, int, PieceType]`

At runtime, the only way to distinguish them is by checking if `move[0] is None` (drop) or `isinstance(move[0], int)` (board move). This pattern is used correctly throughout the codebase (e.g., `shogi_game.py` line 604: `if move_tuple[0] is None`). However, the type annotations don't enforce this convention -- a static type checker sees `Optional[int]` which includes `int`, making the discrimination less precise.

**Why it matters:** The runtime discrimination pattern works but is fragile. If someone creates a move tuple where `move[0]` is `0` (a valid int for row 0 on the board), it could not be confused with a drop move since `None` and `0` are distinct. But the type signatures don't encode this contract. This is a design-level observation, not a bug.

## Observations

### [86-113] PieceType.to_usi_char() covers only droppable pieces
**What:** The `to_usi_char` method raises `ValueError` for `KING` and all promoted piece types. The docstring explains this is for drops, but the method name `to_usi_char` is more general than its implementation.
**Why it matters:** Naming could mislead a developer into calling `PieceType.KING.to_usi_char()` expecting a result. A name like `to_usi_drop_char` would be more precise. Minor maintainability issue.

### [393-509] Piece class is well-implemented
**What:** The `Piece` class has proper `__init__` type validation, `__eq__`, `__hash__`, `__repr__`, `__deepcopy__`, `promote()`, `unpromote()`, and `symbol()` methods. The `is_promoted` state is correctly derived from `piece_type` and maintained through promote/unpromote operations. The `__deepcopy__` method correctly stores the new piece in the `memo` dict.
**Why it matters:** No issues found. The implementation is solid.

### [447-457] Piece.promote() silently does nothing for non-promotable pieces
**What:** Calling `promote()` on a GOLD or KING piece is silently ignored. No error, no warning.
**Why it matters:** This is a reasonable design choice for a game engine -- callers check promotion eligibility before calling. But it could mask bugs if callers forget to check. The silent behavior is documented in the docstring.

### [227-243] PIECE_TYPE_TO_HAND_TYPE does not include KING
**What:** The King type is deliberately excluded from this mapping, with a comment explaining why. Code that captures a king would need to handle this separately.
**Why it matters:** The `shogi_game.py` `add_to_hand` method (line 785-786) explicitly checks for King and returns early, so this is handled correctly. The documentation is adequate.

### [135-148] TerminationReason values mix Japanese and English
**What:** `CHECKMATE = "Tsumi"`, `REPETITION = "Sennichite"`, but `STALEMATE = "stalemate"`, `RESIGNATION = "resignation"`. The inconsistency in capitalization and language (romanized Japanese vs. English) could be confusing.
**Why it matters:** These values are used in string comparisons (e.g., `shogi_game_io.py` termination map) and stored in game state. The comments note that specific values match test expectations, suggesting the values were aligned to existing tests rather than designed from a consistent convention.

### [12] Unnecessary comment on import
**What:** `from dataclasses import dataclass  # Added import` -- the comment "Added import" is a vestige of a code review or commit and adds no lasting value.

### [17-46] Comprehensive __all__ list
**What:** The `__all__` list is thorough, including types, constants, functions, and lookup tables. This is good practice for a foundational module.

### [316-348] get_piece_type_from_symbol handles case normalization
**What:** The function normalizes lowercase and promoted lowercase symbols (e.g., `"p"` to `"P"`, `"+p"` to `"+P"`). It raises `ValueError` for unknown symbols.
**Why it matters:** Defensive input handling is appropriate for a symbol parsing function that may receive user or file input.

## Verdict
**Status:** NEEDS_ATTENTION
**Recommended action:**
1. **Short-term**: Resolve the duplication of observation constants between this module and `keisei/constants.py`. Either have `constants.py` import from here (single source of truth) or remove one set of constants entirely.
2. **Short-term**: Add a clear docstring note to `OBS_UNPROMOTED_ORDER` explicitly warning that it includes KING and should NOT be used for hand piece channel iteration (use `get_unpromoted_types()` instead).
3. **Low priority**: Standardize `TerminationReason` value casing/language. Remove vestigial comments like `# Added import`.
**Confidence:** HIGH -- The module is well-structured with clear type definitions. The warnings are verified by cross-referencing `constants.py` and examining consumer code patterns. No hidden complexity or uncertain behavior.
