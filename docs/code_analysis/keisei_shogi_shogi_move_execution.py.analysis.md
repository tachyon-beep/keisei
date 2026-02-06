# Analysis: keisei/shogi/shogi_move_execution.py

**Lines:** 222
**Role:** Applies and reverts moves on the Shogi board. `apply_move_to_board_state` mutates board and hands for a given move (captures, promotions, drops). `apply_move_to_game` updates game-level state (player switch, move count). `revert_last_applied_move` restores full game state for simulation undo. This is the write path for all game state mutations.
**Key dependencies:** Imports from `shogi_core_definitions` (BASE_TO_PROMOTED_TYPE, PROMOTED_TO_BASE_TYPE, Color, MoveApplicationResult, MoveTuple, Piece, PieceType, TerminationReason). Consumed by `shogi_game.py` (make_move, undo_move). `TerminationReason` is imported but unused in the active code.
**Analysis depth:** FULL

## Summary

The file is well-structured with good input validation and clear separation of concerns. The `apply_move_to_board_state` function has thorough type checking and raises meaningful errors for invalid inputs. However, there is one critical concern: `revert_last_applied_move` performs a shallow copy of nested mutable objects during state restoration, which could lead to shared-state corruption in specific scenarios. There is also an unused import and some dead commented-out code that should be cleaned up. Overall confidence is HIGH.

## Critical Findings

### [211] `revert_last_applied_move` performs shallow copy of hands dictionary values

**What:** When restoring the hands state, line 212 uses `{k: v.copy() for k, v in original_hands_state.items()}`. The `.copy()` on the inner dict (`v`) is a shallow copy, which is correct for `Dict[PieceType, int]` (since ints are immutable). However, line 211 restores the board with `[row[:] for row in original_board_state]`, which is a shallow copy of each row. Since each cell contains either `None` or a `Piece` object, and `Piece` objects are mutable (they have `.promote()` and `.unpromote()` methods that mutate in-place), the restored board shares `Piece` instances with the `original_board_state` snapshot.

**Why it matters:** The `original_board_state` passed to this function comes from `copy.deepcopy(self.board)` in `ShogiGame.make_move` (line 635 of shogi_game.py), so the Piece objects in the snapshot are independent copies. The shallow restore on line 211 therefore creates a board where `game.board[r][c]` points to the same `Piece` objects as `original_board_state[r][c]`. If the caller subsequently modifies `original_board_state` (or the `simulation_undo_details` dict that contains it), the game's board would be silently corrupted. In the current code path, `simulation_undo_details` is a local dict that goes out of scope after `undo_move` returns, so this is not an active bug. But it is a latent hazard: if the undo details are ever retained (e.g., for move history replay or debugging), mutation of those Piece objects would corrupt the live game state.

**Evidence:**
```python
# Line 211: Shallow row copy -- Piece objects are shared, not cloned
game.board = [row[:] for row in original_board_state]
# Line 212: Shallow dict copy -- safe because values are ints
game.hands = {k: v.copy() for k, v in original_hands_state.items()}
```

Compare with `ShogiGame.make_move` which uses `copy.deepcopy`:
```python
# shogi_game.py line 635:
move_details_for_history["original_board_state"] = copy.deepcopy(self.board)
```

The asymmetry (deep copy on save, shallow copy on restore) is a design smell that could cause bugs if the lifecycle of `simulation_undo_details` changes.

## Warnings

### [219-221] `revert_last_applied_move` unconditionally resets `game_over`, `winner`, and `termination_reason`

**What:** Lines 219-221 always set `game.game_over = False`, `game.winner = None`, and `game.termination_reason = None`. This is appropriate for simulation undo (the primary use case), but if this function were ever called for a non-simulation revert, it would destroy legitimate termination state.

**Why it matters:** The function signature does not restrict its use to simulation contexts. If a future developer calls it for a real undo (e.g., implementing "take back" functionality), the termination state would be silently cleared even if the position being reverted to was itself a terminal state. This is a fragile implicit contract.

**Evidence:**
```python
# Lines 219-221: Always clears termination state
game.game_over = False
game.winner = None
game.termination_reason = None  # Use None, not TerminationReason.ACTIVE.value
```

### [109-113] Captured piece added to hand using `.get()` with default 0, bypassing hand initialization

**What:** When a capture occurs, line 112-114 uses `hands[current_player.value].get(captured_piece_type, 0) + 1` to add the captured piece to hand. This `.get()` with default 0 means that even if the hand was not initialized with an entry for `captured_piece_type`, the operation silently succeeds and creates a new entry.

**Why it matters:** In the normal flow, hands are initialized with all 7 droppable piece types set to 0. But if hands initialization is ever incomplete (e.g., from a buggy SFEN parser), this code would silently create entries for unexpected piece types rather than raising an error. Since `PROMOTED_TO_BASE_TYPE` correctly maps promoted types to their base types (line 109-110), the piece type added to hand will always be a valid base type. However, the `.get()` with default masks potential initialization bugs.

**Evidence:**
```python
captured_piece_type = PROMOTED_TO_BASE_TYPE.get(
    target_square_piece.type, target_square_piece.type
)
hands[current_player.value][captured_piece_type] = (
    hands[current_player.value].get(captured_piece_type, 0) + 1
)
```

### [9] Unused import: `TerminationReason`

**What:** `TerminationReason` is imported on line 9 but is not used anywhere in the active code. It appears in commented-out code (lines 177, 181, 185) that was part of the old termination-checking logic that has been moved to `ShogiGame._check_and_update_termination_status`.

**Why it matters:** Dead import. Minor cleanup item, but contributes to import confusion when reading the module.

**Evidence:** Line 9: `from .shogi_core_definitions import TerminationReason` -- grep of active code shows no usage.

### [117-120] Piece identity mutation: moving piece retains original object reference

**What:** On line 117-119, `board[to_r][to_c] = moving_piece` assigns the same `Piece` object to the destination square. If promotion occurs (lines 122-131), a new `Piece` is created at the destination. But for non-promoting moves, the `Piece` object at the destination is the exact same Python object that was at the source. Line 120 then sets `board[from_r][from_c] = None`.

**Why it matters:** This means the Piece object is shared between the board cell and any other reference to it. In practice this is fine because `generate_all_legal_moves` in `shogi_rules_logic.py` only reads piece attributes (type, color) and does not mutate pieces. But if any code path ever calls `piece.promote()` or `piece.unpromote()` on a board piece directly (rather than creating a new Piece), it would mutate the same object potentially still referenced elsewhere. The `Piece` class has mutable `.promote()` and `.unpromote()` methods, making this a latent risk.

**Evidence:**
```python
# Lines 117-120: moving_piece is the SAME object, not a copy
board[to_r][to_c] = moving_piece
board[from_r][from_c] = None
```

## Observations

### [141-186] `apply_move_to_game` is mostly dead code

**What:** The function body is 5 active lines (153-156) and 22 lines of commented-out code (164-185). The commented-out code is the old termination-checking logic that was moved to `_check_and_update_termination_status`.

**Why it matters:** The commented-out code adds noise and makes the function appear more complex than it is. It should be removed now that the refactoring is complete.

### [24-138] `apply_move_to_board_state` has thorough input validation

**What:** The function validates: (a) to_r and to_c are integers (line 50-51), (b) drop moves have PieceType as 5th element (line 56-59), (c) drop piece is in hand (line 65-70), (d) board moves have integer coordinates (line 77-79), (e) board moves have boolean promote flag (line 84-87), (f) source square has a piece (line 91-93), (g) source piece belongs to current player (line 95-98), (h) target square does not contain friendly piece (line 103-107), (i) promotion is valid for the piece type (line 123-127).

**Why it matters:** This is excellent defensive programming. Every invalid input is caught with a clear `ValueError` message. This is the correct approach for a function that sits at the boundary between rule validation and state mutation.

### [188-221] `revert_last_applied_move` has clear but verbose interface

**What:** The function takes 4 separate state parameters (board, hands, player, move_count) rather than a single state snapshot object. This makes the call site in `ShogiGame.undo_move` verbose and error-prone (each parameter must be correctly extracted from the undo details dict).

**Why it matters:** A `GameStateSnapshot` dataclass would simplify the interface and reduce the risk of passing parameters in the wrong order. This is a design observation, not a bug.

### [64] Drop move creates new Piece instance directly on board

**What:** `board[to_r][to_c] = Piece(piece_to_drop_type, current_player)` creates a fresh Piece for every drop. This is correct and avoids sharing objects with the hand (which stores counts, not Piece instances).

**Why it matters:** No issue -- this is the correct approach.

## Verdict

**Status:** SOUND
**Recommended action:** (1) Change the shallow board copy in `revert_last_applied_move` (line 211) to `copy.deepcopy(original_board_state)` to match the deep-copy-on-save pattern and eliminate the shared-Piece-object hazard. (2) Remove the unused `TerminationReason` import. (3) Remove the 22 lines of commented-out termination logic in `apply_move_to_game`. These are minor cleanups; the file is functionally correct for its current use patterns.
**Confidence:** HIGH -- every line was read, all call sites in `shogi_game.py` were traced, and the interaction with `shogi_rules_logic.py` was verified.
