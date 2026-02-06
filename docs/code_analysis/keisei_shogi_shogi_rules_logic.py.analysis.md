# Analysis: keisei/shogi/shogi_rules_logic.py

**Lines:** 696
**Role:** Core Shogi rules engine -- determines legal moves, detects check/checkmate, handles special rules (nifu, uchi fu zume, sennichite). This is the single source of truth for move legality in the entire DRL training pipeline.
**Key dependencies:** Imports from `shogi_core_definitions` (Color, Piece, PieceType, MoveTuple, BASE_TO_PROMOTED_TYPE). Consumed by `shogi_game.py` (all legal move generation, check detection, termination). Also called by `shogi_move_execution.py` indirectly through game orchestration.
**Analysis depth:** FULL

## Summary

The file is functionally correct for the core Shogi rules it implements, and the design of separating rules logic from game state management is sound. However, there are two significant performance/correctness concerns: (1) a massive performance problem in `generate_all_legal_moves` that performs redundant full check-detection work on every candidate move, doubling the already-expensive simulation cost, and (2) `check_for_uchi_fu_zume` mutates game state in-place without exception safety, creating a latent data corruption risk. Confidence in the analysis is HIGH -- every line was read and cross-referenced against the game class.

## Critical Findings

### [527-543] Redundant full attack-check computation on every simulated board move

**What:** Inside the board-move loop of `generate_all_legal_moves`, after calling `game.make_move(move_tuple, is_simulation=True)`, the code performs a full `find_king` + `check_if_square_is_attacked` computation (lines 527-543) that is entirely separate from -- and redundant with -- the authoritative `is_king_in_check_after_simulated_move` call on line 550-552. The result of the first computation (`is_attacked_after_sim`) is assigned to local variables (`king_is_safe_eval`) but **never used to determine legality**. The actual legality decision uses `king_is_safe` from line 550. The identical pattern is repeated for drop moves (lines 589-609).

**Why it matters:** For a DRL training loop, `generate_all_legal_moves` is called on every single step. In the worst case (mid-game with many pieces and hand pieces), there are hundreds of candidate moves. For each candidate, this code performs **two** full 81-square attack scans instead of one. This doubles the cost of legal move generation -- the single most expensive operation in the game engine -- and directly impacts training throughput. For a system designed to run millions of self-play episodes, this is a significant performance drag. While not a correctness bug (the authoritative check on line 550 is correct), this is dead computation with real cost.

**Evidence:**
```python
# Lines 527-543: First check (UNUSED result)
king_pos_trace = find_king(game, original_player_color)
king_r_trace, king_c_trace = (king_pos_trace if king_pos_trace else (-1, -1))
...
is_attacked_after_sim = check_if_square_is_attacked(
    game, king_r_trace, king_c_trace, opponent_color_trace
)
...
king_is_safe_eval = not is_attacked_after_sim  # <-- NEVER USED FOR DECISION

# Lines 550-553: Second check (AUTHORITATIVE)
king_is_safe = not is_king_in_check_after_simulated_move(
    game, original_player_color
)
if king_is_safe:
    legal_moves.append(move_tuple)
```

### [311-359] `check_for_uchi_fu_zume` mutates game state in-place without exception safety

**What:** The function directly mutates `game.board` (via `set_piece`), `game.hands`, and `game.current_player` to simulate a pawn drop, then relies on a manual revert at the end. If any exception occurs between the mutation (lines 313-314) and the revert (lines 351-353) -- for example, if `generate_all_legal_moves` raises, or if `find_king` or `check_if_square_is_attacked` encounters an unexpected state -- the game object is left in a corrupted state with an extra pawn on the board, one fewer pawn in hand, and potentially the wrong current player.

**Why it matters:** This function is called during legal move generation, which is called during training. A corrupted game state means the RL agent trains on an illegal board position. Because the corruption is silent (no exception propagates to the caller if the inner call succeeds but leaves state partially modified), the training signal is poisoned without any alert. The comment at line 467 ("DO NOT COMMENT THIS OUT, IT IS A LOAD BEARING RETURN") suggests this area has been a source of bugs before.

**Evidence:**
```python
# Lines 313-314: Mutation
game.set_piece(drop_row, drop_col, Piece(PieceType.PAWN, color))
game.hands[color.value][PieceType.PAWN] -= 1

# Lines 330-332: Could raise if game state is unexpected
drop_delivers_check = check_if_square_is_attacked(
    game, opp_king_pos[0], opp_king_pos[1], color
)

# Lines 347: Could raise if generate_all_legal_moves encounters issues
opponent_legal_moves = generate_all_legal_moves(game, is_uchi_fu_zume_check=True)

# Lines 351-353: Revert -- only reached if no exception
game.set_piece(drop_row, drop_col, None)
game.hands[color.value][PieceType.PAWN] += 1
game.current_player = original_current_player
```

A `try/finally` block should wrap the mutation-to-revert span to guarantee cleanup.

## Warnings

### [250-272] `check_if_square_is_attacked` is O(81 * max_piece_moves) per call, called in nested loops

**What:** This function iterates all 81 squares looking for attackers, and for each attacker calls `generate_piece_potential_moves` (which itself iterates up to 8 sliding directions x 8 steps). It is called inside `generate_all_legal_moves` for every candidate move (via `is_king_in_check_after_simulated_move`), and redundantly a second time directly (as noted in the critical finding above). In the worst case, legal move generation is O(N_candidates * 81 * M_moves_per_piece).

**Why it matters:** While Shogi boards are small enough that this is "fast enough" for correctness, for a DRL training loop running millions of episodes, this is a bottleneck. The function could be optimized with incremental attack maps or early termination strategies, but this is a design choice rather than a bug.

**Evidence:** Lines 250-272: nested loop over all 81 squares for every attack check.

### [562-566] `PieceType(piece_type_to_drop_val)` conversion may raise ValueError for unexpected hand contents

**What:** In the drop-move generation section, `piece_type_to_drop_val` comes from iterating `game.hands[original_player_color.value].items()`. If the hand dictionary ever contains a key that is not a valid `PieceType` value (due to external corruption or deserialization error), `PieceType(piece_type_to_drop_val)` will raise a `ValueError` that propagates up unhandled.

**Why it matters:** During training, this would crash the episode. While the hands dictionary is typed as `Dict[PieceType, int]`, the actual key type at runtime depends on how hands are initialized and maintained across serialization boundaries.

**Evidence:**
```python
for piece_type_to_drop_val, count in game.hands[
    original_player_color.value
].items():
    if count > 0:
        piece_type_to_drop = PieceType(piece_type_to_drop_val)  # Could raise
```

### [42-55] `is_in_check` returns `True` when king is not found -- masks serious state corruption

**What:** If `find_king` returns `None` (king not on board), the function returns `True` (in check). While the comment says "King not found implies a lost/invalid state", this silently treats a catastrophically corrupt game state as merely "in check" rather than raising an error.

**Why it matters:** If the king is absent due to a bug in move execution or undo logic, returning `True` for "in check" causes the legal move generator to believe the player is in check, potentially generating incorrect moves or declaring checkmate when the real problem is missing pieces. This masks the root cause of the corruption and makes debugging much harder.

**Evidence:**
```python
if not king_pos:
    if debug_recursion:
        ...
        print(...)
    return True  # King not found implies a lost/invalid state, effectively in check.
```

### [208] `list(set(moves))` removes duplicates but destroys move ordering

**What:** At the end of `generate_piece_potential_moves`, duplicate moves are removed via `list(set(moves))`. This can happen when promoted bishop/rook extra offset moves overlap with sliding moves on the same square.

**Why it matters:** The deduplication is correct, but converting through a set destroys any deterministic ordering. Since this feeds into legal move generation which feeds into the RL action space, non-deterministic ordering between runs (even with the same seed) could make training harder to reproduce. Python 3.7+ dicts are ordered but sets are not guaranteed to have stable iteration order across runs (hash randomization).

**Evidence:** Line 208: `return list(set(moves))`

### [326] Unnecessary restoration of `game.current_player` in early-return path of `check_for_uchi_fu_zume`

**What:** On line 326, in the early return when `opp_king_pos` is None, the code restores `game.current_player = original_current_player`. However, `game.current_player` has not been modified at this point in the function (the modification happens on line 345, which is after this branch). This is not a bug but indicates the revert logic was written defensively without tracking which mutations have actually occurred.

**Why it matters:** Minor maintainability concern -- it suggests the developer was uncertain about which state was modified at each point, which increases the risk of future bugs when modifying this function.

**Evidence:**
```python
# Line 326 -- current_player has NOT been changed yet at this point
game.current_player = original_current_player  # Restore original player
```

### [50-51, 244-248] Debug print statements using `print()` instead of logger

**What:** Multiple debug print statements exist throughout the file (lines 50-51, 60-61, 244-248, 255-257, 264-265, etc.) that use raw `print()` rather than the project's unified logger.

**Why it matters:** During training, if `debug_recursion=True` is accidentally left on, these prints go to stdout and pollute the Rich console UI. The project's `CLAUDE.md` explicitly states: "Always use the unified logger (`utils/unified_logger.py`) for consistent Rich-formatted output."

**Evidence:** Lines 50-51, 60-61, 244-248, 255-257, 264-265 all use `print()`.

## Observations

### [486-635] `generate_all_legal_moves` is 150 lines with deep nesting

**What:** This is the most important function in the file, and it spans 150 lines with multiple levels of nesting (iteration over board, iteration over potential moves, promotion options, simulation + undo). The board-move and drop-move sections are structurally similar but not factored into helper functions.

**Why it matters:** The length and nesting depth make this function hard to review and maintain. The duplicated trace/debug blocks for board moves (527-548) and drop moves (589-614) are nearly identical, which increases the risk of one being updated while the other is forgotten.

### [638-695] `check_for_sennichite` relies on hash comparison with subtle timing semantics

**What:** The sennichite check compares the hash stored in `game.move_history[-1]["state_hash"]` against all other entries. The extensive comments (lines 646-677) explain the subtle timing of when the hash is captured relative to the player switch. The implementation is correct as written, but the complexity of the timing semantics (hash captured before player switch, sennichite checked after player switch) is a maintenance risk.

**Why it matters:** If anyone modifies the order of operations in `ShogiGame.make_move` (where the hash is captured and the player is switched), the sennichite detection could silently break without any test catching it, because the hash comparison would stop matching historical entries.

### [382-421] `can_promote_specific_piece` and `must_promote_specific_piece` are clean and correct

**What:** These two functions correctly implement Shogi promotion rules. `can_promote` checks promotion zone eligibility, `must_promote` enforces mandatory promotion for pawns/lances on last rank and knights on last two ranks.

**Why it matters:** No issues found. These are well-implemented and well-tested.

### [70-79] `is_piece_type_sliding` is clean but slightly inconsistent with move generation

**What:** This function declares sliding types as {LANCE, BISHOP, ROOK, PROMOTED_BISHOP, PROMOTED_ROOK}. However, in `generate_piece_potential_moves`, the sliding logic is handled inline with separate branches for each piece type. The `is_piece_type_sliding` function is only used by `ShogiGame._is_sliding_piece_type`, which does not appear to be called anywhere in the codebase (it is a delegating wrapper but no callers were found in the analyzed files).

**Why it matters:** Potential dead code. If `_is_sliding_piece_type` is not called externally, both it and `is_piece_type_sliding` could be removed.

## Verdict

**Status:** NEEDS_ATTENTION
**Recommended action:** (1) Remove the redundant attack-check computation in `generate_all_legal_moves` (lines 527-543 and 589-609) to approximately halve legal-move-generation cost during training. (2) Wrap the mutation span in `check_for_uchi_fu_zume` with a `try/finally` to guarantee state restoration. (3) Consider raising an exception rather than returning `True` when king is not found in `is_in_check`, to surface state corruption immediately rather than masking it.
**Confidence:** HIGH -- every line was read, all imports and callers were traced, and the analysis was cross-referenced against test coverage in `test_shogi_rules_and_validation.py` and `test_shogi_game_core_logic.py`.
