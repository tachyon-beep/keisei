# Analysis: keisei/shogi/shogi_game_io.py

**Lines:** 830
**Role:** Handles all serialization (SFEN/USI format encoding and decoding) and, critically, the neural network observation tensor encoding. This file is the bridge between human-readable game notation and the internal game state, as well as between game semantics and the 46-channel feature tensor that the neural network consumes. Errors here silently corrupt all training.
**Key dependencies:**
- Imports from: `shogi_core_definitions` (Color, Piece, PieceType, MoveTuple, all OBS_* constants, SYMBOL_TO_PIECE_TYPE, BASE_TO_PROMOTED_TYPE, PROMOTED_TYPES_SET, KIF_PIECE_SYMBOLS, get_unpromoted_types, TerminationReason)
- Imported by: `shogi_game.py` (as module), `features.py` (parallel implementation exists)
- Uses TYPE_CHECKING for: `shogi_game.ShogiGame`
**Analysis depth:** FULL

## Summary

The observation encoding in `generate_neural_network_observation` has a significant semantic mismatch in how hand piece planes are indexed compared to board piece planes, using `get_unpromoted_types()` (7 items, no King) for hand channels but `OBS_UNPROMOTED_ORDER` (8 items, includes King) for board channels -- this is correct in purpose but the code at line 502-504 assigns `hand_piece_order = get_unpromoted_types()` and iterates with `enumerate(hand_piece_order)`, producing indices 0-6 that happen to align with `OBS_CURR_PLAYER_HAND_START + 0` through `+6`, which is channel 28-34. This is correct. However, the hand normalization uses a hardcoded `/18.0` divisor that is wrong for all piece types except Pawn. The KIF export has multiple bugs including incorrectly formatted move notation and uses the current game hand state instead of the initial state. The `sys.path.insert` at module level is a code hygiene issue. The SFEN parsing is thorough and well-validated.

## Critical Findings

### [513-514] Hand piece normalization uses hardcoded `/18.0` for ALL piece types

**What:** When encoding hand piece counts into the observation tensor, the code normalizes all piece types by dividing by 18.0. This constant is documented as "max pawns" in the comment. However, the maximum possible count for each piece type in Shogi is: Pawn=18, Lance=4, Knight=4, Silver=4, Gold=4, Bishop=2, Rook=2. Using `/18.0` for all piece types means that a hand with 2 Rooks (the maximum possible) produces a value of `2/18 = 0.111`, while a hand with 2 Pawns produces the same `2/18 = 0.111`. The neural network cannot distinguish "I have all the Rooks" from "I have 2 out of 18 Pawns."

**Why it matters:** This is a **silent training corruption** issue. The observation tensor is the sole input to the neural network. By using an incorrect normalization constant, the network receives a signal that systematically underrepresents the significance of having major pieces (Bishops, Rooks) in hand. A hand with 2 Rooks (maximum power) looks the same as having 2 of 18 Pawns (trivial). This directly degrades the quality of the learned policy and value function. The network can still learn to some degree because the relative ordering within each channel is preserved, but the signal-to-noise ratio is severely degraded for high-value pieces.

**Evidence:**
```python
# Current player's hand
for i, piece_type_enum_player in enumerate(hand_piece_order):
    player_hand_count: int = game.hands[game.current_player.value].get(
        piece_type_enum_player, 0
    )
    if player_hand_count > 0:
        player_ch: int = OBS_CURR_PLAYER_HAND_START + i
        obs[player_ch, :, :] = (
            player_hand_count / 18.0
        )  # Normalize (e.g., by max pawns)
```
The same issue repeats for the opponent's hand at line 527:
```python
obs[opponent_ch, :, :] = opponent_hand_count / 18.0
```

### [34] Module-level `sys.path.insert` pollutes Python path globally

**What:** Line 34 contains `sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))`. This modifies `sys.path` at module import time, inserting the project root at position 0.

**Why it matters:** This is a global side effect that occurs on every import of this module. It can cause import shadowing where the project root's modules override system packages. It also means that importing this module changes the behavior of all subsequent imports in the process. In a production DRL training system, this could cause hard-to-diagnose import conflicts, especially with packages that have common names. Since the package already uses relative imports (`from .shogi_core_definitions import ...`), this `sys.path` modification appears to be unnecessary legacy code.

**Evidence:**
```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
```

### [646-661] KIF export uses current game hands, not initial position hands

**What:** The `game_to_kif` function generates a KIF file with a hardcoded HIRATE (standard) starting position for the board (lines 622-630), but then reads `game.hands[Color.BLACK.value]` and `game.hands[Color.WHITE.value]` for the hand section (lines 646-647). In a standard HIRATE game, hands start empty. But by the time `game_to_kif` is called (typically after a game ends), the hands contain the pieces accumulated during play.

**Why it matters:** The KIF file will contain a standard starting board position but with end-of-game hand contents, creating a physically impossible and invalid initial position. Any KIF parser that attempts to replay the game from this state will produce incorrect results. If KIF files are used for game review, analysis, or dataset creation, the data will be corrupt.

**Evidence:**
```python
# Hardcoded HIRATE board (lines 622-630)
lines.append("P1-KY-KE-GI-KI-OU-KI-GI-KE-KY")
# ...
# But hands come from current game state:
initial_sente_hand: Dict[PieceType, int] = game.hands[Color.BLACK.value]  # line 646
initial_gote_hand: Dict[PieceType, int] = game.hands[Color.WHITE.value]  # line 647
```

## Warnings

### [674-686] KIF move notation format is incorrect

**What:** The KIF move encoding at lines 681-682 produces a format like `"1a2a"` (1-indexed row + letter column) instead of standard USI format `"9a8a"` (file number + rank letter) or standard KIF format with Japanese square notation. The code uses `move_obj[0]+1` (row + 1) and `chr(move_obj[1]+ord('a'))` (column as letter), which produces a non-standard format that no known Shogi tool can parse.

**Why it matters:** KIF files produced by this function will have unparseable move notation. Any tool attempting to read these files (GUIs, analysis engines, databases) will fail.

**Evidence:**
```python
usi_move_str: str = (
    f"{move_obj[0]+1}{chr(move_obj[1]+ord('a'))}{move_obj[2]+1}{chr(move_obj[3]+ord('a'))}"
)
```
For a move from (0,0) to (1,0), this produces `"1a2a"` instead of the correct `"9a8a"`.

### [674-680] KIF export silently skips drop moves

**What:** The KIF move export checks if `move_obj[0] is None or move_obj[1] is None` (drop move indicators) and `continue`s, skipping them entirely. Drop moves are a fundamental part of Shogi and any game record missing them is incomplete and invalid.

**Why it matters:** Games containing drops (which is virtually every Shogi game of significant length) will have incomplete move records in KIF format. The resulting file will be corrupted.

**Evidence:**
```python
if (
    move_obj[0] is None
    or move_obj[1] is None
    or move_obj[2] is None
    or move_obj[3] is None
):
    continue  # Skip malformed move -- but drops have None for [0] and [1]
```

### [466-469] Board perspective flip applies to coordinates but board is already perspective-independent

**What:** In `generate_neural_network_observation`, when the current player is White, the code flips both row and column coordinates: `flipped_r = 8 - r`, `flipped_c = 8 - c`. This means White sees the board rotated 180 degrees. The board representation itself is always stored from Black's perspective (row 0 = rank 1 = White's back rank). The flip ensures that the neural network always sees "my pieces are near the bottom" regardless of which color it's playing.

**Why it matters:** This is actually correct behavior for RL training -- it provides input symmetry. However, it is **critical** that this flipping is consistent with the `PolicyOutputMapper` that translates network outputs back to actual board moves. If the policy mapper does not account for this 180-degree rotation for White, the network will learn to play moves on the wrong squares. This is an **implicit contract** that is not documented anywhere in this file. Any change to the flip logic here without a corresponding change in the policy mapper will silently corrupt all White-side play.

**Evidence:**
```python
flipped_r = r if is_black_perspective else 8 - r
flipped_c = c if is_black_perspective else 8 - c
```

### [502-504] Hand piece order uses `get_unpromoted_types()` which returns a new list each call

**What:** `hand_piece_order = get_unpromoted_types()` is called inside the observation function, which is called on every single step of training. `get_unpromoted_types()` creates and returns a new list `[PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK]` on every call.

**Why it matters:** Minor performance issue. In a hot path called potentially millions of times during training, allocating a new list every time is wasteful. It should be a module-level constant.

**Evidence:**
```python
hand_piece_order: List[PieceType] = (
    get_unpromoted_types()
)  # Use the imported function
```
And from `shogi_core_definitions.py`:
```python
def get_unpromoted_types() -> List[PieceType]:
    return [
        PieceType.PAWN, PieceType.LANCE, PieceType.KNIGHT,
        PieceType.SILVER, PieceType.GOLD, PieceType.BISHOP, PieceType.ROOK,
    ]
```

### [530-531] Current player indicator plane is perspective-inconsistent

**What:** The current player indicator plane is set to `1.0` if `current_player == Color.BLACK` and `0.0` if White. Combined with the board flip for White's perspective, this means the network receives: when Black, indicator=1.0 and board is from Black's viewpoint; when White, indicator=0.0 and board is from White's viewpoint (flipped). The indicator is providing redundant information (the network already knows whose perspective it's seeing from the flip), but more importantly, the indicator's semantics are "am I Black?" rather than "am I the current player" (which is always true).

**Why it matters:** This is a design issue rather than a bug. The plane could be removed or repurposed without loss of information if the board flip is always applied. However, changing it would require retraining.

**Evidence:**
```python
obs[OBS_CURR_PLAYER_INDICATOR, :, :] = (
    1.0 if game.current_player == Color.BLACK else 0.0
)
```

### [690-697] KIF termination reason map is stale relative to `TerminationReason` enum values

**What:** The `termination_map` in `game_to_kif` uses string keys like `"Tsumi"`, `"Sennichite"`, `"Stalemate"`, `"Max moves reached"`. These must match the `.value` strings from `TerminationReason`. The enum defines: `CHECKMATE = "Tsumi"`, `STALEMATE = "stalemate"` (lowercase 's'), `REPETITION = "Sennichite"`, `MAX_MOVES_EXCEEDED = "Max moves reached"`. The map key `"Stalemate"` (capital S) will never match `TerminationReason.STALEMATE.value` which is `"stalemate"` (lowercase s).

**Why it matters:** When a game ends in stalemate, the KIF output will display the raw termination reason string `"stalemate"` instead of the Japanese `"持将棋"` translation. This is a display bug, not a data integrity issue.

**Evidence:**
```python
# In game_to_kif:
termination_map: Dict[str, str] = {
    "Tsumi": "...",
    "Stalemate": "持将棋",  # Capital S -- won't match "stalemate"
    ...
}
# In shogi_core_definitions.py:
STALEMATE = "stalemate"  # lowercase 's'
```

## Observations

### [1-7] Import of `datetime` and `os` are only used by `game_to_kif`

**What:** `datetime` is imported only for the KIF date header, and `os` is imported only for the `sys.path.insert`. Both are module-level imports for functions that may rarely be called.

**Why it matters:** Minor code hygiene. The `os` import supports the problematic `sys.path.insert` and would be unnecessary if that line were removed.

### [40-55] `SFEN_BOARD_CHARS` maps promoted pieces to their base character

**What:** The mapping intentionally maps e.g., `PROMOTED_PAWN: "P"` and `PROMOTED_LANCE: "L"`. This is correct for SFEN, where promoted pieces are represented as `+P`, `+L`, etc., with the `+` prefix added separately in `_get_sfen_board_char`.

**Why it matters:** Observation only -- the design is correct.

### [96-98] `_get_sfen_board_char` lowercases the entire SFEN character for White

**What:** For White pieces, the entire SFEN string (including the `+` for promoted pieces) is lowercased. This produces `+p` for a White promoted pawn. Standard SFEN uses `+p` for White promoted pawn, which is correct.

**Why it matters:** Observation only -- correct behavior.

### [69-76] Feature builder `build_core46` in `features.py` diverges from `generate_neural_network_observation`

**What:** The `features.py` file contains a parallel implementation `build_core46` that "mirrors" `generate_neural_network_observation`. However, `build_core46` calls `piece.is_promoted()` as a method, while the actual `Piece` class uses `piece.is_promoted` as a property (boolean attribute set in `__init__`). Additionally, `build_core46` references `game.OBS_CURR_PLAYER_UNPROMOTED_START` as a game attribute, but these are module-level constants in `shogi_core_definitions.py`, not attributes of the game object. Also, `build_core46` does NOT apply the board perspective flip for White that `generate_neural_network_observation` does.

**Why it matters:** While this is technically in `features.py` (not in the assigned file), it is directly relevant because it claims to mirror the assigned file's function. If `features.py`'s `build_core46` is ever used instead of `generate_neural_network_observation`, it will produce different (and incorrect) observations: no perspective flip, incorrect attribute access patterns, and calling `is_promoted` as a method instead of accessing it as a property. The `build_core46` function appears to be dead/broken code that would crash if called.

### [319-326] SFEN board serialization iterates columns 0-8 directly

**What:** The comment at lines 323-325 discusses file ordering but the actual iteration at line 326 goes from column 0 to 8. In the internal representation, column 0 is file 9 (leftmost from Black's view) and column 8 is file 1. Standard SFEN iterates from file 9 to file 1 (left to right from Black's view), which matches column 0 to 8. This is correct.

**Why it matters:** Observation only -- confirmed correct.

### [779-826] SFEN move parsing is well-implemented

**What:** The `sfen_to_move_tuple` function uses clear regex patterns for both drop and board moves, with proper error handling and validation.

**Why it matters:** This is well-written code with good error messages.

### [186-254] Board SFEN parsing validates thoroughly

**What:** `populate_board_from_sfen_segment` checks for double promotion tokens, digits after promotion tokens, zero digits, column overflow, and column underflow. This is thorough parsing.

**Why it matters:** Positive observation -- this is defensive, well-structured parsing.

## Verdict

**Status:** NEEDS_ATTENTION
**Recommended action:** (1) HIGHEST PRIORITY: Fix the hand piece normalization to use per-piece-type maximum counts (18 for Pawn, 4 for Lance/Knight/Silver/Gold, 2 for Bishop/Rook) -- this directly affects training quality. (2) Remove the `sys.path.insert` at line 34. (3) Fix the KIF export: use initial empty hands for HIRATE games, implement drop move notation, fix the row/column format to match standard notation, and fix the case-sensitive termination reason mismatch. (4) Document the implicit contract between the observation's board perspective flip and the PolicyOutputMapper. (5) Consider caching `get_unpromoted_types()` as a module constant.
**Confidence:** HIGH -- Full read of all 830 lines, cross-referenced with `shogi_core_definitions.py`, `features.py`, and the game class. The hand normalization finding is based on straightforward arithmetic against known Shogi piece counts. The KIF bugs are verifiable by reading the code against KIF format specifications.
