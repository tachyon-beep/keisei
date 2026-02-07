"""
Core Shogi game rules, move generation, and validation logic.
Functions in this module operate on a ShogiGame instance.
"""

from typing import TYPE_CHECKING, List, Optional, Set, Tuple  # Added Set

# Ensure all necessary types are imported:
from .shogi_core_definitions import (
    BASE_TO_PROMOTED_TYPE,
    Color,
    MoveTuple,
    Piece,
    PieceType,
)

if TYPE_CHECKING:
    from .shogi_game import ShogiGame  # For type hinting the 'game' parameter


# --- Helper functions for move generation and validation ---


# ADDED: Centralized king finding function
def find_king(game: "ShogiGame", color: Color) -> Optional[Tuple[int, int]]:
    """Finds the king of the specified color on the board."""
    for r_k in range(9):
        for c_k in range(9):
            p = game.get_piece(r_k, c_k)
            if p and p.type == PieceType.KING and p.color == color:
                return (r_k, c_k)
    return None


# ADDED: is_in_check function to be called by ShogiGame.is_in_check
def is_in_check(
    game: "ShogiGame", player_color: Color, debug_recursion: bool = False
) -> bool:
    """Checks if the king of 'player_color' is in check."""
    king_pos: Optional[Tuple[int, int]] = find_king(game, player_color)

    if not king_pos:
        if debug_recursion:
            # Try to get SFEN, but game object might be in an intermediate state for this print
            sfen_str: str = "unavailable (game object might be partial)"
            try:
                sfen_str = game.to_sfen_string()
            except Exception:  # pylint: disable=broad-except
                pass  # Keep sfen_str as unavailable
            print(
                f"DEBUG_IS_IN_CHECK: King of color {player_color} not found. Game state (SFEN): {sfen_str}. Returning True (check)."
            )
        return (
            True  # King not found implies a lost/invalid state, effectively in check.
        )

    opponent_color: Color = Color.WHITE if player_color == Color.BLACK else Color.BLACK

    if debug_recursion:
        print(
            f"DEBUG_IS_IN_CHECK: [{player_color}] King at {king_pos}. Checking if attacked by {opponent_color}. Debug on."
        )
        # Detailed print for attack check will come from check_if_square_is_attacked

    return check_if_square_is_attacked(
        game, king_pos[0], king_pos[1], opponent_color, debug=debug_recursion
    )


def is_piece_type_sliding(piece_type: PieceType) -> bool:  # Removed 'game' parameter
    """Returns True if the piece type is a sliding piece (Lance, Bishop, Rook or their promoted versions)."""
    sliding_types: Set[PieceType] = {
        PieceType.LANCE,
        PieceType.BISHOP,
        PieceType.ROOK,
        PieceType.PROMOTED_BISHOP,
        PieceType.PROMOTED_ROOK,
    }
    return piece_type in sliding_types


def generate_piece_potential_moves(
    game: "ShogiGame", piece: Piece, r_from: int, c_from: int
) -> List[Tuple[int, int]]:  # Changed to List[Tuple[int, int]]
    """
    Returns a list of (r_to, c_to) tuples for a piece, considering its
    fundamental movement rules and path-blocking by other pieces.
    This function generates squares a piece *attacks* or can move to if empty.
    It stops at the first piece encountered. If that piece is an opponent,
    the square is included (as a capture). If friendly, it's not included.
    (Formerly ShogiGame.get_individual_piece_moves)
    """

    moves: List[Tuple[int, int]] = []
    # Black (Sente, 0) moves towards smaller row indices, White (Gote, 1) towards larger
    forward: int = (
        -1 if piece.color == Color.BLACK else 1
    )  # Assuming game instance has Color enum

    piece_type: PieceType = piece.type

    # Define move offsets
    gold_move_offsets: List[Tuple[int, int]] = [
        (forward, 0),
        (forward, -1),
        (forward, 1),
        (0, -1),
        (0, 1),
        (-forward, 0),  # Backwards for Gold
    ]
    king_move_offsets: List[Tuple[int, int]] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    knight_move_offsets: List[Tuple[int, int]] = [(forward * 2, -1), (forward * 2, 1)]
    silver_move_offsets: List[Tuple[int, int]] = [
        (forward, 0),
        (forward, -1),
        (forward, 1),
        (-forward, -1),
        (-forward, 1),  # Backwards-diagonal
    ]
    promoted_rook_extra_offsets: List[Tuple[int, int]] = [
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]
    promoted_bishop_extra_offsets: List[Tuple[int, int]] = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    current_offsets: List[Tuple[int, int]] = []
    is_sliding: bool = False

    if piece_type == PieceType.PAWN:
        current_offsets = [(forward, 0)]
    elif piece_type == PieceType.KNIGHT:
        current_offsets = knight_move_offsets
    elif piece_type == PieceType.SILVER:
        current_offsets = silver_move_offsets
    elif (
        piece_type == PieceType.GOLD
        or piece_type == PieceType.PROMOTED_PAWN
        or piece_type == PieceType.PROMOTED_LANCE
        or piece_type == PieceType.PROMOTED_KNIGHT
        or piece_type == PieceType.PROMOTED_SILVER
    ):
        current_offsets = gold_move_offsets
    elif piece_type == PieceType.KING:
        current_offsets = king_move_offsets

    for dr, dc in current_offsets:
        nr, nc = r_from + dr, c_from + dc
        if game.is_on_board(nr, nc):
            target_piece: Optional[Piece] = game.get_piece(nr, nc)
            if target_piece is None or target_piece.color != piece.color:
                moves.append((nr, nc))

    sliding_directions: List[Tuple[int, int]] = []
    if piece_type == PieceType.LANCE:
        is_sliding = True
        sliding_directions = [(forward, 0)]
    elif piece_type == PieceType.BISHOP or piece_type == PieceType.PROMOTED_BISHOP:
        is_sliding = True
        sliding_directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        if piece_type == PieceType.PROMOTED_BISHOP:
            for dr, dc in promoted_bishop_extra_offsets:
                nr, nc = r_from + dr, c_from + dc
                if game.is_on_board(nr, nc):
                    target_piece = game.get_piece(nr, nc)
                    if target_piece is None or target_piece.color != piece.color:
                        moves.append((nr, nc))
    elif piece_type == PieceType.ROOK or piece_type == PieceType.PROMOTED_ROOK:
        is_sliding = True
        sliding_directions.extend([(-1, 0), (1, 0), (0, -1), (0, 1)])
        if piece_type == PieceType.PROMOTED_ROOK:
            for dr, dc in promoted_rook_extra_offsets:
                nr, nc = r_from + dr, c_from + dc
                if game.is_on_board(nr, nc):
                    target_piece = game.get_piece(nr, nc)
                    if target_piece is None or target_piece.color != piece.color:
                        moves.append((nr, nc))

    if is_sliding:
        for dr_slide, dc_slide in sliding_directions:
            for i in range(1, 9):  # Max 8 steps
                nr, nc = r_from + dr_slide * i, c_from + dc_slide * i
                if not game.is_on_board(nr, nc):
                    break
                target_piece = game.get_piece(nr, nc)
                if target_piece is None:
                    moves.append((nr, nc))
                else:
                    if target_piece.color != piece.color:
                        moves.append((nr, nc))
                    break

    return list(set(moves))  # Remove duplicates, type is List[Tuple[int, int]]


def check_for_nifu(game: "ShogiGame", color: Color, col: int) -> bool:
    """
    Checks for two unpromoted pawns of the same color on the given file.
    (Formerly ShogiGame.is_nifu)
    """
    for r in range(9):
        p = game.get_piece(r, col)
        if (
            p
            and p.type == PieceType.PAWN
            and p.color == color
            # and not p.is_promoted # This check is redundant if type is PAWN
        ):
            # Found one pawn, need to check for a *second* one.
            # The original logic implies if *any* pawn is found, it's nifu,
            # which is incorrect. Nifu = "two pawns".
            # This function as written in original code is actually "is_pawn_on_file"
            # For a correct nifu, one pawn must already be on the file.
            # This function is used when *dropping* a pawn. So it checks if a pawn *already exists*.
            return True  # A pawn of that color already exists on this file.
    return False


def check_if_square_is_attacked(
    game: "ShogiGame",
    r_target: int,
    c_target: int,
    attacker_color: Color,
    debug: bool = False,  # ADDED debug flag
) -> bool:
    """
    Checks if the square (r_target, c_target) is attacked by any piece of attacker_color.
    """  # Corrected string literal
    if debug:
        print(
            f"DEBUG_CHECK_SQ_ATTACKED: Checking if ({r_target},{c_target}) is attacked by {attacker_color}"
        )
        print(f"DEBUG_CHECK_SQ_ATTACKED: Game state for check: {game.to_sfen_string()}")

    for r_attacker in range(9):
        for c_attacker in range(9):
            piece = game.get_piece(r_attacker, c_attacker)
            if piece and piece.color == attacker_color:
                if debug:
                    print(
                        f"DEBUG_CHECK_SQ_ATTACKED: Checking attacker {piece.type.name} ({piece.color.name}) at ({r_attacker},{c_attacker}) against target ({r_target},{c_target})"
                    )

                potential_moves_of_attacker = generate_piece_potential_moves(
                    game, piece, r_attacker, c_attacker
                )
                if (r_target, c_target) in potential_moves_of_attacker:
                    if debug:
                        print(
                            f"DEBUG_CHECK_SQ_ATTACKED: ***ASSERTION TRIGGER*** YES, {piece.type.name} ({piece.color.name}) at ({r_attacker},{c_attacker}) attacks target ({r_target},{c_target}). Attacker potential moves: {potential_moves_of_attacker}. Game SFEN: {game.to_sfen_string()}"
                        )
                    return True
    if debug:
        print(
            f"DEBUG_CHECK_SQ_ATTACKED: NO, ({r_target},{c_target}) is NOT attacked by {attacker_color}"
        )
    return False


def check_for_uchi_fu_zume(
    game: "ShogiGame", drop_row: int, drop_col: int, color: Color
) -> bool:
    """
    Returns True if dropping a pawn at (drop_row, drop_col) by 'color'
    results in immediate, unescapable checkmate for the opponent.
    This is an illegal move by Shogi rules.

    RECURSION PREVENTION: This function calls generate_all_legal_moves() with
    is_uchi_fu_zume_check=True. This flag prevents infinite recursion by:
    1. Being passed to can_drop_specific_piece() as is_escape_check
    2. When is_escape_check=True, pawn drops skip their own uchi_fu_zume check

    This function does NOT check if the drop itself leaves 'color's king in check;
    that is handled by the main move generation logic.
    """
    # print(f"DEBUG_UCHI_FU_ZUME_DETAILED: Entered for color {color}, drop PAWN at ({drop_row}, {drop_col})")
    # print(f"DEBUG_UCHI_FU_ZUME_DETAILED: Initial game state (SFEN): {game.to_sfen_string()}")
    # print(f"DEBUG_UCHI_FU_ZUME_DETAILED: Current player in initial game: {game.current_player}")

    opp_color = Color.WHITE if color == Color.BLACK else Color.BLACK
    original_current_player = game.current_player

    # Ensure the square is empty (pre-condition)
    if game.get_piece(drop_row, drop_col) is not None:
        return False

    # Ensure the player has a pawn in hand
    if not (
        PieceType.PAWN in game.hands[color.value]
        and game.hands[color.value][PieceType.PAWN] > 0
    ):
        return False

    # Simulate the pawn drop directly on the 'game' object.
    # Use try/finally to guarantee state restoration even if an exception occurs.
    game.set_piece(drop_row, drop_col, Piece(PieceType.PAWN, color))
    game.hands[color.value][PieceType.PAWN] -= 1
    try:
        opp_king_pos = find_king(game, opp_color)

        if not opp_king_pos:
            return False

        # 1. Check if the drop delivers check to the opponent's king.
        drop_delivers_check = check_if_square_is_attacked(
            game, opp_king_pos[0], opp_king_pos[1], color
        )

        if not drop_delivers_check:
            return False

        # 2. Check if the opponent has any legal moves to escape the check.
        game.current_player = opp_color
        opponent_legal_moves = generate_all_legal_moves(
            game, is_uchi_fu_zume_check=True
        )

        # If no legal moves, the opponent is checkmated by the pawn drop (uchi_fu_zume).
        return not opponent_legal_moves
    finally:
        game.set_piece(drop_row, drop_col, None)
        game.hands[color.value][PieceType.PAWN] += 1
        game.current_player = original_current_player


def is_king_in_check_after_simulated_move(
    game: "ShogiGame", player_color: Color
) -> bool:
    """
    Checks if the king of 'player_color' is in check on the current board.
    Assumes the board reflects the state *after* a move has been made.
    (Formerly ShogiGame._king_in_check_after_move)
    """
    king_pos = find_king(game, player_color)

    if not king_pos:
        return True

    opponent_color = Color.WHITE if player_color == Color.BLACK else Color.BLACK
    in_check = check_if_square_is_attacked(
        game, king_pos[0], king_pos[1], opponent_color, debug=False
    )  # debug was from the removed flag
    return in_check


def can_promote_specific_piece(
    game: "ShogiGame", piece: Piece, r_from: int, r_to: int
) -> bool:
    """
    Checks if a piece *can* be promoted given its type and move.
    (Formerly ShogiGame.can_promote_piece)
    """
    if piece.type in [PieceType.GOLD, PieceType.KING] or piece.is_promoted:
        return False

    if piece.type not in BASE_TO_PROMOTED_TYPE:  # Not a base type that can promote
        return False

    # Must start in, end in, or cross promotion zone
    # Assumes game instance has is_in_promotion_zone(row, color) method
    if game.is_in_promotion_zone(r_from, piece.color) or game.is_in_promotion_zone(
        r_to, piece.color
    ):
        return True
    return False


def must_promote_specific_piece(
    piece: Piece, r_to: int
) -> bool:  # Removed 'game' parameter
    """Checks if a piece *must* promote when moving to r_to."""
    # Pawns and Lances must promote if they reach the last rank.
    if piece.type == PieceType.PAWN or piece.type == PieceType.LANCE:
        if (piece.color == Color.BLACK and r_to == 0) or (
            piece.color == Color.WHITE and r_to == 8
        ):
            return True

    # Knights on the last two ranks
    if piece.type == PieceType.KNIGHT:
        if (piece.color == Color.BLACK and r_to <= 1) or (
            piece.color == Color.WHITE and r_to >= 7
        ):
            return True
    return False


def can_drop_specific_piece(
    game: "ShogiGame",
    piece_type: PieceType,
    r_to: int,
    c_to: int,
    color: Color,
    is_escape_check: bool = False,  # Added for uchi_fu_zume fix
) -> bool:
    """
    Checks if a specific piece_type can be legally dropped by 'color' at (r_to, c_to).
    This function checks rules like:
    - Square must be empty.
    - Nifu (two pawns on the same file).
    - Piece cannot be dropped where it has no further moves (Pawn, Lance on last rank; Knight on last two ranks).
    - Uchi Fu Zume (dropping a pawn for an immediate checkmate that cannot be escaped).
    It does NOT check if the drop leaves the current player's king in check;
    that is handled by the calling function (e.g., generate_all_legal_moves).
    """
    # print(f"DEBUG_CAN_DROP: Checking drop of {piece_type} by {color} at ({r_to},{c_to}), is_escape_check={is_escape_check}") # Basic log
    if game.get_piece(r_to, c_to) is not None:
        # print(f"DEBUG_CAN_DROP: Square ({r_to},{c_to}) not empty. Returning False.") # Basic log
        return False  # Square must be empty

    # Determine player-specific "forward" direction and promotion zone boundaries
    last_rank = 0 if color == Color.BLACK else 8
    second_last_rank = 1 if color == Color.BLACK else 7

    if piece_type == PieceType.PAWN:
        # 1. Nifu check: Cannot drop a pawn on a file that already contains an unpromoted pawn of the same color.
        if check_for_nifu(game, color, c_to):
            return False
        # 2. Cannot drop a pawn on the last rank (it would have no moves).
        if r_to == last_rank:
            return False
        # 3. Uchi Fu Zume check: Cannot drop a pawn to give immediate checkmate if that checkmate has no escape.
        #    The check_for_uchi_fu_zume function returns True if it *is* uchi_fu_zume.
        #    This check is skipped if we are evaluating an escape move during an uchi_fu_zume check.
        if (
            not is_escape_check
        ):  # <-- MODIFIED: Only check uchi_fu_zume if not an escape check
            # print(f"DEBUG_CAN_DROP: PAWN drop, is_escape_check is False. Calling check_for_uchi_fu_zume for {color} at ({r_to},{c_to}).") # Basic log
            if check_for_uchi_fu_zume(game, r_to, c_to, color):
                # print(f"DEBUG_CAN_DROP: uchi_fu_zume check for {color} dropping PAWN at ({r_to},{c_to}) returned True. Preventing drop.")
                return False  # DO NOT COMMENT THIS OUT, IT IS A LOAD BEARING RETURN.
        # else: # Basic log
        # print(f"DEBUG_CAN_DROP: PAWN drop, is_escape_check is True. Skipping uchi_fu_zume check for {color} at ({r_to},{c_to}).") # Basic log
    elif piece_type == PieceType.LANCE:
        # Cannot drop a lance on the last rank.
        if r_to == last_rank:
            return False
    elif piece_type == PieceType.KNIGHT:
        # Cannot drop a knight on the last two ranks.
        if r_to == last_rank or r_to == second_last_rank:
            return False

    # Other pieces (Gold, Silver, Bishop, Rook, King) can be dropped on any empty square
    # provided other conditions (like not leaving king in check) are met by the caller.
    # King is not a droppable piece type, but included for completeness if logic changes.
    # For now, hands will only contain P, L, N, S, G, B, R.
    return True


def generate_all_legal_moves(
    game: "ShogiGame",
    is_uchi_fu_zume_check: bool = False,  # Restored is_uchi_fu_zume_check
) -> List[MoveTuple]:
    # Basic entry log
    # print(
    #    f"DEBUG_GALM: Entered for player {game.current_player}. SFEN: {game.to_sfen_string()}"
    # )
    # if is_uchi_fu_zume_check:
    #    print("DEBUG_GALM: Mode: is_uchi_fu_zume_check=True")  # Corrected f-string

    legal_moves: List[MoveTuple] = []
    original_player_color = game.current_player

    # I. Generate Board Moves
    for r_from in range(9):
        for c_from in range(9):
            piece = game.get_piece(r_from, c_from)
            if piece and piece.color == original_player_color:
                potential_squares = generate_piece_potential_moves(
                    game, piece, r_from, c_from
                )
                for r_to, c_to in potential_squares:
                    can_promote = can_promote_specific_piece(game, piece, r_from, r_to)
                    must_promote = must_promote_specific_piece(piece, r_to)
                    possible_promotions = [False]
                    if can_promote:
                        possible_promotions.append(True)
                    if must_promote:
                        possible_promotions = [True]

                    for promote_option in possible_promotions:
                        if must_promote and not promote_option:
                            continue
                        move_tuple = (r_from, c_from, r_to, c_to, promote_option)

                        simulation_details = game.make_move(
                            move_tuple, is_simulation=True
                        )

                        king_is_safe = not is_king_in_check_after_simulated_move(
                            game, original_player_color
                        )
                        if king_is_safe:
                            legal_moves.append(move_tuple)
                        # Ensure we only pass Dict[str, Any] to undo_move
                        if isinstance(simulation_details, dict):
                            game.undo_move(simulation_undo_details=simulation_details)
                        else:
                            game.undo_move()

    # II. Generate Drop Moves
    for piece_type_to_drop_val, count in game.hands[
        original_player_color.value
    ].items():
        if count > 0:
            piece_type_to_drop = PieceType(piece_type_to_drop_val)
            for r_to_drop in range(9):
                for c_to_drop in range(9):
                    if can_drop_specific_piece(
                        game,
                        piece_type_to_drop,
                        r_to_drop,
                        c_to_drop,
                        original_player_color,
                        is_escape_check=is_uchi_fu_zume_check,
                    ):
                        drop_move_tuple = (
                            None,
                            None,
                            r_to_drop,
                            c_to_drop,
                            piece_type_to_drop,
                        )
                        simulation_details_drop = game.make_move(
                            drop_move_tuple, is_simulation=True
                        )

                        king_is_safe_after_drop = (
                            not is_king_in_check_after_simulated_move(
                                game, original_player_color
                            )
                        )
                        if king_is_safe_after_drop:
                            legal_moves.append(drop_move_tuple)
                        # Ensure we only pass Dict[str, Any] to undo_move
                        if isinstance(simulation_details_drop, dict):
                            game.undo_move(
                                simulation_undo_details=simulation_details_drop
                            )
                        else:
                            game.undo_move()

    # Keep this final print for now to confirm the list content before returning
    # print(
    #    f"DEBUG_GALM: FINALIZING for {original_player_color}. Total legal moves: {len(legal_moves)}. Moves: {legal_moves}"
    # )
    return legal_moves


def check_for_sennichite(game: "ShogiGame") -> bool:
    """
    Returns True if the current board state has occurred four times (Sennichite).
    (Formerly ShogiGame.is_sennichite)
    Relies on game.move_history and game._board_state_hash().
    The hash in move_history is for the state *after* a move, including whose turn it was *before* switching players.
    Sennichite rule states: same game position (pieces on board, pieces in hand, and player to move)
    has appeared for the fourth time.
    """
    # The state hash to check for repetition should represent (board, hands, current_player_to_move)
    # In the original make_move:
    # 1. move pieces
    # 2. state_hash = _board_state_hash() (current_player is still P_old who made the move) -> stored in history
    # 3. current_player is switched to P_new
    # 4. if is_sennichite(): ... is called.
    # Inside is_sennichite (this function):
    # game.current_player is P_new.
    # So, state_to_check_for_repetition = game._board_state_hash() will use P_new.
    # This means we are checking if the state (board, hands, P_old_who_just_moved) has repeated.
    # The history stores (board, hands, P_old_who_just_moved). This comparison is subtle.

    # The problem description (II.8) implies the hash in history is the one to count.
    # "The state_hash stored in move_history by make_move is for the board
    # state *after* the move and includes self.current_player *before* it's switched."
    # "is_sennichite is called *after* self.current_player is switched."
    # "If current make_move appends hash for state *after* move & *before* player switch,
    # then is_sennichite is called *after* player switch. The hashes won't match."

    # The original `is_sennichite` code in the prompt:
    # final_state_hash_of_move = (
    #         game.move_history[-1].get("state_hash") if game.move_history else None
    # ) # This is (board_after_P_old_move, hands, P_old_turn)
    # ...
    # count_of_this_state = 0
    # for record in game.move_history:
    #     if record.get("state_hash") == final_state_hash_of_move:
    #         count_of_this_state += 1
    # return count_of_this_state >= 4
    # This means sennichite is declared if the state *just achieved by the previous player*
    # (which includes that player as the one whose turn it was for that hash) has appeared 4 times.
    # This is a common interpretation for how repetition is tracked.

    if not game.move_history:
        return False

    # This hash represents the board state achieved by the *previous* player's move,
    # and importantly, the game._board_state_hash() includes whose turn it *was* when that state was recorded.
    last_recorded_state_hash: Optional[Tuple] = game.move_history[-1].get("state_hash")
    if not last_recorded_state_hash:
        return False  # Should not happen if history is populated correctly

    count: int = 0
    for move_record in game.move_history:
        if move_record.get("state_hash") == last_recorded_state_hash:
            count += 1

    # If this state (board, hands, player_who_just_moved) has now occurred 4 times.
    return count >= 4
