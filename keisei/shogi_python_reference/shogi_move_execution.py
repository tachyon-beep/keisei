# shogi_move_execution.py
"""
Contains functions for applying and reverting moves in the Shogi game.
These functions operate on a ShogiGame instance.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

from .shogi_core_definitions import TerminationReason  # Changed from TerminationStatus
from .shogi_core_definitions import (
    BASE_TO_PROMOTED_TYPE,
    PROMOTED_TO_BASE_TYPE,
    Color,
    MoveApplicationResult,
    MoveTuple,
    Piece,
    PieceType,
)

if TYPE_CHECKING:
    from .shogi_game import ShogiGame


def apply_move_to_board_state(
    board: List[List[Optional[Piece]]],
    hands: Dict[int, Dict[PieceType, int]],  # Changed Color to int for key type
    move: MoveTuple,
    current_player: Color,
) -> MoveApplicationResult:
    """
    Applies a validated move to the given board and hands, mutating them directly.
    Handles piece movement, captures, and promotions.

    Args:
        board: The game board (list of lists of Optional[Piece]).
        hands: The players' hands (dict mapping player color value to dict of PieceType to count).
        move: The MoveTuple to apply.
        current_player: The Color of the player making the move.

    Returns:
        MoveApplicationResult indicating what piece was captured (if any)
        and whether a promotion occurred.
    """
    captured_piece_type: Optional[PieceType] = None
    was_promotion: bool = False

    # Unpack the move tuple. The meaning of the 5th element depends on the move type.
    from_r_raw, from_c_raw, to_r_raw, to_c_raw, promote_or_drop_info = move
    # Ensure to_r and to_c are integers, as they are common to both move types.
    if not isinstance(to_r_raw, int) or not isinstance(to_c_raw, int):
        raise ValueError("to_row and to_col must be integers in MoveTuple")
    to_r: int = to_r_raw
    to_c: int = to_c_raw

    if from_r_raw is None:  # Drop move (from_r_raw and from_c_raw will be None)
        if not isinstance(promote_or_drop_info, PieceType):
            raise ValueError(
                "For drop moves, the 5th element of MoveTuple must be a PieceType."
            )
        piece_to_drop_type: PieceType = promote_or_drop_info

        # Assuming the move is validated, the piece must be in hand and its count > 0.
        # Directly place the piece on the board and decrement hand count.
        board[to_r][to_c] = Piece(piece_to_drop_type, current_player)
        if (
            piece_to_drop_type not in hands[current_player.value]
        ):  # Should not happen if validated
            raise ValueError(
                f"Attempting to drop {piece_to_drop_type} which is not in hand for {current_player}"
            )
        hands[current_player.value][piece_to_drop_type] -= 1
        # Do not delete the key from hands; let the count be 0, consistent with hand initialization.

        # was_promotion remains False for drops
        # captured_piece_type remains None for drops
    else:  # Board move (from_r_raw and from_c_raw will be int)
        if not isinstance(from_r_raw, int) or not isinstance(from_c_raw, int):
            raise ValueError(
                "from_row and from_col must be integers for board moves in MoveTuple"
            )
        from_r: int = from_r_raw
        from_c: int = from_c_raw

        if not isinstance(promote_or_drop_info, bool):
            raise ValueError(
                "For board moves, the 5th element of MoveTuple must be a boolean (promote_flag)."
            )
        promote: bool = promote_or_drop_info

        moving_piece = board[from_r][from_c]
        if not moving_piece:
            raise ValueError(
                f"No piece at source square ({from_r},{from_c}) for board move."
            )
        if moving_piece.color != current_player:
            raise ValueError(
                f"Piece at ({from_r},{from_c}) belongs to {moving_piece.color}, "
                f"but current player is {current_player}."
            )

        target_square_piece = board[to_r][to_c]
        if target_square_piece:
            if target_square_piece.color == current_player:
                raise ValueError(
                    f"Cannot capture own piece at ({to_r},{to_c}). "
                    f"Moving piece from ({from_r},{from_c}) for player {current_player}."
                )
            # Capture piece: add its base type to the current player's hand
            captured_piece_type = PROMOTED_TO_BASE_TYPE.get(
                target_square_piece.type, target_square_piece.type
            )
            hands[current_player.value][captured_piece_type] = (
                hands[current_player.value].get(captured_piece_type, 0) + 1
            )

        # Move the piece
        board[to_r][
            to_c
        ] = moving_piece  # Piece instance carries its color and current type
        board[from_r][from_c] = None

        if promote:
            if moving_piece.type not in BASE_TO_PROMOTED_TYPE:
                raise ValueError(
                    f"Piece type {moving_piece.type} cannot be promoted. "
                    f"Move: {move}, Piece: {moving_piece}"
                )
            promoted_type = BASE_TO_PROMOTED_TYPE[moving_piece.type]
            # Update the piece on the board to its promoted version
            # Create a new Piece instance for the promoted piece, maintaining its color
            board[to_r][to_c] = Piece(promoted_type, moving_piece.color)
            was_promotion = True
        # If not promoting, the piece on board[to_r][to_c] (which is moving_piece)
        # retains its original type (which could be already promoted if it moved from a promotion zone)

    return MoveApplicationResult(
        captured_piece_type=captured_piece_type, was_promotion=was_promotion
    )


def apply_move_to_game(game: "ShogiGame", is_simulation: bool = False) -> None:
    """Updates the game state *after* a move has been applied to the board/hands.

    This includes switching the current player, incrementing the move count,
    and (temporarily) checking for game termination conditions.

    Args:
        game: The ShogiGame instance.
        is_simulation: If True, this is part of a simulation (e.g., for checkmate detection)
                       and certain side effects like history updates might be skipped
                       or handled differently by the caller.
    """
    if not is_simulation:
        game.move_count += 1

    game.current_player = game.current_player.opponent()

    # Termination checks (checkmate, stalemate, max_moves) are now removed from here.
    # They will be handled by ShogiGame._check_and_update_termination_status
    # after this function completes and after history/sennichite is processed in ShogiGame.make_move.

    # The import of is_in_check and generate_all_legal_moves, and the associated logic
    # for termination checking, are removed from this function.
    # if not is_simulation:
    #     from .shogi_rules_logic import (
    #         is_in_check,
    #         generate_all_legal_moves
    #     )
    #     player_to_check = game.current_player
    #     all_opponent_moves = generate_all_legal_moves(game)
    #     if not all_opponent_moves:
    #         if is_in_check(game, player_to_check):
    #             game.game_over = True
    #             # Winner was set using player_who_just_moved, which is now out of scope here.
    #             # This assignment is now handled in _check_and_update_termination_status.
    #             # game.winner = player_who_just_moved
    #             game.termination_reason = TerminationReason.CHECKMATE.value
    #         else:
    #             game.game_over = True
    #             game.winner = None
    #             game.termination_reason = TerminationReason.STALEMATE.value # Ensure enum used
    #     elif game.move_count >= game.max_moves_per_game:
    #         game.game_over = True
    #         game.winner = None
    #         game.termination_reason = TerminationReason.MAX_MOVES_EXCEEDED.value


def revert_last_applied_move(
    game: "ShogiGame",
    original_board_state: List[List[Optional[Piece]]],
    original_hands_state: Dict[int, Dict[PieceType, int]],
    original_current_player: Color,
    original_move_count: int,
    # move_made: MoveTuple, # Unused
    # move_application_result: MoveApplicationResult # Unused
) -> None:
    """Reverts the last move applied to the game state, board, and hands.

    This is primarily used for simulations (e.g., checkmate detection) to undo a trial move.

    Args:
        game: The ShogiGame instance to revert.
        original_board_state: A deep copy of the board *before* the move was made.
        original_hands_state: A deep copy of the hands *before* the move was made.
        original_current_player: The player whose turn it was *before* the move.
        original_move_count: The move count *before* the move.
        # move_made: The MoveTuple that was applied. (Unused due to direct state restoration)
        # move_application_result: The MoveApplicationResult from applying the move. (Unused)
    """
    # Restore board and hands from the copies
    game.board = [row[:] for row in original_board_state]
    game.hands = {k: v.copy() for k, v in original_hands_state.items()}

    # Restore game state variables
    game.current_player = original_current_player
    game.move_count = original_move_count

    # Reset game termination state
    game.game_over = False
    game.winner = None
    game.termination_reason = None  # Use None, not TerminationReason.ACTIVE.value
