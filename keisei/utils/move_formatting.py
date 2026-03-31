"""
move_formatting.py: Contains utilities for formatting Shogi moves.
"""

from keisei.shogi_python_reference.shogi_core_definitions import PieceType


def format_move_with_description(selected_shogi_move, policy_output_mapper, game=None):
    """
    Formats a shogi move with USI notation and English description.

    Args:
        selected_shogi_move: The MoveTuple (either BoardMoveTuple or DropMoveTuple)
        policy_output_mapper: PolicyOutputMapper instance for USI conversion
        game: Optional ShogiGame instance for getting piece information

    Returns:
        str: Formatted string like "7g7f (pawn move to 7f)" or "P*5e (pawn drop to 5e)"
    """
    if selected_shogi_move is None:
        return "None"

    try:
        # Get USI notation
        usi_notation = policy_output_mapper.shogi_move_to_usi(selected_shogi_move)

        # Determine if it's a drop or board move and create description
        if len(selected_shogi_move) == 5 and selected_shogi_move[0] is None:
            # Drop move: (None, None, to_r, to_c, piece_type)
            _, _, to_r, to_c, piece_type = selected_shogi_move
            piece_name = _get_piece_name(piece_type, False)
            to_square = _coords_to_square_name(to_r, to_c)
            description = f"{piece_name} drop to {to_square}"
        else:
            # Board move: (from_r, from_c, to_r, to_c, promote_flag)
            from_r, from_c, to_r, to_c, promote_flag = selected_shogi_move
            from_square = _coords_to_square_name(from_r, from_c)
            to_square = _coords_to_square_name(to_r, to_c)

            # Try to get piece information from game if available
            piece_name = "piece"
            if game is not None:
                try:
                    piece = game.get_piece(from_r, from_c)
                    if piece is not None:
                        piece_name = _get_piece_name(piece.type, promote_flag)
                except (AttributeError, KeyError, TypeError):
                    pass  # Fall back to generic "piece"
            else:
                # Without game context, assume it's a piece that can promote if promote_flag is True
                if promote_flag:
                    piece_name = "piece promoting"

            description = f"{piece_name} moving from {from_square} to {to_square}"

        return f"{usi_notation} - {description}."

    except Exception as e:  # pylint: disable=broad-except
        # Fallback to string representation if formatting fails
        return f"{str(selected_shogi_move)} (format error: {e})"


def format_move_with_description_enhanced(
    selected_shogi_move, policy_output_mapper, piece_info=None
):
    """
    Enhanced move formatting that takes piece info as parameter for better demo logging.

    Args:
        selected_shogi_move: The MoveTuple (either BoardMoveTuple or DropMoveTuple)
        policy_output_mapper: PolicyOutputMapper instance for USI conversion
        piece_info: Piece object from game.get_piece() call made before the move

    Returns:
        str: Formatted string like "7g7f - Fuhyō (Pawn) moving from 7g to 7f."
    """
    if selected_shogi_move is None:
        return "None"

    try:
        # Get USI notation
        usi_notation = policy_output_mapper.shogi_move_to_usi(selected_shogi_move)

        # Determine if it's a drop or board move and create description
        if len(selected_shogi_move) == 5 and selected_shogi_move[0] is None:
            # Drop move: (None, None, to_r, to_c, piece_type)
            _, _, to_r, to_c, piece_type = selected_shogi_move
            piece_name = _get_piece_name(piece_type, False)
            to_square = _coords_to_square_name(to_r, to_c)
            description = f"{piece_name} drop to {to_square}"
        else:
            # Board move: (from_r, from_c, to_r, to_c, promote_flag)
            from_r, from_c, to_r, to_c, promote_flag = selected_shogi_move
            from_square = _coords_to_square_name(from_r, from_c)
            to_square = _coords_to_square_name(to_r, to_c)

            # Use the piece info passed as parameter if available
            piece_name = "piece"
            if piece_info is not None:
                try:
                    piece_name = _get_piece_name(piece_info.type, promote_flag)
                except (AttributeError, KeyError, TypeError):
                    piece_name = "piece"  # Fall back to generic "piece"

            description = f"{piece_name} moving from {from_square} to {to_square}"

        return f"{usi_notation} - {description}."

    except Exception as e:  # pylint: disable=broad-except
        # Fallback to string representation if formatting fails
        return f"{str(selected_shogi_move)} (format error: {e})"


def _get_piece_name(piece_type, is_promoting=False):
    """Convert PieceType enum to Japanese name with English translation."""
    piece_names = {
        PieceType.PAWN: "Fuhyō (Pawn)",
        PieceType.LANCE: "Kyōsha (Lance)",
        PieceType.KNIGHT: "Keima (Knight)",
        PieceType.SILVER: "Ginsho (Silver General)",
        PieceType.GOLD: "Kinshō (Gold General)",
        PieceType.BISHOP: "Kakugyō (Bishop)",
        PieceType.ROOK: "Hisha (Rook)",
        PieceType.KING: "Ōshō (King)",
        PieceType.PROMOTED_PAWN: "Tokin (Promoted Pawn)",
        PieceType.PROMOTED_LANCE: "Narikyo (Promoted Lance)",
        PieceType.PROMOTED_KNIGHT: "Narikei (Promoted Knight)",
        PieceType.PROMOTED_SILVER: "Narigin (Promoted Silver)",
        PieceType.PROMOTED_BISHOP: "Ryūma (Dragon Horse)",
        PieceType.PROMOTED_ROOK: "Ryūō (Dragon King)",
    }

    # If promoting during this move, show the transformation
    if is_promoting:
        base_names = {
            PieceType.PAWN: "Fuhyō (Pawn) → Tokin (Promoted Pawn)",
            PieceType.LANCE: "Kyōsha (Lance) → Narikyo (Promoted Lance)",
            PieceType.KNIGHT: "Keima (Knight) → Narikei (Promoted Knight)",
            PieceType.SILVER: "Ginsho (Silver General) → Narigin (Promoted Silver)",
            PieceType.BISHOP: "Kakugyō (Bishop) → Ryūma (Dragon Horse)",
            PieceType.ROOK: "Hisha (Rook) → Ryūō (Dragon King)",
        }
        return base_names.get(piece_type, piece_names.get(piece_type, str(piece_type)))

    return piece_names.get(piece_type, str(piece_type))


def _coords_to_square_name(row, col):
    """Convert 0-indexed coordinates to square name like '7f'."""
    file = str(
        9 - col
    )  # Convert column to file (9-col because shogi files go 9-1 from left to right)
    rank = chr(ord("a") + row)  # Convert row to rank (a-i)
    return f"{file}{rank}"
