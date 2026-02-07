# shogi_game_io.py

import datetime  # For KIF Date header
import os
import re  # Import the re module
import sys
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .shogi_core_definitions import (  # Observation plane constants
    BASE_TO_PROMOTED_TYPE,
    KIF_PIECE_SYMBOLS,
    OBS_CURR_PLAYER_HAND_START,
    OBS_CURR_PLAYER_INDICATOR,
    OBS_CURR_PLAYER_PROMOTED_START,
    OBS_CURR_PLAYER_UNPROMOTED_START,
    OBS_MOVE_COUNT,
    OBS_OPP_PLAYER_HAND_START,
    OBS_OPP_PLAYER_PROMOTED_START,
    OBS_OPP_PLAYER_UNPROMOTED_START,
    OBS_PROMOTED_ORDER,
    OBS_UNPROMOTED_ORDER,
    PROMOTED_TYPES_SET,
    SYMBOL_TO_PIECE_TYPE,
    Color,
    MoveTuple,
    Piece,
    PieceType,
    TerminationReason,
    get_unpromoted_types,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if TYPE_CHECKING:
    from .shogi_game import ShogiGame  # For type hinting the 'game' parameter

# --- SFEN-related Constants ---
SFEN_BOARD_CHARS: Dict[PieceType, str] = {
    PieceType.PAWN: "P",
    PieceType.LANCE: "L",
    PieceType.KNIGHT: "N",
    PieceType.SILVER: "S",
    PieceType.GOLD: "G",
    PieceType.BISHOP: "B",
    PieceType.ROOK: "R",
    PieceType.KING: "K",
    PieceType.PROMOTED_PAWN: "P",  # Note: SFEN uses the base piece char for promoted pieces on board, promotion is indicated by '+'
    PieceType.PROMOTED_LANCE: "L",
    PieceType.PROMOTED_KNIGHT: "N",
    PieceType.PROMOTED_SILVER: "S",
    PieceType.PROMOTED_BISHOP: "B",
    PieceType.PROMOTED_ROOK: "R",
}

SFEN_HAND_PIECE_CANONICAL_ORDER: List[PieceType] = [
    PieceType.ROOK,
    PieceType.BISHOP,
    PieceType.GOLD,
    PieceType.SILVER,
    PieceType.KNIGHT,
    PieceType.LANCE,
    PieceType.PAWN,
]

# --- SFEN Helper Functions ---


def _sfen_sq(r: int, c: int) -> str:
    """Converts 0-indexed (row, col) to SFEN square string (e.g., (0,0) -> "9a")."""
    if not (0 <= r <= 8 and 0 <= c <= 8):
        raise ValueError(f"Invalid Shogi coordinate for SFEN: row {r}, col {c}")
    file = str(9 - c)
    rank = chr(ord("a") + r)
    return file + rank


def _get_sfen_board_char(piece: Piece) -> str:
    """Helper to get the SFEN character for a piece on the board."""
    if not isinstance(piece, Piece):
        raise TypeError(f"Expected a Piece object, got {type(piece)}")

    # Get the base character (e.g., 'P' for PAWN and PROMOTED_PAWN)
    base_char = SFEN_BOARD_CHARS.get(piece.type)
    if base_char is None:
        # This handles cases like KING which don't have a separate entry for a promoted type
        # but are in SFEN_BOARD_CHARS. It mainly catches truly unknown types.
        raise ValueError(f"Unknown piece type for SFEN board character: {piece.type}")

    sfen_char = ""
    if piece.type in PROMOTED_TYPES_SET:
        sfen_char += "+"
    sfen_char += base_char

    if piece.color == Color.WHITE:
        return sfen_char.lower()
    return sfen_char


def _get_sfen_drop_char(piece_type: PieceType) -> str:
    """Helper to get the uppercase SFEN character for a droppable piece type."""
    # Using a mapping for clarity and directness
    sfen_char_map: Dict[PieceType, str] = {
        PieceType.PAWN: "P",
        PieceType.LANCE: "L",
        PieceType.KNIGHT: "N",
        PieceType.SILVER: "S",
        PieceType.GOLD: "G",
        PieceType.BISHOP: "B",
        PieceType.ROOK: "R",
    }
    char = sfen_char_map.get(piece_type)
    if char is None:
        raise ValueError(
            f"PieceType {piece_type.name if hasattr(piece_type, 'name') else piece_type} "
            f"is not a standard droppable piece for SFEN notation or is invalid."
        )
    return char


def _parse_sfen_board_piece(
    sfen_char_on_board: str, is_promoted_sfen_token: bool
) -> Tuple[PieceType, Color]:
    """
    Parses an SFEN board piece character (e.g., 'P', 'l', 'R') and promotion status
    into (PieceType, Color).
    `sfen_char_on_board` is the actual piece letter (e.g., 'P' from "+P", or 'K').
    `is_promoted_sfen_token` is True if a '+' preceded this character in SFEN.
    Returns (PieceType, Color)
    """
    color = Color.BLACK if "A" <= sfen_char_on_board <= "Z" else Color.WHITE

    base_char_upper = sfen_char_on_board.upper()
    base_piece_type = SYMBOL_TO_PIECE_TYPE.get(base_char_upper)

    if base_piece_type is None:
        raise ValueError(
            f"Invalid SFEN piece character for board: {sfen_char_on_board}"
        )

    if is_promoted_sfen_token:
        if base_piece_type in BASE_TO_PROMOTED_TYPE:
            final_piece_type = BASE_TO_PROMOTED_TYPE[base_piece_type]
        elif (
            base_piece_type in PROMOTED_TYPES_SET
        ):  # Already a promoted type, e.g. if SYMBOL_TO_PIECE_TYPE mapped +P directly
            final_piece_type = base_piece_type
        else:
            # '+' was applied to a non-promotable piece like King or Gold
            raise ValueError(
                f"Invalid promotion: SFEN token '+' applied to non-promotable piece type {base_piece_type.name} (from char '{sfen_char_on_board}')"
            )
    else:  # Not a promoted token
        # If the base_piece_type itself is a promoted type (e.g. if SFEN_BOARD_CHARS had +P: P and no plain P:P)
        # this would be an issue, but SYMBOL_TO_PIECE_TYPE should map to base types.
        if base_piece_type in PROMOTED_TYPES_SET:
            raise ValueError(
                f"Invalid SFEN: Character '{sfen_char_on_board}' (mapped to {base_piece_type.name}) implies promotion, but no '+' prefix found."
            )
        final_piece_type = base_piece_type

    return final_piece_type, color


# --- SFEN String Parsing Functions ---


def parse_sfen_string_components(sfen_str: str) -> Tuple[str, str, str, str]:
    """
    Parses an SFEN string into its components.
    Returns: (board_sfen, turn_sfen, hands_sfen, move_number_sfen)
    """
    sfen_pattern = re.compile(r"^\s*([^ ]+)\s+([bw])\s+([^ ]+)\s+(\d+)\s*$")
    match = sfen_pattern.match(sfen_str.strip())
    if not match:
        raise ValueError(f"Invalid SFEN string structure: '{sfen_str}'")

    groups = match.groups()
    if len(groups) != 4:
        raise ValueError(f"Invalid SFEN string structure: '{sfen_str}'")

    return groups[0], groups[1], groups[2], groups[3]


def populate_board_from_sfen_segment(
    board_array: List[List[Optional[Piece]]], board_sfen_segment: str
) -> None:
    """
    Populates a board array from an SFEN board segment.
    Modifies board_array in place.
    """
    # Parse board
    rows = board_sfen_segment.split("/")
    if len(rows) != 9:
        raise ValueError("Expected 9 ranks")

    for r, row_str in enumerate(rows):
        c = 0
        promoted_flag_active = False
        while c < 9 and row_str:
            char_sfen = row_str[0]
            row_str = row_str[1:]
            if char_sfen == "+":
                if promoted_flag_active:
                    raise ValueError(
                        "Invalid piece character sequence starting with '+'"
                    )
                promoted_flag_active = True
                continue
            elif char_sfen.isdigit():
                if promoted_flag_active:
                    raise ValueError(
                        f"Invalid SFEN: Digit ('{char_sfen}') cannot immediately follow a promotion token ('+')."
                    )
                if char_sfen == "0":
                    raise ValueError("Invalid SFEN piece character for board: 0")
                empty_squares = int(char_sfen)
                if not 1 <= empty_squares <= 9 - c:
                    raise ValueError(
                        f"Row {r+1} ('{rows[r]}') describes {c+empty_squares} columns, expected 9"
                    )
                c += empty_squares
            else:
                piece_char_upper = char_sfen.upper()
                base_piece_type = SYMBOL_TO_PIECE_TYPE.get(piece_char_upper)
                if base_piece_type is None:
                    raise ValueError(
                        f"Invalid SFEN piece character for board: {char_sfen}"
                    )
                piece_color = Color.BLACK if char_sfen.isupper() else Color.WHITE
                final_piece_type = base_piece_type
                if promoted_flag_active:
                    if base_piece_type in BASE_TO_PROMOTED_TYPE:
                        final_piece_type = BASE_TO_PROMOTED_TYPE[base_piece_type]
                    elif base_piece_type in PROMOTED_TYPES_SET:
                        raise ValueError(
                            f"Invalid promotion: SFEN token '+' applied to non-promotable piece type {base_piece_type.name}"
                        )
                    else:
                        raise ValueError(
                            f"Invalid promotion: SFEN token '+' applied to non-promotable piece type {base_piece_type.name}"
                        )
                elif not promoted_flag_active and base_piece_type in PROMOTED_TYPES_SET:
                    raise ValueError(
                        f"Invalid SFEN piece character for board: {char_sfen}"
                    )
                board_array[r][c] = Piece(final_piece_type, piece_color)
                c += 1
                promoted_flag_active = False
        if c != 9:
            raise ValueError(
                f"Row {r+1} ('{rows[r]}') describes {c} columns, expected 9"
            )


def populate_hands_from_sfen_segment(
    hands_dict: Dict[int, Dict[PieceType, int]], hands_sfen_segment: str
) -> None:
    """
    Populates a hands dictionary from an SFEN hands segment.
    Modifies hands_dict in place.
    """
    if hands_sfen_segment != "-":
        hand_segment_pattern = re.compile(r"(\d*)([PLNSGBRplnsgbr])")
        pos = 0
        parsing_white_hand_pieces = False
        while pos < len(hands_sfen_segment):
            match_hand = hand_segment_pattern.match(hands_sfen_segment, pos)
            if not match_hand:
                if hands_sfen_segment[pos:].startswith("K") or hands_sfen_segment[
                    pos:
                ].startswith("k"):
                    raise ValueError(
                        "Invalid piece character 'K' or non-droppable piece type in SFEN hands"
                    )
                raise ValueError("Invalid character sequence in SFEN hands")
            count_str, piece_char = match_hand.groups()
            count = int(count_str) if count_str else 1

            is_current_piece_white = piece_char.islower()
            is_current_piece_black = piece_char.isupper()

            if is_current_piece_white:
                parsing_white_hand_pieces = True
            elif is_current_piece_black and parsing_white_hand_pieces:
                raise ValueError(
                    "Invalid SFEN hands: Black's pieces must precede White's pieces."
                )

            try:
                piece_type_in_hand = SYMBOL_TO_PIECE_TYPE[piece_char.upper()]
                hand_color = Color.BLACK if is_current_piece_black else Color.WHITE
            except KeyError as e:
                raise ValueError("Invalid character sequence in SFEN hands") from e
            if piece_type_in_hand == PieceType.KING:
                raise ValueError(
                    "Invalid piece character 'K' or non-droppable piece type in SFEN hands"
                )
            if piece_type_in_hand in PROMOTED_TYPES_SET:
                raise ValueError("Invalid character sequence in SFEN hands")
            current_hand_for_color = hands_dict[hand_color.value]
            current_hand_for_color[piece_type_in_hand] = (
                current_hand_for_color.get(piece_type_in_hand, 0) + count
            )
            pos = match_hand.end()


# --- SFEN String Generation Functions ---


def convert_game_to_sfen_string(game: "ShogiGame") -> str:
    """
    Serializes the current game state to an SFEN string.
    Format: <board> <turn> <hands> <move_number>
    Example: lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1
    """
    sfen_ranks = []
    # Board: ranks 1-9 (board[0] to board[8]), files 1-9 (col[8] to col[0])
    for r in range(9):  # board row 0 (SFEN rank 1) to board row 8 (SFEN rank 9)
        empty_squares_count = 0
        sfen_rank_str = ""
        # Iterate from file 9 (board column 0) down to file 1 (board column 8) for SFEN
        # Our internal board is board[row][col] where col 0 is file 9 (leftmost from Black's view)
        # So we iterate columns from 0 to 8 for our board, which corresponds to SFEN files 9 to 1.
        for c in range(9):  # Iterate through columns 0 to 8 (SFEN files 9 to 1)
            piece = game.board[r][c]
            if piece:
                if empty_squares_count > 0:
                    sfen_rank_str += str(empty_squares_count)
                    empty_squares_count = 0
                sfen_rank_str += _get_sfen_board_char(piece)
            else:
                empty_squares_count += 1

        if empty_squares_count > 0:  # Append any trailing empty count for the rank
            sfen_rank_str += str(empty_squares_count)
        sfen_ranks.append(sfen_rank_str)
    board_sfen = "/".join(sfen_ranks)

    # Player turn
    turn_sfen = "b" if game.current_player == Color.BLACK else "w"

    # Hands
    hand_sfen_parts = []
    has_black_pieces = False
    # Black's hand (uppercase)
    for piece_type in SFEN_HAND_PIECE_CANONICAL_ORDER:
        count = game.hands[Color.BLACK.value].get(piece_type, 0)
        if count > 0:
            has_black_pieces = True
            sfen_char = SFEN_BOARD_CHARS[
                piece_type
            ]  # Should be uppercase by convention from SFEN_BOARD_CHARS
            if count > 1:
                hand_sfen_parts.append(str(count))
            hand_sfen_parts.append(sfen_char)

    has_white_pieces = False
    # White's hand (lowercase)
    for piece_type in SFEN_HAND_PIECE_CANONICAL_ORDER:
        count = game.hands[Color.WHITE.value].get(piece_type, 0)
        if count > 0:
            has_white_pieces = True
            sfen_char_upper = SFEN_BOARD_CHARS[piece_type]
            sfen_char = sfen_char_upper.lower()  # Ensure lowercase for White
            if count > 1:
                hand_sfen_parts.append(str(count))
            hand_sfen_parts.append(sfen_char)

    hands_sfen = (
        "".join(hand_sfen_parts) if (has_black_pieces or has_white_pieces) else "-"
    )

    # Move number (1-indexed)
    # game.move_count is 0 for the first move to be made, so SFEN move number is move_count + 1
    move_num_sfen = str(game.move_count + 1)

    return f"{board_sfen} {turn_sfen} {hands_sfen} {move_num_sfen}"


def encode_move_to_sfen_string(move_tuple: MoveTuple) -> str:
    """
    Encodes a move in SFEN (Shogi Forsyth-Edwards Notation) format.
    Board move: (from_r, from_c, to_r, to_c, promote_bool) -> e.g., "7g7f" or "2b3a+"
    Drop move: (None, None, to_r, to_c, piece_type) -> e.g., "P*5e"
    """
    if (
        len(move_tuple) == 5
        and isinstance(move_tuple[0], int)
        and isinstance(move_tuple[1], int)
        and isinstance(move_tuple[2], int)
        and isinstance(move_tuple[3], int)
        and isinstance(move_tuple[4], bool)
    ):
        from_r, from_c, to_r, to_c, promote = (
            move_tuple[0],
            move_tuple[1],
            move_tuple[2],
            move_tuple[3],
            move_tuple[4],
        )
        from_sq_str = _sfen_sq(from_r, from_c)
        to_sq_str = _sfen_sq(to_r, to_c)
        promo_char = "+" if promote else ""
        return f"{from_sq_str}{to_sq_str}{promo_char}"
    elif (
        len(move_tuple) == 5
        and move_tuple[0] is None
        and move_tuple[1] is None
        and isinstance(move_tuple[2], int)
        and isinstance(move_tuple[3], int)
        and isinstance(move_tuple[4], PieceType)
    ):
        _, _, to_r, to_c, piece_to_drop = (
            move_tuple[0],
            move_tuple[1],
            move_tuple[2],
            move_tuple[3],
            move_tuple[4],
        )
        piece_char = _get_sfen_drop_char(piece_to_drop)
        to_sq_str = _sfen_sq(to_r, to_c)
        return f"{piece_char}*{to_sq_str}"
    else:
        element_types = [type(el).__name__ for el in move_tuple]
        element_values = [str(el) for el in move_tuple]
        raise ValueError(
            f"Invalid MoveTuple format for SFEN conversion: {move_tuple}. "
            f"Types: {element_types}. Values: {element_values}."
        )


def generate_neural_network_observation(game: "ShogiGame") -> np.ndarray:
    """
    Returns the current board state as a (Channels, 9, 9) NumPy array for RL input.
    Channels:
    Player planes (current player POV):
     0-7: Unpromoted pieces (P, L, N, S, G, B, R, K)
     8-13: Promoted pieces (+P, +L, +N, +S, +B, +R)
    Opponent planes:
     14-21: Unpromoted pieces
     22-27: Promoted pieces
    Hand planes:
     28-34: Current player's hand (P, L, N, S, G, B, R) - 7 types
     35-41: Opponent's hand (P, L, N, S, G, B, R) - 7 types
    Meta planes:
     42: Current player (all 1.0 if Black, all 0.0 if White playing as Black's opponent view)
     43: Move count (normalized)
     44: Reserved for potential future features like repetition count (currently all zeros)
     45: Reserved for potential future features like game phase indicators (currently all zeros)
    Total: 46 channels (14 player board + 14 opponent board + 7 player hand + 7 opponent hand + 4 meta).
    """
    obs = np.zeros((46, 9, 9), dtype=np.float32)
    is_black_perspective = game.current_player == Color.BLACK

    # Map PieceType to its index in OBS_UNPROMOTED_ORDER or OBS_PROMOTED_ORDER
    unpromoted_map: Dict[PieceType, int] = {
        pt: i for i, pt in enumerate(OBS_UNPROMOTED_ORDER)
    }
    promoted_map: Dict[PieceType, int] = {
        pt: i for i, pt in enumerate(OBS_PROMOTED_ORDER)
    }

    for r in range(9):
        for c in range(9):
            # For white's perspective, we need to mirror the board coordinates
            flipped_r = r if is_black_perspective else 8 - r
            flipped_c = c if is_black_perspective else 8 - c

            p: Optional[Piece] = game.board[r][c]
            if p is None:
                continue

            is_current_player_piece: bool = p.color == game.current_player
            channel_offset: int = -1

            if p.is_promoted:
                if p.type in promoted_map:
                    # Offset for promoted planes
                    promoted_block_offset = (
                        OBS_CURR_PLAYER_PROMOTED_START
                        if is_current_player_piece
                        else OBS_OPP_PLAYER_PROMOTED_START
                    )
                    channel_offset = promoted_block_offset + promoted_map[p.type]
            else:  # Unpromoted or non-promotable (King, Gold)
                if p.type in unpromoted_map:
                    # Offset for unpromoted planes
                    unpromoted_block_offset = (
                        OBS_CURR_PLAYER_UNPROMOTED_START
                        if is_current_player_piece
                        else OBS_OPP_PLAYER_UNPROMOTED_START
                    )
                    channel_offset = unpromoted_block_offset + unpromoted_map[p.type]

            if channel_offset != -1:
                # Use the flipped coordinates for setting the observation plane
                obs[channel_offset, flipped_r, flipped_c] = 1.0

    # Pieces in hand (7 channels per player: P,L,N,S,G,B,R)
    hand_piece_order: List[PieceType] = (
        get_unpromoted_types()
    )  # Use the imported function

    # Per-type maximum hand counts for normalization
    hand_max_counts: Dict[PieceType, float] = {
        PieceType.PAWN: 18.0,
        PieceType.LANCE: 4.0,
        PieceType.KNIGHT: 4.0,
        PieceType.SILVER: 4.0,
        PieceType.GOLD: 4.0,
        PieceType.BISHOP: 2.0,
        PieceType.ROOK: 2.0,
    }

    # Current player's hand
    for i, piece_type_enum_player in enumerate(hand_piece_order):
        player_hand_count: int = game.hands[game.current_player.value].get(
            piece_type_enum_player, 0
        )
        if player_hand_count > 0:
            player_ch: int = OBS_CURR_PLAYER_HAND_START + i
            obs[player_ch, :, :] = (
                player_hand_count / hand_max_counts[piece_type_enum_player]
            )

    # Opponent's hand
    opponent_color_val: int = (
        Color.WHITE.value if game.current_player == Color.BLACK else Color.BLACK.value
    )
    for i, piece_type_enum_opponent in enumerate(hand_piece_order):
        opponent_hand_count: int = game.hands[opponent_color_val].get(
            piece_type_enum_opponent, 0
        )
        if opponent_hand_count > 0:
            opponent_ch: int = OBS_OPP_PLAYER_HAND_START + i
            obs[opponent_ch, :, :] = (
                opponent_hand_count / hand_max_counts[piece_type_enum_opponent]
            )

    # Current player indicator plane
    obs[OBS_CURR_PLAYER_INDICATOR, :, :] = (
        1.0 if game.current_player == Color.BLACK else 0.0
    )

    # Move count plane (normalized)
    max_moves = float(game.max_moves_per_game)
    obs[OBS_MOVE_COUNT, :, :] = game.move_count / max_moves if max_moves > 0 else 0.0

    # Planes for OBS_RESERVED_1 and OBS_RESERVED_2 remain zeros.
    return obs


def convert_game_to_text_representation(game: "ShogiGame") -> str:
    """
    Returns a string representation of the Shogi game board and state.
    """
    lines = []
    for r_idx, row_data in enumerate(game.board):
        line_str: str = f"{9-r_idx} "  # Shogi board rank numbers (9 down to 1)
        line_pieces: List[str] = []
        for (
            p_cell
        ) in (
            row_data
        ):  # Renamed p to p_cell to avoid conflict with p in outer scope if any
            if p_cell:
                symbol: str = p_cell.symbol()
                if len(symbol) == 1:  # e.g., P
                    line_pieces.append(f" {symbol} ")  # Results in " P "
                else:  # e.g., +P
                    line_pieces.append(f"{symbol} ")  # Results in "+P "
            else:
                line_pieces.append(" . ")  # Consistent 3-char width
        lines.append(line_str + "".join(line_pieces))
    # Add file numbers at the bottom with consistent single-space formatting
    lines.append(
        "a b c d e f g h i"  # Single space between column labels as expected by tests
    )

    # Add player turn and move info first, then hands info
    lines.append(f"Turn: {game.current_player.name}, Move: {game.move_count+1}")

    black_hand_dict: Dict[str, int] = {
        pt.name: count
        for pt, count in game.hands[Color.BLACK.value].items()
        if count > 0
    }
    lines.append(f"Black's hand: {black_hand_dict}")

    white_hand_dict: Dict[str, int] = {
        pt.name: count
        for pt, count in game.hands[Color.WHITE.value].items()
        if count > 0
    }
    lines.append(f"White's hand: {white_hand_dict}")
    return "\n".join(lines)


def game_to_kif(
    game: "ShogiGame",
    filename: Optional[str] = None,
    sente_player_name: str = "Sente",
    gote_player_name: str = "Gote",
) -> Optional[str]:
    """
    Converts a game to a KIF file or string representation.
    Uses standard KIF piece notation (+FU, -FU, etc.) and includes more headers.

    Args:
        game: The ShogiGame to convert
        filename: If provided, the KIF will be written to this file
        sente_player_name: Name of the black/sente player
        gote_player_name: Name of the white/gote player

    Returns:
        If filename is None, returns a string representation of the KIF.
        Otherwise returns None after writing to the file.
    """

    # Helper function to create KIF content
    def create_kif_content():
        lines = []
        # --- KIF Headers ---
        lines.append("#KIF version=2.0 encoding=UTF-8")
        lines.append("*Event: Casual Game")
        lines.append("*Site: Local Machine")
        lines.append(f"*Date: {datetime.date.today().strftime('%Y/%m/%d')}")
        lines.append(f"*Player Sente: {sente_player_name}")
        lines.append(f"*Player Gote: {gote_player_name}")
        lines.append("*Handicap: HIRATE")

        # Standard HIRATE starting position
        lines.append("P1-KY-KE-GI-KI-OU-KI-GI-KE-KY")
        lines.append("P2 * -HI * * * * * -KA * ")
        lines.append("P3-FU-FU-FU-FU-FU-FU-FU-FU-FU")
        lines.append("P4 * * * * * * * * * ")
        lines.append("P5 * * * * * * * * * ")
        lines.append("P6 * * * * * * * * * ")
        lines.append("P7+FU+FU+FU+FU+FU+FU+FU+FU+FU")
        lines.append("P8 * +KA * * * * * +HI * ")
        lines.append("P9+KY+KE+GI+KI+OU+KI+GI+KE+KY")

        # --- Initial Hands (KIF format: P+00FU00KY... for Sente, P-00FU00KY... for Gote) ---
        # This assumes starting with empty hands for a standard game from initial board setup.
        sente_hand_str: str = "P+"
        gote_hand_str: str = "P-"
        hand_order_for_kif: List[PieceType] = [
            PieceType.ROOK,
            PieceType.BISHOP,
            PieceType.GOLD,
            PieceType.SILVER,
            PieceType.KNIGHT,
            PieceType.LANCE,
            PieceType.PAWN,
        ]  # Common KIF hand order

        initial_sente_hand: Dict[PieceType, int] = game.hands[Color.BLACK.value]
        initial_gote_hand: Dict[PieceType, int] = game.hands[Color.WHITE.value]

        for pt in hand_order_for_kif:
            sente_hand_str += (
                f"{initial_sente_hand.get(pt, 0):02d}{KIF_PIECE_SYMBOLS.get(pt, '??')}"
            )
            gote_hand_str += (
                f"{initial_gote_hand.get(pt, 0):02d}{KIF_PIECE_SYMBOLS.get(pt, '??')}"
            )
        lines.append(f"{sente_hand_str}")
        lines.append(f"{gote_hand_str}")

        # --- Player to move first ---
        lines.append(
            f"{'+' if game.current_player == Color.BLACK else '-'}"
        )  # + for Sente, - for Gote

        lines.append("moves")  # Start of the moves section

        # --- Moves ---
        for i, move_entry in enumerate(game.move_history):
            move_obj: Optional[MoveTuple] = move_entry.get(
                "move"
            )  # Your internal move object/tuple
            if not move_obj:
                continue
            # Defensive: ensure all indices are not None
            if (
                move_obj[0] is None
                or move_obj[1] is None
                or move_obj[2] is None
                or move_obj[3] is None
            ):
                continue  # Skip malformed move
            usi_move_str: str = (
                f"{move_obj[0]+1}{chr(move_obj[1]+ord('a'))}{move_obj[2]+1}{chr(move_obj[3]+ord('a'))}"
            )
            if move_obj[4]:  # Promote flag
                usi_move_str += "+"
            lines.append(f"{i+1} {usi_move_str}")

        # --- Game Termination ---
        if game.game_over:
            termination_map: Dict[str, str] = {
                "Tsumi": "詰み",
                "Toryo": "投了",
                "Sennichite": "千日手",
                "Stalemate": "持将棋",
                "Max moves reached": "持将棋",  # Or "最大手数" or similar
                # Add other mappings for values set in game.termination_reason
            }
            reason_str: Optional[str] = game.termination_reason
            kif_termination_reason_display: str

            if reason_str is None:
                kif_termination_reason_display = ""  # No reason string if None
            else:
                # Now reason_str is str, so termination_map.get(str, str) is used
                kif_termination_reason_display = termination_map.get(
                    reason_str, reason_str
                )

            if kif_termination_reason_display:
                lines.append(kif_termination_reason_display)

            # Append the RESULT line based on winner
            if game.winner == Color.BLACK:
                lines.append("RESULT:SENTE_WIN")
            elif game.winner == Color.WHITE:
                lines.append("RESULT:GOTE_WIN")
            elif game.winner is None:  # Draw conditions
                # More specific draw reasons could be mapped to KIF draw results
                if game.termination_reason in [
                    TerminationReason.REPETITION.value,
                    TerminationReason.IMPASSE.value,
                    TerminationReason.MAX_MOVES_EXCEEDED.value,
                ]:
                    lines.append("RESULT:DRAW")  # Or HIKIWAKE etc.

        lines.append("*EOF")  # Standard KIF end marker
        return "\n".join(lines)

    # Generate KIF content
    kif_content = create_kif_content()

    # Either write to file or return as string
    if filename:
        with open(filename, "w", encoding="utf-8") as kif_file:
            kif_file.write(kif_content)
        return None
    else:
        return kif_content


# --- SFEN Move Parsing ---


def _parse_sfen_square(sfen_sq: str) -> Tuple[int, int]:
    """
    Converts an SFEN square string (e.g., "7g", "5e") to 0-indexed (row, col).
    File (1-9) maps to column (8-0). Rank (a-i) maps to row (0-8).
    Example: "9a" -> (0,0), "1i" -> (8,8)
    """
    if not (
        len(sfen_sq) == 2 and "1" <= sfen_sq[0] <= "9" and "a" <= sfen_sq[1] <= "i"
    ):
        raise ValueError(f"Invalid SFEN square format: {sfen_sq}")

    file_char: str = sfen_sq[0]
    rank_char: str = sfen_sq[1]

    col: int = 9 - int(file_char)
    row: int = ord(rank_char) - ord("a")

    return row, col


def _get_piece_type_from_sfen_char(char: str) -> PieceType:
    """
    Converts an SFEN piece character (e.g., 'P', 'L', 'B') to a PieceType enum.
    SFEN uses uppercase single letters for standard pieces.
    """
    if char in SYMBOL_TO_PIECE_TYPE:
        pt = SYMBOL_TO_PIECE_TYPE[char]
        # Ensure it's a basic piece type that can be dropped
        if (
            pt in get_unpromoted_types() and pt != PieceType.KING
        ):  # King cannot be dropped
            return pt
    raise ValueError(f"Invalid SFEN piece character for drop: {char}")


def sfen_to_move_tuple(sfen_move_str: str) -> MoveTuple:
    """
    Parses an SFEN move string (e.g., "7g7f", "P*5e", "2b3a+")
    and converts it into an internal MoveTuple using regular expressions.
    """
    sfen_move_str = sfen_move_str.strip()

    # Regex for drop moves: e.g., "P*5e"
    # Group 1: Piece character (P, L, N, S, G, B, R)
    # Group 2: Square (e.g., 5e)
    drop_move_pattern = re.compile(r"^([PLNSGBR])\*([1-9][a-i])$")

    # Regex for board moves: e.g., "7g7f", "2b3a+"
    # Group 1: From square (e.g., 7g)
    # Group 2: To square (e.g., 7f)
    # Group 3: Optional promotion character (+)
    board_move_pattern = re.compile(r"^([1-9][a-i])([1-9][a-i])(\+)?$")

    drop_match = drop_move_pattern.match(sfen_move_str)
    if drop_match:
        piece_char: str = drop_match.group(1)
        sfen_sq_to: str = drop_match.group(2)

        try:
            piece_to_drop: PieceType = _get_piece_type_from_sfen_char(piece_char)
            r_to, c_to = _parse_sfen_square(sfen_sq_to)
            return (None, None, r_to, c_to, piece_to_drop)
        except ValueError as e:
            raise ValueError(
                f"Error parsing SFEN drop move '{sfen_move_str}': {e}"
            ) from e

    board_match = board_move_pattern.match(sfen_move_str)
    if board_match:
        sfen_sq_from_str: str = board_match.group(1)
        sfen_sq_to_str: str = board_match.group(2)
        promote_flag: bool = board_match.group(3) is not None

        try:
            r_from, c_from = _parse_sfen_square(sfen_sq_from_str)
            r_to, c_to = _parse_sfen_square(sfen_sq_to_str)
            return (r_from, c_from, r_to, c_to, promote_flag)
        except ValueError as e:
            raise ValueError(
                f"Error parsing SFEN board move '{sfen_move_str}': {e}"
            ) from e

    raise ValueError(f"Invalid SFEN move format: {sfen_move_str}")


# TODO: Consider adding kif_to_game and sfen_to_game functions if needed.
# These would involve more complex parsing of full game states or move sequences.
