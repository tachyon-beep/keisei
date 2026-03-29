"""Convert SFEN (Shogi Forsyth-Edwards Notation) strings to board_state dicts.

The output matches the shape produced by ``state_snapshot.extract_board_state()``,
making it directly consumable by ``render_board()`` in the Streamlit dashboard.
"""

from typing import Any, Dict, List, Optional

# SFEN letter → (piece type name, color)
# Uppercase = black (sente), lowercase = white (gote)
_SFEN_PIECE_MAP: Dict[str, tuple] = {
    "K": ("king", "black"),
    "R": ("rook", "black"),
    "B": ("bishop", "black"),
    "G": ("gold", "black"),
    "S": ("silver", "black"),
    "N": ("knight", "black"),
    "L": ("lance", "black"),
    "P": ("pawn", "black"),
    "k": ("king", "white"),
    "r": ("rook", "white"),
    "b": ("bishop", "white"),
    "g": ("gold", "white"),
    "s": ("silver", "white"),
    "n": ("knight", "white"),
    "l": ("lance", "white"),
    "p": ("pawn", "white"),
}

# SFEN hand letter → hand piece key name (always lowercase base type)
_SFEN_HAND_MAP: Dict[str, str] = {
    "P": "pawn",
    "L": "lance",
    "N": "knight",
    "S": "silver",
    "G": "gold",
    "B": "bishop",
    "R": "rook",
}

# Promoted type names (matching PieceType.name.lower() output)
_PROMOTED_PREFIX = "promoted_"


def sfen_to_board_state(sfen: str) -> Dict[str, Any]:
    """Parse an SFEN string into a board_state dict.

    Args:
        sfen: Standard SFEN string, e.g.
            ``"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"``

    Returns:
        Dict matching the shape of ``extract_board_state()``::

            {
                "board": [[Optional[{"type", "color", "promoted"}], ...], ...],
                "current_player": "black" | "white",
                "move_count": int,
                "game_over": False,
                "winner": None,
                "black_hand": {"pawn": 2, ...},
                "white_hand": {"knight": 1, ...},
            }

    Raises:
        ValueError: If the SFEN string is malformed.
    """
    parts = sfen.strip().split()
    if len(parts) < 3:
        raise ValueError(
            f"SFEN must have at least 3 fields (board, turn, hand), got {len(parts)}: {sfen!r}"
        )

    board_str = parts[0]
    turn = parts[1]
    hand_str = parts[2]
    move_count = int(parts[3]) if len(parts) >= 4 else 1

    board = _parse_board(board_str)
    current_player = _parse_turn(turn)
    black_hand, white_hand = _parse_hand(hand_str)

    return {
        "board": board,
        "current_player": current_player,
        "move_count": move_count,
        "game_over": False,
        "winner": None,
        "black_hand": black_hand,
        "white_hand": white_hand,
    }


def _parse_board(board_str: str) -> List[List[Optional[Dict[str, Any]]]]:
    """Parse the board portion of an SFEN string into a 9x9 grid."""
    rows = board_str.split("/")
    if len(rows) != 9:
        raise ValueError(
            f"SFEN board must have 9 rows separated by '/', got {len(rows)}"
        )

    board: List[List[Optional[Dict[str, Any]]]] = []
    for rank_idx, row_str in enumerate(rows):
        row: List[Optional[Dict[str, Any]]] = []
        promoted = False
        for ch in row_str:
            if ch == "+":
                promoted = True
                continue
            if ch.isdigit():
                if promoted:
                    raise ValueError(
                        f"'+' followed by digit '{ch}' in row {rank_idx + 1}"
                    )
                row.extend([None] * int(ch))
                continue
            if ch not in _SFEN_PIECE_MAP:
                raise ValueError(
                    f"Unknown piece character '{ch}' in row {rank_idx + 1}"
                )
            base_type, color = _SFEN_PIECE_MAP[ch]
            if promoted:
                piece_type = _PROMOTED_PREFIX + base_type
                row.append({"type": piece_type, "color": color, "promoted": True})
                promoted = False
            else:
                row.append({"type": base_type, "color": color, "promoted": False})

        if len(row) != 9:
            raise ValueError(
                f"SFEN row {rank_idx + 1} has {len(row)} squares, expected 9"
            )
        board.append(row)

    return board


def _parse_turn(turn: str) -> str:
    """Parse the turn field into 'black' or 'white'."""
    if turn == "b":
        return "black"
    elif turn == "w":
        return "white"
    else:
        raise ValueError(f"SFEN turn must be 'b' or 'w', got {turn!r}")


def _parse_hand(hand_str: str) -> tuple:
    """Parse the hand field into (black_hand, white_hand) dicts.

    Returns:
        Tuple of (black_hand, white_hand) where each is {piece_name: count}.
    """
    black_hand: Dict[str, int] = {}
    white_hand: Dict[str, int] = {}

    if hand_str == "-":
        return black_hand, white_hand

    count_str = ""
    for ch in hand_str:
        if ch.isdigit():
            count_str += ch
            continue
        count = int(count_str) if count_str else 1
        count_str = ""
        upper = ch.upper()
        if upper not in _SFEN_HAND_MAP:
            raise ValueError(f"Unknown hand piece character '{ch}'")
        piece_name = _SFEN_HAND_MAP[upper]
        if ch.isupper():
            black_hand[piece_name] = black_hand.get(piece_name, 0) + count
        else:
            white_hand[piece_name] = white_hand.get(piece_name, 0) + count

    return black_hand, white_hand
