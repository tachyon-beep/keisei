"""
shogi_core_definitions.py: Core type definitions, enums, constants,
and the Piece class for the Shogi game engine.

This module provides fundamental building blocks for a Shogi game,
including representations for piece colors, types, game termination
reasons, move structures, and the `Piece` class itself. It also defines
constants related to game notation (KIF) and observation tensors for
potential AI applications.
"""

from dataclasses import dataclass  # Added import
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

# --- Public API ---
__all__ = [
    "Color",
    "PieceType",
    "KIF_PIECE_SYMBOLS",
    "TerminationReason",
    "BoardMoveTuple",
    "DropMoveTuple",
    "MoveTuple",
    "get_unpromoted_types",
    "PROMOTED_TYPES_SET",
    "BASE_TO_PROMOTED_TYPE",
    "PROMOTED_TO_BASE_TYPE",
    "PIECE_TYPE_TO_HAND_TYPE",
    "OBS_CURR_PLAYER_UNPROMOTED_START",
    "OBS_CURR_PLAYER_PROMOTED_START",
    "OBS_OPP_PLAYER_UNPROMOTED_START",
    "OBS_OPP_PLAYER_PROMOTED_START",
    "OBS_CURR_PLAYER_HAND_START",
    "OBS_OPP_PLAYER_HAND_START",
    "OBS_CURR_PLAYER_INDICATOR",
    "OBS_MOVE_COUNT",
    "OBS_RESERVED_1",
    "OBS_RESERVED_2",
    "SYMBOL_TO_PIECE_TYPE",
    "get_piece_type_from_symbol",
    "OBS_UNPROMOTED_ORDER",
    "OBS_PROMOTED_ORDER",
    "Piece",
    "MoveApplicationResult",  # Added to __all__
]


# --- Enums and Constants ---
class Color(Enum):
    """
    Represents the player color.
    In Shogi, Black (Sente) typically moves first.
    """

    BLACK = 0  # Sente (先手), typically moves first
    WHITE = 1  # Gote (後手)

    def opponent(self) -> "Color":
        """Returns the opposing color."""
        return Color.WHITE if self == Color.BLACK else Color.BLACK


class PieceType(Enum):
    """
    Represents the type of a Shogi piece, including promoted states.
    """

    PAWN = 0
    LANCE = 1
    KNIGHT = 2
    SILVER = 3
    GOLD = 4
    BISHOP = 5
    ROOK = 6
    KING = 7
    PROMOTED_PAWN = 8  # Tokin (と)
    PROMOTED_LANCE = 9  # Promoted Lance (成香 - Narikyō)
    PROMOTED_KNIGHT = 10  # Promoted Knight (成桂 - Narikei)
    PROMOTED_SILVER = 11  # Promoted Silver (成銀 - Narigin)
    # Gold does not promote
    PROMOTED_BISHOP = 12  # Horse (竜馬 - Ryūma)
    PROMOTED_ROOK = 13  # Dragon (竜王 - Ryūō)
    # King does not promote

    def to_usi_char(self) -> str:
        """Returns the USI character for the piece type (for unpromoted pieces used in drops)."""
        # USI representation for pieces (typically uppercase)
        # P, L, N, S, G, B, R
        match self:
            case PieceType.PAWN:
                return "P"
            case PieceType.LANCE:
                return "L"
            case PieceType.KNIGHT:
                return "N"
            case PieceType.SILVER:
                return "S"
            case PieceType.GOLD:
                return "G"
            case PieceType.BISHOP:
                return "B"
            case PieceType.ROOK:
                return "R"
            # King and promoted pieces are not dropped, so they don't have a simple USI char in this context.
            # However, USI move format for board moves uses piece letters for disambiguation in CSA format,
            # but not typically in standard USI like 7g7f.
            # For drops, it's P*5e.
            # This method is primarily for getting the char for a drop.
            case _:
                raise ValueError(
                    f"Piece type {self.name} cannot be dropped or has no standard single USI drop character."
                )


# KIF Piece Symbol Mapping (Standard two-letter KIF symbols)
KIF_PIECE_SYMBOLS: Dict[PieceType, str] = {
    PieceType.PAWN: "FU",
    PieceType.LANCE: "KY",
    PieceType.KNIGHT: "KE",
    PieceType.SILVER: "GI",
    PieceType.GOLD: "KI",
    PieceType.BISHOP: "KA",
    PieceType.ROOK: "HI",
    PieceType.KING: "OU",  # Or "GY" for Gyoku (玉), "OU" (王) is common for King
    PieceType.PROMOTED_PAWN: "TO",  # Tokin
    PieceType.PROMOTED_LANCE: "NY",  # Nari-Kyo (Promoted Lance)
    PieceType.PROMOTED_KNIGHT: "NK",  # Nari-Kei (Promoted Knight)
    PieceType.PROMOTED_SILVER: "NG",  # Nari-Gin (Promoted Silver)
    PieceType.PROMOTED_BISHOP: "UM",  # Uma (Horse)
    PieceType.PROMOTED_ROOK: "RY",  # Ryu (Dragon)
}


class TerminationReason(Enum):
    """Enumerates reasons for game termination."""

    CHECKMATE = "Tsumi"  # Player is checkmated (Matches test_shogi_game_io.py)
    STALEMATE = "stalemate"  # Player has no legal moves but is not in check (Keep or update if test fails)
    REPETITION = "Sennichite"  # Position repeated (Sennichite - Matches test_shogi_game_core_logic.py)
    MAX_MOVES_EXCEEDED = "Max moves reached"  # Game exceeded maximum allowed moves (Matches test_shogi_game_core_logic.py)
    RESIGNATION = "resignation"
    TIME_FORFEIT = "time_forfeit"
    ILLEGAL_MOVE = "illegal_move"
    AGREEMENT = "agreement"
    IMPASSE = "impasse"
    NO_CONTEST = "no_contest"

    def __str__(self) -> str:
        return self.value


# --- Custom Types for Moves ---

# BoardMoveTuple: (from_row, from_col, to_row, to_col, promote_flag)
#   - from_row, from_col: 0-indexed source square coordinates.
#   - to_row, to_col: 0-indexed destination square coordinates.
#   - promote_flag: Boolean indicating if promotion occurs on this move.
BoardMoveTuple = Tuple[int, int, int, int, bool]

# DropMoveTuple: (None, None, to_row, to_col, piece_type_to_drop)
#   - from_row, from_col: Always None to distinguish from board moves.
#   - to_row, to_col: 0-indexed destination square coordinates for the drop.
#   - piece_type_to_drop: The PieceType (unpromoted) to be dropped.
DropMoveTuple = Tuple[Optional[int], Optional[int], int, int, PieceType]

# MoveTuple is a union of the two types of moves.
MoveTuple = Union[BoardMoveTuple, DropMoveTuple]


@dataclass
class MoveApplicationResult:
    """
    Represents the direct results of applying a move to the board and hands.
    This dataclass is used by shogi_move_execution.apply_move_to_board.
    """

    captured_piece_type: Optional[PieceType] = None
    was_promotion: bool = False
    # Add other direct results of applying the move if necessary,
    # e.g., specific flags for game state changes directly caused by the move mechanics
    # (not game termination, which is handled later).


def get_unpromoted_types() -> List[PieceType]:
    """
    Returns a list of all PieceType enums that represent unpromoted pieces
    capable of being held in hand. King is not included as it cannot be held.
    """
    return [
        PieceType.PAWN,
        PieceType.LANCE,
        PieceType.KNIGHT,
        PieceType.SILVER,
        PieceType.GOLD,
        PieceType.BISHOP,
        PieceType.ROOK,
    ]


PROMOTED_TYPES_SET: Set[PieceType] = {
    PieceType.PROMOTED_PAWN,
    PieceType.PROMOTED_LANCE,
    PieceType.PROMOTED_KNIGHT,
    PieceType.PROMOTED_SILVER,
    PieceType.PROMOTED_BISHOP,
    PieceType.PROMOTED_ROOK,
}
"""A set of all piece types that are in a promoted state."""

BASE_TO_PROMOTED_TYPE: Dict[PieceType, PieceType] = {
    PieceType.PAWN: PieceType.PROMOTED_PAWN,
    PieceType.LANCE: PieceType.PROMOTED_LANCE,
    PieceType.KNIGHT: PieceType.PROMOTED_KNIGHT,
    PieceType.SILVER: PieceType.PROMOTED_SILVER,
    PieceType.BISHOP: PieceType.PROMOTED_BISHOP,
    PieceType.ROOK: PieceType.PROMOTED_ROOK,
}
"""Maps base (unpromoted) piece types to their promoted counterparts."""

PROMOTED_TO_BASE_TYPE: Dict[PieceType, PieceType] = {
    v: k for k, v in BASE_TO_PROMOTED_TYPE.items()
}
"""Maps promoted piece types back to their base (unpromoted) counterparts."""


PIECE_TYPE_TO_HAND_TYPE: Dict[PieceType, PieceType] = {
    PieceType.PAWN: PieceType.PAWN,
    PieceType.LANCE: PieceType.LANCE,
    PieceType.KNIGHT: PieceType.KNIGHT,
    PieceType.SILVER: PieceType.SILVER,
    PieceType.GOLD: PieceType.GOLD,  # Gold is already its base type
    PieceType.BISHOP: PieceType.BISHOP,
    PieceType.ROOK: PieceType.ROOK,
    PieceType.PROMOTED_PAWN: PieceType.PAWN,
    PieceType.PROMOTED_LANCE: PieceType.LANCE,
    PieceType.PROMOTED_KNIGHT: PieceType.KNIGHT,
    PieceType.PROMOTED_SILVER: PieceType.SILVER,
    PieceType.PROMOTED_BISHOP: PieceType.BISHOP,
    PieceType.PROMOTED_ROOK: PieceType.ROOK,
    # Note: Kings (PieceType.KING) are not capturable in a way that adds them to hand.
    # Gold pieces (PieceType.GOLD) do not promote, so they remain Gold when captured.
}
"""
Maps a captured piece type (which could be promoted) to the
PieceType it becomes when added to a player's hand (always unpromoted).
"""

# Observation Plane Constants
# ---------------------------
# These constants define the structure of a (46, 9, 9) observation tensor,
# commonly used as input for neural networks in Shogi AI.
#
# Channel map:
#
# Board Piece Channels (28 total):
#   Channels  0-7: Current player's unpromoted pieces (P, L, N, S, G, B, R, K)
#   Channels  8-13: Current player's promoted pieces (+P, +L, +N, +S, +B, +R)
#   Channels 14-21: Opponent's unpromoted pieces (P, L, N, S, G, B, R, K)
#   Channels 22-27: Opponent's promoted pieces (+P, +L, +N, +S, +B, +R)
#
# Hand Piece Channels (14 total):
#   Channels 28-34: Current player's hand (P, L, N, S, G, B, R count planes)
#   Channels 35-41: Opponent's hand (P, L, N, S, G, B, R count planes)
#
# Meta Information Channels (4 total):
#   Channel 42: Current player indicator (1.0 if Black, 0.0 if White)
#   Channel 43: Move count (normalized or raw)
#   Channel 44: Reserved for future use (e.g., repetition count)
#   Channel 45: Reserved for future use

OBS_CURR_PLAYER_UNPROMOTED_START = 0
OBS_CURR_PLAYER_PROMOTED_START = 8
OBS_OPP_PLAYER_UNPROMOTED_START = 14
OBS_OPP_PLAYER_PROMOTED_START = 22

OBS_CURR_PLAYER_HAND_START = 28
OBS_OPP_PLAYER_HAND_START = 35

OBS_CURR_PLAYER_INDICATOR = 42
OBS_MOVE_COUNT = 43
OBS_RESERVED_1 = 44  # Potentially for repetition count or other game state
OBS_RESERVED_2 = 45  # Potentially for game phase or other features

# Total observation planes: 46 channels
#   - 8 current player unpromoted board pieces
#   - 6 current player promoted board pieces
#   - 8 opponent unpromoted board pieces
#   - 6 opponent promoted board pieces
#   - 7 current player hand pieces (types)
#   - 7 opponent hand pieces (types)
#   - 4 meta information planes


# Symbol to PieceType mapping.
# Uses uppercase symbols as canonical representation.
# E.g., "P" for Pawn, "+P" for Promoted Pawn (Tokin).
SYMBOL_TO_PIECE_TYPE: Dict[str, PieceType] = {
    "P": PieceType.PAWN,
    "L": PieceType.LANCE,
    "N": PieceType.KNIGHT,
    "S": PieceType.SILVER,
    "G": PieceType.GOLD,
    "B": PieceType.BISHOP,
    "R": PieceType.ROOK,
    "K": PieceType.KING,
    "+P": PieceType.PROMOTED_PAWN,
    "+L": PieceType.PROMOTED_LANCE,
    "+N": PieceType.PROMOTED_KNIGHT,
    "+S": PieceType.PROMOTED_SILVER,
    "+B": PieceType.PROMOTED_BISHOP,
    "+R": PieceType.PROMOTED_ROOK,
}


def get_piece_type_from_symbol(symbol: str) -> PieceType:
    """
    Converts a piece symbol string (e.g., "P", "+R", "p", "+r") to a PieceType enum.

    The function prefers canonical uppercase symbols (e.g., "P", "+R") but will
    attempt to normalize common lowercase variations (e.g., "p" to "P", "+p" to "+P").

    Args:
        symbol: The piece symbol string.

    Returns:
        The corresponding PieceType enum.

    Raises:
        ValueError: If the symbol is unknown or malformed.
    """
    # Primary check for canonical symbols (already in SYMBOL_TO_PIECE_TYPE)
    if symbol in SYMBOL_TO_PIECE_TYPE:
        return SYMBOL_TO_PIECE_TYPE[symbol]

    # Handle normalization for common lowercase variants
    # Case 1: Promoted piece like "+p"
    if len(symbol) == 2 and symbol.startswith("+") and symbol[1].islower():
        upper_symbol = "+" + symbol[1].upper()
        if upper_symbol in SYMBOL_TO_PIECE_TYPE:
            return SYMBOL_TO_PIECE_TYPE[upper_symbol]
    # Case 2: Unpromoted piece like "p"
    elif len(symbol) == 1 and symbol.islower():
        upper_symbol = symbol.upper()
        if upper_symbol in SYMBOL_TO_PIECE_TYPE:
            return SYMBOL_TO_PIECE_TYPE[upper_symbol]

    raise ValueError(f"Unknown piece symbol: {symbol}")


# Order of unpromoted pieces for observation channels (excluding King for hand)
OBS_UNPROMOTED_ORDER: List[PieceType] = [
    PieceType.PAWN,
    PieceType.LANCE,
    PieceType.KNIGHT,
    PieceType.SILVER,
    PieceType.GOLD,
    PieceType.BISHOP,
    PieceType.ROOK,
    PieceType.KING,  # King is included for on-board representation
]

# Order of promoted pieces for observation channels
OBS_PROMOTED_ORDER: List[PieceType] = [
    PieceType.PROMOTED_PAWN,
    PieceType.PROMOTED_LANCE,
    PieceType.PROMOTED_KNIGHT,
    PieceType.PROMOTED_SILVER,
    PieceType.PROMOTED_BISHOP,
    PieceType.PROMOTED_ROOK,
]

# Internal mapping for Piece.symbol() method for conciseness.
_PIECE_TYPE_TO_CHAR_SYMBOL: Dict[PieceType, str] = {
    PieceType.PAWN: "P",
    PieceType.LANCE: "L",
    PieceType.KNIGHT: "N",
    PieceType.SILVER: "S",
    PieceType.GOLD: "G",
    PieceType.BISHOP: "B",
    PieceType.ROOK: "R",
    PieceType.KING: "K",
    PieceType.PROMOTED_PAWN: "+P",
    PieceType.PROMOTED_LANCE: "+L",
    PieceType.PROMOTED_KNIGHT: "+N",
    PieceType.PROMOTED_SILVER: "+S",
    PieceType.PROMOTED_BISHOP: "+B",
    PieceType.PROMOTED_ROOK: "+R",
}


# --- Piece Class ---
class Piece:
    """
    Represents a single Shogi piece.

    Attributes:
        type (PieceType): The type of the piece (e.g., PAWN, GOLD, PROMOTED_ROOK).
        color (Color): The color of the piece (BLACK or WHITE).
        is_promoted (bool): True if the piece is in its promoted state, False otherwise.
                            This is derived from `type`.
    """

    def __init__(self, piece_type: PieceType, color: Color):
        """
        Initializes a Piece instance.

        Args:
            piece_type: The type of the piece.
            color: The color of the piece.

        Raises:
            TypeError: If `piece_type` is not a `PieceType` or
                       `color` is not a `Color`.
        """
        if not isinstance(piece_type, PieceType):
            raise TypeError("piece_type must be an instance of PieceType")
        if not isinstance(color, Color):
            raise TypeError("color must be an instance of Color")

        self.type: PieceType = piece_type
        self.color: Color = color
        # is_promoted is derived from the piece_type for consistency
        self.is_promoted: bool = piece_type in PROMOTED_TYPES_SET

    def symbol(self) -> str:
        """
        Returns a string symbol for the piece (e.g., "P", "+P", "k").
        Uppercase for Black (Sente), lowercase for White (Gote).
        Promoted pieces are prefixed with '+'.

        Returns:
            The piece symbol string.

        Raises:
            ValueError: If the piece has an unknown type.
        """
        try:
            base_symbol = _PIECE_TYPE_TO_CHAR_SYMBOL[self.type]
        except KeyError as exc:
            # This should ideally not happen if PieceType enum is comprehensive
            # and _PIECE_TYPE_TO_CHAR_SYMBOL is kept in sync.
            raise ValueError(f"Unknown piece type: {self.type}") from exc

        return base_symbol.lower() if self.color == Color.WHITE else base_symbol

    def promote(self) -> None:
        """
        Promotes the piece if it is promotable and not already promoted.
        If the piece type cannot be promoted or is already promoted,
        this method has no effect.
        Updates `self.type` and `self.is_promoted` accordingly.
        """
        if self.type in BASE_TO_PROMOTED_TYPE and not self.is_promoted:
            self.type = BASE_TO_PROMOTED_TYPE[self.type]
            self.is_promoted = True
        # No change if not promotable or already promoted

    def unpromote(self) -> None:
        """
        Unpromotes the piece if it is currently in a promoted state.
        If the piece is not promoted, this method has no effect.
        Updates `self.type` and `self.is_promoted` accordingly.
        """
        if self.is_promoted and self.type in PROMOTED_TO_BASE_TYPE:
            self.type = PROMOTED_TO_BASE_TYPE[self.type]
            self.is_promoted = False
        # No change if not promoted

    def __repr__(self) -> str:
        """
        Returns an unambiguous string representation of the Piece.
        """
        return f"Piece({self.type.name}, {self.color.name})"

    def __eq__(self, other: object) -> bool:
        """
        Checks if this Piece is equal to another object.
        Two pieces are equal if they have the same type and color.
        """
        if not isinstance(other, Piece):
            return NotImplemented
        return self.type == other.type and self.color == other.color

    def __hash__(self) -> int:
        """
        Returns a hash value for the Piece.
        Based on the piece type and color.
        """
        return hash((self.type, self.color))

    def __deepcopy__(self, memo: Dict[int, "Piece"]) -> "Piece":
        """
        Creates a deep copy of this Piece instance.
        Since Piece instances are relatively simple and their core attributes
        (type, color) define their state, creating a new instance is sufficient.

        Args:
            memo: The memoization dictionary used by `copy.deepcopy`.

        Returns:
            A new Piece instance identical to this one.
        """
        # Piece objects are simple enough that creating a new one with the same
        # attributes serves as a deep copy. `is_promoted` is derived correctly
        # by the __init__ method.
        new_piece = Piece(self.type, self.color)
        memo[id(self)] = new_piece  # Store in memo for `deepcopy` consistency
        return new_piece
