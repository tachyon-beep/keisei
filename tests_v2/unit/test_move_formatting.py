"""
Unit tests for move formatting and USI/SFEN notation round-trip.

Tests cover:
- Board move USI formatting (e.g., "7g7f")
- Drop move USI formatting (e.g., "P*5e")
- Promotion move USI formatting (e.g., "2c2b+")
- USI round-trip: shogi_move_to_usi -> usi_to_shogi_move
- SFEN round-trip: encode_move_to_sfen_string -> sfen_to_move_tuple
- Edge cases at board boundaries (corners, edges)
"""

import pytest

from keisei.shogi.shogi_core_definitions import PieceType
from keisei.shogi.shogi_game_io import encode_move_to_sfen_string, sfen_to_move_tuple
from keisei.utils.utils import PolicyOutputMapper


@pytest.fixture(scope="module")
def mapper():
    """Module-scoped PolicyOutputMapper (expensive to build, safe to share read-only)."""
    return PolicyOutputMapper()


class TestBoardMoveUSIFormatting:
    """Verify that standard board moves produce correct USI strings."""

    def test_simple_pawn_advance(self, mapper):
        """A pawn advancing from 7g to 7f should produce USI '7g7f'."""
        # Row 6, col 2 -> file = 9-2 = 7, rank = chr(ord('a')+6) = 'g'
        # Row 5, col 2 -> file = 7, rank = 'f'
        move = (6, 2, 5, 2, False)
        usi = mapper.shogi_move_to_usi(move)
        assert usi == "7g7f"

    def test_rook_lateral_move(self, mapper):
        """Rook moving from 2h to 6h should produce USI '8h4h'."""
        # 2h: row=7, col=1 -> file=9-1=8, rank='h'
        # 6h: row=7, col=5 -> file=9-5=4, rank='h'
        move = (7, 1, 7, 5, False)
        usi = mapper.shogi_move_to_usi(move)
        assert usi == "8h4h"

    def test_king_diagonal_move(self, mapper):
        """King stepping diagonally from 5i to 4h."""
        # 5i: row=8, col=4 -> file=5, rank='i'
        # 4h: row=7, col=5 -> file=4, rank='h'
        move = (8, 4, 7, 5, False)
        usi = mapper.shogi_move_to_usi(move)
        assert usi == "5i4h"


class TestDropMoveUSIFormatting:
    """Verify that drop moves produce correct USI strings (e.g., 'P*5e')."""

    @pytest.mark.parametrize(
        "piece_type, to_row, to_col, expected_usi",
        [
            (PieceType.PAWN, 4, 4, "P*5e"),
            (PieceType.LANCE, 0, 0, "L*9a"),
            (PieceType.KNIGHT, 2, 7, "N*2c"),
            (PieceType.SILVER, 8, 8, "S*1i"),
            (PieceType.GOLD, 3, 3, "G*6d"),
            (PieceType.BISHOP, 1, 6, "B*3b"),
            (PieceType.ROOK, 5, 5, "R*4f"),
        ],
        ids=["pawn", "lance", "knight", "silver", "gold", "bishop", "rook"],
    )
    def test_drop_all_piece_types(self, mapper, piece_type, to_row, to_col, expected_usi):
        """Each droppable piece type should produce the correct USI drop string."""
        move = (None, None, to_row, to_col, piece_type)
        usi = mapper.shogi_move_to_usi(move)
        assert usi == expected_usi


class TestPromotionMoveUSIFormatting:
    """Verify that promotion moves append '+' to the USI string."""

    def test_pawn_promotes_entering_zone(self, mapper):
        """Pawn moving from 4d (row=3) to 3c (row=2) with promotion -> '6d6c+'."""
        # 4d: row=3, col=3 -> file=9-3=6, rank='d'
        # 3c: row=2, col=3 -> file=6, rank='c'
        move = (3, 3, 2, 3, True)
        usi = mapper.shogi_move_to_usi(move)
        assert usi == "6d6c+"
        assert usi.endswith("+")

    def test_non_promotion_lacks_plus(self, mapper):
        """Same squares without promotion should NOT have '+' suffix."""
        move = (3, 3, 2, 3, False)
        usi = mapper.shogi_move_to_usi(move)
        assert usi == "6d6c"
        assert not usi.endswith("+")

    def test_bishop_promotion(self, mapper):
        """Bishop promoting on a long diagonal move."""
        # 8h (row=7,col=1) to 2b (row=1,col=7) with promotion
        move = (7, 1, 1, 7, True)
        usi = mapper.shogi_move_to_usi(move)
        assert usi == "8h2b+"


class TestUSIRoundTrip:
    """Verify that move -> USI -> move is an identity transformation."""

    def test_board_move_round_trip(self, mapper):
        """A board move should survive a USI round-trip unchanged."""
        original = (6, 2, 5, 2, False)
        usi = mapper.shogi_move_to_usi(original)
        recovered = mapper.usi_to_shogi_move(usi)
        assert recovered == original

    def test_promotion_move_round_trip(self, mapper):
        """A promotion board move should survive a USI round-trip unchanged."""
        original = (7, 1, 1, 7, True)
        usi = mapper.shogi_move_to_usi(original)
        recovered = mapper.usi_to_shogi_move(usi)
        assert recovered == original

    def test_drop_move_round_trip(self, mapper):
        """A drop move should survive a USI round-trip unchanged."""
        original = (None, None, 4, 4, PieceType.PAWN)
        usi = mapper.shogi_move_to_usi(original)
        recovered = mapper.usi_to_shogi_move(usi)
        assert recovered == original

    def test_all_droppable_pieces_round_trip(self, mapper):
        """Every droppable piece type at center square should round-trip correctly."""
        for pt in [
            PieceType.PAWN,
            PieceType.LANCE,
            PieceType.KNIGHT,
            PieceType.SILVER,
            PieceType.GOLD,
            PieceType.BISHOP,
            PieceType.ROOK,
        ]:
            original = (None, None, 4, 4, pt)
            usi = mapper.shogi_move_to_usi(original)
            recovered = mapper.usi_to_shogi_move(usi)
            assert recovered == original, f"Round-trip failed for {pt.name}: {usi}"


class TestSFENRoundTrip:
    """Verify that encode_move_to_sfen_string and sfen_to_move_tuple are inverses."""

    def test_board_move_sfen_round_trip(self):
        """A simple board move should survive SFEN encode -> decode."""
        original = (6, 2, 5, 2, False)
        sfen = encode_move_to_sfen_string(original)
        assert sfen == "7g7f"
        recovered = sfen_to_move_tuple(sfen)
        assert recovered == original

    def test_promotion_sfen_round_trip(self):
        """A promotion move should survive SFEN encode -> decode."""
        original = (7, 1, 1, 7, True)
        sfen = encode_move_to_sfen_string(original)
        assert sfen == "8h2b+"
        recovered = sfen_to_move_tuple(sfen)
        assert recovered == original

    def test_drop_move_sfen_round_trip(self):
        """A drop move should survive SFEN encode -> decode."""
        original = (None, None, 4, 4, PieceType.PAWN)
        sfen = encode_move_to_sfen_string(original)
        assert sfen == "P*5e"
        recovered = sfen_to_move_tuple(sfen)
        assert recovered == original


class TestBoardBoundaryEdgeCases:
    """Verify formatting for moves touching all four board corners and edges."""

    @pytest.mark.parametrize(
        "from_rc, to_rc, expected_from_sq, expected_to_sq",
        [
            # Corner to adjacent: 9a (0,0) -> 8a (0,1)
            ((0, 0), (0, 1), "9a", "8a"),
            # Corner: 1a (0,8) -> 1b (1,8)
            ((0, 8), (1, 8), "1a", "1b"),
            # Corner: 9i (8,0) -> 9h (7,0)
            ((8, 0), (7, 0), "9i", "9h"),
            # Corner: 1i (8,8) -> 2h (7,7)
            ((8, 8), (7, 7), "1i", "2h"),
            # Edge middle: 5a (0,4) -> 5b (1,4)
            ((0, 4), (1, 4), "5a", "5b"),
            # Edge middle: 9e (4,0) -> 8e (4,1)
            ((4, 0), (4, 1), "9e", "8e"),
        ],
        ids=[
            "top_left_corner",
            "top_right_corner",
            "bottom_left_corner",
            "bottom_right_corner",
            "top_edge_middle",
            "left_edge_middle",
        ],
    )
    def test_boundary_moves_produce_correct_usi(
        self, mapper, from_rc, to_rc, expected_from_sq, expected_to_sq
    ):
        """Moves at board boundaries should format to the expected USI squares."""
        move = (from_rc[0], from_rc[1], to_rc[0], to_rc[1], False)
        usi = mapper.shogi_move_to_usi(move)
        assert usi == f"{expected_from_sq}{expected_to_sq}"

    def test_corner_drop(self, mapper):
        """Dropping a lance at 9a (top-left corner, row=0, col=0)."""
        move = (None, None, 0, 0, PieceType.LANCE)
        usi = mapper.shogi_move_to_usi(move)
        assert usi == "L*9a"

    def test_opposite_corner_drop(self, mapper):
        """Dropping a silver at 1i (bottom-right corner, row=8, col=8)."""
        move = (None, None, 8, 8, PieceType.SILVER)
        usi = mapper.shogi_move_to_usi(move)
        assert usi == "S*1i"
