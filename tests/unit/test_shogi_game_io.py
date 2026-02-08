"""
Comprehensive tests for keisei/shogi/shogi_game_io.py

Tests cover:
- SFEN coordinate helpers (_sfen_sq, _parse_sfen_square)
- SFEN round-trip serialization (to_sfen_string / from_sfen)
- SFEN parsing edge cases (parse_sfen_string_components, populate_hands_from_sfen_segment)
- Observation generation (generate_neural_network_observation via ShogiGame.get_observation)
- KIF export (game_to_kif)
- Text representation (convert_game_to_text_representation via ShogiGame.to_string)
- SFEN move encoding/decoding (encode_move_to_sfen_string, sfen_to_move_tuple)
"""

import numpy as np
import pytest

from keisei.shogi.shogi_core_definitions import Color, MoveTuple, Piece, PieceType
from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_game_io import (
    SFEN_HAND_PIECE_CANONICAL_ORDER,
    _get_sfen_board_char,
    _parse_sfen_square,
    _sfen_sq,
    convert_game_to_sfen_string,
    convert_game_to_text_representation,
    encode_move_to_sfen_string,
    game_to_kif,
    generate_neural_network_observation,
    parse_sfen_string_components,
    populate_board_from_sfen_segment,
    populate_hands_from_sfen_segment,
    sfen_to_move_tuple,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def new_game() -> ShogiGame:
    """A fresh ShogiGame in the standard initial position."""
    return ShogiGame()


# ===========================================================================
# 1. SFEN coordinate helpers (~6 tests)
# ===========================================================================


class TestSfenCoordinateHelpers:
    """Tests for _sfen_sq and _parse_sfen_square."""

    def test_sfen_sq_top_left_corner(self):
        """(0,0) is file 9, rank a -> '9a'."""
        assert _sfen_sq(0, 0) == "9a"

    def test_sfen_sq_bottom_right_corner(self):
        """(8,8) is file 1, rank i -> '1i'."""
        assert _sfen_sq(8, 8) == "1i"

    def test_sfen_sq_center(self):
        """(4,4) is file 5, rank e -> '5e'."""
        assert _sfen_sq(4, 4) == "5e"

    def test_sfen_sq_top_right_corner(self):
        """(0,8) is file 1, rank a -> '1a'."""
        assert _sfen_sq(0, 8) == "1a"

    def test_sfen_sq_bottom_left_corner(self):
        """(8,0) is file 9, rank i -> '9i'."""
        assert _sfen_sq(8, 0) == "9i"

    def test_sfen_sq_invalid_row_raises(self):
        with pytest.raises(ValueError, match="Invalid Shogi coordinate"):
            _sfen_sq(9, 0)

    def test_sfen_sq_invalid_col_raises(self):
        with pytest.raises(ValueError, match="Invalid Shogi coordinate"):
            _sfen_sq(0, -1)

    def test_parse_sfen_square_9a(self):
        assert _parse_sfen_square("9a") == (0, 0)

    def test_parse_sfen_square_1i(self):
        assert _parse_sfen_square("1i") == (8, 8)

    def test_parse_sfen_square_5e(self):
        assert _parse_sfen_square("5e") == (4, 4)

    def test_parse_sfen_square_roundtrip_all_corners(self):
        """Every corner and center coordinate round-trips correctly."""
        for r, c, expected_sfen in [
            (0, 0, "9a"),
            (0, 8, "1a"),
            (8, 0, "9i"),
            (8, 8, "1i"),
            (4, 4, "5e"),
        ]:
            sfen_str = _sfen_sq(r, c)
            assert sfen_str == expected_sfen
            assert _parse_sfen_square(sfen_str) == (r, c)

    def test_parse_sfen_square_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid SFEN square format"):
            _parse_sfen_square("z1")

    def test_parse_sfen_square_too_long_raises(self):
        with pytest.raises(ValueError, match="Invalid SFEN square format"):
            _parse_sfen_square("9a1")


# ===========================================================================
# 2. SFEN round-trip (~8 tests)
# ===========================================================================

INITIAL_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


class TestSfenRoundTrip:
    """Tests for to_sfen_string / from_sfen round-trip serialisation."""

    def test_initial_position_sfen(self, new_game: ShogiGame):
        """The default initial position should produce the standard SFEN string."""
        assert new_game.to_sfen_string() == INITIAL_SFEN

    def test_roundtrip_initial_position(self, new_game: ShogiGame):
        """to_sfen -> from_sfen -> to_sfen should be identical."""
        sfen = new_game.to_sfen_string()
        game2 = ShogiGame.from_sfen(sfen)
        assert game2.to_sfen_string() == sfen

    def test_roundtrip_white_to_move(self):
        """SFEN with 'w' turn indicator round-trips correctly."""
        sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 2"
        game = ShogiGame.from_sfen(sfen)
        assert game.current_player == Color.WHITE
        assert game.move_count == 1
        assert game.to_sfen_string() == sfen

    def test_roundtrip_with_pieces_in_hand(self):
        """SFEN with pieces in hand round-trips."""
        sfen = "lnsgkgsnl/1r5b1/pppp1pppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b P 1"
        game = ShogiGame.from_sfen(sfen)
        assert game.hands[Color.BLACK.value][PieceType.PAWN] == 1
        assert game.to_sfen_string() == sfen

    def test_roundtrip_with_multiple_hand_pieces(self):
        """SFEN with multiple hand pieces and counts round-trips."""
        sfen = "lnsgkgsnl/1r5b1/pppp1pppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b 2Pp 1"
        game = ShogiGame.from_sfen(sfen)
        assert game.hands[Color.BLACK.value][PieceType.PAWN] == 2
        assert game.hands[Color.WHITE.value][PieceType.PAWN] == 1
        assert game.to_sfen_string() == sfen

    def test_roundtrip_with_promoted_piece_on_board(self):
        """SFEN with a promoted piece (+P) on the board round-trips."""
        # Place a promoted pawn (tokin) at rank 3, file 5 (row=2, col=4)
        sfen = "lnsgkgsnl/1r5b1/pppp+Ppppp/9/9/9/PPPP1PPPP/1B5R1/LNSGKGSNL w - 2"
        game = ShogiGame.from_sfen(sfen)
        # The promoted pawn should be at row=2, col=4
        piece = game.board[2][4]
        assert piece is not None
        assert piece.type == PieceType.PROMOTED_PAWN
        assert piece.color == Color.BLACK
        assert game.to_sfen_string() == sfen

    def test_roundtrip_with_promoted_bishop_on_board(self):
        """SFEN with a promoted bishop (+B, horse) round-trips."""
        sfen = "lnsgkgsnl/1r5+B1/ppppppppp/9/9/9/PPPPPPPPP/7R1/LNSGKGSNL w - 2"
        game = ShogiGame.from_sfen(sfen)
        # The promoted bishop at row=1, col=7
        piece = game.board[1][7]
        assert piece is not None
        assert piece.type == PieceType.PROMOTED_BISHOP
        assert piece.color == Color.BLACK
        assert game.to_sfen_string() == sfen

    def test_from_sfen_move_number_five(self):
        """SFEN with move number 5 means 4 moves completed."""
        sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 5"
        game = ShogiGame.from_sfen(sfen)
        assert game.move_count == 4


# ===========================================================================
# 3. SFEN parsing edge cases (~4 tests)
# ===========================================================================


class TestSfenParsing:
    """Tests for parse_sfen_string_components and hand parsing."""

    def test_parse_sfen_string_components_initial(self):
        board, turn, hands, move_num = parse_sfen_string_components(INITIAL_SFEN)
        assert board == "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL"
        assert turn == "b"
        assert hands == "-"
        assert move_num == "1"

    def test_parse_sfen_string_components_with_hand(self):
        sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b 2Pp 3"
        board, turn, hands, move_num = parse_sfen_string_components(sfen)
        assert hands == "2Pp"
        assert move_num == "3"

    def test_parse_sfen_invalid_structure_raises(self):
        with pytest.raises(ValueError, match="Invalid SFEN string"):
            parse_sfen_string_components("not a valid sfen")

    def test_parse_sfen_wrong_turn_indicator_raises(self):
        """Turn indicator must be 'b' or 'w'."""
        with pytest.raises(ValueError, match="Invalid SFEN string"):
            parse_sfen_string_components(
                "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL x - 1"
            )

    def test_populate_hands_empty_dash(self):
        """The '-' hand segment produces zero pieces."""
        from keisei.shogi.shogi_core_definitions import get_unpromoted_types

        hands = {
            Color.BLACK.value: {pt: 0 for pt in get_unpromoted_types()},
            Color.WHITE.value: {pt: 0 for pt in get_unpromoted_types()},
        }
        populate_hands_from_sfen_segment(hands, "-")
        for color_val in (Color.BLACK.value, Color.WHITE.value):
            for pt in get_unpromoted_types():
                assert hands[color_val][pt] == 0

    def test_populate_hands_with_count(self):
        """The '2P' hand segment gives Black 2 pawns."""
        from keisei.shogi.shogi_core_definitions import get_unpromoted_types

        hands = {
            Color.BLACK.value: {pt: 0 for pt in get_unpromoted_types()},
            Color.WHITE.value: {pt: 0 for pt in get_unpromoted_types()},
        }
        populate_hands_from_sfen_segment(hands, "2P")
        assert hands[Color.BLACK.value][PieceType.PAWN] == 2

    def test_populate_hands_mixed_colors(self):
        """'RBp' gives Black 1 rook + 1 bishop, White 1 pawn."""
        from keisei.shogi.shogi_core_definitions import get_unpromoted_types

        hands = {
            Color.BLACK.value: {pt: 0 for pt in get_unpromoted_types()},
            Color.WHITE.value: {pt: 0 for pt in get_unpromoted_types()},
        }
        populate_hands_from_sfen_segment(hands, "RBp")
        assert hands[Color.BLACK.value][PieceType.ROOK] == 1
        assert hands[Color.BLACK.value][PieceType.BISHOP] == 1
        assert hands[Color.WHITE.value][PieceType.PAWN] == 1


# ===========================================================================
# 4. Observation generation (~10 tests)
# ===========================================================================


class TestObservationGeneration:
    """Tests for generate_neural_network_observation via ShogiGame.get_observation."""

    def test_observation_shape(self, new_game: ShogiGame):
        obs = new_game.get_observation()
        assert obs.shape == (46, 9, 9)

    def test_observation_dtype(self, new_game: ShogiGame):
        obs = new_game.get_observation()
        assert obs.dtype == np.float32

    def test_initial_color_indicator_plane_black(self, new_game: ShogiGame):
        """Channel 42 should be all 1.0 when Black is to move."""
        obs = new_game.get_observation()
        assert np.all(obs[42, :, :] == 1.0)

    def test_initial_move_count_plane_zero(self, new_game: ShogiGame):
        """Channel 43 should be 0.0 for move 0 (initial position)."""
        obs = new_game.get_observation()
        assert np.all(obs[43, :, :] == 0.0)

    def test_initial_hand_planes_zero(self, new_game: ShogiGame):
        """Channels 28-41 (hand planes) should be 0 at the start."""
        obs = new_game.get_observation()
        hand_planes = obs[28:42, :, :]
        assert np.all(hand_planes == 0.0)

    def test_initial_reserved_planes_zero(self, new_game: ShogiGame):
        """Channels 44 and 45 (reserved) should be all zeros."""
        obs = new_game.get_observation()
        assert np.all(obs[44, :, :] == 0.0)
        assert np.all(obs[45, :, :] == 0.0)

    def test_initial_black_pawn_plane(self, new_game: ShogiGame):
        """
        Channel 0 is current player (Black) PAWN plane.
        In the initial position, Black's pawns are on row 6 (all nine columns).
        Since it is Black's perspective, coordinates are NOT flipped.
        """
        obs = new_game.get_observation()
        pawn_plane = obs[0, :, :]
        # Row 6 should have all 1s (Black's nine pawns)
        for c in range(9):
            assert pawn_plane[6, c] == 1.0, f"Expected pawn at (6,{c})"
        # All other rows should be 0 on the pawn plane
        for r in range(9):
            if r == 6:
                continue
            for c in range(9):
                assert pawn_plane[r, c] == 0.0, f"Expected no pawn at ({r},{c})"

    def test_initial_black_king_plane(self, new_game: ShogiGame):
        """
        Channel 7 is current player (Black) KING plane.
        Black king starts at (8, 4).
        """
        obs = new_game.get_observation()
        king_plane = obs[7, :, :]
        assert king_plane[8, 4] == 1.0
        # Only one king
        assert np.sum(king_plane) == 1.0

    def test_initial_opponent_pawn_plane(self, new_game: ShogiGame):
        """
        Channel 14 is opponent (White) PAWN plane, viewed from Black's perspective.
        White's pawns are at row 2, all nine columns (no flip since Black's turn).
        """
        obs = new_game.get_observation()
        opp_pawn_plane = obs[14, :, :]
        for c in range(9):
            assert opp_pawn_plane[2, c] == 1.0, f"Expected opp pawn at (2,{c})"
        assert np.sum(opp_pawn_plane) == 9.0

    def test_observation_changes_after_move(self, new_game: ShogiGame):
        """
        After making a pawn move, the observation should differ from initial.
        Move Black's pawn at (6,6) to (5,6) -- 7g7f in SFEN.
        """
        obs_before = new_game.get_observation().copy()
        # (from_r=6, from_c=6, to_r=5, to_c=6, promote=False)
        new_game.make_move((6, 6, 5, 6, False))
        obs_after = new_game.get_observation()
        # Observations must differ (at least the turn indicator plane flips)
        assert not np.array_equal(obs_before, obs_after)

    def test_color_indicator_flips_after_move(self, new_game: ShogiGame):
        """After Black moves, channel 42 should be 0.0 (White's turn)."""
        new_game.make_move((6, 6, 5, 6, False))  # Black pawn push
        obs = new_game.get_observation()
        assert np.all(obs[42, :, :] == 0.0)

    def test_observation_hand_pieces_after_capture(self):
        """
        Set up a position where a capture has occurred, then check
        that hand plane is nonzero.
        """
        # Use a custom SFEN where Black has a pawn in hand
        sfen = "lnsgkgsnl/1r5b1/pppp1pppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b P 1"
        game = ShogiGame.from_sfen(sfen)
        obs = game.get_observation()
        # Channel 28 is current player's PAWN hand plane
        # Value should be count / max_count = 1 / 18
        expected_value = 1.0 / 18.0
        assert np.allclose(obs[28, :, :], expected_value)


# ===========================================================================
# 5. KIF export (~4 tests)
# ===========================================================================


class TestKifExport:
    """Tests for game_to_kif."""

    def test_kif_returns_string(self, new_game: ShogiGame):
        kif = game_to_kif(new_game)
        assert isinstance(kif, str)
        assert len(kif) > 0

    def test_kif_contains_header(self, new_game: ShogiGame):
        kif = game_to_kif(new_game)
        assert "#KIF" in kif

    def test_kif_contains_moves_section(self, new_game: ShogiGame):
        kif = game_to_kif(new_game)
        assert "moves" in kif

    def test_kif_with_moves_has_entries(self, new_game: ShogiGame):
        """After making moves, the KIF output should contain move entries."""
        new_game.make_move((6, 6, 5, 6, False))  # 7g7f (Black pawn)
        new_game.make_move((2, 2, 3, 2, False))  # 7c7d (White pawn)
        kif = game_to_kif(new_game)
        # After the "moves" line, there should be move entries like "1 3g3f"
        lines = kif.split("\n")
        moves_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "moves":
                moves_idx = i
                break
        assert moves_idx is not None, "Did not find 'moves' line in KIF"
        # At least one numbered move line should follow
        move_lines = [
            l for l in lines[moves_idx + 1 :] if l.strip() and l.strip()[0].isdigit()
        ]
        assert len(move_lines) >= 2, f"Expected >= 2 move lines, got {len(move_lines)}"

    def test_kif_contains_eof(self, new_game: ShogiGame):
        kif = game_to_kif(new_game)
        assert "*EOF" in kif


# ===========================================================================
# 6. Text representation (~4 tests)
# ===========================================================================


class TestTextRepresentation:
    """Tests for convert_game_to_text_representation via ShogiGame.to_string."""

    def test_returns_multiline_string(self, new_game: ShogiGame):
        text = new_game.to_string()
        assert isinstance(text, str)
        lines = text.split("\n")
        # 9 board rows + column labels + turn info + Black hand + White hand = 13
        assert len(lines) >= 13

    def test_contains_piece_symbols(self, new_game: ShogiGame):
        text = new_game.to_string()
        # Black's pieces are uppercase, White's are lowercase
        assert "K" in text  # Black king
        assert "k" in text  # White king
        assert "P" in text  # Black pawn

    def test_contains_file_labels(self, new_game: ShogiGame):
        text = new_game.to_string()
        # The bottom row should have column labels a through i
        assert "a" in text
        assert "i" in text

    def test_board_rows_count(self, new_game: ShogiGame):
        """The board should have 9 rank lines (labeled 9 down to 1)."""
        text = new_game.to_string()
        lines = text.split("\n")
        # First 9 lines are the board ranks (labeled 9 down to 1 from the board's perspective)
        rank_labels_found = set()
        for line in lines:
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                rank_labels_found.add(stripped[0])
        # Ranks 1 through 9 should all appear
        for expected in "123456789":
            assert expected in rank_labels_found, f"Missing rank {expected}"


# ===========================================================================
# 7. SFEN move encoding / decoding (~4 tests)
# ===========================================================================


class TestSfenMoveEncodingDecoding:
    """Tests for encode_move_to_sfen_string and sfen_to_move_tuple."""

    def test_encode_board_move(self):
        """Board move (6, 0, 5, 0, False) -> '9g9f'."""
        # row=6 -> rank 'g', col=0 -> file '9'
        # row=5 -> rank 'f', col=0 -> file '9'
        result = encode_move_to_sfen_string((6, 0, 5, 0, False))
        assert result == "9g9f"

    def test_encode_board_move_with_promotion(self):
        """Board move with promotion flag (1, 7, 2, 6, True) -> '2b3c+'."""
        # row=1 -> rank 'b', col=7 -> file '2'
        # row=2 -> rank 'c', col=6 -> file '3'
        result = encode_move_to_sfen_string((1, 7, 2, 6, True))
        assert result == "2b3c+"

    def test_encode_drop_move(self):
        """Drop move (None, None, 4, 4, PieceType.PAWN) -> 'P*5e'."""
        result = encode_move_to_sfen_string((None, None, 4, 4, PieceType.PAWN))
        assert result == "P*5e"

    def test_decode_board_move(self):
        """'7g7f' -> (6, 6, 5, 6, False)."""
        # file 7 -> col 2, rank g -> row 6, file 7 -> col 2, rank f -> row 5
        result = sfen_to_move_tuple("7g7f")
        assert result == (6, 2, 5, 2, False)

    def test_decode_drop_move(self):
        """'P*5e' -> (None, None, 4, 4, PieceType.PAWN)."""
        result = sfen_to_move_tuple("P*5e")
        assert result == (None, None, 4, 4, PieceType.PAWN)

    def test_decode_promoted_board_move(self):
        """'2b3a+' -> board move with promote=True."""
        result = sfen_to_move_tuple("2b3a+")
        # file 2 -> col 7, rank b -> row 1
        # file 3 -> col 6, rank a -> row 0
        assert result == (1, 7, 0, 6, True)

    def test_encode_decode_roundtrip_board_move(self):
        """Encoding then decoding a board move returns the original tuple."""
        original = (6, 6, 5, 6, False)
        sfen_str = encode_move_to_sfen_string(original)
        decoded = sfen_to_move_tuple(sfen_str)
        assert decoded == original

    def test_encode_decode_roundtrip_drop_move(self):
        """Encoding then decoding a drop move returns the original tuple."""
        original = (None, None, 4, 4, PieceType.PAWN)
        sfen_str = encode_move_to_sfen_string(original)
        decoded = sfen_to_move_tuple(sfen_str)
        assert decoded == original

    def test_decode_invalid_move_raises(self):
        with pytest.raises(ValueError, match="Invalid SFEN move format"):
            sfen_to_move_tuple("invalid")

    def test_encode_invalid_tuple_raises(self):
        with pytest.raises(ValueError, match="Invalid MoveTuple format"):
            encode_move_to_sfen_string((1, 2, 3))  # type: ignore[arg-type]


# ===========================================================================
# Additional edge-case tests
# ===========================================================================


class TestSfenBoardCharHelper:
    """Tests for the _get_sfen_board_char helper function."""

    def test_black_pawn(self):
        piece = Piece(PieceType.PAWN, Color.BLACK)
        assert _get_sfen_board_char(piece) == "P"

    def test_white_pawn(self):
        piece = Piece(PieceType.PAWN, Color.WHITE)
        assert _get_sfen_board_char(piece) == "p"

    def test_black_promoted_pawn(self):
        piece = Piece(PieceType.PROMOTED_PAWN, Color.BLACK)
        assert _get_sfen_board_char(piece) == "+P"

    def test_white_promoted_rook(self):
        piece = Piece(PieceType.PROMOTED_ROOK, Color.WHITE)
        assert _get_sfen_board_char(piece) == "+r"


class TestSfenHandCanonicalOrder:
    """Tests that the SFEN hand piece canonical order is correct."""

    def test_canonical_order(self):
        expected = [
            PieceType.ROOK,
            PieceType.BISHOP,
            PieceType.GOLD,
            PieceType.SILVER,
            PieceType.KNIGHT,
            PieceType.LANCE,
            PieceType.PAWN,
        ]
        assert SFEN_HAND_PIECE_CANONICAL_ORDER == expected


class TestPopulateBoardFromSfen:
    """Tests for populate_board_from_sfen_segment edge cases."""

    def test_wrong_number_of_ranks_raises(self):
        board = [[None for _ in range(9)] for _ in range(9)]
        with pytest.raises(ValueError, match="Expected 9 ranks"):
            populate_board_from_sfen_segment(board, "9/9/9/9")

    def test_row_too_many_columns_raises(self):
        board = [[None for _ in range(9)] for _ in range(9)]
        # "19" means 1 empty + 9 empty = 10 columns on the first row
        with pytest.raises(ValueError):
            populate_board_from_sfen_segment(
                board, "19/9/9/9/9/9/9/9/9"
            )
