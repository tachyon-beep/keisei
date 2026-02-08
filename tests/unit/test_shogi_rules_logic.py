"""
Comprehensive tests for keisei.shogi.shogi_rules_logic module.

Covers: find_king, is_in_check, is_piece_type_sliding, generate_piece_potential_moves,
check_for_nifu, check_if_square_is_attacked, can_promote_specific_piece,
must_promote_specific_piece, can_drop_specific_piece, generate_all_legal_moves,
check_for_sennichite.
"""

import pytest

from keisei.shogi.shogi_core_definitions import Color, MoveTuple, Piece, PieceType
from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_rules_logic import (
    can_drop_specific_piece,
    can_promote_specific_piece,
    check_for_nifu,
    check_for_sennichite,
    check_if_square_is_attacked,
    find_king,
    generate_all_legal_moves,
    generate_piece_potential_moves,
    is_in_check,
    is_piece_type_sliding,
    must_promote_specific_piece,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def new_game() -> ShogiGame:
    """Standard initial Shogi position."""
    return ShogiGame()


@pytest.fixture
def empty_board_game() -> ShogiGame:
    """Game with only kings on the board (Black king at (8,4), White king at (0,4))."""
    return ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1")


# ===========================================================================
# 1. Legal move generation (~4 tests)
# ===========================================================================


class TestLegalMoveGeneration:
    """Tests for generate_all_legal_moves."""

    def test_initial_position_has_30_legal_moves(self, new_game: ShogiGame) -> None:
        """Black has exactly 30 legal moves in the standard opening position."""
        moves = generate_all_legal_moves(new_game)
        assert len(moves) == 30

    def test_all_moves_are_5_tuples(self, new_game: ShogiGame) -> None:
        """Every returned move must be a 5-element tuple."""
        moves = generate_all_legal_moves(new_game)
        for move in moves:
            assert isinstance(move, tuple), f"Move {move} is not a tuple"
            assert len(move) == 5, f"Move {move} does not have 5 elements"

    def test_no_duplicate_moves(self, new_game: ShogiGame) -> None:
        """Legal moves list must contain no duplicates."""
        moves = generate_all_legal_moves(new_game)
        assert len(moves) == len(set(moves)), "Duplicate moves detected"

    def test_white_has_legal_moves_after_first_move(
        self, new_game: ShogiGame
    ) -> None:
        """After Black plays one move, White should have legal moves."""
        new_game.make_move((6, 4, 5, 4, False))  # Advance center pawn
        assert new_game.current_player == Color.WHITE
        white_moves = generate_all_legal_moves(new_game)
        assert len(white_moves) == 30


# ===========================================================================
# 2. Piece movement patterns (~6 tests)
# ===========================================================================


class TestPieceMovementPatterns:
    """Tests for generate_piece_potential_moves with specific piece placements."""

    def test_pawn_moves_forward_one_square(
        self, empty_board_game: ShogiGame
    ) -> None:
        """A Black pawn at (4,4) should generate exactly one move: (3,4)."""
        game = empty_board_game
        pawn = Piece(PieceType.PAWN, Color.BLACK)
        game.set_piece(4, 4, pawn)

        moves = generate_piece_potential_moves(game, pawn, 4, 4)
        assert moves == [(3, 4)]

    def test_king_moves_all_eight_adjacent(
        self, empty_board_game: ShogiGame
    ) -> None:
        """A king at (4,4) on an otherwise empty board generates all 8 adjacent squares."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4K4/9/9/9/9 b - 1")
        king = game.get_piece(4, 4)
        assert king is not None

        moves = generate_piece_potential_moves(game, king, 4, 4)
        expected = {(3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4), (5, 5)}
        assert set(moves) == expected
        assert len(moves) == 8

    def test_gold_moves_six_directions(
        self, empty_board_game: ShogiGame
    ) -> None:
        """A Black gold at (4,4) generates 6 moves: forward 3, sideways 2, backward 1."""
        game = empty_board_game
        gold = Piece(PieceType.GOLD, Color.BLACK)
        game.set_piece(4, 4, gold)

        moves = generate_piece_potential_moves(game, gold, 4, 4)
        # Forward: (3,3), (3,4), (3,5); Sideways: (4,3), (4,5); Back: (5,4)
        expected = {(3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 4)}
        assert set(moves) == expected
        assert len(moves) == 6

    def test_silver_moves_five_directions(
        self, empty_board_game: ShogiGame
    ) -> None:
        """A Black silver at (4,4) generates 5 moves: forward 3, backward diagonals 2."""
        game = empty_board_game
        silver = Piece(PieceType.SILVER, Color.BLACK)
        game.set_piece(4, 4, silver)

        moves = generate_piece_potential_moves(game, silver, 4, 4)
        # Forward: (3,3), (3,4), (3,5); Backward diags: (5,3), (5,5)
        expected = {(3, 3), (3, 4), (3, 5), (5, 3), (5, 5)}
        assert set(moves) == expected
        assert len(moves) == 5

    def test_knight_moves_two_squares(
        self, empty_board_game: ShogiGame
    ) -> None:
        """A Black knight at (4,4) generates 2 moves: (2,3) and (2,5)."""
        game = empty_board_game
        knight = Piece(PieceType.KNIGHT, Color.BLACK)
        game.set_piece(4, 4, knight)

        moves = generate_piece_potential_moves(game, knight, 4, 4)
        expected = {(2, 3), (2, 5)}
        assert set(moves) == expected
        assert len(moves) == 2

    def test_lance_slides_forward(self, empty_board_game: ShogiGame) -> None:
        """A Black lance at (4,4) slides forward through rows 3, 2, 1, 0 (4 squares)."""
        game = empty_board_game
        lance = Piece(PieceType.LANCE, Color.BLACK)
        game.set_piece(4, 4, lance)

        moves = generate_piece_potential_moves(game, lance, 4, 4)
        # Lance slides forward from row 4 toward row 0, but White king is at (0,4)
        # so it includes (0,4) as a capture target
        expected = {(3, 4), (2, 4), (1, 4), (0, 4)}
        assert set(moves) == expected
        assert len(moves) == 4


# ===========================================================================
# 3. Check detection (~4 tests)
# ===========================================================================


class TestCheckDetection:
    """Tests for is_in_check and check_if_square_is_attacked."""

    def test_initial_position_neither_player_in_check(
        self, new_game: ShogiGame
    ) -> None:
        """Neither player is in check at the start of the game."""
        assert is_in_check(new_game, Color.BLACK) is False
        assert is_in_check(new_game, Color.WHITE) is False

    def test_rook_gives_check_on_file(self) -> None:
        """A White rook on the same file as the Black king with no blockers gives check."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4r4/9/9/9/4K4 b - 1")
        assert is_in_check(game, Color.BLACK) is True

    def test_bishop_gives_check_on_diagonal(self) -> None:
        """A White bishop on the diagonal of the Black king gives check."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/3b5/4K4 b - 1")
        assert is_in_check(game, Color.BLACK) is True

    def test_removing_blocker_reveals_check(self) -> None:
        """Removing a piece that blocks an attack line reveals check (discovered check)."""
        # Black pawn at (6,4) blocks White rook at (4,4) from checking Black king at (8,4)
        game = ShogiGame.from_sfen("4k4/9/9/9/4r4/9/4P4/9/4K4 b - 1")
        assert is_in_check(game, Color.BLACK) is False

        # Remove the blocking pawn
        game.set_piece(6, 4, None)
        assert is_in_check(game, Color.BLACK) is True


# ===========================================================================
# 4. Promotion rules (~4 tests)
# ===========================================================================


class TestPromotionRules:
    """Tests for can_promote_specific_piece and must_promote_specific_piece."""

    def test_can_promote_entering_promotion_zone(self) -> None:
        """A Black pawn moving from row 3 to row 2 (entering promotion zone) can promote."""
        game = ShogiGame.from_sfen("4k4/9/9/4P4/9/9/9/9/4K4 b - 1")
        pawn = game.get_piece(3, 4)
        assert pawn is not None
        assert can_promote_specific_piece(game, pawn, 3, 2) is True

    def test_cannot_promote_outside_promotion_zone(self) -> None:
        """A Black pawn moving within rows 5-6 (outside promotion zone) cannot promote."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        pawn = Piece(PieceType.PAWN, Color.BLACK)
        assert can_promote_specific_piece(game, pawn, 5, 6) is False

    def test_must_promote_pawn_on_last_rank(self) -> None:
        """A Black pawn moving to row 0 must promote."""
        pawn = Piece(PieceType.PAWN, Color.BLACK)
        assert must_promote_specific_piece(pawn, 0) is True

    def test_must_promote_knight_on_last_two_ranks(self) -> None:
        """A Black knight moving to row 0 or row 1 must promote."""
        knight = Piece(PieceType.KNIGHT, Color.BLACK)
        assert must_promote_specific_piece(knight, 0) is True
        assert must_promote_specific_piece(knight, 1) is True
        # Row 2 does NOT require mandatory promotion
        assert must_promote_specific_piece(knight, 2) is False

    def test_gold_king_promoted_cannot_promote(self) -> None:
        """Gold, King, and already-promoted pieces cannot promote."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        assert can_promote_specific_piece(game, Piece(PieceType.GOLD, Color.BLACK), 3, 2) is False
        assert can_promote_specific_piece(game, Piece(PieceType.KING, Color.BLACK), 3, 2) is False
        assert can_promote_specific_piece(game, Piece(PieceType.PROMOTED_PAWN, Color.BLACK), 3, 2) is False


# ===========================================================================
# 5. Drop validation (~6 tests)
# ===========================================================================


class TestDropValidation:
    """Tests for can_drop_specific_piece and check_for_nifu."""

    def test_drop_on_occupied_square_fails(self) -> None:
        """Dropping on an occupied square returns False."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4P4/9/9/9/4K4 b P 1")
        # Square (4,4) has a pawn on it
        assert can_drop_specific_piece(game, PieceType.PAWN, 4, 4, Color.BLACK) is False

    def test_nifu_pawn_in_same_column(self) -> None:
        """Nifu: check_for_nifu returns True when a pawn already exists in the column."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4P4/9/9/9/4K4 b P 1")
        assert check_for_nifu(game, Color.BLACK, 4) is True
        # No pawn in column 3
        assert check_for_nifu(game, Color.BLACK, 3) is False

    def test_drop_pawn_nifu_fails(self) -> None:
        """Dropping a pawn in a column that already has an unpromoted pawn fails (nifu)."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4P4/9/9/9/4K4 b P 1")
        # Column 4 already has a Black pawn
        assert can_drop_specific_piece(game, PieceType.PAWN, 5, 4, Color.BLACK) is False

    def test_drop_pawn_on_last_rank_fails(self) -> None:
        """A pawn cannot be dropped on the last rank (row 0 for Black)."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        assert can_drop_specific_piece(game, PieceType.PAWN, 0, 3, Color.BLACK) is False

    def test_drop_lance_on_last_rank_fails(self) -> None:
        """A lance cannot be dropped on the last rank (row 0 for Black)."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b L 1")
        assert can_drop_specific_piece(game, PieceType.LANCE, 0, 3, Color.BLACK) is False

    def test_drop_knight_on_last_two_ranks_fails(self) -> None:
        """A knight cannot be dropped on the last two ranks (rows 0-1 for Black)."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b N 1")
        assert can_drop_specific_piece(game, PieceType.KNIGHT, 0, 3, Color.BLACK) is False
        assert can_drop_specific_piece(game, PieceType.KNIGHT, 1, 3, Color.BLACK) is False

    def test_valid_drop_on_empty_square(self) -> None:
        """A pawn drop on an empty square with no restrictions succeeds."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        # (5,3) is empty, no pawn in column 3, not last rank
        assert can_drop_specific_piece(game, PieceType.PAWN, 5, 3, Color.BLACK) is True


# ===========================================================================
# 6. Sennichite (~2 tests)
# ===========================================================================


class TestSennichite:
    """Tests for check_for_sennichite."""

    def test_fresh_game_no_sennichite(self, new_game: ShogiGame) -> None:
        """A fresh game with no moves has no sennichite."""
        assert check_for_sennichite(new_game) is False

    def test_sennichite_returns_bool(self, new_game: ShogiGame) -> None:
        """check_for_sennichite returns a boolean value."""
        result = check_for_sennichite(new_game)
        assert isinstance(result, bool)


# ===========================================================================
# 7. Move filtering (pin / self-check) (~2 tests)
# ===========================================================================


class TestMoveFiltering:
    """Tests that illegal moves (leaving own king in check) are excluded."""

    def test_move_leaving_king_in_check_excluded(self) -> None:
        """A pinned piece cannot move in a way that exposes the king to check."""
        # Black silver at (7,4), White rook at (4,4), Black king at (8,4)
        # The silver is pinned on the file
        game = ShogiGame.from_sfen("4k4/9/9/9/4r4/9/9/4S4/4K4 b - 1")
        moves = generate_all_legal_moves(game)

        # Silver moving diagonally (e.g., to (6,3)) would expose king
        perpendicular_move = (7, 4, 6, 3, False)
        assert perpendicular_move not in moves

    def test_pinned_piece_can_move_along_pin_axis(self) -> None:
        """A pinned piece may still move along the pin axis (toward/away from attacker)."""
        # Same position as above
        game = ShogiGame.from_sfen("4k4/9/9/9/4r4/9/9/4S4/4K4 b - 1")
        moves = generate_all_legal_moves(game)

        # Silver can move forward along the file toward the rook
        forward_move = (7, 4, 6, 4, False)
        assert forward_move in moves


# ===========================================================================
# 8. Utility function tests
# ===========================================================================


class TestUtilityFunctions:
    """Tests for find_king, is_piece_type_sliding, check_if_square_is_attacked."""

    def test_find_king_initial_position(self, new_game: ShogiGame) -> None:
        """find_king locates both kings in the initial position."""
        assert find_king(new_game, Color.BLACK) == (8, 4)
        assert find_king(new_game, Color.WHITE) == (0, 4)

    def test_find_king_returns_none_when_absent(self) -> None:
        """find_king returns None when the king is not on the board."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/9 b - 1")
        assert find_king(game, Color.BLACK) is None

    def test_is_in_check_returns_true_when_king_missing(self) -> None:
        """is_in_check returns True when the king is not found (invalid state)."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/9 b - 1")
        assert is_in_check(game, Color.BLACK) is True

    def test_is_piece_type_sliding(self) -> None:
        """Sliding pieces: LANCE, BISHOP, ROOK, PROMOTED_BISHOP, PROMOTED_ROOK."""
        assert is_piece_type_sliding(PieceType.LANCE) is True
        assert is_piece_type_sliding(PieceType.BISHOP) is True
        assert is_piece_type_sliding(PieceType.ROOK) is True
        assert is_piece_type_sliding(PieceType.PROMOTED_BISHOP) is True
        assert is_piece_type_sliding(PieceType.PROMOTED_ROOK) is True

    def test_non_sliding_pieces(self) -> None:
        """Non-sliding pieces: PAWN, KNIGHT, SILVER, GOLD, KING, and promoted minors."""
        assert is_piece_type_sliding(PieceType.PAWN) is False
        assert is_piece_type_sliding(PieceType.KNIGHT) is False
        assert is_piece_type_sliding(PieceType.SILVER) is False
        assert is_piece_type_sliding(PieceType.GOLD) is False
        assert is_piece_type_sliding(PieceType.KING) is False
        assert is_piece_type_sliding(PieceType.PROMOTED_PAWN) is False
        assert is_piece_type_sliding(PieceType.PROMOTED_KNIGHT) is False
        assert is_piece_type_sliding(PieceType.PROMOTED_SILVER) is False
        assert is_piece_type_sliding(PieceType.PROMOTED_LANCE) is False

    def test_check_if_square_is_attacked_by_rook(self) -> None:
        """A rook attacks squares along its rank and file."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4r4/9/9/9/4K4 b - 1")
        # Attacked along the file
        assert check_if_square_is_attacked(game, 8, 4, Color.WHITE) is True
        assert check_if_square_is_attacked(game, 5, 4, Color.WHITE) is True
        # Attacked along the rank
        assert check_if_square_is_attacked(game, 4, 3, Color.WHITE) is True
        # NOT attacked on diagonal
        assert check_if_square_is_attacked(game, 5, 5, Color.WHITE) is False

    def test_white_promotion_rules(self) -> None:
        """White pieces have mirrored promotion/must-promote rules."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        white_pawn = Piece(PieceType.PAWN, Color.WHITE)
        assert must_promote_specific_piece(white_pawn, 8) is True
        assert must_promote_specific_piece(white_pawn, 7) is False

        white_lance = Piece(PieceType.LANCE, Color.WHITE)
        assert must_promote_specific_piece(white_lance, 8) is True

        white_knight = Piece(PieceType.KNIGHT, Color.WHITE)
        assert must_promote_specific_piece(white_knight, 7) is True
        assert must_promote_specific_piece(white_knight, 8) is True
        assert must_promote_specific_piece(white_knight, 6) is False
