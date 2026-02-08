"""Tests for ShogiGame class methods not covered by existing behavioral tests.

Covers: seed(), test_move(), __deepcopy__, get_observation(), property accessors,
reset(), error handling, board utility methods, and miscellaneous aliases.
"""

import copy

import numpy as np
import pytest

from keisei.shogi.shogi_core_definitions import Color, MoveTuple, Piece, PieceType
from keisei.shogi.shogi_game import ShogiGame


@pytest.fixture
def new_game() -> ShogiGame:
    return ShogiGame()


# ---------------------------------------------------------------------------
# A well-known legal opening move for BLACK in the initial position:
# Move the pawn at (6, 6) forward to (5, 6) -- the "7六歩" opening.
# ---------------------------------------------------------------------------
LEGAL_PAWN_PUSH: MoveTuple = (6, 6, 5, 6, False)

# An illegal move: move BLACK's pawn at (6, 0) to (4, 0) -- pawns move one
# square, not two.
ILLEGAL_PAWN_JUMP: MoveTuple = (6, 0, 4, 0, False)


# ===================================================================
# TestSeed
# ===================================================================
class TestSeed:
    """Tests for the seed() method."""

    def test_seed_returns_game_instance_for_chaining(self, new_game: ShogiGame) -> None:
        """seed() should return the game instance so calls can be chained."""
        result = new_game.seed(42)
        assert result is new_game

    def test_seed_stores_value(self, new_game: ShogiGame) -> None:
        """seed() should store the provided value in _seed_value."""
        new_game.seed(42)
        assert new_game._seed_value == 42

        new_game.seed(None)
        assert new_game._seed_value is None


# ===================================================================
# TestTestMove
# ===================================================================
class TestTestMove:
    """Tests for the test_move() method."""

    def test_returns_true_for_legal_move(self, new_game: ShogiGame) -> None:
        """test_move should return True for a legal move."""
        assert new_game.test_move(LEGAL_PAWN_PUSH) is True

    def test_returns_false_for_illegal_move(self, new_game: ShogiGame) -> None:
        """test_move should return False for an illegal move."""
        assert new_game.test_move(ILLEGAL_PAWN_JUMP) is False

    def test_game_state_unchanged_after_test_move(self, new_game: ShogiGame) -> None:
        """Board, hands, current_player, and move_count must not change."""
        board_before = copy.deepcopy(new_game.board)
        hands_before = copy.deepcopy(new_game.hands)
        player_before = new_game.current_player
        move_count_before = new_game.move_count

        new_game.test_move(LEGAL_PAWN_PUSH)

        assert new_game.board == board_before
        assert new_game.hands == hands_before
        assert new_game.current_player == player_before
        assert new_game.move_count == move_count_before

    def test_returns_false_when_game_is_over(self, new_game: ShogiGame) -> None:
        """test_move should return False when the game is already over."""
        new_game.game_over = True
        assert new_game.test_move(LEGAL_PAWN_PUSH) is False


# ===================================================================
# TestDeepCopy
# ===================================================================
class TestDeepCopy:
    """Tests for __deepcopy__ behaviour."""

    def test_deep_copy_produces_independent_object(self, new_game: ShogiGame) -> None:
        """copy.deepcopy should return a distinct object."""
        game_copy = copy.deepcopy(new_game)
        assert game_copy is not new_game
        assert id(game_copy) != id(new_game)

    def test_modifying_copy_does_not_affect_original(
        self, new_game: ShogiGame
    ) -> None:
        """Making a move on the copy must leave the original untouched."""
        game_copy = copy.deepcopy(new_game)

        # Make a move on the copy
        game_copy.make_move(LEGAL_PAWN_PUSH)

        # Original should still be at the initial state
        assert new_game.move_count == 0
        assert new_game.current_player == Color.BLACK
        assert new_game.board[5][6] is None  # Square should still be empty

    def test_copy_has_identical_state(self, new_game: ShogiGame) -> None:
        """The copy should reproduce board, current_player, move_count, hands,
        and move_history exactly."""
        # Make a move first so there is non-trivial state
        new_game.make_move(LEGAL_PAWN_PUSH)

        game_copy = copy.deepcopy(new_game)

        assert game_copy.current_player == new_game.current_player
        assert game_copy.move_count == new_game.move_count
        assert game_copy.board == new_game.board
        assert game_copy.hands == new_game.hands
        assert len(game_copy.move_history) == len(new_game.move_history)


# ===================================================================
# TestGetObservation
# ===================================================================
class TestGetObservation:
    """Tests for get_observation() and its alias get_state()."""

    def test_observation_shape_and_dtype(self, new_game: ShogiGame) -> None:
        """Observation must be a float32 ndarray with shape (46, 9, 9)."""
        obs = new_game.get_observation()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (46, 9, 9)
        assert obs.dtype == np.float32

    def test_different_states_produce_different_observations(
        self, new_game: ShogiGame
    ) -> None:
        """After making a move the observation should change."""
        obs_before = new_game.get_observation().copy()
        new_game.make_move(LEGAL_PAWN_PUSH)
        obs_after = new_game.get_observation()
        assert not np.array_equal(obs_before, obs_after)

    def test_get_state_is_alias(self, new_game: ShogiGame) -> None:
        """get_state() should return the same result as get_observation()."""
        obs = new_game.get_observation()
        state = new_game.get_state()
        np.testing.assert_array_equal(obs, state)


# ===================================================================
# TestPropertyAccessors
# ===================================================================
class TestPropertyAccessors:
    """Tests for public property accessors on a fresh game."""

    def test_current_player_is_black(self, new_game: ShogiGame) -> None:
        assert new_game.current_player == Color.BLACK

    def test_game_over_is_false_and_move_count_zero(
        self, new_game: ShogiGame
    ) -> None:
        assert new_game.game_over is False
        assert new_game.move_count == 0

    def test_max_moves_per_game_default_and_custom(self) -> None:
        """max_moves_per_game should reflect the constructor argument."""
        default_game = ShogiGame()
        assert default_game.max_moves_per_game == 500

        custom_game = ShogiGame(max_moves_per_game=200)
        assert custom_game.max_moves_per_game == 200


# ===================================================================
# TestReset
# ===================================================================
class TestReset:
    """Tests for the reset() method."""

    def test_reset_returns_observation(self, new_game: ShogiGame) -> None:
        """reset() should return a numpy observation array."""
        obs = new_game.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (46, 9, 9)

    def test_reset_restores_initial_state(self, new_game: ShogiGame) -> None:
        """After making moves, reset() should restore the initial state."""
        new_game.make_move(LEGAL_PAWN_PUSH)
        assert new_game.move_count == 1

        new_game.reset()

        assert new_game.move_count == 0
        assert new_game.current_player == Color.BLACK
        assert new_game.game_over is False
        assert new_game.move_history == []
        # The pawn should be back at its starting square
        pawn = new_game.get_piece(6, 6)
        assert pawn is not None
        assert pawn.type == PieceType.PAWN
        assert pawn.color == Color.BLACK


# ===================================================================
# TestErrorHandling
# ===================================================================
class TestErrorHandling:
    """Tests for error-raising and graceful error paths."""

    def test_make_move_invalid_format_raises_value_error(
        self, new_game: ShogiGame
    ) -> None:
        """make_move with a badly-formed tuple should raise ValueError."""
        with pytest.raises(ValueError):
            new_game.make_move(("a", "b", "c"))  # type: ignore[arg-type]

    def test_make_move_on_finished_game_returns_done_true(
        self, new_game: ShogiGame
    ) -> None:
        """make_move on a game_over game should return a tuple with done=True."""
        new_game.game_over = True
        result = new_game.make_move(LEGAL_PAWN_PUSH)
        # Should be the 4-tuple (obs, reward, done, info)
        assert isinstance(result, tuple)
        obs, reward, done, info = result
        assert done is True

    def test_get_piece_out_of_bounds_returns_none(
        self, new_game: ShogiGame
    ) -> None:
        """get_piece with out-of-bounds coordinates should return None."""
        assert new_game.get_piece(9, 0) is None
        assert new_game.get_piece(-1, 4) is None
        assert new_game.get_piece(0, 9) is None

    def test_set_piece_out_of_bounds_raises_value_error(
        self, new_game: ShogiGame
    ) -> None:
        """set_piece with out-of-bounds coordinates should raise ValueError."""
        with pytest.raises(ValueError, match="out of bounds"):
            new_game.set_piece(9, 0, Piece(PieceType.PAWN, Color.BLACK))


# ===================================================================
# TestBoardUtilityMethods
# ===================================================================
class TestBoardUtilityMethods:
    """Tests for is_on_board, get_piece, set_piece, and find_king."""

    def test_is_on_board_valid_squares(self, new_game: ShogiGame) -> None:
        assert new_game.is_on_board(0, 0) is True
        assert new_game.is_on_board(8, 8) is True
        assert new_game.is_on_board(4, 4) is True

    def test_is_on_board_invalid_squares(self, new_game: ShogiGame) -> None:
        assert new_game.is_on_board(9, 0) is False
        assert new_game.is_on_board(0, 9) is False
        assert new_game.is_on_board(-1, 0) is False
        assert new_game.is_on_board(0, -1) is False

    def test_get_piece_returns_piece_or_none(self, new_game: ShogiGame) -> None:
        """get_piece should return a Piece for occupied squares and None for empty ones."""
        # The initial position has a LANCE at (8, 0) for BLACK
        piece = new_game.get_piece(8, 0)
        assert piece is not None
        assert piece.type == PieceType.LANCE
        assert piece.color == Color.BLACK

        # Center of the board is empty in the initial position
        assert new_game.get_piece(4, 4) is None

    def test_set_piece_places_correctly(self, new_game: ShogiGame) -> None:
        """set_piece should place a piece at the specified location."""
        gold = Piece(PieceType.GOLD, Color.WHITE)
        new_game.set_piece(4, 4, gold)
        retrieved = new_game.get_piece(4, 4)
        assert retrieved is not None
        assert retrieved.type == PieceType.GOLD
        assert retrieved.color == Color.WHITE

    def test_find_king_returns_correct_positions(
        self, new_game: ShogiGame
    ) -> None:
        """find_king should locate the kings at their initial positions."""
        black_king_pos = new_game.find_king(Color.BLACK)
        assert black_king_pos == (8, 4)

        white_king_pos = new_game.find_king(Color.WHITE)
        assert white_king_pos == (0, 4)


# ===================================================================
# TestAdditionalMethods
# ===================================================================
class TestAdditionalMethods:
    """Tests for to_string, to_sfen aliases, is_in_promotion_zone,
    and get_board_state_hash."""

    def test_to_string_returns_nonempty(self, new_game: ShogiGame) -> None:
        result = new_game.to_string()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_to_sfen_is_alias_for_to_sfen_string(
        self, new_game: ShogiGame
    ) -> None:
        assert new_game.to_sfen() == new_game.to_sfen_string()

    def test_is_in_promotion_zone_black(self, new_game: ShogiGame) -> None:
        """For BLACK, rows 0-2 are the promotion zone."""
        for row in range(3):
            assert new_game.is_in_promotion_zone(row, Color.BLACK) is True
        for row in range(3, 9):
            assert new_game.is_in_promotion_zone(row, Color.BLACK) is False

    def test_is_in_promotion_zone_white(self, new_game: ShogiGame) -> None:
        """For WHITE, rows 6-8 are the promotion zone."""
        for row in range(6):
            assert new_game.is_in_promotion_zone(row, Color.WHITE) is False
        for row in range(6, 9):
            assert new_game.is_in_promotion_zone(row, Color.WHITE) is True

    def test_get_board_state_hash_returns_tuple(
        self, new_game: ShogiGame
    ) -> None:
        """get_board_state_hash should return a hashable tuple."""
        h = new_game.get_board_state_hash()
        assert isinstance(h, tuple)
        # It should be hashable (usable as a dict key)
        _ = {h: True}
