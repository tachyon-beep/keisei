"""
Tests for reward functionality in ShogiGame, especially in terminal states.
"""

from typing import Optional  # Added Optional for type hinting
from unittest.mock import patch  # Moved import to top level

import pytest  # Fixed import order

from keisei.shogi.shogi_core_definitions import (  # Added this import
    Color,
    MoveTuple,
    Piece,
    PieceType,
    TerminationReason,
    get_unpromoted_types,
)
from keisei.shogi.shogi_game import ShogiGame


@pytest.fixture
def new_game() -> ShogiGame:
    """Returns a ShogiGame instance initialized to the starting position."""
    return ShogiGame(max_moves_per_game=512)


@pytest.mark.parametrize(
    "winner, black_reward, white_reward, reason",
    [
        (Color.BLACK, 1.0, -1.0, "Tsumi"),
        (Color.WHITE, -1.0, 1.0, "Tsumi"),
        (None, 0.0, 0.0, "Sennichite"),
        (None, 0.0, 0.0, "Stalemate"),
        (None, 0.0, 0.0, "Max moves reached"),
    ],
)
def test_reward_in_terminal_states(
    new_game: ShogiGame,
    winner: Optional[Color],
    black_reward: float,
    white_reward: float,
    reason: str,
):
    """
    Test that rewards are correctly assigned for all terminal states.
    - Winner gets +1.0, loser gets -1.0.
    - Both players get 0.0 in a draw.
    """
    # Arrange: Manually set the game to a terminal state
    game = new_game
    game.game_over = True
    game.winner = winner
    game.termination_reason = reason

    # Act & Assert
    assert game.get_reward(perspective_player_color=Color.BLACK) == pytest.approx(
        black_reward
    )
    assert game.get_reward(perspective_player_color=Color.WHITE) == pytest.approx(
        white_reward
    )


def test_reward_ongoing_game(new_game: ShogiGame):
    """Test that rewards are 0 for ongoing game states."""
    # Make initial moves to get into mid-game
    new_game.make_move((6, 6, 5, 6, False))  # Black pawn
    new_game.make_move((2, 3, 3, 3, False))  # White pawn

    # Check reward before game over
    assert new_game.game_over is False  # Changed == to is
    assert new_game.get_reward(perspective_player_color=Color.BLACK) == pytest.approx(
        0.0
    )
    assert new_game.get_reward(perspective_player_color=Color.WHITE) == pytest.approx(
        0.0
    )


def test_make_move_returns_reward_in_tuple(new_game: ShogiGame):  # Use new_game fixture
    """Test that make_move returns reward as part of its return tuple in normal play."""
    game = new_game  # Use the fixture

    # Make a move and check the return value format
    move_outcome = game.make_move((6, 6, 5, 6, False))  # Black pawn

    assert isinstance(move_outcome, tuple)
    assert len(move_outcome) == 4
    _, reward, done, info = move_outcome

    # Mid-game move should have reward 0
    assert reward == pytest.approx(0.0)
    assert done is False  # Changed == to is
    assert isinstance(info, dict)


@pytest.mark.parametrize(
    "player_color, expected_reward",
    [(Color.BLACK, 1.0), (Color.WHITE, -1.0)],
)
def test_make_move_returns_correct_reward_on_checkmate(
    new_game: ShogiGame, player_color: Color, expected_reward: float
):
    game = new_game
    # Setup: Black to deliver checkmate to White
    # WK at (0,0), BG1 at (2,0) (moves to (1,0)), BG2 at (1,1)
    game.board = [[None for _ in range(9)] for _ in range(9)]
    game.hands = {
        Color.BLACK.value: {ptype: 0 for ptype in get_unpromoted_types()},
        Color.WHITE.value: {ptype: 0 for ptype in get_unpromoted_types()},
    }
    game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))  # White King
    game.set_piece(2, 0, Piece(PieceType.GOLD, Color.BLACK))  # Black Gold 1 (to move)
    game.set_piece(
        1, 1, Piece(PieceType.GOLD, Color.BLACK)
    )  # Black Gold 2 (stationary)
    game.set_piece(
        8, 8, Piece(PieceType.KING, Color.BLACK)
    )  # Black King (out of the way)
    game.current_player = Color.BLACK
    game.game_over = False
    game.winner = None
    game.move_count = 10  # Arbitrary move count

    # Move: Black Gold (2,0) -> (1,0) delivering checkmate
    checkmating_move: MoveTuple = (2, 0, 1, 0, False)

    # Act
    _obs, reward, _done, _info = game.make_move(checkmating_move)

    # Assert
    assert game.game_over is True, "Game should be over after checkmate"
    assert game.winner == Color.BLACK, "Black should be the winner"
    assert game.termination_reason == TerminationReason.CHECKMATE.value
    assert reward == pytest.approx(1.0), "Reward for checkmating player should be 1.0"

    # Test get_reward from specified perspective
    assert game.get_reward(perspective_player_color=player_color) == pytest.approx(
        expected_reward
    )


@pytest.mark.parametrize(
    "perspective_player, expected_reward",
    [(Color.BLACK, -1.0), (Color.WHITE, 1.0)],
)
def test_make_move_returns_perspective_specific_reward(
    new_game: ShogiGame, perspective_player: Color, expected_reward: float
):
    game = new_game
    # Setup: White to deliver checkmate to Black
    # BK at (8,0), WG1 at (6,0) (moves to (7,0)), WG2 at (7,1)
    game.board = [[None for _ in range(9)] for _ in range(9)]
    game.hands = {
        Color.BLACK.value: {ptype: 0 for ptype in get_unpromoted_types()},
        Color.WHITE.value: {ptype: 0 for ptype in get_unpromoted_types()},
    }
    game.set_piece(8, 0, Piece(PieceType.KING, Color.BLACK))  # Black King
    game.set_piece(6, 0, Piece(PieceType.GOLD, Color.WHITE))  # White Gold 1 (to move)
    game.set_piece(
        7, 1, Piece(PieceType.GOLD, Color.WHITE)
    )  # White Gold 2 (stationary)
    game.set_piece(
        0, 8, Piece(PieceType.KING, Color.WHITE)
    )  # White King (out of the way)
    game.current_player = Color.WHITE
    game.game_over = False
    game.winner = None
    game.move_count = 10  # Arbitrary move count

    # Move: White Gold (6,0) -> (7,0) delivering checkmate
    checkmating_move: MoveTuple = (6, 0, 7, 0, False)

    # Act
    _obs, reward_for_white, _done, _info = game.make_move(checkmating_move)

    # Assert state after move
    assert game.game_over is True, "Game should be over after checkmate by White"
    assert game.winner == Color.WHITE, "White should be the winner"
    assert game.termination_reason == TerminationReason.CHECKMATE.value
    assert reward_for_white == pytest.approx(
        1.0
    ), "Reward for White (checkmating player) should be 1.0"

    # Assert get_reward from the perspective_player's point of view
    actual_reward_perspective = game.get_reward(
        perspective_player_color=perspective_player
    )
    assert actual_reward_perspective == pytest.approx(expected_reward)


def test_get_reward_ongoing_game(new_game: ShogiGame):  # Use new_game fixture
    """Test get_reward for an ongoing game."""
    game = new_game  # Use the fixture
    # Make a couple of moves to ensure it's not pristine start, but still ongoing
    game.make_move((6, 6, 5, 6, False))  # Black pawn
    game.make_move((2, 2, 3, 2, False))  # White pawn

    # Test for Black's perspective
    reward_black = game.get_reward(perspective_player_color=Color.BLACK)
    assert reward_black == pytest.approx(
        0.0
    ), "Reward for ongoing game (Black) should be 0.0"

    # Test for White's perspective
    reward_white = game.get_reward(perspective_player_color=Color.WHITE)
    assert reward_white == pytest.approx(
        0.0
    ), "Reward for ongoing game (White) should be 0.0"


def test_get_reward_no_perspective_raises_error(
    new_game: ShogiGame,
):  # Use new_game fixture
    """Test that get_reward raises ValueError if perspective_player_color is None."""
    game = new_game  # Use the fixture
    with pytest.raises(ValueError) as excinfo:
        game.get_reward()  # Call without perspective_player_color
    assert "perspective_player_color must be provided" in str(excinfo.value)
