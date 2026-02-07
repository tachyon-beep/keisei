"""Tests for reward calculation from various perspectives using real ShogiGame objects."""

import random

import pytest

from keisei.shogi.shogi_core_definitions import Color
from keisei.shogi.shogi_game import ShogiGame


# ---------------------------------------------------------------------------
# Ongoing game reward tests
# ---------------------------------------------------------------------------


class TestOngoingGameRewards:
    """Rewards during an ongoing (non-terminal) game."""

    def test_get_reward_returns_zero_for_both_perspectives_when_ongoing(self):
        """get_reward returns 0.0 for both Black and White when the game is still in progress."""
        game = ShogiGame()
        assert not game.game_over
        assert game.get_reward(Color.BLACK) == 0.0
        assert game.get_reward(Color.WHITE) == 0.0

    def test_make_move_returns_zero_reward_when_not_over(self):
        """make_move returns reward=0.0 when the game does not end."""
        game = ShogiGame()
        legal_moves = game.get_legal_moves()
        assert len(legal_moves) > 0
        _, reward, done, _ = game.make_move(legal_moves[0])
        assert not done
        assert reward == 0.0

    def test_make_move_info_has_ongoing_reason_when_not_over(self):
        """make_move info dict has 'Game ongoing' as reason when the game is not over."""
        game = ShogiGame()
        legal_moves = game.get_legal_moves()
        _, _, done, info = game.make_move(legal_moves[0])
        assert not done
        assert info["reason"] == "Game ongoing"


# ---------------------------------------------------------------------------
# Win / loss reward tests
# ---------------------------------------------------------------------------


class TestWinLossRewards:
    """Rewards when a game has been won or lost."""

    @staticmethod
    def _play_random_game(seed: int = 42, max_moves: int = 300) -> ShogiGame:
        """Play a random game to completion and return the finished game."""
        game = ShogiGame(max_moves_per_game=max_moves)
        rng = random.Random(seed)
        while not game.game_over:
            legal = game.get_legal_moves()
            if not legal:
                break
            move = rng.choice(legal)
            game.make_move(move)
        return game

    def test_winner_gets_positive_reward(self):
        """When there is a winner, get_reward returns 1.0 from the winner's perspective."""
        game = self._play_random_game()
        if game.winner is not None:
            assert game.get_reward(game.winner) == 1.0
        else:
            pytest.skip("Random game ended in a draw; winner-specific test not applicable")

    def test_loser_gets_negative_reward(self):
        """When there is a winner, get_reward returns -1.0 from the loser's perspective."""
        game = self._play_random_game()
        if game.winner is not None:
            loser = game.winner.opponent()
            assert game.get_reward(loser) == -1.0
        else:
            pytest.skip("Random game ended in a draw; loser-specific test not applicable")

    def test_make_move_checkmate_returns_positive_reward_for_mover(self):
        """The make_move that causes checkmate returns reward=1.0 for the player who moved."""
        # Play a random game, tracking the last move return values
        game = ShogiGame(max_moves_per_game=500)
        rng = random.Random(42)
        last_result = None
        while not game.game_over:
            legal = game.get_legal_moves()
            if not legal:
                break
            move = rng.choice(legal)
            last_result = game.make_move(move)
        if game.winner is not None and last_result is not None:
            _, reward, done, info = last_result
            assert done is True
            assert reward == 1.0
            assert "winner" in info
        else:
            pytest.skip("Random game did not end in checkmate")

    def test_make_move_info_contains_winner_name_on_checkmate(self):
        """When the game ends by checkmate, info dict contains the winner's name."""
        game = self._play_random_game()
        if game.winner is not None:
            # Replay to get the info dict from the last move
            game2 = ShogiGame(max_moves_per_game=500)
            rng = random.Random(42)
            info = None
            while not game2.game_over:
                legal = game2.get_legal_moves()
                if not legal:
                    break
                move = rng.choice(legal)
                _, _, done, info = game2.make_move(move)
            assert info is not None
            assert "winner" in info
            assert info["winner"] in ("BLACK", "WHITE")
        else:
            pytest.skip("Random game ended in a draw; winner info test not applicable")


# ---------------------------------------------------------------------------
# Draw reward tests
# ---------------------------------------------------------------------------


class TestDrawRewards:
    """Rewards when a game ends in a draw."""

    def test_draw_by_max_moves_returns_zero_for_both(self):
        """When a game ends by max moves, get_reward returns 0.0 for both perspectives."""
        game = ShogiGame(max_moves_per_game=10)
        rng = random.Random(99)
        while not game.game_over:
            legal = game.get_legal_moves()
            if not legal:
                break
            move = rng.choice(legal)
            game.make_move(move)
        if game.game_over and game.winner is None:
            assert game.get_reward(Color.BLACK) == 0.0
            assert game.get_reward(Color.WHITE) == 0.0
        else:
            pytest.skip("Game did not end in a draw with this seed/max_moves")

    def test_get_reward_zero_when_game_over_with_no_winner(self):
        """get_reward returns 0.0 when game_over is True but winner is None (draw)."""
        game = ShogiGame()
        # Manually force a draw state for deterministic testing
        game.game_over = True
        game.winner = None
        assert game.get_reward(Color.BLACK) == 0.0
        assert game.get_reward(Color.WHITE) == 0.0


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestRewardErrorHandling:
    """Edge cases and error conditions for get_reward."""

    def test_get_reward_with_none_perspective_raises_value_error(self):
        """get_reward raises ValueError when perspective_player_color is None."""
        game = ShogiGame()
        with pytest.raises(ValueError, match="perspective_player_color must be provided"):
            game.get_reward(None)
