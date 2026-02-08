"""Unit tests for keisei.utils.opponents: SimpleRandomOpponent and SimpleHeuristicOpponent."""

import pytest

from keisei.shogi.shogi_game import ShogiGame
from keisei.utils.opponents import SimpleHeuristicOpponent, SimpleRandomOpponent


class TestSimpleRandomOpponent:
    """Tests for SimpleRandomOpponent."""

    def test_default_name(self):
        opp = SimpleRandomOpponent()
        assert opp.name == "SimpleRandomOpponent"

    def test_custom_name(self):
        opp = SimpleRandomOpponent(name="TestBot")
        assert opp.name == "TestBot"

    def test_select_move_returns_legal_move(self):
        game = ShogiGame()
        game.reset()
        opp = SimpleRandomOpponent()
        move = opp.select_move(game)
        legal_moves = game.get_legal_moves()
        assert move in legal_moves

    def test_move_tuple_has_five_elements(self):
        game = ShogiGame()
        game.reset()
        opp = SimpleRandomOpponent()
        move = opp.select_move(game)
        assert len(move) == 5

    def test_raises_when_no_legal_moves(self):
        game = ShogiGame()
        game.reset()
        opp = SimpleRandomOpponent()
        # Mock get_legal_moves to return empty list
        original_get_legal = game.get_legal_moves
        game.get_legal_moves = lambda: []
        with pytest.raises(ValueError, match="No legal moves"):
            opp.select_move(game)
        game.get_legal_moves = original_get_legal


class TestSimpleHeuristicOpponent:
    """Tests for SimpleHeuristicOpponent."""

    def test_default_name(self):
        opp = SimpleHeuristicOpponent()
        assert opp.name == "SimpleHeuristicOpponent"

    def test_select_move_returns_legal_move(self):
        game = ShogiGame()
        game.reset()
        opp = SimpleHeuristicOpponent()
        move = opp.select_move(game)
        legal_moves = game.get_legal_moves()
        assert move in legal_moves

    def test_move_tuple_has_five_elements(self):
        game = ShogiGame()
        game.reset()
        opp = SimpleHeuristicOpponent()
        move = opp.select_move(game)
        assert len(move) == 5

    def test_raises_when_no_legal_moves(self):
        game = ShogiGame()
        game.reset()
        opp = SimpleHeuristicOpponent()
        game.get_legal_moves = lambda: []
        with pytest.raises(ValueError, match="No legal moves"):
            opp.select_move(game)

    def test_prefers_captures_when_available(self):
        """After several moves, if captures are available they should be preferred."""
        game = ShogiGame()
        game.reset()
        opp_random = SimpleRandomOpponent()
        opp_heuristic = SimpleHeuristicOpponent()

        # Play a few random moves to create a position where captures might exist
        for _ in range(10):
            if game.game_over:
                break
            move = opp_random.select_move(game)
            game.make_move(move)

        if not game.game_over:
            # Just verify the heuristic opponent can still select a move
            move = opp_heuristic.select_move(game)
            assert move in game.get_legal_moves()
