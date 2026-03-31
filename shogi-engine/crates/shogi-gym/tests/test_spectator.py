"""Tests for SpectatorEnv Python bindings."""
import pytest
from shogi_gym import SpectatorEnv


class TestSpectatorEnv:
    def test_create(self):
        env = SpectatorEnv()
        assert env.action_space_size == 13_527
        assert env.current_player == "black"
        assert env.ply == 0
        assert not env.is_over

    def test_reset(self):
        env = SpectatorEnv()
        state = env.reset()
        assert state["current_player"] == "black"
        assert state["ply"] == 0
        assert state["is_over"] is False
        assert state["result"] == "in_progress"
        assert len(state["board"]) == 81

    def test_step(self):
        env = SpectatorEnv()
        env.reset()
        legal = env.legal_actions()
        assert len(legal) > 0
        state = env.step(legal[0])
        assert state["ply"] == 1
        assert state["current_player"] == "white"

    def test_no_auto_reset(self):
        """SpectatorEnv should NOT auto-reset on game end."""
        env = SpectatorEnv(max_ply=1)
        env.reset()
        legal = env.legal_actions()
        env.step(legal[0])
        assert env.is_over, "game should be over after max_ply=1"
        assert env.ply == 1

    def test_step_after_game_over_raises(self):
        """Stepping a finished game should raise RuntimeError."""
        env = SpectatorEnv(max_ply=1)
        env.reset()
        legal = env.legal_actions()
        env.step(legal[0])  # truncated at ply=1
        assert env.is_over
        with pytest.raises(RuntimeError, match="game is already over"):
            env.step(0)

    def test_to_dict(self):
        env = SpectatorEnv()
        env.reset()
        d = env.to_dict()
        assert "board" in d
        assert "hands" in d
        assert "sfen" in d
        assert "move_history" in d

    def test_to_sfen(self):
        env = SpectatorEnv()
        env.reset()
        sfen = env.to_sfen()
        assert "lnsgkgsnl" in sfen.lower()

    def test_move_history(self):
        env = SpectatorEnv()
        env.reset()
        legal = env.legal_actions()
        env.step(legal[0])
        d = env.to_dict()
        assert len(d["move_history"]) == 1
        assert "action" in d["move_history"][0]
        assert "notation" in d["move_history"][0]
