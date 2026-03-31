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
        env = SpectatorEnv(max_ply=2)
        env.reset()
        for _ in range(10):
            if env.is_over:
                break
            legal = env.legal_actions()
            env.step(legal[0])
        if env.is_over:
            assert env.ply <= 2

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
