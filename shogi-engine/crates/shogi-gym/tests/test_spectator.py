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


class TestSpectatorFromSfen:
    STARTPOS_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

    def test_from_sfen_creates_env(self):
        env = SpectatorEnv.from_sfen(self.STARTPOS_SFEN)
        assert env.current_player == "black"
        assert env.ply == 0
        assert not env.is_over
        assert env.action_space_size == 13_527

    def test_from_sfen_roundtrip(self):
        env1 = SpectatorEnv()
        env1.reset()
        legal = env1.legal_actions()
        env1.step(legal[0])
        sfen = env1.to_sfen()
        env2 = SpectatorEnv.from_sfen(sfen)
        assert env2.to_sfen() == sfen

    def test_from_sfen_empty_move_history(self):
        env = SpectatorEnv.from_sfen(self.STARTPOS_SFEN)
        d = env.to_dict()
        assert len(d["move_history"]) == 0

    def test_from_sfen_playable(self):
        env = SpectatorEnv.from_sfen(self.STARTPOS_SFEN)
        legal = env.legal_actions()
        assert len(legal) == 30
        state = env.step(legal[0])
        assert state["ply"] == 1

    def test_from_sfen_custom_max_ply(self):
        env = SpectatorEnv.from_sfen(self.STARTPOS_SFEN, max_ply=1)
        legal = env.legal_actions()
        env.step(legal[0])
        assert env.is_over

    def test_from_sfen_invalid_raises(self):
        with pytest.raises(ValueError):
            SpectatorEnv.from_sfen("not a valid sfen")

    def test_from_sfen_with_hands(self):
        sfen_with_hands = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b 2P 1"
        env = SpectatorEnv.from_sfen(sfen_with_hands)
        d = env.to_dict()
        assert d["hands"]["black"]["pawn"] == 2


class TestSpectatorEnvGetters:
    """Gap #7: Dedicated tests for SpectatorEnv getter properties."""

    def test_get_observation_shape(self):
        """get_observation returns a 3D numpy array with correct shape."""
        import numpy as np
        env = SpectatorEnv()
        env.reset()
        obs = np.asarray(env.get_observation())
        assert obs.shape == (46, 9, 9), f"expected (46, 9, 9), got {obs.shape}"
        assert obs.dtype == np.float32

    def test_get_observation_nonzero_at_start(self):
        """Start position observation should have non-zero values (pieces on the board)."""
        import numpy as np
        env = SpectatorEnv()
        env.reset()
        obs = np.asarray(env.get_observation())
        assert obs.sum() > 0, "Startpos observation should have non-zero values"

    def test_get_observation_changes_after_step(self):
        """Observation changes after a step."""
        import numpy as np
        env = SpectatorEnv()
        env.reset()
        obs_before = np.asarray(env.get_observation()).copy()
        legal = env.legal_actions()
        env.step(legal[0])
        obs_after = np.asarray(env.get_observation())
        assert not np.array_equal(obs_before, obs_after), "Observation should change after step"

    def test_action_space_size_constant(self):
        """action_space_size is 13,527 regardless of position."""
        env = SpectatorEnv()
        assert env.action_space_size == 13_527
        env.reset()
        assert env.action_space_size == 13_527
        legal = env.legal_actions()
        env.step(legal[0])
        assert env.action_space_size == 13_527

    def test_current_player_alternates(self):
        """current_player flips after each step."""
        env = SpectatorEnv()
        env.reset()
        assert env.current_player == "black"
        legal = env.legal_actions()
        env.step(legal[0])
        assert env.current_player == "white"
        legal = env.legal_actions()
        env.step(legal[0])
        assert env.current_player == "black"

    def test_ply_increments(self):
        """ply increments by 1 after each step."""
        env = SpectatorEnv()
        env.reset()
        assert env.ply == 0
        legal = env.legal_actions()
        env.step(legal[0])
        assert env.ply == 1
        legal = env.legal_actions()
        env.step(legal[0])
        assert env.ply == 2

    def test_is_over_false_at_start(self):
        """is_over is False at the start."""
        env = SpectatorEnv()
        env.reset()
        assert env.is_over is False

    def test_legal_actions_opening_count(self):
        """Opening position should have exactly 30 legal actions."""
        env = SpectatorEnv()
        env.reset()
        legal = env.legal_actions()
        assert len(legal) == 30, f"expected 30, got {len(legal)}"

    def test_legal_actions_all_unique(self):
        """All legal action indices should be unique."""
        env = SpectatorEnv()
        env.reset()
        legal = env.legal_actions()
        assert len(legal) == len(set(legal)), "legal actions should be unique"

    def test_legal_actions_in_range(self):
        """All legal action indices should be in [0, action_space_size)."""
        env = SpectatorEnv()
        env.reset()
        legal = env.legal_actions()
        for action in legal:
            assert 0 <= action < env.action_space_size, f"action {action} out of range"

    def test_reset_restores_initial_state(self):
        """After playing moves, reset() returns to initial state."""
        env = SpectatorEnv()
        env.reset()
        legal = env.legal_actions()
        env.step(legal[0])
        env.step(env.legal_actions()[0])
        env.reset()
        assert env.ply == 0
        assert env.current_player == "black"
        assert not env.is_over
