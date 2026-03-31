"""Tests for observation generation via VecEnv."""
import numpy as np
from shogi_gym import VecEnv


class TestObservation:
    def test_startpos_has_pieces(self):
        env = VecEnv(num_envs=1, max_ply=100)
        result = env.reset()
        obs = np.asarray(result.observations)[0]
        assert obs[0:8].sum() > 0, "no current player pieces"
        assert obs[14:22].sum() > 0, "no opponent pieces"

    def test_player_indicator_channel(self):
        env = VecEnv(num_envs=1, max_ply=100)
        result = env.reset()
        obs = np.asarray(result.observations)[0]
        assert np.allclose(obs[42], 1.0), "player indicator should be 1.0 for Black"

    def test_reserved_channels_zero(self):
        env = VecEnv(num_envs=1, max_ply=100)
        result = env.reset()
        obs = np.asarray(result.observations)[0]
        assert np.allclose(obs[44], 0.0)
        assert np.allclose(obs[45], 0.0)

    def test_hand_channels_zero_at_start(self):
        env = VecEnv(num_envs=1, max_ply=100)
        result = env.reset()
        obs = np.asarray(result.observations)[0]
        assert np.allclose(obs[28:42], 0.0)

    def test_observation_dtype(self):
        env = VecEnv(num_envs=1, max_ply=100)
        result = env.reset()
        obs = np.asarray(result.observations)
        assert obs.dtype == np.float32
