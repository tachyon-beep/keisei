"""Integration tests for the KataGo 50-channel observation generator via VecEnv."""

import numpy as np
import pytest

from shogi_gym import VecEnv


@pytest.fixture
def katago_env():
    """Single-env VecEnv with KataGo observation mode."""
    return VecEnv(num_envs=1, max_ply=100, observation_mode="katago", action_mode="default")


@pytest.fixture
def default_env():
    """Single-env VecEnv with default observation mode."""
    return VecEnv(num_envs=1, max_ply=100)


class TestKataGoObservationShape:
    def test_observation_channels(self, katago_env):
        assert katago_env.observation_channels == 50

    def test_reset_observation_shape(self, katago_env):
        result = katago_env.reset()
        obs = np.array(result.observations)
        assert obs.shape == (1, 50, 9, 9)

    def test_step_observation_shape(self, katago_env):
        result = katago_env.reset()
        masks = np.array(result.legal_masks)
        # Pick first legal action
        action = int(np.argmax(masks[0]))
        step_result = katago_env.step([action])
        obs = np.array(step_result.observations)
        assert obs.shape == (1, 50, 9, 9)


class TestKataGoObservationContent:
    def test_channels_0_43_match_default(self, katago_env, default_env):
        """First 44 channels should match the default generator."""
        katago_result = katago_env.reset()
        default_result = default_env.reset()

        katago_obs = np.array(katago_result.observations)[0]  # (50, 9, 9)
        default_obs = np.array(default_result.observations)[0]  # (46, 9, 9)

        np.testing.assert_array_equal(
            katago_obs[:44], default_obs[:44],
            err_msg="Channels 0-43 should be identical between KataGo and default"
        )

    def test_no_repetition_at_startpos(self, katago_env):
        result = katago_env.reset()
        obs = np.array(result.observations)[0]
        # Channels 44-47 should all be zero at startpos
        for ch in range(44, 48):
            assert np.all(obs[ch] == 0.0), f"Channel {ch} should be all zeros at startpos"

    def test_not_in_check_at_startpos(self, katago_env):
        result = katago_env.reset()
        obs = np.array(result.observations)[0]
        assert np.all(obs[48] == 0.0), "Channel 48 should be 0 when not in check"

    def test_reserved_channel_49_zero(self, katago_env):
        result = katago_env.reset()
        obs = np.array(result.observations)[0]
        assert np.all(obs[49] == 0.0), "Channel 49 (reserved) should be all zeros"

    def test_no_nan_in_observation(self, katago_env):
        result = katago_env.reset()
        obs = np.array(result.observations)[0]
        assert not np.any(np.isnan(obs)), "No NaN should be present in observation"


class TestKataGoObservationInvalidMode:
    def test_invalid_observation_mode(self):
        with pytest.raises(ValueError, match="Unknown observation_mode"):
            VecEnv(num_envs=1, max_ply=100, observation_mode="invalid")


class TestDefaultConstructorBackwardCompat:
    """Verify the default constructor (no mode params) still produces old shapes."""

    def test_default_constructor_obs_shape(self):
        env = VecEnv(num_envs=1, max_ply=50)
        assert env.observation_channels == 46
        assert env.action_space_size == 13527
        result = env.reset()
        obs = np.array(result.observations)
        masks = np.array(result.legal_masks)
        assert obs.shape == (1, 46, 9, 9)
        assert masks.shape == (1, 13527)

    def test_default_constructor_step(self):
        env = VecEnv(num_envs=1, max_ply=50)
        result = env.reset()
        masks = np.array(result.legal_masks)
        action = int(np.argmax(masks[0]))
        step_result = env.step([action])
        assert np.array(step_result.observations).shape == (1, 46, 9, 9)
        assert np.array(step_result.legal_masks).shape == (1, 13527)
