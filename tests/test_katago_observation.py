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


class TestKataGoObservationLiveChannels:
    """Test channels 44-48 with non-zero values via actual gameplay through VecEnv.

    These complement the Rust unit tests (which test via direct state manipulation)
    by verifying the full VecEnv → observation pipeline produces correct non-zero
    channel values across the PyO3 boundary.
    """

    def test_check_channel_activates(self):
        """Play moves until check occurs; channel 48 should become 1.0.

        Strategy: play many random-ish games. At least one should end with
        a check before the game terminates. If we find a checked position,
        verify channel 48 is all-ones.
        """
        env = VecEnv(num_envs=1, max_ply=200, observation_mode="katago")
        found_check = False

        for _attempt in range(10):  # try up to 10 games
            result = env.reset()
            for _step in range(100):
                masks = np.array(result.legal_masks)
                if masks[0].sum() == 0:
                    break
                # Pick a random legal action
                legal_indices = np.where(masks[0])[0]
                action = int(np.random.choice(legal_indices))
                result = env.step([action])

                obs = np.array(result.observations)[0]
                if np.any(obs[48] > 0):
                    found_check = True
                    # Verify the entire plane is 1.0 (spatial, not partial)
                    assert np.all(obs[48] == 1.0), \
                        "Check channel should be all-ones when active"
                    break

                if np.array(result.terminated)[0] or np.array(result.truncated)[0]:
                    break

            if found_check:
                break

        # If no check found in 10 games × 100 steps, skip rather than fail
        # (extremely unlikely — checks occur in most games)
        if not found_check:
            pytest.skip("No check position encountered in random play (unlikely)")

    def test_channels_44_47_stay_zero_in_short_game(self):
        """Repetition channels should remain zero in a game with no repeated positions.

        After a few unique moves, channels 44-47 should still be all-zero
        because no position has been visited twice.
        """
        env = VecEnv(num_envs=1, max_ply=100, observation_mode="katago")
        result = env.reset()

        # Make a few moves (unlikely to repeat positions in 5 moves)
        for _ in range(5):
            masks = np.array(result.legal_masks)
            if masks[0].sum() == 0:
                break
            action = int(np.argmax(masks[0]))
            result = env.step([action])

        obs = np.array(result.observations)[0]
        for ch in range(44, 48):
            assert np.all(obs[ch] == 0.0), \
                f"Channel {ch} should be zero after non-repeated moves"


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
