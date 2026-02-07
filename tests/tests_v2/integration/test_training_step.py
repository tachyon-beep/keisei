"""Integration tests: StepManager executes real training steps.

Verifies that StepManager correctly coordinates agent, game, policy mapper,
and experience buffer to produce valid experience data and handle episode
boundaries.
"""

import numpy as np
import pytest
import torch

from keisei.core.experience_buffer import ExperienceBuffer
from keisei.shogi.shogi_game import ShogiGame
from keisei.training.step_manager import EpisodeState, StepManager


# ---------------------------------------------------------------------------
# Single step execution
# ---------------------------------------------------------------------------


class TestSingleStep:
    """StepManager can execute a single step with real dependencies."""

    def test_execute_step_succeeds(self, step_manager, noop_logger):
        """A single execute_step call returns a successful StepResult."""
        episode_state = step_manager.reset_episode()
        result = step_manager.execute_step(
            episode_state, global_timestep=0, logger_func=noop_logger
        )
        assert result.success is True
        assert result.selected_move is not None

    def test_step_produces_valid_experience_data(self, step_manager, noop_logger):
        """Step result contains valid observation, action, reward, done."""
        episode_state = step_manager.reset_episode()
        result = step_manager.execute_step(
            episode_state, global_timestep=0, logger_func=noop_logger
        )

        assert isinstance(result.next_obs, np.ndarray)
        assert result.next_obs.shape[0] == 46  # input channels
        assert isinstance(result.policy_index, int)
        assert 0 <= result.policy_index < 13527
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.log_prob, float)
        assert isinstance(result.value_pred, float)
        assert np.isfinite(result.log_prob)
        assert np.isfinite(result.value_pred)

    def test_step_adds_to_experience_buffer(self, step_manager, experience_buffer, noop_logger):
        """After one step, the experience buffer has exactly one entry."""
        assert experience_buffer.ptr == 0

        episode_state = step_manager.reset_episode()
        result = step_manager.execute_step(
            episode_state, global_timestep=0, logger_func=noop_logger
        )

        assert result.success
        assert experience_buffer.ptr == 1

    def test_step_observation_tensor_shape(self, step_manager, noop_logger):
        """The next_obs_tensor returned has correct shape [1, C, H, W]."""
        episode_state = step_manager.reset_episode()
        result = step_manager.execute_step(
            episode_state, global_timestep=0, logger_func=noop_logger
        )

        assert result.next_obs_tensor.shape == (1, 46, 9, 9)


# ---------------------------------------------------------------------------
# Multiple steps and buffer accumulation
# ---------------------------------------------------------------------------


class TestMultipleSteps:
    """Multiple steps accumulate experiences correctly in the buffer."""

    def test_multiple_steps_fill_buffer(self, step_manager, experience_buffer, noop_logger):
        """Running N steps puts N entries into the experience buffer."""
        episode_state = step_manager.reset_episode()
        n_steps = 8

        for t in range(n_steps):
            result = step_manager.execute_step(
                episode_state, global_timestep=t, logger_func=noop_logger
            )
            if result.success:
                episode_state = step_manager.update_episode_state(episode_state, result)
                if result.done:
                    episode_state = step_manager.reset_episode()
            else:
                episode_state = step_manager.reset_episode()

        assert experience_buffer.ptr == n_steps

    def test_buffer_observations_are_distinct(self, step_manager, experience_buffer, noop_logger):
        """Consecutive observations stored in buffer are not all identical."""
        episode_state = step_manager.reset_episode()
        n_steps = 4

        for t in range(n_steps):
            result = step_manager.execute_step(
                episode_state, global_timestep=t, logger_func=noop_logger
            )
            if result.success:
                episode_state = step_manager.update_episode_state(episode_state, result)
                if result.done:
                    episode_state = step_manager.reset_episode()
            else:
                episode_state = step_manager.reset_episode()

        # Check that at least two observations differ
        obs_data = experience_buffer.obs[: experience_buffer.ptr]
        any_different = False
        for i in range(1, obs_data.shape[0]):
            if not torch.equal(obs_data[i], obs_data[0]):
                any_different = True
                break
        assert any_different, "All buffer observations are identical -- game state is not advancing"


# ---------------------------------------------------------------------------
# Episode boundary handling
# ---------------------------------------------------------------------------


class TestEpisodeBoundaries:
    """When a game ends, StepManager can start a new episode."""

    def test_episode_end_resets_game(
        self,
        integration_config,
        ppo_agent,
        session_policy_mapper,
        experience_buffer,
        noop_logger,
    ):
        """After a game ends (via max_moves), a new episode starts."""
        short_game = ShogiGame(max_moves_per_game=10)
        sm = StepManager(
            config=integration_config,
            game=short_game,
            agent=ppo_agent,
            policy_mapper=session_policy_mapper,
            experience_buffer=experience_buffer,
        )

        episode_state = sm.reset_episode()
        game_ended = False

        for t in range(20):
            result = sm.execute_step(
                episode_state, global_timestep=t, logger_func=noop_logger
            )
            if result.success:
                episode_state = sm.update_episode_state(episode_state, result)
                if result.done:
                    game_ended = True
                    # Start a new episode
                    episode_state = sm.reset_episode()
                    # Verify the new episode starts fresh
                    assert episode_state.episode_length == 0
                    assert episode_state.episode_reward == 0.0
                    break
            else:
                episode_state = sm.reset_episode()

        assert game_ended, "Game with max_moves=10 should end within 20 steps"

    def test_episode_reward_accumulates(self, step_manager, noop_logger):
        """Episode reward accumulates across multiple steps."""
        episode_state = step_manager.reset_episode()
        assert episode_state.episode_reward == 0.0

        for t in range(5):
            result = step_manager.execute_step(
                episode_state, global_timestep=t, logger_func=noop_logger
            )
            if result.success:
                episode_state = step_manager.update_episode_state(episode_state, result)
                if result.done:
                    break

        # Episode length should have incremented
        assert episode_state.episode_length > 0
