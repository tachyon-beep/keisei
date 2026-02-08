"""Unit tests for keisei.training.training_loop_manager: TrainingLoopManager."""

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
import torch

from keisei.training.step_manager import EpisodeState, StepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs():
    return np.zeros((46, 9, 9), dtype=np.float32)


def _make_obs_tensor():
    return torch.zeros(1, 46, 9, 9)


def _make_episode_state():
    return EpisodeState(
        current_obs=_make_obs(),
        current_obs_tensor=_make_obs_tensor(),
        episode_reward=0.0,
        episode_length=0,
    )


def _make_successful_step_result(done=False):
    return StepResult(
        next_obs=_make_obs(),
        next_obs_tensor=_make_obs_tensor(),
        reward=1.0,
        done=done,
        info={},
        selected_move=(0, 0, 1, 0, False),
        policy_index=0,
        log_prob=0.0,
        value_pred=0.0,
        success=True,
    )


def _make_failed_step_result():
    return StepResult(
        next_obs=_make_obs(),
        next_obs_tensor=_make_obs_tensor(),
        reward=0.0,
        done=False,
        info={},
        selected_move=None,
        policy_index=0,
        log_prob=0.0,
        value_pred=0.0,
        success=False,
        error_message="test error",
    )


def _make_trainer(total_timesteps=100, steps_per_epoch=10, render_every_steps=1):
    """Build a minimal mock trainer with all required sub-manager attributes."""
    trainer = MagicMock()
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(
            total_timesteps=total_timesteps,
            steps_per_epoch=steps_per_epoch,
            render_every_steps=render_every_steps,
            rich_display_update_interval_seconds=0.2,
        ),
        parallel=SimpleNamespace(enabled=False),
    )
    trainer.metrics_manager = MagicMock()
    trainer.metrics_manager.global_timestep = 0
    trainer.metrics_manager.total_episodes_completed = 0
    trainer.metrics_manager.black_wins = 0
    trainer.metrics_manager.white_wins = 0
    trainer.metrics_manager.draws = 0
    trainer.metrics_manager.pending_progress_updates = {}

    trainer.step_manager = MagicMock()
    trainer.step_manager.reset_episode.return_value = _make_episode_state()
    trainer.step_manager.execute_step.return_value = _make_successful_step_result()
    trainer.step_manager.update_episode_state.return_value = _make_episode_state()
    trainer.step_manager.move_history = []

    trainer.agent = MagicMock()
    trainer.experience_buffer = MagicMock()
    trainer.display = MagicMock()
    trainer.callbacks = []
    trainer.callback_manager = MagicMock()
    trainer.callback_manager.has_async_callbacks.return_value = False
    trainer.log_both = MagicMock()
    trainer.policy_output_mapper = MagicMock()
    trainer.webui_manager = None

    return trainer


def _make_training_loop_manager(trainer):
    """Create a TrainingLoopManager from a mock trainer, bypassing __init__ import chain."""
    from keisei.training.training_loop_manager import TrainingLoopManager

    tlm = TrainingLoopManager.__new__(TrainingLoopManager)
    tlm.trainer = trainer
    tlm.config = trainer.config
    tlm.agent = trainer.agent
    tlm.buffer = trainer.experience_buffer
    tlm.step_manager = trainer.step_manager
    tlm.display = trainer.display
    tlm.callbacks = trainer.callbacks
    tlm.current_epoch = 0
    tlm.episode_state = None
    tlm.last_time_for_sps = 0.0
    tlm.steps_since_last_time_for_sps = 0
    tlm.last_display_update_time = 0.0
    tlm.parallel_manager = None
    return tlm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSetInitialEpisodeState:
    """Tests for set_initial_episode_state."""

    def test_stores_episode_state(self):
        trainer = _make_trainer()
        tlm = _make_training_loop_manager(trainer)
        state = _make_episode_state()
        tlm.set_initial_episode_state(state)
        assert tlm.episode_state is state


class TestRunPreconditions:
    """Tests for run() precondition checks."""

    def test_raises_if_log_both_not_set(self):
        trainer = _make_trainer()
        trainer.log_both = None
        tlm = _make_training_loop_manager(trainer)
        with pytest.raises(RuntimeError, match="log_both"):
            tlm.run()

    def test_raises_if_episode_state_not_set(self):
        trainer = _make_trainer()
        tlm = _make_training_loop_manager(trainer)
        # episode_state is None by default
        with pytest.raises(RuntimeError, match="episode state"):
            tlm.run()


class TestRunEpochSequential:
    """Tests for _run_epoch_sequential."""

    def test_collects_steps_per_epoch_steps(self):
        trainer = _make_trainer(total_timesteps=100, steps_per_epoch=5)
        # Make global_timestep increment past total_timesteps eventually
        trainer.metrics_manager.global_timestep = 0
        tlm = _make_training_loop_manager(trainer)
        tlm.episode_state = _make_episode_state()

        count = tlm._run_epoch_sequential(trainer.log_both)
        assert count == 5

    def test_stops_when_total_timesteps_reached(self):
        trainer = _make_trainer(total_timesteps=3, steps_per_epoch=10)
        trainer.metrics_manager.global_timestep = 3
        tlm = _make_training_loop_manager(trainer)
        tlm.episode_state = _make_episode_state()

        count = tlm._run_epoch_sequential(trainer.log_both)
        assert count == 0  # Should not collect any steps


class TestHandleSuccessfulStep:
    """Tests for _handle_successful_step."""

    def test_increments_via_update_episode_state(self):
        trainer = _make_trainer()
        tlm = _make_training_loop_manager(trainer)
        state = _make_episode_state()
        step_result = _make_successful_step_result(done=False)

        new_state = tlm._handle_successful_step(state, step_result, trainer.log_both)
        trainer.step_manager.update_episode_state.assert_called_once_with(
            state, step_result
        )

    def test_handles_episode_end_when_done(self):
        trainer = _make_trainer()
        trainer.step_manager.handle_episode_end.return_value = (
            _make_episode_state(),
            "black",
        )
        tlm = _make_training_loop_manager(trainer)
        state = _make_episode_state()
        step_result = _make_successful_step_result(done=True)

        # update_episode_state returns a state with episode_length=0
        updated_state = _make_episode_state()
        updated_state.episode_length = 10
        trainer.step_manager.update_episode_state.return_value = updated_state

        new_state = tlm._handle_successful_step(state, step_result, trainer.log_both)
        trainer.step_manager.handle_episode_end.assert_called_once()
        trainer.metrics_manager.update_episode_stats.assert_called_once()


class TestProcessStepAndHandleEpisode:
    """Tests for _process_step_and_handle_episode."""

    def test_returns_false_when_total_timesteps_reached(self):
        trainer = _make_trainer(total_timesteps=100)
        trainer.metrics_manager.global_timestep = 100
        tlm = _make_training_loop_manager(trainer)
        tlm.episode_state = _make_episode_state()

        assert tlm._process_step_and_handle_episode(trainer.log_both) is False

    def test_resets_on_failed_step(self):
        trainer = _make_trainer()
        trainer.step_manager.execute_step.return_value = _make_failed_step_result()
        tlm = _make_training_loop_manager(trainer)
        tlm.episode_state = _make_episode_state()

        result = tlm._process_step_and_handle_episode(trainer.log_both)
        assert result is True  # Should continue
        trainer.step_manager.reset_episode.assert_called_once()

    def test_increments_global_timestep_on_success(self):
        trainer = _make_trainer()
        tlm = _make_training_loop_manager(trainer)
        tlm.episode_state = _make_episode_state()

        tlm._process_step_and_handle_episode(trainer.log_both)
        trainer.metrics_manager.increment_timestep.assert_called_once()


class TestUpdateDisplayIfNeeded:
    """Tests for _update_display_if_needed throttling."""

    def test_respects_throttle_interval(self):
        trainer = _make_trainer()
        tlm = _make_training_loop_manager(trainer)
        # Set last update time to now so it won't trigger
        import time

        tlm.last_display_update_time = time.time() + 100  # Far in the future
        tlm._update_display_if_needed()
        trainer.display.update_progress.assert_not_called()
