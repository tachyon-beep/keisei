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

    def test_updates_display_when_interval_passed(self):
        """Display is updated when enough time has passed."""
        trainer = _make_trainer()
        tlm = _make_training_loop_manager(trainer)
        tlm.last_display_update_time = 0.0  # Long time ago
        tlm.last_time_for_sps = 0.0
        tlm._update_display_if_needed()
        trainer.display.update_progress.assert_called_once()

    def test_updates_webui_when_available(self):
        """WebUI manager is updated when available and interval passed."""
        trainer = _make_trainer()
        trainer.webui_manager = MagicMock()
        tlm = _make_training_loop_manager(trainer)
        tlm.last_display_update_time = 0.0
        tlm.last_time_for_sps = 0.0
        tlm._update_display_if_needed()
        trainer.webui_manager.update_progress.assert_called_once()

    def test_passes_extra_updates(self):
        """Extra updates are included in the pending progress updates."""
        trainer = _make_trainer()
        tlm = _make_training_loop_manager(trainer)
        tlm.last_display_update_time = 0.0
        tlm.last_time_for_sps = 0.0
        tlm._update_display_if_needed(extra_updates={"my_key": 42})
        assert "my_key" not in trainer.metrics_manager.pending_progress_updates  # cleared after update


# ---------------------------------------------------------------------------
# New test classes: Init, RunMainLoop, RunEpochDispatcher, RunEpochParallel,
#                   LogEpisodeMetrics, BuildConfigs
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for TrainingLoopManager __init__ with parallel enabled/disabled."""

    def test_parallel_disabled_sets_parallel_manager_none(self):
        """When parallel.enabled=False, parallel_manager is None."""
        trainer = _make_trainer()
        tlm = _make_training_loop_manager(trainer)
        assert tlm.parallel_manager is None

    def test_stores_trainer_references(self):
        """Stores references to config, agent, buffer, step_manager, display."""
        trainer = _make_trainer()
        tlm = _make_training_loop_manager(trainer)
        assert tlm.config is trainer.config
        assert tlm.agent is trainer.agent
        assert tlm.buffer is trainer.experience_buffer
        assert tlm.step_manager is trainer.step_manager
        assert tlm.display is trainer.display

    def test_initial_epoch_is_zero(self):
        """current_epoch starts at 0."""
        trainer = _make_trainer()
        tlm = _make_training_loop_manager(trainer)
        assert tlm.current_epoch == 0


class TestRunMainLoop:
    """Tests for run() main loop behavior."""

    def test_stops_when_total_timesteps_reached(self):
        """Loop exits when global_timestep >= total_timesteps."""
        trainer = _make_trainer(total_timesteps=5, steps_per_epoch=5)
        # Simulate timestep incrementing
        call_count = 0

        def increment_timestep():
            nonlocal call_count
            call_count += 1
            trainer.metrics_manager.global_timestep = call_count

        trainer.metrics_manager.increment_timestep.side_effect = increment_timestep

        tlm = _make_training_loop_manager(trainer)
        tlm.episode_state = _make_episode_state()

        tlm.run()

        # Should have completed at least one epoch
        assert tlm.current_epoch >= 1

    def test_handles_keyboard_interrupt(self):
        """KeyboardInterrupt in epoch is re-raised."""
        trainer = _make_trainer(total_timesteps=100, steps_per_epoch=10)
        tlm = _make_training_loop_manager(trainer)
        tlm.episode_state = _make_episode_state()

        # Make _run_epoch raise KeyboardInterrupt
        tlm._run_epoch = MagicMock(side_effect=KeyboardInterrupt)

        with pytest.raises(KeyboardInterrupt):
            tlm.run()

    def test_handles_runtime_error(self):
        """RuntimeError in epoch is re-raised."""
        trainer = _make_trainer(total_timesteps=100, steps_per_epoch=10)
        tlm = _make_training_loop_manager(trainer)
        tlm.episode_state = _make_episode_state()

        tlm._run_epoch = MagicMock(side_effect=RuntimeError("test error"))

        with pytest.raises(RuntimeError, match="test error"):
            tlm.run()

    def test_calls_ppo_update_after_epoch(self):
        """perform_ppo_update is called after epoch when current_obs is set."""
        # Use total_timesteps > steps_per_epoch so the loop completes an epoch
        # *before* hitting the total limit, triggering the PPO update path.
        trainer = _make_trainer(total_timesteps=15, steps_per_epoch=5)

        call_count = 0

        def increment_timestep():
            nonlocal call_count
            call_count += 1
            trainer.metrics_manager.global_timestep = call_count

        trainer.metrics_manager.increment_timestep.side_effect = increment_timestep

        tlm = _make_training_loop_manager(trainer)
        tlm.episode_state = _make_episode_state()
        tlm.run()

        trainer.perform_ppo_update.assert_called()

    def test_executes_step_callbacks(self):
        """callback_manager.execute_step_callbacks is called during loop."""
        trainer = _make_trainer(total_timesteps=15, steps_per_epoch=5)

        call_count = 0

        def increment_timestep():
            nonlocal call_count
            call_count += 1
            trainer.metrics_manager.global_timestep = call_count

        trainer.metrics_manager.increment_timestep.side_effect = increment_timestep

        tlm = _make_training_loop_manager(trainer)
        tlm.episode_state = _make_episode_state()
        tlm.run()

        trainer.callback_manager.execute_step_callbacks.assert_called()


class TestRunEpochDispatcher:
    """Tests for _run_epoch routing to parallel vs sequential."""

    def test_routes_to_sequential_when_no_parallel(self):
        """When parallel_manager is None, _run_epoch_sequential is called."""
        trainer = _make_trainer()
        tlm = _make_training_loop_manager(trainer)
        tlm.episode_state = _make_episode_state()
        tlm._run_epoch_sequential = MagicMock(return_value=5)

        tlm._run_epoch(trainer.log_both)

        tlm._run_epoch_sequential.assert_called_once_with(trainer.log_both)

    def test_routes_to_parallel_when_parallel_manager_exists(self):
        """When parallel_manager is set and enabled, _run_epoch_parallel is called."""
        trainer = _make_trainer()
        trainer.config.parallel.enabled = True
        tlm = _make_training_loop_manager(trainer)
        tlm.parallel_manager = MagicMock()
        tlm.episode_state = _make_episode_state()
        tlm._run_epoch_parallel = MagicMock(return_value=10)

        tlm._run_epoch(trainer.log_both)

        tlm._run_epoch_parallel.assert_called_once_with(trainer.log_both)

    def test_updates_progress_metric(self):
        """_run_epoch sets steps_collected_this_epoch in pending_progress_updates."""
        trainer = _make_trainer()
        tlm = _make_training_loop_manager(trainer)
        tlm.episode_state = _make_episode_state()

        tlm._run_epoch(trainer.log_both)

        assert "steps_collected_this_epoch" in trainer.metrics_manager.pending_progress_updates


class TestRunEpochParallel:
    """Tests for _run_epoch_parallel."""

    def test_falls_back_to_sequential_when_no_parallel_manager(self):
        """Falls back to sequential when parallel_manager is None."""
        trainer = _make_trainer()
        tlm = _make_training_loop_manager(trainer)
        tlm.parallel_manager = None
        tlm.episode_state = _make_episode_state()
        tlm._run_epoch_sequential = MagicMock(return_value=5)

        result = tlm._run_epoch_parallel(trainer.log_both)

        tlm._run_epoch_sequential.assert_called_once()
        assert result == 5

    def test_collects_experiences_from_parallel_manager(self):
        """Collects experiences from parallel_manager.collect_experiences."""
        trainer = _make_trainer(total_timesteps=100, steps_per_epoch=10)
        tlm = _make_training_loop_manager(trainer)
        mock_pm = MagicMock()
        # Return 10 experiences on first call, then break
        mock_pm.collect_experiences.return_value = 10
        mock_pm.sync_model_if_needed.return_value = False
        tlm.parallel_manager = mock_pm
        tlm.episode_state = _make_episode_state()

        result = tlm._run_epoch_parallel(trainer.log_both)

        assert result == 10
        mock_pm.collect_experiences.assert_called()

    def test_falls_back_on_collection_error(self):
        """Falls back to sequential on max collection errors."""
        trainer = _make_trainer(total_timesteps=100, steps_per_epoch=10)
        tlm = _make_training_loop_manager(trainer)
        mock_pm = MagicMock()
        mock_pm.collect_experiences.side_effect = RuntimeError("worker died")
        mock_pm.sync_model_if_needed.return_value = False
        tlm.parallel_manager = mock_pm
        tlm.episode_state = _make_episode_state()
        tlm._run_epoch_sequential = MagicMock(return_value=10)

        result = tlm._run_epoch_parallel(trainer.log_both)

        # Eventually falls back to sequential
        tlm._run_epoch_sequential.assert_called_once()

    def test_handles_no_buffer(self):
        """Breaks when buffer is None."""
        trainer = _make_trainer(total_timesteps=100, steps_per_epoch=10)
        tlm = _make_training_loop_manager(trainer)
        mock_pm = MagicMock()
        mock_pm.sync_model_if_needed.return_value = False
        tlm.parallel_manager = mock_pm
        tlm.buffer = None
        tlm.episode_state = _make_episode_state()

        result = tlm._run_epoch_parallel(trainer.log_both)

        assert result == 0


class TestLogEpisodeMetrics:
    """Tests for _log_episode_metrics."""

    def test_logs_metrics_for_completed_episode(self):
        """Logs episode length, reward, and win rates."""
        trainer = _make_trainer()
        trainer.metrics_manager.black_wins = 5
        trainer.metrics_manager.white_wins = 3
        trainer.metrics_manager.draws = 2
        trainer.metrics_manager.total_episodes_completed = 10

        tlm = _make_training_loop_manager(trainer)
        state = _make_episode_state()
        state.episode_length = 50
        state.episode_reward = 1.0

        tlm._log_episode_metrics(state, "win")

        trainer.metrics_manager.log_episode_metrics.assert_called_once()
        updates = trainer.metrics_manager.pending_progress_updates
        assert "black_win_rate" in updates
        assert "white_win_rate" in updates
        assert "draw_rate" in updates
        assert updates["black_win_rate"] == 0.5
        assert updates["white_win_rate"] == 0.3
        assert updates["draw_rate"] == 0.2

    def test_zero_total_games_does_not_divide_by_zero(self):
        """No division by zero when total_games is 0."""
        trainer = _make_trainer()
        trainer.metrics_manager.black_wins = 0
        trainer.metrics_manager.white_wins = 0
        trainer.metrics_manager.draws = 0
        trainer.metrics_manager.total_episodes_completed = 0

        tlm = _make_training_loop_manager(trainer)
        state = _make_episode_state()
        state.episode_length = 10
        state.episode_reward = 0.0

        # Should not raise
        tlm._log_episode_metrics(state, "draw")

        updates = trainer.metrics_manager.pending_progress_updates
        assert updates["black_win_rate"] == 0.0
        assert updates["white_win_rate"] == 0.0
        assert updates["draw_rate"] == 0.0

    def test_uses_step_manager_move_history(self):
        """Passes move_history from step_manager to metrics_manager."""
        trainer = _make_trainer()
        trainer.step_manager.move_history = ["e2e4", "e7e5"]
        trainer.metrics_manager.black_wins = 1
        trainer.metrics_manager.white_wins = 0
        trainer.metrics_manager.draws = 0

        tlm = _make_training_loop_manager(trainer)
        state = _make_episode_state()
        state.episode_length = 2
        state.episode_reward = 1.0

        tlm._log_episode_metrics(state, "win")

        call_args = trainer.metrics_manager.log_episode_metrics.call_args
        assert call_args[0][4] == ["e2e4", "e7e5"]  # move_history arg

    def test_none_move_history_when_no_step_manager(self):
        """Passes None move_history when step_manager is None."""
        trainer = _make_trainer()
        trainer.step_manager = None
        trainer.metrics_manager.black_wins = 1
        trainer.metrics_manager.white_wins = 0
        trainer.metrics_manager.draws = 0

        tlm = _make_training_loop_manager(trainer)
        tlm.step_manager = None
        state = _make_episode_state()
        state.episode_length = 2
        state.episode_reward = 1.0

        tlm._log_episode_metrics(state, "win")

        call_args = trainer.metrics_manager.log_episode_metrics.call_args
        assert call_args[0][4] is None  # move_history arg


class TestBuildConfigs:
    """Tests for _build_env_config and _build_model_config."""

    def test_build_env_config_structure(self):
        """_build_env_config returns dict with required keys."""
        trainer = _make_trainer()
        trainer.config.env = SimpleNamespace(
            input_channels=46,
            num_actions_total=13527,
            seed=42,
        )
        trainer.config.training.input_features = "core46"
        tlm = _make_training_loop_manager(trainer)

        config = tlm._build_env_config()
        assert config["device"] == "cpu"
        assert config["input_channels"] == 46
        assert config["num_actions_total"] == 13527
        assert config["seed"] == 42
        assert config["input_features"] == "core46"

    def test_build_model_config_structure(self):
        """_build_model_config returns dict with required keys."""
        trainer = _make_trainer()
        trainer.config.training.model_type = "resnet"
        trainer.config.training.tower_depth = 4
        trainer.config.training.tower_width = 128
        trainer.config.training.se_ratio = 0.25
        trainer.config.env = SimpleNamespace(
            input_channels=46,
            num_actions_total=13527,
            seed=42,
        )
        tlm = _make_training_loop_manager(trainer)

        config = tlm._build_model_config()
        assert config["model_type"] == "resnet"
        assert config["tower_depth"] == 4
        assert config["tower_width"] == 128
        assert config["se_ratio"] == 0.25
        assert config["obs_shape"] == (46, 9, 9)
        assert config["num_actions"] == 13527


class TestHandleDisplayUpdates:
    """Tests for _handle_display_updates."""

    def test_refreshes_dashboard_at_render_interval(self):
        """refresh_dashboard_panels is called at render_every_steps intervals."""
        trainer = _make_trainer(render_every_steps=5)
        trainer.metrics_manager.global_timestep = 10  # 10 % 5 == 0
        tlm = _make_training_loop_manager(trainer)
        tlm.last_display_update_time = 0.0
        tlm.last_time_for_sps = 0.0

        tlm._handle_display_updates()

        trainer.display.refresh_dashboard_panels.assert_called_once()

    def test_skips_dashboard_refresh_off_interval(self):
        """refresh_dashboard_panels is NOT called when off-interval."""
        trainer = _make_trainer(render_every_steps=5)
        trainer.metrics_manager.global_timestep = 7  # 7 % 5 != 0
        tlm = _make_training_loop_manager(trainer)
        # Use a far-future time to prevent _update_display_if_needed from triggering
        import time as time_mod

        tlm.last_display_update_time = time_mod.time() + 100
        tlm.last_time_for_sps = time_mod.time()

        tlm._handle_display_updates()

        trainer.display.refresh_dashboard_panels.assert_not_called()

    def test_refreshes_webui_at_render_interval(self):
        """WebUI refresh_dashboard_panels is called at render intervals."""
        trainer = _make_trainer(render_every_steps=5)
        trainer.metrics_manager.global_timestep = 10
        trainer.webui_manager = MagicMock()
        tlm = _make_training_loop_manager(trainer)
        tlm.last_display_update_time = 0.0
        tlm.last_time_for_sps = 0.0

        tlm._handle_display_updates()

        trainer.webui_manager.refresh_dashboard_panels.assert_called_once()


class TestProcessStepEpisodeStateNone:
    """Tests for _process_step_and_handle_episode with None episode state."""

    def test_resets_when_episode_state_is_none(self):
        """Resets episode when episode_state is None."""
        trainer = _make_trainer()
        tlm = _make_training_loop_manager(trainer)
        tlm.episode_state = None

        result = tlm._process_step_and_handle_episode(trainer.log_both)

        assert result is True
        trainer.step_manager.reset_episode.assert_called_once()

    def test_raises_when_step_manager_none_and_episode_state_none(self):
        """Raises RuntimeError when both step_manager and episode_state are None."""
        trainer = _make_trainer()
        tlm = _make_training_loop_manager(trainer)
        tlm.episode_state = None
        tlm.step_manager = None

        with pytest.raises(RuntimeError, match="StepManager"):
            tlm._process_step_and_handle_episode(trainer.log_both)
