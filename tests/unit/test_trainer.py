"""Unit tests for keisei.training.trainer: Trainer methods.

Tests cover perform_ppo_update, _initialize_components, _initialize_game_state,
_finalize_training, and run_training_loop.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs():
    return np.zeros((46, 9, 9), dtype=np.float32)


def _make_trainer_instance():
    """Build a minimal mock that looks like a Trainer for testing perform_ppo_update.

    We don't instantiate the real Trainer because its __init__ has heavy
    side effects (SessionManager, WandB, model loading, etc.).
    """
    trainer = MagicMock()

    # Agent with model
    trainer.agent = MagicMock()
    trainer.agent.get_value.return_value = 0.5
    trainer.agent.learn.return_value = {
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 0.05,
    }
    trainer.agent.model = MagicMock()
    trainer.agent.last_gradient_norm = 0.3

    # Experience buffer
    trainer.experience_buffer = MagicMock()

    # Metrics manager
    trainer.metrics_manager = MagicMock()
    trainer.metrics_manager.global_timestep = 100
    trainer.metrics_manager.history = MagicMock()
    trainer.metrics_manager.format_ppo_metrics.return_value = "PL:0.1 VL:0.2"
    trainer.metrics_manager.format_ppo_metrics_for_logging.return_value = (
        "policy_loss=0.1, value_loss=0.2"
    )

    # WebUI manager (None means no weight tracking)
    trainer.webui_manager = None

    # log_both
    trainer.log_both = MagicMock()

    return trainer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPerformPpoUpdate:
    """Tests for the perform_ppo_update method logic."""

    def test_calls_compute_advantages_and_returns(self):
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        obs = _make_obs()

        # Call the real method on our mock
        Trainer.perform_ppo_update(trainer, obs, trainer.log_both)

        trainer.experience_buffer.compute_advantages_and_returns.assert_called_once()

    def test_calls_agent_learn(self):
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        obs = _make_obs()

        Trainer.perform_ppo_update(trainer, obs, trainer.log_both)

        trainer.agent.learn.assert_called_once_with(trainer.experience_buffer)

    def test_clears_buffer_after_learning(self):
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        obs = _make_obs()

        Trainer.perform_ppo_update(trainer, obs, trainer.log_both)

        trainer.experience_buffer.clear.assert_called_once()

    def test_records_ppo_metrics(self):
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        obs = _make_obs()

        Trainer.perform_ppo_update(trainer, obs, trainer.log_both)

        trainer.metrics_manager.history.add_ppo_data.assert_called_once()
        trainer.metrics_manager.format_ppo_metrics.assert_called_once()

    def test_logs_ppo_update(self):
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        obs = _make_obs()

        Trainer.perform_ppo_update(trainer, obs, trainer.log_both)

        trainer.log_both.assert_called()
        log_call_args = trainer.log_both.call_args[0][0]
        assert "PPO Update" in log_call_args

    def test_skips_when_agent_not_initialized(self):
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.agent = None
        obs = _make_obs()

        Trainer.perform_ppo_update(trainer, obs, trainer.log_both)

        # Should log error but not crash
        trainer.log_both.assert_called()

    def test_skips_when_buffer_not_initialized(self):
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.experience_buffer = None
        obs = _make_obs()

        Trainer.perform_ppo_update(trainer, obs, trainer.log_both)

        trainer.log_both.assert_called()

    def test_stores_gradient_norm(self):
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.last_gradient_norm = 0.0
        obs = _make_obs()

        Trainer.perform_ppo_update(trainer, obs, trainer.log_both)

        assert trainer.last_gradient_norm == 0.3

    def test_weight_tracking_with_webui(self):
        """When webui_manager is not None, weight deltas should be computed."""
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.webui_manager = MagicMock()  # Enable WebUI

        # Create a simple model with named parameters
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        trainer.agent.model.named_parameters.return_value = [("weight", param)]

        obs = _make_obs()
        Trainer.perform_ppo_update(trainer, obs, trainer.log_both)

        # Weight updates should have been computed
        assert isinstance(trainer.last_weight_updates, dict)


# ===========================================================================
# _initialize_components
# ===========================================================================


class TestInitializeComponents:
    """Tests for _initialize_components orchestration."""

    def test_calls_setup_methods_in_order(self):
        """Calls setup_game_components, setup_training_components, setup_step_manager."""
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.setup_manager = MagicMock()
        trainer.env_manager = MagicMock()
        trainer.model_manager = MagicMock()
        trainer.metrics_manager = MagicMock()
        trainer.logger = MagicMock()
        trainer.model_dir = "/tmp/models"
        trainer.args = SimpleNamespace(resume=None)

        # Setup return values
        mock_game = MagicMock()
        mock_mapper = MagicMock()
        trainer.setup_manager.setup_game_components.return_value = (
            mock_game, mock_mapper, 13527, (46, 9, 9)
        )
        mock_model = MagicMock()
        mock_agent = MagicMock()
        mock_buffer = MagicMock()
        trainer.setup_manager.setup_training_components.return_value = (
            mock_model, mock_agent, mock_buffer
        )
        trainer.setup_manager.setup_step_manager.return_value = MagicMock()
        trainer.setup_manager.handle_checkpoint_resume.return_value = False

        Trainer._initialize_components(trainer)

        trainer.setup_manager.setup_game_components.assert_called_once_with(
            trainer.env_manager
        )
        trainer.setup_manager.setup_training_components.assert_called_once_with(
            trainer.model_manager
        )
        trainer.setup_manager.setup_step_manager.assert_called_once()

    def test_stores_game_components(self):
        """Stores game, policy_output_mapper, action_space_size, obs_space_shape."""
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.setup_manager = MagicMock()
        trainer.env_manager = MagicMock()
        trainer.model_manager = MagicMock()
        trainer.metrics_manager = MagicMock()
        trainer.logger = MagicMock()
        trainer.model_dir = "/tmp/models"
        trainer.args = SimpleNamespace(resume=None)

        mock_game = MagicMock()
        mock_mapper = MagicMock()
        trainer.setup_manager.setup_game_components.return_value = (
            mock_game, mock_mapper, 13527, (46, 9, 9)
        )
        trainer.setup_manager.setup_training_components.return_value = (
            MagicMock(), MagicMock(), MagicMock()
        )
        trainer.setup_manager.setup_step_manager.return_value = MagicMock()
        trainer.setup_manager.handle_checkpoint_resume.return_value = False

        Trainer._initialize_components(trainer)

        assert trainer.game is mock_game
        assert trainer.policy_output_mapper is mock_mapper
        assert trainer.action_space_size == 13527
        assert trainer.obs_space_shape == (46, 9, 9)

    def test_propagates_setup_failure(self):
        """Exceptions from setup_manager propagate up."""
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.setup_manager = MagicMock()
        trainer.env_manager = MagicMock()
        trainer.setup_manager.setup_game_components.side_effect = RuntimeError(
            "env failed"
        )

        with pytest.raises(RuntimeError, match="env failed"):
            Trainer._initialize_components(trainer)


# ===========================================================================
# _initialize_game_state
# ===========================================================================


class TestInitializeGameState:
    """Tests for _initialize_game_state."""

    def test_resets_game_and_returns_episode_state(self):
        """Calls env_manager.reset_game and step_manager.reset_episode."""
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.env_manager = MagicMock()
        mock_episode_state = MagicMock()
        trainer.step_manager = MagicMock()
        trainer.step_manager.reset_episode.return_value = mock_episode_state
        trainer.is_train_wandb_active = False
        log_both = MagicMock()

        result = Trainer._initialize_game_state(trainer, log_both)

        trainer.env_manager.reset_game.assert_called_once()
        trainer.step_manager.reset_episode.assert_called_once()
        assert result is mock_episode_state

    def test_raises_when_step_manager_is_none(self):
        """Raises RuntimeError when step_manager is not initialized."""
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.env_manager = MagicMock()
        trainer.step_manager = None
        trainer.is_train_wandb_active = False
        log_both = MagicMock()

        with pytest.raises(RuntimeError, match="Game initialization error"):
            Trainer._initialize_game_state(trainer, log_both)

    @patch("keisei.training.trainer.wandb")
    def test_finishes_wandb_on_reset_failure(self, mock_wandb):
        """Finalizes WandB when game reset fails and wandb is active."""
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.env_manager = MagicMock()
        trainer.env_manager.reset_game.side_effect = RuntimeError("reset failed")
        trainer.step_manager = MagicMock()
        trainer.is_train_wandb_active = True
        mock_wandb.run = MagicMock()
        log_both = MagicMock()

        with pytest.raises(RuntimeError, match="Game initialization error"):
            Trainer._initialize_game_state(trainer, log_both)

        mock_wandb.finish.assert_called_once_with(exit_code=1)


# ===========================================================================
# _finalize_training
# ===========================================================================


class TestFinalizeTraining:
    """Tests for _finalize_training."""

    @patch("keisei.training.trainer.wandb")
    def test_saves_final_model_when_timesteps_completed(self, mock_wandb):
        """Saves final model when all timesteps are completed."""
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.config = SimpleNamespace(
            training=SimpleNamespace(total_timesteps=100)
        )
        trainer.metrics_manager.global_timestep = 100
        trainer.metrics_manager.total_episodes_completed = 50
        trainer.metrics_manager.get_final_stats.return_value = {
            "black_wins": 20, "white_wins": 15, "draws": 15
        }
        trainer.model_manager = MagicMock()
        trainer.model_manager.save_final_model.return_value = (True, "/path/model.pth")
        trainer.model_manager.save_final_checkpoint.return_value = (True, "/path/ckpt.pth")
        trainer.model_dir = "/tmp/models"
        trainer.run_name = "test_run"
        trainer.is_train_wandb_active = False
        trainer.display_manager = MagicMock()
        trainer.run_artifact_dir = "/tmp/artifacts"
        trainer.webui_manager = None
        mock_wandb.run = None
        log_both = MagicMock()

        Trainer._finalize_training(trainer, log_both)

        trainer.model_manager.save_final_model.assert_called_once()
        trainer.model_manager.save_final_checkpoint.assert_called_once()

    @patch("keisei.training.trainer.wandb")
    def test_logs_warning_when_interrupted(self, mock_wandb):
        """Logs warning when training interrupted before completion."""
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.config = SimpleNamespace(
            training=SimpleNamespace(total_timesteps=100)
        )
        trainer.metrics_manager.global_timestep = 50  # Not complete
        trainer.metrics_manager.total_episodes_completed = 25
        trainer.metrics_manager.get_final_stats.return_value = {
            "black_wins": 10, "white_wins": 10, "draws": 5
        }
        trainer.model_manager = MagicMock()
        trainer.model_manager.save_final_checkpoint.return_value = (True, "/path/ckpt.pth")
        trainer.model_dir = "/tmp/models"
        trainer.run_name = "test_run"
        trainer.is_train_wandb_active = False
        trainer.display_manager = MagicMock()
        trainer.run_artifact_dir = "/tmp/artifacts"
        trainer.webui_manager = None
        mock_wandb.run = None
        log_both = MagicMock()

        Trainer._finalize_training(trainer, log_both)

        # Should NOT call save_final_model (interrupted before total_timesteps)
        trainer.model_manager.save_final_model.assert_not_called()
        # Should still save final checkpoint
        trainer.model_manager.save_final_checkpoint.assert_called_once()

    @patch("keisei.training.trainer.wandb")
    def test_handles_no_agent(self, mock_wandb):
        """Logs error and returns when agent is not initialized."""
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.agent = None
        trainer.config = SimpleNamespace(
            training=SimpleNamespace(total_timesteps=100)
        )
        trainer.metrics_manager.global_timestep = 100
        trainer.metrics_manager.total_episodes_completed = 50
        trainer.is_train_wandb_active = False
        trainer.display_manager = MagicMock()
        trainer.webui_manager = None
        mock_wandb.run = None
        log_both = MagicMock()

        # Should not raise
        Trainer._finalize_training(trainer, log_both)

        # Should log error
        assert any("[ERROR]" in str(c) for c in log_both.call_args_list)

    @patch("keisei.training.trainer.wandb")
    def test_stops_webui_manager(self, mock_wandb):
        """Stops WebUI manager during finalization."""
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.config = SimpleNamespace(
            training=SimpleNamespace(total_timesteps=100)
        )
        trainer.metrics_manager.global_timestep = 100
        trainer.metrics_manager.total_episodes_completed = 50
        trainer.metrics_manager.get_final_stats.return_value = {
            "black_wins": 20, "white_wins": 15, "draws": 15
        }
        trainer.model_manager = MagicMock()
        trainer.model_manager.save_final_model.return_value = (True, "/path/model.pth")
        trainer.model_manager.save_final_checkpoint.return_value = (True, "/path/ckpt.pth")
        trainer.model_dir = "/tmp/models"
        trainer.run_name = "test_run"
        trainer.is_train_wandb_active = False
        trainer.display_manager = MagicMock()
        trainer.run_artifact_dir = "/tmp/artifacts"
        trainer.webui_manager = MagicMock()
        mock_wandb.run = None
        log_both = MagicMock()

        Trainer._finalize_training(trainer, log_both)

        trainer.webui_manager.stop.assert_called_once()

    @patch("keisei.training.trainer.wandb")
    def test_finalizes_wandb_session(self, mock_wandb):
        """Finalizes WandB session when active."""
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.config = SimpleNamespace(
            training=SimpleNamespace(total_timesteps=100)
        )
        trainer.metrics_manager.global_timestep = 100
        trainer.metrics_manager.total_episodes_completed = 50
        trainer.metrics_manager.get_final_stats.return_value = {
            "black_wins": 20, "white_wins": 15, "draws": 15
        }
        trainer.model_manager = MagicMock()
        trainer.model_manager.save_final_model.return_value = (True, "/path/model.pth")
        trainer.model_manager.save_final_checkpoint.return_value = (True, "/path/ckpt.pth")
        trainer.model_dir = "/tmp/models"
        trainer.run_name = "test_run"
        trainer.is_train_wandb_active = True
        trainer.session_manager = MagicMock()
        trainer.display_manager = MagicMock()
        trainer.run_artifact_dir = "/tmp/artifacts"
        trainer.webui_manager = None
        mock_wandb.run = MagicMock()  # WandB run is active
        log_both = MagicMock()

        Trainer._finalize_training(trainer, log_both)

        trainer.session_manager.finalize_session.assert_called_once()

    @patch("keisei.training.trainer.wandb")
    def test_handles_save_failure(self, mock_wandb):
        """Continues finalization even when save_final_model fails."""
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.config = SimpleNamespace(
            training=SimpleNamespace(total_timesteps=100)
        )
        trainer.metrics_manager.global_timestep = 100
        trainer.metrics_manager.total_episodes_completed = 50
        trainer.metrics_manager.get_final_stats.return_value = {
            "black_wins": 20, "white_wins": 15, "draws": 15
        }
        trainer.model_manager = MagicMock()
        trainer.model_manager.save_final_model.return_value = (False, None)
        trainer.model_manager.save_final_checkpoint.return_value = (True, "/path/ckpt.pth")
        trainer.model_dir = "/tmp/models"
        trainer.run_name = "test_run"
        trainer.is_train_wandb_active = False
        trainer.display_manager = MagicMock()
        trainer.run_artifact_dir = "/tmp/artifacts"
        trainer.webui_manager = None
        mock_wandb.run = None
        log_both = MagicMock()

        # Should not raise
        Trainer._finalize_training(trainer, log_both)

        # Checkpoint save should still be attempted
        trainer.model_manager.save_final_checkpoint.assert_called_once()


# ===========================================================================
# _log_run_info
# ===========================================================================


class TestLogRunInfo:
    """Tests for _log_run_info delegation."""

    def test_delegates_to_setup_manager(self):
        """Calls setup_manager.log_run_info with correct args."""
        from keisei.training.trainer import Trainer

        trainer = _make_trainer_instance()
        trainer.setup_manager = MagicMock()
        trainer.session_manager = MagicMock()
        trainer.model_manager = MagicMock()
        trainer.metrics_manager = MagicMock()
        log_both = MagicMock()

        Trainer._log_run_info(trainer, log_both)

        trainer.setup_manager.log_run_info.assert_called_once_with(
            trainer.session_manager,
            trainer.model_manager,
            trainer.agent,
            trainer.metrics_manager,
            log_both,
        )
