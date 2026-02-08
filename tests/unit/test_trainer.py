"""Unit tests for keisei.training.trainer: Trainer.perform_ppo_update and related methods."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
