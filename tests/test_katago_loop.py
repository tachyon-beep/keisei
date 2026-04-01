# tests/test_katago_loop.py
"""Integration tests for KataGoTrainingLoop."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from keisei.config import AppConfig, DisplayConfig, ModelConfig, TrainingConfig
from keisei.training.katago_loop import KataGoTrainingLoop


def _make_mock_katago_vecenv(num_envs: int = 2) -> MagicMock:
    """Create a mock VecEnv that returns correct shapes for KataGo mode."""
    mock = MagicMock()
    mock.observation_channels = 50
    mock.action_space_size = 11259
    mock.episodes_completed = 0
    mock.mean_episode_length = 0.0
    mock.truncation_rate = 0.0

    def make_reset_result():
        result = MagicMock()
        result.observations = np.random.randn(num_envs, 50, 9, 9).astype(np.float32)
        result.legal_masks = np.ones((num_envs, 11259), dtype=bool)
        return result

    def make_step_result(actions):
        result = MagicMock()
        result.observations = np.random.randn(num_envs, 50, 9, 9).astype(np.float32)
        result.legal_masks = np.ones((num_envs, 11259), dtype=bool)
        result.rewards = np.zeros(num_envs, dtype=np.float32)
        result.terminated = np.zeros(num_envs, dtype=bool)
        result.truncated = np.zeros(num_envs, dtype=bool)
        result.current_players = np.zeros(num_envs, dtype=np.uint8)
        return result

    mock.reset.side_effect = lambda: make_reset_result()
    mock.step.side_effect = make_step_result
    mock.reset_stats = MagicMock()
    return mock


@pytest.fixture
def katago_config(tmp_path):
    return AppConfig(
        training=TrainingConfig(
            num_games=2,
            max_ply=50,
            algorithm="katago_ppo",
            checkpoint_interval=5,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            algorithm_params={
                "learning_rate": 2e-4,
                "gamma": 0.99,
                "lambda_policy": 1.0,
                "lambda_value": 1.5,
                "lambda_score": 0.02,
                "lambda_entropy": 0.01,
                "score_normalization": 76.0,
                "grad_clip": 1.0,
            },
        ),
        display=DisplayConfig(
            moves_per_minute=0,
            db_path=str(tmp_path / "test.db"),
        ),
        model=ModelConfig(
            display_name="Test-KataGo",
            architecture="se_resnet",
            params={
                "num_blocks": 2,
                "channels": 32,
                "se_reduction": 8,
                "global_pool_channels": 16,
                "policy_channels": 8,
                "value_fc_size": 32,
                "score_fc_size": 16,
                "obs_channels": 50,
            },
        ),
    )


class TestKataGoTrainingLoopInit:
    def test_initialization(self, katago_config):
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        assert loop.num_envs == 2

    def test_model_is_se_resnet(self, katago_config):
        from keisei.training.models.se_resnet import SEResNetModel

        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        base = loop.model.module if hasattr(loop.model, "module") else loop.model
        assert isinstance(base, SEResNetModel)


class TestKataGoTrainingLoopRun:
    def test_run_one_epoch(self, katago_config):
        """Run one epoch of 4 steps — should complete without error."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=4)
        assert loop.epoch == 0
        assert loop.global_step == 4
