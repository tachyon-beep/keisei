"""Integration tests: environment setup pipeline.

Verifies that EnvManager and SetupManager correctly create and configure
game environments, policy mappers, and validate action space consistency.
"""

import numpy as np
import pytest
import torch

from keisei.config_schema import AppConfig, EnvConfig
from keisei.shogi.shogi_game import ShogiGame
from keisei.training.env_manager import EnvManager
from keisei.training.setup_manager import SetupManager
from keisei.utils.utils import PolicyOutputMapper


# ---------------------------------------------------------------------------
# EnvManager setup
# ---------------------------------------------------------------------------


class TestEnvManagerSetup:
    """EnvManager creates and configures game environment correctly."""

    def test_setup_environment_returns_game_and_mapper(self, integration_config):
        """setup_environment returns a valid ShogiGame and PolicyOutputMapper."""
        env_mgr = EnvManager(integration_config)
        game, mapper = env_mgr.setup_environment()

        assert isinstance(game, ShogiGame)
        assert isinstance(mapper, PolicyOutputMapper)

    def test_action_space_matches_config(self, integration_config):
        """Action space size from mapper matches config.env.num_actions_total."""
        env_mgr = EnvManager(integration_config)
        env_mgr.setup_environment()

        assert env_mgr.action_space_size == integration_config.env.num_actions_total
        assert env_mgr.action_space_size == 13527

    def test_observation_space_shape_is_correct(self, integration_config):
        """Observation space shape matches (input_channels, 9, 9)."""
        env_mgr = EnvManager(integration_config)
        env_mgr.setup_environment()

        expected_shape = (integration_config.env.input_channels, 9, 9)
        assert env_mgr.obs_space_shape == expected_shape

    def test_game_produces_valid_initial_observation(self, integration_config):
        """The created game produces a valid initial observation."""
        env_mgr = EnvManager(integration_config)
        game, _ = env_mgr.setup_environment()
        obs = game.get_observation()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (46, 9, 9)
        assert np.isfinite(obs).all()

    def test_validate_environment_passes(self, integration_config):
        """validate_environment returns True after successful setup."""
        env_mgr = EnvManager(integration_config)
        env_mgr.setup_environment()

        assert env_mgr.validate_environment() is True

    def test_action_space_mismatch_raises(self, tmp_path):
        """Mismatched num_actions_total raises ValueError during setup."""
        bad_config = AppConfig(
            env=EnvConfig(
                device="cpu",
                input_channels=46,
                num_actions_total=9999,  # Wrong number
                seed=42,
            ),
            training=pytest.importorskip("keisei.config_schema").TrainingConfig(
                total_timesteps=100,
                steps_per_epoch=16,
                ppo_epochs=2,
                minibatch_size=8,
                enable_torch_compile=False,
            ),
            evaluation=pytest.importorskip("keisei.config_schema").EvaluationConfig(),
            logging=pytest.importorskip("keisei.config_schema").LoggingConfig(
                log_file=str(tmp_path / "log.txt"),
                model_dir=str(tmp_path / "models"),
            ),
            wandb=pytest.importorskip("keisei.config_schema").WandBConfig(enabled=False),
            parallel=pytest.importorskip("keisei.config_schema").ParallelConfig(enabled=False),
        )

        env_mgr = EnvManager(bad_config)
        with pytest.raises(RuntimeError, match="Action space mismatch"):
            env_mgr.setup_environment()


# ---------------------------------------------------------------------------
# SetupManager component creation
# ---------------------------------------------------------------------------


class TestSetupManagerComponents:
    """SetupManager wires env -> game -> agent -> step_manager correctly."""

    def test_setup_game_components(self, integration_config):
        """setup_game_components returns game, mapper, action size, obs shape."""
        env_mgr = EnvManager(integration_config)
        setup_mgr = SetupManager(integration_config, torch.device("cpu"))

        game, mapper, action_size, obs_shape = setup_mgr.setup_game_components(env_mgr)

        assert isinstance(game, ShogiGame)
        assert isinstance(mapper, PolicyOutputMapper)
        assert action_size == 13527
        assert obs_shape == (46, 9, 9)
