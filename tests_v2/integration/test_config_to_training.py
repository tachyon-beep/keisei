"""Integration tests: config -> component creation pipeline.

Verifies that Pydantic AppConfig flows correctly to component creation,
overrides are respected, and invalid config is rejected.
"""

import pytest
import torch

from keisei.config_schema import (
    AppConfig,
    DisplayConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    ParallelConfig,
    TrainingConfig,
    WandBConfig,
    WebUIConfig,
)
from keisei.core.neural_network import ActorCritic
from keisei.core.ppo_agent import PPOAgent
from keisei.training.env_manager import EnvManager
from keisei.training.setup_manager import SetupManager
from keisei.utils.utils import PolicyOutputMapper


# ---------------------------------------------------------------------------
# Config -> component creation
# ---------------------------------------------------------------------------


class TestConfigCreatesComponents:
    """AppConfig correctly drives component creation."""

    def test_config_creates_env_and_agent(self, integration_config, session_policy_mapper):
        """Config -> EnvManager -> game + mapper; then model + agent can be built."""
        env_mgr = EnvManager(integration_config)
        game, mapper = env_mgr.setup_environment()

        model = ActorCritic(
            input_channels=integration_config.env.input_channels,
            num_actions_total=mapper.get_total_actions(),
        )
        agent = PPOAgent(
            model=model,
            config=integration_config,
            device=torch.device(integration_config.env.device),
        )

        # Agent should have the right hyperparameters from config
        assert agent.gamma == integration_config.training.gamma
        assert agent.clip_epsilon == integration_config.training.clip_epsilon
        assert agent.ppo_epochs == integration_config.training.ppo_epochs
        assert agent.minibatch_size == integration_config.training.minibatch_size

    def test_config_overrides_learning_rate(self, tmp_path, session_policy_mapper):
        """Changing learning_rate in config is reflected in the created agent."""
        custom_lr = 1e-5
        config = AppConfig(
            env=EnvConfig(device="cpu", input_channels=46, num_actions_total=13527, seed=42),
            training=TrainingConfig(
                total_timesteps=100,
                steps_per_epoch=16,
                ppo_epochs=2,
                minibatch_size=8,
                learning_rate=custom_lr,
                enable_torch_compile=False,
            ),
            evaluation=EvaluationConfig(num_games=2, max_moves_per_game=50),
            logging=LoggingConfig(
                log_file=str(tmp_path / "lr_test.log"),
                model_dir=str(tmp_path / "models"),
            ),
            wandb=WandBConfig(enabled=False),
            parallel=ParallelConfig(enabled=False),
            display=DisplayConfig(),
            webui=WebUIConfig(enabled=False),
        )

        model = ActorCritic(
            input_channels=46,
            num_actions_total=session_policy_mapper.get_total_actions(),
        )
        agent = PPOAgent(model=model, config=config, device=torch.device("cpu"))

        actual_lr = agent.optimizer.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(custom_lr)


# ---------------------------------------------------------------------------
# Pydantic validation
# ---------------------------------------------------------------------------


class TestPydanticValidation:
    """Invalid config is rejected by Pydantic validation."""

    def test_negative_learning_rate_rejected(self):
        """Pydantic rejects learning_rate <= 0."""
        with pytest.raises(Exception):  # ValidationError
            TrainingConfig(
                total_timesteps=100,
                steps_per_epoch=16,
                ppo_epochs=2,
                minibatch_size=8,
                learning_rate=-0.001,
            )

    def test_minibatch_size_must_exceed_one(self):
        """Pydantic rejects minibatch_size <= 1."""
        with pytest.raises(Exception):
            TrainingConfig(
                total_timesteps=100,
                steps_per_epoch=16,
                ppo_epochs=2,
                minibatch_size=1,
                learning_rate=0.001,
            )

    def test_invalid_evaluation_strategy_rejected(self):
        """Pydantic rejects unknown evaluation strategies."""
        with pytest.raises(Exception):
            EvaluationConfig(strategy="invalid_strategy")


# ---------------------------------------------------------------------------
# SetupManager component wiring
# ---------------------------------------------------------------------------


class TestSetupManagerWiring:
    """SetupManager wires config to components correctly."""

    def test_setup_step_manager_uses_correct_game(self, integration_config, session_policy_mapper):
        """StepManager created by SetupManager references the correct game."""
        env_mgr = EnvManager(integration_config)
        setup_mgr = SetupManager(integration_config, torch.device("cpu"))
        game, mapper, _, _ = setup_mgr.setup_game_components(env_mgr)

        model = ActorCritic(
            input_channels=46,
            num_actions_total=mapper.get_total_actions(),
        )
        agent = PPOAgent(model=model, config=integration_config, device=torch.device("cpu"))

        from keisei.core.experience_buffer import ExperienceBuffer

        buf = ExperienceBuffer(
            buffer_size=integration_config.training.steps_per_epoch,
            gamma=integration_config.training.gamma,
            lambda_gae=integration_config.training.lambda_gae,
            device="cpu",
            num_actions=mapper.get_total_actions(),
        )

        sm = setup_mgr.setup_step_manager(game, agent, mapper, buf)

        # StepManager should reference the same game object
        assert sm.game is game
        assert sm.agent is agent
        assert sm.experience_buffer is buf
