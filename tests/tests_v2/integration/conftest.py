"""Integration test tier: multi-component tests with real dependencies.

Fixtures here build real multi-component chains (agent + game + buffer).
Only system boundaries (wandb, filesystem) are mocked.
"""

import random
from typing import Callable, Tuple

import numpy as np
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
from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.neural_network import ActorCritic
from keisei.core.ppo_agent import PPOAgent
from keisei.shogi.shogi_game import ShogiGame
from keisei.training.step_manager import EpisodeState, StepManager
from keisei.utils.utils import PolicyOutputMapper


# ---------------------------------------------------------------------------
# Shared session fixture: PolicyOutputMapper is expensive (~13k entries)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def session_policy_mapper():
    """PolicyOutputMapper builds 13527 moves -- cache per session."""
    return PolicyOutputMapper()


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def integration_config(tmp_path):
    """Small but realistic AppConfig for integration tests."""
    return AppConfig(
        env=EnvConfig(
            device="cpu",
            input_channels=46,
            num_actions_total=13527,
            seed=42,
            max_moves_per_game=512,
        ),
        training=TrainingConfig(
            total_timesteps=200,
            steps_per_epoch=16,
            ppo_epochs=2,
            minibatch_size=8,
            learning_rate=3e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coef=0.01,
            tower_depth=2,
            tower_width=64,
            model_type="resnet",
            gradient_clip_max_norm=0.5,
            lambda_gae=0.95,
            checkpoint_interval_timesteps=100,
            evaluation_interval_timesteps=100,
            enable_torch_compile=False,
        ),
        evaluation=EvaluationConfig(num_games=2, max_moves_per_game=50),
        logging=LoggingConfig(
            log_file=str(tmp_path / "train.log"),
            model_dir=str(tmp_path / "models"),
        ),
        wandb=WandBConfig(enabled=False),
        parallel=ParallelConfig(enabled=False),
        display=DisplayConfig(display_moves=False),
        webui=WebUIConfig(enabled=False),
    )


# ---------------------------------------------------------------------------
# Component fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cnn_model(session_policy_mapper):
    """Small CNN model for fast integration testing."""
    return ActorCritic(
        input_channels=46,
        num_actions_total=session_policy_mapper.get_total_actions(),
    )


@pytest.fixture
def ppo_agent(cnn_model, integration_config):
    """PPO agent with CNN model on CPU."""
    return PPOAgent(
        model=cnn_model,
        config=integration_config,
        device=torch.device("cpu"),
    )


@pytest.fixture
def shogi_game():
    """Fresh ShogiGame in initial position."""
    return ShogiGame(max_moves_per_game=500)


@pytest.fixture
def experience_buffer(integration_config, session_policy_mapper):
    """Empty experience buffer sized for integration_config.training.steps_per_epoch."""
    return ExperienceBuffer(
        buffer_size=integration_config.training.steps_per_epoch,
        gamma=integration_config.training.gamma,
        lambda_gae=integration_config.training.lambda_gae,
        device="cpu",
        num_actions=session_policy_mapper.get_total_actions(),
    )


@pytest.fixture
def step_manager(integration_config, shogi_game, ppo_agent, session_policy_mapper, experience_buffer):
    """StepManager wired to real game, agent, mapper, and buffer."""
    return StepManager(
        config=integration_config,
        game=shogi_game,
        agent=ppo_agent,
        policy_mapper=session_policy_mapper,
        experience_buffer=experience_buffer,
    )


# ---------------------------------------------------------------------------
# No-op logger fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def noop_logger():
    """No-op logger satisfying StepManager.execute_step's logger_func signature."""

    def _logger(msg, also_to_wandb=False, wandb_data=None, log_level="info"):
        pass

    return _logger


# ---------------------------------------------------------------------------
# Legal-mask helper fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def legal_mask_fn(session_policy_mapper):
    """Return a callable that builds a boolean legal-action mask from legal moves."""

    def _make_mask(legal_moves):
        return session_policy_mapper.get_legal_mask(legal_moves, device=torch.device("cpu"))

    return _make_mask


# ---------------------------------------------------------------------------
# Filled-buffer fixture: plays real game steps to populate a buffer
# ---------------------------------------------------------------------------


def _noop_logger_func(msg, also_to_wandb=False, wandb_data=None, log_level="info"):
    """Module-level no-op logger for use inside fixtures."""
    pass


@pytest.fixture
def filled_buffer(
    integration_config,
    shogi_game,
    ppo_agent,
    session_policy_mapper,
    experience_buffer,
) -> ExperienceBuffer:
    """Play real game steps and return a filled experience buffer.

    Fills up to ``steps_per_epoch`` experiences using the real agent and game.
    """
    sm = StepManager(
        config=integration_config,
        game=shogi_game,
        agent=ppo_agent,
        policy_mapper=session_policy_mapper,
        experience_buffer=experience_buffer,
    )

    episode_state = sm.reset_episode()
    steps_needed = integration_config.training.steps_per_epoch

    for t in range(steps_needed):
        step_result = sm.execute_step(
            episode_state, global_timestep=t, logger_func=_noop_logger_func
        )
        if step_result.success:
            episode_state = sm.update_episode_state(episode_state, step_result)
            if step_result.done:
                episode_state = sm.reset_episode()
        else:
            episode_state = sm.reset_episode()

    # Compute advantages so the buffer is ready for learn()
    last_val = ppo_agent.get_value(episode_state.current_obs)
    experience_buffer.compute_advantages_and_returns(last_val)

    return experience_buffer


# ---------------------------------------------------------------------------
# Utility helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def play_random_moves():
    """Return a callable that plays up to *n* random legal moves on a game."""

    def _play(game: ShogiGame, n: int, seed: int = 42) -> int:
        rng = random.Random(seed)
        played = 0
        for _ in range(n):
            if game.game_over:
                break
            legal = game.get_legal_moves()
            if not legal:
                break
            move = rng.choice(legal)
            game.make_move(move)
            played += 1
        return played

    return _play
