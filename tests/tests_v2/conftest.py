import random
import numpy as np
import pytest
import torch

from keisei.config_schema import (
    AppConfig, EnvConfig, TrainingConfig, EvaluationConfig,
    LoggingConfig, WandBConfig, ParallelConfig, DisplayConfig, WebUIConfig,
)
from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.neural_network import ActorCritic
from keisei.core.ppo_agent import PPOAgent
from keisei.shogi.shogi_core_definitions import Color, MoveTuple, Piece, PieceType
from keisei.shogi.shogi_game import ShogiGame
from keisei.training.models.resnet_tower import ActorCriticResTower
from keisei.utils.utils import PolicyOutputMapper


@pytest.fixture(scope="session")
def session_policy_mapper():
    """PolicyOutputMapper builds 13527 moves - cache per session for speed."""
    return PolicyOutputMapper()


@pytest.fixture
def app_config(tmp_path):
    """Complete real AppConfig with small tower for speed."""
    return AppConfig(
        env=EnvConfig(device="cpu", input_channels=46, num_actions_total=13527, seed=42, max_moves_per_game=512),
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
        logging=LoggingConfig(log_file=str(tmp_path / "train.log"), model_dir=str(tmp_path / "models")),
        wandb=WandBConfig(enabled=False),
        parallel=ParallelConfig(enabled=False),
        display=DisplayConfig(),
        webui=WebUIConfig(enabled=False),
    )


@pytest.fixture
def shogi_game():
    """Fresh ShogiGame in initial position."""
    return ShogiGame(max_moves_per_game=500)


@pytest.fixture
def cnn_model(session_policy_mapper):
    """Small CNN model for fast testing."""
    return ActorCritic(input_channels=46, num_actions_total=session_policy_mapper.get_total_actions())


@pytest.fixture
def resnet_model(session_policy_mapper):
    """Small ResNet model (depth=2, width=64) for fast testing."""
    return ActorCriticResTower(
        input_channels=46,
        num_actions_total=session_policy_mapper.get_total_actions(),
        tower_depth=2,
        tower_width=64,
        se_ratio=0.25,
    )


@pytest.fixture
def ppo_agent(cnn_model, app_config):
    """PPO agent with CNN model on CPU."""
    return PPOAgent(model=cnn_model, config=app_config, device=torch.device("cpu"))


@pytest.fixture
def experience_buffer(app_config, session_policy_mapper):
    """Empty experience buffer sized for app_config.training.steps_per_epoch."""
    return ExperienceBuffer(
        buffer_size=app_config.training.steps_per_epoch,
        gamma=app_config.training.gamma,
        lambda_gae=app_config.training.lambda_gae,
        device="cpu",
        num_actions=session_policy_mapper.get_total_actions(),
    )


# ----------- Helpers (not fixtures) -----------

def play_n_moves(game: ShogiGame, n: int, rng: random.Random = None) -> int:
    """Play n random legal moves, return actual moves played (may be less if game ends)."""
    if rng is None:
        rng = random.Random(42)
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


def make_legal_mask(legal_moves, policy_mapper: PolicyOutputMapper) -> torch.Tensor:
    """Build a boolean legal-action mask from a list of legal ShogiGame moves."""
    total = policy_mapper.get_total_actions()
    mask = torch.zeros(total, dtype=torch.bool)
    for move in legal_moves:
        try:
            idx = policy_mapper.shogi_move_to_policy_index(move)
            mask[idx] = True
        except (KeyError, ValueError):
            pass  # Some moves may not map (edge cases)
    return mask
