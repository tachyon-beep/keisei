"""Shared fixtures for Keisei test suite."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from keisei.config import AppConfig, load_config
from keisei.db import init_db
from keisei.training.algorithm_registry import PPOParams
from keisei.training.models.resnet import ResNetModel, ResNetParams
from keisei.training.ppo import PPOAlgorithm, RolloutBuffer


# ---------------------------------------------------------------------------
# Model / PPO fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_resnet() -> ResNetModel:
    """A minimal ResNet for fast tests."""
    return ResNetModel(ResNetParams(hidden_size=16, num_layers=1))


@pytest.fixture
def small_ppo(small_resnet: ResNetModel) -> PPOAlgorithm:
    """A minimal PPO instance backed by a small ResNet."""
    params = PPOParams(learning_rate=1e-3, batch_size=4, epochs_per_batch=1)
    return PPOAlgorithm(params, small_resnet)


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Path for a temporary SQLite database (not yet initialised)."""
    return tmp_path / "test.db"


@pytest.fixture
def db(db_path: Path) -> Path:
    """An initialised temporary database."""
    init_db(str(db_path))
    return db_path


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

_CONFIG_TEMPLATE = """\
[training]
num_games = {num_games}
max_ply = 100
algorithm = "ppo"
checkpoint_interval = {checkpoint_interval}
checkpoint_dir = "{checkpoint_dir}"

[training.algorithm_params]
learning_rate = 1e-3
gamma = 0.99
clip_epsilon = 0.2
epochs_per_batch = 1
batch_size = 8

[display]
moves_per_minute = {moves_per_minute}
db_path = "{db_path}"

[model]
display_name = "{display_name}"
architecture = "resnet"

[model.params]
hidden_size = 16
num_layers = 1
"""


@pytest.fixture
def make_config(tmp_path: Path):
    """Factory fixture that returns a function to create AppConfig objects."""

    def _make(
        *,
        num_games: int = 4,
        checkpoint_interval: int = 2,
        moves_per_minute: int = 0,
        display_name: str = "TestBot",
    ) -> AppConfig:
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(
            _CONFIG_TEMPLATE.format(
                num_games=num_games,
                checkpoint_interval=checkpoint_interval,
                checkpoint_dir=str(tmp_path / "ckpt"),
                moves_per_minute=moves_per_minute,
                db_path=str(tmp_path / "test.db"),
                display_name=display_name,
            )
        )
        return load_config(toml_path)

    return _make


# ---------------------------------------------------------------------------
# Mock VecEnv
# ---------------------------------------------------------------------------


def make_mock_vecenv(num_envs: int = 4) -> MagicMock:
    """Create a mock VecEnv with correct shapes for integration tests."""
    mock = MagicMock()
    mock.num_envs = num_envs
    mock.action_space_size = 13527
    mock.observation_channels = 46

    reset_result = MagicMock()
    reset_result.observations = np.zeros((num_envs, 46, 9, 9), dtype=np.float32)
    reset_result.legal_masks = np.ones((num_envs, 13527), dtype=bool)
    mock.reset.return_value = reset_result

    step_result = MagicMock()
    step_result.observations = np.zeros((num_envs, 46, 9, 9), dtype=np.float32)
    step_result.legal_masks = np.ones((num_envs, 13527), dtype=bool)
    step_result.rewards = np.zeros(num_envs, dtype=np.float32)
    step_result.terminated = np.zeros(num_envs, dtype=bool)
    step_result.truncated = np.zeros(num_envs, dtype=bool)
    step_result.current_players = np.zeros(num_envs, dtype=np.uint8)
    step_metadata = MagicMock()
    step_metadata.ply_count = np.ones(num_envs, dtype=np.uint16) * 10
    step_result.step_metadata = step_metadata
    mock.step.return_value = step_result

    mock.episodes_completed = 0
    mock.episodes_drawn = 0
    mock.episodes_truncated = 0
    mock.draw_rate = 0.0
    mock.mean_episode_length = 0.0
    mock.truncation_rate = 0.0

    mock.get_spectator_data.return_value = [
        {
            "board": [None] * 81,
            "hands": {"black": {}, "white": {}},
            "current_player": "black",
            "ply": 0,
            "is_over": False,
            "result": "in_progress",
            "sfen": "startpos",
            "in_check": False,
        }
        for _ in range(num_envs)
    ]
    mock.get_sfens.return_value = ["startpos"] * num_envs

    return mock


# ---------------------------------------------------------------------------
# Rollout buffer helper
# ---------------------------------------------------------------------------


def fill_buffer(
    ppo: PPOAlgorithm,
    num_envs: int = 2,
    steps: int = 8,
    *,
    legal_mask: torch.Tensor | None = None,
    all_done: bool = False,
    rewards_val: float = 0.0,
) -> RolloutBuffer:
    """Fill a RolloutBuffer with steps from select_actions."""
    buf = RolloutBuffer(
        num_envs=num_envs, obs_shape=(46, 9, 9), action_space=13527
    )
    if legal_mask is None:
        legal_mask = torch.ones(num_envs, 13527, dtype=torch.bool)
    for _ in range(steps):
        obs = torch.randn(num_envs, 46, 9, 9)
        actions, log_probs, values = ppo.select_actions(obs, legal_mask)
        buf.add(
            obs,
            actions,
            log_probs,
            values,
            torch.full((num_envs,), rewards_val),
            torch.ones(num_envs, dtype=torch.bool)
            if all_done
            else torch.zeros(num_envs, dtype=torch.bool),
            legal_mask,
        )
    return buf
