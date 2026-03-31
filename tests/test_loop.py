"""Integration tests for the training loop. Uses mock VecEnv."""
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from keisei.training.loop import TrainingLoop
from keisei.config import load_config
from keisei.db import init_db, read_metrics_since, read_training_state


def _make_mock_vecenv(num_envs: int = 4) -> MagicMock:
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
    mock.mean_episode_length.return_value = 0.0
    mock.truncation_rate.return_value = 0.0

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


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    config = tmp_path / "test.toml"
    config.write_text(f"""\
[training]
num_games = 4
max_ply = 100
algorithm = "ppo"
checkpoint_interval = 2
checkpoint_dir = "{tmp_path / 'ckpt'}"

[training.algorithm_params]
learning_rate = 1e-3
gamma = 0.99
clip_epsilon = 0.2
epochs_per_batch = 1
batch_size = 8

[display]
moves_per_minute = 0
db_path = "{tmp_path / 'test.db'}"

[model]
display_name = "TestBot"
architecture = "resnet"

[model.params]
hidden_size = 16
num_layers = 1
""")
    return config


def test_training_loop_runs_one_epoch(config_file: Path) -> None:
    config = load_config(config_file)
    mock_vecenv = _make_mock_vecenv(num_envs=config.training.num_games)

    loop = TrainingLoop(config, vecenv=mock_vecenv)
    loop.run(num_epochs=1, steps_per_epoch=8)

    rows = read_metrics_since(config.display.db_path, since_id=0)
    assert len(rows) == 1
    assert rows[0]["epoch"] == 0

    state = read_training_state(config.display.db_path)
    assert state is not None
    assert state["display_name"] == "TestBot"
    assert state["status"] == "running"


def test_training_loop_creates_checkpoint(config_file: Path) -> None:
    config = load_config(config_file)
    mock_vecenv = _make_mock_vecenv(num_envs=config.training.num_games)

    loop = TrainingLoop(config, vecenv=mock_vecenv)
    loop.run(num_epochs=3, steps_per_epoch=8)

    ckpt_dir = Path(config.training.checkpoint_dir)
    assert ckpt_dir.exists()
    checkpoints = list(ckpt_dir.glob("*.pt"))
    assert len(checkpoints) >= 1


def test_training_loop_writes_metrics_each_epoch(config_file: Path) -> None:
    config = load_config(config_file)
    mock_vecenv = _make_mock_vecenv(num_envs=config.training.num_games)

    loop = TrainingLoop(config, vecenv=mock_vecenv)
    loop.run(num_epochs=3, steps_per_epoch=8)

    rows = read_metrics_since(config.display.db_path, since_id=0)
    assert len(rows) == 3
