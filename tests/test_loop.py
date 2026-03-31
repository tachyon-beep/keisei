"""Integration tests for the training loop. Uses mock VecEnv."""
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from keisei.config import load_config
from keisei.db import init_db, read_metrics_since, read_training_state, write_training_state
from keisei.training.loop import TrainingLoop


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


def test_training_loop_resumes_from_checkpoint(config_file: Path) -> None:
    """Train 2 epochs, checkpoint at epoch 2, create new loop, verify it resumes."""
    config = load_config(config_file)
    mock_vecenv = _make_mock_vecenv(num_envs=config.training.num_games)

    loop1 = TrainingLoop(config, vecenv=mock_vecenv)
    loop1.run(num_epochs=2, steps_per_epoch=8)

    # checkpoint_interval=2, so epoch 1 (0-indexed) triggers a checkpoint
    ckpt_dir = Path(config.training.checkpoint_dir)
    checkpoints = list(ckpt_dir.glob("*.pt"))
    assert len(checkpoints) >= 1, "No checkpoint was created"

    state = read_training_state(config.display.db_path)
    assert state is not None
    assert state["checkpoint_path"] is not None

    # Create a new loop — should resume from checkpoint
    mock_vecenv2 = _make_mock_vecenv(num_envs=config.training.num_games)
    loop2 = TrainingLoop(config, vecenv=mock_vecenv2)
    assert loop2.epoch == 2, f"Expected resumed epoch=2, got {loop2.epoch}"
    assert loop2.global_step > 0, "global_step should be restored from checkpoint"


def test_training_loop_metrics_contain_expected_keys(config_file: Path) -> None:
    """Verify all expected metric keys are present in the DB row."""
    config = load_config(config_file)
    mock_vecenv = _make_mock_vecenv(num_envs=config.training.num_games)

    loop = TrainingLoop(config, vecenv=mock_vecenv)
    loop.run(num_epochs=1, steps_per_epoch=4)

    rows = read_metrics_since(config.display.db_path, since_id=0)
    assert len(rows) == 1
    row = rows[0]
    for key in ["epoch", "step", "policy_loss", "value_loss", "entropy", "gradient_norm"]:
        assert key in row, f"Missing metric key: {key}"
        assert row[key] is not None, f"Metric {key} is None"


def test_training_loop_snapshot_disabled(config_file: Path) -> None:
    """With moves_per_minute=0, no snapshots should be written."""
    config = load_config(config_file)
    assert config.display.moves_per_minute == 0
    mock_vecenv = _make_mock_vecenv(num_envs=config.training.num_games)

    loop = TrainingLoop(config, vecenv=mock_vecenv)
    loop.run(num_epochs=1, steps_per_epoch=4)

    from keisei.db import read_game_snapshots
    snapshots = read_game_snapshots(config.display.db_path)
    assert len(snapshots) == 0, "No snapshots expected when moves_per_minute=0"


# ---------------------------------------------------------------------------
# Critical gap: snapshot writing with moves_per_minute > 0
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot_config_file(tmp_path: Path) -> Path:
    """Config with moves_per_minute set very high so snapshots fire immediately."""
    config = tmp_path / "snap.toml"
    config.write_text(f"""\
[training]
num_games = 2
max_ply = 100
algorithm = "ppo"
checkpoint_interval = 100
checkpoint_dir = "{tmp_path / 'ckpt'}"

[training.algorithm_params]
learning_rate = 1e-3
gamma = 0.99
clip_epsilon = 0.2
epochs_per_batch = 1
batch_size = 4

[display]
moves_per_minute = 999999
db_path = "{tmp_path / 'snap.db'}"

[model]
display_name = "SnapBot"
architecture = "resnet"

[model.params]
hidden_size = 16
num_layers = 1
""")
    return config


def test_training_loop_snapshots_written(snapshot_config_file: Path) -> None:
    """With high moves_per_minute, snapshots should be written to the DB."""
    from keisei.db import read_game_snapshots

    config = load_config(snapshot_config_file)
    assert config.display.moves_per_minute > 0
    mock_vecenv = _make_mock_vecenv(num_envs=config.training.num_games)

    loop = TrainingLoop(config, vecenv=mock_vecenv)
    loop.run(num_epochs=1, steps_per_epoch=4)

    snapshots = read_game_snapshots(config.display.db_path)
    assert len(snapshots) > 0, "Expected snapshots with moves_per_minute > 0"
    # Verify snapshot structure
    snap = snapshots[0]
    assert "board_json" in snap
    assert "sfen" in snap
    assert "current_player" in snap
    import json
    board = json.loads(snap["board_json"])
    assert isinstance(board, list)
    hands = json.loads(snap["hands_json"])
    assert isinstance(hands, dict)


# ---------------------------------------------------------------------------
# Critical gap: win_rate metric computation with positive rewards
# ---------------------------------------------------------------------------


def test_training_loop_win_rate_with_rewards(config_file: Path) -> None:
    """When VecEnv returns positive rewards, win_rate should be computed."""
    config = load_config(config_file)
    num_envs = config.training.num_games
    mock_vecenv = _make_mock_vecenv(num_envs=num_envs)

    # Make env 0 win each step (reward=1), rest get reward=0
    rewards = np.zeros(num_envs, dtype=np.float32)
    rewards[0] = 1.0
    mock_vecenv.step.return_value.rewards = rewards
    mock_vecenv.episodes_completed = 4

    loop = TrainingLoop(config, vecenv=mock_vecenv)
    loop.run(num_epochs=1, steps_per_epoch=8)

    rows = read_metrics_since(config.display.db_path, since_id=0)
    assert len(rows) == 1
    row = rows[0]
    assert row["win_rate"] is not None
    assert row["win_rate"] > 0, "Expected positive win_rate with positive rewards"


# ---------------------------------------------------------------------------
# Critical gap: _check_resume with stale checkpoint path
# ---------------------------------------------------------------------------


def test_training_loop_resume_stale_checkpoint(config_file: Path) -> None:
    """If training_state has a checkpoint_path that no longer exists, start fresh."""
    from keisei.db import write_training_state

    config = load_config(config_file)
    init_db(config.display.db_path)

    # Write a training state with a non-existent checkpoint path
    write_training_state(config.display.db_path, {
        "config_json": '{}',
        "display_name": "StaleBot",
        "model_arch": "resnet",
        "algorithm_name": "ppo",
        "started_at": "2026-04-01T00:00:00Z",
        "current_epoch": 10,
        "current_step": 5000,
        "checkpoint_path": "/tmp/nonexistent_checkpoint_xyz.pt",
    })

    mock_vecenv = _make_mock_vecenv(num_envs=config.training.num_games)
    loop = TrainingLoop(config, vecenv=mock_vecenv)

    # Should start fresh (epoch 0), not crash or resume from stale checkpoint
    assert loop.epoch == 0, (
        f"Expected fresh start (epoch=0) with stale checkpoint, got epoch={loop.epoch}"
    )


def test_training_loop_resume_no_checkpoint_path(config_file: Path) -> None:
    """If training_state exists but checkpoint_path is None, start fresh."""
    from keisei.db import write_training_state

    config = load_config(config_file)
    init_db(config.display.db_path)

    write_training_state(config.display.db_path, {
        "config_json": '{}',
        "display_name": "NoCheckpointBot",
        "model_arch": "resnet",
        "algorithm_name": "ppo",
        "started_at": "2026-04-01T00:00:00Z",
        "current_epoch": 5,
        "current_step": 2500,
    })

    mock_vecenv = _make_mock_vecenv(num_envs=config.training.num_games)
    loop = TrainingLoop(config, vecenv=mock_vecenv)

    assert loop.epoch == 0, (
        f"Expected fresh start (epoch=0) with no checkpoint_path, got epoch={loop.epoch}"
    )
