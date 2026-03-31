from pathlib import Path

import pytest

from keisei.config import AppConfig, load_config


@pytest.fixture
def sample_toml(tmp_path: Path) -> Path:
    config_file = tmp_path / "test.toml"
    config_file.write_text("""\
[training]
num_games = 4
max_ply = 300
algorithm = "ppo"
checkpoint_interval = 10
checkpoint_dir = "ckpt/"

[training.algorithm_params]
learning_rate = 1e-3
gamma = 0.99
clip_epsilon = 0.2
epochs_per_batch = 4
batch_size = 128

[display]
moves_per_minute = 60
db_path = "test.db"

[model]
display_name = "TestBot"
architecture = "resnet"

[model.params]
hidden_size = 64
num_layers = 4
""")
    return config_file


def test_load_config_basic(sample_toml: Path) -> None:
    config = load_config(sample_toml)
    assert isinstance(config, AppConfig)
    assert config.training.num_games == 4
    assert config.training.max_ply == 300
    assert config.training.algorithm == "ppo"
    assert config.display.moves_per_minute == 60
    assert config.model.display_name == "TestBot"
    assert config.model.architecture == "resnet"


def test_db_path_resolved_to_absolute(sample_toml: Path) -> None:
    config = load_config(sample_toml)
    assert Path(config.display.db_path).is_absolute()


def test_checkpoint_dir_resolved_to_absolute(sample_toml: Path) -> None:
    config = load_config(sample_toml)
    assert Path(config.training.checkpoint_dir).is_absolute()


def test_num_games_out_of_range(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text("""\
[training]
num_games = 0
max_ply = 500
algorithm = "ppo"
checkpoint_interval = 50
checkpoint_dir = "ckpt/"
[training.algorithm_params]
[display]
moves_per_minute = 30
db_path = "test.db"
[model]
display_name = "X"
architecture = "resnet"
[model.params]
hidden_size = 64
num_layers = 4
""")
    with pytest.raises(ValueError, match="num_games"):
        load_config(config_file)


def test_num_games_too_high(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text("""\
[training]
num_games = 11
max_ply = 500
algorithm = "ppo"
checkpoint_interval = 50
checkpoint_dir = "ckpt/"
[training.algorithm_params]
[display]
moves_per_minute = 30
db_path = "test.db"
[model]
display_name = "X"
architecture = "resnet"
[model.params]
hidden_size = 64
num_layers = 4
""")
    with pytest.raises(ValueError, match="num_games"):
        load_config(config_file)


def test_unknown_architecture(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text("""\
[training]
num_games = 4
max_ply = 500
algorithm = "ppo"
checkpoint_interval = 50
checkpoint_dir = "ckpt/"
[training.algorithm_params]
[display]
moves_per_minute = 30
db_path = "test.db"
[model]
display_name = "X"
architecture = "nonexistent"
[model.params]
hidden_size = 64
num_layers = 4
""")
    with pytest.raises(ValueError, match="architecture"):
        load_config(config_file)


# ---------------------------------------------------------------------------
# High gap: untested validation branches
# ---------------------------------------------------------------------------

_VALID_BASE = """\
[training]
num_games = 4
max_ply = {max_ply}
algorithm = "{algorithm}"
checkpoint_interval = {checkpoint_interval}
checkpoint_dir = "ckpt/"
[training.algorithm_params]
[display]
moves_per_minute = {moves_per_minute}
db_path = "test.db"
[model]
display_name = "X"
architecture = "resnet"
[model.params]
hidden_size = 64
num_layers = 4
"""


def test_max_ply_zero_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text(_VALID_BASE.format(
        max_ply=0, algorithm="ppo", checkpoint_interval=10, moves_per_minute=30,
    ))
    with pytest.raises(ValueError, match="max_ply"):
        load_config(config_file)


def test_max_ply_negative_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text(_VALID_BASE.format(
        max_ply=-5, algorithm="ppo", checkpoint_interval=10, moves_per_minute=30,
    ))
    with pytest.raises(ValueError, match="max_ply"):
        load_config(config_file)


def test_checkpoint_interval_zero_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text(_VALID_BASE.format(
        max_ply=500, algorithm="ppo", checkpoint_interval=0, moves_per_minute=30,
    ))
    with pytest.raises(ValueError, match="checkpoint_interval"):
        load_config(config_file)


def test_checkpoint_interval_negative_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text(_VALID_BASE.format(
        max_ply=500, algorithm="ppo", checkpoint_interval=-1, moves_per_minute=30,
    ))
    with pytest.raises(ValueError, match="checkpoint_interval"):
        load_config(config_file)


def test_moves_per_minute_negative_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text(_VALID_BASE.format(
        max_ply=500, algorithm="ppo", checkpoint_interval=10, moves_per_minute=-1,
    ))
    with pytest.raises(ValueError, match="moves_per_minute"):
        load_config(config_file)


def test_unknown_algorithm_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text(_VALID_BASE.format(
        max_ply=500, algorithm="dqn", checkpoint_interval=10, moves_per_minute=30,
    ))
    with pytest.raises(ValueError, match="algorithm"):
        load_config(config_file)
