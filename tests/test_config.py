from pathlib import Path

import pytest

from keisei.config import (
    AppConfig,
    DynamicConfig,
    FrontierStaticConfig,
    LeagueConfig,
    MatchSchedulerConfig,
    RecentFixedConfig,
    load_config,
)


@pytest.fixture
def sample_toml(tmp_path: Path) -> Path:
    config_file = tmp_path / "test.toml"
    config_file.write_text("""\
[training]
num_games = 4
max_ply = 300
algorithm = "katago_ppo"
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
    assert config.training.algorithm == "katago_ppo"
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
algorithm = "katago_ppo"
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
num_games = 513
max_ply = 500
algorithm = "katago_ppo"
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
algorithm = "katago_ppo"
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
        max_ply=0, algorithm="katago_ppo", checkpoint_interval=10, moves_per_minute=30,
    ))
    with pytest.raises(ValueError, match="max_ply"):
        load_config(config_file)


def test_max_ply_negative_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text(_VALID_BASE.format(
        max_ply=-5, algorithm="katago_ppo", checkpoint_interval=10, moves_per_minute=30,
    ))
    with pytest.raises(ValueError, match="max_ply"):
        load_config(config_file)


def test_checkpoint_interval_zero_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text(_VALID_BASE.format(
        max_ply=500, algorithm="katago_ppo", checkpoint_interval=0, moves_per_minute=30,
    ))
    with pytest.raises(ValueError, match="checkpoint_interval"):
        load_config(config_file)


def test_checkpoint_interval_negative_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text(_VALID_BASE.format(
        max_ply=500, algorithm="katago_ppo", checkpoint_interval=-1, moves_per_minute=30,
    ))
    with pytest.raises(ValueError, match="checkpoint_interval"):
        load_config(config_file)


def test_moves_per_minute_negative_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text(_VALID_BASE.format(
        max_ply=500, algorithm="katago_ppo", checkpoint_interval=10, moves_per_minute=-1,
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


# ---------------------------------------------------------------------------
# H5: LeagueConfig ratio tolerance boundary tests
# ---------------------------------------------------------------------------


class TestTierConfigs:
    def test_frontier_static_defaults(self):
        cfg = FrontierStaticConfig()
        assert cfg.slots == 5
        assert cfg.review_interval_epochs == 250
        assert cfg.min_tenure_epochs == 100
        assert cfg.promotion_margin_elo == 50.0

    def test_recent_fixed_defaults(self):
        cfg = RecentFixedConfig()
        assert cfg.slots == 5
        assert cfg.min_games_for_review == 32
        assert cfg.min_unique_opponents == 6
        assert cfg.promotion_margin_elo == 25.0
        assert cfg.soft_overflow == 1

    def test_dynamic_defaults(self):
        cfg = DynamicConfig()
        assert cfg.slots == 10
        assert cfg.protection_matches == 24
        assert cfg.min_games_before_eviction == 40
        assert cfg.training_enabled is True

    def test_match_scheduler_defaults(self):
        cfg = MatchSchedulerConfig()
        assert cfg.learner_dynamic_ratio == 0.50
        assert cfg.learner_frontier_ratio == 0.30
        assert cfg.learner_recent_ratio == 0.20


class TestLeagueConfigValidation:
    def test_learner_ratios_must_sum_to_one(self):
        bad_sched = MatchSchedulerConfig(
            learner_dynamic_ratio=0.5,
            learner_frontier_ratio=0.5,
            learner_recent_ratio=0.5,
        )
        with pytest.raises(ValueError, match="learner.*ratio.*sum"):
            LeagueConfig(scheduler=bad_sched)

    def test_valid_config_passes(self):
        cfg = LeagueConfig()
        assert cfg.frontier.slots + cfg.recent.slots + cfg.dynamic.slots == 20

    def test_ratios_within_1e6_tolerance_pass(self) -> None:
        """Ratios within 1e-6 of 1.0 should pass validation."""
        sched = MatchSchedulerConfig(
            learner_dynamic_ratio=0.50,
            learner_frontier_ratio=0.30,
            learner_recent_ratio=0.20 + 5e-7,
        )
        cfg = LeagueConfig(scheduler=sched)
        assert cfg is not None

    def test_ratios_outside_1e6_tolerance_rejected(self) -> None:
        """Ratios outside 1e-6 of 1.0 should raise ValueError."""
        bad_sched = MatchSchedulerConfig(
            learner_dynamic_ratio=0.50,
            learner_frontier_ratio=0.30,
            learner_recent_ratio=0.2001,
        )
        with pytest.raises(ValueError, match="learner.*ratio.*sum"):
            LeagueConfig(scheduler=bad_sched)

    def test_ratios_sum_over_one_rejected(self) -> None:
        """Ratios summing to more than 1.0 should raise ValueError."""
        bad_sched = MatchSchedulerConfig(
            learner_dynamic_ratio=0.6,
            learner_frontier_ratio=0.6,
            learner_recent_ratio=0.6,
        )
        with pytest.raises(ValueError, match="learner.*ratio.*sum"):
            LeagueConfig(scheduler=bad_sched)


def test_training_use_amp_must_be_bool(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_text = _VALID_BASE.format(
        max_ply=500, algorithm="katago_ppo", checkpoint_interval=10, moves_per_minute=30,
    ).replace("[training.algorithm_params]", 'use_amp = "false"\n[training.algorithm_params]')
    config_file.write_text(config_text)
    with pytest.raises(ValueError, match="training.use_amp must be a boolean"):
        load_config(config_file)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sync_batchnorm", '"false"'),
        ("find_unused_parameters", '"false"'),
        ("gradient_as_bucket_view", '"false"'),
    ],
)
def test_distributed_flags_must_be_bool(tmp_path: Path, field: str, value: str) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text(_VALID_BASE.format(
        max_ply=500, algorithm="katago_ppo", checkpoint_interval=10, moves_per_minute=30,
    ) + f"\n[distributed]\n{field} = {value}\n")
    with pytest.raises(ValueError, match=f"distributed\\.{field} must be a boolean"):
        load_config(config_file)


def test_league_epochs_per_seat_zero_rejected() -> None:
    from keisei.config import LeagueConfig
    with pytest.raises(ValueError, match="league.epochs_per_seat must be >= 1"):
        LeagueConfig(epochs_per_seat=0)


def test_league_snapshot_interval_zero_rejected() -> None:
    from keisei.config import LeagueConfig
    with pytest.raises(ValueError, match="league.snapshot_interval must be >= 1"):
        LeagueConfig(snapshot_interval=0)
