"""Tests for league and demonstrator config extensions."""


import pytest

from keisei.config import DemonstratorConfig, LeagueConfig, load_config

LEAGUE_TOML = """
[model]
display_name = "Test"
architecture = "se_resnet"

[model.params]
num_blocks = 2
channels = 32
se_reduction = 8
global_pool_channels = 16
policy_channels = 8
value_fc_size = 32
score_fc_size = 16
obs_channels = 50

[training]
algorithm = "katago_ppo"
num_games = 2
max_ply = 50
checkpoint_interval = 10
checkpoint_dir = "checkpoints/"

[training.algorithm_params]
learning_rate = 0.0002
score_normalization = 76.0
grad_clip = 1.0

[display]
moves_per_minute = 0
db_path = "test.db"

[league]
max_pool_size = 20
snapshot_interval = 10
epochs_per_seat = 50
historical_ratio = 0.8
current_best_ratio = 0.2
initial_elo = 1000
elo_k_factor = 32
elo_floor = 500

[demonstrator]
num_games = 3
auto_matchup = true
moves_per_minute = 60
device = "cpu"
"""


def test_load_config_with_league(tmp_path):
    toml_file = tmp_path / "league.toml"
    toml_file.write_text(LEAGUE_TOML)
    config = load_config(toml_file)
    assert config.league is not None
    assert config.league.max_pool_size == 20
    assert config.league.elo_floor == 500


def test_load_config_with_demonstrator(tmp_path):
    toml_file = tmp_path / "demo.toml"
    toml_file.write_text(LEAGUE_TOML)
    config = load_config(toml_file)
    assert config.demonstrator is not None
    assert config.demonstrator.num_games == 3
    assert config.demonstrator.device == "cpu"


def test_league_config_defaults():
    lc = LeagueConfig()
    assert lc.max_pool_size == 20
    assert lc.snapshot_interval == 10
    assert lc.epochs_per_seat == 50
    assert lc.elo_floor == 500


def test_league_config_fairness_defaults():
    """New fairness config fields should default to True."""
    lc = LeagueConfig()
    assert lc.color_randomization is True
    assert lc.per_env_opponents is True


def test_demonstrator_config_defaults():
    dc = DemonstratorConfig()
    assert dc.num_games == 3
    assert dc.device == "cuda"


def test_load_config_without_league_section(tmp_path):
    """Config without [league] should get None."""
    toml = LEAGUE_TOML.split("[league]")[0].rstrip() + "\n"
    toml_file = tmp_path / "noleague.toml"
    toml_file.write_text(toml)
    config = load_config(toml_file)
    assert config.league is None
    assert config.demonstrator is None


def test_league_ratio_validation(tmp_path):
    """historical_ratio + current_best_ratio must equal 1.0."""
    bad_toml = LEAGUE_TOML.replace("historical_ratio = 0.8", "historical_ratio = 0.6")
    toml_file = tmp_path / "badratio.toml"
    toml_file.write_text(bad_toml)
    with pytest.raises(ValueError, match="ratio"):
        load_config(toml_file)
