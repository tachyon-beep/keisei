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
snapshot_interval = 10
epochs_per_seat = 50
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


def test_priority_scorer_config_defaults():
    from keisei.config import PriorityScorerConfig

    c = PriorityScorerConfig()
    assert c.under_sample_weight == 1.0
    assert c.uncertainty_weight == 0.5
    assert c.recent_fixed_bonus == 0.3
    assert c.diversity_weight == 0.3
    assert c.repeat_penalty == -0.5
    assert c.lineage_penalty == -0.3
    assert c.repeat_window_rounds == 5


def test_concurrency_config_defaults():
    from keisei.config import ConcurrencyConfig

    c = ConcurrencyConfig()
    assert c.parallel_matches == 4
    assert c.envs_per_match == 8
    assert c.total_envs == 32
    assert c.max_resident_models == 10


def test_concurrency_config_validation_env_budget():
    from keisei.config import ConcurrencyConfig

    # 4 * 8 = 32 > 16 → should fail
    with pytest.raises(ValueError, match="exceeds total_envs"):
        ConcurrencyConfig(parallel_matches=4, envs_per_match=8, total_envs=16, max_resident_models=10)


def test_concurrency_config_validation_model_budget():
    from keisei.config import ConcurrencyConfig

    # Explicitly specify valid env budget so only model validation fires
    with pytest.raises(ValueError, match="max_resident_models"):
        ConcurrencyConfig(parallel_matches=4, envs_per_match=2, total_envs=8, max_resident_models=1)

    # max_resident < parallel*2 is now allowed (runtime caps active slots)
    c = ConcurrencyConfig(parallel_matches=4, envs_per_match=2, total_envs=8, max_resident_models=4)
    assert c.effective_parallel == 2  # 4 // 2 = 2 slots


def test_league_scheduler_ratio_validation(tmp_path):
    """learner mix ratios must sum to 1.0."""
    # Explicitly set all three ratios to values that sum != 1.0
    bad_toml = LEAGUE_TOML + (
        "\n[league.scheduler]\n"
        "learner_dynamic_ratio = 0.5\n"
        "learner_frontier_ratio = 0.4\n"
        "learner_recent_ratio = 0.4\n"
    )
    toml_file = tmp_path / "badratio.toml"
    toml_file.write_text(bad_toml)
    with pytest.raises(ValueError, match="ratio"):
        load_config(toml_file)
