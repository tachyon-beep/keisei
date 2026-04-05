"""Tests for Phase 3 config extensions (training + promotion parameters)."""

import pytest

from keisei.config import DynamicConfig, FrontierStaticConfig, load_config

# Minimal TOML that exercises the config loader (based on test_league_config.py pattern)
BASE_TOML = """\
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

[display]
moves_per_minute = 0
db_path = "test.db"
"""


# --- DynamicConfig tests ---


def test_dynamic_config_training_defaults():
    """Construct with no args, assert all new fields have expected defaults."""
    dc = DynamicConfig()
    assert dc.training_enabled is True
    assert dc.update_epochs_per_batch == 2
    assert dc.lr_scale == 0.25
    assert dc.grad_clip == 1.0
    assert dc.update_every_matches == 4
    assert dc.max_updates_per_minute == 20
    assert dc.checkpoint_flush_every == 8
    assert dc.disable_on_error is True
    assert dc.max_buffer_depth == 8
    assert dc.max_consecutive_errors == 3


def test_dynamic_config_validation():
    """Validation rejects bad lr_scale, update_every_matches, max_updates_per_minute."""
    # lr_scale=0.0 out of (0, 1.0]
    with pytest.raises(ValueError, match="lr_scale"):
        DynamicConfig(lr_scale=0.0)

    # lr_scale=1.5 out of (0, 1.0]
    with pytest.raises(ValueError, match="lr_scale"):
        DynamicConfig(lr_scale=1.5)

    # lr_scale=1.0 should be OK
    dc = DynamicConfig(lr_scale=1.0)
    assert dc.lr_scale == 1.0

    # update_every_matches=0 should raise
    with pytest.raises(ValueError, match="update_every_matches"):
        DynamicConfig(update_every_matches=0)

    # max_updates_per_minute=0 should raise
    with pytest.raises(ValueError, match="max_updates_per_minute"):
        DynamicConfig(max_updates_per_minute=0)


# --- FrontierStaticConfig tests ---


def test_frontier_config_promotion_defaults():
    """No-args construct, assert new promotion fields."""
    fc = FrontierStaticConfig()
    assert fc.min_games_for_promotion == 100
    assert fc.topk == 3
    assert fc.streak_epochs == 50
    assert fc.max_lineage_overlap == 2


def test_frontier_config_promotion_validation():
    """Individual field bounds are validated independently."""
    with pytest.raises(ValueError, match="min_games_for_promotion must be >= 0"):
        FrontierStaticConfig(min_games_for_promotion=-1)

    with pytest.raises(ValueError, match="min_tenure_epochs must be >= 0"):
        FrontierStaticConfig(min_tenure_epochs=-1)

    with pytest.raises(ValueError, match="slots must be >= 1"):
        FrontierStaticConfig(slots=0)

    # Different units — games < epochs is valid
    fc = FrontierStaticConfig(min_games_for_promotion=64, min_tenure_epochs=100)
    assert fc.min_games_for_promotion == 64


# --- TOML loading tests ---


def test_load_config_with_phase3_fields(tmp_path):
    """TOML with Phase 3 dynamic fields loads correctly."""
    toml_with_phase3 = (
        BASE_TOML
        + "\n[league]\nsnapshot_interval = 10\nepochs_per_seat = 50\n"
        + "\n[league.dynamic]\ntraining_enabled = true\nlr_scale = 0.5\n"
    )
    toml_file = tmp_path / "phase3.toml"
    toml_file.write_text(toml_with_phase3)
    config = load_config(toml_file)

    assert config.league is not None
    assert config.league.dynamic.training_enabled is True
    assert config.league.dynamic.lr_scale == 0.5
    # Other new fields should have defaults
    assert config.league.dynamic.update_epochs_per_batch == 2
    assert config.league.dynamic.grad_clip == 1.0


def test_load_config_without_phase3_fields(tmp_path):
    """TOML without new fields loads with defaults."""
    toml_no_phase3 = (
        BASE_TOML
        + "\n[league]\nsnapshot_interval = 10\nepochs_per_seat = 50\n"
    )
    toml_file = tmp_path / "defaults.toml"
    toml_file.write_text(toml_no_phase3)
    config = load_config(toml_file)

    assert config.league is not None
    assert config.league.dynamic.training_enabled is True
    assert config.league.dynamic.lr_scale == 0.25
    assert config.league.dynamic.update_every_matches == 4
    # Frontier defaults too
    assert config.league.frontier.min_games_for_promotion == 100
    assert config.league.frontier.topk == 3
