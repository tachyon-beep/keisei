"""Tests for league and demonstrator config extensions."""


import pytest

from keisei.config import (
    ConcurrencyConfig,
    DemonstratorConfig,
    DynamicConfig,
    FrontierStaticConfig,
    GauntletConfig,
    HistoricalLibraryConfig,
    LeagueConfig,
    MatchSchedulerConfig,
    PriorityScorerConfig,
    RecentFixedConfig,
    RoleEloConfig,
    StorageConfig,
    load_config,
)

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

    # max_resident < parallel*2 is now allowed (LRU cache shares models)
    c = ConcurrencyConfig(parallel_matches=4, envs_per_match=2, total_envs=8, max_resident_models=4)
    assert c.effective_parallel == 4  # no longer capped


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


class TestAllConfigDefaults:
    """Assert every sub-config's defaults match the plan."""

    def test_frontier_static_defaults(self):
        c = FrontierStaticConfig()
        assert c.slots == 5
        assert c.review_interval_epochs == 250
        assert c.min_tenure_epochs == 100
        assert c.promotion_margin_elo == 50.0
        assert c.min_games_for_promotion == 64
        assert c.topk == 3
        assert c.streak_epochs == 50
        assert c.max_lineage_overlap == 2
        assert c.replace_policy == "weakest_or_stalest_after_cooldown"
        assert c.span_selection is True

    def test_recent_fixed_defaults(self):
        c = RecentFixedConfig()
        assert c.slots == 5
        assert c.min_games_for_review == 32
        assert c.min_unique_opponents == 6
        assert c.promotion_margin_elo == 25.0
        assert c.soft_overflow == 1
        assert c.retire_if_below_dynamic_floor is True

    def test_dynamic_defaults(self):
        c = DynamicConfig()
        assert c.slots == 10
        assert c.protection_matches == 24
        assert c.min_games_before_eviction == 40
        assert c.training_enabled is True
        assert c.update_epochs_per_batch == 2
        assert c.lr_scale == 0.25
        assert c.grad_clip == 1.0
        assert c.update_every_matches == 4
        assert c.max_updates_per_minute == 20
        assert c.checkpoint_flush_every == 8
        assert c.batch_reuse == 1

    def test_historical_library_defaults(self):
        c = HistoricalLibraryConfig()
        assert c.enabled is True
        assert c.slots == 5
        assert c.refresh_interval_epochs == 100
        assert c.selection == "log_spaced"
        assert c.active_league_participation is False

    def test_gauntlet_defaults(self):
        c = GauntletConfig()
        assert c.enabled is True
        assert c.interval_epochs == 100
        assert c.games_per_matchup == 16

    def test_role_elo_defaults(self):
        c = RoleEloConfig()
        assert c.frontier_k == 16.0
        assert c.dynamic_k == 24.0
        assert c.recent_k == 32.0
        assert c.historical_k == 12.0
        assert c.track_role_specific is True

    def test_concurrency_defaults(self):
        c = ConcurrencyConfig()
        assert c.parallel_matches == 4
        assert c.envs_per_match == 8
        assert c.total_envs == 32
        assert c.max_resident_models == 10

    def test_priority_scorer_defaults(self):
        c = PriorityScorerConfig()
        assert c.under_sample_weight == 1.0
        assert c.uncertainty_weight == 0.5
        assert c.recent_fixed_bonus == 0.3
        assert c.diversity_weight == 0.3
        assert c.match_class_weight == 1.0
        assert c.frontier_exposure_weight == 0.4
        assert c.repeat_penalty == -0.5
        assert c.lineage_penalty == -0.3

    def test_storage_defaults(self):
        c = StorageConfig()
        assert c.clone_on_promotion is True
        assert c.persist_optimizer_for_dynamic is True

    def test_match_scheduler_defaults(self):
        c = MatchSchedulerConfig()
        assert c.learner_dynamic_ratio == 0.50
        assert c.learner_frontier_ratio == 0.30
        assert c.learner_recent_ratio == 0.20
        assert c.pairing_policy == "role_weighted_sparse_h2h"
        assert c.dynamic_dynamic_weight == 0.40
        assert c.dynamic_recent_weight == 0.25
        assert c.dynamic_frontier_weight == 0.20
        assert c.recent_frontier_weight == 0.10
        assert c.recent_recent_weight == 0.05


def test_effective_parallel_equals_parallel_matches():
    """effective_parallel no longer caps by max_resident_models."""
    c = ConcurrencyConfig(
        parallel_matches=64,
        envs_per_match=4,
        total_envs=256,
        max_resident_models=22,
    )
    assert c.effective_parallel == 64


def test_league_warns_when_cache_too_small(recwarn):
    """Warn when max_resident_models < max_active_entries."""
    LeagueConfig(
        max_active_entries=20,
        concurrency=ConcurrencyConfig(
            parallel_matches=4,
            envs_per_match=8,
            total_envs=32,
            max_resident_models=10,
        ),
    )
    assert len(recwarn) >= 1
    assert "max_resident_models" in str(recwarn[0].message)


def test_league_no_warn_when_cache_sufficient():
    """No warning when max_resident_models >= max_active_entries."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LeagueConfig(
            max_active_entries=20,
            concurrency=ConcurrencyConfig(
                parallel_matches=64,
                envs_per_match=4,
                total_envs=256,
                max_resident_models=22,
            ),
        )


FULL_LEAGUE_TOML = """\
[model]
display_name = "RoundTrip"
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
num_games = 4
max_ply = 100
checkpoint_interval = 25
checkpoint_dir = "checkpoints/"

[training.algorithm_params]
learning_rate = 0.001

[display]
moves_per_minute = 10
db_path = "roundtrip.db"

[league]
snapshot_interval = 5
epochs_per_seat = 25
initial_elo = 1200.0
elo_k_factor = 16.0
elo_floor = 400.0

[league.frontier]
slots = 3
review_interval_epochs = 200
min_tenure_epochs = 50
promotion_margin_elo = 40.0
min_games_for_promotion = 32
topk = 2
streak_epochs = 30
max_lineage_overlap = 1
span_selection = false

[league.recent]
slots = 4
min_games_for_review = 16
min_unique_opponents = 3
promotion_margin_elo = 20.0
soft_overflow = 2
retire_if_below_dynamic_floor = false

[league.dynamic]
slots = 8
protection_matches = 12
min_games_before_eviction = 20
training_enabled = false
update_epochs_per_batch = 3
lr_scale = 0.5
grad_clip = 2.0
update_every_matches = 2
max_updates_per_minute = 10
checkpoint_flush_every = 4
batch_reuse = 2

[league.scheduler]
learner_dynamic_ratio = 0.40
learner_frontier_ratio = 0.35
learner_recent_ratio = 0.25
pairing_policy = "role_weighted_sparse_h2h"
dynamic_dynamic_weight = 0.35
dynamic_recent_weight = 0.25
dynamic_frontier_weight = 0.20
recent_frontier_weight = 0.10
recent_recent_weight = 0.10

[league.history]
enabled = false
slots = 3
refresh_interval_epochs = 50
selection = "log_spaced"

[league.gauntlet]
enabled = false
interval_epochs = 50
games_per_matchup = 8

[league.elo]
frontier_k = 20.0
dynamic_k = 28.0
recent_k = 36.0
historical_k = 14.0

[league.priority]
under_sample_weight = 2.0
uncertainty_weight = 0.8
recent_fixed_bonus = 0.5
diversity_weight = 0.4
match_class_weight = 1.5
frontier_exposure_weight = 0.6
repeat_penalty = -0.3
lineage_penalty = -0.2

[league.concurrency]
parallel_matches = 2
envs_per_match = 4
total_envs = 16
max_resident_models = 6

[league.storage]
clone_on_promotion = true
persist_optimizer_for_dynamic = true
"""


def test_toml_round_trip_all_sections(tmp_path):
    """Load a TOML with ALL league sub-sections and verify parsed values."""
    toml_file = tmp_path / "full.toml"
    toml_file.write_text(FULL_LEAGUE_TOML)
    config = load_config(toml_file)

    # league is disabled (enabled defaults true, but history.enabled=false
    # doesn't disable league itself) — league should be present
    # Actually league.enabled defaults True and we didn't set it to false,
    # so league should be non-None.
    assert config.league is not None
    lg = config.league

    # Top-level league scalars
    assert lg.snapshot_interval == 5
    assert lg.epochs_per_seat == 25
    assert lg.initial_elo == 1200.0
    assert lg.elo_k_factor == 16.0
    assert lg.elo_floor == 400.0

    # Frontier
    assert lg.frontier.slots == 3
    assert lg.frontier.review_interval_epochs == 200
    assert lg.frontier.min_tenure_epochs == 50
    assert lg.frontier.promotion_margin_elo == 40.0
    assert lg.frontier.min_games_for_promotion == 32
    assert lg.frontier.topk == 2
    assert lg.frontier.streak_epochs == 30
    assert lg.frontier.max_lineage_overlap == 1
    assert lg.frontier.span_selection is False

    # Recent
    assert lg.recent.slots == 4
    assert lg.recent.min_games_for_review == 16
    assert lg.recent.min_unique_opponents == 3
    assert lg.recent.promotion_margin_elo == 20.0
    assert lg.recent.soft_overflow == 2
    assert lg.recent.retire_if_below_dynamic_floor is False

    # Dynamic
    assert lg.dynamic.slots == 8
    assert lg.dynamic.protection_matches == 12
    assert lg.dynamic.min_games_before_eviction == 20
    assert lg.dynamic.training_enabled is False
    assert lg.dynamic.update_epochs_per_batch == 3
    assert lg.dynamic.lr_scale == 0.5
    assert lg.dynamic.grad_clip == 2.0
    assert lg.dynamic.update_every_matches == 2
    assert lg.dynamic.max_updates_per_minute == 10
    assert lg.dynamic.checkpoint_flush_every == 4
    assert lg.dynamic.batch_reuse == 2

    # Scheduler
    assert lg.scheduler.learner_dynamic_ratio == 0.40
    assert lg.scheduler.learner_frontier_ratio == 0.35
    assert lg.scheduler.learner_recent_ratio == 0.25
    assert lg.scheduler.pairing_policy == "role_weighted_sparse_h2h"
    assert lg.scheduler.dynamic_dynamic_weight == 0.35
    assert lg.scheduler.dynamic_recent_weight == 0.25
    assert lg.scheduler.dynamic_frontier_weight == 0.20
    assert lg.scheduler.recent_frontier_weight == 0.10
    assert lg.scheduler.recent_recent_weight == 0.10

    # History (disabled but still parsed)
    # Note: history.enabled=false doesn't disable the whole league,
    # but the HistoricalLibraryConfig validator rejects active_league_participation=True.
    # enabled=false is fine.

    # Gauntlet
    assert lg.gauntlet.enabled is False
    assert lg.gauntlet.interval_epochs == 50
    assert lg.gauntlet.games_per_matchup == 8

    # Elo
    assert lg.elo.frontier_k == 20.0
    assert lg.elo.dynamic_k == 28.0
    assert lg.elo.recent_k == 36.0
    assert lg.elo.historical_k == 14.0

    # Priority
    assert lg.priority.under_sample_weight == 2.0
    assert lg.priority.uncertainty_weight == 0.8
    assert lg.priority.recent_fixed_bonus == 0.5
    assert lg.priority.diversity_weight == 0.4
    assert lg.priority.match_class_weight == 1.5
    assert lg.priority.frontier_exposure_weight == 0.6
    assert lg.priority.repeat_penalty == -0.3
    assert lg.priority.lineage_penalty == -0.2

    # Concurrency
    assert lg.concurrency.parallel_matches == 2
    assert lg.concurrency.envs_per_match == 4
    assert lg.concurrency.total_envs == 16
    assert lg.concurrency.max_resident_models == 6

    # Storage
    assert lg.storage.clone_on_promotion is True
    assert lg.storage.persist_optimizer_for_dynamic is True
