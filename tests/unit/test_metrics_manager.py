"""Unit tests for MetricsManager: statistics tracking, win rates, formatting, checkpoints."""

import json

import pytest

from keisei.shogi.shogi_core_definitions import Color
from keisei.training.metrics_manager import MetricsHistory, MetricsManager, TrainingStats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def manager():
    """Create a fresh MetricsManager with default settings."""
    return MetricsManager()


@pytest.fixture
def ppo_metrics():
    """Standard PPO metrics dict for formatting tests."""
    return {
        MetricsHistory.PPO_LEARNING_RATE: 3e-4,
        MetricsHistory.PPO_POLICY_LOSS: 0.1234,
        MetricsHistory.PPO_VALUE_LOSS: 0.5678,
        MetricsHistory.PPO_KL_DIVERGENCE: 0.0042,
        MetricsHistory.PPO_ENTROPY: 5.1234,
        MetricsHistory.PPO_CLIP_FRACTION: 0.15,
    }


# ---------------------------------------------------------------------------
# 1. update_episode_stats: win/draw counting
# ---------------------------------------------------------------------------
class TestUpdateEpisodeStats:
    """update_episode_stats correctly increments counters and returns win rates."""

    def test_black_win_increments_black_wins_counter(self, manager):
        manager.update_episode_stats(Color.BLACK)
        assert manager.stats.black_wins == 1
        assert manager.stats.white_wins == 0
        assert manager.stats.draws == 0

    def test_white_win_increments_white_wins_counter(self, manager):
        manager.update_episode_stats(Color.WHITE)
        assert manager.stats.white_wins == 1
        assert manager.stats.black_wins == 0
        assert manager.stats.draws == 0

    def test_draw_increments_draws_counter(self, manager):
        """Explicit draw (any value that is not BLACK or WHITE) increments draws."""
        manager.update_episode_stats("draw")
        assert manager.stats.draws == 1
        assert manager.stats.black_wins == 0
        assert manager.stats.white_wins == 0

    def test_none_winner_increments_draws_counter(self, manager):
        manager.update_episode_stats(None)
        assert manager.stats.draws == 1
        assert manager.stats.black_wins == 0
        assert manager.stats.white_wins == 0

    def test_increments_total_episodes_completed(self, manager):
        manager.update_episode_stats(Color.BLACK)
        assert manager.stats.total_episodes_completed == 1

    def test_returns_win_rates_dict(self, manager):
        result = manager.update_episode_stats(Color.BLACK)
        assert "win_rate_black" in result
        assert "win_rate_white" in result
        assert "win_rate_draw" in result

    def test_multiple_games_accumulate_correctly(self, manager):
        manager.update_episode_stats(Color.BLACK)
        manager.update_episode_stats(Color.BLACK)
        manager.update_episode_stats(Color.WHITE)
        manager.update_episode_stats(None)
        assert manager.stats.black_wins == 2
        assert manager.stats.white_wins == 1
        assert manager.stats.draws == 1
        assert manager.stats.total_episodes_completed == 4

    def test_adds_win_rate_data_to_history(self, manager):
        manager.update_episode_stats(Color.BLACK)
        assert len(manager.history.win_rates_history) == 1


# ---------------------------------------------------------------------------
# 2. get_win_rates: percentage calculations
# ---------------------------------------------------------------------------
class TestGetWinRates:
    """get_win_rates returns correct percentage tuples."""

    def test_returns_zero_tuple_with_no_games(self, manager):
        black_rate, white_rate, draw_rate = manager.get_win_rates()
        assert black_rate == 0.0
        assert white_rate == 0.0
        assert draw_rate == 0.0

    def test_returns_correct_rates_after_known_games(self, manager):
        # 2 black wins, 1 white win, 1 draw = 4 games
        manager.update_episode_stats(Color.BLACK)
        manager.update_episode_stats(Color.BLACK)
        manager.update_episode_stats(Color.WHITE)
        manager.update_episode_stats(None)

        black_rate, white_rate, draw_rate = manager.get_win_rates()
        assert black_rate == pytest.approx(50.0)
        assert white_rate == pytest.approx(25.0)
        assert draw_rate == pytest.approx(25.0)

    def test_all_black_wins_gives_100_percent(self, manager):
        for _ in range(5):
            manager.update_episode_stats(Color.BLACK)
        black_rate, white_rate, draw_rate = manager.get_win_rates()
        assert black_rate == pytest.approx(100.0)
        assert white_rate == pytest.approx(0.0)
        assert draw_rate == pytest.approx(0.0)

    def test_rates_sum_to_100(self, manager):
        manager.update_episode_stats(Color.BLACK)
        manager.update_episode_stats(Color.WHITE)
        manager.update_episode_stats(None)
        black_rate, white_rate, draw_rate = manager.get_win_rates()
        assert black_rate + white_rate + draw_rate == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 3. get_win_rates_dict
# ---------------------------------------------------------------------------
class TestGetWinRatesDict:
    """get_win_rates_dict returns a dictionary with expected keys and values."""

    def test_returns_dict_with_correct_keys(self, manager):
        result = manager.get_win_rates_dict()
        assert set(result.keys()) == {"win_rate_black", "win_rate_white", "win_rate_draw"}

    def test_values_match_tuple_form(self, manager):
        manager.update_episode_stats(Color.BLACK)
        manager.update_episode_stats(Color.WHITE)

        d = manager.get_win_rates_dict()
        black_rate, white_rate, draw_rate = manager.get_win_rates()
        assert d["win_rate_black"] == pytest.approx(black_rate)
        assert d["win_rate_white"] == pytest.approx(white_rate)
        assert d["win_rate_draw"] == pytest.approx(draw_rate)


# ---------------------------------------------------------------------------
# 4. Timestep management
# ---------------------------------------------------------------------------
class TestTimestepManagement:
    """increment_timestep, increment_timestep_by, and global_timestep property."""

    def test_increment_timestep_adds_one(self, manager):
        assert manager.global_timestep == 0
        manager.increment_timestep()
        assert manager.global_timestep == 1

    def test_increment_timestep_multiple_times(self, manager):
        for _ in range(5):
            manager.increment_timestep()
        assert manager.global_timestep == 5

    def test_increment_timestep_by_adds_specified_amount(self, manager):
        manager.increment_timestep_by(100)
        assert manager.global_timestep == 100

    def test_increment_timestep_by_zero_is_noop(self, manager):
        manager.increment_timestep_by(0)
        assert manager.global_timestep == 0

    def test_increment_timestep_by_negative_raises_value_error(self, manager):
        with pytest.raises(ValueError, match="non-negative"):
            manager.increment_timestep_by(-1)

    def test_global_timestep_property_returns_correct_value(self, manager):
        manager.stats.global_timestep = 42
        assert manager.global_timestep == 42

    def test_global_timestep_setter_works(self, manager):
        manager.global_timestep = 99
        assert manager.stats.global_timestep == 99


# ---------------------------------------------------------------------------
# 5. Episode count property
# ---------------------------------------------------------------------------
class TestTotalEpisodesCompleted:
    """total_episodes_completed property tracks episode count."""

    def test_starts_at_zero(self, manager):
        assert manager.total_episodes_completed == 0

    def test_incremented_by_update_episode_stats(self, manager):
        manager.update_episode_stats(Color.BLACK)
        manager.update_episode_stats(Color.WHITE)
        assert manager.total_episodes_completed == 2

    def test_setter_works(self, manager):
        manager.total_episodes_completed = 50
        assert manager.stats.total_episodes_completed == 50


# ---------------------------------------------------------------------------
# 6. MetricsHistory: add_episode_data and add_ppo_data
# ---------------------------------------------------------------------------
class TestMetricsHistory:
    """MetricsHistory stores episode and PPO data correctly."""

    def test_add_episode_data_stores_win_rates(self, manager):
        win_rates = {"win_rate_black": 50.0, "win_rate_white": 30.0, "win_rate_draw": 20.0}
        manager.history.add_episode_data(win_rates)
        assert len(manager.history.win_rates_history) == 1
        assert manager.history.win_rates_history[0] == win_rates

    def test_add_ppo_data_stores_all_metrics(self, manager, ppo_metrics):
        manager.history.add_ppo_data(ppo_metrics)
        assert len(manager.history.learning_rates) == 1
        assert len(manager.history.policy_losses) == 1
        assert len(manager.history.value_losses) == 1
        assert len(manager.history.kl_divergences) == 1
        assert len(manager.history.entropies) == 1
        assert len(manager.history.clip_fractions) == 1

    def test_add_ppo_data_stores_correct_values(self, manager, ppo_metrics):
        manager.history.add_ppo_data(ppo_metrics)
        assert manager.history.learning_rates[0] == pytest.approx(3e-4)
        assert manager.history.policy_losses[0] == pytest.approx(0.1234)

    def test_add_ppo_data_handles_partial_metrics(self, manager):
        partial = {MetricsHistory.PPO_POLICY_LOSS: 0.5}
        manager.history.add_ppo_data(partial)
        assert len(manager.history.policy_losses) == 1
        assert len(manager.history.learning_rates) == 0

    def test_history_respects_max_history(self):
        history = MetricsHistory(max_history=3)
        for i in range(5):
            history.add_episode_data({"val": float(i)})
        assert len(history.win_rates_history) == 3
        # Should keep the latest entries
        assert history.win_rates_history[0] == {"val": 2.0}


# ---------------------------------------------------------------------------
# 7. format_episode_metrics
# ---------------------------------------------------------------------------
class TestFormatEpisodeMetrics:
    """format_episode_metrics returns a properly formatted string."""

    def test_contains_episode_number(self, manager):
        manager.update_episode_stats(Color.BLACK)
        result = manager.format_episode_metrics(episode_length=50, episode_reward=1.0)
        assert "Ep 1" in result

    def test_contains_episode_length(self, manager):
        result = manager.format_episode_metrics(episode_length=120, episode_reward=0.5)
        assert "Len=120" in result

    def test_contains_reward(self, manager):
        result = manager.format_episode_metrics(episode_length=50, episode_reward=1.234)
        assert "R=1.234" in result

    def test_contains_win_rate_percentages(self, manager):
        manager.update_episode_stats(Color.BLACK)
        result = manager.format_episode_metrics(episode_length=50, episode_reward=1.0)
        assert "B=" in result
        assert "W=" in result
        assert "D=" in result
        assert "%" in result


# ---------------------------------------------------------------------------
# 8. format_ppo_metrics
# ---------------------------------------------------------------------------
class TestFormatPpoMetrics:
    """format_ppo_metrics returns formatted string with all expected fields."""

    def test_contains_all_metric_abbreviations(self, manager, ppo_metrics):
        result = manager.format_ppo_metrics(ppo_metrics)
        assert "LR:" in result
        assert "KL:" in result
        assert "PolL:" in result
        assert "ValL:" in result
        assert "Ent:" in result
        assert "CF:" in result

    def test_does_not_store_data_in_history(self, manager, ppo_metrics):
        """format_ppo_metrics is a pure query — it does not record to history."""
        manager.format_ppo_metrics(ppo_metrics)
        assert len(manager.history.policy_losses) == 0

    def test_empty_metrics_returns_empty_string(self, manager):
        result = manager.format_ppo_metrics({})
        assert result == ""

    def test_partial_metrics_includes_only_present_fields(self, manager):
        partial = {MetricsHistory.PPO_POLICY_LOSS: 0.1234}
        result = manager.format_ppo_metrics(partial)
        assert "PolL:0.1234" in result
        assert "LR:" not in result


# ---------------------------------------------------------------------------
# 9. format_ppo_metrics_for_logging
# ---------------------------------------------------------------------------
class TestFormatPpoMetricsForLogging:
    """format_ppo_metrics_for_logging returns JSON-formatted string."""

    def test_returns_valid_json(self, manager, ppo_metrics):
        result = manager.format_ppo_metrics_for_logging(ppo_metrics)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_all_values_formatted_to_4_decimal_places(self, manager):
        metrics = {"some_metric": 1.23456789}
        result = manager.format_ppo_metrics_for_logging(metrics)
        parsed = json.loads(result)
        assert parsed["some_metric"] == "1.2346"


# ---------------------------------------------------------------------------
# 10. get_final_stats
# ---------------------------------------------------------------------------
class TestGetFinalStats:
    """get_final_stats returns a complete stats dictionary."""

    def test_returns_dict_with_all_keys(self, manager):
        result = manager.get_final_stats()
        expected_keys = {
            "black_wins",
            "white_wins",
            "draws",
            "total_episodes_completed",
            "global_timestep",
            "elo_state",
            "metrics_history",
        }
        assert set(result.keys()) == expected_keys

    def test_reflects_current_state(self, manager):
        manager.update_episode_stats(Color.BLACK)
        manager.update_episode_stats(Color.WHITE)
        manager.increment_timestep_by(100)

        result = manager.get_final_stats()
        assert result["black_wins"] == 1
        assert result["white_wins"] == 1
        assert result["draws"] == 0
        assert result["total_episodes_completed"] == 2
        assert result["global_timestep"] == 100


# ---------------------------------------------------------------------------
# 11. Progress updates management
# ---------------------------------------------------------------------------
class TestProgressUpdates:
    """update_progress_metrics, get_progress_updates, clear_progress_updates."""

    def test_update_stores_key_value_pair(self, manager):
        manager.update_progress_metrics("speed", 42.0)
        assert manager.pending_progress_updates["speed"] == 42.0

    def test_get_returns_copy_of_updates(self, manager):
        manager.update_progress_metrics("epoch", 5)
        updates = manager.get_progress_updates()
        assert updates == {"epoch": 5}
        # Verify it is a copy
        updates["epoch"] = 999
        assert manager.pending_progress_updates["epoch"] == 5

    def test_clear_empties_updates(self, manager):
        manager.update_progress_metrics("speed", 42.0)
        manager.update_progress_metrics("epoch", 1)
        manager.clear_progress_updates()
        assert manager.get_progress_updates() == {}

    def test_multiple_updates_overwrite_same_key(self, manager):
        manager.update_progress_metrics("speed", 10.0)
        manager.update_progress_metrics("speed", 20.0)
        assert manager.pending_progress_updates["speed"] == 20.0


# ---------------------------------------------------------------------------
# 12. restore_from_checkpoint
# ---------------------------------------------------------------------------
class TestRestoreFromCheckpoint:
    """restore_from_checkpoint restores all counters from checkpoint data."""

    def test_restores_all_counters(self, manager):
        checkpoint_data = {
            "global_timestep": 5000,
            "total_episodes_completed": 200,
            "black_wins": 80,
            "white_wins": 70,
            "draws": 50,
        }
        manager.restore_from_checkpoint(checkpoint_data)

        assert manager.global_timestep == 5000
        assert manager.total_episodes_completed == 200
        assert manager.stats.black_wins == 80
        assert manager.stats.white_wins == 70
        assert manager.stats.draws == 50

    def test_handles_missing_keys_with_defaults(self, manager):
        manager.restore_from_checkpoint({})
        assert manager.global_timestep == 0
        assert manager.total_episodes_completed == 0
        assert manager.stats.black_wins == 0
        assert manager.stats.white_wins == 0
        assert manager.stats.draws == 0

    def test_handles_partial_checkpoint(self, manager):
        checkpoint_data = {
            "global_timestep": 1000,
            "black_wins": 50,
        }
        manager.restore_from_checkpoint(checkpoint_data)
        assert manager.global_timestep == 1000
        assert manager.stats.black_wins == 50
        assert manager.total_episodes_completed == 0
        assert manager.stats.white_wins == 0
        assert manager.stats.draws == 0

    def test_roundtrip_get_final_stats_then_restore(self, manager):
        """get_final_stats output can be fed back into restore_from_checkpoint."""
        manager.update_episode_stats(Color.BLACK)
        manager.update_episode_stats(Color.WHITE)
        manager.update_episode_stats(None)
        manager.increment_timestep_by(500)

        saved = manager.get_final_stats()

        new_manager = MetricsManager()
        new_manager.restore_from_checkpoint(saved)

        assert new_manager.global_timestep == 500
        assert new_manager.total_episodes_completed == 3
        assert new_manager.stats.black_wins == 1
        assert new_manager.stats.white_wins == 1
        assert new_manager.stats.draws == 1

    def test_roundtrip_preserves_elo_state(self, manager):
        """Elo ratings survive a save/restore roundtrip."""
        # Play some games to move Elo ratings away from default
        manager.update_episode_stats(Color.BLACK)
        manager.update_episode_stats(Color.BLACK)
        manager.update_episode_stats(Color.WHITE)
        original_black = manager.elo_system.black_rating
        original_white = manager.elo_system.white_rating

        saved = manager.get_final_stats()
        new_manager = MetricsManager()
        new_manager.restore_from_checkpoint(saved)

        assert new_manager.elo_system.black_rating == original_black
        assert new_manager.elo_system.white_rating == original_white
        assert len(new_manager.elo_system.rating_history) == 3

    def test_roundtrip_preserves_metrics_history(self, manager):
        """PPO trend history survives a save/restore roundtrip."""
        manager.history.add_ppo_data({
            "ppo/policy_loss": 0.5,
            "ppo/value_loss": 1.2,
            "ppo/entropy": 3.0,
        })
        manager.history.add_ppo_data({
            "ppo/policy_loss": 0.4,
            "ppo/value_loss": 1.0,
            "ppo/entropy": 2.8,
        })

        saved = manager.get_final_stats()
        new_manager = MetricsManager()
        new_manager.restore_from_checkpoint(saved)

        assert list(new_manager.history.policy_losses) == [0.5, 0.4]
        assert list(new_manager.history.value_losses) == [1.2, 1.0]
        assert list(new_manager.history.entropies) == [3.0, 2.8]

    def test_restore_old_checkpoint_without_elo_or_history(self, manager):
        """Old checkpoints without elo_state/metrics_history restore gracefully."""
        old_checkpoint = {
            "global_timestep": 1000,
            "total_episodes_completed": 50,
            "black_wins": 20,
            "white_wins": 20,
            "draws": 10,
        }
        manager.restore_from_checkpoint(old_checkpoint)

        # Basic stats restored
        assert manager.global_timestep == 1000
        # Elo stays at defaults
        assert manager.elo_system.black_rating == 1500.0
        # History stays empty
        assert len(manager.history.policy_losses) == 0


# ---------------------------------------------------------------------------
# 13. Processing state
# ---------------------------------------------------------------------------
class TestProcessingState:
    """set_processing controls the processing flag."""

    def test_default_is_false(self, manager):
        assert manager.processing is False

    def test_set_processing_true(self, manager):
        manager.set_processing(True)
        assert manager.processing is True

    def test_set_processing_false(self, manager):
        manager.set_processing(True)
        manager.set_processing(False)
        assert manager.processing is False


# ---------------------------------------------------------------------------
# 14. Property setters for backward compatibility
# ---------------------------------------------------------------------------
class TestPropertySetters:
    """Backward-compatible property setters write to underlying stats."""

    def test_black_wins_setter(self, manager):
        manager.black_wins = 10
        assert manager.stats.black_wins == 10

    def test_white_wins_setter(self, manager):
        manager.white_wins = 20
        assert manager.stats.white_wins == 20

    def test_draws_setter(self, manager):
        manager.draws = 5
        assert manager.stats.draws == 5


# ---------------------------------------------------------------------------
# 15. TrainingStats dataclass
# ---------------------------------------------------------------------------
class TestTrainingStats:
    """TrainingStats defaults and structure."""

    def test_defaults_are_zero(self):
        stats = TrainingStats()
        assert stats.global_timestep == 0
        assert stats.total_episodes_completed == 0
        assert stats.black_wins == 0
        assert stats.white_wins == 0
        assert stats.draws == 0


# ---------------------------------------------------------------------------
# 16. Initialization
# ---------------------------------------------------------------------------
class TestMetricsManagerInit:
    """MetricsManager __init__ sets up all internal structures."""

    def test_custom_history_size(self):
        mm = MetricsManager(history_size=50)
        assert mm.history.max_history == 50

    def test_custom_elo_params(self):
        mm = MetricsManager(elo_initial_rating=1200.0, elo_k_factor=16.0)
        assert mm.elo_system.initial_rating == 1200.0
        assert mm.elo_system.k_factor == 16.0

    def test_deques_start_empty(self, manager):
        assert len(manager.moves_per_game) == 0
        assert len(manager.turns_per_game) == 0
        assert len(manager.games_completed_timestamps) == 0
        assert len(manager.win_loss_draw_history) == 0
        assert len(manager.sente_opening_history) == 0
        assert len(manager.gote_opening_history) == 0
        assert len(manager.square_usage) == 0
