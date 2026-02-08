"""Schema lock tests — detect accidental drift in v1 contract definitions.

These tests pin the exact key sets, constants, and type structures of the
v1 broadcast contract.  Any addition, removal, or rename of a required field
will break at least one test here.  This is intentional: schema changes must
be deliberate and accompanied by a schema_version bump.

Covers eva.4.1 acceptance criteria.
"""

import time
from types import SimpleNamespace

import pytest

from keisei.webui.envelope_parser import EnvelopeParser
from keisei.webui.state_snapshot import build_snapshot
from keisei.webui.view_contracts import (
    SCHEMA_VERSION,
    STALE_THRESHOLD_SECONDS,
    VIEW_KEYS,
    _REQUIRED_ENVELOPE_KEYS,
    _VALID_HEALTH_STATUSES,
    make_health_map,
    sanitize_pending_updates,
    validate_envelope,
)


# ---------------------------------------------------------------------------
# Constant locks
# ---------------------------------------------------------------------------


class TestConstantLocks:
    """Contract constants are frozen at their v1 values."""

    def test_schema_version_is_v1_0_0(self):
        assert SCHEMA_VERSION == "v1.0.0"

    def test_stale_threshold_is_30_seconds(self):
        assert STALE_THRESHOLD_SECONDS == 30.0

    def test_view_keys_exact(self):
        assert VIEW_KEYS == (
            "training",
            "league",
            "lineage",
            "skill_differential",
            "model_profile",
        )

    def test_valid_health_statuses_exact(self):
        assert _VALID_HEALTH_STATUSES == {"ok", "stale", "missing", "error"}

    def test_required_envelope_keys_exact(self):
        assert _REQUIRED_ENVELOPE_KEYS == {
            "schema_version",
            "timestamp",
            "speed",
            "mode",
            "active_views",
            "health",
            "training",
            "pending_updates",
        }


# ---------------------------------------------------------------------------
# Training view key locks
# ---------------------------------------------------------------------------


class TestTrainingViewKeyLock:
    """TrainingViewState has exactly these 5 sub-keys."""

    EXPECTED_TRAINING_KEYS = {
        "board_state",
        "metrics",
        "step_info",
        "buffer_info",
        "model_info",
    }

    def test_producer_emits_exact_training_keys(self):
        """build_snapshot training payload has no extra or missing keys."""
        mm = SimpleNamespace(
            global_timestep=0, total_episodes_completed=0,
            black_wins=0, white_wins=0, draws=0, processing=False,
            history=SimpleNamespace(
                policy_losses=[], value_losses=[], entropies=[],
                kl_divergences=[], clip_fractions=[], learning_rates=[],
                episode_lengths=[], episode_rewards=[], win_rates_history=[],
            ),
            get_hot_squares=lambda n: [],
        )
        trainer = SimpleNamespace(
            config=None, game=None, metrics_manager=mm,
            step_manager=None, experience_buffer=None,
            last_gradient_norm=0.0,
        )
        snapshot = build_snapshot(trainer)
        assert set(snapshot["training"].keys()) == self.EXPECTED_TRAINING_KEYS

    def test_parser_exposes_all_training_keys(self):
        """EnvelopeParser has a property for each training sub-key."""
        parser_props = {"board_state", "metrics", "step_info", "buffer_info", "model_info"}
        actual = {name for name in dir(EnvelopeParser) if name in parser_props}
        assert actual == parser_props


# ---------------------------------------------------------------------------
# Learning curves key lock
# ---------------------------------------------------------------------------


class TestLearningCurvesKeyLock:
    """LearningCurves has exactly these 8 metric series."""

    EXPECTED_CURVE_KEYS = {
        "policy_losses",
        "value_losses",
        "entropies",
        "kl_divergences",
        "clip_fractions",
        "learning_rates",
        "episode_lengths",
        "episode_rewards",
    }

    def test_learning_curves_keys_match(self):
        """Metrics from producer include exact learning curve keys."""
        mm = SimpleNamespace(
            global_timestep=10, total_episodes_completed=1,
            black_wins=0, white_wins=0, draws=1, processing=False,
            history=SimpleNamespace(
                policy_losses=[0.5], value_losses=[0.3], entropies=[1.0],
                kl_divergences=[0.01], clip_fractions=[0.1],
                learning_rates=[3e-4], episode_lengths=[100.0],
                episode_rewards=[0.5], win_rates_history=[],
            ),
            get_hot_squares=lambda n: [],
        )
        trainer = SimpleNamespace(
            config=None, game=None, metrics_manager=mm,
            step_manager=None, experience_buffer=None,
            last_gradient_norm=0.0,
        )
        snapshot = build_snapshot(trainer)
        curves = snapshot["training"]["metrics"]["learning_curves"]
        assert set(curves.keys()) == self.EXPECTED_CURVE_KEYS


# ---------------------------------------------------------------------------
# Envelope key lock
# ---------------------------------------------------------------------------


class TestEnvelopeKeyLock:
    """Envelope from build_snapshot has exactly the required keys."""

    def test_envelope_has_no_extra_top_level_keys(self):
        """Snapshot has only required envelope keys (no legacy top-level leaks)."""
        mm = SimpleNamespace(
            global_timestep=0, total_episodes_completed=0,
            black_wins=0, white_wins=0, draws=0, processing=False,
            history=SimpleNamespace(
                policy_losses=[], value_losses=[], entropies=[],
                kl_divergences=[], clip_fractions=[], learning_rates=[],
                episode_lengths=[], episode_rewards=[], win_rates_history=[],
            ),
            get_hot_squares=lambda n: [],
        )
        trainer = SimpleNamespace(
            config=None, game=None, metrics_manager=mm,
            step_manager=None, experience_buffer=None,
            last_gradient_norm=0.0,
        )
        snapshot = build_snapshot(trainer)
        assert set(snapshot.keys()) == _REQUIRED_ENVELOPE_KEYS

    def test_legacy_flat_keys_absent(self):
        """Old top-level keys (board_state, metrics, etc.) don't appear at envelope level."""
        mm = SimpleNamespace(
            global_timestep=0, total_episodes_completed=0,
            black_wins=0, white_wins=0, draws=0, processing=False,
            history=SimpleNamespace(
                policy_losses=[], value_losses=[], entropies=[],
                kl_divergences=[], clip_fractions=[], learning_rates=[],
                episode_lengths=[], episode_rewards=[], win_rates_history=[],
            ),
            get_hot_squares=lambda n: [],
        )
        trainer = SimpleNamespace(
            config=None, game=None, metrics_manager=mm,
            step_manager=None, experience_buffer=None,
            last_gradient_norm=0.0,
        )
        snapshot = build_snapshot(trainer)
        legacy_keys = {"board_state", "metrics", "step_info", "buffer_info", "model_info"}
        assert legacy_keys.isdisjoint(snapshot.keys())


# ---------------------------------------------------------------------------
# Round-trip: producer → JSON → parser
# ---------------------------------------------------------------------------


class TestProducerParserRoundTrip:
    """Snapshot from build_snapshot round-trips through EnvelopeParser."""

    def _make_trainer(self):
        mm = SimpleNamespace(
            global_timestep=42, total_episodes_completed=7,
            black_wins=4, white_wins=2, draws=1, processing=True,
            history=SimpleNamespace(
                policy_losses=[0.5, 0.4], value_losses=[0.3, 0.2],
                entropies=[1.0, 0.9], kl_divergences=[0.01, 0.02],
                clip_fractions=[0.1, 0.08], learning_rates=[3e-4, 3e-4],
                episode_lengths=[120.0, 130.0], episode_rewards=[0.5, 0.6],
                win_rates_history=[{"win_rate_black": 0.6}],
            ),
            get_hot_squares=lambda n: [],
        )
        eval_cfg = SimpleNamespace(enable_periodic_evaluation=True, strategy="tournament")
        config = SimpleNamespace(evaluation=eval_cfg)
        return SimpleNamespace(
            config=config, game=None, metrics_manager=mm,
            step_manager=None, experience_buffer=None,
            last_gradient_norm=1.5,
        )

    def test_round_trip_schema_version(self):
        snapshot = build_snapshot(self._make_trainer())
        p = EnvelopeParser(snapshot)
        assert p.schema_version == SCHEMA_VERSION

    def test_round_trip_mode(self):
        snapshot = build_snapshot(self._make_trainer())
        p = EnvelopeParser(snapshot)
        assert p.mode == "tournament"

    def test_round_trip_speed(self):
        snapshot = build_snapshot(self._make_trainer(), speed=55.5)
        p = EnvelopeParser(snapshot)
        assert p.speed == 55.5

    def test_round_trip_metrics(self):
        snapshot = build_snapshot(self._make_trainer())
        p = EnvelopeParser(snapshot)
        assert p.metrics["global_timestep"] == 42
        assert p.metrics["total_episodes"] == 7

    def test_round_trip_pending_updates(self):
        snapshot = build_snapshot(self._make_trainer(), pending_updates={"lr": 0.001})
        p = EnvelopeParser(snapshot)
        assert p.pending_updates == {"lr": 0.001}

    def test_round_trip_validates(self):
        snapshot = build_snapshot(self._make_trainer(), speed=10.0)
        errors = validate_envelope(snapshot)
        assert errors == [], f"Validation errors: {errors}"

    def test_round_trip_health(self):
        snapshot = build_snapshot(self._make_trainer())
        p = EnvelopeParser(snapshot)
        assert p.view_health("training") == "ok"
        assert p.view_health("league") == "missing"

    def test_round_trip_not_stale(self):
        snapshot = build_snapshot(self._make_trainer())
        p = EnvelopeParser(snapshot)
        assert p.is_stale() is False

    def test_round_trip_nullable_sub_keys(self):
        """With no game/step/buffer, parser returns None for those sub-keys."""
        snapshot = build_snapshot(self._make_trainer())
        p = EnvelopeParser(snapshot)
        assert p.board_state is None
        assert p.step_info is None
        assert p.buffer_info is None
        assert p.model_info["gradient_norm"] == 1.5


# ---------------------------------------------------------------------------
# Health map lock
# ---------------------------------------------------------------------------


class TestHealthMapLock:
    """Health map structure is locked to VIEW_KEYS."""

    def test_health_map_covers_all_view_keys(self):
        hm = make_health_map()
        assert set(hm.keys()) == set(VIEW_KEYS)

    def test_health_map_default_values(self):
        hm = make_health_map()
        for key in VIEW_KEYS:
            assert hm[key] == "missing"

    def test_producer_health_map_matches_view_keys(self):
        mm = SimpleNamespace(
            global_timestep=0, total_episodes_completed=0,
            black_wins=0, white_wins=0, draws=0, processing=False,
            history=SimpleNamespace(
                policy_losses=[], value_losses=[], entropies=[],
                kl_divergences=[], clip_fractions=[], learning_rates=[],
                episode_lengths=[], episode_rewards=[], win_rates_history=[],
            ),
            get_hot_squares=lambda n: [],
        )
        trainer = SimpleNamespace(
            config=None, game=None, metrics_manager=mm,
            step_manager=None, experience_buffer=None,
            last_gradient_norm=0.0,
        )
        snapshot = build_snapshot(trainer)
        assert set(snapshot["health"].keys()) == set(VIEW_KEYS)
