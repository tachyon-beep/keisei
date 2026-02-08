"""Tests: EnvelopeParser read-only access layer.

Covers eva.3 acceptance criteria:
- Parser provides typed access to envelope fields (eva.3.1)
- Missing optional views are correctly reported (eva.3.2)
- Stale detection uses contract threshold (eva.3.3)
"""

import time

import pytest

from keisei.webui.envelope_parser import EnvelopeParser
from keisei.webui.view_contracts import (
    SCHEMA_VERSION,
    STALE_THRESHOLD_SECONDS,
    VIEW_KEYS,
    make_health_map,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_envelope(ts=None, speed=42.5, mode="single_opponent", **overrides):
    """Build a minimal valid envelope for parser tests."""
    env = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": ts if ts is not None else time.time(),
        "speed": speed,
        "mode": mode,
        "active_views": ["training"],
        "health": make_health_map(training="ok"),
        "training": {
            "board_state": {
                "board": [[None] * 9 for _ in range(9)],
                "current_player": "black",
                "move_count": 5,
                "game_over": False,
                "winner": None,
                "black_hand": {},
                "white_hand": {},
            },
            "metrics": {
                "global_timestep": 100,
                "total_episodes": 5,
                "black_wins": 3,
                "white_wins": 1,
                "draws": 1,
                "processing": False,
                "learning_curves": {
                    "policy_losses": [0.5],
                    "value_losses": [0.3],
                    "entropies": [1.0],
                    "kl_divergences": [0.01],
                    "clip_fractions": [0.1],
                    "learning_rates": [3e-4],
                    "episode_lengths": [120.0],
                    "episode_rewards": [0.5],
                },
                "win_rates_history": [],
                "hot_squares": [],
            },
            "step_info": {
                "move_log": ["P7g-7f"],
                "sente_capture_count": 0,
                "gote_capture_count": 0,
                "sente_drop_count": 0,
                "gote_drop_count": 0,
                "sente_promo_count": 0,
                "gote_promo_count": 0,
            },
            "buffer_info": {"size": 50, "capacity": 2048},
            "model_info": {"gradient_norm": 1.23},
        },
        "pending_updates": {"epoch": 3},
    }
    env.update(overrides)
    return env


# ---------------------------------------------------------------------------
# Eva.3.1: Envelope metadata access
# ---------------------------------------------------------------------------


class TestEnvelopeMetadata:
    """Parser provides correct envelope-level fields."""

    def test_schema_version(self):
        p = EnvelopeParser(_make_envelope())
        assert p.schema_version == SCHEMA_VERSION

    def test_timestamp(self):
        ts = 1700000000.0
        p = EnvelopeParser(_make_envelope(ts=ts))
        assert p.timestamp == ts

    def test_speed(self):
        p = EnvelopeParser(_make_envelope(speed=99.9))
        assert p.speed == 99.9

    def test_mode(self):
        p = EnvelopeParser(_make_envelope(mode="tournament"))
        assert p.mode == "tournament"

    def test_active_views(self):
        p = EnvelopeParser(_make_envelope())
        assert p.active_views == ["training"]

    def test_health(self):
        p = EnvelopeParser(_make_envelope())
        assert p.health["training"] == "ok"

    def test_pending_updates(self):
        p = EnvelopeParser(_make_envelope())
        assert p.pending_updates == {"epoch": 3}


class TestTrainingViewAccess:
    """Parser navigates training view sub-keys."""

    def test_board_state(self):
        p = EnvelopeParser(_make_envelope())
        assert p.board_state is not None
        assert p.board_state["move_count"] == 5

    def test_metrics(self):
        p = EnvelopeParser(_make_envelope())
        assert p.metrics["global_timestep"] == 100

    def test_step_info(self):
        p = EnvelopeParser(_make_envelope())
        assert p.step_info["move_log"] == ["P7g-7f"]

    def test_buffer_info(self):
        p = EnvelopeParser(_make_envelope())
        assert p.buffer_info["size"] == 50

    def test_model_info(self):
        p = EnvelopeParser(_make_envelope())
        assert p.model_info["gradient_norm"] == 1.23


class TestNullableTrainingKeys:
    """Parser returns None/empty when training sub-keys are absent."""

    def test_no_training_key(self):
        raw = _make_envelope()
        del raw["training"]
        p = EnvelopeParser(raw)
        assert p.training is None
        assert p.board_state is None
        assert p.metrics == {}
        assert p.step_info is None
        assert p.buffer_info is None
        assert p.model_info == {}

    def test_nullable_board_state(self):
        raw = _make_envelope()
        raw["training"]["board_state"] = None
        p = EnvelopeParser(raw)
        assert p.board_state is None

    def test_nullable_step_info(self):
        raw = _make_envelope()
        raw["training"]["step_info"] = None
        p = EnvelopeParser(raw)
        assert p.step_info is None

    def test_nullable_buffer_info(self):
        raw = _make_envelope()
        raw["training"]["buffer_info"] = None
        p = EnvelopeParser(raw)
        assert p.buffer_info is None


class TestDefaults:
    """Parser returns sensible defaults for empty/missing raw data."""

    def test_empty_raw(self):
        p = EnvelopeParser({})
        assert p.schema_version == ""
        assert p.timestamp == 0.0
        assert p.speed == 0.0
        assert p.mode == "training_only"
        assert p.active_views == []
        assert p.health == {}
        assert p.pending_updates == {}
        assert p.training is None
        assert p.board_state is None
        assert p.metrics == {}


# ---------------------------------------------------------------------------
# Eva.3.2: Optional view tracking
# ---------------------------------------------------------------------------


class TestOptionalViews:
    """Parser tracks which optional views are available/missing."""

    def test_has_view_training(self):
        p = EnvelopeParser(_make_envelope())
        assert p.has_view("training") is True

    def test_has_view_league_missing(self):
        p = EnvelopeParser(_make_envelope())
        assert p.has_view("league") is False

    def test_missing_optional_views_default(self):
        p = EnvelopeParser(_make_envelope())
        missing = p.missing_optional_views()
        assert set(missing) == {"league", "lineage", "skill_differential", "model_profile"}

    def test_available_optional_views_default(self):
        p = EnvelopeParser(_make_envelope())
        assert p.available_optional_views() == []

    def test_with_league_active(self):
        raw = _make_envelope()
        raw["active_views"] = ["training", "league"]
        p = EnvelopeParser(raw)
        assert p.has_view("league") is True
        assert "league" not in p.missing_optional_views()
        assert p.available_optional_views() == ["league"]

    def test_view_health_for_missing(self):
        p = EnvelopeParser(_make_envelope())
        assert p.view_health("league") == "missing"

    def test_view_health_for_training(self):
        p = EnvelopeParser(_make_envelope())
        assert p.view_health("training") == "ok"

    def test_view_health_unknown_key(self):
        p = EnvelopeParser(_make_envelope())
        assert p.view_health("nonexistent") == "missing"


# ---------------------------------------------------------------------------
# Eva.3.3: Stale detection
# ---------------------------------------------------------------------------


class TestStaleDetection:
    """Stale detection uses contract-defined threshold."""

    def test_fresh_snapshot_is_not_stale(self):
        p = EnvelopeParser(_make_envelope(ts=time.time()))
        assert p.is_stale() is False

    def test_old_snapshot_is_stale(self):
        old_ts = time.time() - STALE_THRESHOLD_SECONDS - 10
        p = EnvelopeParser(_make_envelope(ts=old_ts))
        assert p.is_stale() is True

    def test_zero_timestamp_is_stale(self):
        p = EnvelopeParser(_make_envelope(ts=0))
        assert p.is_stale() is True

    def test_custom_threshold(self):
        ts = time.time() - 5  # 5 seconds old
        p = EnvelopeParser(_make_envelope(ts=ts))
        assert p.is_stale(threshold=3.0) is True
        assert p.is_stale(threshold=10.0) is False

    def test_age_seconds_fresh(self):
        ts = time.time()
        p = EnvelopeParser(_make_envelope(ts=ts))
        assert p.age_seconds() < 1.0

    def test_age_seconds_zero_timestamp(self):
        p = EnvelopeParser(_make_envelope(ts=0))
        assert p.age_seconds() == float("inf")

    def test_missing_timestamp(self):
        p = EnvelopeParser({})
        assert p.is_stale() is True
        assert p.age_seconds() == float("inf")
