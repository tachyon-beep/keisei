"""Renderer fallback tests — verify graceful degradation.

Tests that the EnvelopeParser and rendering-adjacent logic handle
absent optional views, stale envelopes, and partial data without
exceptions.  These do NOT require a running Streamlit process —
they test the data layer that feeds the renderer.

Covers eva.4.2 acceptance criteria.
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
# Helpers
# ---------------------------------------------------------------------------


def _minimal_envelope(**overrides):
    """Minimal valid envelope for fallback tests."""
    env = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.time(),
        "speed": 10.0,
        "mode": "training_only",
        "active_views": ["training"],
        "health": make_health_map(training="ok"),
        "training": {
            "board_state": None,
            "metrics": {
                "global_timestep": 0,
                "total_episodes": 0,
                "black_wins": 0,
                "white_wins": 0,
                "draws": 0,
                "processing": False,
                "learning_curves": {
                    "policy_losses": [],
                    "value_losses": [],
                    "entropies": [],
                    "kl_divergences": [],
                    "clip_fractions": [],
                    "learning_rates": [],
                    "episode_lengths": [],
                    "episode_rewards": [],
                },
                "win_rates_history": [],
                "hot_squares": [],
            },
            "step_info": None,
            "buffer_info": None,
            "model_info": {"gradient_norm": 0.0},
        },
        "pending_updates": {},
    }
    env.update(overrides)
    return env


# ---------------------------------------------------------------------------
# Missing optional views
# ---------------------------------------------------------------------------


class TestMissingOptionalViews:
    """Parser handles missing optional views gracefully."""

    def test_all_optional_views_missing_by_default(self):
        p = EnvelopeParser(_minimal_envelope())
        for view in ("league", "lineage", "skill_differential", "model_profile"):
            assert p.view_health(view) == "missing"
            assert p.has_view(view) is False

    def test_missing_views_list(self):
        p = EnvelopeParser(_minimal_envelope())
        missing = p.missing_optional_views()
        assert len(missing) == 4
        assert "training" not in missing

    def test_partial_optional_views(self):
        """When some optional views become active, others stay missing."""
        raw = _minimal_envelope()
        raw["active_views"] = ["training", "league"]
        raw["health"]["league"] = "ok"
        p = EnvelopeParser(raw)
        assert p.has_view("league") is True
        assert p.has_view("lineage") is False
        assert "league" not in p.missing_optional_views()
        assert "lineage" in p.missing_optional_views()


# ---------------------------------------------------------------------------
# Stale envelopes
# ---------------------------------------------------------------------------


class TestStaleEnvelopes:
    """Parser handles stale and partially-available states."""

    def test_very_old_snapshot(self):
        raw = _minimal_envelope(timestamp=1000000000.0)  # year 2001
        p = EnvelopeParser(raw)
        assert p.is_stale() is True
        assert p.age_seconds() > 1000

    def test_missing_timestamp_field(self):
        raw = _minimal_envelope()
        del raw["timestamp"]
        p = EnvelopeParser(raw)
        assert p.is_stale() is True
        assert p.timestamp == 0.0

    def test_stale_with_health_error(self):
        """Stale snapshot with error health is handled without exception."""
        raw = _minimal_envelope(timestamp=time.time() - 60)
        raw["health"]["training"] = "error"
        p = EnvelopeParser(raw)
        assert p.is_stale() is True
        assert p.view_health("training") == "error"

    def test_just_under_threshold_is_not_stale(self):
        ts = time.time() - (STALE_THRESHOLD_SECONDS - 1)
        p = EnvelopeParser(_minimal_envelope(timestamp=ts))
        assert p.is_stale() is False

    def test_just_over_threshold_is_stale(self):
        ts = time.time() - (STALE_THRESHOLD_SECONDS + 1)
        p = EnvelopeParser(_minimal_envelope(timestamp=ts))
        assert p.is_stale() is True


# ---------------------------------------------------------------------------
# Partial / degraded training data
# ---------------------------------------------------------------------------


class TestPartialTrainingData:
    """Parser tolerates partial training data without exceptions."""

    def test_empty_learning_curves(self):
        p = EnvelopeParser(_minimal_envelope())
        curves = p.metrics.get("learning_curves", {})
        for key in ("policy_losses", "value_losses", "entropies"):
            assert curves.get(key) == []

    def test_no_board_state(self):
        p = EnvelopeParser(_minimal_envelope())
        assert p.board_state is None

    def test_no_step_info(self):
        p = EnvelopeParser(_minimal_envelope())
        assert p.step_info is None

    def test_no_buffer_info(self):
        p = EnvelopeParser(_minimal_envelope())
        assert p.buffer_info is None

    def test_metrics_always_present(self):
        p = EnvelopeParser(_minimal_envelope())
        assert isinstance(p.metrics, dict)
        assert "global_timestep" in p.metrics

    def test_model_info_always_present(self):
        p = EnvelopeParser(_minimal_envelope())
        assert isinstance(p.model_info, dict)

    def test_completely_empty_training_dict(self):
        """Training is an empty dict — all sub-keys return None/empty."""
        raw = _minimal_envelope()
        raw["training"] = {}
        p = EnvelopeParser(raw)
        assert p.board_state is None
        assert p.metrics == {}
        assert p.step_info is None
        assert p.buffer_info is None
        assert p.model_info == {}

    def test_no_training_key_at_all(self):
        """Missing 'training' key — parser degrades cleanly."""
        raw = _minimal_envelope()
        del raw["training"]
        p = EnvelopeParser(raw)
        assert p.training is None
        assert p.board_state is None
        assert p.metrics == {}


# ---------------------------------------------------------------------------
# Corrupt / unexpected data
# ---------------------------------------------------------------------------


class TestUnexpectedData:
    """Parser doesn't crash on malformed or unexpected values."""

    def test_non_dict_health(self):
        raw = _minimal_envelope()
        raw["health"] = "broken"
        p = EnvelopeParser(raw)
        # health property returns whatever is in the raw dict
        assert p.view_health("training") == "missing"  # .get on non-dict falls through

    def test_non_list_active_views(self):
        raw = _minimal_envelope()
        raw["active_views"] = "training"
        p = EnvelopeParser(raw)
        # has_view should still work (string "in" test)
        assert p.has_view("training") is True

    def test_extra_envelope_keys_ignored(self):
        raw = _minimal_envelope()
        raw["unexpected_future_field"] = {"data": True}
        p = EnvelopeParser(raw)
        assert p.schema_version == SCHEMA_VERSION
        assert p.metrics.get("global_timestep") == 0
