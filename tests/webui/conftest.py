"""WebUI test tier: contract, snapshot, and renderer tests.

Fixtures here provide canonical v1 broadcast envelope payloads — both valid
and intentionally invalid — for locking schema behaviour.
"""

import time

import pytest

from keisei.webui.view_contracts import (
    SCHEMA_VERSION,
    make_health_map,
)


# ---------------------------------------------------------------------------
# Canonical valid payload factories
# ---------------------------------------------------------------------------


def _make_valid_training_view():
    """Minimal valid TrainingViewState with all required sub-keys."""
    return {
        "board_state": {
            "board": [[None] * 9 for _ in range(9)],
            "current_player": "black",
            "move_count": 0,
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
                "policy_losses": [0.5, 0.4],
                "value_losses": [0.3, 0.2],
                "entropies": [1.0, 0.9],
                "kl_divergences": [0.01, 0.02],
                "clip_fractions": [0.1, 0.08],
                "learning_rates": [3e-4, 3e-4],
                "episode_lengths": [120.0, 130.0],
                "episode_rewards": [0.5, 0.6],
            },
            "win_rates_history": [
                {"win_rate_black": 0.6, "win_rate_white": 0.2, "win_rate_draw": 0.2},
            ],
            "hot_squares": [],
        },
        "step_info": {
            "move_log": ["P7g-7f", "P3c-3d"],
            "sente_capture_count": 0,
            "gote_capture_count": 0,
            "sente_drop_count": 0,
            "gote_drop_count": 0,
            "sente_promo_count": 0,
            "gote_promo_count": 0,
        },
        "buffer_info": {
            "size": 50,
            "capacity": 2048,
        },
        "model_info": {
            "gradient_norm": 1.23,
        },
    }


def _make_valid_envelope(ts=None):
    """Build a canonical valid BroadcastStateEnvelope."""
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": ts if ts is not None else time.time(),
        "speed": 42.5,
        "mode": "single_opponent",
        "active_views": ["training"],
        "health": make_health_map(training="ok"),
        "training": _make_valid_training_view(),
        "pending_updates": {"epoch": 3, "lr": 0.0003},
    }


def _make_valid_envelope_nullable_training(ts=None):
    """Valid envelope with nullable training sub-keys (between episodes)."""
    env = _make_valid_envelope(ts)
    env["training"]["board_state"] = None
    env["training"]["step_info"] = None
    env["training"]["buffer_info"] = None
    return env


def _make_valid_envelope_training_only_mode(ts=None):
    """Valid envelope with mode='training_only' (no evaluation)."""
    env = _make_valid_envelope(ts)
    env["mode"] = "training_only"
    return env


# ---------------------------------------------------------------------------
# Canonical invalid payload factories
# ---------------------------------------------------------------------------


def _make_invalid_missing_speed():
    """Invalid: missing required 'speed' field."""
    env = _make_valid_envelope()
    del env["speed"]
    return env


def _make_invalid_missing_schema_version():
    """Invalid: missing required 'schema_version' field."""
    env = _make_valid_envelope()
    del env["schema_version"]
    return env


def _make_invalid_bad_health_status():
    """Invalid: health map contains unrecognised status value."""
    env = _make_valid_envelope()
    env["health"]["training"] = "borked"
    return env


def _make_invalid_incomplete_health():
    """Invalid: health map missing view keys."""
    env = _make_valid_envelope()
    env["health"] = {"training": "ok"}  # missing 4 keys
    return env


def _make_invalid_non_scalar_pending():
    """Invalid: pending_updates contains a list (non-scalar)."""
    env = _make_valid_envelope()
    env["pending_updates"] = {"ok": 1, "bad": [1, 2, 3]}
    return env


def _make_invalid_legacy_top_level_keys():
    """Invalid: legacy top-level key layout (pre-v1 snapshot shape).

    Simulates the current build_snapshot() output where board_state,
    metrics, etc. are top-level keys with no wrapping 'training' key.
    The validator should catch the missing 'training' required key.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.time(),
        "speed": 10.0,
        "mode": "single_opponent",
        "active_views": ["training"],
        "health": make_health_map(training="ok"),
        "pending_updates": {},
        # Legacy: sub-keys at top level instead of nested under 'training'
        "board_state": None,
        "metrics": {},
        "step_info": None,
        "buffer_info": None,
        "model_info": {},
    }


# ---------------------------------------------------------------------------
# Edge-case payload factories (structurally valid)
# ---------------------------------------------------------------------------


def _make_envelope_unknown_mode():
    """Envelope with an unknown mode value (structurally valid).

    validate_envelope() only checks that mode is a string, not that it's
    a known BroadcastMode value.  This is intentional per Decision Freeze
    #6: unknown modes must be treated as opaque strings by renderers.
    """
    env = _make_valid_envelope()
    env["mode"] = "quantum_evaluation"
    return env


# ---------------------------------------------------------------------------
# Pytest fixtures — valid envelopes
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_envelope():
    """Canonical valid v1 BroadcastStateEnvelope."""
    return _make_valid_envelope()


@pytest.fixture
def valid_envelope_nullable():
    """Valid envelope with nullable training sub-keys."""
    return _make_valid_envelope_nullable_training()


@pytest.fixture
def valid_envelope_training_only():
    """Valid envelope with mode='training_only'."""
    return _make_valid_envelope_training_only_mode()


# ---------------------------------------------------------------------------
# Pytest fixtures — invalid envelopes (parametrised + individual)
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        ("missing_speed", _make_invalid_missing_speed),
        ("missing_schema_version", _make_invalid_missing_schema_version),
        ("bad_health_status", _make_invalid_bad_health_status),
        ("incomplete_health", _make_invalid_incomplete_health),
        ("non_scalar_pending", _make_invalid_non_scalar_pending),
        ("legacy_top_level_keys", _make_invalid_legacy_top_level_keys),
    ],
    ids=lambda p: p[0],
)
def invalid_envelope(request):
    """Parametrised fixture yielding each invalid envelope variant.

    Each variant is a (name, payload) tuple.  Tests receive the payload;
    the ``name`` is used for readable test IDs.
    """
    name, factory = request.param
    return factory()


@pytest.fixture
def invalid_missing_speed():
    """Invalid envelope: missing required 'speed' field."""
    return _make_invalid_missing_speed()


@pytest.fixture
def invalid_missing_schema_version():
    """Invalid envelope: missing required 'schema_version' field."""
    return _make_invalid_missing_schema_version()


@pytest.fixture
def invalid_bad_health_status():
    """Invalid envelope: unrecognised health status value."""
    return _make_invalid_bad_health_status()


@pytest.fixture
def invalid_incomplete_health():
    """Invalid envelope: health map missing view keys."""
    return _make_invalid_incomplete_health()


@pytest.fixture
def invalid_non_scalar_pending():
    """Invalid envelope: pending_updates contains non-scalar value."""
    return _make_invalid_non_scalar_pending()


@pytest.fixture
def invalid_legacy_top_level_keys():
    """Invalid envelope: legacy top-level key layout (pre-v1 shape)."""
    return _make_invalid_legacy_top_level_keys()


# ---------------------------------------------------------------------------
# Pytest fixtures — edge cases
# ---------------------------------------------------------------------------


@pytest.fixture
def envelope_unknown_mode():
    """Envelope with unknown mode value (valid per Decision Freeze #6)."""
    return _make_envelope_unknown_mode()
