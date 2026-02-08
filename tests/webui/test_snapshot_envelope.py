"""Tests: snapshot builder emits contract-compliant envelopes.

Covers eva.2 acceptance criteria:
- Envelope structure and required keys (eva.2.1)
- Health/status population (eva.2.2)
- pending_updates sanitization (eva.2.3)
- Mode resolution from config (eva.2.2)
"""

import time
from types import SimpleNamespace

import pytest

from keisei.webui.state_snapshot import (
    _build_training_view,
    _resolve_mode,
    build_snapshot,
)
from keisei.webui.view_contracts import (
    SCHEMA_VERSION,
    VIEW_KEYS,
    validate_envelope,
)


# ---------------------------------------------------------------------------
# Helpers: lightweight stubs (no real training components needed)
# ---------------------------------------------------------------------------


def _stub_metrics_manager():
    """Minimal MetricsManager-like object."""
    history = SimpleNamespace(
        policy_losses=[0.5],
        value_losses=[0.3],
        entropies=[1.0],
        kl_divergences=[0.01],
        clip_fractions=[0.1],
        learning_rates=[3e-4],
        episode_lengths=[100.0],
        episode_rewards=[0.5],
        win_rates_history=[],
    )
    return SimpleNamespace(
        global_timestep=10,
        total_episodes_completed=2,
        black_wins=1,
        white_wins=0,
        draws=1,
        processing=False,
        history=history,
        get_hot_squares=lambda n: [],
    )


def _stub_trainer(config=None, game=None, step_manager=None, buffer=None):
    """Minimal trainer stub with optional config."""
    return SimpleNamespace(
        config=config,
        game=game,
        metrics_manager=_stub_metrics_manager(),
        step_manager=step_manager,
        experience_buffer=buffer,
        last_gradient_norm=0.5,
    )


def _stub_eval_config(strategy="single_opponent", enabled=True):
    """Minimal evaluation config stub."""
    return SimpleNamespace(
        enable_periodic_evaluation=enabled,
        strategy=strategy,
    )


def _stub_app_config(eval_config=None):
    """Minimal AppConfig stub."""
    return SimpleNamespace(
        evaluation=eval_config or _stub_eval_config(),
    )


# ---------------------------------------------------------------------------
# Eva.2.1: Envelope structure
# ---------------------------------------------------------------------------


class TestEnvelopeStructure:
    """build_snapshot returns a valid BroadcastStateEnvelope."""

    def test_has_all_required_keys(self):
        trainer = _stub_trainer()
        snapshot = build_snapshot(trainer)

        required = {
            "schema_version", "timestamp", "speed", "mode",
            "active_views", "health", "training", "pending_updates",
        }
        assert required.issubset(snapshot.keys())

    def test_schema_version_is_v1(self):
        trainer = _stub_trainer()
        snapshot = build_snapshot(trainer)
        assert snapshot["schema_version"] == SCHEMA_VERSION

    def test_active_views_contains_training(self):
        trainer = _stub_trainer()
        snapshot = build_snapshot(trainer)
        assert snapshot["active_views"] == ["training"]

    def test_training_has_five_sub_keys(self):
        trainer = _stub_trainer()
        snapshot = build_snapshot(trainer)
        training = snapshot["training"]
        expected = {"board_state", "metrics", "step_info", "buffer_info", "model_info"}
        assert expected == set(training.keys())

    def test_validates_against_contract(self):
        trainer = _stub_trainer()
        snapshot = build_snapshot(trainer)
        errors = validate_envelope(snapshot)
        assert errors == [], f"Validation errors: {errors}"

    def test_speed_passthrough(self):
        trainer = _stub_trainer()
        snapshot = build_snapshot(trainer, speed=99.9)
        assert snapshot["speed"] == 99.9

    def test_timestamp_is_recent(self):
        before = time.time()
        trainer = _stub_trainer()
        snapshot = build_snapshot(trainer)
        after = time.time()
        assert before <= snapshot["timestamp"] <= after

    def test_nullable_training_sub_keys(self):
        """When game/step_manager/buffer are None, sub-keys are None."""
        trainer = _stub_trainer(game=None, step_manager=None, buffer=None)
        snapshot = build_snapshot(trainer)
        training = snapshot["training"]
        assert training["board_state"] is None
        assert training["step_info"] is None
        assert training["buffer_info"] is None
        # metrics and model_info are always present
        assert training["metrics"] is not None
        assert training["model_info"] is not None


# ---------------------------------------------------------------------------
# Eva.2.2: Mode resolution and health population
# ---------------------------------------------------------------------------


class TestModeResolution:
    """_resolve_mode derives mode from trainer config."""

    def test_no_config_returns_training_only(self):
        assert _resolve_mode(SimpleNamespace()) == "training_only"

    def test_no_evaluation_config_returns_training_only(self):
        config = SimpleNamespace()  # no evaluation attr
        assert _resolve_mode(SimpleNamespace(config=config)) == "training_only"

    def test_evaluation_disabled_returns_training_only(self):
        eval_cfg = _stub_eval_config(enabled=False, strategy="tournament")
        config = _stub_app_config(eval_cfg)
        assert _resolve_mode(_stub_trainer(config=config)) == "training_only"

    def test_evaluation_enabled_returns_strategy(self):
        eval_cfg = _stub_eval_config(enabled=True, strategy="tournament")
        config = _stub_app_config(eval_cfg)
        assert _resolve_mode(_stub_trainer(config=config)) == "tournament"

    def test_single_opponent_strategy(self):
        eval_cfg = _stub_eval_config(enabled=True, strategy="single_opponent")
        config = _stub_app_config(eval_cfg)
        assert _resolve_mode(_stub_trainer(config=config)) == "single_opponent"

    def test_mode_appears_in_envelope(self):
        eval_cfg = _stub_eval_config(enabled=True, strategy="ladder")
        config = _stub_app_config(eval_cfg)
        trainer = _stub_trainer(config=config)
        snapshot = build_snapshot(trainer)
        assert snapshot["mode"] == "ladder"


class TestHealthPopulation:
    """Health map is correctly populated in the envelope."""

    def test_training_health_is_ok(self):
        trainer = _stub_trainer()
        snapshot = build_snapshot(trainer)
        assert snapshot["health"]["training"] == "ok"

    def test_non_training_views_are_missing(self):
        trainer = _stub_trainer()
        snapshot = build_snapshot(trainer)
        for key in VIEW_KEYS:
            if key != "training":
                assert snapshot["health"][key] == "missing"

    def test_health_covers_all_view_keys(self):
        trainer = _stub_trainer()
        snapshot = build_snapshot(trainer)
        assert set(snapshot["health"].keys()) == set(VIEW_KEYS)


# ---------------------------------------------------------------------------
# Eva.2.3: pending_updates sanitization
# ---------------------------------------------------------------------------


class TestPendingUpdatesSanitization:
    """pending_updates are sanitized through the contract helper."""

    def test_none_pending_returns_empty_dict(self):
        trainer = _stub_trainer()
        snapshot = build_snapshot(trainer, pending_updates=None)
        assert snapshot["pending_updates"] == {}

    def test_scalars_preserved(self):
        trainer = _stub_trainer()
        updates = {"epoch": 5, "lr": 0.001, "phase": "warmup", "done": True}
        snapshot = build_snapshot(trainer, pending_updates=updates)
        assert snapshot["pending_updates"] == updates

    def test_non_scalars_stripped(self):
        trainer = _stub_trainer()
        updates = {"epoch": 5, "bad_list": [1, 2], "bad_dict": {"a": 1}}
        snapshot = build_snapshot(trainer, pending_updates=updates)
        assert snapshot["pending_updates"] == {"epoch": 5}

    def test_empty_pending_returns_empty_dict(self):
        trainer = _stub_trainer()
        snapshot = build_snapshot(trainer, pending_updates={})
        assert snapshot["pending_updates"] == {}

    def test_envelope_with_sanitized_pending_validates(self):
        trainer = _stub_trainer()
        updates = {"ok": 1, "also_ok": "yes", "bad": [1, 2, 3]}
        snapshot = build_snapshot(trainer, pending_updates=updates)
        errors = validate_envelope(snapshot)
        assert errors == [], f"Validation errors: {errors}"
