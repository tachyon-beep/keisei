"""Tests for v1 broadcast view contracts.

These tests lock the canonical fixture payloads against the contract
validators, ensuring that schema changes are detected immediately.
"""

from keisei.webui.view_contracts import (
    SCHEMA_VERSION,
    STALE_THRESHOLD_SECONDS,
    VIEW_KEYS,
    make_health_map,
    sanitize_pending_updates,
    validate_envelope,
)


# ---------------------------------------------------------------------------
# Valid envelope tests
# ---------------------------------------------------------------------------


class TestValidEnvelope:
    """Canonical valid envelopes pass validation with zero errors."""

    def test_standard_envelope_is_valid(self, valid_envelope):
        errors = validate_envelope(valid_envelope)
        assert errors == [], f"Expected no errors: {errors}"

    def test_nullable_training_is_valid(self, valid_envelope_nullable):
        errors = validate_envelope(valid_envelope_nullable)
        assert errors == [], f"Expected no errors: {errors}"

    def test_training_only_mode_is_valid(self, valid_envelope_training_only):
        errors = validate_envelope(valid_envelope_training_only)
        assert errors == [], f"Expected no errors: {errors}"

    def test_valid_envelope_has_all_required_keys(self, valid_envelope):
        required = {
            "schema_version", "timestamp", "speed", "mode",
            "active_views", "health", "training", "pending_updates",
        }
        assert required.issubset(valid_envelope.keys())

    def test_valid_envelope_schema_version(self, valid_envelope):
        assert valid_envelope["schema_version"] == SCHEMA_VERSION

    def test_valid_envelope_health_covers_all_views(self, valid_envelope):
        for vk in VIEW_KEYS:
            assert vk in valid_envelope["health"]

    def test_valid_envelope_training_has_sub_keys(self, valid_envelope):
        training = valid_envelope["training"]
        assert "board_state" in training
        assert "metrics" in training
        assert "step_info" in training
        assert "buffer_info" in training
        assert "model_info" in training


# ---------------------------------------------------------------------------
# Invalid envelope tests
# ---------------------------------------------------------------------------


class TestInvalidEnvelope:
    """Every canonical invalid envelope produces at least one error."""

    def test_invalid_envelope_fails_validation(self, invalid_envelope):
        errors = validate_envelope(invalid_envelope)
        assert len(errors) > 0, "Expected validation errors for invalid envelope"


class TestSpecificInvalidCases:
    """Targeted tests for specific contract violations."""

    def test_missing_speed_detected(self, invalid_missing_speed):
        errors = validate_envelope(invalid_missing_speed)
        assert any("speed" in e for e in errors)

    def test_missing_schema_version_detected(self, invalid_missing_schema_version):
        errors = validate_envelope(invalid_missing_schema_version)
        assert any("schema_version" in e for e in errors)

    def test_bad_health_status_detected(self, invalid_bad_health_status):
        errors = validate_envelope(invalid_bad_health_status)
        assert any("invalid status" in e for e in errors)

    def test_incomplete_health_detected(self, invalid_incomplete_health):
        errors = validate_envelope(invalid_incomplete_health)
        assert any("missing view key" in e for e in errors)

    def test_non_scalar_pending_detected(self, invalid_non_scalar_pending):
        errors = validate_envelope(invalid_non_scalar_pending)
        assert any("non-scalar" in e for e in errors)

    def test_legacy_top_level_layout_detected(self, invalid_legacy_top_level_keys):
        errors = validate_envelope(invalid_legacy_top_level_keys)
        assert any("training" in e for e in errors)

    def test_unknown_mode_passes_validation(self, envelope_unknown_mode):
        """Unknown mode strings are allowed per spike Decision Freeze #6."""
        errors = validate_envelope(envelope_unknown_mode)
        assert errors == [], (
            "Unknown mode values must NOT cause validation failure — "
            "renderers must treat them as opaque strings"
        )


# ---------------------------------------------------------------------------
# Validation helper tests
# ---------------------------------------------------------------------------


class TestMakeHealthMap:

    def test_default_all_missing(self):
        hm = make_health_map()
        for vk in VIEW_KEYS:
            assert hm[vk] == "missing"

    def test_override_single(self):
        hm = make_health_map(training="ok")
        assert hm["training"] == "ok"
        assert hm["league"] == "missing"

    def test_override_multiple(self):
        hm = make_health_map(training="ok", league="stale", lineage="error")
        assert hm["training"] == "ok"
        assert hm["league"] == "stale"
        assert hm["lineage"] == "error"

    def test_unknown_key_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown view key"):
            make_health_map(nonexistent="ok")


class TestSanitizePendingUpdates:

    def test_none_returns_empty(self):
        assert sanitize_pending_updates(None) == {}

    def test_empty_returns_empty(self):
        assert sanitize_pending_updates({}) == {}

    def test_scalars_preserved(self):
        raw = {"a": 1, "b": "x", "c": True, "d": 3.14, "e": None}
        assert sanitize_pending_updates(raw) == raw

    def test_non_scalars_dropped(self):
        raw = {"ok": 1, "list": [1, 2], "dict": {"a": 1}, "set": {1, 2}}
        result = sanitize_pending_updates(raw)
        assert result == {"ok": 1}

    def test_mixed(self):
        raw = {"epoch": 3, "bad": [1], "lr": 0.001, "nested": {"x": 1}}
        result = sanitize_pending_updates(raw)
        assert result == {"epoch": 3, "lr": 0.001}


class TestValidateEnvelopeEdgeCases:

    def test_empty_dict_reports_all_missing(self):
        errors = validate_envelope({})
        assert len(errors) == 8  # one per required key

    def test_wrong_type_timestamp(self, valid_envelope):
        valid_envelope["timestamp"] = "not_a_number"
        errors = validate_envelope(valid_envelope)
        assert any("timestamp" in e and "numeric" in e for e in errors)

    def test_active_views_wrong_type(self, valid_envelope):
        valid_envelope["active_views"] = "training"  # should be list
        errors = validate_envelope(valid_envelope)
        assert any("active_views" in e for e in errors)

    def test_health_wrong_type(self, valid_envelope):
        valid_envelope["health"] = "ok"
        errors = validate_envelope(valid_envelope)
        assert any("health" in e and "dict" in e for e in errors)

    def test_training_wrong_type(self, valid_envelope):
        valid_envelope["training"] = 42
        errors = validate_envelope(valid_envelope)
        assert any("training" in e and "dict" in e for e in errors)

    def test_pending_updates_wrong_type(self, valid_envelope):
        valid_envelope["pending_updates"] = "nope"
        errors = validate_envelope(valid_envelope)
        assert any("pending_updates" in e and "dict" in e for e in errors)
