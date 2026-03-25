"""
Tests for lineage event schema — constants, factory helpers, and validation.
"""

import re

import pytest

from keisei.lineage.event_schema import (
    EVENT_TYPES,
    LINEAGE_SCHEMA_VERSION,
    CheckpointCreatedPayload,
    LineageEvent,
    MatchCompletedPayload,
    ModelPromotedPayload,
    TrainingResumedPayload,
    TrainingStartedPayload,
    make_event,
    make_event_id,
    make_model_id,
    validate_event,
)


# ---------------------------------------------------------------------------
# Constants lock
# ---------------------------------------------------------------------------


class TestConstants:
    def test_schema_version_is_v1(self):
        """The schema version must be pinned to v1.0.0."""
        assert LINEAGE_SCHEMA_VERSION == "v1.0.0"

    def test_event_types_is_tuple(self):
        assert isinstance(EVENT_TYPES, tuple)

    def test_event_types_contains_expected_values(self):
        expected = {
            "checkpoint_created",
            "model_promoted",
            "match_completed",
            "training_started",
            "training_resumed",
        }
        assert set(EVENT_TYPES) == expected

    def test_event_types_count(self):
        assert len(EVENT_TYPES) == 5


# ---------------------------------------------------------------------------
# make_event_id
# ---------------------------------------------------------------------------


class TestMakeEventId:
    def test_format_matches_spec(self):
        """ID format: {seq:06d}_{iso_utc}_{uuid8}"""
        eid = make_event_id(42)
        # Pattern: 6-digit seq, underscore, ISO compact UTC, underscore, 8 hex chars
        pattern = r"^000042_\d{8}T\d{6}Z_[0-9a-f]{8}$"
        assert re.match(pattern, eid), f"Event ID {eid!r} does not match pattern"

    def test_zero_sequence(self):
        eid = make_event_id(0)
        assert eid.startswith("000000_")

    def test_large_sequence(self):
        eid = make_event_id(999999)
        assert eid.startswith("999999_")

    def test_uniqueness(self):
        """Two calls with the same seq should still produce different IDs (UUID suffix)."""
        id1 = make_event_id(1)
        id2 = make_event_id(1)
        assert id1 != id2

    def test_lexicographic_sort_matches_sequence_order(self):
        """IDs with increasing seq numbers must sort lexicographically in order."""
        ids = [make_event_id(i) for i in range(10)]
        assert ids == sorted(ids)

    def test_return_type_is_str(self):
        assert isinstance(make_event_id(0), str)


# ---------------------------------------------------------------------------
# make_model_id
# ---------------------------------------------------------------------------


class TestMakeModelId:
    def test_format(self):
        mid = make_model_id("run-abc", 10000)
        assert mid == "run-abc::checkpoint_ts10000"

    def test_zero_timestep(self):
        mid = make_model_id("my_run", 0)
        assert mid == "my_run::checkpoint_ts0"

    def test_contains_run_name(self):
        mid = make_model_id("fancy-run", 500)
        assert "fancy-run" in mid

    def test_contains_timestep(self):
        mid = make_model_id("r", 12345)
        assert "12345" in mid


# ---------------------------------------------------------------------------
# make_event
# ---------------------------------------------------------------------------


class TestMakeEvent:
    def test_produces_valid_event(self):
        """make_event output must pass validate_event with zero errors."""
        event = make_event(
            seq=0,
            event_type="checkpoint_created",
            run_name="test-run",
            model_id="test-run::checkpoint_ts0",
            payload={
                "checkpoint_path": "/tmp/ckpt.pt",
                "global_timestep": 0,
                "total_episodes": 0,
                "parent_model_id": None,
            },
        )
        errors = validate_event(event)
        assert errors == [], f"Unexpected validation errors: {errors}"

    def test_schema_version_is_current(self):
        event = make_event(
            seq=0,
            event_type="training_started",
            run_name="r",
            model_id="r::checkpoint_ts0",
            payload={"config_snapshot": {}, "parent_model_id": None},
        )
        assert event["schema_version"] == LINEAGE_SCHEMA_VERSION

    def test_event_type_preserved(self):
        event = make_event(
            seq=1,
            event_type="match_completed",
            run_name="r",
            model_id="m",
            payload={
                "opponent_model_id": "opp",
                "result": "win",
                "num_games": 10,
                "win_rate": 0.7,
                "agent_rating": 1500.0,
                "opponent_rating": 1400.0,
            },
        )
        assert event["event_type"] == "match_completed"

    def test_emitted_at_is_iso_format(self):
        event = make_event(
            seq=0,
            event_type="training_started",
            run_name="r",
            model_id="m",
            payload={"config_snapshot": {}, "parent_model_id": None},
        )
        # ISO-8601 should contain 'T' and '+' or 'Z'
        assert "T" in event["emitted_at"]

    def test_run_name_and_model_id_preserved(self):
        event = make_event(
            seq=0,
            event_type="training_started",
            run_name="my-run",
            model_id="my-run::checkpoint_ts0",
            payload={"config_snapshot": {}, "parent_model_id": None},
        )
        assert event["run_name"] == "my-run"
        assert event["model_id"] == "my-run::checkpoint_ts0"

    def test_payload_preserved(self):
        payload = {
            "checkpoint_path": "/tmp/x.pt",
            "global_timestep": 100,
            "total_episodes": 50,
            "parent_model_id": "parent::checkpoint_ts0",
        }
        event = make_event(
            seq=5,
            event_type="checkpoint_created",
            run_name="r",
            model_id="m",
            payload=payload,
        )
        assert event["payload"] == payload


# ---------------------------------------------------------------------------
# validate_event
# ---------------------------------------------------------------------------


class TestValidateEvent:
    def _make_valid_event(self) -> dict:
        """Return a minimal valid event dict."""
        return {
            "event_id": "000001_20260101T000000Z_abcdef01",
            "event_type": "checkpoint_created",
            "schema_version": "v1.0.0",
            "emitted_at": "2026-01-01T00:00:00+00:00",
            "run_name": "test-run",
            "model_id": "test-run::checkpoint_ts0",
            "payload": {
                "checkpoint_path": "/tmp/ckpt.pt",
                "global_timestep": 0,
                "total_episodes": 0,
                "parent_model_id": None,
            },
        }

    def test_valid_event_has_no_errors(self):
        assert validate_event(self._make_valid_event()) == []

    def test_missing_required_key(self):
        event = self._make_valid_event()
        del event["event_id"]
        errors = validate_event(event)
        assert any("event_id" in e for e in errors)

    def test_missing_multiple_keys(self):
        event = self._make_valid_event()
        del event["event_id"]
        del event["payload"]
        errors = validate_event(event)
        assert len(errors) >= 2

    def test_unknown_event_type(self):
        event = self._make_valid_event()
        event["event_type"] = "unknown_type"
        errors = validate_event(event)
        assert any("not a recognised type" in e for e in errors)

    def test_event_type_not_string(self):
        event = self._make_valid_event()
        event["event_type"] = 123
        errors = validate_event(event)
        assert any("event_type" in e for e in errors)

    def test_empty_event_id(self):
        event = self._make_valid_event()
        event["event_id"] = ""
        errors = validate_event(event)
        assert any("event_id" in e for e in errors)

    def test_empty_schema_version(self):
        event = self._make_valid_event()
        event["schema_version"] = ""
        errors = validate_event(event)
        assert any("schema_version" in e for e in errors)

    def test_payload_not_dict(self):
        event = self._make_valid_event()
        event["payload"] = "not a dict"
        errors = validate_event(event)
        assert any("payload" in e for e in errors)

    def test_run_name_not_string(self):
        event = self._make_valid_event()
        event["run_name"] = 42
        errors = validate_event(event)
        assert any("run_name" in e for e in errors)

    def test_model_id_not_string(self):
        event = self._make_valid_event()
        event["model_id"] = None
        errors = validate_event(event)
        assert any("model_id" in e for e in errors)

    def test_empty_dict_returns_errors(self):
        errors = validate_event({})
        assert len(errors) == len(
            {"event_id", "event_type", "schema_version", "emitted_at",
             "run_name", "model_id", "payload"}
        )


# ---------------------------------------------------------------------------
# Payload TypedDicts (smoke tests for type existence)
# ---------------------------------------------------------------------------


class TestPayloadTypes:
    def test_checkpoint_created_payload_keys(self):
        p = CheckpointCreatedPayload(
            checkpoint_path="/tmp/x.pt",
            global_timestep=100,
            total_episodes=50,
            parent_model_id=None,
        )
        assert "checkpoint_path" in p

    def test_model_promoted_payload_keys(self):
        p = ModelPromotedPayload(
            from_rating=1400.0, to_rating=1600.0, promotion_reason="new best"
        )
        assert "promotion_reason" in p

    def test_match_completed_payload_keys(self):
        p = MatchCompletedPayload(
            opponent_model_id="opp",
            result="win",
            num_games=10,
            win_rate=0.7,
            agent_rating=1500.0,
            opponent_rating=1400.0,
        )
        assert "win_rate" in p

    def test_training_started_payload_keys(self):
        p = TrainingStartedPayload(config_snapshot={}, parent_model_id=None)
        assert "config_snapshot" in p

    def test_training_resumed_payload_keys(self):
        p = TrainingResumedPayload(
            resumed_from_checkpoint="/tmp/ckpt.pt",
            global_timestep_at_resume=1000,
            parent_model_id=None,
        )
        assert "resumed_from_checkpoint" in p
