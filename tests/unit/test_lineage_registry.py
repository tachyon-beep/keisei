"""
Tests for LineageRegistry — append-only JSONL persistence.
"""

import json

import pytest

from keisei.lineage.event_schema import (
    LINEAGE_SCHEMA_VERSION,
    make_event,
    make_event_id,
    validate_event,
)
from keisei.lineage.registry import LineageRegistry


def _make_event(seq: int = 0, **overrides) -> dict:
    """Build a valid event dict, with optional field overrides."""
    base = make_event(
        seq=seq,
        event_type=overrides.pop("event_type", "checkpoint_created"),
        run_name=overrides.pop("run_name", "test-run"),
        model_id=overrides.pop("model_id", f"test-run::checkpoint_ts{seq * 1000}"),
        payload=overrides.pop(
            "payload",
            {
                "checkpoint_path": f"/tmp/ckpt_{seq}.pt",
                "global_timestep": seq * 1000,
                "total_episodes": seq * 100,
                "parent_model_id": None,
            },
        ),
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Init / load
# ---------------------------------------------------------------------------


class TestRegistryInit:
    def test_init_with_no_file(self, tmp_path):
        """Registry should start empty if file does not exist."""
        reg = LineageRegistry(tmp_path / "events.jsonl")
        assert reg.event_count == 0
        assert reg.next_sequence_number == 0

    def test_init_from_existing_jsonl(self, tmp_path):
        """Registry should load events from an existing file."""
        path = tmp_path / "events.jsonl"
        event = _make_event(0)
        path.write_text(json.dumps(event, separators=(",", ":")) + "\n")

        reg = LineageRegistry(path)
        assert reg.event_count == 1

    def test_parent_dir_created_on_append(self, tmp_path):
        """Appending to a file in a non-existent directory should create it."""
        path = tmp_path / "sub" / "dir" / "events.jsonl"
        reg = LineageRegistry(path)
        reg.append(_make_event(0))
        assert path.exists()


# ---------------------------------------------------------------------------
# Append
# ---------------------------------------------------------------------------


class TestRegistryAppend:
    def test_append_writes_to_file(self, tmp_path):
        path = tmp_path / "events.jsonl"
        reg = LineageRegistry(path)
        event = _make_event(0)
        result = reg.append(event)

        assert result is True
        assert reg.event_count == 1
        assert path.read_text().strip() != ""

    def test_append_preserves_order(self, tmp_path):
        path = tmp_path / "events.jsonl"
        reg = LineageRegistry(path)

        events = [_make_event(i) for i in range(5)]
        for e in events:
            reg.append(e)

        loaded = reg.load_all()
        assert [e["event_id"] for e in loaded] == [e["event_id"] for e in events]

    def test_idempotent_duplicate_skip(self, tmp_path):
        """Appending the same event_id twice should skip the second."""
        path = tmp_path / "events.jsonl"
        reg = LineageRegistry(path)
        event = _make_event(0)

        assert reg.append(event) is True
        assert reg.append(event) is False
        assert reg.event_count == 1

    def test_invalid_event_raises_value_error(self, tmp_path):
        path = tmp_path / "events.jsonl"
        reg = LineageRegistry(path)

        with pytest.raises(ValueError, match="Invalid lineage event"):
            reg.append({"bad": "data"})

    def test_sequence_number_increments(self, tmp_path):
        path = tmp_path / "events.jsonl"
        reg = LineageRegistry(path)

        assert reg.next_sequence_number == 0
        reg.append(_make_event(0))
        assert reg.next_sequence_number == 1
        reg.append(_make_event(1))
        assert reg.next_sequence_number == 2


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------


class TestRegistryReadAPI:
    @pytest.fixture
    def populated_registry(self, tmp_path):
        """Registry with 3 events of different types."""
        path = tmp_path / "events.jsonl"
        reg = LineageRegistry(path)

        reg.append(
            _make_event(0, event_type="checkpoint_created", run_name="run-a")
        )
        reg.append(
            _make_event(
                1,
                event_type="training_started",
                run_name="run-b",
                payload={"config_snapshot": {}, "parent_model_id": None},
            )
        )
        reg.append(
            _make_event(2, event_type="checkpoint_created", run_name="run-a")
        )
        return reg

    def test_load_all_returns_defensive_copy(self, populated_registry):
        """Mutations to the returned list must not affect the registry."""
        all1 = populated_registry.load_all()
        all1.clear()
        assert populated_registry.event_count == 3

    def test_filter_by_type(self, populated_registry):
        result = populated_registry.filter_by_type("checkpoint_created")
        assert len(result) == 2
        assert all(e["event_type"] == "checkpoint_created" for e in result)

    def test_filter_by_type_no_match(self, populated_registry):
        result = populated_registry.filter_by_type("match_completed")
        assert result == []

    def test_filter_by_run(self, populated_registry):
        result = populated_registry.filter_by_run("run-a")
        assert len(result) == 2

    def test_filter_by_model(self, tmp_path):
        path = tmp_path / "events.jsonl"
        reg = LineageRegistry(path)
        reg.append(_make_event(0, model_id="model-A"))
        reg.append(_make_event(1, model_id="model-B"))

        result = reg.filter_by_model("model-A")
        assert len(result) == 1
        assert result[0]["model_id"] == "model-A"

    def test_get_latest_event(self, populated_registry):
        latest = populated_registry.get_latest_event()
        assert latest is not None
        assert latest["event_type"] == "checkpoint_created"
        # It's the third event (seq=2)

    def test_get_latest_event_empty_registry(self, tmp_path):
        reg = LineageRegistry(tmp_path / "events.jsonl")
        assert reg.get_latest_event() is None


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


class TestRegistryResilience:
    def test_corrupt_line_is_skipped(self, tmp_path):
        """A non-JSON line should be logged and skipped."""
        path = tmp_path / "events.jsonl"
        event = _make_event(0)
        lines = [
            json.dumps(event, separators=(",", ":")),
            "THIS IS NOT JSON",
            json.dumps(_make_event(1), separators=(",", ":")),
        ]
        path.write_text("\n".join(lines) + "\n")

        reg = LineageRegistry(path)
        assert reg.event_count == 2

    def test_blank_lines_ignored(self, tmp_path):
        path = tmp_path / "events.jsonl"
        event = _make_event(0)
        content = (
            "\n"
            + json.dumps(event, separators=(",", ":"))
            + "\n\n\n"
            + json.dumps(_make_event(1), separators=(",", ":"))
            + "\n\n"
        )
        path.write_text(content)

        reg = LineageRegistry(path)
        assert reg.event_count == 2

    def test_dedup_on_load(self, tmp_path):
        """Duplicate event_ids in the file should be deduplicated."""
        path = tmp_path / "events.jsonl"
        event = _make_event(0)
        # Write the same event twice
        content = (
            json.dumps(event, separators=(",", ":"))
            + "\n"
            + json.dumps(event, separators=(",", ":"))
            + "\n"
        )
        path.write_text(content)

        reg = LineageRegistry(path)
        assert reg.event_count == 1

    def test_write_close_reopen_roundtrip(self, tmp_path):
        """Events survive a write-close-reopen cycle."""
        path = tmp_path / "events.jsonl"

        # Write phase
        reg1 = LineageRegistry(path)
        reg1.append(_make_event(0))
        reg1.append(_make_event(1))
        assert reg1.event_count == 2

        # Reopen phase
        reg2 = LineageRegistry(path)
        assert reg2.event_count == 2
        assert reg2.load_all()[0]["event_id"] == reg1.load_all()[0]["event_id"]

    def test_compact_json_separators_in_file(self, tmp_path):
        """Lines should use compact separators (no extra whitespace)."""
        path = tmp_path / "events.jsonl"
        reg = LineageRegistry(path)
        reg.append(_make_event(0))

        raw = path.read_text().strip()
        # Compact JSON has no spaces after : or ,
        assert ": " not in raw
        assert ", " not in raw

    def test_invalid_event_on_load_is_skipped(self, tmp_path):
        """An event missing required fields in the file should be skipped."""
        path = tmp_path / "events.jsonl"
        valid = _make_event(0)
        invalid = {"event_id": "bad", "partial": True}
        content = (
            json.dumps(valid, separators=(",", ":"))
            + "\n"
            + json.dumps(invalid, separators=(",", ":"))
            + "\n"
        )
        path.write_text(content)

        reg = LineageRegistry(path)
        assert reg.event_count == 1
