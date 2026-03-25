"""Tests for lineage view: extraction, envelope parsing, and panel rendering."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from keisei.webui.view_contracts import SCHEMA_VERSION, make_health_map, validate_envelope


def _make_lineage_view() -> Dict[str, Any]:
    """Minimal valid LineageViewState with expanded fields."""
    return {
        "event_count": 5,
        "latest_checkpoint_id": "run-1::checkpoint_ts5000",
        "parent_id": "run-1::checkpoint_ts2500",
        "model_id": "run-1::checkpoint_ts5000",
        "run_name": "run-1",
        "generation": 3,
        "latest_rating": 1050.0,
        "recent_events": [
            {
                "event_type": "checkpoint_created",
                "model_id": "run-1::checkpoint_ts5000",
                "emitted_at": "2026-03-26T10:00:00Z",
            },
        ],
        "ancestor_chain": ["run-1::checkpoint_ts2500", "run-1::init"],
    }


def _make_envelope_with_lineage() -> Dict[str, Any]:
    """Build a valid envelope with lineage view active."""
    from tests.webui.conftest import _make_valid_training_view

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": 1707400000.0,
        "speed": 42.5,
        "mode": "single_opponent",
        "active_views": ["training", "lineage"],
        "health": make_health_map(training="ok", lineage="ok"),
        "training": _make_valid_training_view(),
        "lineage": _make_lineage_view(),
        "pending_updates": {},
    }


@pytest.mark.unit
class TestLineageViewState:
    def test_envelope_with_lineage_validates(self):
        env = _make_envelope_with_lineage()
        errors = validate_envelope(env)
        assert errors == [], f"Validation errors: {errors}"

    def test_envelope_without_lineage_validates(self):
        env = _make_envelope_with_lineage()
        del env["lineage"]
        env["active_views"] = ["training"]
        env["health"]["lineage"] = "missing"
        errors = validate_envelope(env)
        assert errors == [], f"Validation errors: {errors}"


@pytest.mark.unit
class TestLineageExtraction:
    def test_extract_lineage_summary_with_events(self):
        from keisei.lineage.graph import LineageGraph
        from keisei.lineage.registry import LineageRegistry
        from keisei.webui.state_snapshot import extract_lineage_summary

        events = [
            {
                "event_id": "evt-001",
                "event_type": "training_started",
                "model_id": "run::init",
                "run_name": "run",
                "emitted_at": "2026-01-01T00:00:00Z",
                "sequence_number": 0,
                "payload": {},
            },
            {
                "event_id": "evt-002",
                "event_type": "checkpoint_created",
                "model_id": "run::cp1",
                "run_name": "run",
                "emitted_at": "2026-01-01T01:00:00Z",
                "sequence_number": 1,
                "payload": {
                    "parent_model_id": "run::init",
                    "checkpoint_path": "/tmp/cp1.pt",
                    "global_timestep": 1000,
                },
            },
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
            tmp_path = Path(f.name)

        try:
            registry = LineageRegistry(tmp_path)
            graph = LineageGraph.from_events(registry.load_all())
            result = extract_lineage_summary(
                registry, graph, current_model_id="run::cp1"
            )

            assert result["event_count"] == 2
            assert result["model_id"] == "run::cp1"
            assert result["parent_id"] == "run::init"
            assert result["generation"] == 2  # init -> cp1
            assert "run::init" in result["ancestor_chain"]
        finally:
            tmp_path.unlink()

    def test_extract_lineage_summary_empty_registry(self):
        from keisei.lineage.graph import LineageGraph
        from keisei.lineage.registry import LineageRegistry
        from keisei.webui.state_snapshot import extract_lineage_summary

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            tmp_path = Path(f.name)

        try:
            registry = LineageRegistry(tmp_path)
            graph = LineageGraph.from_events(registry.load_all())
            result = extract_lineage_summary(
                registry, graph, current_model_id=None
            )

            assert result["event_count"] == 0
            assert result["model_id"] is None
            assert result["generation"] == 0
            assert result["ancestor_chain"] == []
        finally:
            tmp_path.unlink()


@pytest.mark.unit
class TestEnvelopeParserLineage:
    def test_lineage_present(self):
        from keisei.webui.envelope_parser import EnvelopeParser

        env = _make_envelope_with_lineage()
        parser = EnvelopeParser(env)
        assert parser.lineage is not None
        assert parser.lineage["model_id"] == "run-1::checkpoint_ts5000"
        assert parser.lineage["generation"] == 3

    def test_lineage_absent(self):
        from keisei.webui.envelope_parser import EnvelopeParser

        env = _make_envelope_with_lineage()
        del env["lineage"]
        parser = EnvelopeParser(env)
        assert parser.lineage is None

    def test_lineage_in_available_optional_views(self):
        from keisei.webui.envelope_parser import EnvelopeParser

        env = _make_envelope_with_lineage()
        parser = EnvelopeParser(env)
        assert "lineage" in parser.available_optional_views()

    def test_lineage_not_in_missing_views_when_active(self):
        from keisei.webui.envelope_parser import EnvelopeParser

        env = _make_envelope_with_lineage()
        parser = EnvelopeParser(env)
        assert "lineage" not in parser.missing_optional_views()
