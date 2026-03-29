"""Tests for Elo rating timeline builder."""

import json
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _write_log(entries, path):
    """Write match result entries to a JSONL file."""
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _make_result(model_a, model_b, delta_a, delta_b, ts=1.0):
    return {
        "model_a": model_a,
        "model_b": model_b,
        "winner": model_a if delta_a > 0 else model_b,
        "elo_delta_a": delta_a,
        "elo_delta_b": delta_b,
        "move_count": 50,
        "reason": "checkmate",
        "timestamp": ts,
    }


class TestBuildEloTimelines:
    """Tests for build_elo_timelines()."""

    def test_missing_file_returns_empty(self, tmp_path):
        from keisei.webui.elo_chart import build_elo_timelines

        result = build_elo_timelines(tmp_path / "nonexistent.jsonl")
        assert result == {}

    def test_empty_file_returns_empty(self, tmp_path):
        from keisei.webui.elo_chart import build_elo_timelines

        log = tmp_path / "empty.jsonl"
        log.write_text("")
        result = build_elo_timelines(log)
        assert result == {}

    def test_single_match(self, tmp_path):
        from keisei.webui.elo_chart import build_elo_timelines

        log = tmp_path / "log.jsonl"
        _write_log([_make_result("A", "B", 12.0, -12.0)], log)
        result = build_elo_timelines(log)
        assert "A" in result
        assert "B" in result
        assert result["A"] == [1500.0, 1512.0]
        assert result["B"] == [1500.0, 1488.0]

    def test_multiple_matches_cumulative(self, tmp_path):
        from keisei.webui.elo_chart import build_elo_timelines

        log = tmp_path / "log.jsonl"
        _write_log(
            [
                _make_result("A", "B", 12.0, -12.0, ts=1.0),
                _make_result("A", "B", 10.0, -10.0, ts=2.0),
            ],
            log,
        )
        result = build_elo_timelines(log)
        assert result["A"] == [1500.0, 1512.0, 1522.0]
        assert result["B"] == [1500.0, 1488.0, 1478.0]

    def test_forward_fill(self, tmp_path):
        from keisei.webui.elo_chart import build_elo_timelines

        log = tmp_path / "log.jsonl"
        _write_log(
            [
                _make_result("A", "B", 12.0, -12.0, ts=1.0),
                _make_result("A", "C", 8.0, -8.0, ts=2.0),
            ],
            log,
        )
        result = build_elo_timelines(log)
        assert result["B"] == [1500.0, 1488.0, 1488.0]
        assert result["C"] == [1500.0, 1500.0, 1492.0]

    def test_top_n_filtering(self, tmp_path):
        from keisei.webui.elo_chart import build_elo_timelines

        log = tmp_path / "log.jsonl"
        entries = []
        for i in range(12):
            name = f"model_{i:02d}"
            entries.append(_make_result(name, "baseline", float(i + 1), float(-(i + 1)), ts=float(i)))
        _write_log(entries, log)
        result = build_elo_timelines(log, top_n=3)
        assert len(result) == 3
        assert "model_11" in result
        assert "model_10" in result
        assert "model_09" in result

    def test_top_n_with_leaderboard(self, tmp_path):
        from keisei.webui.elo_chart import build_elo_timelines

        log = tmp_path / "log.jsonl"
        _write_log(
            [
                _make_result("A", "B", 12.0, -12.0),
                _make_result("C", "B", 5.0, -5.0),
            ],
            log,
        )
        leaderboard = [
            {"name": "B", "elo": 1483.0},
            {"name": "C", "elo": 1505.0},
        ]
        result = build_elo_timelines(log, top_n=2, leaderboard=leaderboard)
        assert "B" in result
        assert "C" in result
        assert "A" not in result

    def test_malformed_line_skipped(self, tmp_path):
        from keisei.webui.elo_chart import build_elo_timelines

        log = tmp_path / "log.jsonl"
        with open(log, "w") as f:
            f.write(json.dumps(_make_result("A", "B", 12.0, -12.0)) + "\n")
            f.write("this is not valid json\n")
            f.write(json.dumps(_make_result("A", "B", 10.0, -10.0)) + "\n")
        result = build_elo_timelines(log)
        assert result["A"] == [1500.0, 1512.0, 1522.0]
