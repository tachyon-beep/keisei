"""Gap-analysis tests for db.py: empty snapshots, heartbeat on empty table,
timestamp-filtered snapshot reads."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from keisei.db import (
    init_db,
    read_game_snapshots,
    read_game_snapshots_since,
    read_training_state,
    update_heartbeat,
    write_game_snapshots,
)


# ===================================================================
# H2 — write_game_snapshots with empty list
# ===================================================================


def test_write_game_snapshots_empty_list(db: Path) -> None:
    """An empty snapshot list should not raise and should leave the table empty."""
    write_game_snapshots(str(db), [])
    result = read_game_snapshots(str(db))
    assert result == []


def test_write_game_snapshots_empty_then_real(db: Path) -> None:
    """Empty write followed by a real write should work normally."""
    write_game_snapshots(str(db), [])
    snap = {
        "game_id": 0, "board_json": "[]", "hands_json": "{}",
        "current_player": "black", "ply": 5, "is_over": 0,
        "result": "in_progress", "sfen": "startpos", "in_check": 0,
        "move_history_json": "[]",
        "value_estimate": 0.0,
    }
    write_game_snapshots(str(db), [snap])
    result = read_game_snapshots(str(db))
    assert len(result) == 1
    assert result[0]["ply"] == 5


# ===================================================================
# H3 — update_heartbeat on a DB with no training_state row
# ===================================================================


def test_update_heartbeat_no_training_state_row(db: Path) -> None:
    """Calling update_heartbeat when no training_state row exists should
    silently update 0 rows (no error) — documents the current contract."""
    # This should NOT raise — it just updates 0 rows
    update_heartbeat(str(db))
    # Verify no row was magically created
    state = read_training_state(str(db))
    assert state is None


# ===================================================================
# M2 — read_game_snapshots_since() timestamp filtering
# ===================================================================

def _make_snapshot(game_id: int) -> dict:
    """Helper: minimal valid game snapshot."""
    return {
        "game_id": game_id, "board_json": "[]", "hands_json": "{}",
        "current_player": "black", "ply": game_id * 10, "is_over": 0,
        "result": "in_progress", "sfen": "startpos", "in_check": 0,
        "move_history_json": "[]", "value_estimate": 0.0,
    }


def _set_updated_at(db_path: str, game_id: int, ts: str) -> None:
    """Directly set the updated_at timestamp for a game snapshot."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE game_snapshots SET updated_at = ? WHERE game_id = ?",
        (ts, game_id),
    )
    conn.commit()
    conn.close()


class TestReadGameSnapshotsSince:
    """Tests for read_game_snapshots_since() — timestamp filtering and max_ts."""

    def test_returns_only_newer_snapshots(self, db: Path) -> None:
        """Only snapshots with updated_at > since should be returned."""
        path = str(db)
        write_game_snapshots(path, [_make_snapshot(0), _make_snapshot(1), _make_snapshot(2)])
        _set_updated_at(path, 0, "2026-01-01T00:00:00Z")
        _set_updated_at(path, 1, "2026-01-02T00:00:00Z")
        _set_updated_at(path, 2, "2026-01-03T00:00:00Z")

        rows, max_ts = read_game_snapshots_since(path, "2026-01-01T00:00:00Z")

        game_ids = [r["game_id"] for r in rows]
        assert game_ids == [1, 2]

    def test_returns_correct_max_ts(self, db: Path) -> None:
        """max_ts should be the maximum updated_at among returned rows."""
        path = str(db)
        write_game_snapshots(path, [_make_snapshot(0), _make_snapshot(1)])
        _set_updated_at(path, 0, "2026-01-02T00:00:00Z")
        _set_updated_at(path, 1, "2026-01-03T00:00:00Z")

        _, max_ts = read_game_snapshots_since(path, "2026-01-01T00:00:00Z")

        assert max_ts == "2026-01-03T00:00:00Z"

    def test_empty_when_no_matches(self, db: Path) -> None:
        """When all snapshots are older than since, returns empty list and since as max_ts."""
        path = str(db)
        write_game_snapshots(path, [_make_snapshot(0)])
        _set_updated_at(path, 0, "2026-01-01T00:00:00Z")

        rows, max_ts = read_game_snapshots_since(path, "2026-12-31T00:00:00Z")

        assert rows == []
        assert max_ts == "2026-12-31T00:00:00Z"  # Falls back to the since value

    def test_exact_timestamp_excluded(self, db: Path) -> None:
        """Boundary: a snapshot with updated_at == since should NOT be included (strict >)."""
        path = str(db)
        write_game_snapshots(path, [_make_snapshot(0)])
        _set_updated_at(path, 0, "2026-01-01T12:00:00Z")

        rows, max_ts = read_game_snapshots_since(path, "2026-01-01T12:00:00Z")

        assert rows == []
        assert max_ts == "2026-01-01T12:00:00Z"

    def test_empty_table(self, db: Path) -> None:
        """No snapshots at all — returns empty list and the since value."""
        rows, max_ts = read_game_snapshots_since(str(db), "2026-01-01T00:00:00Z")
        assert rows == []
        assert max_ts == "2026-01-01T00:00:00Z"
