"""Gap-analysis tests for db.py: empty snapshots, heartbeat on empty table."""

from __future__ import annotations

from pathlib import Path

import pytest

from keisei.db import (
    init_db,
    read_game_snapshots,
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
