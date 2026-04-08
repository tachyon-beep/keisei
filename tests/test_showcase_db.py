"""Tests for showcase database tables and operations."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from keisei.db import init_db, _connect


@pytest.fixture
def db(tmp_path: Path) -> str:
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


class TestShowcaseSchema:
    def test_schema_version_is_3(self, db: str) -> None:
        conn = _connect(db)
        try:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            assert row["version"] == 3
        finally:
            conn.close()

    def test_showcase_queue_table_exists(self, db: str) -> None:
        conn = _connect(db)
        try:
            conn.execute("SELECT id, entry_id_1, entry_id_2, speed, status, requested_at, started_at, completed_at FROM showcase_queue LIMIT 0")
        finally:
            conn.close()

    def test_showcase_games_table_exists(self, db: str) -> None:
        conn = _connect(db)
        try:
            conn.execute("SELECT id, queue_id, entry_id_black, entry_id_white, elo_black, elo_white, name_black, name_white, status, abandon_reason, started_at, completed_at, total_ply FROM showcase_games LIMIT 0")
        finally:
            conn.close()

    def test_showcase_moves_table_exists(self, db: str) -> None:
        conn = _connect(db)
        try:
            conn.execute("SELECT id, game_id, ply, action_index, usi_notation, board_json, hands_json, current_player, in_check, value_estimate, top_candidates, move_time_ms, created_at FROM showcase_moves LIMIT 0")
        finally:
            conn.close()

    def test_showcase_heartbeat_table_exists(self, db: str) -> None:
        conn = _connect(db)
        try:
            conn.execute("SELECT id, last_heartbeat, runner_pid FROM showcase_heartbeat LIMIT 0")
        finally:
            conn.close()

    def test_showcase_queue_one_running_constraint(self, db: str) -> None:
        """Only one queue entry can have status='running' at a time."""
        conn = _connect(db)
        try:
            conn.execute(
                "INSERT INTO showcase_queue (entry_id_1, entry_id_2, speed, status, requested_at) VALUES ('a', 'b', 'normal', 'running', '2026-01-01T00:00:00Z')"
            )
            conn.commit()
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO showcase_queue (entry_id_1, entry_id_2, speed, status, requested_at) VALUES ('c', 'd', 'normal', 'running', '2026-01-01T00:00:00Z')"
                )
                conn.commit()
        finally:
            conn.close()
