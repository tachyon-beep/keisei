"""Tests for DB schema v2: league tables and game_snapshots extensions."""

import sqlite3

import pytest

from keisei.db import init_db


def _get_schema_version(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def _get_table_columns(db_path: str, table: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cursor.fetchall()]
    finally:
        conn.close()


def _table_exists(db_path: str, table: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


class TestSchemaV2:
    def test_creates_league_tables(self, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        assert _table_exists(db_path, "league_entries")
        assert _table_exists(db_path, "league_results")

    def test_game_snapshots_has_new_columns(self, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        cols = _get_table_columns(db_path, "game_snapshots")
        assert "game_type" in cols
        assert "demo_slot" in cols

    def test_schema_version_is_2(self, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        assert _get_schema_version(db_path) == 2

    def test_league_entries_columns(self, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        cols = _get_table_columns(db_path, "league_entries")
        assert "architecture" in cols
        assert "elo_rating" in cols
        assert "checkpoint_path" in cols
        assert "created_epoch" in cols

    def test_league_results_columns(self, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        cols = _get_table_columns(db_path, "league_results")
        assert "learner_id" in cols
        assert "opponent_id" in cols
        assert "wins" in cols
        assert "draws" in cols
