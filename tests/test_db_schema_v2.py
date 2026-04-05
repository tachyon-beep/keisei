"""Tests for DB schema: league tables, game_snapshots, and schema version."""

import sqlite3

from keisei.db import SCHEMA_VERSION, init_db


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

    def test_schema_version_is_current(self, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        assert _get_schema_version(db_path) == SCHEMA_VERSION

    def test_league_entries_columns(self, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        cols = _get_table_columns(db_path, "league_entries")
        assert "display_name" in cols
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

    def test_creates_elo_history_table(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "elo_history" in tables

    def test_elo_history_columns(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(elo_history)").fetchall()]
        conn.close()
        assert cols == ["id", "entry_id", "epoch", "elo_rating", "recorded_at"]

    def test_game_snapshots_has_opponent_id(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(game_snapshots)").fetchall()]
        conn.close()
        assert "opponent_id" in cols


class TestSchemaV5:
    """Phase 2: historical library, gauntlet results, role Elo columns."""

    def test_league_entries_has_role_elo_columns(self, tmp_path):
        db_path = str(tmp_path / "v5.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(league_entries)").fetchall()]
        conn.close()
        assert "elo_frontier" in cols
        assert "elo_dynamic" in cols
        assert "elo_recent" in cols
        assert "elo_historical" in cols

    def test_role_elo_defaults_to_1000(self, tmp_path):
        db_path = str(tmp_path / "v5.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO league_entries (architecture, model_params, checkpoint_path, created_epoch) "
            "VALUES ('resnet', '{}', '/tmp/x.pt', 1)"
        )
        conn.commit()
        row = conn.execute("SELECT elo_frontier, elo_dynamic, elo_recent, elo_historical FROM league_entries").fetchone()
        conn.close()
        assert row == (1000.0, 1000.0, 1000.0, 1000.0)

    def test_historical_library_table_exists(self, tmp_path):
        db_path = str(tmp_path / "v5.db")
        init_db(db_path)
        assert _table_exists(db_path, "historical_library")

    def test_historical_library_columns(self, tmp_path):
        db_path = str(tmp_path / "v5.db")
        init_db(db_path)
        cols = _get_table_columns(db_path, "historical_library")
        assert "slot_index" in cols
        assert "target_epoch" in cols
        assert "entry_id" in cols
        assert "actual_epoch" in cols
        assert "selected_at" in cols
        assert "selection_mode" in cols

    def test_gauntlet_results_table_exists(self, tmp_path):
        db_path = str(tmp_path / "v5.db")
        init_db(db_path)
        assert _table_exists(db_path, "gauntlet_results")

    def test_gauntlet_results_columns(self, tmp_path):
        db_path = str(tmp_path / "v5.db")
        init_db(db_path)
        cols = _get_table_columns(db_path, "gauntlet_results")
        assert "epoch" in cols
        assert "entry_id" in cols
        assert "historical_slot" in cols
        assert "historical_entry_id" in cols
        assert "wins" in cols
        assert "losses" in cols
        assert "draws" in cols
        assert "elo_before" in cols
        assert "elo_after" in cols

    def test_mismatched_version_raises(self, tmp_path):
        """A database with a different schema version should raise RuntimeError."""
        db_path = str(tmp_path / "old.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version VALUES (99)")
        conn.commit()
        conn.close()
        import pytest
        with pytest.raises(RuntimeError, match="schema version 99"):
            init_db(db_path)

    def test_idempotent_init(self, tmp_path):
        """Running init_db twice should be a no-op."""
        db_path = str(tmp_path / "idempotent.db")
        init_db(db_path)
        init_db(db_path)  # Should not raise
        assert _get_schema_version(db_path) == SCHEMA_VERSION
