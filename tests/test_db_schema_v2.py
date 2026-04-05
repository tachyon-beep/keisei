"""Tests for DB schema v2: league tables and game_snapshots extensions."""

import sqlite3

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

    def test_schema_version_is_current(self, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        assert _get_schema_version(db_path) == 6

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

    def test_v4_to_v5_migration(self, tmp_path):
        """Simulate a v4 DB and verify migration to v5."""
        db_path = str(tmp_path / "v4_to_v5.db")
        # Create a minimal v4 DB
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version VALUES (4)")
        conn.execute("""
            CREATE TABLE league_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                display_name TEXT NOT NULL DEFAULT '',
                flavour_facts TEXT NOT NULL DEFAULT '[]',
                architecture TEXT NOT NULL,
                model_params TEXT NOT NULL,
                checkpoint_path TEXT NOT NULL,
                elo_rating REAL NOT NULL DEFAULT 1000.0,
                created_epoch INTEGER NOT NULL,
                games_played INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT 'unassigned',
                status TEXT NOT NULL DEFAULT 'active',
                parent_entry_id INTEGER,
                lineage_group TEXT,
                protection_remaining INTEGER NOT NULL DEFAULT 0,
                last_match_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE league_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER NOT NULL,
                learner_id INTEGER NOT NULL,
                opponent_id INTEGER NOT NULL,
                wins INTEGER NOT NULL,
                losses INTEGER NOT NULL,
                draws INTEGER NOT NULL,
                elo_delta_a REAL NOT NULL DEFAULT 0.0,
                elo_delta_b REAL NOT NULL DEFAULT 0.0,
                recorded_at TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_league_results_epoch ON league_results(epoch)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_league_entries_elo ON league_entries(elo_rating)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_league_results_learner ON league_results(learner_id)")
        conn.execute("""
            CREATE TABLE elo_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                elo_rating REAL NOT NULL,
                recorded_at TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.execute("CREATE TABLE training_state (id INTEGER PRIMARY KEY CHECK (id = 1), config_json TEXT NOT NULL, display_name TEXT NOT NULL, model_arch TEXT NOT NULL, algorithm_name TEXT NOT NULL, started_at TEXT NOT NULL, current_epoch INTEGER NOT NULL DEFAULT 0, current_step INTEGER NOT NULL DEFAULT 0, checkpoint_path TEXT, total_epochs INTEGER, status TEXT NOT NULL DEFAULT 'running', phase TEXT NOT NULL DEFAULT 'init', heartbeat_at TEXT NOT NULL DEFAULT '')")
        conn.execute("CREATE TABLE metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, epoch INTEGER, step INTEGER, timestamp TEXT NOT NULL DEFAULT '')")
        conn.execute("CREATE TABLE game_snapshots (game_id INTEGER PRIMARY KEY, board_json TEXT NOT NULL, hands_json TEXT NOT NULL, current_player TEXT NOT NULL, ply INTEGER NOT NULL, is_over INTEGER NOT NULL, result TEXT NOT NULL, sfen TEXT NOT NULL, in_check INTEGER NOT NULL, move_history_json TEXT NOT NULL, value_estimate REAL NOT NULL DEFAULT 0.0, game_type TEXT NOT NULL DEFAULT 'live', demo_slot INTEGER, opponent_id INTEGER, updated_at TEXT NOT NULL DEFAULT '')")
        conn.execute("""
            CREATE TABLE league_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER NOT NULL,
                from_role TEXT, to_role TEXT, from_status TEXT, to_status TEXT, reason TEXT,
                created_at TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.execute("CREATE TABLE league_meta (id INTEGER PRIMARY KEY CHECK (id = 1), bootstrapped INTEGER NOT NULL DEFAULT 0)")
        conn.execute("INSERT INTO league_meta (id, bootstrapped) VALUES (1, 0)")
        # Insert a test entry pre-migration
        conn.execute(
            "INSERT INTO league_entries (architecture, model_params, checkpoint_path, created_epoch) "
            "VALUES ('resnet', '{}', '/tmp/x.pt', 42)"
        )
        conn.commit()
        conn.close()

        # Run migration
        init_db(db_path)

        # Verify
        assert _get_schema_version(db_path) == 6
        cols = _get_table_columns(db_path, "league_entries")
        assert "elo_frontier" in cols
        assert "elo_dynamic" in cols
        assert "elo_recent" in cols
        assert "elo_historical" in cols
        assert _table_exists(db_path, "historical_library")
        assert _table_exists(db_path, "gauntlet_results")

        # Pre-existing entry should have defaults
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT elo_frontier, elo_dynamic, elo_recent, elo_historical FROM league_entries").fetchone()
        conn.close()
        assert row == (1000.0, 1000.0, 1000.0, 1000.0)

    def test_idempotent_migration(self, tmp_path):
        """Running init_db twice on a v6 DB should be a no-op."""
        db_path = str(tmp_path / "idempotent.db")
        init_db(db_path)
        init_db(db_path)  # Should not raise
        assert _get_schema_version(db_path) == 6
