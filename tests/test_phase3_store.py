"""Tests for Phase 3 store features: schema v6, new OpponentEntry fields, new methods."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from keisei.db import SCHEMA_VERSION, init_db, read_league_data
from keisei.training.opponent_store import (
    EntryStatus,
    OpponentEntry,
    OpponentStore,
    Role,
)


@pytest.fixture
def store_db(tmp_path):
    db_path = str(tmp_path / "phase3.db")
    init_db(db_path)
    return db_path


@pytest.fixture
def league_dir(tmp_path):
    d = tmp_path / "checkpoints" / "league"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def store(store_db, league_dir):
    s = OpponentStore(db_path=store_db, league_dir=str(league_dir))
    yield s
    s.close()


# ---------- Task 1: Schema v6 migration ----------


class TestSchemaV6Migration:
    def test_schema_v6_migration_idempotent(self, tmp_path):
        """Re-init on an already-v6 DB is a no-op (idempotency check)."""
        db_path = str(tmp_path / "migrate.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        # Downgrade version to 5 but leave v6 columns in place
        conn.execute("UPDATE schema_version SET version = 5")
        conn.commit()
        conn.close()

        # Re-init should trigger the v5->v6 migration path but columns already exist
        init_db(db_path)

        conn = sqlite3.connect(db_path)
        cols = [c[1] for c in conn.execute("PRAGMA table_info(league_entries)").fetchall()]
        assert "optimizer_path" in cols
        assert "update_count" in cols
        assert "last_train_at" in cols
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row[0] == SCHEMA_VERSION
        conn.close()

    def test_schema_v6_migration_real(self, tmp_path):
        """Test actual v5->v6 migration by starting from a genuine v5 schema."""
        db_path = str(tmp_path / "v5_test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode = WAL")
        # Create schema_version with version 5
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version VALUES (5)")
        # Create league_entries WITHOUT v6 columns
        conn.execute("""CREATE TABLE league_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            display_name TEXT NOT NULL DEFAULT '',
            flavour_facts TEXT NOT NULL DEFAULT '[]',
            architecture TEXT NOT NULL,
            model_params TEXT NOT NULL,
            checkpoint_path TEXT NOT NULL,
            elo_rating REAL NOT NULL DEFAULT 1000.0,
            created_epoch INTEGER NOT NULL,
            games_played INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            role TEXT NOT NULL DEFAULT 'unassigned',
            status TEXT NOT NULL DEFAULT 'active',
            parent_entry_id INTEGER,
            lineage_group TEXT,
            protection_remaining INTEGER NOT NULL DEFAULT 0,
            last_match_at TEXT,
            elo_frontier REAL NOT NULL DEFAULT 1000.0,
            elo_dynamic REAL NOT NULL DEFAULT 1000.0,
            elo_recent REAL NOT NULL DEFAULT 1000.0,
            elo_historical REAL NOT NULL DEFAULT 1000.0
        )""")
        # Create other required tables with enough columns for indexes
        conn.execute("""CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT, epoch INTEGER NOT NULL,
            step INTEGER NOT NULL DEFAULT 0)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_epoch ON metrics(epoch)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_id ON metrics(id)")
        conn.execute("CREATE TABLE IF NOT EXISTS game_snapshots (game_id INTEGER PRIMARY KEY)")
        conn.execute("""CREATE TABLE IF NOT EXISTS training_state (
            id INTEGER PRIMARY KEY CHECK (id = 1))""")
        conn.execute("""CREATE TABLE IF NOT EXISTS league_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch INTEGER NOT NULL, learner_id INTEGER NOT NULL,
            opponent_id INTEGER NOT NULL,
            wins INTEGER NOT NULL, losses INTEGER NOT NULL,
            draws INTEGER NOT NULL,
            elo_delta_a REAL NOT NULL DEFAULT 0.0,
            elo_delta_b REAL NOT NULL DEFAULT 0.0,
            recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')))""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_league_results_epoch ON league_results(epoch)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_league_entries_elo ON league_entries(elo_rating)")
        conn.execute("""CREATE TABLE IF NOT EXISTS elo_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id INTEGER NOT NULL, epoch INTEGER NOT NULL,
            elo_rating REAL NOT NULL,
            recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')))""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_elo_history_entry ON elo_history(entry_id)")
        conn.execute("""CREATE TABLE IF NOT EXISTS league_transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id INTEGER NOT NULL)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transitions_entry ON league_transitions(entry_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_league_results_learner ON league_results(learner_id)")
        conn.execute("""CREATE TABLE IF NOT EXISTS league_meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            bootstrapped INTEGER NOT NULL DEFAULT 0)""")
        conn.execute("INSERT OR IGNORE INTO league_meta (id, bootstrapped) VALUES (1, 0)")
        conn.execute("""CREATE TABLE IF NOT EXISTS historical_library (
            slot_index INTEGER NOT NULL PRIMARY KEY,
            target_epoch INTEGER NOT NULL,
            entry_id INTEGER,
            actual_epoch INTEGER,
            selected_at TEXT NOT NULL DEFAULT '',
            selection_mode TEXT NOT NULL DEFAULT '')""")
        conn.execute("""CREATE TABLE IF NOT EXISTS gauntlet_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch INTEGER NOT NULL,
            entry_id INTEGER NOT NULL,
            historical_slot INTEGER NOT NULL,
            historical_entry_id INTEGER NOT NULL,
            wins INTEGER NOT NULL, losses INTEGER NOT NULL,
            draws INTEGER NOT NULL,
            elo_before REAL, elo_after REAL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')))""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gauntlet_epoch ON gauntlet_results(epoch)")
        conn.commit()
        conn.close()

        # Verify columns don't exist yet
        conn = sqlite3.connect(db_path)
        cols = [c[1] for c in conn.execute("PRAGMA table_info(league_entries)").fetchall()]
        assert "optimizer_path" not in cols
        assert "update_count" not in cols
        assert "last_train_at" not in cols
        conn.close()

        # Run migration
        init_db(db_path)

        # Verify columns now exist
        conn = sqlite3.connect(db_path)
        cols = [c[1] for c in conn.execute("PRAGMA table_info(league_entries)").fetchall()]
        assert "optimizer_path" in cols
        assert "update_count" in cols
        assert "last_train_at" in cols
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == SCHEMA_VERSION
        conn.close()

    def test_fresh_db_has_v6_columns(self, tmp_path):
        """A brand new DB should have the v6 columns in the CREATE TABLE."""
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cols = [c[1] for c in conn.execute("PRAGMA table_info(league_entries)").fetchall()]
        assert "optimizer_path" in cols
        assert "update_count" in cols
        assert "last_train_at" in cols
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row[0] == SCHEMA_VERSION
        conn.close()

    def test_schema_version_constant_is_7(self):
        assert SCHEMA_VERSION == 7


# ---------- Task 2: OpponentEntry new fields ----------


class TestOpponentEntryNewFields:
    def test_defaults(self):
        """New fields should have proper defaults."""
        entry = OpponentEntry(
            id=1,
            display_name="Test",
            architecture="resnet",
            model_params={},
            checkpoint_path="/path/to/model.pt",
            elo_rating=1000.0,
            created_epoch=0,
            games_played=0,
            created_at="2026-01-01T00:00:00Z",
            flavour_facts=[],
        )
        assert entry.optimizer_path is None
        assert entry.update_count == 0
        assert entry.last_train_at is None

    def test_explicit_values(self):
        """New fields should accept explicit values."""
        entry = OpponentEntry(
            id=1,
            display_name="Test",
            architecture="resnet",
            model_params={},
            checkpoint_path="/path/to/model.pt",
            elo_rating=1000.0,
            created_epoch=0,
            games_played=0,
            created_at="2026-01-01T00:00:00Z",
            flavour_facts=[],
            optimizer_path="/opt.pt",
            update_count=5,
            last_train_at="2026-04-01T00:00:00Z",
        )
        assert entry.optimizer_path == "/opt.pt"
        assert entry.update_count == 5
        assert entry.last_train_at == "2026-04-01T00:00:00Z"

    def test_from_db_row_with_new_columns(self, tmp_path):
        """from_db_row should read new columns when present."""
        db_path = str(tmp_path / "entry.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """INSERT INTO league_entries
               (display_name, flavour_facts, architecture, model_params,
                checkpoint_path, created_epoch, role, status,
                optimizer_path, update_count, last_train_at)
               VALUES ('Takeshi', '[]', 'resnet', '{}', '/p', 10,
                       'frontier_static', 'active',
                       '/opt.pt', 3, '2026-04-01T00:00:00Z')"""
        )
        conn.commit()
        row = conn.execute("SELECT * FROM league_entries WHERE id = 1").fetchone()
        entry = OpponentEntry.from_db_row(row)
        assert entry.optimizer_path == "/opt.pt"
        assert entry.update_count == 3
        assert entry.last_train_at == "2026-04-01T00:00:00Z"
        conn.close()

    def test_from_db_row_without_new_columns(self):
        """from_db_row should use fallback defaults when columns are missing."""
        # Simulate a pre-v6 row by using an in-memory DB without the new columns
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE league_entries (
                id INTEGER PRIMARY KEY,
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
                last_match_at TEXT,
                elo_frontier REAL NOT NULL DEFAULT 1000.0,
                elo_dynamic REAL NOT NULL DEFAULT 1000.0,
                elo_recent REAL NOT NULL DEFAULT 1000.0,
                elo_historical REAL NOT NULL DEFAULT 1000.0
            )
        """)
        conn.execute(
            """INSERT INTO league_entries
               (display_name, architecture, model_params, checkpoint_path, created_epoch)
               VALUES ('Test', 'resnet', '{}', '/p', 5)"""
        )
        conn.commit()
        row = conn.execute("SELECT * FROM league_entries WHERE id = 1").fetchone()
        entry = OpponentEntry.from_db_row(row)
        assert entry.optimizer_path is None
        assert entry.update_count == 0
        assert entry.last_train_at is None
        conn.close()


# ---------- Task 3: OpponentStore new methods ----------


def _make_dummy_model():
    """Create a tiny model for testing."""
    return torch.nn.Linear(4, 2)


def _add_test_entry(store, league_dir):
    """Add a test entry via the store and return it."""
    model = _make_dummy_model()
    with patch("keisei.training.opponent_store.build_model", return_value=_make_dummy_model()):
        entry = store.add_entry(
            model=model,
            architecture="linear",
            model_params={"in_features": 4, "out_features": 2},
            epoch=1,
        )
    return entry


class TestSaveLoadOptimizer:
    def test_round_trip(self, store, league_dir):
        """Save optimizer state, load it back, verify content matches."""
        entry = _add_test_entry(store, league_dir)
        optimizer = torch.optim.SGD(_make_dummy_model().parameters(), lr=0.01)
        # Do a dummy step so state is non-trivial
        optimizer.zero_grad()
        dummy_loss = _make_dummy_model()(torch.randn(1, 4)).sum()
        dummy_loss.backward()
        optimizer.step()

        state_dict = optimizer.state_dict()
        store.save_optimizer(entry.id, state_dict)

        loaded = store.load_optimizer(entry.id)
        assert loaded is not None
        # Check param_groups match
        assert loaded["param_groups"] == state_dict["param_groups"]

    def test_load_returns_none_when_no_optimizer(self, store, league_dir):
        """load_optimizer returns None when no optimizer has been saved."""
        entry = _add_test_entry(store, league_dir)
        result = store.load_optimizer(entry.id)
        assert result is None

    def test_load_returns_none_for_nonexistent_entry(self, store, league_dir):
        """load_optimizer returns None for non-existent entry."""
        result = store.load_optimizer(99999)
        assert result is None

    def test_load_warns_on_missing_file(self, store, league_dir, caplog):
        """If optimizer file was deleted, log warning and return None."""
        entry = _add_test_entry(store, league_dir)
        state_dict = torch.optim.SGD(_make_dummy_model().parameters(), lr=0.01).state_dict()
        store.save_optimizer(entry.id, state_dict)

        # Verify saved entry has optimizer_path
        updated = store.get_entry(entry.id)
        assert updated is not None
        assert updated.optimizer_path is not None

        # Delete the file
        Path(updated.optimizer_path).unlink()

        import logging
        with caplog.at_level(logging.WARNING):
            result = store.load_optimizer(entry.id)
        assert result is None
        assert "does not exist" in caplog.text.lower()

    def test_atomic_write_no_tmp_left(self, store, league_dir):
        """After save_optimizer, no .tmp files should remain."""
        entry = _add_test_entry(store, league_dir)
        state_dict = torch.optim.SGD(_make_dummy_model().parameters(), lr=0.01).state_dict()
        store.save_optimizer(entry.id, state_dict)

        tmp_files = list(league_dir.glob("*.tmp"))
        assert tmp_files == []

    def test_save_optimizer_raises_for_missing_entry(self, store, league_dir):
        """save_optimizer raises ValueError for non-existent entry."""
        state_dict = torch.optim.SGD(_make_dummy_model().parameters(), lr=0.01).state_dict()
        with pytest.raises(ValueError):
            store.save_optimizer(99999, state_dict)


class TestIncrementUpdateCount:
    def test_increment(self, store, league_dir):
        """update_count increments and last_train_at is set."""
        entry = _add_test_entry(store, league_dir)
        assert entry.update_count == 0

        store.increment_update_count(entry.id)
        updated = store.get_entry(entry.id)
        assert updated is not None
        assert updated.update_count == 1
        assert updated.last_train_at is not None

        store.increment_update_count(entry.id)
        updated2 = store.get_entry(entry.id)
        assert updated2 is not None
        assert updated2.update_count == 2


class TestReadLeagueDataPhase3:
    def test_includes_phase3_fields(self, store_db, league_dir):
        """read_league_data should include the new phase3 columns."""
        conn = sqlite3.connect(store_db)
        conn.execute(
            """INSERT INTO league_entries
               (display_name, flavour_facts, architecture, model_params,
                checkpoint_path, created_epoch, role, status,
                optimizer_path, update_count, last_train_at)
               VALUES ('Takeshi', '[]', 'resnet', '{}', '/p', 10,
                       'frontier_static', 'active',
                       '/opt.pt', 3, '2026-04-01T00:00:00Z')"""
        )
        conn.commit()
        conn.close()

        data = read_league_data(store_db)
        assert len(data["entries"]) == 1
        entry = data["entries"][0]
        assert entry["optimizer_path"] == "/opt.pt"
        assert entry["update_count"] == 3
        assert entry["last_train_at"] == "2026-04-01T00:00:00Z"
