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


class TestSchemaVersion:
    def test_fresh_db_has_all_columns(self, tmp_path):
        """A brand new DB should have all columns from the current schema."""
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

    def test_schema_version_constant_is_1(self):
        assert SCHEMA_VERSION == 1

    def test_mismatched_version_raises(self, tmp_path):
        """A database with a different schema version should raise RuntimeError."""
        db_path = str(tmp_path / "old.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version VALUES (5)")
        conn.commit()
        conn.close()
        with pytest.raises(RuntimeError, match="schema version 5"):
            init_db(db_path)


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


# ---------- record_result self-play ----------


class TestRecordResultSelfPlay:
    def test_record_result_self_play(self, store, league_dir):
        """When learner_id == opponent_id, protection_remaining decrements only once."""
        entry = _add_test_entry(store, league_dir)
        store.set_protection(entry.id, 5)

        # Verify starting protection
        before = store.get_entry(entry.id)
        assert before is not None
        assert before.protection_remaining == 5

        store.record_result(
            epoch=1, learner_id=entry.id, opponent_id=entry.id,
            wins=1, losses=0, draws=0,
        )

        after = store.get_entry(entry.id)
        assert after is not None
        # Should be 4 (decremented once), NOT 3 (double-decremented)
        assert after.protection_remaining == 4


# ---------- Pin / Unpin ----------


class TestPinUnpin:
    def test_pin_adds_to_pinned_set(self, store, league_dir):
        """Pin an entry and verify it's in the internal pinned set."""
        entry = _add_test_entry(store, league_dir)
        store.pin(entry.id)
        assert entry.id in store._pinned

    def test_unpin_removes_from_pinned_set(self, store, league_dir):
        """Pin then unpin an entry, verify it's no longer pinned."""
        entry = _add_test_entry(store, league_dir)
        store.pin(entry.id)
        assert entry.id in store._pinned
        store.unpin(entry.id)
        assert entry.id not in store._pinned

    def test_pin_idempotent(self, store, league_dir):
        """Pinning the same entry twice should not raise an error."""
        entry = _add_test_entry(store, league_dir)
        store.pin(entry.id)
        store.pin(entry.id)  # no error
        assert entry.id in store._pinned

    def test_unpin_nonexistent_no_error(self, store, league_dir):
        """Unpinning an entry that was never pinned should not raise."""
        entry = _add_test_entry(store, league_dir)
        store.unpin(entry.id)  # no error
        assert entry.id not in store._pinned


# ---------- log_transition ----------


class TestLogTransition:
    def test_log_transition_writes_to_db(self, store, league_dir):
        """log_transition should insert a row in league_transitions."""
        entry = _add_test_entry(store, league_dir)

        store.log_transition(
            entry_id=entry.id,
            from_role="dynamic",
            to_role="frontier_static",
            from_status="active",
            to_status="active",
            reason="promoted",
        )

        with store._lock:
            rows = store._conn.execute(
                "SELECT * FROM league_transitions WHERE entry_id = ? AND reason = ?",
                (entry.id, "promoted"),
            ).fetchall()

        assert len(rows) >= 1
        row = dict(rows[-1])  # last matching row
        assert row["from_role"] == "dynamic"
        assert row["to_role"] == "frontier_static"
        assert row["from_status"] == "active"
        assert row["to_status"] == "active"
        assert row["reason"] == "promoted"


# ---------- list_all_entries vs list_entries ----------


class TestListAllVsListEntries:
    def test_list_all_includes_retired(self, store, league_dir):
        """list_all_entries returns all statuses; list_entries returns only active."""
        e1 = _add_test_entry(store, league_dir)
        e2 = _add_test_entry(store, league_dir)
        store.retire_entry(e2.id, "obsolete")

        all_entries = store.list_all_entries()
        active_entries = store.list_entries()

        assert len(all_entries) == 2
        assert len(active_entries) == 1
        assert active_entries[0].id == e1.id


# ---------- clone_entry error path ----------


class TestCloneEntryErrorPath:
    def test_clone_entry_source_not_found(self, store, league_dir):
        """Cloning a non-existent entry raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            store.clone_entry(source_entry_id=99999, new_role=Role.DYNAMIC, reason="test")


# ---------- load_opponent missing checkpoint ----------


class TestLoadOpponentMissingCheckpoint:
    def test_load_opponent_missing_checkpoint(self, store, league_dir):
        """load_opponent raises FileNotFoundError when checkpoint file is missing."""
        entry = _add_test_entry(store, league_dir)

        # Delete the checkpoint file
        Path(entry.checkpoint_path).unlink()

        with pytest.raises(FileNotFoundError, match="Checkpoint missing"):
            store.load_opponent(entry)


# ---------- Transaction nesting ----------


class TestTransactionNesting:
    def test_inner_exception_rolls_back_outer(self, store, league_dir):
        """An exception in a nested transaction should rollback the outer transaction."""
        entry = _add_test_entry(store, league_dir)
        store.set_protection(entry.id, 10)

        with pytest.raises(RuntimeError):
            with store.transaction():
                # Outer write
                store._conn.execute(
                    "UPDATE league_entries SET protection_remaining = 5 WHERE id = ?",
                    (entry.id,),
                )
                with store.transaction():
                    # Inner write
                    store._conn.execute(
                        "UPDATE league_entries SET protection_remaining = 1 WHERE id = ?",
                        (entry.id,),
                    )
                    raise RuntimeError("inner failure")

        # Outer transaction should have rolled back, restoring to 10
        after = store.get_entry(entry.id)
        assert after is not None
        assert after.protection_remaining == 10

    def test_nested_success_commits(self, store, league_dir):
        """Successful nested transactions commit only when outermost completes."""
        entry = _add_test_entry(store, league_dir)
        store.set_protection(entry.id, 10)

        with store.transaction():
            store._conn.execute(
                "UPDATE league_entries SET protection_remaining = 7 WHERE id = ?",
                (entry.id,),
            )
            with store.transaction():
                store._conn.execute(
                    "UPDATE league_entries SET protection_remaining = 3 WHERE id = ?",
                    (entry.id,),
                )

        after = store.get_entry(entry.id)
        assert after is not None
        assert after.protection_remaining == 3
