"""Tests for OpponentStore -- the tiered pool storage layer."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from keisei.db import init_db
from keisei.training.opponent_store import (
    OpponentEntry,
    OpponentStore,
    Role,
    EntryStatus,
    compute_elo_update,
)


class TestEnums:
    def test_role_values(self):
        assert Role.FRONTIER_STATIC == "frontier_static"
        assert Role.RECENT_FIXED == "recent_fixed"
        assert Role.DYNAMIC == "dynamic"
        assert Role.UNASSIGNED == "unassigned"

    def test_entry_status_values(self):
        assert EntryStatus.ACTIVE == "active"
        assert EntryStatus.RETIRED == "retired"
        assert EntryStatus.ARCHIVED == "archived"

    def test_role_from_string(self):
        assert Role("frontier_static") is Role.FRONTIER_STATIC

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError):
            Role("invalid")


@pytest.fixture
def store_db(tmp_path):
    db_path = str(tmp_path / "store.db")
    init_db(db_path)
    return db_path


@pytest.fixture
def league_dir(tmp_path):
    d = tmp_path / "checkpoints" / "league"
    d.mkdir(parents=True)
    return d


class TestOpponentEntry:
    def test_from_db_row_with_new_fields(self, tmp_path):
        db_path = str(tmp_path / "entry.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """INSERT INTO league_entries
               (display_name, flavour_facts, architecture, model_params,
                checkpoint_path, created_epoch, role, status)
               VALUES ('Takeshi', '[]', 'resnet', '{}', '/p', 10,
                       'frontier_static', 'active')"""
        )
        conn.commit()
        row = conn.execute("SELECT * FROM league_entries WHERE id = 1").fetchone()
        conn.close()
        entry = OpponentEntry.from_db_row(row)
        assert entry.role is Role.FRONTIER_STATIC
        assert entry.status is EntryStatus.ACTIVE
        assert entry.parent_entry_id is None
        assert entry.protection_remaining == 0


class TestOpponentStoreBasics:
    def test_add_entry_with_role(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        entry = store.add_entry(model, "resnet", {}, epoch=1, role=Role.RECENT_FIXED)
        assert entry.role is Role.RECENT_FIXED
        assert entry.status is EntryStatus.ACTIVE

    def test_list_by_role(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        store.add_entry(model, "resnet", {}, epoch=1, role=Role.FRONTIER_STATIC)
        store.add_entry(model, "resnet", {}, epoch=2, role=Role.RECENT_FIXED)
        store.add_entry(model, "resnet", {}, epoch=3, role=Role.DYNAMIC)
        assert len(store.list_by_role(Role.FRONTIER_STATIC)) == 1
        assert len(store.list_by_role(Role.RECENT_FIXED)) == 1
        assert len(store.list_by_role(Role.DYNAMIC)) == 1

    def test_retire_entry_logs_transition(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        entry = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        store.retire_entry(entry.id, "evicted")
        active = store.list_entries()
        assert len(active) == 0
        conn = sqlite3.connect(store_db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM league_transitions WHERE entry_id = ?", (entry.id,)
        ).fetchall()
        conn.close()
        assert len(rows) >= 1
        assert rows[-1]["to_status"] == "retired"

    def test_retire_entry_does_not_delete_checkpoint(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        entry = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        ckpt = Path(entry.checkpoint_path)
        assert ckpt.exists()
        store.retire_entry(entry.id, "evicted")
        assert ckpt.exists()

    def test_clone_entry(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        source = store.add_entry(model, "resnet", {"h": 16}, epoch=5, role=Role.RECENT_FIXED)
        clone = store.clone_entry(source.id, Role.DYNAMIC, "promoted")
        assert clone.role is Role.DYNAMIC
        assert clone.parent_entry_id == source.id
        assert clone.architecture == source.architecture
        assert clone.model_params == source.model_params
        assert clone.checkpoint_path != source.checkpoint_path
        assert Path(clone.checkpoint_path).exists()
        assert Path(source.checkpoint_path).exists()

    def test_update_role(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        entry = store.add_entry(model, "resnet", {}, epoch=1, role=Role.UNASSIGNED)
        store.update_role(entry.id, Role.FRONTIER_STATIC, "bootstrap")
        updated = store.list_by_role(Role.FRONTIER_STATIC)
        assert len(updated) == 1
        assert updated[0].id == entry.id


class TestOpponentStoreTransaction:
    def test_transaction_commits_on_exit(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        with store.transaction():
            store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        assert len(store.list_entries()) == 1

    def test_transaction_rolls_back_on_exception(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        with pytest.raises(RuntimeError):
            with store.transaction():
                store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
                raise RuntimeError("abort")
        assert len(store.list_entries()) == 0


class TestRecordResultUpdates:
    def test_record_result_updates_last_match_at(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        a = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        b = store.add_entry(model, "resnet", {}, epoch=2, role=Role.FRONTIER_STATIC)
        store.record_result(epoch=1, learner_id=a.id, opponent_id=b.id,
                            wins=3, losses=1, draws=0)
        updated_a = store.list_entries()[0] if store.list_entries()[0].id == a.id else store.list_entries()[1]
        assert updated_a.last_match_at is not None

    def test_record_result_decrements_protection(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        with store.transaction():
            a = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
            store._conn.execute(
                "UPDATE league_entries SET protection_remaining = 5 WHERE id = ?",
                (a.id,),
            )
        store.record_result(epoch=1, learner_id=a.id, opponent_id=a.id,
                            wins=1, losses=0, draws=0)
        conn = sqlite3.connect(store_db)
        row = conn.execute(
            "SELECT protection_remaining FROM league_entries WHERE id = ?", (a.id,)
        ).fetchone()
        conn.close()
        assert row[0] == 4


class TestEloCalculation:
    def test_equal_elo_win(self):
        new_a, new_b = compute_elo_update(1000.0, 1000.0, result=1.0, k=32)
        assert abs(new_a - 1016.0) < 0.1
        assert abs(new_b - 984.0) < 0.1

    def test_draw_against_equal(self):
        new_a, new_b = compute_elo_update(1000.0, 1000.0, result=0.5, k=32)
        assert abs(new_a - 1000.0) < 0.1
