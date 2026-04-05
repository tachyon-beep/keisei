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

pytestmark = pytest.mark.integration


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
        # Phase 2: role Elo columns
        assert entry.elo_frontier == 1000.0
        assert entry.elo_dynamic == 1000.0
        assert entry.elo_recent == 1000.0
        assert entry.elo_historical == 1000.0

    def test_from_db_row_missing_elo_columns(self, tmp_path):
        """Entries from a pre-v5 DB without Elo columns should default to 1000."""
        db_path = str(tmp_path / "old.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # Minimal table WITHOUT the Phase 2 columns
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
        conn.execute(
            "INSERT INTO league_entries (architecture, model_params, checkpoint_path, created_epoch) "
            "VALUES ('resnet', '{}', '/p', 10)"
        )
        conn.commit()
        row = conn.execute("SELECT * FROM league_entries WHERE id = 1").fetchone()
        conn.close()
        entry = OpponentEntry.from_db_row(row)
        assert entry.elo_frontier == 1000.0
        assert entry.elo_dynamic == 1000.0
        assert entry.elo_recent == 1000.0
        assert entry.elo_historical == 1000.0


class TestOpponentStoreBasics:
    def test_add_entry_with_role(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        entry = store.add_entry(model, "resnet", {}, epoch=1, role=Role.RECENT_FIXED)
        assert entry.role is Role.RECENT_FIXED
        assert entry.status is EntryStatus.ACTIVE
        store.close()

    def test_list_by_role(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        store.add_entry(model, "resnet", {}, epoch=1, role=Role.FRONTIER_STATIC)
        store.add_entry(model, "resnet", {}, epoch=2, role=Role.RECENT_FIXED)
        store.add_entry(model, "resnet", {}, epoch=3, role=Role.DYNAMIC)
        assert len(store.list_by_role(Role.FRONTIER_STATIC)) == 1
        assert len(store.list_by_role(Role.RECENT_FIXED)) == 1
        assert len(store.list_by_role(Role.DYNAMIC)) == 1
        store.close()

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
        store.close()

    def test_retire_entry_does_not_delete_checkpoint(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        entry = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        ckpt = Path(entry.checkpoint_path)
        assert ckpt.exists()
        store.retire_entry(entry.id, "evicted")
        assert ckpt.exists()
        store.close()

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
        store.close()

    def test_update_role(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        entry = store.add_entry(model, "resnet", {}, epoch=1, role=Role.UNASSIGNED)
        store.update_role(entry.id, Role.FRONTIER_STATIC, "bootstrap")
        updated = store.list_by_role(Role.FRONTIER_STATIC)
        assert len(updated) == 1
        assert updated[0].id == entry.id
        store.close()


class TestOpponentStoreTransaction:
    def test_transaction_commits_on_exit(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        with store.transaction():
            store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        assert len(store.list_entries()) == 1
        store.close()

    def test_transaction_rolls_back_on_exception(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        with pytest.raises(RuntimeError):
            with store.transaction():
                store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
                raise RuntimeError("abort")
        assert len(store.list_entries()) == 0
        store.close()

    def test_rollback_cleans_up_checkpoint_files(self, store_db, league_dir):
        """Outer transaction rollback must delete checkpoint files written by add_entry."""
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        with pytest.raises(RuntimeError):
            with store.transaction():
                store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
                raise RuntimeError("abort")
        # DB should be clean
        assert len(store.list_entries()) == 0
        # Filesystem should also be clean — no orphaned weights.pt
        weight_files = list(Path(league_dir).rglob("weights.pt"))
        assert weight_files == [], f"Orphaned files after rollback: {weight_files}"
        store.close()

    def test_rollback_cleans_up_cloned_files(self, store_db, league_dir):
        """Outer transaction rollback must delete files written by clone_entry."""
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        source = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        with pytest.raises(RuntimeError):
            with store.transaction():
                store.clone_entry(source.id, Role.FRONTIER_STATIC, "test")
                raise RuntimeError("abort")
        # Only the original entry should exist
        assert len(store.list_entries()) == 1
        # Only the original entry's directory should have weight files
        weight_files = list(Path(league_dir).rglob("weights.pt"))
        assert len(weight_files) == 1, f"Expected 1 weights.pt, got {weight_files}"
        store.close()

    def test_rollback_preserves_prior_optimizer(self, store_db, league_dir):
        """save_optimizer rollback must restore the prior optimizer.pt, not delete it."""
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        entry = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)

        # Save an initial optimizer state
        original_state = {"step": 1, "param_groups": []}
        store.save_optimizer(entry.id, original_state)
        opt_path = Path(league_dir) / "entries" / f"{entry.id:06d}" / "optimizer.pt"
        assert opt_path.exists()

        # Now attempt a save_optimizer inside an outer transaction that rolls back
        new_state = {"step": 99, "param_groups": []}
        with pytest.raises(RuntimeError):
            with store.transaction():
                store.save_optimizer(entry.id, new_state)
                raise RuntimeError("abort")

        # The prior optimizer.pt must be RESTORED, not deleted
        assert opt_path.exists(), "Rollback deleted the prior optimizer.pt"
        restored = torch.load(opt_path, weights_only=False)
        assert restored["step"] == 1, (
            f"Rollback did not restore prior state: got step={restored['step']}"
        )
        store.close()


class TestRecordResultUpdates:
    def test_record_result_updates_last_match_at(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        a = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        b = store.add_entry(model, "resnet", {}, epoch=2, role=Role.FRONTIER_STATIC)
        store.record_result(epoch=1, entry_a_id=a.id, entry_b_id=b.id,
                            wins_a=3, wins_b=1, draws=0, match_type="calibration")
        updated_a = store.get_entry(a.id)
        assert updated_a is not None
        assert updated_a.last_match_at is not None
        store.close()

    def test_record_result_decrements_protection(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        b = store.add_entry(model, "resnet", {}, epoch=2, role=Role.FRONTIER_STATIC)
        with store.transaction():
            a = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
            store._conn.execute(
                "UPDATE league_entries SET protection_remaining = 5 WHERE id = ?",
                (a.id,),
            )
        store.record_result(epoch=1, entry_a_id=a.id, entry_b_id=b.id,
                            wins_a=1, wins_b=0, draws=0, match_type="calibration")
        conn = sqlite3.connect(store_db)
        row = conn.execute(
            "SELECT protection_remaining FROM league_entries WHERE id = ?", (a.id,)
        ).fetchone()
        conn.close()
        assert row[0] == 4
        store.close()


class TestEloCalculation:
    def test_equal_elo_win(self):
        new_a, new_b = compute_elo_update(1000.0, 1000.0, result=1.0, k=32)
        assert abs(new_a - 1016.0) < 0.1
        assert abs(new_b - 984.0) < 0.1

    def test_draw_against_equal(self):
        new_a, new_b = compute_elo_update(1000.0, 1000.0, result=0.5, k=32)
        assert abs(new_a - 1000.0) < 0.1


class TestOptimizerSaveLoadRoundTrip:
    """T5: Optimizer save/load round-trip with state verification."""

    def test_save_load_round_trip_state_matches(self, store_db, league_dir):
        """Create Dynamic entry, save optimizer, load it back, verify state matches."""
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        entry = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)

        # Build a non-trivial optimizer state
        opt_model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(opt_model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss = opt_model(torch.randn(2, 10)).sum()
        loss.backward()
        optimizer.step()

        state_dict = optimizer.state_dict()
        store.save_optimizer(entry.id, state_dict)
        loaded = store.load_optimizer(entry.id)

        assert loaded is not None
        # param_groups should match exactly
        assert loaded["param_groups"] == state_dict["param_groups"]
        # state keys should match
        assert set(loaded["state"].keys()) == set(state_dict["state"].keys())
        # Verify tensor values in state match
        for k in state_dict["state"]:
            for field in state_dict["state"][k]:
                orig = state_dict["state"][k][field]
                rest = loaded["state"][k][field]
                if isinstance(orig, torch.Tensor):
                    assert torch.equal(orig, rest), f"Mismatch in state[{k}][{field}]"
                else:
                    assert orig == rest, f"Mismatch in state[{k}][{field}]"
        store.close()


class TestDynamicToFrontierStaticClone:
    """T5: Clone Dynamic -> Frontier Static with role/field verification."""

    def test_clone_dynamic_to_frontier_static(self, store_db, league_dir):
        """Clone a Dynamic entry to Frontier Static, verify properties."""
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        source = store.add_entry(model, "resnet", {"h": 16}, epoch=5, role=Role.DYNAMIC)

        # Save optimizer on source so we can verify clone does NOT have one
        opt_state = {"step": 1, "param_groups": []}
        store.save_optimizer(source.id, opt_state)
        source_updated = store.get_entry(source.id)
        assert source_updated is not None
        assert source_updated.optimizer_path is not None

        clone = store.clone_entry(source.id, Role.FRONTIER_STATIC, "promotion")

        assert clone.role is Role.FRONTIER_STATIC
        assert clone.checkpoint_path != source.checkpoint_path
        assert clone.parent_entry_id == source.id
        assert clone.training_enabled is False
        assert clone.optimizer_path is None
        # Weights file should exist at the clone's path
        assert Path(clone.checkpoint_path).exists()
        store.close()


class TestFilesystemLayout:
    """T5: Verify directory structure for entries of each role."""

    def test_directory_structure_per_role(self, store_db, league_dir):
        """Create entries of each role, verify weights.pt/metadata.json layout."""
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)

        fs_entry = store.add_entry(model, "resnet", {}, epoch=1, role=Role.FRONTIER_STATIC)
        rf_entry = store.add_entry(model, "resnet", {}, epoch=2, role=Role.RECENT_FIXED)
        dyn_entry = store.add_entry(model, "resnet", {}, epoch=3, role=Role.DYNAMIC)

        for entry in [fs_entry, rf_entry, dyn_entry]:
            entry_dir = Path(league_dir) / "entries" / f"{entry.id:06d}"
            assert entry_dir.exists(), f"Missing entry dir for {entry.role}"
            assert (entry_dir / "weights.pt").exists(), f"Missing weights.pt for {entry.role}"
            assert (entry_dir / "metadata.json").exists(), f"Missing metadata.json for {entry.role}"

        # Save optimizer only on Dynamic
        opt_state = {"step": 1, "param_groups": []}
        store.save_optimizer(dyn_entry.id, opt_state)

        # optimizer.pt should exist only for Dynamic
        dyn_dir = Path(league_dir) / "entries" / f"{dyn_entry.id:06d}"
        assert (dyn_dir / "optimizer.pt").exists(), "Dynamic entry should have optimizer.pt"

        fs_dir = Path(league_dir) / "entries" / f"{fs_entry.id:06d}"
        assert not (fs_dir / "optimizer.pt").exists(), "Frontier Static should NOT have optimizer.pt"

        rf_dir = Path(league_dir) / "entries" / f"{rf_entry.id:06d}"
        assert not (rf_dir / "optimizer.pt").exists(), "Recent Fixed should NOT have optimizer.pt"

        store.close()
