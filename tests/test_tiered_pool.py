"""Tests for TieredPool -- the orchestrator."""

import sqlite3

import pytest
import torch

from keisei.config import LeagueConfig
from keisei.db import init_db
from keisei.training.opponent_store import OpponentStore, Role, EntryStatus
from keisei.training.tiered_pool import TieredPool


@pytest.fixture
def pool_setup(tmp_path):
    db_path = str(tmp_path / "pool.db")
    init_db(db_path)
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    store = OpponentStore(db_path, str(league_dir))
    config = LeagueConfig()
    pool = TieredPool(store, config)
    return pool, store, db_path


class TestSnapshotLearner:
    def test_creates_recent_fixed_entry(self, pool_setup):
        pool, store, _ = pool_setup
        model = torch.nn.Linear(10, 10)
        entry = pool.snapshot_learner(model, "resnet", {}, epoch=1)
        assert entry.role is Role.RECENT_FIXED
        assert len(store.list_by_role(Role.RECENT_FIXED)) == 1


class TestEntriesByRole:
    def test_groups_correctly(self, pool_setup):
        pool, store, _ = pool_setup
        model = torch.nn.Linear(10, 10)
        store.add_entry(model, "resnet", {}, epoch=1, role=Role.FRONTIER_STATIC)
        store.add_entry(model, "resnet", {}, epoch=2, role=Role.RECENT_FIXED)
        store.add_entry(model, "resnet", {}, epoch=3, role=Role.DYNAMIC)
        by_role = pool.entries_by_role()
        assert len(by_role[Role.FRONTIER_STATIC]) == 1
        assert len(by_role[Role.RECENT_FIXED]) == 1
        assert len(by_role[Role.DYNAMIC]) == 1


class TestBootstrapFromFlatPool:
    def test_full_pool_20_entries(self, pool_setup):
        pool, store, db_path = pool_setup
        model = torch.nn.Linear(10, 10)
        for i in range(20):
            e = store.add_entry(model, "resnet", {}, epoch=i, role=Role.UNASSIGNED)
            store.update_elo(e.id, 800 + i * 20)
        pool.bootstrap_from_flat_pool()
        fs = store.list_by_role(Role.FRONTIER_STATIC)
        rf = store.list_by_role(Role.RECENT_FIXED)
        dy = store.list_by_role(Role.DYNAMIC)
        assert len(fs) == 5
        assert len(rf) == 5
        assert len(dy) == 10
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT bootstrapped FROM league_meta WHERE id = 1").fetchone()
        conn.close()
        assert row[0] == 1

    def test_small_pool_8_entries(self, pool_setup):
        pool, store, _ = pool_setup
        model = torch.nn.Linear(10, 10)
        for i in range(8):
            e = store.add_entry(model, "resnet", {}, epoch=i, role=Role.UNASSIGNED)
            store.update_elo(e.id, 800 + i * 50)
        pool.bootstrap_from_flat_pool()
        fs = store.list_by_role(Role.FRONTIER_STATIC)
        rf = store.list_by_role(Role.RECENT_FIXED)
        dy = store.list_by_role(Role.DYNAMIC)
        total_active = len(fs) + len(rf) + len(dy)
        assert total_active == 8
        assert len(fs) >= 1
        assert len(rf) >= 1
        assert len(dy) >= 1

    def test_bootstrap_is_idempotent(self, pool_setup):
        pool, store, db_path = pool_setup
        model = torch.nn.Linear(10, 10)
        for i in range(5):
            store.add_entry(model, "resnet", {}, epoch=i, role=Role.UNASSIGNED)
        pool.bootstrap_from_flat_pool()
        pool.bootstrap_from_flat_pool()
        assert len(store.list_entries()) == 5


class TestFullLifecycle:
    def test_lifecycle_snapshot_to_eviction(self, pool_setup):
        pool, store, db_path = pool_setup
        model = torch.nn.Linear(10, 10)

        for i in range(1, 6):
            entry = pool.snapshot_learner(model, "resnet", {}, epoch=i)
            assert entry.role is Role.RECENT_FIXED
        assert len(store.list_by_role(Role.RECENT_FIXED)) == 5

        for i in range(10):
            e = store.add_entry(model, "resnet", {}, epoch=100 + i, role=Role.DYNAMIC)
            store.update_elo(e.id, 900 + i * 10)

        pool.snapshot_learner(model, "resnet", {}, epoch=7)
        rf = store.list_by_role(Role.RECENT_FIXED)
        assert all(e.created_epoch != 1 for e in rf)

        dynamic_before = store.list_by_role(Role.DYNAMIC)
        assert len(dynamic_before) == 10


class TestListAllActive:
    def test_returns_all_tiers(self, pool_setup):
        pool, store, _ = pool_setup
        model = torch.nn.Linear(10, 10)
        store.add_entry(model, "resnet", {}, epoch=1, role=Role.FRONTIER_STATIC)
        store.add_entry(model, "resnet", {}, epoch=2, role=Role.RECENT_FIXED)
        store.add_entry(model, "resnet", {}, epoch=3, role=Role.DYNAMIC)
        assert len(pool.list_all_active()) == 3

    def test_excludes_retired(self, pool_setup):
        pool, store, _ = pool_setup
        model = torch.nn.Linear(10, 10)
        e = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        store.retire_entry(e.id, "test")
        assert len(pool.list_all_active()) == 0
