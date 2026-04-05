"""Tests for TieredPool -- the orchestrator."""

import sqlite3
from unittest.mock import patch

import pytest
import torch

from keisei.config import (
    DynamicConfig,
    FrontierStaticConfig,
    HistoricalLibraryConfig,
    LeagueConfig,
    RecentFixedConfig,
)
from keisei.db import init_db
from keisei.training.opponent_store import OpponentStore, Role, EntryStatus
from keisei.training.tiered_pool import TieredPool

pytestmark = pytest.mark.integration


@pytest.fixture
def pool_setup(tmp_path):
    db_path = str(tmp_path / "pool.db")
    init_db(db_path)
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    store = OpponentStore(db_path, str(league_dir))
    config = LeagueConfig()
    pool = TieredPool(store, config)
    yield pool, store, db_path
    store.close()


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
        # Capture role distribution after first bootstrap
        fs_first = len(store.list_by_role(Role.FRONTIER_STATIC))
        rf_first = len(store.list_by_role(Role.RECENT_FIXED))
        dy_first = len(store.list_by_role(Role.DYNAMIC))
        pool.bootstrap_from_flat_pool()
        # Second bootstrap should not change count or role distribution
        assert len(store.list_entries()) == 5
        assert len(store.list_by_role(Role.FRONTIER_STATIC)) == fs_first
        assert len(store.list_by_role(Role.RECENT_FIXED)) == rf_first
        assert len(store.list_by_role(Role.DYNAMIC)) == dy_first


class TestFullLifecycle:
    def test_lifecycle_snapshot_overflow_retires_oldest(self, tmp_path):
        """Snapshot fills Recent Fixed; overflow triggers retirement of oldest entry.

        Uses soft_overflow=0 so the first overflow immediately retires the
        under-calibrated oldest entry instead of delaying it.  Total pool must
        be at or above capacity for the retirement to actually proceed (the
        capacity guard skips retirement when spare capacity exists).
        """
        db_path = str(tmp_path / "overflow.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))
        # Tiny total capacity: frontier=1, recent=5, dynamic=10 → 16
        config = LeagueConfig(
            recent=RecentFixedConfig(soft_overflow=0),
            frontier=FrontierStaticConfig(slots=1),
        )
        pool = TieredPool(store, config)

        model = torch.nn.Linear(10, 10)

        for i in range(1, 6):
            entry = pool.snapshot_learner(model, "resnet", {}, epoch=i)
            assert entry.role is Role.RECENT_FIXED
        assert len(store.list_by_role(Role.RECENT_FIXED)) == 5

        for i in range(10):
            e = store.add_entry(model, "resnet", {}, epoch=100 + i, role=Role.DYNAMIC)
            store.update_elo(e.id, 900 + i * 10)

        # Fill frontier to reach total capacity (5 RF + 10 D + 1 FS = 16 = capacity)
        store.add_entry(model, "resnet", {}, epoch=200, role=Role.FRONTIER_STATIC)

        # This snapshot overflows Recent Fixed AND exceeds total capacity → retires oldest
        pool.snapshot_learner(model, "resnet", {}, epoch=7)
        rf = store.list_by_role(Role.RECENT_FIXED)
        assert all(e.created_epoch != 1 for e in rf)

        dynamic_before = store.list_by_role(Role.DYNAMIC)
        assert len(dynamic_before) == 10
        store.close()


class TestCapacityGuard:
    """Entries must not be retired when the total pool has spare capacity."""

    def test_no_retirement_when_pool_has_spare_capacity(self, tmp_path):
        """snapshot_learner skips retirement when total active < total capacity.

        With soft_overflow=0, review_oldest returns RETIRE immediately for
        under-calibrated entries. But the capacity guard should prevent the
        actual retirement.
        """
        db_path = str(tmp_path / "capacity.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))
        # slots=2, soft_overflow=0: overflow immediately triggers RETIRE review
        # Total capacity = frontier(5) + recent(2) + dynamic(10) = 17
        config = LeagueConfig(recent=RecentFixedConfig(slots=2, soft_overflow=0))
        pool = TieredPool(store, config)
        model = torch.nn.Linear(10, 10)

        # Fill Recent Fixed to capacity
        pool.snapshot_learner(model, "resnet", {}, epoch=1)
        pool.snapshot_learner(model, "resnet", {}, epoch=2)
        assert len(store.list_by_role(Role.RECENT_FIXED)) == 2

        # Overflow: review returns RETIRE but pool has spare capacity (3/17)
        pool.snapshot_learner(model, "resnet", {}, epoch=3)
        rf = store.list_by_role(Role.RECENT_FIXED)
        # All 3 entries should survive — capacity guard prevents retirement
        assert len(rf) == 3
        assert any(e.created_epoch == 1 for e in rf), "oldest entry should NOT be retired"
        store.close()

    def test_retirement_allowed_when_pool_at_capacity(self, tmp_path):
        """snapshot_learner retires oldest when total pool is at capacity."""
        db_path = str(tmp_path / "full.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))
        # Tiny total capacity: frontier=1, recent=2, dynamic=1 → total=4
        config = LeagueConfig(
            recent=RecentFixedConfig(slots=2, soft_overflow=0),
            dynamic=DynamicConfig(slots=1),
            frontier=FrontierStaticConfig(slots=1),
        )
        pool = TieredPool(store, config)
        model = torch.nn.Linear(10, 10)

        # Fill Recent Fixed to capacity
        pool.snapshot_learner(model, "resnet", {}, epoch=1)
        pool.snapshot_learner(model, "resnet", {}, epoch=2)
        # Add entries to other tiers to fill total capacity
        store.add_entry(model, "resnet", {}, epoch=50, role=Role.DYNAMIC)
        store.add_entry(model, "resnet", {}, epoch=51, role=Role.FRONTIER_STATIC)
        assert len(store.list_entries()) == 4  # at capacity

        # Overflow: pool is at capacity, so retirement should proceed
        pool.snapshot_learner(model, "resnet", {}, epoch=3)
        rf = store.list_by_role(Role.RECENT_FIXED)
        # 5 total after new snapshot, capacity=4 → oldest should be retired
        # leaving 4 total (2 RF + 1 D + 1 FS)
        assert not any(e.created_epoch == 1 for e in rf), "oldest should be retired"
        store.close()


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


class TestOnEpochEndHistoricalRefresh:
    """Regression: on_epoch_end must respect min_epoch_for_selection."""

    def test_no_refresh_before_min_epoch(self, tmp_path):
        """Historical refresh should NOT fire at epochs below min_epoch_for_selection."""
        db_path = str(tmp_path / "pool.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))
        # refresh_interval=5, min_epoch=20: epoch 5/10/15 should NOT refresh
        config = LeagueConfig(
            history=HistoricalLibraryConfig(
                refresh_interval_epochs=5,
                min_epoch_for_selection=20,
            )
        )
        pool = TieredPool(store, config)

        with patch.object(pool.historical_library, "refresh") as mock_refresh:
            pool.on_epoch_end(5)
            pool.on_epoch_end(10)
            pool.on_epoch_end(15)
            mock_refresh.assert_not_called()

            # Epoch 20 should trigger
            pool.on_epoch_end(20)
            mock_refresh.assert_called_once_with(20)

        store.close()

    def test_refresh_at_valid_epoch(self, pool_setup):
        """Historical refresh fires at epochs meeting both interval and min_epoch."""
        pool, store, _ = pool_setup
        # Default: refresh_interval=100, min_epoch=10
        with patch.object(pool.historical_library, "refresh") as mock_refresh:
            pool.on_epoch_end(100)
            mock_refresh.assert_called_once_with(100)


# ===========================================================================
# DELAY outcome in snapshot_learner
# ===========================================================================


class TestSnapshotLearnerDelay:
    """Test DELAY path: entry stays in overflow when under-calibrated."""

    def test_snapshot_learner_delay_undercalibrated(self, tmp_path):
        """Entry stays in overflow when under min_opponents threshold (DELAY)."""
        db_path = str(tmp_path / "delay.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))
        # soft_overflow=2 gives room for delay; min_games_for_review=32 means
        # entries with 0 games are under-calibrated
        config = LeagueConfig(recent=RecentFixedConfig(
            slots=2, soft_overflow=2, min_games_for_review=32,
        ))
        pool = TieredPool(store, config)
        model = torch.nn.Linear(10, 10)

        # Fill to capacity
        pool.snapshot_learner(model, "resnet", {}, epoch=1)
        pool.snapshot_learner(model, "resnet", {}, epoch=2)
        assert len(store.list_by_role(Role.RECENT_FIXED)) == 2

        # Overflow: triggers review_oldest, but oldest has 0 games -> DELAY
        pool.snapshot_learner(model, "resnet", {}, epoch=3)
        # All 3 entries should remain (oldest was delayed, not retired/promoted)
        rf = store.list_by_role(Role.RECENT_FIXED)
        assert len(rf) == 3
        # The oldest entry (epoch=1) should still be present
        assert any(e.created_epoch == 1 for e in rf)

        store.close()


# ===========================================================================
# bootstrap_from_flat_pool edge cases
# ===========================================================================


class TestBootstrapEdgeCases:
    """Edge cases for bootstrap_from_flat_pool with very small pools."""

    def test_bootstrap_single_entry(self, pool_setup):
        """1 entry -> assigned to some tier without crash."""
        pool, store, _ = pool_setup
        model = torch.nn.Linear(10, 10)
        store.add_entry(model, "resnet", {}, epoch=1, role=Role.UNASSIGNED)
        pool.bootstrap_from_flat_pool()

        # Entry should no longer be UNASSIGNED
        all_entries = store.list_entries()
        assert len(all_entries) == 1
        assert all_entries[0].role is not Role.UNASSIGNED

    def test_bootstrap_two_entries(self, pool_setup):
        """2 entries -> each gets a role (no crash, no UNASSIGNED remaining)."""
        pool, store, _ = pool_setup
        model = torch.nn.Linear(10, 10)
        store.add_entry(model, "resnet", {}, epoch=1, role=Role.UNASSIGNED)
        store.add_entry(model, "resnet", {}, epoch=2, role=Role.UNASSIGNED)
        pool.bootstrap_from_flat_pool()

        all_entries = store.list_entries()
        assert len(all_entries) == 2
        for e in all_entries:
            assert e.role is not Role.UNASSIGNED
        # Both entries should have some role assigned
        roles = {e.role for e in all_entries}
        assert len(roles) >= 1  # at least one role type used


# ===========================================================================
# training_enabled=False path
# ===========================================================================


class TestTrainingDisabled:
    """Test that training_enabled=False produces no DynamicTrainer."""

    def test_init_training_disabled(self, tmp_path):
        """DynamicConfig(training_enabled=False) -> pool.dynamic_trainer is None."""
        from keisei.config import DynamicConfig as DC

        db_path = str(tmp_path / "disabled.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))
        config = LeagueConfig(dynamic=DC(training_enabled=False))
        pool = TieredPool(store, config)
        assert pool.dynamic_trainer is None
        store.close()
