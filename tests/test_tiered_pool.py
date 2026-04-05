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
from keisei.training.historical_gauntlet import HistoricalGauntlet
from keisei.training.match_scheduler import MatchScheduler
from keisei.training.opponent_store import EloColumn, OpponentStore, Role, EntryStatus
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


class TestSampleOpponentForLearner:
    def test_returns_entry_from_populated_tier(self, pool_setup):
        """sample_opponent_for_learner returns an entry from one of the tiers."""
        pool, store, _ = pool_setup
        model = torch.nn.Linear(10, 10)
        store.add_entry(model, "resnet", {}, epoch=1, role=Role.FRONTIER_STATIC)
        store.add_entry(model, "resnet", {}, epoch=2, role=Role.DYNAMIC)
        store.add_entry(model, "resnet", {}, epoch=3, role=Role.RECENT_FIXED)

        result = pool.sample_opponent_for_learner()
        assert result.role in (Role.FRONTIER_STATIC, Role.DYNAMIC, Role.RECENT_FIXED)

    def test_raises_when_no_entries(self, pool_setup):
        """sample_opponent_for_learner raises when pool is empty."""
        pool, store, _ = pool_setup
        with pytest.raises(ValueError, match="No entries available"):
            pool.sample_opponent_for_learner()

    def test_scheduler_attribute_exists(self, pool_setup):
        """TieredPool exposes a scheduler attribute."""
        pool, store, _ = pool_setup
        assert hasattr(pool, 'scheduler')
        assert isinstance(pool.scheduler, MatchScheduler)


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


class TestGauntletIntegration:
    """TieredPool creates and exposes gauntlet when enabled."""

    def test_gauntlet_created_when_enabled(self, pool_setup):
        pool, store, _ = pool_setup
        assert pool.gauntlet is not None
        assert isinstance(pool.gauntlet, HistoricalGauntlet)

    def test_gauntlet_not_created_when_disabled(self, tmp_path):
        from keisei.config import GauntletConfig
        db_path = str(tmp_path / "pool.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))
        config = LeagueConfig(gauntlet=GauntletConfig(enabled=False))
        pool = TieredPool(store, config)
        assert pool.gauntlet is None
        assert pool.is_gauntlet_due(100) is False
        store.close()

    def test_is_gauntlet_due_delegates(self, pool_setup):
        pool, store, _ = pool_setup
        # Default gauntlet interval is 100
        assert pool.is_gauntlet_due(100) is True
        assert pool.is_gauntlet_due(50) is False


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


# ===========================================================================
# T1: Core lifecycle paths
# ===========================================================================


class TestRecentFixedToDynamicPromote:
    """Recent Fixed -> Dynamic PROMOTE path via snapshot_learner overflow."""

    def test_promote_oldest_to_dynamic(self, tmp_path):
        """Fill Recent Fixed to capacity, qualify the oldest, trigger overflow.

        The oldest RF entry should be cloned into Dynamic with parent_entry_id
        linking back to the source, and the source RF entry should be retired.
        """
        db_path = str(tmp_path / "promote.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))

        # Small RF tier (slots=3), soft_overflow=0 so overflow triggers review
        # immediately. min_games=4 and min_unique_opponents=3 for fast qualification.
        # max_elo_spread=999 so spread check always passes.
        config = LeagueConfig(
            recent=RecentFixedConfig(
                slots=3,
                soft_overflow=0,
                min_games_for_review=4,
                min_unique_opponents=3,
                promotion_margin_elo=0.0,
                max_elo_spread=999.0,
            ),
            dynamic=DynamicConfig(slots=10),
            frontier=FrontierStaticConfig(slots=5),
        )
        pool = TieredPool(store, config)
        model = torch.nn.Linear(10, 10)

        # Fill Recent Fixed to capacity (3 entries)
        entries = []
        for i in range(1, 4):
            entries.append(pool.snapshot_learner(model, "resnet", {}, epoch=i))
        assert len(store.list_by_role(Role.RECENT_FIXED)) == 3

        # The oldest entry is entries[0] (epoch=1). Give it enough games and
        # unique opponents to qualify for promotion.
        oldest = entries[0]
        # Add a Dynamic entry so weakest_dynamic_elo returns a value
        dyn = store.add_entry(model, "resnet", {}, epoch=50, role=Role.DYNAMIC)
        store.update_elo(dyn.id, 800.0)
        # Ensure oldest RF has Elo above Dynamic floor
        store.update_elo(oldest.id, 900.0)
        # Ensure protection is zero and enough games for eviction on Dynamic side
        store.set_protection(dyn.id, 0)
        # Record elo_spread history so it stays low
        store.update_elo(oldest.id, 900.0, epoch=1)
        store.update_elo(oldest.id, 905.0, epoch=2)

        # Record results to give oldest entry enough games (4) with 3 unique opponents
        # Create some dummy opponent entries to record results against
        opp_ids = []
        for i in range(3):
            opp = store.add_entry(model, "resnet", {}, epoch=60 + i, role=Role.DYNAMIC)
            opp_ids.append(opp.id)
        for opp_id in opp_ids:
            store.record_result(
                epoch=10, entry_a_id=oldest.id, entry_b_id=opp_id,
                wins_a=1, wins_b=1, draws=0, match_type="tournament",
            )
        # oldest now has 6 games (3 opponents x 2 games each) and 3 unique opponents

        # Snapshot again to trigger overflow (4th entry in 3-slot tier)
        pool.snapshot_learner(model, "resnet", {}, epoch=4)

        # Assert: oldest RF should be retired
        oldest_refreshed = store.get_entry(oldest.id)
        assert oldest_refreshed is not None
        assert oldest_refreshed.status == EntryStatus.RETIRED

        # Assert: a new Dynamic entry exists with parent_entry_id == oldest.id
        dynamic_entries = store.list_by_role(Role.DYNAMIC)
        promoted = [e for e in dynamic_entries if e.parent_entry_id == oldest.id]
        assert len(promoted) == 1, f"Expected 1 promoted Dynamic clone, got {len(promoted)}"
        assert promoted[0].role is Role.DYNAMIC

        store.close()


class TestDynamicToFrontierViaEpochEnd:
    """Dynamic -> Frontier Static clone via on_epoch_end."""

    def test_frontier_promotion_via_on_epoch_end(self, tmp_path):
        """Set up a Dynamic entry meeting all FrontierPromoter criteria.

        Call on_epoch_end at the review interval and verify a new Frontier
        Static clone exists.
        """
        db_path = str(tmp_path / "frontier_promo.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))

        review_interval = 10
        streak_required = 5
        config = LeagueConfig(
            frontier=FrontierStaticConfig(
                slots=5,
                review_interval_epochs=review_interval,
                min_games_for_promotion=4,
                topk=3,
                streak_epochs=streak_required,
                promotion_margin_elo=10.0,
                min_tenure_epochs=0,
            ),
            dynamic=DynamicConfig(slots=10),
            recent=RecentFixedConfig(slots=5),
        )
        pool = TieredPool(store, config)
        model = torch.nn.Linear(10, 10)

        # Create a Frontier entry (so the tier is non-empty but below capacity)
        fs = store.add_entry(model, "resnet", {}, epoch=1, role=Role.FRONTIER_STATIC)
        # Set its elo_frontier low so Dynamic candidate can beat it by margin
        store.update_role_elo(fs.id, EloColumn.FRONTIER, 800.0)

        # Create a Dynamic entry that will be the promotion candidate
        dyn = store.add_entry(model, "resnet", {}, epoch=2, role=Role.DYNAMIC)
        # Give it high elo_frontier (> weakest frontier + margin = 800 + 10 = 810)
        store.update_role_elo(dyn.id, EloColumn.FRONTIER, 900.0)
        # Create a second Dynamic entry as a match opponent, with low elo_frontier
        # so it doesn't accidentally qualify for promotion ahead of dyn
        opp = store.add_entry(model, "resnet", {}, epoch=3, role=Role.DYNAMIC)
        store.update_role_elo(opp.id, EloColumn.FRONTIER, 700.0)
        # Give dyn enough games via match results
        store.record_result(
            epoch=5, entry_a_id=dyn.id, entry_b_id=opp.id,
            wins_a=2, wins_b=2, draws=0, match_type="tournament",
        )
        # dyn now has 4 games_played

        # Build up the streak by calling evaluate() at each epoch in the streak window.
        # The FrontierPromoter is inside pool.frontier_manager._promoter.
        promoter = pool.frontier_manager._promoter
        assert promoter is not None

        # Seed the streak: call evaluate at epoch 0 so the candidate enters top-K
        dynamic_entries = store.list_by_role(Role.DYNAMIC)
        frontier_entries = store.list_by_role(Role.FRONTIER_STATIC)
        for ep in range(0, streak_required):
            promoter.evaluate(dynamic_entries, frontier_entries, ep)

        # Now at epoch = streak_required, the streak should be long enough.
        # Trigger on_epoch_end at a review_interval multiple >= streak_required
        review_epoch = review_interval  # = 10, which is >= streak_required=5
        frontier_before = len(store.list_by_role(Role.FRONTIER_STATIC))
        pool.on_epoch_end(review_epoch)
        frontier_after = store.list_by_role(Role.FRONTIER_STATIC)

        assert len(frontier_after) == frontier_before + 1
        # The new Frontier entry should be a clone of dyn
        clones = [e for e in frontier_after if e.parent_entry_id == dyn.id]
        assert len(clones) == 1
        assert clones[0].role is Role.FRONTIER_STATIC

        store.close()


class TestDynamicEvictionWhenFull:
    """Dynamic eviction when the tier is full and a new entry is promoted in."""

    def test_evicts_lowest_elo_dynamic(self, tmp_path):
        """Fill Dynamic to 10, promote a RF entry. The lowest-Elo unprotected
        Dynamic should be evicted (status='retired').
        """
        db_path = str(tmp_path / "eviction.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))

        config = LeagueConfig(
            recent=RecentFixedConfig(
                slots=3,
                soft_overflow=0,
                min_games_for_review=2,
                min_unique_opponents=1,
                promotion_margin_elo=0.0,
                max_elo_spread=999.0,
            ),
            dynamic=DynamicConfig(
                slots=10,
                min_games_before_eviction=2,
                protection_matches=0,
            ),
            frontier=FrontierStaticConfig(slots=5),
        )
        pool = TieredPool(store, config)
        model = torch.nn.Linear(10, 10)

        # Fill Dynamic tier to capacity (10 entries)
        dyn_entries = []
        for i in range(10):
            e = store.add_entry(model, "resnet", {}, epoch=100 + i, role=Role.DYNAMIC)
            elo = 900.0 + i * 10  # 900, 910, ..., 990
            store.update_elo(e.id, elo)
            store.update_role_elo(e.id, EloColumn.DYNAMIC, elo)
            store.set_protection(e.id, 0)
            dyn_entries.append(e)
        # Give all Dynamic entries enough games for eviction eligibility
        for e in dyn_entries:
            store.record_result(
                epoch=10, entry_a_id=e.id, entry_b_id=dyn_entries[-1].id,
                wins_a=1, wins_b=1, draws=0, match_type="tournament",
            )

        weakest_dyn = dyn_entries[0]  # elo_dynamic = 900

        # Fill RF to capacity, then qualify the oldest for promotion
        rf_entries = []
        for i in range(1, 4):
            rf_entries.append(pool.snapshot_learner(model, "resnet", {}, epoch=i))
        oldest_rf = rf_entries[0]
        store.update_elo(oldest_rf.id, 950.0)
        store.update_elo(oldest_rf.id, 950.0, epoch=1)
        store.update_elo(oldest_rf.id, 955.0, epoch=2)
        # Record a result to satisfy min_games=2 and min_unique_opponents=1
        store.record_result(
            epoch=10, entry_a_id=oldest_rf.id, entry_b_id=dyn_entries[0].id,
            wins_a=1, wins_b=1, draws=0, match_type="tournament",
        )

        # Trigger overflow -> PROMOTE -> Dynamic.admit evicts weakest
        pool.snapshot_learner(model, "resnet", {}, epoch=4)

        # The weakest Dynamic entry should be retired
        weakest_refreshed = store.get_entry(weakest_dyn.id)
        assert weakest_refreshed is not None
        assert weakest_refreshed.status == EntryStatus.RETIRED

        # Dynamic tier should still have exactly 10 entries
        assert len(store.list_by_role(Role.DYNAMIC)) == 10

        store.close()


class TestProtectedCandidateIds:
    """Protected candidates survive eviction per section 7.2."""

    def test_protected_candidate_survives_eviction(self, tmp_path):
        """The lowest-Elo Dynamic entry tracked in FrontierPromoter._topk_streaks
        should be protected from eviction; the next-lowest is evicted instead.
        """
        db_path = str(tmp_path / "protect.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))

        config = LeagueConfig(
            recent=RecentFixedConfig(
                slots=3,
                soft_overflow=0,
                min_games_for_review=2,
                min_unique_opponents=1,
                promotion_margin_elo=0.0,
                max_elo_spread=999.0,
            ),
            dynamic=DynamicConfig(
                slots=10,
                min_games_before_eviction=2,
                protection_matches=0,
            ),
            frontier=FrontierStaticConfig(slots=5),
        )
        pool = TieredPool(store, config)
        model = torch.nn.Linear(10, 10)

        # Fill Dynamic tier to capacity
        dyn_entries = []
        for i in range(10):
            e = store.add_entry(model, "resnet", {}, epoch=100 + i, role=Role.DYNAMIC)
            elo = 900.0 + i * 10  # 900, 910, ..., 990
            store.update_elo(e.id, elo)
            store.update_role_elo(e.id, EloColumn.DYNAMIC, elo)
            store.set_protection(e.id, 0)
            dyn_entries.append(e)

        # Give all Dynamic entries enough games for eviction eligibility
        for e in dyn_entries:
            store.record_result(
                epoch=10, entry_a_id=e.id, entry_b_id=dyn_entries[-1].id,
                wins_a=1, wins_b=1, draws=0, match_type="tournament",
            )

        lowest_dyn = dyn_entries[0]   # elo_dynamic = 900
        second_lowest = dyn_entries[1]  # elo_dynamic = 910

        # Put the lowest-Elo Dynamic entry into FrontierPromoter's top-K streaks
        # so it becomes a protected candidate
        promoter = pool.frontier_manager._promoter
        assert promoter is not None
        promoter._topk_streaks[lowest_dyn.id] = 0  # epoch when first seen

        # Fill RF and qualify oldest for promotion
        rf_entries = []
        for i in range(1, 4):
            rf_entries.append(pool.snapshot_learner(model, "resnet", {}, epoch=i))
        oldest_rf = rf_entries[0]
        store.update_elo(oldest_rf.id, 950.0)
        store.update_elo(oldest_rf.id, 950.0, epoch=1)
        store.update_elo(oldest_rf.id, 955.0, epoch=2)
        store.record_result(
            epoch=10, entry_a_id=oldest_rf.id, entry_b_id=dyn_entries[0].id,
            wins_a=1, wins_b=1, draws=0, match_type="tournament",
        )

        # Trigger overflow -> PROMOTE -> Dynamic.admit attempts eviction
        pool.snapshot_learner(model, "resnet", {}, epoch=4)

        # The protected lowest-Elo entry should survive
        lowest_refreshed = store.get_entry(lowest_dyn.id)
        assert lowest_refreshed is not None
        assert lowest_refreshed.status == EntryStatus.ACTIVE, (
            f"Protected entry {lowest_dyn.id} should survive eviction"
        )

        # The second-lowest should be evicted instead
        second_refreshed = store.get_entry(second_lowest.id)
        assert second_refreshed is not None
        assert second_refreshed.status == EntryStatus.RETIRED, (
            f"Unprotected entry {second_lowest.id} should be evicted"
        )

        store.close()
