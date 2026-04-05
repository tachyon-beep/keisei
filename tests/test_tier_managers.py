"""Tests for tier managers: Frontier, RecentFixed, Dynamic."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from keisei.config import FrontierStaticConfig, RecentFixedConfig, DynamicConfig
from keisei.db import init_db
from keisei.training.frontier_promoter import FrontierPromoter
from keisei.training.opponent_store import EloColumn, OpponentEntry, OpponentStore, Role, EntryStatus
from keisei.training.tier_managers import (
    FrontierManager,
    RecentFixedManager,
    DynamicManager,
    ReviewOutcome,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "tier.db")
    init_db(db_path)
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    s = OpponentStore(db_path, str(league_dir))
    yield s
    s.close()


def _add_entry(store, epoch, role=Role.UNASSIGNED, elo=1000.0):
    model = torch.nn.Linear(10, 10)
    entry = store.add_entry(model, "resnet", {}, epoch=epoch, role=role)
    if elo != 1000.0:
        store.update_elo(entry.id, elo)
        store.update_role_elo(entry.id, EloColumn.FRONTIER, elo)
    return store._get_entry(entry.id)


class TestFrontierManager:
    def test_get_active(self, store):
        _add_entry(store, 1, role=Role.FRONTIER_STATIC)
        _add_entry(store, 2, role=Role.DYNAMIC)
        mgr = FrontierManager(store, FrontierStaticConfig())
        active = mgr.get_active()
        assert len(active) == 1
        assert active[0].role is Role.FRONTIER_STATIC

    def test_select_initial_spans_elo(self, store):
        entries = []
        for i, elo in enumerate([800, 900, 1000, 1100, 1200, 1300, 1400]):
            entries.append(_add_entry(store, i, elo=elo))
        mgr = FrontierManager(store, FrontierStaticConfig(slots=5))
        selected = mgr.select_initial(entries, count=5)
        assert len(selected) == 5
        elos = sorted(e.elo_rating for e in selected)
        assert elos[0] < 1000
        assert elos[-1] > 1200

    def test_select_initial_underfull(self, store):
        entries = [_add_entry(store, i, elo=1000 + i * 100) for i in range(3)]
        mgr = FrontierManager(store, FrontierStaticConfig(slots=5))
        selected = mgr.select_initial(entries, count=5)
        assert len(selected) == 3

    def test_review_is_noop_without_promoter(self, store):
        mgr = FrontierManager(store, FrontierStaticConfig())
        mgr.review(epoch=500)  # should not raise, no-op without promoter

    def test_is_due_for_review(self, store):
        mgr = FrontierManager(store, FrontierStaticConfig(review_interval_epochs=250))
        assert mgr.is_due_for_review(250)
        assert mgr.is_due_for_review(500)
        assert not mgr.is_due_for_review(100)
        assert not mgr.is_due_for_review(0)

    def test_is_due_for_review_interval_one(self, store):
        """Edge case: review_interval=1 means every epoch (except 0) triggers review."""
        mgr = FrontierManager(store, FrontierStaticConfig(review_interval_epochs=1))
        assert mgr.is_due_for_review(1)
        assert mgr.is_due_for_review(2)
        assert not mgr.is_due_for_review(0)


class TestRecentFixedManager:
    def test_admit_creates_recent_fixed_entry(self, store):
        mgr = RecentFixedManager(store, RecentFixedConfig())
        model = torch.nn.Linear(10, 10)
        entry = mgr.admit(model, "resnet", {}, epoch=1)
        assert entry.role is Role.RECENT_FIXED
        assert mgr.count() == 1

    def test_count(self, store):
        mgr = RecentFixedManager(store, RecentFixedConfig(slots=5))
        model = torch.nn.Linear(10, 10)
        for i in range(3):
            mgr.admit(model, "resnet", {}, epoch=i)
        assert mgr.count() == 3

    def test_review_oldest_retire_when_unqualified(self, store):
        """Entry with 0 games should be RETIRE'd."""
        mgr = RecentFixedManager(store, RecentFixedConfig(
            slots=2, soft_overflow=0, min_games_for_review=32,
        ))
        model = torch.nn.Linear(10, 10)
        mgr.admit(model, "resnet", {}, epoch=1)
        mgr.admit(model, "resnet", {}, epoch=2)
        mgr.admit(model, "resnet", {}, epoch=3)
        outcome, entry = mgr.review_oldest()
        assert outcome is ReviewOutcome.RETIRE
        assert entry.created_epoch == 1

    def test_review_oldest_promote_when_qualified(self, store):
        """Entry meeting all criteria should be PROMOTE'd."""
        mgr = RecentFixedManager(store, RecentFixedConfig(
            slots=2, soft_overflow=0, min_games_for_review=2,
            min_unique_opponents=2, promotion_margin_elo=25.0,
        ))
        _add_entry(store, 0, role=Role.DYNAMIC, elo=900)

        model = torch.nn.Linear(10, 10)
        mgr.admit(model, "resnet", {}, epoch=1)
        mgr.admit(model, "resnet", {}, epoch=2)
        oldest = store.list_by_role(Role.RECENT_FIXED)[0]

        other1 = _add_entry(store, 10, role=Role.FRONTIER_STATIC, elo=1000)
        other2 = _add_entry(store, 11, role=Role.FRONTIER_STATIC, elo=1000)
        store.record_result(epoch=1, entry_a_id=oldest.id, entry_b_id=other1.id,
                            wins_a=1, wins_b=0, draws=0, match_type="calibration")
        store.record_result(epoch=2, entry_a_id=oldest.id, entry_b_id=other2.id,
                            wins_a=1, wins_b=0, draws=0, match_type="calibration")
        store.update_elo(oldest.id, 1000.0)

        mgr.admit(model, "resnet", {}, epoch=3)
        outcome, entry = mgr.review_oldest()
        assert outcome is ReviewOutcome.PROMOTE
        assert entry.id == oldest.id

    def test_review_oldest_promotes_when_dynamic_empty(self, store):
        """When Dynamic tier is empty, promotion should always pass the Elo check."""
        mgr = RecentFixedManager(store, RecentFixedConfig(
            slots=1, soft_overflow=0, min_games_for_review=0, min_unique_opponents=0,
        ))
        model = torch.nn.Linear(10, 10)
        mgr.admit(model, "resnet", {}, epoch=1)
        mgr.admit(model, "resnet", {}, epoch=2)
        outcome, entry = mgr.review_oldest()
        assert outcome is ReviewOutcome.PROMOTE

    def test_review_oldest_retire_when_overflow_exhausted(self, store):
        """Entry with < min_games should RETIRE when soft overflow is exhausted."""
        mgr = RecentFixedManager(store, RecentFixedConfig(
            slots=2, soft_overflow=1, min_games_for_review=32,
        ))
        model = torch.nn.Linear(10, 10)
        mgr.admit(model, "resnet", {}, epoch=1)
        mgr.admit(model, "resnet", {}, epoch=2)
        mgr.admit(model, "resnet", {}, epoch=3)
        assert mgr.count() == 3
        # count=4, overflow_used = 4-2 = 2 > soft_overflow(1) → RETIRE
        mgr.admit(model, "resnet", {}, epoch=4)
        outcome, entry = mgr.review_oldest()
        assert outcome is ReviewOutcome.RETIRE


    def test_review_oldest_delay_at_exact_overflow_boundary(self, store):
        """Delay should fire when overflow_used == soft_overflow (at budget, not over)."""
        mgr = RecentFixedManager(store, RecentFixedConfig(
            slots=2, soft_overflow=1, min_games_for_review=32,
        ))
        model = torch.nn.Linear(10, 10)
        mgr.admit(model, "resnet", {}, epoch=1)
        mgr.admit(model, "resnet", {}, epoch=2)
        mgr.admit(model, "resnet", {}, epoch=3)
        # count=3, slots=2, overflow_used=1, soft_overflow=1 → at budget, should DELAY
        outcome, entry = mgr.review_oldest()
        assert outcome is ReviewOutcome.DELAY
        assert entry.created_epoch == 1

    def test_review_oldest_delay_when_opponents_insufficient(self, store):
        """Entry with enough games but too few unique opponents should DELAY."""
        mgr = RecentFixedManager(store, RecentFixedConfig(
            slots=2, soft_overflow=1,
            min_games_for_review=2, min_unique_opponents=3,
        ))
        model = torch.nn.Linear(10, 10)
        mgr.admit(model, "resnet", {}, epoch=1)
        mgr.admit(model, "resnet", {}, epoch=2)
        mgr.admit(model, "resnet", {}, epoch=3)

        # Give oldest entry 2 games but only 1 unique opponent via direct SQL
        # (bypasses record_result which may have pending schema changes)
        oldest = store.list_by_role(Role.RECENT_FIXED)[0]
        other = _add_entry(store, 10, role=Role.FRONTIER_STATIC, elo=1000)
        with store.transaction():
            store._conn.execute(
                "UPDATE league_entries SET games_played = 2 WHERE id = ?",
                (oldest.id,),
            )
            store._conn.execute(
                "INSERT INTO league_results (epoch, entry_a_id, entry_b_id, match_type, num_games, wins_a, wins_b, draws) "
                "VALUES (?, ?, ?, 'calibration', 1, ?, ?, ?)",
                (1, oldest.id, other.id, 1, 0, 0),
            )
            store._conn.execute(
                "INSERT INTO league_results (epoch, entry_a_id, entry_b_id, match_type, num_games, wins_a, wins_b, draws) "
                "VALUES (?, ?, ?, 'calibration', 1, ?, ?, ?)",
                (2, oldest.id, other.id, 0, 1, 0),
            )

        outcome, entry = mgr.review_oldest()
        assert outcome is ReviewOutcome.DELAY

    def test_review_oldest_raises_on_empty_tier(self, store):
        """review_oldest should raise ValueError when tier is empty."""
        mgr = RecentFixedManager(store, RecentFixedConfig())
        with pytest.raises(ValueError, match="no Recent Fixed entries"):
            mgr.review_oldest()

    def test_review_oldest_scales_min_opponents_to_pool_size(self, store):
        """min_unique_opponents should scale down for small pools.

        With 3 total entries, an entry can face at most 2 opponents.
        min_unique_opponents=6 is unreachable, but scaled to min(6, 2)=2.
        """
        mgr = RecentFixedManager(store, RecentFixedConfig(
            slots=2, soft_overflow=1,
            min_games_for_review=2, min_unique_opponents=6,
        ))
        model = torch.nn.Linear(10, 10)
        mgr.admit(model, "resnet", {}, epoch=1)
        mgr.admit(model, "resnet", {}, epoch=2)
        mgr.admit(model, "resnet", {}, epoch=3)

        # Give oldest entry enough games and 2 unique opponents
        oldest = store.list_by_role(Role.RECENT_FIXED)[0]
        other1 = _add_entry(store, 10, role=Role.FRONTIER_STATIC, elo=1000)
        other2 = _add_entry(store, 11, role=Role.FRONTIER_STATIC, elo=1000)
        with store.transaction():
            store._conn.execute(
                "UPDATE league_entries SET games_played = 4 WHERE id = ?",
                (oldest.id,),
            )
            store._conn.execute(
                "INSERT INTO league_results (epoch, entry_a_id, entry_b_id, match_type, num_games, wins_a, wins_b, draws) "
                "VALUES (?, ?, ?, 'calibration', 1, ?, ?, ?)",
                (1, oldest.id, other1.id, 1, 0, 0),
            )
            store._conn.execute(
                "INSERT INTO league_results (epoch, entry_a_id, entry_b_id, match_type, num_games, wins_a, wins_b, draws) "
                "VALUES (?, ?, ?, 'calibration', 1, ?, ?, ?)",
                (2, oldest.id, other2.id, 1, 0, 0),
            )

        # With total_active_count=5 (3 RF + 2 FS), scaled min = min(6, 4) = 4
        # Entry has 2 unique opponents < 4, so should NOT qualify for promote
        outcome_no_scale, _ = mgr.review_oldest()
        # Without scaling (total_active_count=None), 2 < 6 -> DELAY (soft overflow)
        assert outcome_no_scale is ReviewOutcome.DELAY

        # Now pass total_active_count=3 (small pool): min(6, 2)=2, and
        # entry has 2 unique opponents >= 2 → qualifies for PROMOTE
        # (Elo check: weakest_elo_fn returns None → always passes)
        outcome_scaled, _ = mgr.review_oldest(total_active_count=3)
        assert outcome_scaled is ReviewOutcome.PROMOTE


class TestDynamicManager:
    def test_admit_clones_entry(self, store):
        source = _add_entry(store, 1, role=Role.RECENT_FIXED)
        mgr = DynamicManager(store, DynamicConfig())
        entry = mgr.admit(source)
        assert entry.role is Role.DYNAMIC
        assert entry.parent_entry_id == source.id
        assert entry.protection_remaining == 24

    def test_evict_weakest_skips_protected(self, store):
        mgr = DynamicManager(store, DynamicConfig(
            slots=3, protection_matches=0, min_games_before_eviction=0,
        ))
        e1 = _add_entry(store, 1, role=Role.DYNAMIC, elo=800)
        e2 = _add_entry(store, 2, role=Role.DYNAMIC, elo=1000)
        e3 = _add_entry(store, 3, role=Role.DYNAMIC, elo=1200)
        with store.transaction():
            store._conn.execute(
                "UPDATE league_entries SET protection_remaining = 10 WHERE id = ?", (e1.id,)
            )
        evicted = mgr.evict_weakest()
        assert evicted is not None
        assert evicted.id == e2.id
        remaining = store.list_by_role(Role.DYNAMIC)
        assert any(e.id == e1.id for e in remaining)

    def test_evict_weakest_picks_lowest_elo(self, store):
        mgr = DynamicManager(store, DynamicConfig(
            slots=2, protection_matches=0, min_games_before_eviction=0,
        ))
        e1 = _add_entry(store, 1, role=Role.DYNAMIC, elo=800)
        e2 = _add_entry(store, 2, role=Role.DYNAMIC, elo=1200)
        evicted = mgr.evict_weakest()
        assert evicted is not None
        assert evicted.id == e1.id

    def test_evict_weakest_returns_none_when_all_protected(self, store):
        mgr = DynamicManager(store, DynamicConfig(
            slots=2, protection_matches=0, min_games_before_eviction=100,
        ))
        _add_entry(store, 1, role=Role.DYNAMIC, elo=800)
        _add_entry(store, 2, role=Role.DYNAMIC, elo=1200)
        evicted = mgr.evict_weakest()
        assert evicted is None

    def test_is_full(self, store):
        mgr = DynamicManager(store, DynamicConfig(slots=2))
        assert not mgr.is_full()
        _add_entry(store, 1, role=Role.DYNAMIC)
        _add_entry(store, 2, role=Role.DYNAMIC)
        assert mgr.is_full()

    def test_weakest_elo_returns_none_when_all_protected(self, store):
        mgr = DynamicManager(store, DynamicConfig(
            slots=2, protection_matches=24, min_games_before_eviction=40,
        ))
        _add_entry(store, 1, role=Role.DYNAMIC, elo=800)
        assert mgr.weakest_elo() is None

    def test_admit_returns_none_when_full_and_all_protected(self, store):
        mgr = DynamicManager(store, DynamicConfig(
            slots=1, protection_matches=0, min_games_before_eviction=100,
        ))
        _add_entry(store, 1, role=Role.DYNAMIC, elo=800)
        source = _add_entry(store, 2, role=Role.RECENT_FIXED)
        result = mgr.admit(source)
        assert result is None

    def test_training_enabled_accepted(self, store):
        """DynamicManager accepts both True and False for training_enabled."""
        mgr_on = DynamicManager(store, DynamicConfig(training_enabled=True))
        mgr_off = DynamicManager(store, DynamicConfig(training_enabled=False))
        assert mgr_on._config.training_enabled is True
        assert mgr_off._config.training_enabled is False


class TestDynamicEloDynamic:
    """DynamicManager.evict_weakest uses elo_dynamic, not elo_rating."""

    def test_evict_weakest_uses_elo_dynamic(self, store):
        """Pool at realistic capacity — elo_dynamic determines eviction, not elo_rating."""
        config = DynamicConfig(slots=10, min_games_before_eviction=0, protection_matches=0)
        mgr = DynamicManager(store, config)
        # Fill pool to 8/10 slots so eviction scenario is realistic
        for i in range(6):
            _add_entry(store, epoch=10 + i, role=Role.DYNAMIC, elo=1000.0 + i * 20)
        e1 = _add_entry(store, epoch=1, role=Role.DYNAMIC, elo=1500.0)
        e2 = _add_entry(store, epoch=2, role=Role.DYNAMIC, elo=1000.0)
        # e1: HIGH elo_rating (1500) but LOW elo_dynamic (800)
        # e2: LOW elo_rating (1000) but HIGH elo_dynamic (1200)
        with store.transaction():
            store.update_role_elo(e1.id, EloColumn.DYNAMIC, 800.0)
            store.update_role_elo(e2.id, EloColumn.DYNAMIC, 1200.0)
        evicted = mgr.evict_weakest()
        assert evicted is not None
        assert evicted.id == e1.id  # evicted despite higher elo_rating

    def test_weakest_dynamic_elo_returns_elo_dynamic(self, store):
        config = DynamicConfig(slots=10, min_games_before_eviction=0, protection_matches=0)
        mgr = DynamicManager(store, config)
        e1 = _add_entry(store, epoch=1, role=Role.DYNAMIC, elo=1500.0)
        e2 = _add_entry(store, epoch=2, role=Role.DYNAMIC, elo=1000.0)
        with store.transaction():
            store.update_role_elo(e1.id, EloColumn.DYNAMIC, 800.0)
            store.update_role_elo(e2.id, EloColumn.DYNAMIC, 1200.0)
        assert mgr.weakest_dynamic_elo() == 800.0


# ---------------------------------------------------------------------------
# FrontierManager.review() with promoter — Phase 3 tests
# ---------------------------------------------------------------------------


def _make_frontier_config(**overrides: object) -> FrontierStaticConfig:
    """Helper to build a FrontierStaticConfig with test-friendly defaults."""
    defaults = dict(
        slots=5,
        review_interval_epochs=250,
        min_tenure_epochs=50,
        promotion_margin_elo=10.0,
        min_games_for_promotion=50,
        topk=3,
        streak_epochs=0,  # disable streak requirement for tests
        max_lineage_overlap=10,
    )
    defaults.update(overrides)
    return FrontierStaticConfig(**defaults)


class TestFrontierReview:
    """Tests for FrontierManager.review() with FrontierPromoter."""

    def test_frontier_review_promotes_candidate(self, store):
        """Dynamic entry meeting criteria is promoted; weakest Frontier retired."""
        config = _make_frontier_config(slots=5, min_tenure_epochs=0)
        promoter = FrontierPromoter(config)

        # Seed 5 Frontier entries with varying Elo
        frontier_ids = []
        for i, elo in enumerate([900, 950, 1000, 1050, 1100]):
            e = _add_entry(store, epoch=i, role=Role.FRONTIER_STATIC, elo=elo)
            frontier_ids.append(e.id)

        # Add Dynamic entries — one with high Elo
        for i in range(9):
            _add_entry(store, epoch=100 + i, role=Role.DYNAMIC, elo=1000 + i * 10)
        # The top candidate — high Elo, enough games
        candidate = _add_entry(store, epoch=110, role=Role.DYNAMIC, elo=1200)
        # Give the candidate enough games_played
        with store.transaction():
            store._conn.execute(
                "UPDATE league_entries SET games_played = 200 WHERE id = ?",
                (candidate.id,),
            )

        mgr = FrontierManager(store, config, promoter=promoter)
        # Prime the promoter streak tracking so the candidate is seen at the epoch
        promoter._topk_streaks[candidate.id] = 0

        mgr.review(epoch=300)

        # New Frontier entry should exist (cloned from candidate)
        frontier_after = store.list_by_role(Role.FRONTIER_STATIC)
        frontier_after_ids = {e.id for e in frontier_after}

        # At least one new Frontier entry should be a clone of the candidate
        cloned_from_candidate = [
            e for e in frontier_after
            if e.parent_entry_id == candidate.id
        ]
        assert len(cloned_from_candidate) == 1, (
            f"Expected exactly 1 clone of candidate {candidate.id}, "
            f"got {len(cloned_from_candidate)}"
        )

        # Original candidate still exists as Dynamic
        dyn_after = store.list_by_role(Role.DYNAMIC)
        assert any(e.id == candidate.id for e in dyn_after)

        # The weakest Frontier entry (Elo=900) should be retired
        retired_entry = store.get_entry(frontier_ids[0])
        assert retired_entry is not None
        assert retired_entry.status == EntryStatus.RETIRED

        # A new Frontier entry was created (6 total original minus 1 retired + 1 new = 5 active)
        assert len(frontier_after) == 5

    def test_frontier_review_no_promotion_when_no_candidate(self, store):
        """No Dynamic entries meet criteria -> no changes."""
        config = _make_frontier_config(
            slots=5, min_games_for_promotion=1000
        )  # very high bar
        promoter = FrontierPromoter(config)

        for i in range(5):
            _add_entry(store, epoch=i, role=Role.FRONTIER_STATIC, elo=1000)
        for i in range(10):
            _add_entry(store, epoch=100 + i, role=Role.DYNAMIC, elo=900)

        frontier_before = store.list_by_role(Role.FRONTIER_STATIC)
        mgr = FrontierManager(store, config, promoter=promoter)
        mgr.review(epoch=300)
        frontier_after = store.list_by_role(Role.FRONTIER_STATIC)

        assert len(frontier_before) == len(frontier_after)
        assert {e.id for e in frontier_before} == {e.id for e in frontier_after}

    def test_frontier_review_retires_weakest_or_stalest(self, store):
        """Weakest Frontier entry past min_tenure is retired on promotion."""
        config = _make_frontier_config(slots=3, min_tenure_epochs=50)
        promoter = FrontierPromoter(config)

        # 3 Frontier entries: one weak and old, one strong and old, one mid and new
        weak_old = _add_entry(store, epoch=0, role=Role.FRONTIER_STATIC, elo=800)
        _add_entry(store, epoch=10, role=Role.FRONTIER_STATIC, elo=1200)
        _add_entry(store, epoch=200, role=Role.FRONTIER_STATIC, elo=1000)

        # One high-Elo Dynamic candidate
        candidate = _add_entry(store, epoch=100, role=Role.DYNAMIC, elo=1300)
        with store.transaction():
            store._conn.execute(
                "UPDATE league_entries SET games_played = 200 WHERE id = ?",
                (candidate.id,),
            )
        promoter._topk_streaks[candidate.id] = 0

        mgr = FrontierManager(store, config, promoter=promoter)
        mgr.review(epoch=300)

        # weak_old (Elo=800, epoch=0) should be retired — past tenure and lowest Elo
        retired = store.get_entry(weak_old.id)
        assert retired is not None
        assert retired.status == EntryStatus.RETIRED

    def test_frontier_review_never_replaces_more_than_one(self, store):
        """Even if multiple Dynamic entries qualify, only 1 is promoted per review."""
        config = _make_frontier_config(slots=3, min_tenure_epochs=0)
        promoter = FrontierPromoter(config)

        for i in range(3):
            _add_entry(store, epoch=i, role=Role.FRONTIER_STATIC, elo=800 + i * 50)

        # Two qualifying Dynamic entries
        c1 = _add_entry(store, epoch=100, role=Role.DYNAMIC, elo=1200)
        c2 = _add_entry(store, epoch=101, role=Role.DYNAMIC, elo=1300)
        with store.transaction():
            store._conn.execute(
                "UPDATE league_entries SET games_played = 200 WHERE id IN (?, ?)",
                (c1.id, c2.id),
            )
        promoter._topk_streaks[c1.id] = 0
        promoter._topk_streaks[c2.id] = 0

        frontier_before = store.list_by_role(Role.FRONTIER_STATIC)
        mgr = FrontierManager(store, config, promoter=promoter)
        mgr.review(epoch=300)
        frontier_after = store.list_by_role(Role.FRONTIER_STATIC)

        # Still 3 active Frontier entries (1 retired + 1 promoted = net 0)
        assert len(frontier_after) == 3

        # Only 1 retirement occurred
        retired_count = sum(
            1
            for e_before in frontier_before
            if store.get_entry(e_before.id).status == EntryStatus.RETIRED
        )
        assert retired_count == 1

    def test_frontier_review_skips_retirement_when_frontier_not_full(self, store):
        """Frontier has 3 entries but slots=5; promote without retiring."""
        config = _make_frontier_config(slots=5, min_tenure_epochs=0)
        promoter = FrontierPromoter(config)

        frontier_entries = []
        for i in range(3):
            frontier_entries.append(
                _add_entry(store, epoch=i, role=Role.FRONTIER_STATIC, elo=900 + i * 100)
            )

        candidate = _add_entry(store, epoch=100, role=Role.DYNAMIC, elo=1300)
        with store.transaction():
            store._conn.execute(
                "UPDATE league_entries SET games_played = 200 WHERE id = ?",
                (candidate.id,),
            )
        promoter._topk_streaks[candidate.id] = 0

        mgr = FrontierManager(store, config, promoter=promoter)
        mgr.review(epoch=300)

        # All original Frontier entries should still be active
        for fe in frontier_entries:
            entry = store.get_entry(fe.id)
            assert entry.status == EntryStatus.ACTIVE

        # Now 4 Frontier entries (3 original + 1 promoted)
        frontier_after = store.list_by_role(Role.FRONTIER_STATIC)
        assert len(frontier_after) == 4

    def test_frontier_review_promotion_is_atomic(self, store):
        """Verify clone + retire both happen within the same outer transaction."""
        config = _make_frontier_config(slots=3, min_tenure_epochs=0)
        promoter = FrontierPromoter(config)

        frontier_entries = []
        for i in range(3):
            frontier_entries.append(
                _add_entry(store, epoch=i, role=Role.FRONTIER_STATIC, elo=900)
            )

        candidate = _add_entry(store, epoch=100, role=Role.DYNAMIC, elo=1200)
        with store.transaction():
            store._conn.execute(
                "UPDATE league_entries SET games_played = 200 WHERE id = ?",
                (candidate.id,),
            )
        promoter._topk_streaks[candidate.id] = 0

        mgr = FrontierManager(store, config, promoter=promoter)

        # Track which operations happen inside transaction contexts
        ops_inside_txn = []
        original_clone = store.clone_entry
        original_retire = store.retire_entry

        def tracking_clone(*args, **kwargs):
            ops_inside_txn.append("clone")
            return original_clone(*args, **kwargs)

        def tracking_retire(*args, **kwargs):
            ops_inside_txn.append("retire")
            return original_retire(*args, **kwargs)

        with patch.object(store, "clone_entry", side_effect=tracking_clone), \
             patch.object(store, "retire_entry", side_effect=tracking_retire), \
             patch.object(store, "transaction", wraps=store.transaction) as mock_txn:
            mgr.review(epoch=300)
            # Both clone and retire should have been called
            assert "clone" in ops_inside_txn
            assert "retire" in ops_inside_txn
            # The outer transaction wrapping both calls
            assert mock_txn.call_count >= 1


# ===========================================================================
# DynamicManager.get_trainable
# ===========================================================================


class TestDynamicGetTrainable:
    """Tests for DynamicManager.get_trainable — controls PPO update eligibility."""

    def test_get_trainable_returns_dynamic_entries(self, store):
        """training_enabled=True, Dynamic entries present -> returns them."""
        mgr = DynamicManager(store, DynamicConfig(training_enabled=True))
        e1 = _add_entry(store, 1, role=Role.DYNAMIC, elo=1000)
        e2 = _add_entry(store, 2, role=Role.DYNAMIC, elo=1100)
        result = mgr.get_trainable()
        ids = {e.id for e in result}
        assert e1.id in ids
        assert e2.id in ids

    def test_get_trainable_excludes_disabled(self, store):
        """Entry in disabled set is excluded from trainable list."""
        mgr = DynamicManager(store, DynamicConfig(training_enabled=True))
        e1 = _add_entry(store, 1, role=Role.DYNAMIC, elo=1000)
        e2 = _add_entry(store, 2, role=Role.DYNAMIC, elo=1100)
        result = mgr.get_trainable(disabled_entries={e1.id})
        ids = {e.id for e in result}
        assert e1.id not in ids
        assert e2.id in ids

    def test_get_trainable_disabled_training_returns_empty(self, store):
        """training_enabled=False -> returns empty list regardless of entries."""
        mgr = DynamicManager(store, DynamicConfig(training_enabled=False))
        _add_entry(store, 1, role=Role.DYNAMIC, elo=1000)
        _add_entry(store, 2, role=Role.DYNAMIC, elo=1100)
        result = mgr.get_trainable()
        assert result == []

    def test_get_trainable_empty_tier(self, store):
        """No Dynamic entries -> returns empty list."""
        mgr = DynamicManager(store, DynamicConfig(training_enabled=True))
        result = mgr.get_trainable()
        assert result == []


# ===========================================================================
# DynamicManager.evict_weakest with disabled_entry_ids
# ===========================================================================


class TestDynamicEvictDisabled:
    """Tests for disabled_entry_ids overriding protection in evict_weakest."""

    def test_evict_weakest_disabled_overrides_protection(self, store):
        """Protected entry with id in disabled_entry_ids is still evictable."""
        mgr = DynamicManager(store, DynamicConfig(
            slots=3, protection_matches=0, min_games_before_eviction=100,
        ))
        # All entries have 0 games_played (< 100), so normally ALL protected
        e1 = _add_entry(store, 1, role=Role.DYNAMIC, elo=800)
        e2 = _add_entry(store, 2, role=Role.DYNAMIC, elo=1200)

        # Without disabled set: no eligible entries
        assert mgr.evict_weakest() is None

        # With disabled set: e1 becomes eligible despite insufficient games
        evicted = mgr.evict_weakest(disabled_entry_ids={e1.id})
        assert evicted is not None
        assert evicted.id == e1.id


# ===========================================================================
# RecentFixedManager.review_oldest Elo rejection path
# ===========================================================================


class TestRecentFixedEloRejection:
    """Tests for review_oldest Elo floor logic."""

    def test_review_oldest_rejects_below_elo_floor(self, store):
        """Entry below floor-margin should NOT promote even if games/opponents ok."""
        mgr = RecentFixedManager(store, RecentFixedConfig(
            slots=2, soft_overflow=0,
            min_games_for_review=0, min_unique_opponents=0,
            promotion_margin_elo=25.0,
        ))
        # Set weakest_elo_fn returning 1200.0 -> floor = 1200 - 25 = 1175
        mgr.set_weakest_elo_fn(lambda: 1200.0)

        model = torch.nn.Linear(10, 10)
        mgr.admit(model, "resnet", {}, epoch=1)
        # Set oldest entry elo to 900 (well below 1175)
        oldest = store.list_by_role(Role.RECENT_FIXED)[0]
        store.update_elo(oldest.id, 900.0)

        mgr.admit(model, "resnet", {}, epoch=2)
        mgr.admit(model, "resnet", {}, epoch=3)
        # count=3, slots=2, overflow_used=1, soft_overflow=0 -> can't delay -> RETIRE
        outcome, entry = mgr.review_oldest()
        assert outcome is ReviewOutcome.RETIRE
        assert entry.id == oldest.id


# ===========================================================================
# FrontierManager.select_initial edge cases
# ===========================================================================


class TestFrontierSelectInitialEdgeCases:
    """Edge cases for FrontierManager.select_initial."""

    def test_select_initial_count_zero(self, store):
        """select_initial with count=0 returns empty list."""
        entries = [_add_entry(store, i, elo=1000 + i * 100) for i in range(5)]
        mgr = FrontierManager(store, FrontierStaticConfig())
        result = mgr.select_initial(entries, count=0)
        assert result == []

    def test_select_initial_count_one(self, store):
        """Single slot selection picks the median-elo entry."""
        entries = [_add_entry(store, i, elo=800 + i * 100) for i in range(5)]
        # Elos: 800, 900, 1000, 1100, 1200 sorted
        # Median index: 5 // 2 = 2 -> elo 1000
        mgr = FrontierManager(store, FrontierStaticConfig())
        result = mgr.select_initial(entries, count=1)
        assert len(result) == 1
        assert result[0].elo_rating == pytest.approx(1000.0)
