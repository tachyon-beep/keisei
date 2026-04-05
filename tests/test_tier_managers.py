"""Tests for tier managers: Frontier, RecentFixed, Dynamic."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from keisei.config import FrontierStaticConfig, RecentFixedConfig, DynamicConfig
from keisei.db import init_db
from keisei.training.frontier_promoter import FrontierPromoter
from keisei.training.opponent_store import OpponentEntry, OpponentStore, Role, EntryStatus
from keisei.training.tier_managers import (
    FrontierManager,
    RecentFixedManager,
    DynamicManager,
    ReviewOutcome,
)


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "tier.db")
    init_db(db_path)
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    return OpponentStore(db_path, str(league_dir))


def _add_entry(store, epoch, role=Role.UNASSIGNED, elo=1000.0):
    model = torch.nn.Linear(10, 10)
    entry = store.add_entry(model, "resnet", {}, epoch=epoch, role=role)
    if elo != 1000.0:
        store.update_elo(entry.id, elo)
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

    def test_review_is_noop_phase1(self, store):
        mgr = FrontierManager(store, FrontierStaticConfig())
        mgr.review(epoch=500)  # should not raise

    def test_is_due_for_review(self, store):
        mgr = FrontierManager(store, FrontierStaticConfig(review_interval_epochs=250))
        assert mgr.is_due_for_review(250)
        assert mgr.is_due_for_review(500)
        assert not mgr.is_due_for_review(100)
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
        store.record_result(epoch=1, learner_id=oldest.id, opponent_id=other1.id,
                            wins=1, losses=0, draws=0)
        store.record_result(epoch=2, learner_id=oldest.id, opponent_id=other2.id,
                            wins=1, losses=0, draws=0)
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

    def test_review_oldest_delay_when_undercalibrated(self, store):
        """Entry with < min_games should DELAY if soft overflow remains."""
        mgr = RecentFixedManager(store, RecentFixedConfig(
            slots=2, soft_overflow=1, min_games_for_review=32,
        ))
        model = torch.nn.Linear(10, 10)
        mgr.admit(model, "resnet", {}, epoch=1)
        mgr.admit(model, "resnet", {}, epoch=2)
        mgr.admit(model, "resnet", {}, epoch=3)
        assert mgr.count() == 3
        mgr.admit(model, "resnet", {}, epoch=4)
        outcome, entry = mgr.review_oldest()
        assert outcome is ReviewOutcome.RETIRE


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
        """DynamicManager now accepts training_enabled=True (Phase 3)."""
        mgr = DynamicManager(store, DynamicConfig(training_enabled=True))
        assert mgr._config.training_enabled is True
