"""Tests for RoleEloTracker -- per-context Elo updates."""

import pytest
import torch

from keisei.config import RoleEloConfig
from keisei.db import init_db
from keisei.training.opponent_store import EloColumn, OpponentStore, Role
from keisei.training.role_elo import RoleEloTracker


@pytest.fixture
def elo_setup(tmp_path):
    db_path = str(tmp_path / "elo.db")
    init_db(db_path)
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    store = OpponentStore(db_path, str(league_dir))
    config = RoleEloConfig()
    tracker = RoleEloTracker(store, config)
    model = torch.nn.Linear(10, 10)
    yield tracker, store, model
    store.close()


class TestUpdateFromResult:
    def test_frontier_match_updates_elo_frontier(self, elo_setup):
        tracker, store, model = elo_setup
        a = store.add_entry(model, "resnet", {}, epoch=1, role=Role.FRONTIER_STATIC)
        b = store.add_entry(model, "resnet", {}, epoch=2, role=Role.DYNAMIC)

        tracker.update_from_result(a, b, result_score=1.0, match_context="frontier")

        a_after = store._get_entry(a.id)
        b_after = store._get_entry(b.id)
        assert a_after.elo_frontier > 1000.0
        assert b_after.elo_frontier < 1000.0
        # Composite elo_rating should NOT be affected
        assert a_after.elo_rating == 1000.0
        assert b_after.elo_rating == 1000.0

    def test_dynamic_match_updates_elo_dynamic(self, elo_setup):
        tracker, store, model = elo_setup
        a = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        b = store.add_entry(model, "resnet", {}, epoch=2, role=Role.DYNAMIC)

        tracker.update_from_result(a, b, result_score=0.75, match_context="dynamic")

        a_after = store._get_entry(a.id)
        b_after = store._get_entry(b.id)
        assert a_after.elo_dynamic > 1000.0
        assert b_after.elo_dynamic < 1000.0

    def test_recent_match_updates_elo_recent(self, elo_setup):
        tracker, store, model = elo_setup
        a = store.add_entry(model, "resnet", {}, epoch=1, role=Role.RECENT_FIXED)
        b = store.add_entry(model, "resnet", {}, epoch=2, role=Role.RECENT_FIXED)

        tracker.update_from_result(a, b, result_score=1.0, match_context="recent")

        a_after = store._get_entry(a.id)
        assert a_after.elo_recent > 1000.0

    def test_historical_match_updates_learner_only(self, elo_setup):
        """Historical context: learner (a) gets updated, anchor (b) stays frozen."""
        tracker, store, model = elo_setup
        a = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        b = store.add_entry(model, "resnet", {}, epoch=2, role=Role.DYNAMIC)

        tracker.update_from_result(a, b, result_score=1.0, match_context="historical")

        a_after = store._get_entry(a.id)
        b_after = store._get_entry(b.id)
        assert a_after.elo_historical > 1000.0
        # result_score=1.0 means a wins; if b were updated it would DROP below 1000.
        # Staying at exactly 1000.0 proves the code skipped b's update.
        assert b_after.elo_historical == 1000.0  # anchor stays frozen
        # Other columns should be untouched for both
        assert a_after.elo_dynamic == 1000.0
        assert b_after.elo_dynamic == 1000.0

    def test_cross_dynamic_recent(self, elo_setup):
        tracker, store, model = elo_setup
        dyn = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        rec = store.add_entry(model, "resnet", {}, epoch=2, role=Role.RECENT_FIXED)

        tracker.update_from_result(dyn, rec, result_score=1.0, match_context="cross_dynamic_recent")

        dyn_after = store._get_entry(dyn.id)
        rec_after = store._get_entry(rec.id)
        # Dynamic entry's elo_dynamic should go up
        assert dyn_after.elo_dynamic > 1000.0
        # Recent entry's elo_recent should go down
        assert rec_after.elo_recent < 1000.0
        # Non-updated columns must stay at default
        assert dyn_after.elo_recent == 1000.0
        assert rec_after.elo_dynamic == 1000.0

    def test_k_factors_differ_by_context(self, elo_setup):
        tracker, store, model = elo_setup
        # Two matches with same result_score but different contexts
        a1 = store.add_entry(model, "resnet", {}, epoch=1, role=Role.FRONTIER_STATIC)
        b1 = store.add_entry(model, "resnet", {}, epoch=2, role=Role.DYNAMIC)
        tracker.update_from_result(a1, b1, result_score=1.0, match_context="frontier")
        a1_after = store._get_entry(a1.id)

        a2 = store.add_entry(model, "resnet", {}, epoch=3, role=Role.RECENT_FIXED)
        b2 = store.add_entry(model, "resnet", {}, epoch=4, role=Role.RECENT_FIXED)
        tracker.update_from_result(a2, b2, result_score=1.0, match_context="recent")
        a2_after = store._get_entry(a2.id)

        # frontier_k=16, recent_k=32, so recent delta should be larger
        frontier_delta = a1_after.elo_frontier - 1000.0
        recent_delta = a2_after.elo_recent - 1000.0
        assert recent_delta > frontier_delta

    def test_composite_elo_not_modified(self, elo_setup):
        tracker, store, model = elo_setup
        a = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        b = store.add_entry(model, "resnet", {}, epoch=2, role=Role.DYNAMIC)

        tracker.update_from_result(a, b, result_score=1.0, match_context="dynamic")

        a_after = store._get_entry(a.id)
        b_after = store._get_entry(b.id)
        # Prove the update actually happened (elo_dynamic changed)...
        assert a_after.elo_dynamic > 1000.0
        assert b_after.elo_dynamic < 1000.0
        # ...then verify composite elo_rating was NOT touched
        assert a_after.elo_rating == 1000.0
        assert b_after.elo_rating == 1000.0

    def test_atomic_update_both_entries(self, elo_setup):
        """Both entries are updated in a single transaction — winner up, loser down."""
        tracker, store, model = elo_setup
        a = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        b = store.add_entry(model, "resnet", {}, epoch=2, role=Role.DYNAMIC)

        tracker.update_from_result(a, b, result_score=1.0, match_context="dynamic")

        a_after = store._get_entry(a.id)
        b_after = store._get_entry(b.id)
        # Both entries must have been updated: a goes up, b goes down
        assert a_after.elo_dynamic > 1000.0
        assert b_after.elo_dynamic < 1000.0
        # Elo is zero-sum: combined delta should be zero
        delta_a = a_after.elo_dynamic - 1000.0
        delta_b = b_after.elo_dynamic - 1000.0
        assert abs(delta_a + delta_b) < 0.01

    def test_unknown_context_raises(self, elo_setup):
        tracker, store, model = elo_setup
        a = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        b = store.add_entry(model, "resnet", {}, epoch=2, role=Role.DYNAMIC)

        with pytest.raises(ValueError, match="Unknown match context"):
            tracker.update_from_result(a, b, result_score=1.0, match_context="invalid")


class TestDetermineMatchContext:
    def test_frontier_present(self):
        # We can't easily create OpponentEntry without a DB, so test at the static method level
        from keisei.training.opponent_store import OpponentEntry

        a = OpponentEntry(
            id=1, display_name="A", architecture="resnet", model_params={},
            checkpoint_path="", elo_rating=1000, created_epoch=1,
            games_played=0, created_at="", flavour_facts=[],
            role=Role.FRONTIER_STATIC,
        )
        b = OpponentEntry(
            id=2, display_name="B", architecture="resnet", model_params={},
            checkpoint_path="", elo_rating=1000, created_epoch=2,
            games_played=0, created_at="", flavour_facts=[],
            role=Role.DYNAMIC,
        )
        assert RoleEloTracker.determine_match_context(a, b) == "frontier"

    def test_dynamic_vs_dynamic(self):
        from keisei.training.opponent_store import OpponentEntry

        a = OpponentEntry(
            id=1, display_name="A", architecture="resnet", model_params={},
            checkpoint_path="", elo_rating=1000, created_epoch=1,
            games_played=0, created_at="", flavour_facts=[],
            role=Role.DYNAMIC,
        )
        b = OpponentEntry(
            id=2, display_name="B", architecture="resnet", model_params={},
            checkpoint_path="", elo_rating=1000, created_epoch=2,
            games_played=0, created_at="", flavour_facts=[],
            role=Role.DYNAMIC,
        )
        assert RoleEloTracker.determine_match_context(a, b) == "dynamic"

    def test_recent_vs_recent(self):
        from keisei.training.opponent_store import OpponentEntry

        a = OpponentEntry(
            id=1, display_name="A", architecture="resnet", model_params={},
            checkpoint_path="", elo_rating=1000, created_epoch=1,
            games_played=0, created_at="", flavour_facts=[],
            role=Role.RECENT_FIXED,
        )
        b = OpponentEntry(
            id=2, display_name="B", architecture="resnet", model_params={},
            checkpoint_path="", elo_rating=1000, created_epoch=2,
            games_played=0, created_at="", flavour_facts=[],
            role=Role.RECENT_FIXED,
        )
        assert RoleEloTracker.determine_match_context(a, b) == "recent"

    def test_dynamic_vs_recent(self):
        from keisei.training.opponent_store import OpponentEntry

        a = OpponentEntry(
            id=1, display_name="A", architecture="resnet", model_params={},
            checkpoint_path="", elo_rating=1000, created_epoch=1,
            games_played=0, created_at="", flavour_facts=[],
            role=Role.DYNAMIC,
        )
        b = OpponentEntry(
            id=2, display_name="B", architecture="resnet", model_params={},
            checkpoint_path="", elo_rating=1000, created_epoch=2,
            games_played=0, created_at="", flavour_facts=[],
            role=Role.RECENT_FIXED,
        )
        assert RoleEloTracker.determine_match_context(a, b) == "cross_dynamic_recent"


class TestDetermineMatchContextFallback:
    def test_unassigned_vs_unassigned_falls_back_to_dynamic(self):
        from keisei.training.opponent_store import OpponentEntry

        a = OpponentEntry(
            id=1, display_name="A", architecture="resnet", model_params={},
            checkpoint_path="", elo_rating=1000, created_epoch=1,
            games_played=0, created_at="", flavour_facts=[],
            role=Role.UNASSIGNED,
        )
        b = OpponentEntry(
            id=2, display_name="B", architecture="resnet", model_params={},
            checkpoint_path="", elo_rating=1000, created_epoch=2,
            games_played=0, created_at="", flavour_facts=[],
            role=Role.UNASSIGNED,
        )
        assert RoleEloTracker.determine_match_context(a, b) == "dynamic"


class TestGetRoleElos:
    def test_returns_all_four(self, elo_setup):
        tracker, store, model = elo_setup
        e = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        elos = tracker.get_role_elos(e.id)
        assert EloColumn.FRONTIER in elos
        assert EloColumn.DYNAMIC in elos
        assert EloColumn.RECENT in elos
        assert EloColumn.HISTORICAL in elos

    def test_nonexistent_entry_returns_empty(self, elo_setup):
        tracker, _, _ = elo_setup
        assert tracker.get_role_elos(9999) == {}
