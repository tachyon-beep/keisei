"""Tests for HistoricalGauntlet -- periodic benchmark runner."""

import threading

import pytest
import torch

from keisei.config import GauntletConfig, RoleEloConfig
from keisei.db import init_db
from keisei.training.historical_gauntlet import HistoricalGauntlet
from keisei.training.historical_library import HistoricalSlot
from keisei.training.opponent_store import EloColumn, OpponentStore, Role
from keisei.training.role_elo import RoleEloTracker


@pytest.fixture
def gauntlet_setup(tmp_path):
    db_path = str(tmp_path / "gauntlet.db")
    init_db(db_path)
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    store = OpponentStore(db_path, str(league_dir))
    role_elo = RoleEloTracker(store, RoleEloConfig())
    config = GauntletConfig(games_per_matchup=16)
    stop_event = threading.Event()
    gauntlet = HistoricalGauntlet(
        store=store,
        role_elo_tracker=role_elo,
        config=config,
        stop_event=stop_event,
        device="cpu",
        num_envs=4,
        max_ply=10,
    )
    return gauntlet, store, stop_event


class TestIsDue:
    def test_false_at_epoch_0(self, gauntlet_setup):
        gauntlet, _, _ = gauntlet_setup
        assert not gauntlet.is_due(0)

    def test_false_at_epoch_99(self, gauntlet_setup):
        gauntlet, _, _ = gauntlet_setup
        assert not gauntlet.is_due(99)

    def test_true_at_epoch_100(self, gauntlet_setup):
        gauntlet, _, _ = gauntlet_setup
        assert gauntlet.is_due(100)

    def test_true_at_epoch_200(self, gauntlet_setup):
        gauntlet, _, _ = gauntlet_setup
        assert gauntlet.is_due(200)

    def test_false_when_disabled(self, tmp_path):
        db_path = str(tmp_path / "disabled.db")
        init_db(db_path)
        store = OpponentStore(db_path, str(tmp_path / "league"))
        role_elo = RoleEloTracker(store, RoleEloConfig())
        config = GauntletConfig(enabled=False)
        gauntlet = HistoricalGauntlet(
            store=store, role_elo_tracker=role_elo, config=config,
        )
        assert not gauntlet.is_due(100)
        store.close()


class TestRunGauntlet:
    def test_skips_empty_slots(self, gauntlet_setup):
        """Gauntlet with all-empty slots should log warning and return."""
        gauntlet, store, _ = gauntlet_setup
        model = torch.nn.Linear(10, 10)
        learner = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)

        empty_slots = [
            HistoricalSlot(i, target_epoch=i * 100, entry_id=None, actual_epoch=None, selection_mode="fallback")
            for i in range(5)
        ]

        # Should not raise
        gauntlet.run_gauntlet(epoch=100, learner_entry=learner, historical_slots=empty_slots)

        # No gauntlet results should be recorded
        with store._lock:
            rows = store._conn.execute("SELECT COUNT(*) FROM gauntlet_results").fetchone()
            assert rows[0] == 0

    def test_stop_event_interrupts(self, gauntlet_setup):
        """Setting stop_event should cause early exit."""
        gauntlet, store, stop_event = gauntlet_setup
        model = torch.nn.Linear(10, 10)
        learner = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        hist = store.add_entry(model, "resnet", {}, epoch=50, role=Role.RECENT_FIXED)
        store.retire_entry(hist.id, "archive")

        slots = [
            HistoricalSlot(0, target_epoch=50, entry_id=hist.id, actual_epoch=50, selection_mode="log_spaced"),
            HistoricalSlot(1, target_epoch=100, entry_id=hist.id, actual_epoch=50, selection_mode="log_spaced"),
        ]

        # Set stop before running
        stop_event.set()
        gauntlet.run_gauntlet(epoch=100, learner_entry=learner, historical_slots=slots)

        # Stop was set before call — zero results expected
        with store._lock:
            rows = store._conn.execute("SELECT COUNT(*) FROM gauntlet_results").fetchone()
            assert rows[0] == 0


class TestRecordResult:
    def test_records_to_db(self, gauntlet_setup):
        """Verify record_gauntlet_result writes to the DB."""
        _, store, _ = gauntlet_setup
        model = torch.nn.Linear(10, 10)
        a = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        b = store.add_entry(model, "resnet", {}, epoch=50, role=Role.RECENT_FIXED)

        store.record_gauntlet_result(
            epoch=100, entry_id=a.id, historical_slot=0,
            historical_entry_id=b.id, wins=10, losses=5, draws=1,
            elo_before=1000.0, elo_after=1008.0,
        )

        with store._lock:
            rows = store._conn.execute("SELECT * FROM gauntlet_results").fetchall()
            assert len(rows) == 1
            r = dict(rows[0])
            assert r["epoch"] == 100
            assert r["wins"] == 10
            assert r["losses"] == 5
            assert r["draws"] == 1
            assert r["elo_before"] == 1000.0
            assert r["elo_after"] == 1008.0


class TestEloColumnUpdate:
    def test_update_role_elo(self, gauntlet_setup):
        """Verify update_role_elo writes the correct column."""
        _, store, _ = gauntlet_setup
        model = torch.nn.Linear(10, 10)
        e = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        assert store._get_entry(e.id).elo_historical == 1000.0

        store.update_role_elo(e.id, EloColumn.HISTORICAL, 1050.0)

        after = store._get_entry(e.id)
        assert after.elo_historical == 1050.0
        # Other columns unchanged
        assert after.elo_frontier == 1000.0
        assert after.elo_dynamic == 1000.0
        assert after.elo_recent == 1000.0
        # Composite unchanged
        assert after.elo_rating == 1000.0

    def test_update_role_elo_rejects_invalid(self, gauntlet_setup):
        _, store, _ = gauntlet_setup
        model = torch.nn.Linear(10, 10)
        e = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)

        with pytest.raises(ValueError, match="Invalid Elo column"):
            store.update_role_elo(e.id, "not_a_column", 1050.0)  # type: ignore[arg-type]
