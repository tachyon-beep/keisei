"""Tests for HistoricalGauntlet -- periodic benchmark runner."""

import logging
import threading
from unittest.mock import patch

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
    yield gauntlet, store, stop_event
    store.close()


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
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))
        try:
            role_elo = RoleEloTracker(store, RoleEloConfig())
            config = GauntletConfig(enabled=False)
            gauntlet = HistoricalGauntlet(
                store=store, role_elo_tracker=role_elo, config=config,
            )
            assert not gauntlet.is_due(100)
        finally:
            store.close()


class TestRunGauntlet:
    def test_skips_empty_slots(self, gauntlet_setup, caplog):
        """Gauntlet with all-empty slots should log warning and return."""
        gauntlet, store, _ = gauntlet_setup
        model = torch.nn.Linear(10, 10)
        learner = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)

        empty_slots = [
            HistoricalSlot(i, target_epoch=i * 100, entry_id=None, actual_epoch=None, selection_mode="fallback")
            for i in range(5)
        ]

        with caplog.at_level(logging.WARNING, logger="keisei.training.historical_gauntlet"):
            gauntlet.run_gauntlet(epoch=100, learner_entry=learner, historical_slots=empty_slots)

        # Verify warning was logged
        assert any("no filled historical slots" in r.message.lower() or "empty" in r.message.lower()
                    for r in caplog.records), "Expected a warning about empty/no filled slots"

        # No gauntlet results should be recorded
        with store._lock:
            rows = store._conn.execute("SELECT COUNT(*) FROM gauntlet_results").fetchone()
            assert rows[0] == 0

    def test_stop_event_interrupts(self, gauntlet_setup):
        """Pre-set stop_event causes early exit before any slot is played."""
        gauntlet, store, stop_event = gauntlet_setup
        model = torch.nn.Linear(10, 10)
        learner = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        hist = store.add_entry(model, "resnet", {}, epoch=50, role=Role.RECENT_FIXED)
        store.retire_entry(hist.id, "archive")

        slots = [
            HistoricalSlot(0, target_epoch=50, entry_id=hist.id, actual_epoch=50, selection_mode="log_spaced"),
            HistoricalSlot(1, target_epoch=100, entry_id=hist.id, actual_epoch=50, selection_mode="log_spaced"),
        ]

        # Set stop before running — the slot loop checks stop_event at each iteration
        stop_event.set()
        gauntlet.run_gauntlet(epoch=100, learner_entry=learner, historical_slots=slots)

        # Stop was set before call — zero gauntlet results expected
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
            assert r["entry_id"] == a.id
            assert r["historical_slot"] == 0
            assert r["historical_entry_id"] == b.id
            assert r["wins"] == 10
            assert r["losses"] == 5
            assert r["draws"] == 1
            assert r["elo_before"] == 1000.0
            assert r["elo_after"] == 1008.0


class TestEloColumnUpdate:
    def test_update_role_elo(self, gauntlet_setup):
        """Verify update_role_elo writes the correct column without touching others."""
        _, store, _ = gauntlet_setup
        model = torch.nn.Linear(10, 10)
        e = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        assert store._get_entry(e.id).elo_historical == 1000.0

        store.update_role_elo(e.id, EloColumn.HISTORICAL, 1050.0)

        after = store._get_entry(e.id)
        # The targeted column was actually written
        assert after.elo_historical == 1050.0
        # Other role columns unchanged (non-vacuous: elo_historical proves the
        # function writes, so 1000.0 here means it didn't write these columns)
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


class TestStaleEloAccumulation:
    """Regression: multi-slot gauntlet must accumulate Elo, not reset each slot."""

    def test_elo_accumulates_across_slots(self, gauntlet_setup):
        """Two sequential wins should accumulate: 1000→X→Y, not 1000→X→X."""
        gauntlet, store, _ = gauntlet_setup
        model = torch.nn.Linear(10, 10)
        learner = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        hist_a = store.add_entry(model, "resnet", {}, epoch=50, role=Role.RECENT_FIXED)
        hist_b = store.add_entry(model, "resnet", {}, epoch=100, role=Role.RECENT_FIXED)
        store.retire_entry(hist_a.id, "archive")
        store.retire_entry(hist_b.id, "archive")

        slots = [
            HistoricalSlot(0, target_epoch=50, entry_id=hist_a.id, actual_epoch=50, selection_mode="log_spaced"),
            HistoricalSlot(1, target_epoch=100, entry_id=hist_b.id, actual_epoch=100, selection_mode="log_spaced"),
        ]

        dummy_model = object()
        dummy_vecenv = object()

        with (
            patch.object(store, "load_opponent", return_value=dummy_model),
            patch("keisei.training.historical_gauntlet.play_match", return_value=(16, 0, 0)),
            patch("keisei.training.historical_gauntlet.release_models"),
        ):
            gauntlet.run_gauntlet(epoch=100, learner_entry=learner, historical_slots=slots, vecenv=dummy_vecenv)

        # Read gauntlet results
        with store._lock:
            rows = store._conn.execute(
                "SELECT elo_before, elo_after FROM gauntlet_results ORDER BY historical_slot"
            ).fetchall()

        assert len(rows) == 2
        slot0_before, slot0_after = rows[0]
        slot1_before, slot1_after = rows[1]

        # Slot 0: starts at 1000, wins → elo increases
        assert slot0_before == 1000.0
        assert slot0_after > 1000.0

        # Slot 1: must start where slot 0 left off (accumulated), not reset to 1000
        assert slot1_before == slot0_after, (
            f"Stale learner bug: slot1 started at {slot1_before} instead of {slot0_after}"
        )
        assert slot1_after > slot1_before


class TestRunGauntletFailurePaths:
    """Tests for various failure/edge-case paths in run_gauntlet."""

    def test_run_gauntlet_learner_load_failure(self, gauntlet_setup, caplog):
        """If loading the learner model fails, gauntlet aborts gracefully."""
        gauntlet, store, _ = gauntlet_setup
        model = torch.nn.Linear(10, 10)
        learner = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        hist = store.add_entry(model, "resnet", {}, epoch=50, role=Role.RECENT_FIXED)

        slots = [
            HistoricalSlot(0, target_epoch=50, entry_id=hist.id, actual_epoch=50, selection_mode="log_spaced"),
        ]

        dummy_vecenv = object()

        with (
            patch.object(store, "load_opponent", side_effect=RuntimeError("corrupt checkpoint")),
            patch("keisei.training.historical_gauntlet.release_models") as mock_release,
            caplog.at_level(logging.ERROR, logger="keisei.training.historical_gauntlet"),
        ):
            gauntlet.run_gauntlet(epoch=100, learner_entry=learner, historical_slots=slots, vecenv=dummy_vecenv)

        # No gauntlet results recorded
        with store._lock:
            rows = store._conn.execute("SELECT COUNT(*) FROM gauntlet_results").fetchone()
            assert rows[0] == 0

        # release_models should NOT have been called (learner never loaded)
        mock_release.assert_not_called()

    def test_run_gauntlet_hist_model_load_failure(self, gauntlet_setup):
        """If one historical model fails to load, remaining slots still proceed."""
        gauntlet, store, _ = gauntlet_setup
        model = torch.nn.Linear(10, 10)
        learner = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        hist_a = store.add_entry(model, "resnet", {}, epoch=50, role=Role.RECENT_FIXED)
        hist_b = store.add_entry(model, "resnet", {}, epoch=100, role=Role.RECENT_FIXED)

        slots = [
            HistoricalSlot(0, target_epoch=50, entry_id=hist_a.id, actual_epoch=50, selection_mode="log_spaced"),
            HistoricalSlot(1, target_epoch=100, entry_id=hist_b.id, actual_epoch=100, selection_mode="log_spaced"),
        ]

        dummy_model = object()
        dummy_vecenv = object()
        call_count = 0

        def selective_load(entry, device="cpu"):
            nonlocal call_count
            call_count += 1
            # First call is learner (succeeds), second is hist_a (fails), third is hist_b (succeeds)
            if call_count == 1:
                return dummy_model  # learner
            elif call_count == 2:
                raise RuntimeError("corrupt slot 0 model")  # hist_a
            else:
                return dummy_model  # hist_b

        with (
            patch.object(store, "load_opponent", side_effect=selective_load),
            patch("keisei.training.historical_gauntlet.play_match", return_value=(8, 4, 4)),
            patch("keisei.training.historical_gauntlet.release_models"),
        ):
            gauntlet.run_gauntlet(epoch=100, learner_entry=learner, historical_slots=slots, vecenv=dummy_vecenv)

        # Only slot 1 should have a result (slot 0 failed to load)
        with store._lock:
            rows = store._conn.execute("SELECT * FROM gauntlet_results").fetchall()
            assert len(rows) == 1
            assert dict(rows[0])["historical_slot"] == 1

    def test_run_gauntlet_zero_games_skip(self, gauntlet_setup):
        """If play_match returns (0,0,0), no gauntlet result is recorded for that slot."""
        gauntlet, store, _ = gauntlet_setup
        model = torch.nn.Linear(10, 10)
        learner = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        hist = store.add_entry(model, "resnet", {}, epoch=50, role=Role.RECENT_FIXED)

        slots = [
            HistoricalSlot(0, target_epoch=50, entry_id=hist.id, actual_epoch=50, selection_mode="log_spaced"),
        ]

        dummy_model = object()
        dummy_vecenv = object()

        with (
            patch.object(store, "load_opponent", return_value=dummy_model),
            patch("keisei.training.historical_gauntlet.play_match", return_value=(0, 0, 0)),
            patch("keisei.training.historical_gauntlet.release_models"),
        ):
            gauntlet.run_gauntlet(epoch=100, learner_entry=learner, historical_slots=slots, vecenv=dummy_vecenv)

        # Zero games → no result recorded
        with store._lock:
            rows = store._conn.execute("SELECT COUNT(*) FROM gauntlet_results").fetchone()
            assert rows[0] == 0

    def test_run_gauntlet_hist_entry_not_found(self, gauntlet_setup):
        """If store.get_entry returns None for a slot's entry_id, that slot is skipped."""
        gauntlet, store, _ = gauntlet_setup
        model = torch.nn.Linear(10, 10)
        learner = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        hist_real = store.add_entry(model, "resnet", {}, epoch=100, role=Role.RECENT_FIXED)

        # Slot 0 refers to a non-existent entry_id (9999)
        # Slot 1 refers to a real entry
        slots = [
            HistoricalSlot(0, target_epoch=50, entry_id=9999, actual_epoch=50, selection_mode="log_spaced"),
            HistoricalSlot(1, target_epoch=100, entry_id=hist_real.id, actual_epoch=100, selection_mode="log_spaced"),
        ]

        dummy_model = object()
        dummy_vecenv = object()

        with (
            patch.object(store, "load_opponent", return_value=dummy_model),
            patch("keisei.training.historical_gauntlet.play_match", return_value=(10, 2, 4)),
            patch("keisei.training.historical_gauntlet.release_models"),
        ):
            gauntlet.run_gauntlet(epoch=100, learner_entry=learner, historical_slots=slots, vecenv=dummy_vecenv)

        # Only slot 1 should have a result (slot 0's entry_id doesn't exist)
        with store._lock:
            rows = store._conn.execute("SELECT * FROM gauntlet_results").fetchall()
            assert len(rows) == 1
            assert dict(rows[0])["historical_slot"] == 1
