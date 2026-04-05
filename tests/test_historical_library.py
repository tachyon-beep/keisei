"""Tests for HistoricalLibrary -- log-spaced milestone manager."""

import pytest
import torch

from keisei.config import HistoricalLibraryConfig, LeagueConfig
from keisei.db import init_db
from keisei.training.historical_library import HistoricalLibrary
from keisei.training.opponent_store import EntryStatus, OpponentStore, Role


@pytest.fixture
def library_setup(tmp_path):
    db_path = str(tmp_path / "lib.db")
    init_db(db_path)
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    store = OpponentStore(db_path, str(league_dir))
    config = HistoricalLibraryConfig()
    library = HistoricalLibrary(store, config)
    yield library, store
    store.close()


class TestComputeTargets:
    def test_targets_at_epoch_10(self):
        targets = HistoricalLibrary._compute_targets(10)
        assert len(targets) == 5
        assert targets[0] == 1
        assert targets[-1] == 10
        # All targets should be distinct at E=10
        assert len(set(targets)) >= 4

    def test_targets_at_epoch_1000(self):
        targets = HistoricalLibrary._compute_targets(1000)
        assert targets[0] == 1
        assert targets[-1] == 1000
        assert len(set(targets)) == 5

    def test_targets_at_epoch_100000(self):
        targets = HistoricalLibrary._compute_targets(100000)
        assert targets[0] == 1
        assert targets[-1] == 100000
        assert len(set(targets)) == 5
        # Should be roughly log-spaced
        assert targets[1] < targets[2] < targets[3]

    def test_targets_at_epoch_2(self):
        targets = HistoricalLibrary._compute_targets(2)
        assert len(targets) == 5
        assert targets[0] == 1
        assert targets[-1] == 2

    def test_targets_at_epoch_1_clamped(self):
        """epoch=1 is clamped to 2 by max(E, 2)."""
        targets = HistoricalLibrary._compute_targets(1)
        assert targets == HistoricalLibrary._compute_targets(2)


class TestIsDueForRefresh:
    def test_false_before_min_epoch(self, library_setup):
        library, _ = library_setup
        assert not library.is_due_for_refresh(5)

    def test_false_at_epoch_99(self, library_setup):
        library, _ = library_setup
        assert not library.is_due_for_refresh(99)

    def test_true_at_epoch_100(self, library_setup):
        library, _ = library_setup
        assert library.is_due_for_refresh(100)

    def test_true_at_epoch_200(self, library_setup):
        library, _ = library_setup
        assert library.is_due_for_refresh(200)


class TestRefresh:
    def test_empty_candidates_fills_fallback_slots(self, library_setup):
        library, store = library_setup
        library.refresh(100)
        slots = library.get_slots()
        assert len(slots) == 5
        for s in slots:
            assert s.entry_id is None
            assert s.selection_mode == "fallback"

    def test_snaps_to_nearest_checkpoint(self, library_setup):
        library, store = library_setup
        model = torch.nn.Linear(10, 10)
        # Create entries at various epochs, densely covering the log-space range
        for epoch in [1, 3, 6, 10, 20, 50, 100, 200, 500, 1000]:
            e = store.add_entry(model, "resnet", {}, epoch=epoch, role=Role.RECENT_FIXED)
            store.retire_entry(e.id, "test archive")

        library.refresh(1000)
        slots = library.get_slots()
        filled = [s for s in slots if s.entry_id is not None]
        assert len(filled) == 5  # Dense coverage should fill all slots

    def test_refresh_idempotency(self, library_setup):
        library, store = library_setup
        model = torch.nn.Linear(10, 10)
        for epoch in [1, 10, 100, 500, 1000]:
            e = store.add_entry(model, "resnet", {}, epoch=epoch, role=Role.RECENT_FIXED)
            store.retire_entry(e.id, "archive")

        library.refresh(1000)
        slots_1 = [(s.slot_index, s.entry_id) for s in library.get_slots()]

        library.refresh(1000)
        slots_2 = [(s.slot_index, s.entry_id) for s in library.get_slots()]

        assert slots_1 == slots_2

    def test_early_training_fallback(self, library_setup):
        library, store = library_setup
        model = torch.nn.Linear(10, 10)
        # Only 2 entries — fewer than 5 slots
        e1 = store.add_entry(model, "resnet", {}, epoch=5, role=Role.RECENT_FIXED)
        store.retire_entry(e1.id, "archive")
        e2 = store.add_entry(model, "resnet", {}, epoch=10, role=Role.RECENT_FIXED)
        store.retire_entry(e2.id, "archive")

        library.refresh(100)
        slots = library.get_slots()
        filled = [s for s in slots if s.entry_id is not None]
        assert len(filled) >= 1
        for s in slots:
            if s.entry_id is not None:
                assert s.selection_mode == "fallback"

    def test_prefers_retired_over_active(self, library_setup):
        library, store = library_setup
        model = torch.nn.Linear(10, 10)
        # Create an active entry and a retired entry at the same epoch
        active = store.add_entry(model, "resnet", {}, epoch=100, role=Role.DYNAMIC)
        retired = store.add_entry(model, "resnet", {}, epoch=100, role=Role.RECENT_FIXED)
        store.retire_entry(retired.id, "archive")

        library.refresh(100)
        slots = library.get_slots()
        # The slot closest to epoch 100 should prefer the retired entry
        filled = [s for s in slots if s.entry_id is not None]
        assert len(filled) >= 1, "At least one slot should be filled"
        closest_to_100 = min(filled, key=lambda s: abs((s.actual_epoch or 0) - 100))
        assert closest_to_100.entry_id == retired.id


class TestProximityThreshold:
    def test_distant_checkpoint_leaves_slot_empty(self, library_setup):
        """A candidate too far from the target epoch leaves the slot empty."""
        library, store = library_setup
        model = torch.nn.Linear(10, 10)
        # Only create entries at epoch 1 and 10000 — nothing near the middle targets
        e1 = store.add_entry(model, "resnet", {}, epoch=1, role=Role.RECENT_FIXED)
        store.retire_entry(e1.id, "archive")
        e2 = store.add_entry(model, "resnet", {}, epoch=10000, role=Role.RECENT_FIXED)
        store.retire_entry(e2.id, "archive")

        library.refresh(10000)
        slots = library.get_slots()
        # Only 2 checkpoints (epoch 1 and 10000) — endpoints filled, middle slots empty
        filled = [s for s in slots if s.entry_id is not None]
        empty = [s for s in slots if s.entry_id is None]
        assert len(filled) == 2  # exactly the two endpoints
        assert len(empty) == 3  # middle slots rejected by proximity threshold


class TestGetSlots:
    def test_returns_empty_before_refresh(self, library_setup):
        library, _ = library_setup
        slots = library.get_slots()
        assert slots == []

    def test_returns_five_after_refresh(self, library_setup):
        library, _ = library_setup
        library.refresh(100)
        slots = library.get_slots()
        assert len(slots) == 5
