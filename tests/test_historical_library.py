"""Tests for HistoricalLibrary -- log-spaced milestone manager."""

import pytest
import torch

from keisei.config import HistoricalLibraryConfig, LeagueConfig
from keisei.db import init_db
from keisei.training.historical_library import HistoricalLibrary
from keisei.training.opponent_store import EntryStatus, OpponentStore, Role

pytestmark = pytest.mark.integration


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
        # At E=10 the log-spacing produces [1, 2, 3, 6, 10] — all distinct
        assert len(set(targets)) == 5

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
        # At E=2 the rounding collapses to [1, 1, 1, 2, 2]
        assert targets == [1, 1, 1, 2, 2]

    def test_targets_at_epoch_1_clamped(self):
        """epoch=1 is clamped to 2 by max(E, 2), producing the same [1,1,1,2,2]."""
        targets = HistoricalLibrary._compute_targets(1)
        assert targets == [1, 1, 1, 2, 2]
        # Verify the clamping equivalence explicitly
        assert targets == HistoricalLibrary._compute_targets(2)

    def test_targets_with_single_slot(self):
        """num_slots=1 should return [current_epoch] without ZeroDivisionError."""
        targets = HistoricalLibrary._compute_targets(500, num_slots=1)
        assert targets == [500]

    def test_targets_with_single_slot_epoch_1(self):
        """num_slots=1 at epoch 1 returns [1] (the current epoch)."""
        targets = HistoricalLibrary._compute_targets(1, num_slots=1)
        assert targets == [1]


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
        # With >= 5 candidates, filled slots should use log_spaced mode
        for s in filled:
            assert s.selection_mode == "log_spaced"

    def test_refresh_idempotency(self, library_setup):
        library, store = library_setup
        model = torch.nn.Linear(10, 10)
        for epoch in [1, 10, 100, 500, 1000]:
            e = store.add_entry(model, "resnet", {}, epoch=epoch, role=Role.RECENT_FIXED)
            store.retire_entry(e.id, "archive")

        library.refresh(1000)
        slots_1 = [
            (s.slot_index, s.entry_id, s.actual_epoch, s.selection_mode)
            for s in library.get_slots()
        ]

        library.refresh(1000)
        slots_2 = [
            (s.slot_index, s.entry_id, s.actual_epoch, s.selection_mode)
            for s in library.get_slots()
        ]

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
        # 2 candidates, targets at E=100 are [1,3,10,32,100].
        # Epoch 5 snaps to target 3 (dist=2, threshold=1.0 → rejected) or target 10 (dist=5).
        # Epoch 10 snaps to target 10 (dist=0).
        # With proximity threshold, at least epoch 10→target 10 always fills.
        assert len(filled) >= 1
        assert len(filled) <= 2
        for s in slots:
            if s.entry_id is not None:
                # Fewer candidates than slots → fallback mode
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
        # Targets at E=100: [1, 3, 10, 32, 100]. Only target=100 is within
        # proximity threshold of epoch-100 candidates (dist=0).
        # _snap_to_nearest prefers retired entries on distance ties.
        filled = [s for s in slots if s.entry_id is not None]
        assert len(filled) >= 1, "At least one slot should be filled"
        slot_100 = [s for s in filled if s.target_epoch == 100]
        assert len(slot_100) == 1, "Exactly one slot targets epoch 100"
        assert slot_100[0].entry_id == retired.id


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
        # Verify the filled slots are the two endpoints
        filled_epochs = {s.actual_epoch for s in filled}
        assert filled_epochs == {1, 10000}
        filled_ids = {s.entry_id for s in filled}
        assert filled_ids == {e1.id, e2.id}


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
