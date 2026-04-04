"""Tests for the opponent league: pool, sampler, Elo."""

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from keisei.db import init_db
from keisei.training.league import (
    OpponentEntry,
    OpponentPool,
    OpponentSampler,
    compute_elo_update,
)


@pytest.fixture
def league_db(tmp_path):
    db_path = str(tmp_path / "league.db")
    init_db(db_path)
    return db_path


@pytest.fixture
def league_dir(tmp_path):
    d = tmp_path / "checkpoints" / "league"
    d.mkdir(parents=True)
    return d


class TestOpponentEntry:
    def test_from_db_row(self, tmp_path):
        """Verify from_db_row correctly deserializes a sqlite3.Row."""
        import sqlite3

        db_path = str(tmp_path / "test_entry.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE t (id INT, display_name TEXT, flavour_facts TEXT, "
            "architecture TEXT, model_params TEXT, "
            "checkpoint_path TEXT, elo_rating REAL, created_epoch INT, "
            "games_played INT, created_at TEXT)"
        )
        conn.execute(
            """INSERT INTO t VALUES (1, 'Takeshi', '[["Favourite piece","Rook"]]', """
            """'resnet', '{"hidden_size": 16}', """
            """'/path/to/ckpt.pt', 1000.0, 10, 5, '2026-04-01T00:00:00Z')"""
        )
        row = conn.execute("SELECT * FROM t").fetchone()
        conn.close()

        entry = OpponentEntry.from_db_row(row)
        assert entry.id == 1
        assert entry.display_name == "Takeshi"
        assert entry.flavour_facts == [["Favourite piece", "Rook"]]
        assert entry.architecture == "resnet"
        assert entry.model_params == {"hidden_size": 16}
        assert entry.elo_rating == 1000.0
        assert entry.created_epoch == 10
        assert entry.games_played == 5


class TestOpponentPool:
    def test_add_snapshot(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=5)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {"hidden_size": 16}, epoch=10)
        entries = pool.list_entries()
        assert len(entries) == 1
        assert entries[0].architecture == "resnet"
        assert entries[0].created_epoch == 10
        assert Path(entries[0].checkpoint_path).exists()

    def test_eviction_respects_max_pool_size(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=3)
        model = torch.nn.Linear(10, 10)
        for epoch in range(5):
            pool.add_snapshot(model, "resnet", {"hidden_size": 16}, epoch=epoch)
        entries = pool.list_entries()
        assert len(entries) == 3
        epochs = [e.created_epoch for e in entries]
        assert 0 not in epochs
        assert 1 not in epochs

    def test_eviction_skips_pinned_entries(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=2)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)
        entry_0 = pool.list_entries()[0]

        pool.pin(entry_0.id)
        pool.add_snapshot(model, "resnet", {}, epoch=1)
        pool.add_snapshot(model, "resnet", {}, epoch=2)

        entries = pool.list_entries()
        epochs = [e.created_epoch for e in entries]
        assert 0 in epochs  # pinned, survived
        assert 2 in epochs  # most recent

        pool.unpin(entry_0.id)

    def test_load_opponent(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=5)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {"hidden_size": 16}, epoch=5)
        entry = pool.list_entries()[0]

        with patch("keisei.training.league.build_model") as mock_build:
            mock_model = torch.nn.Linear(10, 10)
            mock_build.return_value = mock_model
            loaded = pool.load_opponent(entry)
            mock_build.assert_called_once_with("resnet", {"hidden_size": 16})
            assert not loaded.training  # should be in eval mode

    def test_empty_pool_list(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=5)
        assert pool.list_entries() == []

    def test_update_elo(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=5)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)
        entry = pool.list_entries()[0]
        pool.update_elo(entry.id, 1050.0)
        updated = pool.list_entries()[0]
        assert updated.elo_rating == 1050.0

    def test_record_result(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=5)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)
        pool.add_snapshot(model, "resnet", {}, epoch=1)
        entries = pool.list_entries()
        learner_id, opponent_id = entries[0].id, entries[1].id
        pool.record_result(
            epoch=1, learner_id=learner_id, opponent_id=opponent_id,
            wins=3, losses=1, draws=1,
        )
        # Both learner and opponent should have games_played updated
        updated = {e.id: e for e in pool.list_entries()}
        assert updated[learner_id].games_played == 5
        assert updated[opponent_id].games_played == 5

    def test_update_elo_writes_history(self, league_db, league_dir):
        import sqlite3
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=5)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)
        entry = pool.list_entries()[0]

        pool.update_elo(entry.id, 1050.0, epoch=3)
        pool.update_elo(entry.id, 1100.0, epoch=5)

        conn = sqlite3.connect(league_db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT entry_id, epoch, elo_rating FROM elo_history ORDER BY epoch"
        ).fetchall()
        conn.close()
        assert len(rows) == 2
        assert dict(rows[0]) == {"entry_id": entry.id, "epoch": 3, "elo_rating": 1050.0}
        assert dict(rows[1]) == {"entry_id": entry.id, "epoch": 5, "elo_rating": 1100.0}

    def test_delete_entry_cascades_elo_history(self, league_db, league_dir):
        import sqlite3
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=1)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)
        entry = pool.list_entries()[0]
        pool.update_elo(entry.id, 1050.0, epoch=1)

        # Adding a second snapshot triggers eviction of the first
        pool.add_snapshot(model, "resnet", {}, epoch=1)

        conn = sqlite3.connect(league_db)
        count = conn.execute("SELECT COUNT(*) FROM elo_history WHERE entry_id = ?", (entry.id,)).fetchone()[0]
        conn.close()
        assert count == 0

    def test_load_opponent_missing_checkpoint_raises(self, league_db, league_dir):
        """load_opponent() should raise FileNotFoundError for deleted checkpoints."""
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        import dataclasses
        params = SEResNetParams(num_blocks=2, channels=32, se_reduction=8,
                                global_pool_channels=16, policy_channels=8,
                                value_fc_size=32, score_fc_size=16, obs_channels=50)
        model = SEResNetModel(params)
        pool = OpponentPool(league_db, str(league_dir))
        entry = pool.add_snapshot(model, "se_resnet", dataclasses.asdict(params), epoch=0)

        # Delete the checkpoint file
        import os
        os.remove(entry.checkpoint_path)

        with pytest.raises(FileNotFoundError, match="Checkpoint missing"):
            pool.load_opponent(entry)


class TestEloCalculation:
    def _assert_conservation(self, ra, rb, new_a, new_b):
        """Elo total must be conserved."""
        assert abs((new_a + new_b) - (ra + rb)) < 1e-9

    def test_equal_elo_expected_is_half(self):
        new_a, new_b = compute_elo_update(1000.0, 1000.0, result=1.0, k=32)
        assert abs(new_a - 1016.0) < 0.1
        assert abs(new_b - 984.0) < 0.1
        self._assert_conservation(1000.0, 1000.0, new_a, new_b)

    def test_draw_against_equal(self):
        new_a, new_b = compute_elo_update(1000.0, 1000.0, result=0.5, k=32)
        assert abs(new_a - 1000.0) < 0.1
        assert abs(new_b - 1000.0) < 0.1
        self._assert_conservation(1000.0, 1000.0, new_a, new_b)

    def test_upset_gives_more_elo(self):
        new_a, new_b = compute_elo_update(800.0, 1200.0, result=1.0, k=32)
        assert new_a > 825
        assert new_b < 1175
        self._assert_conservation(800.0, 1200.0, new_a, new_b)

    def test_symmetry(self):
        new_a, new_b = compute_elo_update(1000.0, 1200.0, result=1.0, k=32)
        new_c, new_d = compute_elo_update(1200.0, 1000.0, result=0.0, k=32)
        assert abs(new_a - new_d) < 0.01
        assert abs(new_b - new_c) < 0.01
        self._assert_conservation(1000.0, 1200.0, new_a, new_b)

    def test_equal_ratings_win(self):
        """Player A wins against equal opponent — gains exactly K/2."""
        new_a, new_b = compute_elo_update(1500.0, 1500.0, 1.0, k=32.0)
        assert new_a == pytest.approx(1516.0)
        assert new_b == pytest.approx(1484.0)

    def test_equal_ratings_draw(self):
        """Draw between equals — no rating change."""
        new_a, new_b = compute_elo_update(1500.0, 1500.0, 0.5, k=32.0)
        assert new_a == pytest.approx(1500.0)
        assert new_b == pytest.approx(1500.0)

    def test_upset_win_large_gain(self):
        """Weak player beats strong player — gains more than K/2."""
        new_a, new_b = compute_elo_update(1200.0, 1800.0, 1.0, k=32.0)
        assert new_a > 1200.0 + 30.0
        assert new_b < 1800.0 - 30.0

    def test_custom_k_factor(self):
        """K-factor scales the update magnitude."""
        new_a_k16, _ = compute_elo_update(1500.0, 1500.0, 1.0, k=16.0)
        new_a_k64, _ = compute_elo_update(1500.0, 1500.0, 1.0, k=64.0)
        delta_16 = new_a_k16 - 1500.0
        delta_64 = new_a_k64 - 1500.0
        assert delta_64 == pytest.approx(delta_16 * 4.0)


class TestOpponentSampler:
    def test_sample_from_pool(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        for i in range(5):
            pool.add_snapshot(model, "resnet", {}, epoch=i)

        sampler = OpponentSampler(pool, historical_ratio=0.8, current_best_ratio=0.2)
        pool_ids = {e.id for e in pool.list_entries()}
        for _ in range(10):
            entry = sampler.sample()
            assert isinstance(entry, OpponentEntry)
            assert entry.id in pool_ids

    def test_current_best_is_highest_elo(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        for i in range(5):
            pool.add_snapshot(model, "resnet", {}, epoch=i)

        # Make epoch 2 the strongest
        entries = pool.list_entries()
        pool.update_elo(entries[2].id, 1200.0)

        sampler = OpponentSampler(pool, historical_ratio=0.0, current_best_ratio=1.0)
        for _ in range(10):
            entry = sampler.sample()
            assert entry.created_epoch == 2

    def test_single_entry_returns_it(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)

        sampler = OpponentSampler(pool, historical_ratio=0.8, current_best_ratio=0.2)
        entry = sampler.sample()
        assert entry.created_epoch == 0

    def test_elo_floor_excludes_weak_from_historical(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)
        pool.add_snapshot(model, "resnet", {}, epoch=1)
        pool.add_snapshot(model, "resnet", {}, epoch=2)

        entries = pool.list_entries()
        pool.update_elo(entries[0].id, 400.0)  # below floor

        sampler = OpponentSampler(
            pool, historical_ratio=1.0, current_best_ratio=0.0, elo_floor=500.0,
        )
        # Historical sampling should only return epoch 1 (above floor), not epoch 0
        for _ in range(10):
            entry = sampler.sample()
            assert entry.created_epoch != 0

    def test_all_below_floor_returns_current_best(self, league_db, league_dir):
        """When no historical entries clear the floor, returns current_best."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)
        pool.add_snapshot(model, "resnet", {}, epoch=1)

        entries = pool.list_entries()
        pool.update_elo(entries[0].id, 400.0)
        pool.update_elo(entries[1].id, 400.0)

        sampler = OpponentSampler(
            pool, historical_ratio=1.0, current_best_ratio=0.0, elo_floor=500.0,
        )
        # Both below floor → returns current_best (highest Elo, which is tied at 400)
        entry = sampler.sample()
        assert entry is not None
        assert entry.id in {e.id for e in entries}

    def test_empty_pool_raises(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        sampler = OpponentSampler(pool)
        with pytest.raises(ValueError, match="empty"):
            sampler.sample()


class TestOpponentSamplerSampleFrom:
    """Tests for sample_from() with pre-fetched entries."""

    def test_sample_from_matches_sample(self, league_db, league_dir):
        """sample_from(entries) should draw from both historical and current_best."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        for i in range(5):
            pool.add_snapshot(model, "resnet", {}, epoch=i)

        sampler = OpponentSampler(pool, historical_ratio=0.8, current_best_ratio=0.2)
        entries = pool.list_entries()
        pool_ids = {e.id for e in entries}

        # Draw enough samples to confirm diversity (not always current_best)
        seen_ids: set[int] = set()
        for _ in range(200):
            entry = sampler.sample_from(entries)
            assert isinstance(entry, OpponentEntry)
            assert entry.id in pool_ids
            seen_ids.add(entry.id)
        # With 5 entries and 200 draws at 80/20 split, we should see at least 2 distinct IDs
        assert len(seen_ids) >= 2, f"Expected diversity, only saw {seen_ids}"

    def test_sample_from_single_entry(self, league_db, league_dir):
        """Single-entry list should always return that entry."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)

        sampler = OpponentSampler(pool)
        entries = pool.list_entries()
        assert sampler.sample_from(entries).created_epoch == 0

    def test_sample_from_empty_raises(self, league_db, league_dir):
        """Empty entries list should raise ValueError."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        sampler = OpponentSampler(pool)
        with pytest.raises(ValueError, match="empty"):
            sampler.sample_from([])

    def test_sample_from_respects_elo_floor(self, league_db, league_dir):
        """Entries below elo_floor excluded from historical sampling."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)
        pool.add_snapshot(model, "resnet", {}, epoch=1)
        pool.add_snapshot(model, "resnet", {}, epoch=2)

        entries = pool.list_entries()
        pool.update_elo(entries[0].id, 400.0)
        # Refresh entries after Elo update
        entries = pool.list_entries()

        sampler = OpponentSampler(
            pool, historical_ratio=1.0, current_best_ratio=0.0, elo_floor=500.0,
        )
        for _ in range(10):
            entry = sampler.sample_from(entries)
            assert entry.created_epoch != 0


class TestDeleteEntryMissingFile:
    """H3: _delete_entry handles missing checkpoint file gracefully."""

    def test_eviction_with_missing_checkpoint_file(self, league_db, league_dir):
        """Add a snapshot, delete its file from disk, then trigger eviction.
        Should not raise and the DB row should be cleaned up."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=2)
        model = torch.nn.Linear(10, 10)

        # Add two snapshots to fill the pool
        pool.add_snapshot(model, "resnet", {}, epoch=0)
        pool.add_snapshot(model, "resnet", {}, epoch=1)

        # Manually delete the checkpoint file for epoch 0
        entries = pool.list_entries()
        epoch_0_entry = [e for e in entries if e.created_epoch == 0][0]
        Path(epoch_0_entry.checkpoint_path).unlink()

        # Adding a third should trigger eviction of epoch 0 (missing file)
        pool.add_snapshot(model, "resnet", {}, epoch=2)

        # Verify: epoch 0 should be gone from DB, no exception raised
        remaining = pool.list_entries()
        remaining_epochs = [e.created_epoch for e in remaining]
        assert 0 not in remaining_epochs
        assert len(remaining) == 2


class TestAllPinnedEviction:
    """H3: When all entries are pinned and pool exceeds max_pool_size,
    a warning is logged and no crash occurs."""

    def test_all_pinned_logs_warning_no_crash(self, league_db, league_dir, caplog):
        """Pin all entries then call _evict_if_needed directly.
        Verify warning is logged and no crash occurs."""
        import logging

        # Use max_pool_size=2 but add 3 entries and pin all of them
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=2)
        model = torch.nn.Linear(10, 10)

        # Add three entries with a large enough pool first
        pool.max_pool_size = 10
        pool.add_snapshot(model, "resnet", {}, epoch=0)
        pool.add_snapshot(model, "resnet", {}, epoch=1)
        pool.add_snapshot(model, "resnet", {}, epoch=2)

        # Pin all entries
        for entry in pool.list_entries():
            pool.pin(entry.id)

        # Now shrink pool size and trigger eviction
        pool.max_pool_size = 2
        with caplog.at_level(logging.WARNING, logger="keisei.training.league"):
            pool._evict_if_needed()

        # Should have logged a warning about all entries being pinned
        assert any(
            "All entries pinned" in record.message
            for record in caplog.records
        ), f"Expected 'All entries pinned' warning, got: {[r.message for r in caplog.records]}"

        # All three entries should still exist (none evicted)
        entries = pool.list_entries()
        assert len(entries) == 3

        # Cleanup pins
        for entry in entries:
            pool.unpin(entry.id)


class TestOpponentSamplerEdgeCases:
    """Edge cases for OpponentSampler.sample()."""

    def _make_pool_with_entries(self, tmp_path, entries_data):
        """Create a real OpponentPool with specific Elo ratings."""
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams

        db_path = str(tmp_path / "sampler_test.db")
        init_db(db_path)
        league_dir = tmp_path / "sampler_league"
        league_dir.mkdir()
        pool = OpponentPool(db_path, str(league_dir), max_pool_size=20)

        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8, global_pool_channels=16,
            policy_channels=8, value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        model = SEResNetModel(params)

        entries = []
        for i, elo in enumerate(entries_data):
            entry = pool.add_snapshot(model, "se_resnet", dataclasses.asdict(params), epoch=i)
            pool.update_elo(entry.id, elo)
            entries.append(entry)
        return pool, entries

    def test_single_entry_returns_that_entry(self, tmp_path):
        """Pool with 1 entry — sample() should return it directly."""
        pool, entries = self._make_pool_with_entries(tmp_path, [1500.0])
        sampler = OpponentSampler(pool, elo_floor=500.0)
        result = sampler.sample()
        assert result.id == entries[0].id

    def test_all_below_elo_floor_returns_current_best(self, tmp_path):
        """When all historical entries are below floor, always return current_best."""
        pool, entries = self._make_pool_with_entries(tmp_path, [400.0, 300.0, 450.0])
        sampler = OpponentSampler(pool, elo_floor=500.0, current_best_ratio=0.5)
        # All below 500 floor. Best is the one with elo 450.
        # Historical (above floor, excluding best) → empty → always current_best
        results = [sampler.sample() for _ in range(20)]
        # All should be the same entry (the best one)
        ids = {r.id for r in results}
        assert len(ids) == 1

    def test_current_best_ratio_respected(self, tmp_path):
        """Statistical test: current_best should be sampled ~current_best_ratio of the time."""
        import random as stdlib_random

        pool, entries = self._make_pool_with_entries(tmp_path, [1000.0, 1500.0, 1200.0])
        sampler = OpponentSampler(
            pool, elo_floor=500.0, current_best_ratio=0.3, historical_ratio=0.7
        )
        stdlib_random.seed(42)
        n_samples = 200
        # The best entry has elo 1500
        best_entry = max(pool.list_entries(), key=lambda e: e.elo_rating)
        best_count = sum(1 for _ in range(n_samples) if sampler.sample().id == best_entry.id)
        ratio = best_count / n_samples
        assert 0.1 < ratio < 0.6, f"Current best sampled {ratio:.0%}, expected ~30%"

    def test_empty_pool_raises(self, tmp_path):
        """Empty pool should raise ValueError."""
        db_path = str(tmp_path / "empty.db")
        init_db(db_path)
        league_dir = tmp_path / "empty_league"
        league_dir.mkdir()
        pool = OpponentPool(db_path, str(league_dir))
        sampler = OpponentSampler(pool)
        with pytest.raises(ValueError, match="empty opponent pool"):
            sampler.sample()


class TestDeleteEntryCleansGameSnapshots:
    """Evicting a league entry must null out game_snapshots.opponent_id
    referencing that entry, so snapshots remain valid display data."""

    def test_eviction_nulls_game_snapshot_opponent_id(self, league_db, league_dir):
        import sqlite3

        pool = OpponentPool(league_db, str(league_dir), max_pool_size=1)
        model = torch.nn.Linear(10, 10)

        # Add an entry, record its id
        entry = pool.add_snapshot(model, "resnet", {}, epoch=0)

        # Insert a game_snapshot referencing this entry
        conn = sqlite3.connect(league_db)
        conn.execute(
            "INSERT INTO game_snapshots "
            "(game_id, board_json, hands_json, current_player, ply, is_over, "
            "result, sfen, in_check, move_history_json, opponent_id) "
            "VALUES (1, '{}', '{}', 'black', 0, 0, 'in_progress', 'startpos', 0, '[]', ?)",
            (entry.id,),
        )
        conn.commit()
        conn.close()

        # Add a second entry — triggers eviction of the first
        pool.add_snapshot(model, "resnet", {}, epoch=1)

        # The game_snapshot should still exist but with opponent_id = NULL
        conn = sqlite3.connect(league_db)
        row = conn.execute(
            "SELECT opponent_id FROM game_snapshots WHERE game_id = 1"
        ).fetchone()
        conn.close()
        assert row is not None, "game_snapshot row should still exist"
        assert row[0] is None, (
            f"opponent_id should be NULL after eviction, got {row[0]}"
        )


class TestLoadAllOpponents:
    """Tests for OpponentPool.load_all_opponents."""

    def test_loads_all_entries(self, league_db, league_dir):
        """Should return a dict with one model per pool entry."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        for i in range(3):
            pool.add_snapshot(model, "resnet", {"hidden_size": 16, "num_layers": 2}, epoch=i)

        with patch("keisei.training.league.build_model") as mock_build:
            mock_build.side_effect = lambda arch, params: torch.nn.Linear(10, 10)
            models = pool.load_all_opponents(device="cpu")

        entries = pool.list_entries()
        assert len(models) == 3
        for entry in entries:
            assert entry.id in models
            assert isinstance(models[entry.id], torch.nn.Module)

    def test_skips_corrupt_checkpoint(self, league_db, league_dir):
        """Corrupt checkpoint should be skipped with a warning, not crash."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {"hidden_size": 16, "num_layers": 2}, epoch=0)
        pool.add_snapshot(model, "resnet", {"hidden_size": 16, "num_layers": 2}, epoch=1)

        # Corrupt the first entry's checkpoint
        entries = pool.list_entries()
        Path(entries[0].checkpoint_path).write_text("not a valid checkpoint")

        with patch("keisei.training.league.build_model") as mock_build:
            mock_build.side_effect = lambda arch, params: torch.nn.Linear(10, 10)
            models = pool.load_all_opponents(device="cpu")

        assert len(models) == 1
        assert entries[1].id in models
        assert entries[0].id not in models

    def test_skips_missing_checkpoint(self, league_db, league_dir):
        """Missing checkpoint file should be skipped, not crash."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {"hidden_size": 16, "num_layers": 2}, epoch=0)
        pool.add_snapshot(model, "resnet", {"hidden_size": 16, "num_layers": 2}, epoch=1)

        # Delete the first entry's checkpoint
        entries = pool.list_entries()
        Path(entries[0].checkpoint_path).unlink()

        with patch("keisei.training.league.build_model") as mock_build:
            mock_build.side_effect = lambda arch, params: torch.nn.Linear(10, 10)
            models = pool.load_all_opponents(device="cpu")

        assert len(models) == 1
        assert entries[1].id in models

    def test_empty_pool_returns_empty_dict(self, league_db, league_dir):
        """Empty pool should return empty dict."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        models = pool.load_all_opponents(device="cpu")
        assert models == {}


class TestThreadSafety:
    """Regression: OpponentPool must be usable from background threads."""

    def test_cross_thread_list_entries(self, league_db, league_dir):
        """list_entries() from a background thread must not raise."""
        import threading

        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)

        result = {}

        def worker():
            try:
                entries = pool.list_entries()
                result["entries"] = entries
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)

        assert "error" not in result, f"Background thread raised: {result.get('error')}"
        assert len(result["entries"]) == 1

    def test_cross_thread_pin_unpin(self, league_db, league_dir):
        """pin/unpin from a background thread must not corrupt state."""
        import threading

        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        entry = pool.add_snapshot(model, "resnet", {}, epoch=0)

        errors = []

        def worker():
            try:
                pool.pin(entry.id)
                pool.unpin(entry.id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Threads raised: {errors}"


class TestSnapshotFilenameUniqueness:
    """Regression: snapshot filenames must be unique across restarts."""

    def test_same_epoch_different_entries_get_unique_files(self, league_db, league_dir):
        """Two snapshots at epoch=0 (e.g., restart) must not share a filename."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)

        entry1 = pool.add_snapshot(model, "resnet", {}, epoch=0)
        entry2 = pool.add_snapshot(model, "resnet", {}, epoch=0)

        assert entry1.checkpoint_path != entry2.checkpoint_path, (
            f"Both entries share checkpoint path: {entry1.checkpoint_path}"
        )
        assert Path(entry1.checkpoint_path).exists()
        assert Path(entry2.checkpoint_path).exists()
