"""Tests for the opponent league: pool, sampler, Elo."""

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
            "CREATE TABLE t (id INT, architecture TEXT, model_params TEXT, "
            "checkpoint_path TEXT, elo_rating REAL, created_epoch INT, "
            "games_played INT, created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO t VALUES (1, 'resnet', '{\"hidden_size\": 16}', "
            "'/path/to/ckpt.pt', 1000.0, 10, 5, '2026-04-01T00:00:00Z')"
        )
        row = conn.execute("SELECT * FROM t").fetchone()
        conn.close()

        entry = OpponentEntry.from_db_row(row)
        assert entry.id == 1
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
