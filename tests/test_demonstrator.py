"""Tests for the DemonstratorRunner — inference-only exhibition matches."""

import time
from unittest.mock import MagicMock

import pytest
import torch

from keisei.training.demonstrator import DemonstratorRunner


def _make_mock_model():
    model = MagicMock()

    def forward(obs):
        batch = obs.shape[0]
        output = MagicMock()
        output.policy_logits = torch.randn(batch, 9, 9, 139)
        return output

    model.__call__ = forward
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    return model


def _make_mock_pool(num_entries=3):
    from keisei.training.league import OpponentEntry

    pool = MagicMock()
    entries = [
        OpponentEntry(
            id=i, architecture="resnet", model_params={"hidden_size": 16},
            checkpoint_path=f"/fake/ckpt_{i}.pt", elo_rating=1000.0 + i * 50,
            created_epoch=i * 10, games_played=0, created_at="2026-01-01",
        )
        for i in range(num_entries)
    ]
    pool.list_entries.return_value = entries
    pool.load_opponent.return_value = _make_mock_model()
    pool.pin = MagicMock()
    pool.unpin = MagicMock()
    return pool


class TestDemonstratorRunner:
    def test_init_creates_runner(self):
        pool = _make_mock_pool()
        runner = DemonstratorRunner(
            pool=pool, db_path="/tmp/test.db",
            num_slots=3, moves_per_minute=600, device="cpu",
        )
        assert runner.num_slots == 3
        assert not runner.is_alive()

    def test_start_and_stop(self):
        pool = _make_mock_pool()
        runner = DemonstratorRunner(
            pool=pool, db_path="/tmp/test.db",
            num_slots=1, moves_per_minute=6000, device="cpu",
        )
        runner.start()
        assert runner.is_alive()
        time.sleep(0.1)
        runner.stop()
        runner.join(timeout=2.0)
        assert not runner.is_alive()

    def test_crash_is_non_fatal(self):
        """Per-slot crashes should be caught and logged, not kill the thread."""
        pool = _make_mock_pool(num_entries=3)
        pool.load_opponent.side_effect = RuntimeError("simulated crash")
        runner = DemonstratorRunner(
            pool=pool, db_path="/tmp/test.db",
            num_slots=1, moves_per_minute=6000, device="cpu",
        )
        runner.start()
        time.sleep(0.2)  # let it attempt and fail
        # Thread should still be alive — per-slot crashes don't kill the loop
        assert runner.is_alive()
        runner.stop()
        runner.join(timeout=2.0)
        assert not runner.is_alive()

    def test_slot_fallback_insufficient_entries(self):
        """With < 2 entries, slots should be inactive."""
        pool = _make_mock_pool(num_entries=1)
        runner = DemonstratorRunner(
            pool=pool, db_path="/tmp/test.db",
            num_slots=3, moves_per_minute=6000, device="cpu",
        )
        matchups = runner._select_matchups()
        assert len(matchups) == 0

    def test_select_matchups_with_3_entries(self):
        """With 3 entries and 3 slots, should get 3 matchups."""
        pool = _make_mock_pool(num_entries=3)
        runner = DemonstratorRunner(
            pool=pool, db_path="/tmp/test.db",
            num_slots=3, moves_per_minute=6000, device="cpu",
        )
        matchups = runner._select_matchups()
        assert len(matchups) == 3
        assert matchups[0].slot == 1  # championship
