"""Tests for the DemonstratorRunner — inference-only exhibition matches."""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from keisei.training.demonstrator import DemoMatchup, DemonstratorRunner


def _make_mock_model():
    model = MagicMock()

    def forward(obs):
        batch = obs.shape[0]
        output = MagicMock()
        output.policy_logits = torch.randn(batch, 9, 9, 139)
        return output

    model.__call__ = forward  # type: ignore[method-assign]
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    return model


def _make_mock_pool(num_entries=3):
    from keisei.training.opponent_store import OpponentEntry

    pool = MagicMock()
    entries = [
        OpponentEntry(
            id=i, display_name=f"Test {i}", architecture="resnet",
            model_params={"hidden_size": 16},
            checkpoint_path=f"/fake/ckpt_{i}.pt", elo_rating=1000.0 + i * 50,
            created_epoch=i * 10, games_played=0, created_at="2026-01-01",
            flavour_facts=[],
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
            store=pool, db_path="/tmp/test.db",
            num_slots=3, moves_per_minute=600, device="cpu",
        )
        assert runner.num_slots == 3
        assert not runner.is_alive()

    def test_start_and_stop(self):
        pool = _make_mock_pool()
        runner = DemonstratorRunner(
            store=pool, db_path="/tmp/test.db",
            num_slots=1, moves_per_minute=6000, device="cpu",
        )
        runner.start()
        # Poll for thread to become alive (avoids fixed sleep on slow CI)
        deadline = time.monotonic() + 2.0
        while not runner.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert runner.is_alive()
        runner.stop()
        runner.join(timeout=5.0)
        assert not runner.is_alive()

    def test_crash_is_non_fatal(self):
        """Per-slot crashes should be caught and logged, not kill the thread."""
        pool = _make_mock_pool(num_entries=3)
        pool.load_opponent.side_effect = RuntimeError("simulated crash")
        runner = DemonstratorRunner(
            store=pool, db_path="/tmp/test.db",
            num_slots=1, moves_per_minute=6000, device="cpu",
        )
        runner.start()
        # Wait for the thread to attempt at least one game cycle
        deadline = time.monotonic() + 2.0
        while pool.load_opponent.call_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pool.load_opponent.call_count > 0, "Thread never attempted load_opponent"
        # Thread should still be alive — per-slot crashes don't kill the loop
        assert runner.is_alive()
        runner.stop()
        runner.join(timeout=5.0)
        assert not runner.is_alive()

    def test_slot_fallback_insufficient_entries(self):
        """With < 2 entries, slots should be inactive."""
        pool = _make_mock_pool(num_entries=1)
        runner = DemonstratorRunner(
            store=pool, db_path="/tmp/test.db",
            num_slots=3, moves_per_minute=6000, device="cpu",
        )
        matchups = runner._select_matchups()
        assert len(matchups) == 0

    def test_select_matchups_with_3_entries(self):
        """With 3 entries and 3 slots, should get 3 matchups with correct slot numbers."""
        pool = _make_mock_pool(num_entries=3)
        runner = DemonstratorRunner(
            store=pool, db_path="/tmp/test.db",
            num_slots=3, moves_per_minute=6000, device="cpu",
        )
        matchups = runner._select_matchups()
        assert len(matchups) == 3
        assert matchups[0].slot == 1  # championship
        assert matchups[1].slot == 2  # cross-architecture or random
        assert matchups[2].slot == 3  # random


class TestPlayGamePinUnpin:
    """Tests for _play_game pin/unpin lifecycle (GAP H2)."""

    def test_pin_called_before_load_unpin_called_after(self):
        """pin() is called for both entries before load_opponent, unpin() after."""
        import sys

        pool = _make_mock_pool(num_entries=2)
        runner = DemonstratorRunner(
            store=pool, db_path="/tmp/test.db",
            num_slots=1, moves_per_minute=6000, device="cpu",
        )
        entries = pool.list_entries()
        matchup = DemoMatchup(slot=1, entry_a=entries[0], entry_b=entries[1])

        # Block shogi_gym import so _play_game returns after load (pin/unpin
        # lifecycle is what we're testing, not the game loop).
        saved = sys.modules.get("shogi_gym")
        sys.modules["shogi_gym"] = None  # type: ignore[assignment]  # forces ImportError
        try:
            runner._play_game(matchup)
        finally:
            if saved is not None:
                sys.modules["shogi_gym"] = saved
            else:
                sys.modules.pop("shogi_gym", None)

        # Verify pin was called for both entries
        pin_calls = [c.args[0] for c in pool.pin.call_args_list]
        assert entries[0].id in pin_calls
        assert entries[1].id in pin_calls

        # Verify unpin was called for both entries
        unpin_calls = [c.args[0] for c in pool.unpin.call_args_list]
        assert entries[0].id in unpin_calls
        assert entries[1].id in unpin_calls

    def test_unpin_called_on_file_not_found_error(self):
        """FileNotFoundError during checkpoint load still calls unpin for both."""
        pool = _make_mock_pool(num_entries=2)
        pool.load_opponent.side_effect = FileNotFoundError("missing checkpoint")
        runner = DemonstratorRunner(
            store=pool, db_path="/tmp/test.db",
            num_slots=1, moves_per_minute=6000, device="cpu",
        )
        entries = pool.list_entries()
        matchup = DemoMatchup(slot=1, entry_a=entries[0], entry_b=entries[1])

        # Should not raise — the FileNotFoundError is caught and logged
        runner._play_game(matchup)

        # Pin was called for both
        assert pool.pin.call_count == 2

        # Unpin was called for both even though load failed
        assert pool.unpin.call_count == 2
        unpin_calls = [c.args[0] for c in pool.unpin.call_args_list]
        assert entries[0].id in unpin_calls
        assert entries[1].id in unpin_calls

    def test_unpin_called_on_unexpected_exception(self):
        """Unpin fires in finally even for unexpected exceptions from load_opponent."""
        pool = _make_mock_pool(num_entries=2)
        pool.load_opponent.side_effect = RuntimeError("unexpected failure")
        runner = DemonstratorRunner(
            store=pool, db_path="/tmp/test.db",
            num_slots=1, moves_per_minute=6000, device="cpu",
        )
        entries = pool.list_entries()
        matchup = DemoMatchup(slot=1, entry_a=entries[0], entry_b=entries[1])

        # RuntimeError is NOT caught by _play_game — it propagates
        with pytest.raises(RuntimeError, match="unexpected failure"):
            runner._play_game(matchup)

        # But unpin must still have been called (finally block)
        assert pool.unpin.call_count == 2

    def test_file_not_found_returns_without_crashing(self):
        """FileNotFoundError fast path logs warning and returns gracefully."""
        pool = _make_mock_pool(num_entries=2)
        pool.load_opponent.side_effect = FileNotFoundError("gone")
        runner = DemonstratorRunner(
            store=pool, db_path="/tmp/test.db",
            num_slots=1, moves_per_minute=6000, device="cpu",
        )
        entries = pool.list_entries()
        matchup = DemoMatchup(slot=1, entry_a=entries[0], entry_b=entries[1])

        # Must not raise an exception
        runner._play_game(matchup)

    def test_pin_order_before_load(self):
        """pin() calls happen before load_opponent() calls."""
        import sys

        pool = _make_mock_pool(num_entries=2)
        call_order = []
        pool.pin.side_effect = lambda id_: call_order.append(("pin", id_))
        def _load_side_effect(entry: object, device: object = None) -> object:
            call_order.append(("load", entry.id))  # type: ignore[attr-defined]
            return _make_mock_model()
        pool.load_opponent.side_effect = _load_side_effect
        pool.unpin.side_effect = lambda id_: call_order.append(("unpin", id_))

        runner = DemonstratorRunner(
            store=pool, db_path="/tmp/test.db",
            num_slots=1, moves_per_minute=6000, device="cpu",
        )
        entries = pool.list_entries()
        matchup = DemoMatchup(slot=1, entry_a=entries[0], entry_b=entries[1])

        # Block shogi_gym import — we're testing pin/load ordering, not game play.
        saved = sys.modules.get("shogi_gym")
        sys.modules["shogi_gym"] = None  # type: ignore[assignment]
        try:
            runner._play_game(matchup)
        finally:
            if saved is not None:
                sys.modules["shogi_gym"] = saved
            else:
                sys.modules.pop("shogi_gym", None)

        # Both pins come before any loads
        pin_indices = [i for i, (op, _) in enumerate(call_order) if op == "pin"]
        load_indices = [i for i, (op, _) in enumerate(call_order) if op == "load"]
        assert len(load_indices) > 0, "load_opponent should have been called"
        assert max(pin_indices) < min(load_indices)


class TestPlayGameWithMockedVecEnv:
    """MED-1: Exercise _play_game() while-loop body with a mocked shogi_gym.VecEnv."""

    def _make_mock_vecenv(self, terminate_after: int = 3):
        """Create a mock VecEnv that terminates after N steps."""
        step_count = [0]
        mock_env = MagicMock()

        def make_reset():
            result = MagicMock()
            result.observations = np.zeros((1, 50, 9, 9), dtype=np.float32)
            result.legal_masks = np.ones((1, 11259), dtype=bool)
            return result

        def make_step(actions):
            step_count[0] += 1
            result = MagicMock()
            result.observations = np.zeros((1, 50, 9, 9), dtype=np.float32)
            result.legal_masks = np.ones((1, 11259), dtype=bool)
            result.terminated = np.array([step_count[0] >= terminate_after])
            result.truncated = np.array([False])
            result.current_players = np.array([step_count[0] % 2], dtype=np.uint8)
            return result

        mock_env.reset.side_effect = lambda: make_reset()
        mock_env.step.side_effect = make_step
        return mock_env, step_count

    @staticmethod
    def _make_real_model():
        """Create a real (tiny) SEResNetModel for demonstrator game loop testing."""
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams

        params = SEResNetParams(
            num_blocks=1, channels=16, se_reduction=4,
            global_pool_channels=8, policy_channels=4,
            value_fc_size=16, score_fc_size=8, obs_channels=50,
        )
        model = SEResNetModel(params)
        model.eval()
        return model

    def test_play_game_runs_to_completion(self):
        """_play_game() should run the game loop until termination with mocked VecEnv."""
        import sys
        import types

        mock_vecenv, step_count = self._make_mock_vecenv(terminate_after=3)

        fake_shogi_gym = types.ModuleType("shogi_gym")
        fake_shogi_gym.VecEnv = MagicMock(return_value=mock_vecenv)  # type: ignore[attr-defined]

        pool = _make_mock_pool(num_entries=2)
        # Use real models so forward passes produce real tensors
        real_model = self._make_real_model()
        pool.load_opponent.return_value = real_model

        runner = DemonstratorRunner(
            store=pool, db_path="/tmp/test.db",
            num_slots=1, moves_per_minute=600_000,  # very fast to avoid sleep
            device="cpu",
        )
        entries = pool.list_entries()
        matchup = DemoMatchup(slot=1, entry_a=entries[0], entry_b=entries[1])

        old_module = sys.modules.get("shogi_gym")
        sys.modules["shogi_gym"] = fake_shogi_gym
        try:
            runner._play_game(matchup)
        finally:
            if old_module is not None:
                sys.modules["shogi_gym"] = old_module
            else:
                sys.modules.pop("shogi_gym", None)

        # Verify the game loop actually ran
        assert step_count[0] == 3, f"Expected 3 steps, got {step_count[0]}"
        fake_shogi_gym.VecEnv.assert_called_once()
        assert mock_vecenv.reset.call_count == 1
        assert mock_vecenv.step.call_count == 3

    def test_play_game_respects_stop_event(self):
        """_play_game() should exit early when stop event is set."""
        import sys
        import types

        mock_vecenv, step_count = self._make_mock_vecenv(terminate_after=999)

        fake_shogi_gym = types.ModuleType("shogi_gym")
        fake_shogi_gym.VecEnv = MagicMock(return_value=mock_vecenv)  # type: ignore[attr-defined]

        pool = _make_mock_pool(num_entries=2)
        real_model = self._make_real_model()
        pool.load_opponent.return_value = real_model

        runner = DemonstratorRunner(
            store=pool, db_path="/tmp/test.db",
            num_slots=1, moves_per_minute=600_000,
            device="cpu",
        )
        entries = pool.list_entries()
        matchup = DemoMatchup(slot=1, entry_a=entries[0], entry_b=entries[1])

        # Set stop event before the game starts — it should exit after first check
        runner._stop_event.set()

        old_module = sys.modules.get("shogi_gym")
        sys.modules["shogi_gym"] = fake_shogi_gym
        try:
            runner._play_game(matchup)
        finally:
            if old_module is not None:
                sys.modules["shogi_gym"] = old_module
            else:
                sys.modules.pop("shogi_gym", None)

        # Stop was set before the game loop — the while condition checks
        # _stop_event.is_set() immediately, so zero steps should execute
        assert step_count[0] == 0


class TestMoveDelay:
    """Tests for move_delay calculation (GAP H2)."""

    def test_move_delay_normal(self):
        """move_delay = 60.0 / moves_per_minute for normal values."""
        pool = _make_mock_pool()
        runner = DemonstratorRunner(
            store=pool, db_path="/tmp/test.db",
            moves_per_minute=60, device="cpu",
        )
        assert runner.move_delay == pytest.approx(1.0)

    def test_move_delay_high_speed(self):
        """Higher moves_per_minute gives shorter delay."""
        pool = _make_mock_pool()
        runner = DemonstratorRunner(
            store=pool, db_path="/tmp/test.db",
            moves_per_minute=6000, device="cpu",
        )
        assert runner.move_delay == pytest.approx(0.01)

    def test_move_delay_zero_moves_per_minute(self):
        """Zero moves_per_minute is guarded by max(..., 1), giving 60s delay."""
        pool = _make_mock_pool()
        runner = DemonstratorRunner(
            store=pool, db_path="/tmp/test.db",
            moves_per_minute=0, device="cpu",
        )
        # max(0, 1) = 1 => 60.0 / 1 = 60.0
        assert runner.move_delay == pytest.approx(60.0)

    def test_move_delay_negative_moves_per_minute(self):
        """Negative moves_per_minute is guarded by max(..., 1)."""
        pool = _make_mock_pool()
        runner = DemonstratorRunner(
            store=pool, db_path="/tmp/test.db",
            moves_per_minute=-10, device="cpu",
        )
        assert runner.move_delay == pytest.approx(60.0)

    def test_move_delay_one(self):
        """moves_per_minute=1 gives 60s delay."""
        pool = _make_mock_pool()
        runner = DemonstratorRunner(
            store=pool, db_path="/tmp/test.db",
            moves_per_minute=1, device="cpu",
        )
        assert runner.move_delay == pytest.approx(60.0)
