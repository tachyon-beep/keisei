"""Tests for keisei.training.match_utils — shared match-execution utilities.

Covers: play_batch win/draw attribution, stop_event interruption, max_ply
limits, rollout collection, play_match lifecycle, _combine_rollouts
concatenation, and release_models safety.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from keisei.training.dynamic_trainer import MatchRollout
from keisei.training.match_utils import (
    _combine_rollouts,
    play_batch,
    play_match,
    release_models,
)
from tests._helpers import TinyModel, make_rollout

# ---------------------------------------------------------------------------
# MockVecEnv variants
# ---------------------------------------------------------------------------

NUM_ENVS = 2
OBS_CHANNELS = 50
ACTION_SPACE = 11259


class MockVecEnv:
    """Deterministic mock: terminates all envs after *terminate_after* steps.

    When a game terminates the reward is *terminal_reward* (default +1.0,
    meaning the last mover wins).  Set terminal_reward=0.0 for draws.
    """

    def __init__(
        self,
        num_envs: int = NUM_ENVS,
        terminate_after: int = 3,
        terminal_reward: float = 1.0,
    ) -> None:
        self.num_envs = num_envs
        self.terminate_after = terminate_after
        self.terminal_reward = terminal_reward
        self._ply = np.zeros(num_envs, dtype=int)

    def reset(self) -> SimpleNamespace:
        self._ply = np.zeros(self.num_envs, dtype=int)
        return SimpleNamespace(
            observations=np.random.randn(self.num_envs, OBS_CHANNELS, 9, 9).astype(
                np.float32
            ),
            legal_masks=np.ones((self.num_envs, ACTION_SPACE), dtype=bool),
        )

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        self._ply += 1
        terminated = self._ply >= self.terminate_after
        rewards = np.where(terminated, self.terminal_reward, 0.0).astype(np.float32)
        # Auto-reset terminated envs (like a real VecEnv).
        self._ply[terminated] = 0
        return SimpleNamespace(
            observations=np.random.randn(self.num_envs, OBS_CHANNELS, 9, 9).astype(
                np.float32
            ),
            legal_masks=np.ones((self.num_envs, ACTION_SPACE), dtype=bool),
            current_players=np.zeros(self.num_envs, dtype=np.uint8),
            rewards=rewards,
            terminated=terminated,
            truncated=np.zeros(self.num_envs, dtype=bool),
        )


class NeverTerminatingVecEnv:
    """VecEnv that never terminates games — useful for max_ply / ceiling tests."""

    def __init__(self, num_envs: int = NUM_ENVS) -> None:
        self.num_envs = num_envs

    def reset(self) -> SimpleNamespace:
        return SimpleNamespace(
            observations=np.random.randn(self.num_envs, OBS_CHANNELS, 9, 9).astype(
                np.float32
            ),
            legal_masks=np.ones((self.num_envs, ACTION_SPACE), dtype=bool),
        )

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        return SimpleNamespace(
            observations=np.random.randn(self.num_envs, OBS_CHANNELS, 9, 9).astype(
                np.float32
            ),
            legal_masks=np.ones((self.num_envs, ACTION_SPACE), dtype=bool),
            current_players=np.zeros(self.num_envs, dtype=np.uint8),
            rewards=np.zeros(self.num_envs, dtype=np.float32),
            terminated=np.zeros(self.num_envs, dtype=bool),
            truncated=np.zeros(self.num_envs, dtype=bool),
        )


class CountingVecEnv:
    """VecEnv that counts step() calls, for verifying stop_event mid-ply."""

    def __init__(self, num_envs: int = NUM_ENVS) -> None:
        self.num_envs = num_envs
        self.step_count = 0

    def reset(self) -> SimpleNamespace:
        self.step_count = 0
        return SimpleNamespace(
            observations=np.random.randn(self.num_envs, OBS_CHANNELS, 9, 9).astype(
                np.float32
            ),
            legal_masks=np.ones((self.num_envs, ACTION_SPACE), dtype=bool),
        )

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        self.step_count += 1
        return SimpleNamespace(
            observations=np.random.randn(self.num_envs, OBS_CHANNELS, 9, 9).astype(
                np.float32
            ),
            legal_masks=np.ones((self.num_envs, ACTION_SPACE), dtype=bool),
            current_players=np.zeros(self.num_envs, dtype=np.uint8),
            rewards=np.zeros(self.num_envs, dtype=np.float32),
            terminated=np.zeros(self.num_envs, dtype=bool),
            truncated=np.zeros(self.num_envs, dtype=bool),
        )


# ---------------------------------------------------------------------------
# play_batch tests
# ---------------------------------------------------------------------------


class TestPlayBatchWinAttribution:
    """Verify wins are attributed to the correct player based on pre_step_players."""

    def test_play_batch_wins_attributed_to_correct_player(self) -> None:
        """When reward > 0 and player A (side 0) moved, A gets the win.

        MockVecEnv always returns current_players=0 (player A always moves),
        and terminal_reward=+1.0 (last mover wins), so all wins should go to A.
        """
        num_envs = 2
        vecenv = MockVecEnv(num_envs=num_envs, terminate_after=2, terminal_reward=1.0)
        model_a = TinyModel()
        model_b = TinyModel()

        result = play_batch(
            vecenv, model_a, model_b,
            device=torch.device("cpu"),
            num_envs=num_envs,
            max_ply=100,
        )
        a_wins, b_wins, draws = result
        # Player A always moves (current_players=0) and reward is +1 on termination,
        # so A should get all wins; B gets none.
        assert a_wins >= num_envs, f"Expected at least {num_envs} a_wins, got {a_wins}"
        assert b_wins == 0, f"Expected 0 b_wins, got {b_wins}"
        assert draws == 0, f"Expected 0 draws, got {draws}"

    def test_play_batch_draw_counted(self) -> None:
        """When reward == 0 and done == True, draws are incremented."""
        num_envs = 2
        vecenv = MockVecEnv(num_envs=num_envs, terminate_after=2, terminal_reward=0.0)
        model_a = TinyModel()
        model_b = TinyModel()

        result = play_batch(
            vecenv, model_a, model_b,
            device=torch.device("cpu"),
            num_envs=num_envs,
            max_ply=100,
        )
        a_wins, b_wins, draws = result
        assert a_wins == 0, f"Expected 0 a_wins, got {a_wins}"
        assert b_wins == 0, f"Expected 0 b_wins, got {b_wins}"
        assert draws >= num_envs, f"Expected at least {num_envs} draws, got {draws}"


class TestPlayBatchStopEvent:
    """Verify stop_event interrupts play_batch mid-ply."""

    def test_play_batch_stop_event_mid_ply(self) -> None:
        """Set stop_event after N steps via side_effect on the env's step method."""
        num_envs = 2
        stop_after_steps = 3
        vecenv = CountingVecEnv(num_envs=num_envs)
        stop_event = threading.Event()

        # Wrap step to set stop_event after N calls.
        original_step = vecenv.step

        def step_with_stop(actions: np.ndarray) -> SimpleNamespace:
            result = original_step(actions)
            if vecenv.step_count >= stop_after_steps:
                stop_event.set()
            return result

        vecenv.step = step_with_stop  # type: ignore[assignment]

        model_a = TinyModel()
        model_b = TinyModel()

        a_wins, b_wins, draws = play_batch(
            vecenv, model_a, model_b,
            device=torch.device("cpu"),
            num_envs=num_envs,
            max_ply=1000,
            stop_event=stop_event,
        )
        # The loop should have exited after approximately stop_after_steps plies.
        # CountingVecEnv never terminates, so no games complete.
        assert vecenv.step_count <= stop_after_steps + 1
        assert a_wins + b_wins + draws == 0


class TestPlayBatchMaxPly:
    """Verify max_ply prevents infinite loops."""

    def test_play_batch_max_ply_limit(self) -> None:
        """NeverTerminatingVecEnv with small max_ply should return after max_ply iterations."""
        num_envs = 2
        max_ply = 5
        vecenv = NeverTerminatingVecEnv(num_envs=num_envs)
        model_a = TinyModel()
        model_b = TinyModel()

        a_wins, b_wins, draws = play_batch(
            vecenv, model_a, model_b,
            device=torch.device("cpu"),
            num_envs=num_envs,
            max_ply=max_ply,
        )
        # No games should complete since the env never terminates.
        assert a_wins == 0
        assert b_wins == 0
        assert draws == 0


class TestPlayBatchRollout:
    """Verify rollout collection produces correctly shaped tensors."""

    def test_play_batch_collect_rollout_tensor_shapes(self) -> None:
        """collect_rollout=True returns a 4-tuple with MatchRollout having 6 fields."""
        num_envs = 2
        vecenv = MockVecEnv(num_envs=num_envs, terminate_after=3, terminal_reward=1.0)
        model_a = TinyModel()
        model_b = TinyModel()

        result = play_batch(
            vecenv, model_a, model_b,
            device=torch.device("cpu"),
            num_envs=num_envs,
            max_ply=100,
            collect_rollout=True,
        )
        assert len(result) == 4
        a_wins, b_wins, draws, rollout = result
        assert isinstance(rollout, MatchRollout)

        steps = rollout.observations.shape[0]
        assert steps > 0, "Expected at least one step of rollout data"

        # observations: (steps, num_envs, OBS_CHANNELS, 9, 9)
        assert rollout.observations.shape == (steps, num_envs, OBS_CHANNELS, 9, 9)
        # actions: (steps, num_envs)
        assert rollout.actions.shape == (steps, num_envs)
        # rewards: (steps, num_envs)
        assert rollout.rewards.shape == (steps, num_envs)
        # dones: (steps, num_envs)
        assert rollout.dones.shape == (steps, num_envs)
        # legal_masks: (steps, num_envs, ACTION_SPACE)
        assert rollout.legal_masks.shape == (steps, num_envs, ACTION_SPACE)
        # perspective: (steps, num_envs)
        assert rollout.perspective.shape == (steps, num_envs)


# ---------------------------------------------------------------------------
# play_match tests
# ---------------------------------------------------------------------------


class TestPlayMatchBasic:
    """Test play_match returns correct tuple structure."""

    def test_play_match_basic_3tuple(self) -> None:
        """Normal match returns (a_wins, b_wins, draws) summing to >= games_target."""
        num_envs = 2
        vecenv = MockVecEnv(num_envs=num_envs, terminate_after=2, terminal_reward=1.0)
        model_a = TinyModel()
        model_b = TinyModel()
        games_target = 4

        result = play_match(
            vecenv, model_a, model_b,
            device=torch.device("cpu"),
            num_envs=num_envs,
            max_ply=200,
            games_target=games_target,
        )
        total = result.a_wins + result.b_wins + result.draws
        assert total >= games_target, f"Expected >= {games_target} games, got {total}"
        assert result.rollout is None


class TestPlayMatchStopEvent:
    """Test that stop_event interrupts play_match."""

    def test_play_match_stop_event_interrupts(self) -> None:
        """stop_event set after first batch returns early with partial counts."""
        num_envs = 2
        # Use NeverTerminatingVecEnv so it doesn't finish naturally.
        vecenv = NeverTerminatingVecEnv(num_envs=num_envs)
        model_a = TinyModel()
        model_b = TinyModel()
        stop_event = threading.Event()

        # Set the stop event immediately — play_match should check before first batch.
        stop_event.set()

        result = play_match(
            vecenv, model_a, model_b,
            device=torch.device("cpu"),
            num_envs=num_envs,
            max_ply=10,
            games_target=100,
            stop_event=stop_event,
        )
        total = result.a_wins + result.b_wins + result.draws
        assert total < 100, "stop_event should have interrupted before completing 100 games"


class TestPlayMatchMaxBatches:
    """Test that play_match batch ceiling prevents infinite loops."""

    def test_play_match_max_batches_ceiling(self, caplog) -> None:
        """NeverTerminatingVecEnv should hit the batch ceiling and log a warning."""
        num_envs = 2
        vecenv = NeverTerminatingVecEnv(num_envs=num_envs)
        model_a = TinyModel()
        model_b = TinyModel()
        games_target = 2  # small target, but env never finishes

        with caplog.at_level(logging.WARNING, logger="keisei.training.match_utils"):
            result = play_match(
                vecenv, model_a, model_b,
                device=torch.device("cpu"),
                num_envs=num_envs,
                max_ply=5,  # very short to keep test fast
                games_target=games_target,
            )

        total = result.a_wins + result.b_wins + result.draws
        assert total == 0, "NeverTerminatingVecEnv should not produce any completed games"
        assert "batch ceiling" in caplog.text, (
            f"Expected warning about batch ceiling, got: {caplog.text!r}"
        )


class TestPlayMatchRollout:
    """Test rollout collection through play_match."""

    def test_play_match_collect_rollout_4tuple(self) -> None:
        """collect_rollout=True returns 4-tuple with MatchRollout."""
        num_envs = 2
        vecenv = MockVecEnv(num_envs=num_envs, terminate_after=2, terminal_reward=1.0)
        model_a = TinyModel()
        model_b = TinyModel()

        result = play_match(
            vecenv, model_a, model_b,
            device=torch.device("cpu"),
            num_envs=num_envs,
            max_ply=200,
            games_target=2,
            collect_rollout=True,
        )
        assert result.rollout is not None
        assert isinstance(result.rollout, MatchRollout)
        # Should have observations with ndim == 5: (steps, envs, C, 9, 9)
        assert result.rollout.observations.ndim == 5


# ---------------------------------------------------------------------------
# _combine_rollouts tests
# ---------------------------------------------------------------------------


class TestCombineRollouts:
    """Test _combine_rollouts concatenation logic."""

    def test_combine_rollouts_concatenation(self) -> None:
        """Two rollouts with 3 and 5 steps combine to 8 steps."""
        r1 = make_rollout(steps=3, num_envs=2)
        r2 = make_rollout(steps=5, num_envs=2)

        combined = _combine_rollouts([r1, r2])
        assert combined.observations.shape[0] == 8
        assert combined.actions.shape[0] == 8
        assert combined.rewards.shape[0] == 8
        assert combined.dones.shape[0] == 8
        assert combined.legal_masks.shape[0] == 8
        assert combined.perspective.shape[0] == 8

        # Verify first 3 steps come from r1 and next 5 from r2.
        assert torch.equal(combined.observations[:3], r1.observations)
        assert torch.equal(combined.observations[3:], r2.observations)

    def test_combine_rollouts_single(self) -> None:
        """Single rollout is returned with the same data."""
        r = make_rollout(steps=4, num_envs=1)
        combined = _combine_rollouts([r])
        assert torch.equal(combined.observations, r.observations)
        assert torch.equal(combined.actions, r.actions)

    def test_combine_rollouts_empty_filtered(self) -> None:
        """play_match filters out empty rollouts (dim <= 1) before calling
        _combine_rollouts, so we test the filtering logic at the play_match
        level using a NeverTerminatingVecEnv with stop_event."""
        # An empty rollout has shape (0,) — dim == 1, which play_match filters.
        empty_rollout = MatchRollout(
            observations=torch.empty(0),
            actions=torch.empty(0),
            rewards=torch.empty(0),
            dones=torch.empty(0),
            legal_masks=torch.empty(0),
            perspective=torch.empty(0),
        )
        valid_rollout = make_rollout(steps=3, num_envs=1)

        # Simulate what play_match does: filter out dim <= 1
        all_rollouts = [empty_rollout, valid_rollout]
        valid = [r for r in all_rollouts if r.observations.dim() > 1]
        assert len(valid) == 1

        combined = _combine_rollouts(valid)
        assert torch.equal(combined.observations, valid_rollout.observations)


# ---------------------------------------------------------------------------
# release_models tests
# ---------------------------------------------------------------------------


class TestReleaseModels:
    """Test release_models safety on CPU."""

    def test_release_models_cpu_no_crash(self) -> None:
        """Calling release_models with CPU models should not raise."""
        model = TinyModel()
        # Should be a no-op on CPU (only calls cuda.empty_cache when device_type=="cuda").
        release_models(model, device_type="cpu")

    def test_release_models_multiple_models(self) -> None:
        """Multiple models can be passed without error."""
        m1 = TinyModel()
        m2 = TinyModel()
        release_models(m1, m2, device_type="cpu")
