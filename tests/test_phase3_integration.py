"""Phase 3 integration tests — rollout collection from match_utils."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from keisei.training.dynamic_trainer import MatchRollout
from keisei.training.match_utils import play_batch, play_match


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockModel(nn.Module):
    """Minimal model that returns random policy logits and a dummy value."""

    def __init__(self, action_dim: int = 11259) -> None:
        super().__init__()
        self.action_dim = action_dim
        # Need at least one parameter so PyTorch considers it a Module
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        batch = x.shape[0]
        return SimpleNamespace(
            policy_logits=torch.randn(batch, self.action_dim),
            value=torch.zeros(batch, 1),
        )


class MockVecEnv:
    """Minimal VecEnv mock for testing match_utils."""

    def __init__(self, num_envs: int = 2, obs_channels: int = 50, end_ply: int = 5) -> None:
        self.num_envs = num_envs
        self.obs_channels = obs_channels
        self.end_ply = end_ply
        self._step_count = 0
        self._current_player = 0  # alternates each step

    def reset(self) -> SimpleNamespace:
        self._step_count = 0
        self._current_player = 0
        return SimpleNamespace(
            observations=np.random.randn(
                self.num_envs, self.obs_channels, 9, 9
            ).astype(np.float32),
            legal_masks=np.ones((self.num_envs, 11259), dtype=bool),
        )

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        self._step_count += 1
        old_player = self._current_player
        self._current_player = 1 - self._current_player

        terminated = self._step_count >= self.end_ply
        rewards = np.array(
            [1.0 if terminated else 0.0] * self.num_envs, dtype=np.float32
        )

        return SimpleNamespace(
            observations=np.random.randn(
                self.num_envs, self.obs_channels, 9, 9
            ).astype(np.float32),
            legal_masks=np.ones((self.num_envs, 11259), dtype=bool),
            current_players=np.full(
                self.num_envs, self._current_player, dtype=np.uint8
            ),
            rewards=rewards,
            terminated=np.full(self.num_envs, terminated),
            truncated=np.full(self.num_envs, False),
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_play_batch_collects_rollout_when_requested():
    """play_batch with collect_rollout=True returns a 4-tuple with MatchRollout."""
    vecenv = MockVecEnv(num_envs=2, end_ply=5)
    model_a = _MockModel()
    model_b = _MockModel()

    result = play_batch(
        vecenv, model_a, model_b,
        device=torch.device("cpu"),
        num_envs=2,
        max_ply=20,
        collect_rollout=True,
    )

    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple"
    a_wins, b_wins, draws, rollout = result
    assert isinstance(rollout, MatchRollout)
    assert rollout.observations.shape[0] > 0
    # All tensors should be on CPU
    for name in ("observations", "actions", "rewards", "dones", "legal_masks", "perspective"):
        t = getattr(rollout, name)
        assert t.device == torch.device("cpu"), f"{name} not on CPU"


def test_play_batch_no_rollout_by_default():
    """play_batch without collect_rollout returns a 3-tuple."""
    vecenv = MockVecEnv(num_envs=2, end_ply=3)
    model_a = _MockModel()
    model_b = _MockModel()

    result = play_batch(
        vecenv, model_a, model_b,
        device=torch.device("cpu"),
        num_envs=2,
        max_ply=20,
    )

    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
    a_wins, b_wins, draws = result
    assert isinstance(a_wins, int)


def test_play_batch_rollout_has_correct_perspective():
    """Perspective tensor alternates between 0 and 1 each ply."""
    vecenv = MockVecEnv(num_envs=2, end_ply=6)
    model_a = _MockModel()
    model_b = _MockModel()

    result = play_batch(
        vecenv, model_a, model_b,
        device=torch.device("cpu"),
        num_envs=2,
        max_ply=20,
        collect_rollout=True,
    )
    _, _, _, rollout = result
    perspective = rollout.perspective

    # MockVecEnv starts current_player=0 and alternates each step.
    # Perspective is captured BEFORE step, so:
    # ply 0: current_players=0 (init), ply 1: current_players=1 (after step flips), etc.
    for ply in range(perspective.shape[0]):
        expected = ply % 2
        assert (perspective[ply] == expected).all(), (
            f"Ply {ply}: expected all {expected}, got {perspective[ply]}"
        )


def test_play_batch_perspective_captured_before_step():
    """Perspective at ply 0 reflects the INITIAL current_players (before first step)."""
    vecenv = MockVecEnv(num_envs=2, end_ply=3)
    model_a = _MockModel()
    model_b = _MockModel()

    result = play_batch(
        vecenv, model_a, model_b,
        device=torch.device("cpu"),
        num_envs=2,
        max_ply=20,
        collect_rollout=True,
    )
    _, _, _, rollout = result

    # current_players is initialized to zeros before the loop.
    # Perspective should capture this BEFORE the first step() call.
    assert (rollout.perspective[0] == 0).all(), (
        f"First perspective should be 0 (initial), got {rollout.perspective[0]}"
    )


def test_play_match_collects_rollout():
    """play_match with collect_rollout=True threads through and returns 4-tuple."""
    vecenv = MockVecEnv(num_envs=2, end_ply=3)
    model_a = _MockModel()
    model_b = _MockModel()

    result = play_match(
        vecenv, model_a, model_b,
        device=torch.device("cpu"),
        num_envs=2,
        max_ply=20,
        games_target=4,
        collect_rollout=True,
    )

    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple"
    a_wins, b_wins, draws, rollout = result
    assert isinstance(rollout, MatchRollout)
    assert rollout.observations.shape[0] > 0
