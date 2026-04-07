"""Tests for ConcurrentMatchPool.run_round() reward attribution logic.

GAP-C1: The reward attribution block (concurrent_matches.py lines 390-410)
maps pre_step_players → a_wins/b_wins. These tests exercise all four branches:
  - Player A moved last, reward > 0 → a_wins
  - Player A moved last, reward < 0 → b_wins
  - Player B moved last, reward > 0 → b_wins
  - Player B moved last, reward < 0 → a_wins
  - Reward == 0 (draw) → draws
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from keisei.config import ConcurrencyConfig
from keisei.training.concurrent_matches import ConcurrentMatchPool
from keisei.training.opponent_store import OpponentEntry, Role
from tests._helpers import TinyModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(entry_id: int, role: Role = Role.FRONTIER_STATIC) -> OpponentEntry:
    return OpponentEntry(
        id=entry_id,
        display_name=f"test-{entry_id}",
        architecture="resnet",
        model_params={},
        checkpoint_path="/fake/path.pt",
        elo_rating=1000.0,
        created_epoch=0,
        games_played=0,
        created_at="2026-01-01T00:00:00",
        flavour_facts=[],
        role=role,
    )


def _make_pool(
    parallel_matches: int = 1,
    envs_per_match: int = 2,
) -> ConcurrentMatchPool:
    total_envs = parallel_matches * envs_per_match
    config = ConcurrencyConfig(
        parallel_matches=parallel_matches,
        envs_per_match=envs_per_match,
        total_envs=total_envs,
        max_resident_models=parallel_matches * 2,
    )
    return ConcurrentMatchPool(config)


class RewardControlVecEnv:
    """Mock VecEnv that gives precise control over who moves and who wins.

    On each step:
      - current_players alternates based on ply count (even=0, odd=1)
      - After ``terminate_after`` plies, returns done=True with the
        configured ``terminal_reward``.
      - Auto-resets on termination (ply counter resets to 0).
    """

    def __init__(
        self,
        num_envs: int,
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
            observations=np.zeros((self.num_envs, 50, 9, 9), dtype=np.float32),
            legal_masks=np.ones((self.num_envs, 11259), dtype=bool),
        )

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        self._ply += 1
        terminated = self._ply >= self.terminate_after
        rewards = np.where(terminated, self.terminal_reward, 0.0).astype(np.float32)
        # Auto-reset
        self._ply[terminated] = 0
        # current_players alternates: even ply → player 0, odd ply → player 1
        current_players = (self._ply % 2).astype(np.uint8)
        return SimpleNamespace(
            observations=np.zeros((self.num_envs, 50, 9, 9), dtype=np.float32),
            legal_masks=np.ones((self.num_envs, 11259), dtype=bool),
            current_players=current_players,
            rewards=rewards,
            terminated=terminated,
            truncated=np.zeros(self.num_envs, dtype=bool),
        )


class FixedPlayerVecEnv:
    """Mock VecEnv where the moving player at termination is controlled explicitly.

    All envs terminate after ``terminate_after`` steps.  The player who is
    moving at ply ``terminate_after - 1`` (the ply before termination) is
    ``terminal_player``.  This lets us test all reward-attribution branches:
      - terminal_player=0 + reward=+1 → A won (a_wins)
      - terminal_player=0 + reward=-1 → A lost (b_wins)
      - terminal_player=1 + reward=+1 → B won (b_wins)
      - terminal_player=1 + reward=-1 → B lost (a_wins)
    """

    def __init__(
        self,
        num_envs: int,
        terminate_after: int = 2,
        terminal_reward: float = 1.0,
        terminal_player: int = 0,
    ) -> None:
        self.num_envs = num_envs
        self.terminate_after = terminate_after
        self.terminal_reward = terminal_reward
        self.terminal_player = terminal_player
        self._ply = np.zeros(num_envs, dtype=int)

    def reset(self) -> SimpleNamespace:
        self._ply = np.zeros(self.num_envs, dtype=int)
        # Start with whatever player is needed so that at termination
        # the correct player is moving.  Player at ply N = (start + N) % 2.
        # We want player at ply (terminate_after - 1) = terminal_player.
        # So start = (terminal_player - (terminate_after - 1)) % 2.
        return SimpleNamespace(
            observations=np.zeros((self.num_envs, 50, 9, 9), dtype=np.float32),
            legal_masks=np.ones((self.num_envs, 11259), dtype=bool),
        )

    def _player_at_ply(self, ply: int) -> int:
        """Which player moves at the given ply."""
        # We want ply (terminate_after - 1) to be terminal_player.
        offset = (self.terminal_player - (self.terminate_after - 1)) % 2
        return (offset + ply) % 2

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        self._ply += 1
        terminated = self._ply >= self.terminate_after
        rewards = np.where(terminated, self.terminal_reward, 0.0).astype(np.float32)

        # current_players for the NEXT ply (post-step state).
        # pre_step_players (what run_round saves before step) is what matters
        # for attribution.  The env reports current_players for the new state.
        next_players = np.array(
            [self._player_at_ply(int(p)) for p in self._ply], dtype=np.uint8
        )

        # Auto-reset
        self._ply[terminated] = 0

        return SimpleNamespace(
            observations=np.zeros((self.num_envs, 50, 9, 9), dtype=np.float32),
            legal_masks=np.ones((self.num_envs, 11259), dtype=bool),
            current_players=next_players,
            rewards=rewards,
            terminated=terminated,
            truncated=np.zeros(self.num_envs, dtype=bool),
        )


# ---------------------------------------------------------------------------
# Tests: Reward attribution correctness
# ---------------------------------------------------------------------------


class TestRewardAttribution:
    """Verify that run_round() correctly attributes wins/losses/draws."""

    def test_player_a_wins_when_a_moved_last_positive_reward(self) -> None:
        """Player A (0) moved last → reward +1 → a_wins incremented."""
        # terminate_after=1: game ends after 1 ply.
        # At ply 0 (the step that causes termination), current_players starts
        # at 0 (Black/player A always moves first in shogi).
        # pre_step_players will be [0, 0], reward = +1.0
        # So: a_moved=True, r>0 → a_wins.
        pool = _make_pool(envs_per_match=2)
        vecenv = FixedPlayerVecEnv(
            num_envs=2, terminate_after=1,
            terminal_reward=1.0, terminal_player=0,
        )
        entries = [_make_entry(1), _make_entry(2)]

        results, stats = pool.run_round(
            vecenv, [(entries[0], entries[1])],
            load_fn=lambda e: TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=2,
            max_ply=512,
        )

        assert len(results) == 1
        r = results[0]
        assert r.a_wins >= 2, f"Expected a_wins >= 2, got a={r.a_wins} b={r.b_wins} d={r.draws}"
        assert r.b_wins == 0

    def test_player_b_wins_when_b_moved_last_positive_reward(self) -> None:
        """Player B (1) moved last → reward +1 → b_wins incremented."""
        pool = _make_pool(envs_per_match=2)
        vecenv = FixedPlayerVecEnv(
            num_envs=2, terminate_after=2,
            terminal_reward=1.0, terminal_player=1,
        )
        entries = [_make_entry(1), _make_entry(2)]

        results, stats = pool.run_round(
            vecenv, [(entries[0], entries[1])],
            load_fn=lambda e: TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=2,
            max_ply=512,
        )

        assert len(results) == 1
        r = results[0]
        assert r.b_wins >= 2, f"Expected b_wins >= 2, got a={r.a_wins} b={r.b_wins} d={r.draws}"
        assert r.a_wins == 0

    def test_player_a_loses_when_a_moved_last_negative_reward(self) -> None:
        """Player A moved last → reward -1 → b_wins incremented (A lost)."""
        pool = _make_pool(envs_per_match=2)
        vecenv = FixedPlayerVecEnv(
            num_envs=2, terminate_after=1,
            terminal_reward=-1.0, terminal_player=0,
        )
        entries = [_make_entry(1), _make_entry(2)]

        results, stats = pool.run_round(
            vecenv, [(entries[0], entries[1])],
            load_fn=lambda e: TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=2,
            max_ply=512,
        )

        assert len(results) == 1
        r = results[0]
        assert r.b_wins >= 2, f"Expected b_wins >= 2 (A lost), got a={r.a_wins} b={r.b_wins} d={r.draws}"
        assert r.a_wins == 0

    def test_player_b_loses_when_b_moved_last_negative_reward(self) -> None:
        """Player B moved last → reward -1 → a_wins incremented (B lost)."""
        pool = _make_pool(envs_per_match=2)
        vecenv = FixedPlayerVecEnv(
            num_envs=2, terminate_after=2,
            terminal_reward=-1.0, terminal_player=1,
        )
        entries = [_make_entry(1), _make_entry(2)]

        results, stats = pool.run_round(
            vecenv, [(entries[0], entries[1])],
            load_fn=lambda e: TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=2,
            max_ply=512,
        )

        assert len(results) == 1
        r = results[0]
        assert r.a_wins >= 2, f"Expected a_wins >= 2 (B lost), got a={r.a_wins} b={r.b_wins} d={r.draws}"
        assert r.b_wins == 0

    def test_draw_on_zero_reward(self) -> None:
        """Reward == 0 at termination → draws incremented."""
        pool = _make_pool(envs_per_match=2)
        vecenv = FixedPlayerVecEnv(
            num_envs=2, terminate_after=1,
            terminal_reward=0.0, terminal_player=0,
        )
        entries = [_make_entry(1), _make_entry(2)]

        results, stats = pool.run_round(
            vecenv, [(entries[0], entries[1])],
            load_fn=lambda e: TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=2,
            max_ply=512,
        )

        assert len(results) == 1
        r = results[0]
        assert r.draws >= 2, f"Expected draws >= 2, got a={r.a_wins} b={r.b_wins} d={r.draws}"
        assert r.a_wins == 0
        assert r.b_wins == 0


class TestRewardAttributionWithRollout:
    """Verify rollout buffers are populated when trainable_fn returns True."""

    def test_rollout_populated_with_correct_shapes(self) -> None:
        """trainable_fn=True → rollout has obs, actions, rewards, dones, masks, perspective."""
        pool = _make_pool(envs_per_match=2)
        vecenv = RewardControlVecEnv(num_envs=2, terminate_after=3, terminal_reward=1.0)
        entries = [_make_entry(1), _make_entry(2)]

        results, _stats = pool.run_round(
            vecenv, [(entries[0], entries[1])],
            load_fn=lambda e: TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=2,
            trainable_fn=lambda a, b: True,
        )

        assert len(results) == 1
        rollout = results[0].rollout
        assert rollout is not None, "Rollout should be collected when trainable_fn=True"

        # Rollout should have multiple timesteps
        T = rollout.observations.shape[0]
        assert T > 0, "Rollout should have at least one timestep"

        # Shape checks: (timesteps, envs_per_match, ...)
        assert rollout.observations.shape == (T, 2, 50, 9, 9)
        assert rollout.actions.shape == (T, 2)
        assert rollout.rewards.shape == (T, 2)
        assert rollout.dones.shape == (T, 2)
        assert rollout.legal_masks.shape == (T, 2, 11259)
        assert rollout.perspective.shape == (T, 2)

    def test_rollout_rewards_contain_terminal_signal(self) -> None:
        """Rollout rewards should contain non-zero values at terminal steps."""
        pool = _make_pool(envs_per_match=2)
        vecenv = RewardControlVecEnv(num_envs=2, terminate_after=2, terminal_reward=1.0)
        entries = [_make_entry(1), _make_entry(2)]

        results, _stats = pool.run_round(
            vecenv, [(entries[0], entries[1])],
            load_fn=lambda e: TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=2,
            trainable_fn=lambda a, b: True,
        )

        rollout = results[0].rollout
        assert rollout is not None
        # At least some rewards should be non-zero (terminal rewards)
        assert rollout.rewards.abs().sum() > 0, "Rollout should contain terminal rewards"
        # Dones should have some True values
        assert rollout.dones.sum() > 0, "Rollout should contain done flags"

    def test_rollout_perspective_tracks_current_player(self) -> None:
        """Rollout perspective should contain both player 0 and player 1 values."""
        pool = _make_pool(envs_per_match=2)
        vecenv = RewardControlVecEnv(num_envs=2, terminate_after=4, terminal_reward=1.0)
        entries = [_make_entry(1), _make_entry(2)]

        results, _stats = pool.run_round(
            vecenv, [(entries[0], entries[1])],
            load_fn=lambda e: TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=2,
            trainable_fn=lambda a, b: True,
        )

        rollout = results[0].rollout
        assert rollout is not None
        # With alternating players, perspective should contain both 0 and 1
        unique_perspectives = rollout.perspective.unique().tolist()
        assert 0 in unique_perspectives, "Perspective should include player 0"
        # Player 1 should also appear since players alternate
        assert 1 in unique_perspectives, "Perspective should include player 1"


class TestStopEventRewardAttribution:
    """Verify that stop_event mid-round releases models without double-release."""

    def test_stop_event_mid_game_releases_models(self) -> None:
        """Setting stop_event after a few steps should release all loaded models."""
        pool = _make_pool(envs_per_match=2)
        stop_event = threading.Event()
        step_count = 0

        class StoppingEnv(RewardControlVecEnv):
            def step(self, actions: np.ndarray) -> SimpleNamespace:
                nonlocal step_count
                step_count += 1
                if step_count >= 2:
                    stop_event.set()
                return super().step(actions)

        vecenv = StoppingEnv(num_envs=2, terminate_after=100, terminal_reward=1.0)
        entries = [_make_entry(1), _make_entry(2)]
        released: list[object] = []

        results, _stats = pool.run_round(
            vecenv, [(entries[0], entries[1])],
            load_fn=lambda e: TinyModel(),
            release_fn=lambda ma, mb: (released.append(ma), released.append(mb)),
            device="cpu",
            games_per_match=1000,
            stop_event=stop_event,
        )

        assert len(results) == 0, "No games should complete when stopped early"
        assert len(released) == 2, f"Both models should be released, got {len(released)}"
