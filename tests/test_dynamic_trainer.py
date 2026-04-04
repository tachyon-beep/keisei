"""Tests for DynamicTrainer and MatchRollout."""

from __future__ import annotations

import torch

from keisei.training.dynamic_trainer import MatchRollout


class TestMatchRolloutConstruction:
    """test_match_rollout_construction — synthetic tensor construction."""

    def test_match_rollout_construction(self) -> None:
        steps, num_envs, obs_channels, action_space = 10, 3, 50, 11259

        observations = torch.zeros(steps, num_envs, obs_channels, 9, 9)
        actions = torch.zeros(steps, num_envs, dtype=torch.long)
        rewards = torch.zeros(steps, num_envs)
        dones = torch.zeros(steps, num_envs)
        legal_masks = torch.zeros(steps, num_envs, action_space)
        perspective = torch.zeros(steps, num_envs, dtype=torch.long)

        rollout = MatchRollout(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            legal_masks=legal_masks,
            perspective=perspective,
        )

        # All fields accessible
        assert rollout.observations is observations
        assert rollout.actions is actions
        assert rollout.rewards is rewards
        assert rollout.dones is dones
        assert rollout.legal_masks is legal_masks
        assert rollout.perspective is perspective

        # Correct shapes
        assert rollout.observations.shape == (steps, num_envs, obs_channels, 9, 9)
        assert rollout.actions.shape == (steps, num_envs)
        assert rollout.rewards.shape == (steps, num_envs)
        assert rollout.dones.shape == (steps, num_envs)
        assert rollout.legal_masks.shape == (steps, num_envs, action_space)
        assert rollout.perspective.shape == (steps, num_envs)

        # All on CPU
        assert rollout.observations.device.type == "cpu"
        assert rollout.actions.device.type == "cpu"
        assert rollout.rewards.device.type == "cpu"
        assert rollout.dones.device.type == "cpu"
        assert rollout.legal_masks.device.type == "cpu"
        assert rollout.perspective.device.type == "cpu"


class TestMatchRolloutFilterByPerspective:
    """test_match_rollout_filter_by_perspective — filter by player perspective."""

    def test_match_rollout_filter_by_perspective(self) -> None:
        steps, num_envs, obs_channels, action_space = 10, 3, 50, 11259

        observations = torch.randn(steps, num_envs, obs_channels, 9, 9)
        actions = torch.randint(0, action_space, (steps, num_envs))
        rewards = torch.randn(steps, num_envs)
        dones = torch.zeros(steps, num_envs)
        legal_masks = torch.ones(steps, num_envs, action_space)

        # Create mixed perspectives: first 6 steps player A (0), last 4 player B (1)
        perspective = torch.zeros(steps, num_envs, dtype=torch.long)
        perspective[6:, :] = 1

        rollout = MatchRollout(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            legal_masks=legal_masks,
            perspective=perspective,
        )

        # Filter where perspective == 0 (player A)
        mask = rollout.perspective == 0
        filtered_obs = rollout.observations[mask]
        filtered_actions = rollout.actions[mask]
        filtered_rewards = rollout.rewards[mask]

        # 6 steps * 3 envs = 18 entries for player A
        expected_count = 6 * num_envs
        assert filtered_obs.shape[0] == expected_count
        assert filtered_actions.shape[0] == expected_count
        assert filtered_rewards.shape[0] == expected_count

        # Verify player B count
        mask_b = rollout.perspective == 1
        filtered_b = rollout.observations[mask_b]
        expected_b = 4 * num_envs
        assert filtered_b.shape[0] == expected_b
