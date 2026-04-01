"""Tests for the split-merge step logic in the unified training loop."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from keisei.training.katago_loop import split_merge_step


def _make_mock_model(action_space: int = 11259):
    """Create a mock model that returns deterministic actions."""
    model = MagicMock()

    def forward(obs):
        batch = obs.shape[0]
        output = MagicMock()
        output.policy_logits = torch.randn(batch, 9, 9, 139)
        output.value_logits = torch.randn(batch, 3)
        output.score_lead = torch.randn(batch, 1)
        return output

    model.side_effect = forward
    model.__call__ = forward
    return model


class TestSplitMergeStep:
    def test_actions_shape_matches_num_envs(self):
        """Merged actions should cover all environments."""
        num_envs = 8
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        current_players = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)

        result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=_make_mock_model(),
            opponent_model=_make_mock_model(),
            learner_side=0,
        )

        assert result.actions.shape == (num_envs,)
        assert result.learner_mask.sum() == 4
        assert result.opponent_mask.sum() == 4

    def test_learner_actions_have_log_probs(self):
        """Learner actions should come with log_probs and values for the buffer."""
        num_envs = 4
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        current_players = np.array([0, 1, 0, 1], dtype=np.uint8)

        result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=_make_mock_model(),
            opponent_model=_make_mock_model(),
            learner_side=0,
        )

        assert result.learner_log_probs.shape == (2,)
        assert result.learner_values.shape == (2,)

    def test_all_learner_envs(self):
        """When all envs are learner's turn, opponent model should not be called."""
        num_envs = 4
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        current_players = np.array([0, 0, 0, 0], dtype=np.uint8)

        result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=_make_mock_model(),
            opponent_model=_make_mock_model(),
            learner_side=0,
        )

        assert result.actions.shape == (num_envs,)
        assert result.learner_mask.all()

    def test_all_opponent_envs(self):
        """When all envs are opponent's turn, learner model should not be called."""
        num_envs = 4
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        current_players = np.array([1, 1, 1, 1], dtype=np.uint8)

        result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=_make_mock_model(),
            opponent_model=_make_mock_model(),
            learner_side=0,
        )

        assert result.actions.shape == (num_envs,)
        assert result.opponent_mask.all()
        assert result.learner_log_probs.shape == (0,)
