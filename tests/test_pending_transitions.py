"""Tests for split-merge pending transition logic."""

import numpy as np
import pytest
import torch

from keisei.training.katago_loop import to_learner_perspective, sign_correct_bootstrap


class TestToLearnerPerspective:
    def test_learner_move_reward_unchanged(self):
        """Reward from learner's own move stays positive."""
        rewards = torch.tensor([1.0, 0.0, -1.0])
        pre_players = np.array([0, 0, 0], dtype=np.uint8)
        result = to_learner_perspective(rewards, pre_players, learner_side=0)
        assert torch.equal(result, torch.tensor([1.0, 0.0, -1.0]))

    def test_opponent_move_reward_negated(self):
        """Reward from opponent's move is negated for learner perspective."""
        rewards = torch.tensor([1.0, 0.0, -1.0])
        pre_players = np.array([1, 1, 1], dtype=np.uint8)
        result = to_learner_perspective(rewards, pre_players, learner_side=0)
        assert torch.equal(result, torch.tensor([-1.0, 0.0, 1.0]))

    def test_mixed_turns(self):
        """Mixed learner/opponent turns apply selective negation."""
        rewards = torch.tensor([1.0, 1.0, 0.5, -0.5])
        pre_players = np.array([0, 1, 0, 1], dtype=np.uint8)
        result = to_learner_perspective(rewards, pre_players, learner_side=0)
        expected = torch.tensor([1.0, -1.0, 0.5, 0.5])
        assert torch.equal(result, expected)

    def test_does_not_mutate_input(self):
        """Input tensor must not be modified in place."""
        rewards = torch.tensor([1.0, -1.0])
        original = rewards.clone()
        pre_players = np.array([1, 0], dtype=np.uint8)
        to_learner_perspective(rewards, pre_players, learner_side=0)
        assert torch.equal(rewards, original)

    def test_empty_tensor(self):
        """Handles zero-env edge case without error."""
        rewards = torch.tensor([])
        pre_players = np.array([], dtype=np.uint8)
        result = to_learner_perspective(rewards, pre_players, learner_side=0)
        assert result.numel() == 0


class TestSignCorrectBootstrap:
    def test_learner_to_move_unchanged(self):
        """Bootstrap stays positive when learner is to move."""
        next_values = torch.tensor([0.5, -0.3, 0.8])
        current_players = np.array([0, 0, 0], dtype=np.uint8)
        result = sign_correct_bootstrap(next_values, current_players, learner_side=0)
        assert torch.equal(result, next_values)

    def test_opponent_to_move_negated(self):
        """Bootstrap negated when opponent is to move (value is from opponent POV)."""
        next_values = torch.tensor([0.5, -0.3, 0.8])
        current_players = np.array([1, 1, 1], dtype=np.uint8)
        result = sign_correct_bootstrap(next_values, current_players, learner_side=0)
        assert torch.equal(result, torch.tensor([-0.5, 0.3, -0.8]))

    def test_mixed_perspective(self):
        """Mixed to-move states: only opponent-to-move envs are negated."""
        next_values = torch.tensor([0.5, 0.5, 0.5, 0.5])
        current_players = np.array([0, 1, 0, 1], dtype=np.uint8)
        result = sign_correct_bootstrap(next_values, current_players, learner_side=0)
        expected = torch.tensor([0.5, -0.5, 0.5, -0.5])
        assert torch.equal(result, expected)

    def test_does_not_mutate_input(self):
        """Input tensor must not be modified in place."""
        next_values = torch.tensor([0.5, -0.3])
        original = next_values.clone()
        current_players = np.array([1, 0], dtype=np.uint8)
        sign_correct_bootstrap(next_values, current_players, learner_side=0)
        assert torch.equal(next_values, original)

    def test_empty_tensor(self):
        """Handles zero-env edge case without error."""
        next_values = torch.tensor([])
        current_players = np.array([], dtype=np.uint8)
        result = sign_correct_bootstrap(next_values, current_players, learner_side=0)
        assert result.numel() == 0
