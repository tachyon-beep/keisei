"""Tests for split-merge pending transition logic."""

import numpy as np
import pytest
import torch

from keisei.training.katago_loop import to_learner_perspective, sign_correct_bootstrap, PendingTransitions


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


class TestPendingTransitions:
    """Test the PendingTransitions state container."""

    def _make_pending(self, num_envs=4, obs_channels=50, action_space=11259):
        return PendingTransitions(
            num_envs=num_envs,
            obs_shape=(obs_channels, 9, 9),
            action_space=action_space,
            device=torch.device("cpu"),
        )

    def test_initially_no_valid_pending(self):
        pt = self._make_pending()
        assert not pt.valid.any()

    def test_create_sets_valid_and_stores_data(self):
        pt = self._make_pending(num_envs=4)
        env_mask = torch.tensor([True, False, True, False], dtype=torch.bool)
        obs = torch.randn(4, 50, 9, 9)
        actions = torch.tensor([10, 20, 30, 40])
        log_probs = torch.tensor([0.0, 0.0, -0.5, 0.0])
        values = torch.tensor([0.0, 0.0, 0.3, 0.0])
        legal_masks = torch.ones(4, 11259, dtype=torch.bool)
        rewards = torch.tensor([0.0, 0.0, 0.0, 0.0])
        score_targets = torch.tensor([0.1, 0.0, 0.2, 0.0])

        pt.create(env_mask, obs, actions, log_probs, values,
                  legal_masks, rewards, score_targets)

        assert pt.valid[0] and pt.valid[2]
        assert not pt.valid[1] and not pt.valid[3]
        assert torch.equal(pt.actions[0], torch.tensor(10))
        assert torch.equal(pt.actions[2], torch.tensor(30))
        assert pt.log_probs[2].item() == pytest.approx(-0.5)

    def test_create_rejects_overwrite_of_valid_env(self):
        """create() must not be called on an env with an already-open pending
        transition. This catches protocol violations where finalize was skipped."""
        pt = self._make_pending(num_envs=2)
        env_mask = torch.tensor([True, False], dtype=torch.bool)
        obs = torch.randn(2, 50, 9, 9)
        actions = torch.zeros(2, dtype=torch.long)
        log_probs = torch.zeros(2)
        values = torch.zeros(2)
        legal_masks = torch.ones(2, 11259, dtype=torch.bool)
        rewards = torch.zeros(2)
        score_targets = torch.zeros(2)

        pt.create(env_mask, obs, actions, log_probs, values,
                  legal_masks, rewards, score_targets)

        # Attempting to create again on the same env should fail
        with pytest.raises(AssertionError):
            pt.create(env_mask, obs, actions, log_probs, values,
                      legal_masks, rewards, score_targets)

    def test_accumulate_adds_reward(self):
        """Rewards accumulate across steps. Initial reward=0.5, accumulated=-0.2
        gives final=0.3 for env 0. Initial=-0.5, accumulated=0.3 gives -0.2 for env 1."""
        pt = self._make_pending(num_envs=2)
        env_mask = torch.tensor([True, True], dtype=torch.bool)
        obs = torch.randn(2, 50, 9, 9)
        actions = torch.zeros(2, dtype=torch.long)
        log_probs = torch.zeros(2)
        values = torch.zeros(2)
        legal_masks = torch.ones(2, 11259, dtype=torch.bool)
        rewards = torch.tensor([0.5, -0.5])
        score_targets = torch.zeros(2)

        pt.create(env_mask, obs, actions, log_probs, values,
                  legal_masks, rewards, score_targets)
        # Accumulate opponent-turn reward
        pt.accumulate_reward(torch.tensor([-0.2, 0.3]))
        assert pt.rewards[0].item() == pytest.approx(0.3)   # 0.5 + (-0.2)
        assert pt.rewards[1].item() == pytest.approx(-0.2)  # -0.5 + 0.3

    def test_accumulate_before_create_is_noop(self):
        """accumulate_reward before any create should be a safe no-op."""
        pt = self._make_pending(num_envs=2)
        pt.accumulate_reward(torch.tensor([1.0, -1.0]))
        assert pt.rewards[0].item() == 0.0
        assert pt.rewards[1].item() == 0.0

    def test_finalize_returns_data_and_clears(self):
        pt = self._make_pending(num_envs=3)
        env_mask = torch.tensor([True, True, True], dtype=torch.bool)
        obs = torch.randn(3, 50, 9, 9)
        actions = torch.tensor([5, 10, 15])
        log_probs = torch.tensor([-0.1, -0.2, -0.3])
        values = torch.tensor([0.5, 0.6, 0.7])
        legal_masks = torch.ones(3, 11259, dtype=torch.bool)
        rewards = torch.tensor([0.0, 0.0, 0.0])
        score_targets = torch.tensor([0.1, 0.2, 0.3])

        pt.create(env_mask, obs, actions, log_probs, values,
                  legal_masks, rewards, score_targets)

        # Finalize envs 0 and 2 (terminal)
        finalize_mask = torch.tensor([True, False, True], dtype=torch.bool)
        dones = torch.tensor([True, False, True])
        result = pt.finalize(finalize_mask, dones)

        assert result is not None
        assert result["env_ids"].shape == (2,)
        assert torch.equal(result["env_ids"], torch.tensor([0, 2]))
        assert torch.equal(result["actions"], torch.tensor([5, 15]))
        assert torch.equal(result["dones"], torch.tensor([1.0, 1.0]))
        # Finalized envs are cleared
        assert not pt.valid[0] and not pt.valid[2]
        # Non-finalized env stays valid
        assert pt.valid[1]

    def test_finalize_nonterminal_sets_done_false(self):
        pt = self._make_pending(num_envs=2)
        env_mask = torch.tensor([True, True], dtype=torch.bool)
        obs = torch.randn(2, 50, 9, 9)
        actions = torch.zeros(2, dtype=torch.long)
        log_probs = torch.zeros(2)
        values = torch.zeros(2)
        legal_masks = torch.ones(2, 11259, dtype=torch.bool)
        rewards = torch.zeros(2)
        score_targets = torch.zeros(2)

        pt.create(env_mask, obs, actions, log_probs, values,
                  legal_masks, rewards, score_targets)

        finalize_mask = torch.tensor([True, False], dtype=torch.bool)
        dones = torch.tensor([False, False])
        result = pt.finalize(finalize_mask, dones)

        assert result is not None
        assert result["dones"][0].item() == 0.0

    def test_finalize_none_returns_none(self):
        pt = self._make_pending(num_envs=2)
        finalize_mask = torch.tensor([False, False], dtype=torch.bool)
        dones = torch.tensor([False, False])
        result = pt.finalize(finalize_mask, dones)
        assert result is None

    def test_finalize_mask_on_invalid_env_is_safe(self):
        """finalize_mask may include envs where valid=False. These are
        silently skipped via the `to_finalize = finalize_mask & self.valid` guard."""
        pt = self._make_pending(num_envs=2)
        # No pending created — both are invalid
        finalize_mask = torch.tensor([True, True], dtype=torch.bool)
        dones = torch.tensor([True, True])
        result = pt.finalize(finalize_mask, dones)
        assert result is None
