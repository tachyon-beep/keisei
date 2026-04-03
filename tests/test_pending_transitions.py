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


class TestSplitMergeCollection:
    """Integration test: verify buffer receives correct transitions for
    known game sequences in split-merge mode."""

    def test_opponent_terminal_reaches_buffer(self):
        """When the opponent checkmates, the learner's last transition must
        appear in the buffer with done=True and a negative reward."""
        num_envs = 1
        obs_shape = (50, 9, 9)
        action_space = 11259
        device = torch.device("cpu")
        learner_side = 0

        pt = PendingTransitions(num_envs, obs_shape, action_space, device)

        # --- Step 1: Learner moves, game continues ---
        pre_players_1 = np.array([0], dtype=np.uint8)  # learner to move
        obs_1 = torch.randn(1, *obs_shape)
        actions_1 = torch.tensor([42])
        log_probs_full = torch.tensor([-0.5])
        values_full = torch.tensor([0.3])
        legal_masks_1 = torch.ones(1, action_space, dtype=torch.bool)
        rewards_1 = torch.tensor([0.0])  # non-terminal
        dones_1 = torch.tensor([False])
        score_targets_1 = torch.tensor([0.1])
        current_players_after_1 = np.array([1], dtype=np.uint8)  # opponent next

        learner_rewards_1 = to_learner_perspective(rewards_1, pre_players_1, learner_side)

        # Accumulate (no pending yet, so no-op)
        pt.accumulate_reward(learner_rewards_1)
        # Finalize (nothing to finalize)
        finalize_mask = pt.valid & (
            dones_1.bool()
            | torch.tensor(current_players_after_1 == learner_side, dtype=torch.bool)
        )
        result = pt.finalize(finalize_mask, dones_1)
        assert result is None

        # Create pending for learner's move
        learner_moved = torch.tensor(pre_players_1 == learner_side, dtype=torch.bool)
        pt.create(learner_moved, obs_1, actions_1, log_probs_full, values_full,
                  legal_masks_1, learner_rewards_1, score_targets_1)
        assert pt.valid[0]

        # Check immediate terminal (learner moved + terminal)
        imm_terminal = learner_moved & dones_1.bool()
        assert not imm_terminal.any()

        # --- Step 2: Opponent moves, checkmates learner ---
        pre_players_2 = np.array([1], dtype=np.uint8)  # opponent to move
        rewards_2 = torch.tensor([1.0])  # opponent won, from opponent POV
        dones_2 = torch.tensor([True])   # game over
        current_players_after_2 = np.array([0], dtype=np.uint8)  # reset to start

        learner_rewards_2 = to_learner_perspective(rewards_2, pre_players_2, learner_side)
        assert learner_rewards_2[0].item() == -1.0  # negated for learner

        # Accumulate
        pt.accumulate_reward(learner_rewards_2)
        assert pt.rewards[0].item() == pytest.approx(-1.0)

        # Finalize (terminal)
        finalize_mask = pt.valid & (
            dones_2.bool()
            | torch.tensor(current_players_after_2 == learner_side, dtype=torch.bool)
        )
        result = pt.finalize(finalize_mask, dones_2)

        assert result is not None
        assert result["env_ids"].tolist() == [0]
        assert result["rewards"][0].item() == pytest.approx(-1.0)
        assert result["dones"][0].item() == 1.0
        assert result["actions"][0].item() == 42
        assert result["values"][0].item() == pytest.approx(0.3)
        assert not pt.valid[0]

    def test_learner_terminal_finalized_immediately(self):
        """When the learner checkmates, the pending transition is created
        and immediately finalized in the same step."""
        num_envs = 1
        obs_shape = (50, 9, 9)
        action_space = 11259
        device = torch.device("cpu")
        learner_side = 0

        pt = PendingTransitions(num_envs, obs_shape, action_space, device)

        pre_players = np.array([0], dtype=np.uint8)
        obs = torch.randn(1, *obs_shape)
        actions = torch.tensor([99])
        log_probs_full = torch.tensor([-0.1])
        values_full = torch.tensor([0.8])
        legal_masks = torch.ones(1, action_space, dtype=torch.bool)
        rewards = torch.tensor([1.0])  # learner won, from learner POV
        dones = torch.tensor([True])
        score_targets = torch.tensor([0.5])
        current_players_after = np.array([0], dtype=np.uint8)  # reset

        learner_rewards = to_learner_perspective(rewards, pre_players, learner_side)

        # Accumulate (no pending yet)
        pt.accumulate_reward(learner_rewards)

        # Finalize existing (none)
        finalize_mask = pt.valid & (
            dones.bool()
            | torch.tensor(current_players_after == learner_side, dtype=torch.bool)
        )
        result = pt.finalize(finalize_mask, dones)
        assert result is None

        # Create pending
        learner_moved = torch.tensor(pre_players == learner_side, dtype=torch.bool)
        pt.create(learner_moved, obs, actions, log_probs_full, values_full,
                  legal_masks, learner_rewards, score_targets)
        assert pt.valid[0]

        # Immediate terminal finalize
        imm_terminal = learner_moved & dones.bool()
        result = pt.finalize(imm_terminal, dones)

        assert result is not None
        assert result["rewards"][0].item() == pytest.approx(1.0)
        assert result["dones"][0].item() == 1.0
        assert not pt.valid[0]

    def test_nonterminal_finalized_when_turn_returns(self):
        """Non-terminal transitions are finalized when the turn returns
        to the learner (opponent moved, game continues)."""
        num_envs = 1
        obs_shape = (50, 9, 9)
        action_space = 11259
        device = torch.device("cpu")
        learner_side = 0

        pt = PendingTransitions(num_envs, obs_shape, action_space, device)

        # Step 1: Learner moves
        pre_players_1 = np.array([0], dtype=np.uint8)
        obs_1 = torch.randn(1, *obs_shape)
        actions_1 = torch.tensor([7])
        log_probs_1 = torch.tensor([-0.3])
        values_1 = torch.tensor([0.2])
        legal_masks_1 = torch.ones(1, action_space, dtype=torch.bool)
        rewards_1 = torch.tensor([0.0])
        dones_1 = torch.tensor([False])
        score_targets_1 = torch.tensor([0.0])
        current_players_after_1 = np.array([1], dtype=np.uint8)

        learner_rewards_1 = to_learner_perspective(rewards_1, pre_players_1, learner_side)
        pt.accumulate_reward(learner_rewards_1)
        pt.finalize(
            pt.valid & (
                dones_1.bool()
                | torch.tensor(current_players_after_1 == learner_side, dtype=torch.bool)
            ),
            dones_1,
        )
        learner_moved = torch.tensor(pre_players_1 == learner_side, dtype=torch.bool)
        pt.create(learner_moved, obs_1, actions_1, log_probs_1, values_1,
                  legal_masks_1, learner_rewards_1, score_targets_1)

        # Step 2: Opponent moves, game continues
        pre_players_2 = np.array([1], dtype=np.uint8)
        rewards_2 = torch.tensor([0.0])
        dones_2 = torch.tensor([False])
        current_players_after_2 = np.array([0], dtype=np.uint8)  # back to learner

        learner_rewards_2 = to_learner_perspective(rewards_2, pre_players_2, learner_side)
        pt.accumulate_reward(learner_rewards_2)

        # Finalize: non-terminal, turn returns to learner
        finalize_mask = pt.valid & (
            dones_2.bool()
            | torch.tensor(current_players_after_2 == learner_side, dtype=torch.bool)
        )
        result = pt.finalize(finalize_mask, dones_2)

        assert result is not None
        assert result["dones"][0].item() == 0.0
        assert result["rewards"][0].item() == pytest.approx(0.0)
        assert not pt.valid[0]

    def test_multi_env_heterogeneous_terminal(self):
        """In the same step, env 0 has an opponent-turn terminal while
        env 1 has a non-terminal opponent move. Both must be handled correctly."""
        num_envs = 2
        obs_shape = (50, 9, 9)
        action_space = 11259
        device = torch.device("cpu")
        learner_side = 0

        pt = PendingTransitions(num_envs, obs_shape, action_space, device)

        # Step 1: Both envs have learner to move
        pre_players_1 = np.array([0, 0], dtype=np.uint8)
        obs_1 = torch.randn(2, *obs_shape)
        actions_1 = torch.tensor([10, 20])
        log_probs_1 = torch.tensor([-0.1, -0.2])
        values_1 = torch.tensor([0.5, 0.6])
        legal_masks_1 = torch.ones(2, action_space, dtype=torch.bool)
        rewards_1 = torch.tensor([0.0, 0.0])
        dones_1 = torch.tensor([False, False])
        score_targets_1 = torch.tensor([0.1, 0.2])
        current_players_after_1 = np.array([1, 1], dtype=np.uint8)

        learner_rewards_1 = to_learner_perspective(rewards_1, pre_players_1, learner_side)
        pt.accumulate_reward(learner_rewards_1)
        pt.finalize(
            pt.valid & (
                dones_1.bool()
                | torch.tensor(current_players_after_1 == learner_side, dtype=torch.bool)
            ),
            dones_1,
        )
        learner_moved = torch.tensor(pre_players_1 == learner_side, dtype=torch.bool)
        pt.create(learner_moved, obs_1, actions_1, log_probs_1, values_1,
                  legal_masks_1, learner_rewards_1, score_targets_1)

        # Step 2: Both envs have opponent to move
        # Env 0: opponent checkmates (terminal)
        # Env 1: opponent moves, game continues
        pre_players_2 = np.array([1, 1], dtype=np.uint8)
        rewards_2 = torch.tensor([1.0, 0.0])  # env 0: opponent won; env 1: non-terminal
        dones_2 = torch.tensor([True, False])
        current_players_after_2 = np.array([0, 0], dtype=np.uint8)  # env 0: reset; env 1: back to learner

        learner_rewards_2 = to_learner_perspective(rewards_2, pre_players_2, learner_side)
        pt.accumulate_reward(learner_rewards_2)

        finalize_mask = pt.valid & (
            dones_2.bool()
            | torch.tensor(current_players_after_2 == learner_side, dtype=torch.bool)
        )
        result = pt.finalize(finalize_mask, dones_2)

        assert result is not None
        assert result["env_ids"].tolist() == [0, 1]
        # Env 0: terminal loss — reward = -1.0, done = 1.0
        assert result["rewards"][0].item() == pytest.approx(-1.0)
        assert result["dones"][0].item() == 1.0
        assert result["actions"][0].item() == 10
        # Env 1: non-terminal — reward = 0.0, done = 0.0
        assert result["rewards"][1].item() == pytest.approx(0.0)
        assert result["dones"][1].item() == 0.0
        assert result["actions"][1].item() == 20

        # Both cleared
        assert not pt.valid.any()

    def test_epoch_end_flush(self):
        """Pending transitions remaining at epoch end are finalized with
        done=False and value_cat=-1 (non-terminal bootstrap)."""
        num_envs = 2
        obs_shape = (50, 9, 9)
        action_space = 11259
        device = torch.device("cpu")
        learner_side = 0

        pt = PendingTransitions(num_envs, obs_shape, action_space, device)

        # Step 1: Both envs have learner to move
        pre_players = np.array([0, 0], dtype=np.uint8)
        obs = torch.randn(2, *obs_shape)
        actions = torch.tensor([5, 6])
        log_probs = torch.tensor([-0.1, -0.2])
        values = torch.tensor([0.4, 0.5])
        legal_masks = torch.ones(2, action_space, dtype=torch.bool)
        rewards = torch.tensor([0.0, 0.0])
        score_targets = torch.tensor([0.0, 0.0])

        learner_rewards = to_learner_perspective(rewards, pre_players, learner_side)
        learner_moved = torch.tensor(pre_players == learner_side, dtype=torch.bool)
        pt.create(learner_moved, obs, actions, log_probs, values,
                  legal_masks, learner_rewards, score_targets)

        # Epoch ends — no more steps. Flush remaining pending.
        remaining_mask = pt.valid.clone()
        remaining_dones = torch.zeros(num_envs)
        result = pt.finalize(remaining_mask, remaining_dones)

        assert result is not None
        assert result["env_ids"].tolist() == [0, 1]
        # All flushed as non-terminal
        assert result["dones"][0].item() == 0.0
        assert result["dones"][1].item() == 0.0
        assert result["values"][0].item() == pytest.approx(0.4)
        assert not pt.valid.any()
