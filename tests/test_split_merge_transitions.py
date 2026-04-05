"""Tests for split-merge pending transition logic."""

import numpy as np
import pytest
import torch

from keisei.training.katago_loop import (
    PendingTransitions,
    _compute_value_cats,
    sign_correct_bootstrap,
    to_learner_perspective,
)


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
        with pytest.raises(RuntimeError):
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
        result = pt.finalize(finalize_mask, dones, dones)

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
        result = pt.finalize(finalize_mask, dones, dones)

        assert result is not None
        assert result["dones"][0].item() == 0.0

    def test_finalize_none_returns_none(self):
        pt = self._make_pending(num_envs=2)
        finalize_mask = torch.tensor([False, False], dtype=torch.bool)
        dones = torch.tensor([False, False])
        result = pt.finalize(finalize_mask, dones, dones)
        assert result is None

    def test_finalize_mask_on_invalid_env_is_safe(self):
        """finalize_mask may include envs where valid=False. These are
        silently skipped via the `to_finalize = finalize_mask & self.valid` guard."""
        pt = self._make_pending(num_envs=2)
        # No pending created — both are invalid
        finalize_mask = torch.tensor([True, True], dtype=torch.bool)
        dones = torch.tensor([True, True])
        result = pt.finalize(finalize_mask, dones, dones)
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
        result = pt.finalize(finalize_mask, dones_1, dones_1)
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
        result = pt.finalize(finalize_mask, dones_2, dones_2)

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
        result = pt.finalize(finalize_mask, dones, dones)
        assert result is None

        # Create pending
        learner_moved = torch.tensor(pre_players == learner_side, dtype=torch.bool)
        pt.create(learner_moved, obs, actions, log_probs_full, values_full,
                  legal_masks, learner_rewards, score_targets)
        assert pt.valid[0]

        # Immediate terminal finalize
        imm_terminal = learner_moved & dones.bool()
        result = pt.finalize(imm_terminal, dones, dones)

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
        result = pt.finalize(finalize_mask, dones_2, dones_2)

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
        result = pt.finalize(finalize_mask, dones_2, dones_2)

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
        result = pt.finalize(remaining_mask, remaining_dones, remaining_dones)

        assert result is not None
        assert result["env_ids"].tolist() == [0, 1]
        # All flushed as non-terminal
        assert result["dones"][0].item() == 0.0
        assert result["dones"][1].item() == 0.0
        assert result["values"][0].item() == pytest.approx(0.4)
        assert not pt.valid.any()


class TestLearnerSideOne:
    """Verify perspective correction works when learner plays White (side 1)."""

    def test_to_learner_perspective_side_1(self):
        """When learner_side=1, opponent is side 0. Rewards from side 0
        moves should be negated."""
        rewards = torch.tensor([1.0, -1.0, 0.0])
        pre_players = np.array([0, 1, 0], dtype=np.uint8)
        result = to_learner_perspective(rewards, pre_players, learner_side=1)
        # Side 0 moved in envs 0,2 → opponent moved → negate
        # Side 1 moved in env 1 → learner moved → keep
        expected = torch.tensor([-1.0, -1.0, 0.0])
        assert torch.equal(result, expected)

    def test_sign_correct_bootstrap_side_1(self):
        """When learner_side=1, negate bootstrap for envs where side 0 is to-move."""
        next_values = torch.tensor([0.5, -0.5])
        current_players = np.array([0, 1], dtype=np.uint8)
        result = sign_correct_bootstrap(next_values, current_players, learner_side=1)
        # Env 0: side 0 to move = opponent → negate
        # Env 1: side 1 to move = learner → keep
        expected = torch.tensor([-0.5, -0.5])
        assert torch.equal(result, expected)

    def test_opponent_terminal_side_1(self):
        """Full protocol with learner_side=1: opponent (side 0) checkmates,
        learner (side 1) should see done=True with negative reward."""
        num_envs = 1
        obs_shape = (50, 9, 9)
        action_space = 11259
        device = torch.device("cpu")
        learner_side = 1

        pt = PendingTransitions(num_envs, obs_shape, action_space, device)

        # Step 1: Learner (side 1) moves, game continues
        pre_players_1 = np.array([1], dtype=np.uint8)
        obs_1 = torch.randn(1, *obs_shape)
        actions_1 = torch.tensor([42])
        log_probs_1 = torch.tensor([-0.5])
        values_1 = torch.tensor([0.3])
        legal_masks_1 = torch.ones(1, action_space, dtype=torch.bool)
        rewards_1 = torch.tensor([0.0])
        dones_1 = torch.tensor([False])
        score_targets_1 = torch.tensor([0.1])
        current_players_after_1 = np.array([0], dtype=np.uint8)

        learner_rewards_1 = to_learner_perspective(rewards_1, pre_players_1, learner_side)
        pt.accumulate_reward(learner_rewards_1)
        pt.finalize(
            pt.valid & (
                dones_1.bool()
                | torch.tensor(current_players_after_1 == learner_side, dtype=torch.bool)
            ),
            dones_1,
            dones_1,
        )
        learner_moved = torch.tensor(pre_players_1 == learner_side, dtype=torch.bool)
        pt.create(learner_moved, obs_1, actions_1, log_probs_1, values_1,
                  legal_masks_1, learner_rewards_1, score_targets_1)

        # Step 2: Opponent (side 0) checkmates
        pre_players_2 = np.array([0], dtype=np.uint8)
        rewards_2 = torch.tensor([1.0])  # from side 0 (opponent) POV
        dones_2 = torch.tensor([True])
        current_players_after_2 = np.array([1], dtype=np.uint8)

        learner_rewards_2 = to_learner_perspective(rewards_2, pre_players_2, learner_side)
        assert learner_rewards_2[0].item() == -1.0

        pt.accumulate_reward(learner_rewards_2)
        finalize_mask = pt.valid & (
            dones_2.bool()
            | torch.tensor(current_players_after_2 == learner_side, dtype=torch.bool)
        )
        result = pt.finalize(finalize_mask, dones_2, dones_2)

        assert result is not None
        assert result["rewards"][0].item() == pytest.approx(-1.0)
        assert result["dones"][0].item() == 1.0


class TestDrawTerminal:
    """Verify draw outcomes are handled correctly at the protocol level."""

    def test_draw_terminal_produces_value_cat_1(self):
        """A terminal draw (done=True, reward=0.0) should produce value_cat=1."""
        rewards = torch.tensor([0.0, 1.0, -1.0])
        dones_bool = torch.tensor([True, True, True])
        cats = _compute_value_cats(rewards, dones_bool, torch.device("cpu"))
        assert cats[0].item() == 1  # draw
        assert cats[1].item() == 0  # win
        assert cats[2].item() == 2  # loss

    def test_draw_terminal_in_protocol(self):
        """A game ending in a draw (reward=0.0 from both perspectives)
        should finalize with done=True and reward=0.0."""
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
            dones_1,
        )
        learner_moved = torch.tensor(pre_players_1 == learner_side, dtype=torch.bool)
        pt.create(learner_moved, obs_1, actions_1, log_probs_1, values_1,
                  legal_masks_1, learner_rewards_1, score_targets_1)

        # Step 2: Opponent moves, game draws (e.g. repetition)
        pre_players_2 = np.array([1], dtype=np.uint8)
        rewards_2 = torch.tensor([0.0])  # draw from opponent POV
        dones_2 = torch.tensor([True])
        current_players_after_2 = np.array([0], dtype=np.uint8)

        learner_rewards_2 = to_learner_perspective(rewards_2, pre_players_2, learner_side)
        assert learner_rewards_2[0].item() == 0.0  # negating 0 is still 0

        pt.accumulate_reward(learner_rewards_2)
        finalize_mask = pt.valid & (
            dones_2.bool()
            | torch.tensor(current_players_after_2 == learner_side, dtype=torch.bool)
        )
        result = pt.finalize(finalize_mask, dones_2, dones_2)

        assert result is not None
        assert result["rewards"][0].item() == pytest.approx(0.0)
        assert result["dones"][0].item() == 1.0


class TestComputeValueCatsTerminated:
    """R1: _compute_value_cats must use terminated, not dones."""

    def test_truncated_gets_draw_under_dones_but_ignore_under_terminated(self):
        from keisei.training.katago_loop import _compute_value_cats
        device = torch.device("cpu")
        rewards = torch.tensor([0.0, 1.0, 0.0])

        # With merged dones: truncated game (reward=0) gets labeled as draw — WRONG
        dones_merged = torch.tensor([True, True, False])
        cats_wrong = _compute_value_cats(rewards, dones_merged, device)
        assert cats_wrong[0].item() == 1  # WRONG: labeled as draw

        # With terminated only: truncated game (not terminated) gets ignored — CORRECT
        terminated = torch.tensor([False, True, False])
        cats_correct = _compute_value_cats(rewards, terminated, device)
        assert cats_correct[0].item() == -1  # CORRECT: ignored
        assert cats_correct[1].item() == 0   # win
        assert cats_correct[2].item() == -1  # still playing


class TestPendingTransitionsTerminated:
    """R1: finalize() must return terminated separately from dones."""

    def test_finalize_three_populations(self):
        from keisei.training.katago_loop import PendingTransitions
        device = torch.device("cpu")
        pt = PendingTransitions(num_envs=3, obs_shape=(4, 9, 9), action_space=100, device=device)

        env_mask = torch.ones(3, dtype=torch.bool, device=device)
        pt.create(
            env_mask,
            obs=torch.zeros(3, 4, 9, 9, device=device),
            actions=torch.zeros(3, dtype=torch.long, device=device),
            log_probs=torch.zeros(3, device=device),
            values=torch.zeros(3, device=device),
            legal_masks=torch.zeros(3, 100, dtype=torch.bool, device=device),
            rewards=torch.zeros(3, device=device),
            score_targets=torch.zeros(3, device=device),
        )

        # A: terminated (game ended), B: truncated (max_ply), C: epoch flush
        dones = torch.tensor([1.0, 1.0, 0.0], device=device)
        terminated = torch.tensor([1.0, 0.0, 0.0], device=device)

        finalize_mask = torch.ones(3, dtype=torch.bool, device=device)
        result = pt.finalize(finalize_mask, dones, terminated)

        assert result is not None
        assert "terminated" in result
        assert result["terminated"][0].item() == 1.0  # A: terminated
        assert result["terminated"][1].item() == 0.0  # B: truncated
        assert result["terminated"][2].item() == 0.0  # C: flush
        assert result["dones"][0].item() == 1.0
        assert result["dones"][1].item() == 1.0
        assert result["dones"][2].item() == 0.0


class TestToLearnerPerspectiveArray:
    """Test to_learner_perspective with per-env ndarray learner_side."""

    def test_mixed_learner_sides(self):
        """Each env uses its own learner_side for perspective correction."""
        rewards = torch.tensor([1.0, 1.0, -1.0, -1.0])
        pre_players = np.array([0, 1, 0, 1], dtype=np.uint8)
        # env0: learner=0, pre=0 → learner moved → unchanged
        # env1: learner=1, pre=1 → learner moved → unchanged
        # env2: learner=1, pre=0 → opponent moved → negated
        # env3: learner=0, pre=1 → opponent moved → negated
        learner_side = np.array([0, 1, 1, 0], dtype=np.uint8)
        result = to_learner_perspective(rewards, pre_players, learner_side)
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0])
        assert torch.equal(result, expected)

    def test_all_same_side_matches_scalar(self):
        """Array of identical values should match scalar behavior."""
        rewards = torch.tensor([1.0, -1.0, 0.5])
        pre_players = np.array([0, 1, 0], dtype=np.uint8)
        learner_side_scalar = 0
        learner_side_array = np.array([0, 0, 0], dtype=np.uint8)
        result_scalar = to_learner_perspective(rewards, pre_players, learner_side_scalar)
        result_array = to_learner_perspective(rewards, pre_players, learner_side_array)
        assert torch.equal(result_scalar, result_array)


class TestSignCorrectBootstrapArray:
    """Test sign_correct_bootstrap with per-env ndarray learner_side."""

    def test_mixed_learner_sides(self):
        """Each env uses its own learner_side for sign correction."""
        next_values = torch.tensor([0.5, 0.5, 0.5, 0.5])
        current_players = np.array([0, 1, 0, 1], dtype=np.uint8)
        # env0: learner=0, current=0 → learner to move → unchanged
        # env1: learner=0, current=1 → opponent to move → negated
        # env2: learner=1, current=0 → opponent to move → negated
        # env3: learner=1, current=1 → learner to move → unchanged
        learner_side = np.array([0, 0, 1, 1], dtype=np.uint8)
        result = sign_correct_bootstrap(next_values, current_players, learner_side)
        expected = torch.tensor([0.5, -0.5, -0.5, 0.5])
        assert torch.equal(result, expected)

    def test_all_same_side_matches_scalar(self):
        """Array of identical values should match scalar behavior."""
        next_values = torch.tensor([0.5, -0.3, 0.8])
        current_players = np.array([0, 1, 0], dtype=np.uint8)
        result_scalar = sign_correct_bootstrap(next_values, current_players, learner_side=0)
        result_array = sign_correct_bootstrap(
            next_values, current_players,
            learner_side=np.array([0, 0, 0], dtype=np.uint8),
        )
        assert torch.equal(result_scalar, result_array)


class TestColorReRandomizationInvariant:
    """Verify re-randomization timing invariant for sign_correct_bootstrap."""

    def test_non_done_envs_unchanged_after_rerandomization(self):
        """learner_side for non-done envs must not change during re-randomization."""
        learner_side = np.array([0, 1, 0, 1], dtype=np.uint8)
        original = learner_side.copy()

        # Simulate dones: envs 0 and 2 completed, envs 1 and 3 still mid-game
        done_np = np.array([True, False, True, False])

        # Re-randomize only done envs (same logic as katago_loop.py)
        new_sides = np.random.randint(0, 2, size=int(done_np.sum()), dtype=np.uint8)
        learner_side[done_np] = new_sides

        # Non-done envs must be unchanged — their games are still in progress
        assert learner_side[1] == original[1], "Non-done env 1 should be unchanged"
        assert learner_side[3] == original[3], "Non-done env 3 should be unchanged"

    def test_sign_correct_bootstrap_uses_current_game_color(self):
        """At epoch end, sign_correct_bootstrap should use the color of the
        currently-running (truncated) game, not the completed game's color.

        For done envs that auto-reset, learner_side reflects the NEXT game's
        color — but their bootstrap values are irrelevant (terminal episodes
        don't need bootstrap correction in PPO).

        For non-done (truncated) envs, learner_side still reflects the
        in-progress game's color — which is correct for bootstrap.
        """
        # 4 envs: envs 0,2 are mid-game (truncated), envs 1,3 completed (done)
        next_values = torch.tensor([0.5, 0.5, 0.5, 0.5])
        # current_players after step: mid-game envs have their game state,
        # done envs have reset to player 0
        current_players = np.array([1, 0, 0, 0], dtype=np.uint8)

        # Pre-re-randomization: learner_side reflects current games
        learner_side_before = np.array([0, 1, 1, 0], dtype=np.uint8)

        # Simulate re-randomization for done envs only
        done_np = np.array([False, True, False, True])
        learner_side = learner_side_before.copy()
        learner_side[done_np] = np.array([1, 1], dtype=np.uint8)  # new random sides

        # sign_correct_bootstrap for truncated envs (0 and 2):
        # env0: learner=0, current=1 → opponent to move → negate
        # env2: learner=1, current=0 → opponent to move → negate
        # These are UNCHANGED by re-randomization — correct for bootstrap.
        result = sign_correct_bootstrap(next_values, current_players, learner_side)

        # Truncated envs: sign correction uses their original (unchanged) learner_side
        assert result[0].item() == -0.5  # env0: learner=0, current=1 → negated
        assert result[2].item() == -0.5  # env2: learner=1, current=0 → negated
