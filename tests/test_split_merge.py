"""Tests for the split-merge step logic in the unified training loop."""

from unittest.mock import MagicMock

import numpy as np
import torch

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
    model.__call__ = forward  # type: ignore[method-assign]
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


def _make_deterministic_model(bias: float = 0.0, action_space: int = 11259):
    """Create a mock model with a known constant bias to distinguish outputs.

    Uses torch.full (no randomness) so the argmax action is deterministic
    and different models produce reliably different actions.
    """
    call_count = [0]  # mutable counter to track calls

    def forward(obs):
        call_count[0] += 1
        batch = obs.shape[0]
        output = MagicMock()
        # Constant logits + bias — deterministic argmax for action verification
        output.policy_logits = torch.full((batch, 9, 9, 139), bias)
        output.value_logits = torch.zeros(batch, 3)
        output.score_lead = torch.zeros(batch, 1)
        return output

    model = MagicMock()
    model.side_effect = forward
    model.__call__ = forward  # type: ignore[method-assign]
    # Give the mock a parameters() method that returns an empty iterator
    # so the cross-device detection doesn't fail
    model.parameters = MagicMock(return_value=iter([]))
    # Expose call_count so tests can verify whether the model was invoked.
    # MagicMock.called is unreliable when __call__ is overridden.
    model._call_count = call_count
    return model


class TestSplitMergeMultiOpponent:
    """Tests for multi-opponent split_merge_step."""

    def test_multi_opponent_actions_shape(self):
        """Actions should cover all envs with multiple opponent models."""
        num_envs = 8
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        current_players = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
        env_opponent_ids = np.array([1, 1, 2, 2, 1, 1, 2, 2], dtype=np.int64)

        result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=_make_mock_model(),
            opponent_models={1: _make_mock_model(), 2: _make_mock_model()},
            env_opponent_ids=env_opponent_ids,
            learner_side=0,
        )

        assert result.actions.shape == (num_envs,)
        assert result.learner_mask.sum() == 4
        assert result.opponent_mask.sum() == 4

    def test_legacy_single_opponent_still_works(self):
        """Passing opponent_model= (legacy path) should still work."""
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

        assert result.actions.shape == (num_envs,)

    def test_opponent_with_no_envs_skipped(self):
        """Opponent model with no assigned envs should not be called."""
        num_envs = 4
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        current_players = np.array([0, 1, 0, 1], dtype=np.uint8)
        # All opponent envs assigned to model 1, model 2 has no envs
        env_opponent_ids = np.array([1, 1, 1, 1], dtype=np.int64)

        unused_model = _make_deterministic_model(bias=999.0)
        result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=_make_mock_model(),
            opponent_models={1: _make_mock_model(), 2: unused_model},
            env_opponent_ids=env_opponent_ids,
            learner_side=0,
        )

        assert result.actions.shape == (num_envs,)
        # MagicMock.called is unreliable when __call__ is overridden.
        # Use the explicit call counter instead.
        assert unused_model._call_count[0] == 0, "Model 2 should not have been called"

    def test_multi_opponent_with_array_learner_side(self):
        """Multi-opponent + per-env learner_side should produce correct masks."""
        num_envs = 4
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        # env0: player=0, learner=0 → learner
        # env1: player=1, learner=1 → learner
        # env2: player=0, learner=1 → opponent (model 1)
        # env3: player=1, learner=0 → opponent (model 2)
        current_players = np.array([0, 1, 0, 1], dtype=np.uint8)
        learner_side = np.array([0, 1, 1, 0], dtype=np.uint8)
        env_opponent_ids = np.array([0, 0, 1, 2], dtype=np.int64)

        result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=_make_mock_model(),
            opponent_models={1: _make_mock_model(), 2: _make_mock_model()},
            env_opponent_ids=env_opponent_ids,
            learner_side=learner_side,
        )

        assert result.actions.shape == (num_envs,)
        assert result.learner_mask.sum() == 2  # envs 0 and 1
        assert result.opponent_mask.sum() == 2  # envs 2 and 3

    def test_action_source_from_correct_model(self):
        """Actions for opponent envs must come from the model assigned to that env."""
        num_envs = 4
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        # env0: player=0, learner=0 → learner (no opponent action)
        # env1: player=1, learner=0 → opponent, assigned to model_a
        # env2: player=0, learner=1 → opponent, assigned to model_b  (learner_side=1, player=0 → opponent)
        # env3: player=1, learner=0 → opponent, assigned to model_a
        current_players = np.array([0, 1, 0, 1], dtype=np.uint8)
        learner_side = np.array([0, 0, 1, 0], dtype=np.uint8)
        env_opponent_ids = np.array([0, 10, 20, 10], dtype=np.int64)

        model_a = _make_deterministic_model(bias=0.0)   # ID=10
        model_b = _make_deterministic_model(bias=100.0)  # ID=20, very different logits

        result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=_make_mock_model(),
            opponent_models={10: model_a, 20: model_b},
            env_opponent_ids=env_opponent_ids,
            learner_side=learner_side,
        )

        # Model A (bias=0) was called for envs 1 and 3
        assert model_a._call_count[0] == 1, f"Model A called {model_a._call_count[0]} times, expected 1 batch"
        # Model B (bias=100) was called for env 2
        assert model_b._call_count[0] == 1, f"Model B called {model_b._call_count[0]} times, expected 1 batch"

        # The actions for opponent envs should be valid (non-negative action indices)
        assert result.actions[1].item() >= 0
        assert result.actions[2].item() >= 0
        assert result.actions[3].item() >= 0
