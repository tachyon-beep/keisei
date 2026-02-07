"""
Test to verify the base actor-critic refactoring works correctly.
"""

import pytest
import torch

from keisei.core.base_actor_critic import BaseActorCriticModel
from keisei.core.neural_network import ActorCritic
from keisei.training.models.resnet_tower import ActorCriticResTower


def test_both_models_inherit_from_base():
    """Test that both ActorCritic and ActorCriticResTower inherit from BaseActorCriticModel."""
    model1 = ActorCritic(input_channels=46, num_actions_total=100)
    model2 = ActorCriticResTower(
        input_channels=46, num_actions_total=100, tower_depth=2, tower_width=32
    )

    assert isinstance(model1, BaseActorCriticModel)
    assert isinstance(model2, BaseActorCriticModel)


def test_shared_methods_work_identically():
    """Test that both models provide the same interface through shared methods."""
    model1 = ActorCritic(input_channels=46, num_actions_total=100)
    model2 = ActorCriticResTower(
        input_channels=46, num_actions_total=100, tower_depth=2, tower_width=32
    )

    obs = torch.randn(1, 46, 9, 9)
    legal_mask = torch.ones(1, 100, dtype=torch.bool)
    actions = torch.randint(0, 100, (1,))

    # Test get_action_and_value
    action1, log_prob1, value1 = model1.get_action_and_value(obs, legal_mask)
    action2, log_prob2, value2 = model2.get_action_and_value(obs, legal_mask)

    # Test shapes are consistent
    assert action1.shape == action2.shape
    assert log_prob1.shape == log_prob2.shape
    assert value1.shape == value2.shape

    # Test evaluate_actions
    log_probs1, entropy1, value1_eval = model1.evaluate_actions(
        obs, actions, legal_mask
    )
    log_probs2, entropy2, value2_eval = model2.evaluate_actions(
        obs, actions, legal_mask
    )

    # Test shapes are consistent
    assert log_probs1.shape == log_probs2.shape
    assert entropy1.shape == entropy2.shape
    assert value1_eval.shape == value2_eval.shape


def test_deterministic_mode():
    """Test that deterministic mode works for both models."""
    model1 = ActorCritic(input_channels=46, num_actions_total=100)
    model2 = ActorCriticResTower(
        input_channels=46, num_actions_total=100, tower_depth=2, tower_width=32
    )

    obs = torch.randn(1, 46, 9, 9)

    # Test deterministic mode produces same action for same input
    action1_det, _, _ = model1.get_action_and_value(obs, deterministic=True)
    action1_det2, _, _ = model1.get_action_and_value(obs, deterministic=True)

    action2_det, _, _ = model2.get_action_and_value(obs, deterministic=True)
    action2_det2, _, _ = model2.get_action_and_value(obs, deterministic=True)

    assert torch.equal(action1_det, action1_det2)
    assert torch.equal(action2_det, action2_det2)


def test_get_action_all_masked_returns_safe_dummy():
    """Regression test for P2-07: all-masked legal_mask must not produce NaN
    or a distribution over illegal actions."""
    model = ActorCritic(input_channels=46, num_actions_total=100)
    obs = torch.randn(1, 46, 9, 9)
    legal_mask = torch.zeros(1, 100, dtype=torch.bool)  # No legal actions

    action, log_prob, value = model.get_action_and_value(obs, legal_mask)

    assert not torch.isnan(action).any(), "Action must not be NaN"
    assert not torch.isnan(log_prob).any(), "log_prob must not be NaN"
    assert not torch.isnan(value).any(), "Value must not be NaN"
    assert action.item() == 0, "Dummy action should be 0"
    assert log_prob.item() == 0.0, "Dummy log_prob should be 0.0"


def test_get_action_all_masked_1d_legal_mask():
    """P2-07: all-masked with unbatched (1D) legal_mask and batched logits."""
    model = ActorCritic(input_channels=46, num_actions_total=100)
    obs = torch.randn(1, 46, 9, 9)
    legal_mask = torch.zeros(100, dtype=torch.bool)  # 1D, no legal actions

    action, log_prob, value = model.get_action_and_value(obs, legal_mask)

    assert not torch.isnan(log_prob).any()
    assert log_prob.item() == 0.0


def test_evaluate_actions_all_masked_rows():
    """Regression test for P2-07: all-masked rows in evaluate_actions must
    produce zero log_probs and entropy (PPO ratio = 1, no gradient)."""
    model = ActorCritic(input_channels=46, num_actions_total=100)
    obs = torch.randn(4, 46, 9, 9)
    actions = torch.randint(0, 100, (4,))

    # Rows 0 and 2 have legal actions; rows 1 and 3 are all-masked
    legal_mask = torch.zeros(4, 100, dtype=torch.bool)
    legal_mask[0, :10] = True
    legal_mask[2, 50:60] = True

    log_probs, entropy, value = model.evaluate_actions(obs, actions, legal_mask)

    assert not torch.isnan(log_probs).any(), "log_probs must not contain NaN"
    assert not torch.isnan(entropy).any(), "entropy must not contain NaN"

    # All-masked rows must have zero log_prob and zero entropy
    assert log_probs[1].item() == 0.0
    assert log_probs[3].item() == 0.0
    assert entropy[1].item() == 0.0
    assert entropy[3].item() == 0.0

    # Rows with legal actions must have non-zero values
    assert log_probs[0].item() != 0.0
    assert log_probs[2].item() != 0.0
    assert entropy[0].item() > 0.0
    assert entropy[2].item() > 0.0
