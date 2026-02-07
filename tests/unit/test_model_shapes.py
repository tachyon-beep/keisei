"""Tests that CNN and ResNet model architectures produce correct output shapes
and conform to the ActorCriticProtocol."""

import pytest
import torch
import torch.nn as nn

from keisei.core.neural_network import ActorCritic
from keisei.training.models.resnet_tower import ActorCriticResTower
from keisei.core.actor_critic_protocol import ActorCriticProtocol

NUM_ACTIONS = 13527
INPUT_CHANNELS = 46
BOARD_SIZE = 9


@pytest.fixture(params=["cnn", "resnet"])
def model(request):
    """Create a CNN or ResNet model for parametrized testing."""
    if request.param == "cnn":
        return ActorCritic(input_channels=INPUT_CHANNELS, num_actions_total=NUM_ACTIONS)
    else:
        return ActorCriticResTower(
            input_channels=INPUT_CHANNELS,
            num_actions_total=NUM_ACTIONS,
            tower_depth=2,
            tower_width=64,
            se_ratio=0.25,
        )


@pytest.fixture
def dummy_obs():
    """Single observation tensor with batch size 1."""
    return torch.randn(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)


@pytest.fixture
def batch_obs():
    """Batch of 4 observation tensors."""
    return torch.randn(4, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)


class TestForwardOutputStructure:
    """Tests for the forward() method output structure and shapes."""

    def test_forward_returns_tuple_of_two_tensors(self, model, dummy_obs):
        """forward() should return a tuple of exactly 2 tensors."""
        result = model.forward(dummy_obs)
        assert isinstance(result, tuple), "forward() must return a tuple"
        assert len(result) == 2, "forward() must return exactly 2 elements"
        assert isinstance(result[0], torch.Tensor), "policy_logits must be a Tensor"
        assert isinstance(result[1], torch.Tensor), "value must be a Tensor"

    def test_policy_logits_shape_batch_1(self, model, dummy_obs):
        """Policy logits should have shape (1, num_actions) for batch_size=1."""
        policy_logits, _ = model.forward(dummy_obs)
        assert policy_logits.shape == (1, NUM_ACTIONS), (
            f"Expected policy shape (1, {NUM_ACTIONS}), got {policy_logits.shape}"
        )

    def test_value_shape_is_scalar_per_batch_item(self, model, dummy_obs):
        """Value should be a scalar per batch item.

        CNN returns (B, 1), ResNet returns (B,). Both are acceptable as the
        base class squeezes (B, 1) -> (B,) in get_action_and_value and
        evaluate_actions.
        """
        _, value = model.forward(dummy_obs)
        # Accept either (1,) or (1, 1) from forward -- both are valid
        assert value.numel() == 1, (
            f"Expected 1 scalar value for batch=1, got shape {value.shape}"
        )

    def test_forward_batch_4_policy_shape(self, model, batch_obs):
        """Policy logits should have shape (4, num_actions) for batch_size=4."""
        policy_logits, _ = model.forward(batch_obs)
        assert policy_logits.shape == (4, NUM_ACTIONS), (
            f"Expected policy shape (4, {NUM_ACTIONS}), got {policy_logits.shape}"
        )


class TestGetActionAndValue:
    """Tests for the get_action_and_value() method."""

    def test_returns_three_tensors(self, model, dummy_obs):
        """get_action_and_value should return (action, log_prob, value)."""
        result = model.get_action_and_value(dummy_obs)
        assert isinstance(result, tuple) and len(result) == 3, (
            "get_action_and_value must return a 3-tuple"
        )
        action, log_prob, value = result
        assert isinstance(action, torch.Tensor)
        assert isinstance(log_prob, torch.Tensor)
        assert isinstance(value, torch.Tensor)

    def test_action_is_valid_index(self, model, dummy_obs):
        """Action must be a valid index in [0, num_actions)."""
        action, _, _ = model.get_action_and_value(dummy_obs)
        assert action.dtype == torch.long, "Action must be a long (int64) tensor"
        assert 0 <= action.item() < NUM_ACTIONS, (
            f"Action {action.item()} is out of range [0, {NUM_ACTIONS})"
        )

    def test_legal_mask_constrains_action(self, model, dummy_obs):
        """With a legal_mask, the selected action must be among legal ones."""
        # Create a mask with only 3 legal actions
        legal_indices = [0, 100, 5000]
        legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        for idx in legal_indices:
            legal_mask[idx] = True

        # Run multiple times to reduce flakiness from stochastic sampling
        for _ in range(10):
            action, _, _ = model.get_action_and_value(dummy_obs, legal_mask=legal_mask)
            assert action.item() in legal_indices, (
                f"Action {action.item()} is not in legal set {legal_indices}"
            )


class TestEvaluateActions:
    """Tests for the evaluate_actions() method."""

    def test_returns_correct_shapes(self, model, batch_obs):
        """evaluate_actions should return (log_probs, entropy, value) with batch shapes."""
        batch_size = batch_obs.shape[0]
        actions = torch.randint(0, NUM_ACTIONS, (batch_size,))
        log_probs, entropy, value = model.evaluate_actions(batch_obs, actions)

        assert log_probs.shape == (batch_size,), (
            f"log_probs shape {log_probs.shape} != ({batch_size},)"
        )
        assert entropy.shape == (batch_size,), (
            f"entropy shape {entropy.shape} != ({batch_size},)"
        )
        assert value.shape == (batch_size,), (
            f"value shape {value.shape} != ({batch_size},)"
        )

    def test_entropy_is_non_negative(self, model, batch_obs):
        """Entropy of the action distribution should be non-negative."""
        batch_size = batch_obs.shape[0]
        actions = torch.randint(0, NUM_ACTIONS, (batch_size,))
        _, entropy, _ = model.evaluate_actions(batch_obs, actions)

        assert torch.all(entropy >= 0), (
            f"Entropy has negative values: {entropy}"
        )


class TestModelProperties:
    """Tests for general model properties (nn.Module compatibility, etc.)."""

    def test_model_is_nn_module(self, model):
        """Model must be an nn.Module so we can call .to(), .parameters(), etc."""
        assert isinstance(model, nn.Module), (
            f"Model type {type(model).__name__} is not an nn.Module"
        )

    def test_model_has_trainable_parameters(self, model):
        """Model must have at least one trainable parameter."""
        params = list(model.parameters())
        assert len(params) > 0, "Model has no parameters"
        assert any(p.requires_grad for p in params), (
            "Model has no trainable (requires_grad=True) parameters"
        )

    def test_outputs_are_finite(self, model, dummy_obs):
        """Both policy logits and value must be finite (no NaN or Inf)."""
        policy_logits, value = model.forward(dummy_obs)
        assert torch.isfinite(policy_logits).all(), (
            "policy_logits contains NaN or Inf values"
        )
        assert torch.isfinite(value).all(), (
            "value contains NaN or Inf values"
        )
