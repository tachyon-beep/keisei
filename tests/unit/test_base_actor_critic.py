"""Tests for BaseActorCriticModel's legal masking and safety behavior.

Uses ActorCritic (the simple CNN) as a concrete subclass for testing,
since BaseActorCriticModel is abstract.
"""

import pytest
import torch

from keisei.core.neural_network import ActorCritic

NUM_ACTIONS = 13527
INPUT_CHANNELS = 46
BOARD_SIZE = 9


@pytest.fixture
def model():
    """Create a CNN model (concrete BaseActorCriticModel subclass)."""
    return ActorCritic(input_channels=INPUT_CHANNELS, num_actions_total=NUM_ACTIONS)


@pytest.fixture
def obs():
    """Single observation tensor with batch size 1."""
    return torch.randn(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)


@pytest.fixture
def batch_obs():
    """Batch of 4 observation tensors."""
    return torch.randn(4, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)


class TestLegalMaskBehavior:
    """Tests for legal mask application in get_action_and_value."""

    def test_action_respects_legal_mask(self, model, obs):
        """With a legal_mask, the selected action must have mask[action] == True."""
        legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        legal_mask[42] = True
        legal_mask[999] = True
        legal_mask[13000] = True

        for _ in range(20):
            action, _, _ = model.get_action_and_value(obs, legal_mask=legal_mask)
            assert legal_mask[action.item()], (
                f"Action {action.item()} was selected but is not legal"
            )

    def test_single_legal_action_always_selected(self, model, obs):
        """When only one action is legal, it must always be selected."""
        legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        legal_mask[7777] = True

        for _ in range(10):
            action, _, _ = model.get_action_and_value(obs, legal_mask=legal_mask)
            assert action.item() == 7777, (
                f"Expected action 7777 but got {action.item()}"
            )

    def test_deterministic_mode_consistent(self, model, obs):
        """Deterministic mode should select the same action every time."""
        legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        legal_mask[100] = True
        legal_mask[200] = True
        legal_mask[300] = True

        actions = []
        for _ in range(10):
            action, _, _ = model.get_action_and_value(
                obs, legal_mask=legal_mask, deterministic=True
            )
            actions.append(action.item())

        assert len(set(actions)) == 1, (
            f"Deterministic mode produced different actions: {set(actions)}"
        )

    def test_stochastic_mode_finite_log_prob(self, model, obs):
        """Stochastic mode should produce a valid (finite) log probability."""
        legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        legal_mask[50] = True
        legal_mask[100] = True
        legal_mask[150] = True

        _, log_prob, _ = model.get_action_and_value(
            obs, legal_mask=legal_mask, deterministic=False
        )
        assert torch.isfinite(log_prob).all(), (
            f"log_prob is not finite: {log_prob}"
        )


class TestAllMaskedSafety:
    """Tests for the safety behavior when all actions are masked (terminal state)."""

    def test_all_masked_returns_safe_dummy_values(self, model, obs):
        """All-masked legal_mask should return action=0 and log_prob=0."""
        legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)

        action, log_prob, value = model.get_action_and_value(
            obs, legal_mask=legal_mask
        )
        assert action.item() == 0, (
            f"Expected dummy action 0 for all-masked, got {action.item()}"
        )
        assert log_prob.item() == 0.0, (
            f"Expected dummy log_prob 0.0 for all-masked, got {log_prob.item()}"
        )

    def test_evaluate_actions_all_masked_zeros_log_probs_and_entropy(self, model):
        """evaluate_actions with all-masked rows should zero out log_probs and
        entropy for those rows."""
        batch_size = 4
        batch_obs = torch.randn(batch_size, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        actions = torch.zeros(batch_size, dtype=torch.long)

        # All rows are fully masked
        legal_mask = torch.zeros(batch_size, NUM_ACTIONS, dtype=torch.bool)

        log_probs, entropy, value = model.evaluate_actions(
            batch_obs, actions, legal_mask=legal_mask
        )
        assert torch.all(log_probs == 0.0), (
            f"Expected log_probs all 0.0 for all-masked, got {log_probs}"
        )
        assert torch.all(entropy == 0.0), (
            f"Expected entropy all 0.0 for all-masked, got {entropy}"
        )


class TestEvaluateActionsShapes:
    """Tests for evaluate_actions output shapes and value squeezing."""

    def test_evaluate_actions_returns_correct_shapes(self, model, batch_obs):
        """evaluate_actions should return matching batch-sized tensors."""
        batch_size = batch_obs.shape[0]
        actions = torch.randint(0, NUM_ACTIONS, (batch_size,))

        log_probs, entropy, value = model.evaluate_actions(batch_obs, actions)
        assert log_probs.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert value.shape == (batch_size,)

    def test_value_squeezed_correctly(self, model, obs):
        """Value should have no extra trailing dimension of size 1 after
        get_action_and_value squeezes it."""
        _, _, value = model.get_action_and_value(obs)
        # After squeezing, value for batch=1 should be shape (1,) not (1,1)
        assert value.shape == (1,), (
            f"Expected value shape (1,) after squeezing, got {value.shape}"
        )

    def test_legal_mask_broadcasts_for_batch_1(self, model, obs):
        """A 1-D legal_mask should work correctly with batch_size=1 input."""
        legal_mask = torch.ones(NUM_ACTIONS, dtype=torch.bool)
        # Should not raise; the base class unsqueezes for batch=1
        action, log_prob, value = model.get_action_and_value(
            obs, legal_mask=legal_mask
        )
        assert torch.isfinite(log_prob).all()
        assert torch.isfinite(value).all()


class TestEntropyConstraint:
    """Tests for entropy behavior under different constraint levels."""

    def test_entropy_lower_with_fewer_legal_actions(self, model, obs):
        """Entropy should be lower when fewer actions are legal (more constrained)."""
        # Mask with many legal actions
        wide_mask = torch.ones(NUM_ACTIONS, dtype=torch.bool)
        # Mask with very few legal actions
        narrow_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        narrow_mask[0] = True
        narrow_mask[1] = True

        # Use evaluate_actions with batch dim
        actions_wide = torch.tensor([0])
        actions_narrow = torch.tensor([0])

        wide_mask_batch = wide_mask.unsqueeze(0)
        narrow_mask_batch = narrow_mask.unsqueeze(0)

        _, entropy_wide, _ = model.evaluate_actions(
            obs, actions_wide, legal_mask=wide_mask_batch
        )
        _, entropy_narrow, _ = model.evaluate_actions(
            obs, actions_narrow, legal_mask=narrow_mask_batch
        )

        assert entropy_narrow.item() < entropy_wide.item(), (
            f"Expected narrow entropy ({entropy_narrow.item()}) < wide entropy "
            f"({entropy_wide.item()})"
        )
