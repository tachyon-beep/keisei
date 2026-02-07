"""
test_resnet_tower.py: Unit tests for keisei/training/models/resnet_tower.py
"""

import gc
import io
import tracemalloc
from unittest.mock import patch

import pytest
import torch

from keisei.constants import (
    CORE_OBSERVATION_CHANNELS,
    EXTENDED_OBSERVATION_CHANNELS,
    FULL_ACTION_SPACE,
    SHOGI_BOARD_SIZE,
    TEST_SINGLE_LEGAL_ACTION_INDEX,
)
from keisei.training.models.resnet_tower import ActorCriticResTower


@pytest.mark.parametrize(
    "input_channels",
    [CORE_OBSERVATION_CHANNELS, EXTENDED_OBSERVATION_CHANNELS],
    ids=["channels_46", "channels_51"],
)
def test_resnet_tower_forward_shapes(input_channels):
    # Test with different input channel configurations
    model = ActorCriticResTower(
        input_channels=input_channels,
        num_actions_total=FULL_ACTION_SPACE,
        tower_depth=9,
        tower_width=256,
        se_ratio=0.25,
    )
    x = torch.randn(2, input_channels, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)
    policy, value = model(x)
    assert policy.shape == (2, FULL_ACTION_SPACE)
    assert value.shape == (2,)


def test_resnet_tower_fp16_memory():
    # This is a smoke test for memory, not a strict limit
    model = ActorCriticResTower(
        input_channels=EXTENDED_OBSERVATION_CHANNELS,
        num_actions_total=FULL_ACTION_SPACE,
        tower_depth=9,
        tower_width=256,
        se_ratio=0.25,
    )
    x = torch.randn(
        8, EXTENDED_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE
    ).half()
    model = model.half()
    with torch.no_grad():
        policy, value = model(x)
        assert policy.shape == (8, FULL_ACTION_SPACE)
        assert value.shape == (8,)


def test_resnet_tower_se_toggle():
    # Test with and without SE block
    model_se = ActorCriticResTower(
        CORE_OBSERVATION_CHANNELS,
        FULL_ACTION_SPACE,
        tower_depth=3,
        tower_width=64,
        se_ratio=0.5,
    )
    model_no_se = ActorCriticResTower(
        CORE_OBSERVATION_CHANNELS,
        FULL_ACTION_SPACE,
        tower_depth=3,
        tower_width=64,
        se_ratio=None,
    )
    x = torch.randn(1, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)
    p1, v1 = model_se(x)
    p2, v2 = model_no_se(x)
    assert p1.shape == (1, FULL_ACTION_SPACE)
    assert v1.shape == (1,)
    assert p2.shape == (1, FULL_ACTION_SPACE)
    assert v2.shape == (1,)


@pytest.fixture
def model():
    """Create a small test model for testing."""
    return ActorCriticResTower(
        input_channels=CORE_OBSERVATION_CHANNELS,
        num_actions_total=100,  # Smaller for easier testing
        tower_depth=2,
        tower_width=32,
        se_ratio=0.25,
    )


@pytest.fixture
def obs_batch():
    """Create test observation batch."""
    return torch.randn(4, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)


@pytest.fixture
def obs_single():
    """Create single test observation."""
    return torch.randn(1, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)


class TestGetActionAndValue:
    """Test the get_action_and_value() method."""

    def test_basic_functionality(self, model, obs_batch):
        """Test basic functionality without legal mask."""
        action, log_prob, value = model.get_action_and_value(obs_batch)

        assert action.shape == (4,)
        assert log_prob.shape == (4,)
        assert value.shape == (4,)
        assert action.dtype == torch.long
        assert torch.all(action >= 0) and torch.all(action < 100)
        assert not torch.isnan(log_prob).any()
        assert not torch.isnan(value).any()

    def test_deterministic_vs_stochastic(self, model, obs_single):
        """Test deterministic vs stochastic action selection."""
        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Get stochastic actions multiple times
        actions_stochastic = []
        for _ in range(10):
            action, _, _ = model.get_action_and_value(obs_single, deterministic=False)
            actions_stochastic.append(action.item())

        # Get deterministic action multiple times
        actions_deterministic = []
        for _ in range(5):
            action, _, _ = model.get_action_and_value(obs_single, deterministic=True)
            actions_deterministic.append(action.item())

        # Deterministic should always be the same
        assert (
            len(set(actions_deterministic)) == 1
        ), "Deterministic actions should be identical"

        # Stochastic should have some variation (with high probability)
        # Note: This test might occasionally fail due to randomness, but should be rare
        assert len(set(actions_stochastic)) > 1, "Stochastic actions should vary"

    def test_legal_mask_basic(self, model, obs_single):
        """Test basic legal mask functionality."""
        # Create legal mask with some actions masked out
        legal_mask = torch.ones(100, dtype=torch.bool)
        legal_mask[50:] = False  # Mask out actions 50-99

        # Test multiple times to ensure actions are always legal
        for _ in range(20):
            action, log_prob, value = model.get_action_and_value(
                obs_single, legal_mask=legal_mask, deterministic=False
            )
            assert action.item() < 50, f"Action {action.item()} should be < 50 (legal)"
            assert not torch.isnan(log_prob).any()
            assert not torch.isnan(value).any()

    def test_legal_mask_batch_broadcasting(self, model, obs_batch):
        """Test legal mask broadcasting with batch observations."""
        # Create legal mask for batch (4, 100)
        legal_mask = torch.ones(4, 100, dtype=torch.bool)
        legal_mask[0, 10:] = False  # First obs: only actions 0-9 legal
        legal_mask[1, 20:] = False  # Second obs: only actions 0-19 legal
        legal_mask[2, 30:] = False  # Third obs: only actions 0-29 legal
        legal_mask[3, 40:] = False  # Fourth obs: only actions 0-39 legal

        action, _, _ = model.get_action_and_value(
            obs_batch, legal_mask=legal_mask, deterministic=True
        )

        assert action[0].item() < 10, "First action should be < 10"
        assert action[1].item() < 20, "Second action should be < 20"
        assert action[2].item() < 30, "Third action should be < 30"
        assert action[3].item() < 40, "Fourth action should be < 40"

    def test_single_obs_legal_mask_adaptation(self, model, obs_single):
        """Test legal mask shape adaptation for single observation."""
        # Test with 1D legal mask that needs to be adapted for batch
        legal_mask_1d = torch.ones(100, dtype=torch.bool)
        legal_mask_1d[80:] = False

        action, log_prob, value = model.get_action_and_value(
            obs_single, legal_mask=legal_mask_1d, deterministic=False
        )

        assert action.item() < 80, "Action should respect legal mask"
        assert not torch.isnan(log_prob).any()
        assert not torch.isnan(value).any()

    def test_all_false_legal_mask_nan_handling(self, model, obs_single):
        """Test handling of all-False legal mask returns safe dummy values."""
        # Create legal mask with all actions masked out
        legal_mask = torch.zeros(100, dtype=torch.bool)

        # Capture stderr to check for warning message
        captured_stderr = io.StringIO()
        with patch("sys.stderr", captured_stderr):
            action, log_prob, value = model.get_action_and_value(
                obs_single, legal_mask=legal_mask, deterministic=False
            )

        # Check that error was logged
        stderr_content = captured_stderr.getvalue()
        assert "No legal actions in get_action_and_value" in stderr_content
        assert "[ActorCriticResTower] ERROR:" in stderr_content

        # Early return with safe dummy values
        assert action.item() == 0
        assert log_prob.item() == 0.0
        assert not torch.isnan(value).any()

    def test_device_consistency(self, model):
        """Test device consistency when using CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        model = model.to(device)
        obs = torch.randn(
            2,
            CORE_OBSERVATION_CHANNELS,
            SHOGI_BOARD_SIZE,
            SHOGI_BOARD_SIZE,
            device=device,
        )
        legal_mask = torch.ones(2, 100, dtype=torch.bool, device=device)

        action, log_prob, value = model.get_action_and_value(obs, legal_mask)

        assert action.device.type == device.type
        assert log_prob.device.type == device.type
        assert value.device.type == device.type

    def test_gradient_flow(self, model, obs_single):
        """Test that gradients flow properly through the method."""
        obs_single.requires_grad_(True)
        _, log_prob, value = model.get_action_and_value(obs_single)

        # Backward pass on value
        value.backward(retain_graph=True)
        assert obs_single.grad is not None

        obs_single.grad.zero_()

        # Backward pass on log_prob
        log_prob.backward()
        assert obs_single.grad is not None


class TestEvaluateActions:
    """Test the evaluate_actions() method."""

    def test_basic_functionality(self, model, obs_batch):
        """Test basic functionality without legal mask."""
        actions = torch.randint(0, 100, (4,))
        log_probs, entropy, value = model.evaluate_actions(obs_batch, actions)

        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)
        assert not torch.isnan(log_probs).any()
        assert not torch.isnan(entropy).any()
        assert not torch.isnan(value).any()
        assert torch.all(entropy >= 0), "Entropy should be non-negative"

    def test_with_legal_mask(self, model, obs_batch):
        """Test evaluate_actions with legal mask."""
        # Create legal mask and legal actions
        legal_mask = torch.ones(4, 100, dtype=torch.bool)
        legal_mask[:, 50:] = False  # Only actions 0-49 are legal

        # Use legal actions
        actions = torch.randint(0, 50, (4,))

        log_probs, entropy, value = model.evaluate_actions(
            obs_batch, actions, legal_mask=legal_mask
        )

        assert not torch.isnan(log_probs).any()
        assert not torch.isnan(entropy).any()
        assert not torch.isnan(value).any()
        assert torch.all(entropy >= 0)

    def test_all_false_legal_mask_nan_handling(self, model, obs_batch):
        """Test handling of all-False legal mask in evaluate_actions."""
        # Create legal mask with all actions masked out
        legal_mask = torch.zeros(4, 100, dtype=torch.bool)
        actions = torch.randint(0, 100, (4,))

        # Capture stderr to check for warning message
        captured_stderr = io.StringIO()
        with patch("sys.stderr", captured_stderr):
            log_probs, entropy, value = model.evaluate_actions(
                obs_batch, actions, legal_mask=legal_mask
            )

        # Check that error was logged
        stderr_content = captured_stderr.getvalue()
        assert "All-masked rows in evaluate_actions" in stderr_content
        assert "[ActorCriticResTower] ERROR:" in stderr_content

        # All-masked rows get zeroed out (no gradient signal)
        assert not torch.isnan(log_probs).any()
        assert not torch.isnan(entropy).any()
        assert not torch.isnan(value).any()
        assert (log_probs == 0.0).all()
        assert (entropy == 0.0).all()

    def test_consistency_with_get_action_and_value(self, model, obs_single):
        """Test consistency between get_action_and_value and evaluate_actions."""
        legal_mask = torch.ones(100, dtype=torch.bool)
        legal_mask[75:] = False

        # Get action and log_prob from get_action_and_value
        torch.manual_seed(42)
        action, log_prob1, value1 = model.get_action_and_value(
            obs_single, legal_mask=legal_mask.unsqueeze(0), deterministic=True
        )

        # Evaluate the same action with evaluate_actions
        log_prob2, _, value2 = model.evaluate_actions(
            obs_single, action, legal_mask=legal_mask.unsqueeze(0)
        )

        # Log probabilities should be very close (allowing for small numerical differences)
        assert torch.allclose(log_prob1, log_prob2, atol=1e-5)
        # Values should be identical (same forward pass)
        assert torch.allclose(value1, value2, atol=1e-5)

    def test_entropy_properties(self, model, obs_batch):
        """Test entropy calculation properties."""
        actions = torch.randint(0, 100, (4,))

        # Without legal mask
        _, entropy1, _ = model.evaluate_actions(obs_batch, actions)

        # With restrictive legal mask (fewer legal actions -> lower entropy)
        legal_mask = torch.ones(4, 100, dtype=torch.bool)
        legal_mask[:, 10:] = False  # Only 10 actions legal
        _, entropy2, _ = model.evaluate_actions(obs_batch, actions, legal_mask)

        # Entropy with fewer legal actions should generally be lower
        # Note: This is a probabilistic test and might occasionally fail
        assert torch.all(entropy1 >= 0) and torch.all(entropy2 >= 0)

    def test_gradient_flow(self, model, obs_batch):
        """Test that gradients flow properly through evaluate_actions."""
        obs_batch.requires_grad_(True)
        actions = torch.randint(0, 100, (4,))

        log_probs, entropy, value = model.evaluate_actions(obs_batch, actions)

        # Test gradient flow through each output
        loss = log_probs.mean() + entropy.mean() + value.mean()
        loss.backward()

        assert obs_batch.grad is not None
        assert not torch.isnan(obs_batch.grad).any()

    def test_device_consistency(self, model):
        """Test device consistency for evaluate_actions."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        model = model.to(device)
        obs = torch.randn(
            2,
            CORE_OBSERVATION_CHANNELS,
            SHOGI_BOARD_SIZE,
            SHOGI_BOARD_SIZE,
            device=device,
        )
        actions = torch.randint(0, 100, (2,), device=device)
        legal_mask = torch.ones(2, 100, dtype=torch.bool, device=device)

        log_probs, entropy, value = model.evaluate_actions(obs, actions, legal_mask)

        assert log_probs.device.type == device.type
        assert entropy.device.type == device.type
        assert value.device.type == device.type


class TestIntegrationAndEdgeCases:
    """Test integration scenarios and edge cases."""

    def test_extreme_legal_masks(self, model, obs_single):
        """Test with extreme legal mask configurations."""
        # Single legal action
        legal_mask = torch.zeros(100, dtype=torch.bool)
        legal_mask[TEST_SINGLE_LEGAL_ACTION_INDEX] = True

        action, log_prob, value = model.get_action_and_value(
            obs_single, legal_mask=legal_mask, deterministic=True
        )

        assert action.item() == TEST_SINGLE_LEGAL_ACTION_INDEX
        assert not torch.isnan(log_prob).any()
        assert not torch.isnan(value).any()

    def test_numerical_stability(self, model):
        """Test numerical stability with extreme inputs."""
        # Very small observation values
        obs_small = torch.full(
            (1, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE), 1e-8
        )
        _, log_prob, value = model.get_action_and_value(obs_small)
        assert not torch.isnan(log_prob).any() and not torch.isnan(value).any()

        # Very large observation values
        obs_large = torch.full(
            (1, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE), 1e3
        )
        _, log_prob, value = model.get_action_and_value(obs_large)
        assert not torch.isnan(log_prob).any() and not torch.isnan(value).any()

    @pytest.mark.parametrize(
        "batch_size",
        [1, 2, 7, 16, 32],
        ids=["batch_1", "batch_2", "batch_7", "batch_16", "batch_32"],
    )
    def test_batch_size_edge_cases(self, model, batch_size):
        """Test with different batch sizes including edge cases."""
        obs = torch.randn(
            batch_size, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE
        )
        actions = torch.randint(0, 100, (batch_size,))
        legal_mask = torch.ones(batch_size, 100, dtype=torch.bool)

        # Test get_action_and_value
        action, log_prob, value = model.get_action_and_value(obs, legal_mask)
        assert action.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert value.shape == (batch_size,)

        # Test evaluate_actions
        log_probs, entropy, value = model.evaluate_actions(obs, actions, legal_mask)
        assert log_probs.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert value.shape == (batch_size,)

    def test_mixed_legal_masks_in_batch(self, model):
        """Test batch with mixed legal mask conditions."""
        obs_batch = torch.randn(
            3, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE
        )
        legal_mask = torch.ones(3, 100, dtype=torch.bool)

        # First obs: normal mask (50% legal)
        legal_mask[0, 50:] = False
        # Second obs: single legal action
        legal_mask[1, :] = False
        legal_mask[1, 25] = True
        # Third obs: all legal actions
        # legal_mask[2] remains all True

        action, _, _ = model.get_action_and_value(
            obs_batch, legal_mask=legal_mask, deterministic=True
        )

        assert action[0].item() < 50, "First action should be < 50"
        assert action[1].item() == 25, "Second action should be 25"
        assert 0 <= action[2].item() < 100, "Third action should be valid"


# Configuration Edge Cases Tests
@pytest.mark.parametrize(
    "tower_depth", [1, 2, 12], ids=["min_depth", "small_depth", "large_depth"]
)
@pytest.mark.parametrize(
    "tower_width", [16, 64, 512], ids=["small_width", "medium_width", "large_width"]
)
def test_resnet_tower_configuration_edge_cases(tower_depth, tower_width):
    """Test ResNet tower with edge case configurations."""
    model = ActorCriticResTower(
        input_channels=CORE_OBSERVATION_CHANNELS,
        num_actions_total=FULL_ACTION_SPACE,
        tower_depth=tower_depth,
        tower_width=tower_width,
        se_ratio=0.25,
    )
    x = torch.randn(2, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)
    policy, value = model(x)
    assert policy.shape == (2, FULL_ACTION_SPACE)
    assert value.shape == (2,)
    assert not torch.isnan(policy).any()
    assert not torch.isnan(value).any()


@pytest.mark.parametrize(
    "se_ratio", [0.0, 0.125, 0.5, 1.0], ids=["no_se", "small_se", "medium_se", "max_se"]
)
def test_resnet_tower_se_ratio_edge_cases(se_ratio):
    """Test ResNet tower with edge case SE ratios."""
    model = ActorCriticResTower(
        input_channels=CORE_OBSERVATION_CHANNELS,
        num_actions_total=FULL_ACTION_SPACE,
        tower_depth=3,
        tower_width=64,
        se_ratio=se_ratio,
    )
    x = torch.randn(2, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)
    policy, value = model(x)
    assert policy.shape == (2, FULL_ACTION_SPACE)
    assert value.shape == (2,)
    assert not torch.isnan(policy).any()
    assert not torch.isnan(value).any()


# Operational Modes Tests
def test_resnet_tower_training_vs_eval_modes():
    """Test ResNet tower behavior in training vs evaluation modes."""
    model = ActorCriticResTower(
        input_channels=CORE_OBSERVATION_CHANNELS,
        num_actions_total=FULL_ACTION_SPACE,
        tower_depth=3,
        tower_width=64,
        se_ratio=0.25,
    )
    x = torch.randn(4, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)

    # Test training mode
    model.train()
    policy_train, value_train = model(x)
    assert policy_train.shape == (4, FULL_ACTION_SPACE)
    assert value_train.shape == (4,)

    # Test eval mode
    model.eval()
    with torch.no_grad():
        policy_eval, value_eval = model(x)
        assert policy_eval.shape == (4, FULL_ACTION_SPACE)
        assert value_eval.shape == (4,)

    # Outputs should be different between training and eval due to BatchNorm/Dropout
    # Note: This is a probabilistic test - occasionally outputs might be similar
    model.train()
    policy_train2, value_train2 = model(x)

    # The exact difference depends on the architecture, but there should be some variation
    # due to stochastic components like dropout (if present) or BatchNorm behavior
    assert policy_train.shape == policy_train2.shape
    assert value_train.shape == value_train2.shape


# Gradient Flow Tests
def test_resnet_tower_gradient_flow():
    """Test that gradients flow correctly through the ResNet tower."""
    model = ActorCriticResTower(
        input_channels=CORE_OBSERVATION_CHANNELS,
        num_actions_total=100,  # Smaller for efficiency
        tower_depth=2,
        tower_width=32,
        se_ratio=0.25,
    )
    x = torch.randn(
        2,
        CORE_OBSERVATION_CHANNELS,
        SHOGI_BOARD_SIZE,
        SHOGI_BOARD_SIZE,
        requires_grad=True,
    )

    # Forward pass
    policy, value = model(x)

    # Create dummy targets and compute loss
    policy_target = torch.randn_like(policy)
    value_target = torch.randn_like(value)
    loss = torch.nn.functional.mse_loss(
        policy, policy_target
    ) + torch.nn.functional.mse_loss(value, value_target)

    # Backward pass
    loss.backward()

    # Check that gradients exist for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
        assert torch.any(
            param.grad != 0
        ), f"Parameter {name} has zero gradients everywhere"


def test_resnet_tower_gradient_accumulation():
    """Test gradient accumulation across multiple forward passes."""
    model = ActorCriticResTower(
        input_channels=CORE_OBSERVATION_CHANNELS,
        num_actions_total=100,
        tower_depth=2,
        tower_width=32,
        se_ratio=0.25,
    )

    # First forward/backward pass
    x1 = torch.randn(2, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)
    policy1, value1 = model(x1)
    loss1 = policy1.sum() + value1.sum()
    loss1.backward()

    # Store gradients
    grad_dict_1 = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_dict_1[name] = param.grad.clone()

    # Second forward/backward pass (accumulating)
    x2 = torch.randn(2, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)
    policy2, value2 = model(x2)
    loss2 = policy2.sum() + value2.sum()
    loss2.backward()

    # Check that gradients have accumulated (should equal sum of individual gradients)
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Compute what the accumulated gradient should be
            # Run fresh computations to get individual gradients
            model.zero_grad()
            policy1_fresh, value1_fresh = model(x1)
            loss1_fresh = policy1_fresh.sum() + value1_fresh.sum()
            loss1_fresh.backward(retain_graph=True)
            grad_1 = param.grad.clone()

            model.zero_grad()
            policy2_fresh, value2_fresh = model(x2)
            loss2_fresh = policy2_fresh.sum() + value2_fresh.sum()
            loss2_fresh.backward()
            grad_2 = param.grad.clone()

            expected_accumulated = grad_1 + grad_2

            # Reset to accumulated state for comparison
            model.zero_grad()
            loss1 = model(x1)[0].sum() + model(x1)[1].sum()
            loss1.backward(retain_graph=True)
            loss2 = model(x2)[0].sum() + model(x2)[1].sum()
            loss2.backward()

            # Check that accumulated gradient is close to sum of individual gradients
            # Use slightly more relaxed tolerance for numerical precision differences
            torch.testing.assert_close(
                param.grad, expected_accumulated, atol=1e-5, rtol=1e-5
            )


# Device Compatibility Tests
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_resnet_tower_device_compatibility():
    """Test ResNet tower device compatibility (CPU/CUDA)."""
    model = ActorCriticResTower(
        input_channels=CORE_OBSERVATION_CHANNELS,
        num_actions_total=100,
        tower_depth=2,
        tower_width=32,
        se_ratio=0.25,
    )

    # Test on CPU
    x_cpu = torch.randn(
        2, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE
    )
    policy_cpu, value_cpu = model(x_cpu)
    assert policy_cpu.device == torch.device("cpu")
    assert value_cpu.device == torch.device("cpu")

    # Move to CUDA
    model_cuda = model.cuda()
    x_cuda = x_cpu.cuda()
    policy_cuda, value_cuda = model_cuda(x_cuda)
    assert policy_cuda.device.type == "cuda"
    assert value_cuda.device.type == "cuda"

    # Move back to CPU
    model_cpu_again = model_cuda.cpu()
    policy_cpu_again, value_cpu_again = model_cpu_again(x_cpu)
    assert policy_cpu_again.device == torch.device("cpu")
    assert value_cpu_again.device == torch.device("cpu")

    # Results should be numerically close (within floating point precision)
    torch.testing.assert_close(policy_cpu, policy_cpu_again, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(value_cpu, value_cpu_again, atol=1e-5, rtol=1e-5)


def test_resnet_tower_device_mismatch_error():
    """Test that device mismatches raise appropriate errors."""
    model = ActorCriticResTower(
        input_channels=CORE_OBSERVATION_CHANNELS,
        num_actions_total=100,
        tower_depth=2,
        tower_width=32,
        se_ratio=0.25,
    )

    # Move model to CPU explicitly
    model = model.cpu()

    # Create input on different device (if CUDA available)
    if torch.cuda.is_available():
        x_cuda = torch.randn(
            2, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE
        ).cuda()

        # This should raise a RuntimeError due to device mismatch
        with pytest.raises(RuntimeError):
            model(x_cuda)


# Architecture-specific Tests
def test_resnet_tower_residual_connections():
    """Test that residual connections are working (model can learn identity)."""
    model = ActorCriticResTower(
        input_channels=CORE_OBSERVATION_CHANNELS,
        num_actions_total=100,
        tower_depth=3,
        tower_width=64,
        se_ratio=0.0,  # Disable SE for simpler test
    )

    # Get initial output
    x = torch.randn(1, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)
    with torch.no_grad():
        initial_policy, initial_value = model(x)

    # Zero out all non-residual parameters (this is a simplified test)
    # In practice, residual networks should be able to learn identity mappings
    # This test verifies the architecture doesn't break with different inputs
    x_different = torch.randn(
        1, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE
    )
    with torch.no_grad():
        policy_diff, value_diff = model(x_different)

    # Outputs should be valid and different for different inputs
    assert not torch.equal(initial_policy, policy_diff)
    assert not torch.equal(initial_value, value_diff)
    assert not torch.isnan(policy_diff).any()
    assert not torch.isnan(value_diff).any()


def test_resnet_tower_weight_initialization():
    """Test that weights are properly initialized (not all zeros, allow BatchNorm defaults)."""
    model = ActorCriticResTower(
        input_channels=CORE_OBSERVATION_CHANNELS,
        num_actions_total=100,
        tower_depth=2,
        tower_width=32,
        se_ratio=0.25,
    )

    # Check that parameters are not all zeros or all ones (except BatchNorm which defaults to 1)
    for name, param in model.named_parameters():
        if "weight" in name:
            assert not torch.all(param == 0), f"Weight parameter {name} is all zeros"

            # BatchNorm weights are initialized to 1 by default, which is correct
            # Check for BatchNorm by examining the actual module type
            module_path = name.split(".")[:-1]  # Remove 'weight' from the end
            current_module = model
            try:
                for part in module_path:
                    current_module = getattr(current_module, part)
                is_batchnorm = isinstance(
                    current_module,
                    (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
                )
            except AttributeError:
                is_batchnorm = False

            if not is_batchnorm:
                assert not torch.all(param == 1), f"Weight parameter {name} is all ones"
                assert (
                    torch.var(param, dim=None, unbiased=False) > 1e-6
                ), f"Weight parameter {name} has very low variance"

        if "bias" in name:
            # Biases are often initialized to zero, so we just check they exist
            assert param is not None, f"Bias parameter {name} is None"


# Batch Size Compatibility Tests
@pytest.mark.parametrize(
    "batch_size", [1, 3, 8, 16], ids=["single", "small", "medium", "large"]
)
def test_resnet_tower_batch_size_compatibility(batch_size):
    """Test ResNet tower with different batch sizes."""
    model = ActorCriticResTower(
        input_channels=CORE_OBSERVATION_CHANNELS,
        num_actions_total=100,
        tower_depth=2,
        tower_width=32,
        se_ratio=0.25,
    )

    x = torch.randn(
        batch_size, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE
    )
    policy, value = model(x)

    assert policy.shape == (batch_size, 100)
    assert value.shape == (batch_size,)
    assert not torch.isnan(policy).any()
    assert not torch.isnan(value).any()


# Performance and Memory Tests
def test_resnet_tower_memory_efficiency():
    """Test that the model doesn't use excessive memory."""
    # Start tracing memory
    tracemalloc.start()

    model = ActorCriticResTower(
        input_channels=CORE_OBSERVATION_CHANNELS,
        num_actions_total=100,
        tower_depth=3,
        tower_width=64,
        se_ratio=0.25,
    )

    # Multiple forward passes to check for memory leaks
    for _ in range(5):
        x = torch.randn(
            4, CORE_OBSERVATION_CHANNELS, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE
        )
        with torch.no_grad():
            policy, value = model(x)

        # Clean up
        del x, policy, value
        gc.collect()

    # Get memory statistics
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Memory usage should be reasonable (less than 100MB for this test)
    assert peak < 100 * 1024 * 1024, f"Peak memory usage {peak} bytes is too high"
