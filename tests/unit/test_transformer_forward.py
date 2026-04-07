"""Tests for TransformerModel.forward() with valid inputs.

GAP-M2: Only validation error paths are tested; the actual forward pass
with valid inputs was never exercised.
"""

from __future__ import annotations

import pytest
import torch

from keisei.training.models.transformer import TransformerModel, TransformerParams


class TestTransformerForward:
    """Test TransformerModel forward pass with valid inputs."""

    @pytest.fixture
    def model(self) -> TransformerModel:
        """Small transformer for testing."""
        params = TransformerParams(d_model=32, nhead=4, num_layers=2)
        return TransformerModel(params)

    def test_forward_output_shapes(self, model: TransformerModel) -> None:
        """Forward pass should return (policy_logits, value) with correct shapes."""
        batch_size = 4
        obs = torch.randn(batch_size, 50, 9, 9)
        policy_logits, value = model(obs)

        assert policy_logits.shape == (batch_size, 11259), (
            f"Expected policy_logits shape (4, 11259), got {policy_logits.shape}"
        )
        assert value.shape == (batch_size, 1), (
            f"Expected value shape (4, 1), got {value.shape}"
        )

    def test_forward_single_sample(self, model: TransformerModel) -> None:
        """Forward pass with batch_size=1."""
        obs = torch.randn(1, 50, 9, 9)
        policy_logits, value = model(obs)
        assert policy_logits.shape == (1, 11259)
        assert value.shape == (1, 1)

    def test_value_in_tanh_range(self, model: TransformerModel) -> None:
        """Value output should be in [-1, 1] (tanh activation)."""
        obs = torch.randn(8, 50, 9, 9)
        _, value = model(obs)
        assert value.min() >= -1.0
        assert value.max() <= 1.0

    def test_forward_gradients_flow(self, model: TransformerModel) -> None:
        """Gradients should flow through both policy and value heads."""
        obs = torch.randn(2, 50, 9, 9)
        policy_logits, value = model(obs)

        loss = policy_logits.sum() + value.sum()
        loss.backward()

        # Check that gradients exist on key parameters
        assert model.input_proj.weight.grad is not None
        assert model.policy_fc.weight.grad is not None
        assert model.value_fc1.weight.grad is not None
        assert model.value_fc2.weight.grad is not None
        assert model.row_embed.weight.grad is not None
        assert model.col_embed.weight.grad is not None

    def test_forward_different_d_model_configs(self) -> None:
        """Forward pass works with different d_model/nhead combinations."""
        configs = [
            TransformerParams(d_model=16, nhead=2, num_layers=1),
            TransformerParams(d_model=64, nhead=8, num_layers=1),
            TransformerParams(d_model=48, nhead=6, num_layers=3),
        ]
        obs = torch.randn(2, 50, 9, 9)
        for params in configs:
            model = TransformerModel(params)
            policy_logits, value = model(obs)
            assert policy_logits.shape == (2, 11259)
            assert value.shape == (2, 1)

    def test_forward_eval_mode(self, model: TransformerModel) -> None:
        """Forward pass works in eval mode (no dropout variation)."""
        model.eval()
        obs = torch.randn(2, 50, 9, 9)
        with torch.no_grad():
            policy_logits, value = model(obs)
        assert policy_logits.shape == (2, 11259)
        assert value.shape == (2, 1)

    def test_invalid_input_raises(self, model: TransformerModel) -> None:
        """Invalid input shape should raise ValueError."""
        # Wrong number of channels
        with pytest.raises(ValueError, match="Expected obs shape"):
            model(torch.randn(2, 3, 9, 9))

        # Wrong spatial dimensions
        with pytest.raises(ValueError, match="Expected obs shape"):
            model(torch.randn(2, 50, 8, 8))

        # NHWC instead of NCHW
        with pytest.raises(ValueError, match="NHWC"):
            model(torch.randn(2, 9, 9, 50))
