"""Gap-analysis tests for models: degenerate configs (zero layers, empty hidden)."""

from __future__ import annotations

import torch

from keisei.training.models.mlp import MLPModel, MLPParams
from keisei.training.models.resnet import ResNetModel, ResNetParams


# ===================================================================
# M1 — ResNet with num_layers=0
# ===================================================================


class TestResNetZeroLayers:
    """num_layers=0 produces an empty Sequential (identity).
    The model should still produce correct output shapes."""

    def test_forward_shapes(self) -> None:
        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=0))
        obs = torch.randn(2, 46, 9, 9)
        policy, value = model(obs)
        assert policy.shape == (2, 13527)
        assert value.shape == (2, 1)

    def test_value_bounded(self) -> None:
        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=0))
        obs = torch.randn(8, 46, 9, 9)
        _, value = model(obs)
        assert value.min() >= -1.0
        assert value.max() <= 1.0

    def test_gradient_flow(self) -> None:
        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=0))
        model.train()
        obs = torch.randn(4, 46, 9, 9)
        policy, value = model(obs)
        loss = policy.sum() + value.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


# ===================================================================
# M2 — MLP with empty hidden_sizes
# ===================================================================


class TestMLPEmptyHidden:
    """hidden_sizes=[] creates a direct input→output model (no hidden layers).
    The trunk is an empty Sequential (identity)."""

    def test_forward_shapes(self) -> None:
        model = MLPModel(MLPParams(hidden_sizes=[]))
        obs = torch.randn(2, 46, 9, 9)
        policy, value = model(obs)
        assert policy.shape == (2, 13527)
        assert value.shape == (2, 1)

    def test_value_bounded(self) -> None:
        model = MLPModel(MLPParams(hidden_sizes=[]))
        obs = torch.randn(8, 46, 9, 9)
        _, value = model(obs)
        assert value.min() >= -1.0
        assert value.max() <= 1.0

    def test_no_layernorm_with_empty_hidden(self) -> None:
        """With no hidden layers, there should be no LayerNorm modules."""
        model = MLPModel(MLPParams(hidden_sizes=[]))
        ln_layers = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
        assert len(ln_layers) == 0, (
            "MLP with no hidden layers should have no LayerNorm"
        )

    def test_gradient_flow(self) -> None:
        model = MLPModel(MLPParams(hidden_sizes=[]))
        model.train()
        obs = torch.randn(4, 46, 9, 9)
        policy, value = model(obs)
        loss = policy.sum() + value.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
