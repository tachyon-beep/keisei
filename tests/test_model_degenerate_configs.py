"""Gap-analysis tests for models: degenerate configs (zero layers, empty hidden)."""

from __future__ import annotations

import pytest
import torch

from keisei.training.model_registry import validate_model_params
from keisei.training.models.mlp import MLPModel, MLPParams
from keisei.training.models.resnet import ResNetModel, ResNetParams
from keisei.training.models.transformer import TransformerModel, TransformerParams

# ===================================================================
# M1 — ResNet with num_layers=0
# ===================================================================


class TestResNetZeroLayers:
    """num_layers=0 produces an empty Sequential (identity).
    The model should still produce correct output shapes."""

    def test_forward_shapes(self) -> None:
        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=0))
        obs = torch.randn(2, 50, 9, 9)
        policy, value = model(obs)
        assert policy.shape == (2, 11259)
        assert value.shape == (2, 1)

    def test_value_bounded(self) -> None:
        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=0))
        obs = torch.randn(8, 50, 9, 9)
        _, value = model(obs)
        assert value.min() >= -1.0
        assert value.max() <= 1.0

    def test_gradient_flow(self) -> None:
        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=0))
        model.train()
        obs = torch.randn(4, 50, 9, 9)
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
        obs = torch.randn(2, 50, 9, 9)
        policy, value = model(obs)
        assert policy.shape == (2, 11259)
        assert value.shape == (2, 1)

    def test_value_bounded(self) -> None:
        model = MLPModel(MLPParams(hidden_sizes=[]))
        obs = torch.randn(8, 50, 9, 9)
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
        obs = torch.randn(4, 50, 9, 9)
        policy, value = model(obs)
        loss = policy.sum() + value.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


# ===================================================================
# Rejection: TransformerParams invalid configs
# ===================================================================


class TestTransformerParamsRejection:
    """TransformerParams must reject invalid configs at construction."""

    def test_num_layers_zero(self) -> None:
        with pytest.raises(ValueError, match="num_layers"):
            TransformerParams(d_model=32, nhead=4, num_layers=0)

    def test_num_layers_negative(self) -> None:
        with pytest.raises(ValueError, match="num_layers"):
            TransformerParams(d_model=32, nhead=4, num_layers=-1)

    def test_d_model_zero(self) -> None:
        with pytest.raises(ValueError, match="d_model"):
            TransformerParams(d_model=0, nhead=4, num_layers=2)

    def test_nhead_zero(self) -> None:
        with pytest.raises(ValueError, match="nhead"):
            TransformerParams(d_model=32, nhead=0, num_layers=2)

    def test_d_model_not_divisible_by_nhead(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            TransformerParams(d_model=33, nhead=4, num_layers=2)


# ===================================================================
# Rejection: ResNetParams invalid configs
# ===================================================================


class TestResNetParamsRejection:
    """ResNetParams must reject invalid hidden_size."""

    def test_hidden_size_zero(self) -> None:
        with pytest.raises(ValueError, match="hidden_size"):
            ResNetParams(hidden_size=0, num_layers=2)

    def test_hidden_size_negative(self) -> None:
        with pytest.raises(ValueError, match="hidden_size"):
            ResNetParams(hidden_size=-1, num_layers=2)

    def test_num_layers_negative(self) -> None:
        with pytest.raises(ValueError, match="num_layers"):
            ResNetParams(hidden_size=16, num_layers=-1)


# ===================================================================
# Rejection: MLPParams invalid configs
# ===================================================================


class TestMLPParamsRejection:
    """MLPParams must reject zero/negative hidden sizes."""

    def test_hidden_size_zero_entry(self) -> None:
        with pytest.raises(ValueError, match="hidden_sizes"):
            MLPParams(hidden_sizes=[128, 0, 64])

    def test_hidden_size_negative_entry(self) -> None:
        with pytest.raises(ValueError, match="hidden_sizes"):
            MLPParams(hidden_sizes=[-1])


# ===================================================================
# Rejection: model_registry for resnet/mlp
# ===================================================================


class TestRegistryRejectsInvalidConfigs:
    """validate_model_params must reject invalid configs for all architectures."""

    def test_resnet_hidden_size_zero(self) -> None:
        with pytest.raises(ValueError, match="hidden_size"):
            validate_model_params("resnet", {"hidden_size": 0, "num_layers": 2})

    def test_mlp_hidden_size_zero(self) -> None:
        with pytest.raises(ValueError, match="hidden_sizes"):
            validate_model_params("mlp", {"hidden_sizes": [0]})

    def test_transformer_num_layers_zero(self) -> None:
        with pytest.raises(ValueError, match="num_layers"):
            validate_model_params("transformer", {"d_model": 32, "nhead": 4, "num_layers": 0})


# ===================================================================
# Forward shape guards: ResNet and MLP
# ===================================================================


class TestResNetForwardShapeGuard:
    """ResNetModel.forward must reject wrong obs shapes."""

    def test_nhwc_rejected(self) -> None:
        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
        nhwc = torch.randn(2, 9, 9, 50)
        with pytest.raises(ValueError, match="NHWC"):
            model(nhwc)

    def test_wrong_channels(self) -> None:
        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
        wrong = torch.randn(2, 46, 9, 9)
        with pytest.raises(ValueError, match="Expected obs shape"):
            model(wrong)

    def test_wrong_ndim(self) -> None:
        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
        wrong = torch.randn(50, 9, 9)
        with pytest.raises(ValueError, match="Expected obs shape"):
            model(wrong)


class TestMLPForwardShapeGuard:
    """MLPModel.forward must reject wrong obs shapes."""

    def test_nhwc_rejected(self) -> None:
        model = MLPModel(MLPParams(hidden_sizes=[64]))
        nhwc = torch.randn(2, 9, 9, 50)
        with pytest.raises(ValueError, match="NHWC"):
            model(nhwc)

    def test_wrong_channels(self) -> None:
        model = MLPModel(MLPParams(hidden_sizes=[64]))
        wrong = torch.randn(2, 46, 9, 9)
        with pytest.raises(ValueError, match="Expected obs shape"):
            model(wrong)

    def test_wrong_ndim(self) -> None:
        model = MLPModel(MLPParams(hidden_sizes=[64]))
        wrong = torch.randn(50, 9, 9)
        with pytest.raises(ValueError, match="Expected obs shape"):
            model(wrong)
