import pytest
import torch
from keisei.training.models.base import BaseModel
from keisei.training.models.mlp import MLPModel, MLPParams
from keisei.training.models.resnet import ResNetModel, ResNetParams
from keisei.training.models.transformer import TransformerModel, TransformerParams


def test_base_model_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseModel()  # type: ignore[abstract]


class TestResNet:
    def test_forward_shapes(self) -> None:
        params = ResNetParams(hidden_size=32, num_layers=2)
        model = ResNetModel(params)
        obs = torch.randn(4, 46, 9, 9)
        policy_logits, value = model(obs)
        assert policy_logits.shape == (4, 13527)
        assert value.shape == (4, 1)

    def test_value_bounded(self) -> None:
        params = ResNetParams(hidden_size=32, num_layers=2)
        model = ResNetModel(params)
        obs = torch.randn(8, 46, 9, 9)
        _, value = model(obs)
        assert value.min() >= -1.0
        assert value.max() <= 1.0

    def test_single_sample(self) -> None:
        params = ResNetParams(hidden_size=32, num_layers=2)
        model = ResNetModel(params)
        obs = torch.randn(1, 46, 9, 9)
        policy_logits, value = model(obs)
        assert policy_logits.shape == (1, 13527)
        assert value.shape == (1, 1)

    def test_has_batchnorm(self) -> None:
        params = ResNetParams(hidden_size=32, num_layers=2)
        model = ResNetModel(params)
        bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        assert len(bn_layers) > 0, "ResNet must use BatchNorm2d"


class TestMLP:
    def test_forward_shapes(self) -> None:
        params = MLPParams(hidden_sizes=[128, 64])
        model = MLPModel(params)
        obs = torch.randn(4, 46, 9, 9)
        policy_logits, value = model(obs)
        assert policy_logits.shape == (4, 13527)
        assert value.shape == (4, 1)

    def test_value_bounded(self) -> None:
        params = MLPParams(hidden_sizes=[128, 64])
        model = MLPModel(params)
        obs = torch.randn(8, 46, 9, 9)
        _, value = model(obs)
        assert value.min() >= -1.0
        assert value.max() <= 1.0

    def test_has_layernorm(self) -> None:
        params = MLPParams(hidden_sizes=[128, 64])
        model = MLPModel(params)
        ln_layers = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
        assert len(ln_layers) > 0, "MLP must use LayerNorm"


class TestTransformer:
    def test_forward_shapes(self) -> None:
        params = TransformerParams(d_model=32, nhead=4, num_layers=2)
        model = TransformerModel(params)
        obs = torch.randn(4, 46, 9, 9)
        policy_logits, value = model(obs)
        assert policy_logits.shape == (4, 13527)
        assert value.shape == (4, 1)

    def test_value_bounded(self) -> None:
        params = TransformerParams(d_model=32, nhead=4, num_layers=2)
        model = TransformerModel(params)
        obs = torch.randn(8, 46, 9, 9)
        _, value = model(obs)
        assert value.min() >= -1.0
        assert value.max() <= 1.0

    def test_has_positional_encoding(self) -> None:
        params = TransformerParams(d_model=32, nhead=4, num_layers=2)
        model = TransformerModel(params)
        assert hasattr(model, "row_embed"), "Transformer must have 2D row embeddings"
        assert hasattr(model, "col_embed"), "Transformer must have 2D column embeddings"
