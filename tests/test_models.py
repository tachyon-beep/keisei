import pytest
import torch

from keisei.training.models.base import BaseModel
from keisei.training.models.mlp import MLPModel, MLPParams
from keisei.training.models.resnet import ResNetModel, ResNetParams
from keisei.training.models.transformer import TransformerModel, TransformerParams


def _check_gradient_flow(model: BaseModel) -> None:
    """Verify all parameters receive finite gradients via both heads."""
    model.train()
    obs = torch.randn(4, 46, 9, 9)
    policy, value = model(obs)
    # Loss that depends on both heads
    loss = policy.sum() + value.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


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

    def test_gradient_flow(self) -> None:
        model = ResNetModel(ResNetParams(hidden_size=32, num_layers=2))
        _check_gradient_flow(model)

    def test_eval_mode_deterministic(self) -> None:
        """In eval mode, same input should produce same output (no batch stats noise)."""
        model = ResNetModel(ResNetParams(hidden_size=32, num_layers=2))
        # First run a training forward pass to populate running stats
        model.train()
        model(torch.randn(4, 46, 9, 9))
        # Now eval mode should be deterministic
        model.eval()
        obs = torch.randn(1, 46, 9, 9)
        with torch.no_grad():
            p1, v1 = model(obs)
            p2, v2 = model(obs)
        assert torch.allclose(p1, p2), "eval mode should be deterministic"
        assert torch.allclose(v1, v2), "eval mode should be deterministic"

    def test_train_mode_batch_size_one(self) -> None:
        """ResNet with BatchNorm must not crash on batch_size=1."""
        model = ResNetModel(ResNetParams(hidden_size=32, num_layers=2))
        model.train()
        obs = torch.randn(1, 46, 9, 9)
        # BatchNorm with batch_size=1 can be problematic in some configs
        policy, value = model(obs)
        assert policy.shape == (1, 13527)
        assert value.shape == (1, 1)


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

    def test_gradient_flow(self) -> None:
        model = MLPModel(MLPParams(hidden_sizes=[128, 64]))
        _check_gradient_flow(model)


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

    def test_gradient_flow(self) -> None:
        model = TransformerModel(TransformerParams(d_model=32, nhead=4, num_layers=2))
        _check_gradient_flow(model)

    def test_nhead_must_divide_d_model(self) -> None:
        """nhead that doesn't divide d_model should raise at construction time."""
        # d_model=32 is not divisible by nhead=5
        with pytest.raises((AssertionError, ValueError)):
            TransformerModel(TransformerParams(d_model=32, nhead=5, num_layers=1))

    def test_positional_encoding_affects_output(self) -> None:
        """Two observations that differ only in spatial layout should get different logits."""
        model = TransformerModel(TransformerParams(d_model=32, nhead=4, num_layers=1))
        model.eval()
        # obs1: all weight on square (0,0)
        obs1 = torch.zeros(1, 46, 9, 9)
        obs1[0, 0, 0, 0] = 1.0
        # obs2: all weight on square (8,8)
        obs2 = torch.zeros(1, 46, 9, 9)
        obs2[0, 0, 8, 8] = 1.0
        with torch.no_grad():
            p1, _ = model(obs1)
            p2, _ = model(obs2)
        assert not torch.allclose(p1, p2, atol=1e-4), (
            "Positional encoding should make spatially different inputs produce different outputs"
        )
