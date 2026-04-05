# tests/test_katago_model.py
"""Tests for the KataGo model architecture."""

import pytest
import torch

from keisei.training.model_registry import VALID_ARCHITECTURES, build_model, validate_model_params
from keisei.training.models.katago_base import KataGoBaseModel, KataGoOutput
from keisei.training.models.se_resnet import GlobalPoolBiasBlock, SEResNetModel, SEResNetParams


def test_katago_output_fields():
    """KataGoOutput should have policy_logits, value_logits, score_lead."""
    output = KataGoOutput(
        policy_logits=torch.randn(2, 9, 9, 139),
        value_logits=torch.randn(2, 3),
        score_lead=torch.randn(2, 1),
    )
    assert output.policy_logits.shape == (2, 9, 9, 139)
    assert output.value_logits.shape == (2, 3)
    assert output.score_lead.shape == (2, 1)


def test_katago_base_model_is_abstract():
    """KataGoBaseModel should not be instantiable directly."""
    with pytest.raises(TypeError):
        KataGoBaseModel()  # type: ignore[abstract]


class TestGlobalPoolBiasBlock:
    def test_output_shape(self):
        block = GlobalPoolBiasBlock(channels=64, se_reduction=16, global_pool_channels=32)
        x = torch.randn(4, 64, 9, 9)
        out = block(x)
        assert out.shape == (4, 64, 9, 9)

    def test_residual_connection(self):
        """Output should differ from input (block is not identity)."""
        block = GlobalPoolBiasBlock(channels=64, se_reduction=16, global_pool_channels=32)
        x = torch.randn(4, 64, 9, 9)
        out = block(x)
        assert not torch.allclose(out, x), "Block should not be identity"

    def test_gradient_flows(self):
        block = GlobalPoolBiasBlock(channels=64, se_reduction=16, global_pool_channels=32)
        x = torch.randn(4, 64, 9, 9, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestSEResNetModel:
    @pytest.fixture
    def model(self):
        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        return SEResNetModel(params)

    def test_output_types(self, model):
        obs = torch.randn(4, 50, 9, 9)
        output = model(obs)
        assert isinstance(output, KataGoOutput)

    def test_policy_shape(self, model):
        obs = torch.randn(4, 50, 9, 9)
        output = model(obs)
        assert output.policy_logits.shape == (4, 9, 9, 139)

    def test_value_shape(self, model):
        obs = torch.randn(4, 50, 9, 9)
        output = model(obs)
        assert output.value_logits.shape == (4, 3)

    def test_score_shape(self, model):
        obs = torch.randn(4, 50, 9, 9)
        output = model(obs)
        assert output.score_lead.shape == (4, 1)

    def test_value_logits_are_raw(self, model):
        """Value logits should be raw (not softmaxed). Deterministic check."""
        obs = torch.randn(4, 50, 9, 9)
        output = model(obs)
        row_sums = output.value_logits.sum(dim=-1)
        assert not torch.allclose(row_sums, torch.ones_like(row_sums)), \
            "Value logits should be raw, not already a probability distribution"

    def test_gradient_through_all_heads(self, model):
        obs = torch.randn(4, 50, 9, 9, requires_grad=True)
        output = model(obs)
        loss = (
            output.policy_logits.sum()
            + output.value_logits.sum()
            + output.score_lead.sum()
        )
        loss.backward()
        assert obs.grad is not None
        assert obs.grad.abs().sum() > 0

    def test_wrong_input_channels_raises(self, model):
        obs = torch.randn(4, 46, 9, 9)  # wrong: 46 instead of 50
        with pytest.raises(ValueError, match="Expected obs shape"):
            model(obs)

    def test_batch_size_1(self, model):
        model.eval()
        obs = torch.randn(1, 50, 9, 9)
        output = model(obs)
        assert output.policy_logits.shape == (1, 9, 9, 139)
        assert output.value_logits.shape == (1, 3)
        assert output.score_lead.shape == (1, 1)


class TestModelRegistry:
    def test_se_resnet_in_valid_architectures(self):
        assert "se_resnet" in VALID_ARCHITECTURES

    def test_build_se_resnet(self):
        params = {
            "num_blocks": 2, "channels": 32, "se_reduction": 8,
            "global_pool_channels": 16, "policy_channels": 8,
            "value_fc_size": 32, "score_fc_size": 16, "obs_channels": 50,
        }
        model = build_model("se_resnet", params)
        assert isinstance(model, SEResNetModel)

    def test_validate_se_resnet_params(self):
        params = {"num_blocks": 2, "channels": 32}
        validated = validate_model_params("se_resnet", params)
        assert isinstance(validated, SEResNetParams)
        assert validated.num_blocks == 2
        assert validated.channels == 32
        # Defaults should apply
        assert validated.obs_channels == 50


class TestGlobalPoolBiasBlockBatchSizeOne:
    """MED-2: GlobalPoolBiasBlock with batch_size=1 must produce finite output.

    With batch_size=1, population std (correction=0) returns 0 for constant
    channels rather than NaN. This is the correct behavior, but needs explicit
    verification since it's a known edge case for pooling operations.
    """

    def test_batch_size_one_finite_output(self):
        block = GlobalPoolBiasBlock(channels=32, se_reduction=8, global_pool_channels=16)
        x = torch.randn(1, 32, 9, 9)
        out = block(x)
        assert out.shape == (1, 32, 9, 9)
        assert torch.isfinite(out).all(), "Output contains NaN/Inf with batch_size=1"

    def test_batch_size_one_constant_input(self):
        """Constant spatial input → std=0, should not produce NaN."""
        block = GlobalPoolBiasBlock(channels=16, se_reduction=4, global_pool_channels=8)
        x = torch.full((1, 16, 9, 9), 1.0)
        out = block(x)
        assert torch.isfinite(out).all(), "Constant input with batch_size=1 produced NaN/Inf"

    def test_batch_size_one_gradient_flow(self):
        """Gradients should flow through the block with batch_size=1."""
        block = GlobalPoolBiasBlock(channels=32, se_reduction=8, global_pool_channels=16)
        x = torch.randn(1, 32, 9, 9, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestGlobalPoolFunction:
    """M3: Test _global_pool as a pure function."""

    def test_constant_tensor_std_is_zero(self):
        """A constant feature map should have std=0.0 everywhere."""
        from keisei.training.models.se_resnet import _global_pool

        # Constant tensor: all values are 5.0
        x = torch.full((2, 8, 9, 9), 5.0)
        result = _global_pool(x)

        # Shape should be (B, 3*C) = (2, 24)
        assert result.shape == (2, 8 * 3)

        # Split into mean, max, std components
        g_mean = result[:, :8]
        g_max = result[:, 8:16]
        g_std = result[:, 16:]

        # Mean and max should both be 5.0
        assert torch.allclose(g_mean, torch.full_like(g_mean, 5.0))
        assert torch.allclose(g_max, torch.full_like(g_max, 5.0))
        # Std should be exactly 0.0 (population std of constant)
        assert torch.allclose(g_std, torch.zeros_like(g_std))

    def test_output_shape(self):
        """_global_pool(B, C, H, W) -> (B, 3C)."""
        from keisei.training.models.se_resnet import _global_pool

        x = torch.randn(3, 16, 9, 9)
        result = _global_pool(x)
        assert result.shape == (3, 16 * 3)

    def test_known_values(self):
        """Verify mean/max/std with a hand-crafted input."""
        from keisei.training.models.se_resnet import _global_pool

        # Single batch, single channel, 2x2 spatial: values [1, 2, 3, 4]
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # (1, 1, 2, 2)
        result = _global_pool(x)
        # mean=2.5, max=4.0, std=sqrt(((1-2.5)^2+(2-2.5)^2+(3-2.5)^2+(4-2.5)^2)/4)=sqrt(1.25)
        expected_mean = 2.5
        expected_max = 4.0
        expected_std = (1.25 ** 0.5)
        assert abs(result[0, 0].item() - expected_mean) < 1e-5
        assert abs(result[0, 1].item() - expected_max) < 1e-5
        assert abs(result[0, 2].item() - expected_std) < 1e-5


class TestSEResNetNumericalStability:
    """T8: Verify SE-ResNet produces finite outputs for extreme inputs."""

    @pytest.fixture
    def model(self):
        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        model = SEResNetModel(params)
        model.eval()
        return model

    def test_all_zero_input(self, model):
        """All-zero observation should produce finite output."""
        obs = torch.zeros(2, 50, 9, 9)
        with torch.no_grad():
            output = model(obs)
        assert torch.isfinite(output.policy_logits).all(), "Policy NaN/Inf with zero input"
        assert torch.isfinite(output.value_logits).all(), "Value NaN/Inf with zero input"
        assert torch.isfinite(output.score_lead).all(), "Score NaN/Inf with zero input"

    def test_large_input_values(self, model):
        """Observations with large values should not produce NaN/Inf."""
        obs = torch.full((2, 50, 9, 9), 100.0)
        with torch.no_grad():
            output = model(obs)
        assert torch.isfinite(output.policy_logits).all(), "Policy NaN/Inf with large input"
        assert torch.isfinite(output.value_logits).all(), "Value NaN/Inf with large input"
        assert torch.isfinite(output.score_lead).all(), "Score NaN/Inf with large input"

    def test_small_input_values(self, model):
        """Very small observations should produce finite output."""
        obs = torch.full((2, 50, 9, 9), 1e-7)
        with torch.no_grad():
            output = model(obs)
        assert torch.isfinite(output.policy_logits).all(), "Policy NaN/Inf with tiny input"
        assert torch.isfinite(output.value_logits).all(), "Value NaN/Inf with tiny input"
        assert torch.isfinite(output.score_lead).all(), "Score NaN/Inf with tiny input"

    def test_negative_large_input(self, model):
        """Large negative values should produce finite output."""
        obs = torch.full((2, 50, 9, 9), -50.0)
        with torch.no_grad():
            output = model(obs)
        assert torch.isfinite(output.policy_logits).all()
        assert torch.isfinite(output.value_logits).all()
        assert torch.isfinite(output.score_lead).all()
