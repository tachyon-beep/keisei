# tests/test_katago_model.py
"""Tests for the KataGo model architecture."""

import torch
import pytest

from keisei.training.models.katago_base import KataGoBaseModel, KataGoOutput
from keisei.training.models.se_resnet import SEResNetParams, GlobalPoolBiasBlock, SEResNetModel
from keisei.training.model_registry import build_model, validate_model_params, VALID_ARCHITECTURES


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
        KataGoBaseModel()


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
        with pytest.raises(ValueError, match="Expected 50 input channels"):
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
