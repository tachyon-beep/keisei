# tests/test_katago_model.py
"""Tests for the KataGo model architecture."""

import torch
import torch.nn.functional as F
import pytest

from keisei.training.models.katago_base import KataGoBaseModel, KataGoOutput
from keisei.training.models.se_resnet import SEResNetParams, GlobalPoolBiasBlock


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
