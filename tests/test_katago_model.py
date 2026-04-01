# tests/test_katago_model.py
"""Tests for the KataGo model architecture."""

import torch
import torch.nn.functional as F
import pytest

from keisei.training.models.katago_base import KataGoBaseModel, KataGoOutput


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
