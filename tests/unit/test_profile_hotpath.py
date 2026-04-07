"""Tests for profiling script utilities."""

import pytest
import torch

from scripts.profile_hotpath import create_model_at_scale, SCALES


def test_create_model_small_scale():
    model, params = create_model_at_scale("small", "cpu")
    assert params.num_blocks == 10
    assert params.channels == 128
    obs = torch.randn(2, 50, 9, 9)
    out = model(obs)
    assert out.policy_logits.shape == (2, 9, 9, 139)
    assert out.value_logits.shape == (2, 3)
    assert out.score_lead.shape == (2, 1)


def test_create_model_production_scale():
    model, params = create_model_at_scale("production", "cpu")
    assert params.num_blocks == 40
    assert params.channels == 256
    obs = torch.randn(2, 50, 9, 9)
    out = model(obs)
    assert out.policy_logits.shape == (2, 9, 9, 139)


def test_invalid_scale_raises():
    with pytest.raises(KeyError):
        create_model_at_scale("nonexistent", "cpu")
