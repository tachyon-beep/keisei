# tests/test_katago_checkpoint.py
"""Tests for enhanced checkpoint with architecture metadata."""

import tempfile
from pathlib import Path

import pytest
import torch

from keisei.training.checkpoint import save_checkpoint, load_checkpoint
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


@pytest.fixture
def model():
    params = SEResNetParams(
        num_blocks=2, channels=32, se_reduction=8,
        global_pool_channels=16, policy_channels=8,
        value_fc_size=32, score_fc_size=16, obs_channels=50,
    )
    return SEResNetModel(params)


def test_save_with_architecture_metadata(model):
    optimizer = torch.optim.Adam(model.parameters())
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        save_checkpoint(path, model, optimizer, 10, 100, architecture="se_resnet")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        assert ckpt["architecture"] == "se_resnet"


def test_load_with_architecture_check(model):
    optimizer = torch.optim.Adam(model.parameters())
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        save_checkpoint(path, model, optimizer, 10, 100, architecture="se_resnet")
        meta = load_checkpoint(path, model, optimizer, expected_architecture="se_resnet")
        assert meta["epoch"] == 10


def test_load_architecture_mismatch_raises(model):
    optimizer = torch.optim.Adam(model.parameters())
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        save_checkpoint(path, model, optimizer, 10, 100, architecture="se_resnet")
        with pytest.raises(ValueError, match="architecture mismatch"):
            load_checkpoint(path, model, optimizer, expected_architecture="resnet")


def test_load_legacy_checkpoint_no_architecture(model):
    """Old checkpoints without architecture field should load when no check requested."""
    optimizer = torch.optim.Adam(model.parameters())
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        # Save without architecture (legacy format)
        save_checkpoint(path, model, optimizer, 5, 50)
        meta = load_checkpoint(path, model, optimizer)
        assert meta["epoch"] == 5
