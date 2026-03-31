from pathlib import Path

import pytest
import torch

from keisei.training.checkpoint import load_checkpoint, save_checkpoint
from keisei.training.models.resnet import ResNetModel, ResNetParams


@pytest.fixture
def model() -> ResNetModel:
    return ResNetModel(ResNetParams(hidden_size=16, num_layers=1))


def test_save_and_load_round_trip(tmp_path: Path, model: ResNetModel) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(path, model, optimizer, epoch=10, step=1000)
    assert path.exists()

    loaded = load_checkpoint(path, model, optimizer)
    assert loaded["epoch"] == 10
    assert loaded["step"] == 1000


def test_model_weights_preserved(tmp_path: Path, model: ResNetModel) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"

    obs = torch.randn(1, 46, 9, 9)
    with torch.no_grad():
        original_policy, original_value = model(obs)

    save_checkpoint(path, model, optimizer, epoch=1, step=100)

    for p in model.parameters():
        p.data.add_(torch.randn_like(p))

    load_checkpoint(path, model, optimizer)

    with torch.no_grad():
        restored_policy, restored_value = model(obs)

    assert torch.allclose(original_policy, restored_policy, atol=1e-6)
    assert torch.allclose(original_value, restored_value, atol=1e-6)


def test_load_nonexistent_raises(tmp_path: Path, model: ResNetModel) -> None:
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", model, optimizer)
