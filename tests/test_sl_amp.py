"""Tests for AMP mixed precision in SL trainer."""

from __future__ import annotations

import torch
from pathlib import Path

import pytest

from keisei.sl.trainer import SLTrainer, SLConfig
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


def _small_model() -> SEResNetModel:
    """Return a tiny SEResNetModel suitable for unit tests."""
    params = SEResNetParams(
        num_blocks=1,
        channels=8,
        se_reduction=2,
        global_pool_channels=4,
        policy_channels=4,
        value_fc_size=8,
        score_fc_size=8,
        obs_channels=50,
    )
    return SEResNetModel(params)


def _write_shard(shard_dir: Path) -> None:
    """Write a minimal shard file with 8 samples."""
    obs = torch.randn(8, 50, 9, 9)
    policy = torch.randint(0, 81 * 139, (8,))
    value = torch.randint(0, 3, (8,))
    score = torch.randn(8).clamp(-1.5, 1.5)
    torch.save(
        {
            "observation": obs,
            "policy_target": policy,
            "value_target": value,
            "score_target": score,
        },
        shard_dir / "shard_000.pt",
    )


class TestSLAmp:
    def test_sl_epoch_with_amp(self, tmp_path: Path) -> None:
        """SL train_epoch with use_amp=True completes without error."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_shard(shard_dir)

        model = _small_model()
        config = SLConfig(data_dir=str(shard_dir), batch_size=4, use_amp=True)
        trainer = SLTrainer(model, config)
        metrics = trainer.train_epoch()

        assert all(
            torch.isfinite(torch.tensor(v))
            for v in metrics.values()
            if isinstance(v, float)
        )

    def test_sl_epoch_without_amp(self, tmp_path: Path) -> None:
        """Default use_amp=False still works."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_shard(shard_dir)

        model = _small_model()
        config = SLConfig(data_dir=str(shard_dir), batch_size=4)
        trainer = SLTrainer(model, config)
        metrics = trainer.train_epoch()

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "score_loss" in metrics
