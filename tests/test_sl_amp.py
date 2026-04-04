"""Tests for AMP mixed precision in SL trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from keisei.sl.dataset import OBS_SIZE, write_shard
from keisei.sl.trainer import SLConfig, SLTrainer
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


def _write_binary_shard(shard_dir: Path, n: int = 16) -> None:
    """Write a minimal binary shard with n samples in the format SLDataset expects."""
    observations = np.random.randn(n, OBS_SIZE).astype(np.float32)
    policy_targets = np.random.randint(0, 81 * 139, size=(n,)).astype(np.int64)
    value_targets = np.random.randint(0, 3, size=(n,)).astype(np.int64)
    score_targets = np.random.uniform(-1.5, 1.5, size=(n,)).astype(np.float32)
    write_shard(shard_dir / "shard_000.bin", observations, policy_targets,
                value_targets, score_targets)


class TestSLAmp:
    def test_sl_epoch_with_amp(self, tmp_path: Path) -> None:
        """SL train_epoch with use_amp=True completes and produces non-zero losses."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_binary_shard(shard_dir, n=16)

        model = _small_model()
        config = SLConfig(data_dir=str(shard_dir), batch_size=8, use_amp=True)
        trainer = SLTrainer(model, config)
        metrics = trainer.train_epoch()

        # Verify the trainer actually processed data (not vacuous)
        assert metrics["policy_loss"] > 0.0, "policy_loss is zero — no data was trained"
        assert metrics["value_loss"] > 0.0, "value_loss is zero — no data was trained"
        assert all(
            np.isfinite(v) for v in metrics.values() if isinstance(v, float)
        )

    def test_sl_epoch_without_amp(self, tmp_path: Path) -> None:
        """Default use_amp=False still works and trains on data."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_binary_shard(shard_dir, n=16)

        model = _small_model()
        config = SLConfig(data_dir=str(shard_dir), batch_size=8)
        trainer = SLTrainer(model, config)
        metrics = trainer.train_epoch()

        assert metrics["policy_loss"] > 0.0, "policy_loss is zero — no data was trained"
        assert metrics["value_loss"] > 0.0, "value_loss is zero — no data was trained"
        assert "score_loss" in metrics

    def test_sl_amp_updates_model_weights(self, tmp_path: Path) -> None:
        """AMP training should actually update model parameters."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_binary_shard(shard_dir, n=16)

        model = _small_model()
        params_before = {n: p.data.clone() for n, p in model.named_parameters()}

        config = SLConfig(data_dir=str(shard_dir), batch_size=8, use_amp=True)
        trainer = SLTrainer(model, config)
        trainer.train_epoch()

        changed = sum(
            1 for n, p in model.named_parameters()
            if not torch.equal(params_before[n], p.data)
        )
        assert changed > 0, "No parameters changed — AMP training had no effect"
