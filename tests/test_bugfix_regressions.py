"""Regression tests for bugs fixed in the P1/P2 scan findings batch.

Each test is tagged with the bug it guards against so future changes
don't reintroduce the same issue.
"""

from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from keisei.training.checkpoint import load_checkpoint, save_checkpoint
from keisei.training.distributed import setup_distributed, DistributedContext
from keisei.training.models.base import BaseModel
from keisei.training.models.transformer import TransformerModel, TransformerParams


# ---------------------------------------------------------------------------
# Bug: Distributed resume restores rank-0 RNG to all ranks
# Fix: Skip RNG restore when current_world_size > 1
# ---------------------------------------------------------------------------


class TestCheckpointDistributedRNG:
    def test_rng_skipped_when_world_size_gt_1(self, tmp_path: Path) -> None:
        """RNG states should NOT be restored when resuming in distributed mode."""
        from keisei.training.models.resnet import ResNetModel, ResNetParams

        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        path = tmp_path / "ckpt.pt"

        # Save checkpoint (captures current RNG state)
        save_checkpoint(path, model, optimizer, epoch=1, step=100)

        # Advance RNG to different state
        for _ in range(100):
            random.random()
            np.random.random()
            torch.randn(10)

        pre_load_py = random.getstate()
        pre_load_np = np.random.get_state()[1].copy()
        pre_load_torch = torch.random.get_rng_state().clone()

        # Load with world_size > 1 — should NOT restore RNG
        load_checkpoint(path, model, optimizer, current_world_size=2)

        assert random.getstate() == pre_load_py
        assert np.array_equal(np.random.get_state()[1], pre_load_np)
        assert torch.equal(torch.random.get_rng_state(), pre_load_torch)

    def test_rng_restored_when_world_size_1(self, tmp_path: Path) -> None:
        """RNG states SHOULD be restored for single-process resume."""
        from keisei.training.models.resnet import ResNetModel, ResNetParams

        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        path = tmp_path / "ckpt.pt"

        save_checkpoint(path, model, optimizer, epoch=1, step=100)
        saved_py = random.getstate()

        # Advance RNG
        for _ in range(100):
            random.random()

        # Load with world_size=1 — should restore
        load_checkpoint(path, model, optimizer, current_world_size=1)
        assert random.getstate() == saved_py


# ---------------------------------------------------------------------------
# Bug: NCCL default on CPU-only — no auto-fallback to gloo
# Fix: Auto-select backend based on CUDA availability
# ---------------------------------------------------------------------------


class TestDistributedBackendFallback:
    def test_explicit_nccl_without_cuda_raises(self) -> None:
        """Explicitly requesting NCCL without CUDA should raise immediately."""
        ctx = DistributedContext(rank=0, local_rank=0, world_size=2, is_distributed=True)
        with patch("keisei.training.distributed.torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="backend='nccl' requires CUDA"):
                setup_distributed(ctx, backend="nccl")

    def test_auto_selects_gloo_on_cpu(self) -> None:
        """With backend=None and no CUDA, should select gloo."""
        ctx = DistributedContext(rank=0, local_rank=0, world_size=2, is_distributed=True)
        with patch("keisei.training.distributed.torch.cuda.is_available", return_value=False), \
             patch("keisei.training.distributed.dist.init_process_group") as mock_init:
            setup_distributed(ctx, backend=None)
            mock_init.assert_called_once_with(backend="gloo")


# ---------------------------------------------------------------------------
# Bug: TransformerModel silently accepts NHWC observations
# Fix: Validate input shape at top of forward()
# ---------------------------------------------------------------------------


class TestTransformerInputValidation:
    def test_nhwc_input_rejected(self) -> None:
        """NHWC input should be rejected with a helpful hint."""
        model = TransformerModel(TransformerParams(d_model=32, nhead=4, num_layers=1))
        nhwc_obs = torch.randn(2, 9, 9, 50)  # NHWC instead of NCHW
        with pytest.raises(ValueError, match="NHWC"):
            model(nhwc_obs)

    def test_wrong_channels_rejected(self) -> None:
        """Wrong channel count should be rejected."""
        model = TransformerModel(TransformerParams(d_model=32, nhead=4, num_layers=1))
        wrong_obs = torch.randn(2, 46, 9, 9)
        with pytest.raises(ValueError, match="Expected obs shape"):
            model(wrong_obs)

    def test_correct_nchw_accepted(self) -> None:
        """Correct NCHW input should work."""
        model = TransformerModel(TransformerParams(d_model=32, nhead=4, num_layers=1))
        obs = torch.randn(2, 50, 9, 9)
        policy, value = model(obs)
        assert policy.shape == (2, 11259)
        assert value.shape == (2, 1)


# ---------------------------------------------------------------------------
# Bug: BaseModel stale constants (46→50, 13527→11259)
# Fix: Updated OBS_CHANNELS and ACTION_SPACE
# ---------------------------------------------------------------------------


class TestBaseModelConstants:
    def test_obs_channels_matches_katago(self) -> None:
        assert BaseModel.OBS_CHANNELS == 50

    def test_action_space_matches_spatial(self) -> None:
        assert BaseModel.ACTION_SPACE == 11259


# ---------------------------------------------------------------------------
# Bug: SLConfig allows grad_clip <= 0
# Fix: __post_init__ validation
# ---------------------------------------------------------------------------


class TestSLConfigValidation:
    def test_negative_grad_clip_rejected(self) -> None:
        from keisei.sl.trainer import SLConfig
        with pytest.raises(ValueError, match="grad_clip must be > 0"):
            SLConfig(data_dir="/tmp", grad_clip=-0.5)

    def test_zero_grad_clip_rejected(self) -> None:
        from keisei.sl.trainer import SLConfig
        with pytest.raises(ValueError, match="grad_clip must be > 0"):
            SLConfig(data_dir="/tmp", grad_clip=0.0)

    def test_negative_total_epochs_rejected(self) -> None:
        from keisei.sl.trainer import SLConfig
        with pytest.raises(ValueError, match="total_epochs must be >= 0"):
            SLConfig(data_dir="/tmp", total_epochs=-1)

    def test_zero_total_epochs_allowed(self) -> None:
        """total_epochs=0 is valid (skip SL, go straight to RL)."""
        from keisei.sl.trainer import SLConfig
        config = SLConfig(data_dir="/tmp", total_epochs=0)
        assert config.total_epochs == 0


# ---------------------------------------------------------------------------
# Bug: KataGoPPOParams allows invalid hyperparameters
# Fix: __post_init__ validation
# ---------------------------------------------------------------------------


class TestKataGoPPOParamsValidation:
    def test_zero_batch_size_rejected(self) -> None:
        from keisei.training.katago_ppo import KataGoPPOParams
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            KataGoPPOParams(batch_size=0)

    def test_gamma_out_of_range_rejected(self) -> None:
        from keisei.training.katago_ppo import KataGoPPOParams
        with pytest.raises(ValueError, match="gamma must be in"):
            KataGoPPOParams(gamma=1.5)

    def test_negative_grad_clip_rejected(self) -> None:
        from keisei.training.katago_ppo import KataGoPPOParams
        with pytest.raises(ValueError, match="grad_clip must be > 0"):
            KataGoPPOParams(grad_clip=-1.0)


# ---------------------------------------------------------------------------
# Bug: model_registry allows invalid transformer/se_resnet params
# Fix: Semantic validation in validate_model_params
# ---------------------------------------------------------------------------


class TestModelRegistryValidation:
    def test_transformer_d_model_not_divisible_by_nhead(self) -> None:
        from keisei.training.model_registry import validate_model_params
        with pytest.raises(ValueError, match="divisible"):
            validate_model_params("transformer", {"d_model": 32, "nhead": 5, "num_layers": 1})

    def test_se_resnet_se_reduction_too_large(self) -> None:
        from keisei.training.model_registry import validate_model_params
        with pytest.raises(ValueError, match="se_reduction"):
            validate_model_params("se_resnet", {"channels": 8, "se_reduction": 16})


# ---------------------------------------------------------------------------
# Bug: SEResNetParams allows degenerate configs
# Fix: __post_init__ validation
# ---------------------------------------------------------------------------


class TestSEResNetParamsValidation:
    def test_zero_channels_rejected(self) -> None:
        from keisei.training.models.se_resnet import SEResNetParams
        with pytest.raises(ValueError, match="channels must be >= 1"):
            SEResNetParams(channels=0)

    def test_se_reduction_exceeds_channels(self) -> None:
        from keisei.training.models.se_resnet import SEResNetParams
        with pytest.raises(ValueError, match="channels.*se_reduction"):
            SEResNetParams(channels=8, se_reduction=16)


# ---------------------------------------------------------------------------
# Bug: SLDataset sorts shards lexicographically, not numerically
# Fix: Parse numeric index for sort key
# ---------------------------------------------------------------------------


class TestShardSortOrder:
    def test_numeric_sort_order(self, tmp_path: Path) -> None:
        """Shards 9, 10, 11 should be in numeric order, not lex order."""
        from keisei.sl.dataset import SLDataset, RECORD_SIZE

        # Create shards with names that would mis-sort lexicographically
        for idx in [9, 10, 11, 100]:
            shard = tmp_path / f"shard_{idx:03d}.bin"
            # Write exactly 1 record per shard
            shard.write_bytes(b"\x00" * RECORD_SIZE)

        dataset = SLDataset(tmp_path, allow_placeholder=True)
        shard_names = [s[0].name for s in dataset.shards]
        assert shard_names == [
            "shard_009.bin", "shard_010.bin", "shard_011.bin", "shard_100.bin"
        ]


# ---------------------------------------------------------------------------
# Bug: SEResNetModel only validates channels, not board geometry
# Fix: Check all 4 dims of input tensor
# ---------------------------------------------------------------------------


class TestSEResNetBoardGeometry:
    def test_wrong_board_size_rejected(self) -> None:
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        params = SEResNetParams(
            num_blocks=1, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16,
        )
        model = SEResNetModel(params)
        model.eval()
        wrong_obs = torch.randn(2, 50, 8, 8)  # 8x8 instead of 9x9
        with pytest.raises(ValueError, match="Expected obs shape"):
            model(wrong_obs)
