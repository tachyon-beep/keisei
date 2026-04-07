"""Tests for setup_distributed() and cleanup_distributed() error branches.

GAP-H1: The existing test_distributed.py covers DistributedContext and
get_distributed_context well, but setup_distributed/cleanup_distributed
error branches are undertested.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
import torch

from keisei.training.distributed import (
    DistributedContext,
    setup_distributed,
    cleanup_distributed,
)


class TestSetupDistributed:
    """Test setup_distributed() branches."""

    def test_noop_when_not_distributed(self) -> None:
        """Non-distributed context → setup is a no-op, no init_process_group call."""
        ctx = DistributedContext(
            rank=0, local_rank=0, world_size=1, is_distributed=False,
        )
        with patch("keisei.training.distributed.dist.init_process_group") as mock_init:
            setup_distributed(ctx)
            mock_init.assert_not_called()

    def test_nccl_without_cuda_raises(self) -> None:
        """Explicit backend='nccl' with no CUDA should raise RuntimeError."""
        with patch("torch.cuda.is_available", return_value=False):
            ctx = DistributedContext(
                rank=0, local_rank=0, world_size=2, is_distributed=True,
            )
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="backend='nccl' requires CUDA"):
                setup_distributed(ctx, backend="nccl")

    def test_auto_selects_gloo_when_no_cuda(self) -> None:
        """backend=None on CPU-only host → auto-selects 'gloo'."""
        with patch("torch.cuda.is_available", return_value=False):
            ctx = DistributedContext(
                rank=0, local_rank=0, world_size=2, is_distributed=True,
            )
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("keisei.training.distributed.dist.init_process_group") as mock_init,
        ):
            setup_distributed(ctx, backend=None)
            mock_init.assert_called_once_with(backend="gloo")

    def test_auto_selects_nccl_when_cuda_available(self) -> None:
        """backend=None with CUDA → auto-selects 'nccl' and sets device."""
        with patch("torch.cuda.is_available", return_value=True):
            ctx = DistributedContext(
                rank=0, local_rank=0, world_size=2, is_distributed=True,
            )
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.set_device") as mock_set_device,
            patch("keisei.training.distributed.dist.init_process_group") as mock_init,
        ):
            setup_distributed(ctx, backend=None)
            mock_set_device.assert_called_once_with(0)
            mock_init.assert_called_once_with(backend="nccl")

    def test_explicit_gloo_no_cuda_skips_set_device(self) -> None:
        """backend='gloo' on CPU-only host → no cuda.set_device call."""
        with patch("torch.cuda.is_available", return_value=False):
            ctx = DistributedContext(
                rank=0, local_rank=0, world_size=2, is_distributed=True,
            )
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.set_device") as mock_set_device,
            patch("keisei.training.distributed.dist.init_process_group") as mock_init,
        ):
            setup_distributed(ctx, backend="gloo")
            mock_set_device.assert_not_called()
            mock_init.assert_called_once_with(backend="gloo")

    def test_explicit_gloo_with_cuda_still_pins_device(self) -> None:
        """backend='gloo' on CUDA host → must still call cuda.set_device."""
        with patch("torch.cuda.is_available", return_value=True):
            ctx = DistributedContext(
                rank=1, local_rank=1, world_size=2, is_distributed=True,
            )
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.set_device") as mock_set_device,
            patch("keisei.training.distributed.dist.init_process_group") as mock_init,
        ):
            setup_distributed(ctx, backend="gloo")
            mock_set_device.assert_called_once_with(1)
            mock_init.assert_called_once_with(backend="gloo")

    def test_init_process_group_failure_reraises(self) -> None:
        """If dist.init_process_group raises, setup_distributed re-raises."""
        with patch("torch.cuda.is_available", return_value=False):
            ctx = DistributedContext(
                rank=0, local_rank=0, world_size=2, is_distributed=True,
            )
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "keisei.training.distributed.dist.init_process_group",
                side_effect=RuntimeError("mock init failed"),
            ),
        ):
            with pytest.raises(RuntimeError, match="mock init failed"):
                setup_distributed(ctx, backend="gloo")


class TestCleanupDistributed:
    """Test cleanup_distributed() branches."""

    def test_noop_when_not_distributed(self) -> None:
        """Non-distributed context → cleanup is a no-op."""
        ctx = DistributedContext(
            rank=0, local_rank=0, world_size=1, is_distributed=False,
        )
        with patch("keisei.training.distributed.dist.destroy_process_group") as mock_destroy:
            cleanup_distributed(ctx)
            mock_destroy.assert_not_called()

    def test_noop_when_not_initialized(self) -> None:
        """Distributed context but process group never initialized → no-op."""
        with patch("torch.cuda.is_available", return_value=False):
            ctx = DistributedContext(
                rank=0, local_rank=0, world_size=2, is_distributed=True,
            )
        with (
            patch("keisei.training.distributed.dist.is_initialized", return_value=False),
            patch("keisei.training.distributed.dist.destroy_process_group") as mock_destroy,
        ):
            cleanup_distributed(ctx)
            mock_destroy.assert_not_called()

    def test_destroys_when_initialized(self) -> None:
        """Distributed + initialized → destroy_process_group called."""
        with patch("torch.cuda.is_available", return_value=False):
            ctx = DistributedContext(
                rank=0, local_rank=0, world_size=2, is_distributed=True,
            )
        with (
            patch("keisei.training.distributed.dist.is_initialized", return_value=True),
            patch("keisei.training.distributed.dist.destroy_process_group") as mock_destroy,
        ):
            cleanup_distributed(ctx)
            mock_destroy.assert_called_once()
