"""Unit tests for distributed training utilities."""

import os
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from keisei.training.distributed import (
    DistributedContext,
    get_distributed_context,
    setup_distributed,
    cleanup_distributed,
    seed_all_ranks,
    broadcast_string,
)


class TestDistributedContext:
    def test_single_gpu_context(self):
        """Non-distributed mode returns rank=0, world_size=1."""
        with patch("torch.cuda.is_available", return_value=False):
            ctx = DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False)
            assert ctx.rank == 0
            assert ctx.local_rank == 0
            assert ctx.world_size == 1
            assert ctx.is_distributed is False
            assert ctx.is_main is True
            assert ctx.device == torch.device("cpu")

    def test_multi_gpu_context_rank0(self):
        ctx = DistributedContext(rank=0, local_rank=0, world_size=4, is_distributed=True)
        assert ctx.is_main is True

    def test_multi_gpu_context_rank1(self):
        ctx = DistributedContext(rank=1, local_rank=1, world_size=4, is_distributed=True)
        assert ctx.is_main is False

    @patch("torch.cuda.is_available", return_value=False)
    def test_device_cpu_when_no_cuda(self, _mock):
        """Device is CPU when CUDA is unavailable."""
        ctx = DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False)
        assert ctx.device == torch.device("cpu")

    @patch("torch.cuda.is_available", return_value=True)
    def test_device_cuda_when_not_distributed(self, _mock):
        """Non-distributed with CUDA returns cuda:0."""
        ctx = DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False)
        assert ctx.device == torch.device("cuda")

    @patch("torch.cuda.is_available", return_value=True)
    def test_device_cuda_distributed_uses_local_rank(self, _mock):
        """Distributed mode returns cuda:{local_rank}."""
        ctx = DistributedContext(rank=3, local_rank=2, world_size=4, is_distributed=True)
        assert ctx.device == torch.device("cuda:2")

    def test_device_is_stable(self):
        """Device property returns the same object on repeated access."""
        with patch("torch.cuda.is_available", return_value=False):
            ctx = DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False)
            assert ctx.device is ctx.device


class TestGetDistributedContext:
    def test_returns_single_gpu_when_no_env(self):
        """Without torchrun env vars, returns non-distributed context."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("RANK", "LOCAL_RANK", "WORLD_SIZE")}
        with patch.dict(os.environ, env, clear=True):
            ctx = get_distributed_context()
            assert ctx.is_distributed is False
            assert ctx.world_size == 1

    def test_returns_distributed_when_env_set(self):
        """With torchrun env vars, returns distributed context."""
        env = {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "4"}
        with patch.dict(os.environ, env):
            ctx = get_distributed_context()
            assert ctx.is_distributed is True
            assert ctx.rank == 1
            assert ctx.world_size == 4

    def test_raises_on_partial_env(self):
        """RANK set but LOCAL_RANK missing raises RuntimeError."""
        env = {"RANK": "0"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("LOCAL_RANK", None)
            os.environ.pop("WORLD_SIZE", None)
            with pytest.raises(RuntimeError, match="LOCAL_RANK"):
                get_distributed_context()


class TestSeedAllRanks:
    def test_seeds_all_rngs(self):
        """seed_all_ranks sets torch, numpy, and Python RNG."""
        seed_all_ranks(123)
        a = torch.randn(3)
        b = np.random.rand(3)
        c = random.random()

        seed_all_ranks(123)
        assert torch.equal(torch.randn(3), a)
        assert np.array_equal(np.random.rand(3), b)
        assert random.random() == c
