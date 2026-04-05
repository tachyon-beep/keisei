"""Integration test: DDP training with 2 ranks on CPU (gloo backend)."""

import os
import socket
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from keisei.config import (
    AppConfig,
    DisplayConfig,
    DistributedConfig,
    ModelConfig,
    TrainingConfig,
)
from keisei.training.distributed import (
    DistributedContext,
    cleanup_distributed,
    seed_all_ranks,
    setup_distributed,
)
from keisei.training.katago_loop import KataGoTrainingLoop

pytestmark = pytest.mark.integration


def _find_free_port() -> int:
    """Find a free TCP port for MASTER_PORT to avoid collisions in CI."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]  # type: ignore[no-any-return]


def _make_mock_vecenv(num_envs: int = 2, seed: int = 42) -> MagicMock:
    """Minimal mock VecEnv for DDP integration test.

    Each call to step() returns a fresh result with new random observations.
    """
    rng = np.random.default_rng(seed)
    mock = MagicMock()
    mock.observation_channels = 50
    mock.action_space_size = 11259
    mock.episodes_completed = 0
    mock.mean_episode_length = 0.0
    mock.truncation_rate = 0.0

    def make_result():
        result = MagicMock()
        result.observations = rng.standard_normal((num_envs, 50, 9, 9)).astype(np.float32)
        result.legal_masks = np.ones((num_envs, 11259), dtype=bool)
        result.rewards = np.zeros(num_envs, dtype=np.float32)
        result.terminated = np.zeros(num_envs, dtype=bool)
        result.truncated = np.zeros(num_envs, dtype=bool)
        result.current_players = np.zeros(num_envs, dtype=np.uint8)
        result.step_metadata = MagicMock()
        result.step_metadata.material_balance = np.zeros(num_envs, dtype=np.int32)
        return result

    mock.reset.side_effect = lambda: make_result()
    mock.step.side_effect = lambda actions: make_result()
    mock.reset_stats = MagicMock()
    return mock


def _worker(rank: int, world_size: int, tmp_dir: str, port: int) -> None:
    """Worker function for each DDP rank."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    ctx = DistributedContext(
        rank=rank, local_rank=rank, world_size=world_size, is_distributed=True,
    )

    # Use the actual setup_distributed function with gloo backend
    setup_distributed(ctx, backend="gloo")

    try:
        seed_all_ranks(42 + rank)

        config = AppConfig(
            training=TrainingConfig(
                num_games=2, max_ply=500, algorithm="katago_ppo",
                checkpoint_interval=1, checkpoint_dir=str(Path(tmp_dir) / "ckpt"),
                algorithm_params={
                    "learning_rate": 1e-3, "epochs_per_batch": 1, "batch_size": 4,
                },
                use_amp=False,
            ),
            display=DisplayConfig(
                moves_per_minute=0,
                db_path=str(Path(tmp_dir) / f"test_rank{rank}.db"),
            ),
            model=ModelConfig(
                display_name="TestDDP", architecture="se_resnet",
                params={"num_blocks": 2, "channels": 32, "obs_channels": 50},
            ),
            distributed=DistributedConfig(sync_batchnorm=False),
        )

        mock_env = _make_mock_vecenv(num_envs=2, seed=42 + rank)
        loop = KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)
        loop.run(num_epochs=1, steps_per_epoch=4)

        state = {k: v.cpu() for k, v in loop._base_model.state_dict().items()}
        torch.save(state, Path(tmp_dir) / f"weights_rank{rank}.pt")

    finally:
        cleanup_distributed(ctx)


@pytest.mark.slow
def test_ddp_two_ranks():
    """End-to-end: 2-rank DDP training produces synchronized weights."""
    port = _find_free_port()
    with tempfile.TemporaryDirectory() as tmp_dir:
        mp.spawn(_worker, args=(2, tmp_dir, port), nprocs=2, join=True)  # type: ignore[attr-defined]

        w0 = torch.load(Path(tmp_dir) / "weights_rank0.pt", weights_only=True)
        w1 = torch.load(Path(tmp_dir) / "weights_rank1.pt", weights_only=True)

        # BatchNorm running statistics (running_mean, running_var,
        # num_batches_tracked) are NOT synchronized by DDP when
        # sync_batchnorm=False — each rank accumulates its own stats from
        # different data.  Only learned parameters (weights/biases) must match.
        bn_running = {"running_mean", "running_var", "num_batches_tracked"}
        for key in w0:
            if any(key.endswith(suffix) for suffix in bn_running):
                continue
            torch.testing.assert_close(
                w0[key], w1[key],
                msg=f"Weight mismatch on parameter '{key}' between rank 0 and rank 1",
            )

        ckpt_dir = Path(tmp_dir) / "ckpt"
        ckpts = list(ckpt_dir.glob("epoch_*.pt"))
        assert len(ckpts) >= 1, "Rank 0 should have saved at least one checkpoint"

        ckpt = torch.load(ckpts[0], weights_only=True)
        assert ckpt["world_size"] == 2
