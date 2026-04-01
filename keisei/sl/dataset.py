"""Memory-mapped SL dataset for training positions."""

from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Per-position record layout in shard:
# observation: float32[50*81] = 16200 bytes
# policy_target: int64         = 8 bytes
# value_target: int64          = 8 bytes
# score_target: float32        = 4 bytes
# Total: 16220 bytes per position

OBS_SIZE = 50 * 81
OBS_BYTES = OBS_SIZE * 4  # float32
RECORD_SIZE = OBS_BYTES + 8 + 8 + 4  # 16220 bytes


def write_shard(
    path: Path,
    observations: np.ndarray,
    policy_targets: np.ndarray,
    value_targets: np.ndarray,
    score_targets: np.ndarray,
) -> None:
    """Write positions to a binary shard file."""
    n = observations.shape[0]
    assert observations.shape == (n, OBS_SIZE)
    assert policy_targets.shape == (n,)
    assert value_targets.shape == (n,)
    assert score_targets.shape == (n,)

    with open(path, "wb") as f:
        for i in range(n):
            f.write(observations[i].astype(np.float32).tobytes())
            f.write(policy_targets[i].astype(np.int64).tobytes())
            f.write(value_targets[i].astype(np.int64).tobytes())
            f.write(score_targets[i].astype(np.float32).tobytes())


class SLDataset(Dataset):
    """Memory-mapped dataset reading from binary shard files."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.shards: list[tuple[Path, int]] = []  # (path, num_positions)
        self._cumulative: list[int] = []

        shard_files = sorted(data_dir.glob("shard_*.bin"))
        total = 0
        for shard_path in shard_files:
            file_size = shard_path.stat().st_size
            n_positions = file_size // RECORD_SIZE
            if n_positions > 0:
                self.shards.append((shard_path, n_positions))
                total += n_positions
                self._cumulative.append(total)

        self._total = total
        self._mmap_cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return self._total

    def _get_mmap(self, path: Path) -> np.ndarray:
        if path not in self._mmap_cache:
            self._mmap_cache[path] = np.memmap(path, dtype=np.uint8, mode="r")
        return self._mmap_cache[path]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self._total:
            raise IndexError(
                f"index {idx} out of range for dataset with {self._total} positions"
            )

        # Find which shard this index belongs to
        shard_idx = bisect.bisect_right(self._cumulative, idx)
        local_idx = idx - (self._cumulative[shard_idx - 1] if shard_idx > 0 else 0)

        shard_path, _ = self.shards[shard_idx]
        mmap = self._get_mmap(shard_path)

        offset = local_idx * RECORD_SIZE

        # Read observation
        obs_bytes = mmap[offset : offset + OBS_BYTES]
        obs = np.frombuffer(obs_bytes, dtype=np.float32).reshape(50, 9, 9).copy()

        # Read targets
        offset += OBS_BYTES
        policy = np.frombuffer(mmap[offset : offset + 8], dtype=np.int64).copy()
        offset += 8
        value = np.frombuffer(mmap[offset : offset + 8], dtype=np.int64).copy()
        offset += 8
        score = np.frombuffer(mmap[offset : offset + 4], dtype=np.float32).copy()

        return {
            "observation": torch.from_numpy(obs),
            "policy_target": torch.tensor(policy[0], dtype=torch.long),
            "value_target": torch.tensor(value[0], dtype=torch.long),
            "score_target": torch.tensor(score[0], dtype=torch.float32),
        }
