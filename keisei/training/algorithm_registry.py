"""Algorithm registry: algorithm name -> params dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOParams:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    epochs_per_batch: int = 4
    batch_size: int = 256
