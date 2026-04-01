# keisei/training/katago_ppo.py
"""KataGo-style multi-head PPO: W/D/L value, score lead, spatial policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KataGoPPOParams:
    learning_rate: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95        # GAE lambda -- exposed as config, not hardcoded
    clip_epsilon: float = 0.2
    epochs_per_batch: int = 4
    batch_size: int = 256
    lambda_policy: float = 1.0
    lambda_value: float = 1.5
    lambda_score: float = 0.02
    lambda_entropy: float = 0.01
    score_normalization: float = 76.0  # used by KataGoTrainingLoop to normalize targets at buffer level
    grad_clip: float = 1.0
