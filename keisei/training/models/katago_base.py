# keisei/training/models/katago_base.py
"""Abstract base model for KataGo-style multi-head architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class KataGoOutput:
    """Output container for KataGo-style models.

    Fields:
        policy_logits: (batch, 9, 9, 139) -- spatial, raw, unmasked
        value_logits:  (batch, 3) -- [W, D, L] logits (pre-softmax)
        score_lead:    (batch, 1) -- predicted point advantage
    """

    policy_logits: torch.Tensor
    value_logits: torch.Tensor
    score_lead: torch.Tensor


class KataGoBaseModel(ABC, nn.Module):
    """Abstract base for KataGo-style multi-head architectures.

    Contract:
    - Input: observation tensor (batch, obs_channels, 9, 9)
    - Output: KataGoOutput
    """

    BOARD_SIZE = 9
    SPATIAL_MOVE_TYPES = 139
    SPATIAL_ACTION_SPACE = 81 * 139  # 11,259

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> KataGoOutput: ...
