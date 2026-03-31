"""Abstract base model for all Keisei architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base for all Keisei model architectures.

    Contract:
    - Input: observation tensor (batch, 46, 9, 9)
    - Output: (policy_logits, value) where:
        - policy_logits: (batch, 13527) -- RAW, UNMASKED logits
        - value: (batch, 1) -- scalar value estimate, tanh-activated
    """

    OBS_CHANNELS = 46
    BOARD_SIZE = 9
    ACTION_SPACE = 13527

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...
