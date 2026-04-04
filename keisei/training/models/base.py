"""Abstract base model for all Keisei architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base for all Keisei model architectures.

    Contract:
    - Input: observation tensor (batch, 50, 9, 9)
    - Output: (policy_logits, value) where:
        - policy_logits: (batch, 11259) -- RAW, UNMASKED logits
        - value: (batch, 1) -- scalar value estimate, tanh-activated
    """

    OBS_CHANNELS = 50
    BOARD_SIZE = 9
    ACTION_SPACE = 11259

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...
