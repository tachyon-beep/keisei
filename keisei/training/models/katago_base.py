# keisei/training/models/katago_base.py
"""Abstract base model for KataGo-style multi-head architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.amp import autocast  # type: ignore[attr-defined]  # stubs lag behind PyTorch 2.x


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

    AMP: call ``configure_amp()`` once after construction.  ``forward()``
    then applies ``torch.amp.autocast`` internally so that inductor can
    fuse the dtype casts with the first conv/BN ops.
    """

    BOARD_SIZE = 9
    SPATIAL_MOVE_TYPES = 139
    SPATIAL_ACTION_SPACE = 81 * 139  # 11,259

    def __init__(self) -> None:
        super().__init__()
        self._amp_enabled: bool = False
        self._amp_dtype: torch.dtype = torch.float16
        self._amp_device_type: str = "cpu"
        self._amp_frozen: bool = False

    def configure_amp(
        self,
        enabled: bool,
        dtype: torch.dtype = torch.float16,
        device_type: str = "cuda",
    ) -> None:
        """Set AMP parameters used by forward()'s internal autocast."""
        if self._amp_frozen:
            raise RuntimeError(
                "configure_amp() must not be called after torch.compile() — "
                "changing AMP attributes would trigger silent recompilation"
            )
        self._amp_enabled = enabled
        self._amp_dtype = dtype
        self._amp_device_type = device_type

    def forward(self, obs: torch.Tensor) -> KataGoOutput:
        if not self._amp_enabled:
            return self._forward_impl(obs)
        with autocast(
            device_type=self._amp_device_type,
            dtype=self._amp_dtype,
        ):
            return self._forward_impl(obs)

    @abstractmethod
    def _forward_impl(self, obs: torch.Tensor) -> KataGoOutput: ...
