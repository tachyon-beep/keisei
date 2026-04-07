"""Transformer architecture with 2D positional encoding for Shogi."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .base import BaseModel


@dataclass(frozen=True)
class TransformerParams:
    d_model: int
    nhead: int
    num_layers: int

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")
        if self.nhead <= 0:
            raise ValueError(f"nhead must be > 0, got {self.nhead}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )


class TransformerModel(BaseModel):
    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        d = params.d_model

        self.input_proj = nn.Linear(self.OBS_CHANNELS, d)

        self.row_embed = nn.Embedding(self.BOARD_SIZE, d)
        self.col_embed = nn.Embedding(self.BOARD_SIZE, d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=params.nhead,
            dim_feedforward=d * 4,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=params.num_layers,
            enable_nested_tensor=False,
        )

        self.policy_fc = nn.Linear(
            d * self.BOARD_SIZE * self.BOARD_SIZE, self.ACTION_SPACE
        )

        self.value_fc1 = nn.Linear(d, d)
        self.value_fc2 = nn.Linear(d, 1)

        # Pre-compute positional index tensors — embedding weights are learned
        # but the index grids are constant. Registered as buffers so they
        # move with the model on .to(device) calls.
        self.register_buffer("_row_idx", torch.arange(self.BOARD_SIZE), persistent=False)
        self.register_buffer("_col_idx", torch.arange(self.BOARD_SIZE), persistent=False)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim != 4 or obs.shape[1] != self.OBS_CHANNELS or obs.shape[2] != self.BOARD_SIZE or obs.shape[3] != self.BOARD_SIZE:
            hint = ""
            if obs.ndim == 4 and obs.shape[-1] == self.OBS_CHANNELS:
                hint = " (input appears to be NHWC — expected NCHW)"
            raise ValueError(
                f"Expected obs shape (batch, {self.OBS_CHANNELS}, {self.BOARD_SIZE}, {self.BOARD_SIZE}), "
                f"got {tuple(obs.shape)}{hint}"
            )
        batch = obs.shape[0]
        num_squares = self.BOARD_SIZE * self.BOARD_SIZE

        x = obs.permute(0, 2, 3, 1).reshape(batch, num_squares, self.OBS_CHANNELS)
        x = self.input_proj(x)

        row_emb = self.row_embed(self._row_idx)
        col_emb = self.col_embed(self._col_idx)
        pos = (row_emb.unsqueeze(1) + col_emb.unsqueeze(0)).reshape(num_squares, -1)
        x = x + pos.unsqueeze(0)

        x = self.encoder(x)

        policy_logits = self.policy_fc(x.reshape(batch, -1))

        pooled = x.mean(dim=1)
        v = torch.relu(self.value_fc1(pooled))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value
