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
            encoder_layer, num_layers=params.num_layers
        )

        self.policy_fc = nn.Linear(
            d * self.BOARD_SIZE * self.BOARD_SIZE, self.ACTION_SPACE
        )

        self.value_fc1 = nn.Linear(d, d)
        self.value_fc2 = nn.Linear(d, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = obs.shape[0]

        x = obs.permute(0, 2, 3, 1).reshape(batch, 81, self.OBS_CHANNELS)
        x = self.input_proj(x)

        rows = torch.arange(self.BOARD_SIZE, device=obs.device)
        cols = torch.arange(self.BOARD_SIZE, device=obs.device)
        row_emb = self.row_embed(rows)
        col_emb = self.col_embed(cols)
        pos = (row_emb.unsqueeze(1) + col_emb.unsqueeze(0)).reshape(81, -1)
        x = x + pos.unsqueeze(0)

        x = self.encoder(x)

        policy_logits = self.policy_fc(x.reshape(batch, -1))

        pooled = x.mean(dim=1)
        v = torch.relu(self.value_fc1(pooled))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value
