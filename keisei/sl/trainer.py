"""Supervised learning trainer for KataGo models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from keisei.sl.dataset import SLDataset
from keisei.training.models.katago_base import KataGoBaseModel


@dataclass
class SLConfig:
    data_dir: str
    batch_size: int = 4096
    learning_rate: float = 1e-3
    total_epochs: int = 30
    num_workers: int = 0
    lambda_policy: float = 1.0
    lambda_value: float = 1.5
    lambda_score: float = 0.02
    grad_clip: float = 0.5


logger = logging.getLogger(__name__)


class SLTrainer:
    """Supervised learning trainer. Trains one epoch at a time.

    Checkpoint management is the caller's responsibility — call
    save_checkpoint() between epochs and load_checkpoint() to resume.
    """

    def __init__(self, model: KataGoBaseModel, config: SLConfig) -> None:
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.total_epochs, eta_min=1e-6
        )
        self.dataset = SLDataset(Path(config.data_dir))
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_policy = 0.0
        total_value = 0.0
        total_score = 0.0
        num_batches = 0

        for batch in self.dataloader:
            obs = batch["observation"].to(self.device)
            policy_targets = batch["policy_target"].to(self.device)
            value_targets = batch["value_target"].to(self.device)
            score_targets = batch["score_target"].to(self.device)

            output = self.model(obs)

            policy_loss = F.cross_entropy(
                output.policy_logits.reshape(obs.shape[0], -1), policy_targets
            )
            value_loss = F.cross_entropy(output.value_logits, value_targets)
            score_loss = F.mse_loss(output.score_lead.squeeze(-1), score_targets)

            loss = (
                self.config.lambda_policy * policy_loss
                + self.config.lambda_value * value_loss
                + self.config.lambda_score * score_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.optimizer.step()

            total_policy += policy_loss.item()
            total_value += value_loss.item()
            total_score += score_loss.item()
            num_batches += 1

        self.scheduler.step()

        denom = max(num_batches, 1)
        metrics = {
            "policy_loss": total_policy / denom,
            "value_loss": total_value / denom,
            "score_loss": total_score / denom,
        }
        logger.info(
            "SL epoch | policy=%.4f value=%.4f score=%.4f lr=%.6f",
            metrics["policy_loss"],
            metrics["value_loss"],
            metrics["score_loss"],
            self.optimizer.param_groups[0]["lr"],
        )
        return metrics
