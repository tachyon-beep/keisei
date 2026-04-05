"""Supervised learning trainer for KataGo models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]  # stubs lag behind PyTorch 2.x
from torch.utils.data import DataLoader, get_worker_info

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
    use_amp: bool = False
    allow_placeholder: bool = False

    def __post_init__(self) -> None:
        if self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be > 0, got {self.grad_clip}")
        if self.total_epochs < 0:
            raise ValueError(f"total_epochs must be >= 0, got {self.total_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")


logger = logging.getLogger(__name__)


def _sl_worker_init(worker_id: int) -> None:
    """Clear inherited mmap cache after fork so each worker opens its own."""
    info = get_worker_info()
    if info is None:
        return  # called from main process (e.g., in tests)
    dataset = info.dataset
    # Walk wrapper chain (e.g., Subset wraps .dataset)
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    if hasattr(dataset, "clear_cache"):
        dataset.clear_cache()


class SLTrainer:
    """Supervised learning trainer. Trains one epoch at a time.

    Checkpoint management is the caller's responsibility — call
    save_checkpoint() between epochs and load_checkpoint() to resume.

    AMP/mixed precision is enabled via ``SLConfig.use_amp=True``.  When
    enabled, forward and loss computation run under ``torch.autocast`` and
    gradients are scaled with ``GradScaler``.
    """

    def __init__(self, model: KataGoBaseModel, config: SLConfig) -> None:
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scaler = GradScaler(enabled=config.use_amp and self.device.type == "cuda")
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(config.total_epochs, 1), eta_min=1e-6
        )
        # Compute AMP dtype once; used by both model.configure_amp() and
        # train_epoch()'s loss-computation autocast.  Branch on actual device
        # type — CPU autocast only supports bfloat16, not float16.
        if not config.use_amp:
            self._amp_dtype = torch.float16  # placeholder — ignored when disabled
        elif self.device.type == "cpu":
            self._amp_dtype = torch.bfloat16
        elif torch.cuda.is_bf16_supported():
            self._amp_dtype = torch.bfloat16
        else:
            self._amp_dtype = torch.float16
        self._amp_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        model.configure_amp(
            enabled=config.use_amp, dtype=self._amp_dtype,
            device_type=self._amp_device_type,
        )

        self.dataset = SLDataset(
            Path(config.data_dir), allow_placeholder=config.allow_placeholder,
        )
        is_cuda = self.device.type == "cuda"
        has_data = len(self.dataset) > 0
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=has_data,  # RandomSampler rejects empty datasets
            num_workers=config.num_workers if has_data else 0,
            pin_memory=is_cuda and config.num_workers > 0 and has_data,
            persistent_workers=config.num_workers > 0 and has_data,
            worker_init_fn=_sl_worker_init if config.num_workers > 0 and has_data else None,
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

            # Model's forward() handles its own autocast for inductor fusion.
            # Loss computation uses autocast separately.
            output = self.model(obs)

            with autocast(
                device_type=self._amp_device_type,
                dtype=self._amp_dtype,
                enabled=self.config.use_amp,
            ):
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

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_policy += policy_loss.item()
            total_value += value_loss.item()
            total_score += score_loss.item()
            num_batches += 1

        # Only advance scheduler if we actually trained on data.
        # An empty dataset would burn through all annealing ticks without learning.
        if num_batches > 0:
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
