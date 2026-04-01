"""KataGo training loop orchestrator."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from keisei.config import AppConfig
from keisei.db import (
    init_db,
    read_training_state,
    update_training_progress,
    write_metrics,
    write_training_state,
)
from keisei.training.algorithm_registry import validate_algorithm_params
from keisei.training.checkpoint import load_checkpoint, save_checkpoint
from keisei.training.katago_ppo import (
    KataGoPPOAlgorithm,
    KataGoPPOParams,
    KataGoRolloutBuffer,
)
from keisei.training.model_registry import build_model

logger = logging.getLogger(__name__)


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule_type: str = "plateau",
    factor: float = 0.5,
    patience: int = 50,
    min_lr: float = 1e-5,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Create an LR scheduler from config parameters.

    Returns the PyTorch scheduler directly — the training loop is responsible
    for extracting the monitored metric and calling scheduler.step(value).
    """
    if schedule_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr,
        )
    else:
        raise ValueError(f"Unknown schedule type '{schedule_type}'")


class KataGoTrainingLoop:
    def __init__(self, config: AppConfig, vecenv: Any = None) -> None:
        self.config = config
        self.db_path = config.display.db_path

        init_db(self.db_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_model(config.model.architecture, config.model.params)
        self.model = self.model.to(self.device)

        gpu_count = torch.cuda.device_count()
        # DataParallel's gather doesn't support custom dataclass outputs
        # (KataGoOutput). Multi-GPU for KataGo models requires DistributedDataParallel
        # with a custom output wrapper — deferred to Plan D.
        if gpu_count > 1:
            logger.info(
                "Found %d GPUs; DataParallel skipped for KataGo (use DDP instead)",
                gpu_count,
            )

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Model: %s (%s), params: %d, device: %s, gpus: %d",
            config.model.display_name,
            config.model.architecture,
            param_count,
            self.device,
            gpu_count,
        )

        # Validate architecture-algorithm compatibility.
        # KataGoPPO requires a KataGoBaseModel (returns KataGoOutput).
        # Passing a BaseModel (returns tuple) would crash deep in the update loop.
        from keisei.training.models.katago_base import KataGoBaseModel

        base_model = self.model.module if hasattr(self.model, "module") else self.model
        if config.training.algorithm == "katago_ppo" and not isinstance(
            base_model, KataGoBaseModel
        ):
            raise ValueError(
                f"algorithm='katago_ppo' requires a KataGoBaseModel architecture "
                f"(e.g. 'se_resnet'), got '{config.model.architecture}' "
                f"which is a {type(base_model).__name__}"
            )

        # Extract nested config sections BEFORE validate_algorithm_params,
        # which constructs KataGoPPOParams(**params) and rejects unknown keys.
        algo_params = dict(config.training.algorithm_params)
        lr_config = algo_params.pop("lr_schedule", {})
        rl_warmup_config = algo_params.pop("rl_warmup", {})

        ppo_params = validate_algorithm_params(
            config.training.algorithm, algo_params
        )
        if not isinstance(ppo_params, KataGoPPOParams):
            raise TypeError(
                f"Expected KataGoPPOParams, got {type(ppo_params).__name__}"
            )

        self.ppo = KataGoPPOAlgorithm(ppo_params, base_model, forward_model=self.model)

        # LR scheduler (optional — only if lr_schedule config is present)
        if lr_config:
            self.lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = (
                create_lr_scheduler(
                    self.ppo.optimizer,
                    schedule_type=lr_config.get("type", "plateau"),
                    factor=lr_config.get("factor", 0.5),
                    patience=lr_config.get("patience", 50),
                    min_lr=lr_config.get("min_lr", 1e-5),
                )
            )
        else:
            self.lr_scheduler = None

        # Store warmup config for use in run()
        self._rl_warmup_epochs = rl_warmup_config.get("epochs", 0)
        self._rl_warmup_entropy = rl_warmup_config.get("entropy_bonus", 0.05)

        if vecenv is not None:
            self.vecenv = vecenv
        else:
            from shogi_gym import VecEnv

            self.vecenv = VecEnv(
                num_envs=config.training.num_games,
                max_ply=config.training.max_ply,
                observation_mode="katago",
                action_mode="spatial",
            )

        # Fail fast on config mismatch — these must survive python -O
        expected_obs = config.model.params.get("obs_channels", 50)
        if self.vecenv.observation_channels != expected_obs:
            raise ValueError(
                f"VecEnv produces {self.vecenv.observation_channels} observation "
                f"channels but model expects {expected_obs}"
            )
        if self.vecenv.action_space_size != 11259:
            raise ValueError(
                f"Expected spatial action space 11259, "
                f"got {self.vecenv.action_space_size}"
            )

        self.num_envs = config.training.num_games
        obs_channels = self.vecenv.observation_channels
        action_space = self.vecenv.action_space_size
        self.buffer = KataGoRolloutBuffer(
            num_envs=self.num_envs,
            obs_shape=(obs_channels, 9, 9),
            action_space=action_space,
        )

        self.score_norm = ppo_params.score_normalization
        self.moves_per_minute = config.display.moves_per_minute
        self._last_snapshot_time = 0.0
        self.epoch = 0
        self.global_step = 0
        self._last_heartbeat = time.monotonic()

        self._check_resume()

    def _check_resume(self) -> None:
        # NOTE: When resuming from an SL checkpoint into RL training, the SL
        # optimizer state is intentionally discarded. KataGoTrainingLoop creates
        # a fresh Adam optimizer. The SL optimizer has momentum from supervised
        # gradients that would fight the RL gradient signal. The RL warmup
        # elevated entropy (Plan D Task 3) compensates for the overconfident
        # SL policy by encouraging exploration in early RL epochs.
        state = read_training_state(self.db_path)
        if state is not None and state.get("checkpoint_path"):
            checkpoint_path = Path(state["checkpoint_path"])
            if checkpoint_path.exists():
                logger.warning(
                    "Resuming from checkpoint: %s (epoch %d)",
                    checkpoint_path,
                    state["current_epoch"],
                )
                base_model = (
                    self.model.module
                    if hasattr(self.model, "module")
                    else self.model
                )
                meta = load_checkpoint(
                    checkpoint_path,
                    base_model,
                    self.ppo.optimizer,
                    expected_architecture=self.config.model.architecture,
                )
                self.epoch = meta["epoch"]
                self.global_step = meta["step"]
                return

        write_training_state(
            self.db_path,
            {
                "config_json": json.dumps(
                    {
                        "training": {
                            "num_games": self.config.training.num_games,
                            "algorithm": self.config.training.algorithm,
                        },
                        "model": {"architecture": self.config.model.architecture},
                    }
                ),
                "display_name": self.config.model.display_name,
                "model_arch": self.config.model.architecture,
                "algorithm_name": self.config.training.algorithm,
                "started_at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            },
        )

    def run(self, num_epochs: int, steps_per_epoch: int) -> None:
        reset_result = self.vecenv.reset()
        obs = torch.from_numpy(np.asarray(reset_result.observations)).to(self.device)
        legal_masks = torch.from_numpy(np.asarray(reset_result.legal_masks)).to(
            self.device
        )

        start_epoch = self.epoch
        for epoch_i in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch_i

            for step_i in range(steps_per_epoch):
                self.global_step += 1

                actions, log_probs, values = self.ppo.select_actions(obs, legal_masks)
                action_list = actions.tolist()
                step_result = self.vecenv.step(action_list)

                rewards = torch.from_numpy(np.asarray(step_result.rewards)).to(
                    self.device
                )
                terminated = torch.from_numpy(np.asarray(step_result.terminated)).to(
                    self.device
                )
                truncated = torch.from_numpy(np.asarray(step_result.truncated)).to(
                    self.device
                )
                dones = terminated | truncated

                # Value categories: -1 (ignore) for non-terminal steps.
                # Only terminal steps get a real label (0=W, 1=D, 2=L).
                # F.cross_entropy(ignore_index=-1) skips these in the loss.
                # Score targets: NaN for non-terminal (masked in PPO update).
                terminal_mask = dones.bool()
                value_cats = torch.full(
                    (self.num_envs,), -1, dtype=torch.long, device=self.device
                )
                value_cats[terminal_mask & (rewards > 0)] = 0  # Win
                value_cats[terminal_mask & (rewards == 0)] = 1  # Draw
                value_cats[terminal_mask & (rewards < 0)] = 2  # Loss

                score_targets = torch.full(
                    (self.num_envs,), float("nan"), device=self.device
                )
                score_targets[terminal_mask] = rewards[terminal_mask] / self.score_norm

                self.buffer.add(
                    obs,
                    actions,
                    log_probs,
                    values,
                    rewards,
                    dones,
                    legal_masks,
                    value_cats,
                    score_targets,
                )

                obs = torch.from_numpy(np.asarray(step_result.observations)).to(
                    self.device
                )
                legal_masks = torch.from_numpy(
                    np.asarray(step_result.legal_masks)
                ).to(self.device)

                self._maybe_update_heartbeat()

            # Bootstrap value for GAE
            self.model.eval()
            with torch.no_grad():
                output = self.model(obs)
                next_values = KataGoPPOAlgorithm.scalar_value(output.value_logits)
            self.model.train()

            # Set epoch-dependent entropy coefficient
            self.ppo.current_entropy_coeff = self.ppo.get_entropy_coeff(
                epoch_i, self._rl_warmup_epochs, self._rl_warmup_entropy,
            )
            if epoch_i == 0 or (epoch_i == self._rl_warmup_epochs):
                logger.info(
                    "Entropy coefficient: %.4f (warmup=%d, epoch=%d)",
                    self.ppo.current_entropy_coeff, self._rl_warmup_epochs, epoch_i,
                )

            losses = self.ppo.update(self.buffer, next_values)

            # Reset LR scheduler fully at warmup boundary BEFORE the step,
            # so the boundary epoch's anomalous loss doesn't leak into the
            # post-warmup patience window.
            if epoch_i == self._rl_warmup_epochs and self.lr_scheduler is not None:
                self.lr_scheduler.best = self.lr_scheduler.mode_worse  # +inf for mode='min'
                self.lr_scheduler.num_bad_epochs = 0
                logger.info("LR scheduler fully reset at warmup boundary (epoch %d)", epoch_i)

            # LR scheduler step (monitors value_loss)
            if self.lr_scheduler is not None:
                monitor_value = losses.get("value_loss")
                if monitor_value is not None:
                    old_lr = self.ppo.optimizer.param_groups[0]["lr"]
                    self.lr_scheduler.step(monitor_value)
                    new_lr = self.ppo.optimizer.param_groups[0]["lr"]
                    if new_lr != old_lr:
                        logger.info(
                            "LR reduced: %.6f -> %.6f (value_loss=%.4f)",
                            old_lr, new_lr, monitor_value,
                        )

            ep_completed = getattr(self.vecenv, "episodes_completed", 0)
            metrics = {
                "epoch": epoch_i,
                "step": self.global_step,
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "entropy": losses["entropy"],
                "gradient_norm": losses["gradient_norm"],
                "episodes_completed": ep_completed,
                "score_loss": losses["score_loss"],
            }
            write_metrics(self.db_path, metrics)

            if hasattr(self.vecenv, "reset_stats"):
                self.vecenv.reset_stats()

            update_training_progress(self.db_path, epoch_i, self.global_step)

            logger.info(
                "Epoch %d | step %d | policy=%.4f value=%.4f score=%.4f entropy=%.4f",
                epoch_i,
                self.global_step,
                losses["policy_loss"],
                losses["value_loss"],
                losses["score_loss"],
                losses["entropy"],
            )

            if (epoch_i + 1) % self.config.training.checkpoint_interval == 0:
                ckpt_path = (
                    Path(self.config.training.checkpoint_dir)
                    / f"epoch_{epoch_i:05d}.pt"
                )
                base_model = (
                    self.model.module
                    if hasattr(self.model, "module")
                    else self.model
                )
                save_checkpoint(
                    ckpt_path,
                    base_model,
                    self.ppo.optimizer,
                    epoch_i + 1,
                    self.global_step,
                    architecture=self.config.model.architecture,
                )
                update_training_progress(
                    self.db_path, epoch_i + 1, self.global_step, str(ckpt_path)
                )
                logger.info("Checkpoint saved: %s", ckpt_path)

    def _maybe_update_heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat >= 10.0:
            self._last_heartbeat = now
            update_training_progress(self.db_path, self.epoch, self.global_step)
