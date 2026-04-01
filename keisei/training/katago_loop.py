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


class KataGoTrainingLoop:
    def __init__(self, config: AppConfig, vecenv: Any = None) -> None:
        self.config = config
        self.db_path = config.display.db_path

        init_db(self.db_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Validate architecture-algorithm compatibility BEFORE building the model.
        # build_model() allocates the full weight matrix (potentially ~1.5 GB VRAM
        # for a 40-block model). Catching the mismatch here avoids wasting VRAM
        # on a misconfigured run.
        _KATAGO_ARCHITECTURES = {"se_resnet"}
        if config.training.algorithm == "katago_ppo" and config.model.architecture not in _KATAGO_ARCHITECTURES:
            raise ValueError(
                f"algorithm='katago_ppo' requires a KataGoBaseModel architecture "
                f"(one of {_KATAGO_ARCHITECTURES}), got '{config.model.architecture}'"
            )

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

        ppo_params = validate_algorithm_params(
            config.training.algorithm, config.training.algorithm_params
        )
        if not isinstance(ppo_params, KataGoPPOParams):
            raise TypeError(
                f"Expected KataGoPPOParams, got {type(ppo_params).__name__}"
            )

        base_model = self.model.module if hasattr(self.model, "module") else self.model
        self.ppo = KataGoPPOAlgorithm(ppo_params, base_model, forward_model=self.model)

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

            losses = self.ppo.update(self.buffer, next_values)

            # Log zero-gradient situations so operators can distinguish
            # "no terminals this epoch" from "perfectly calibrated head"
            if losses["value_loss"] == 0.0:
                logger.info(
                    "Epoch %d: value_loss=0.0 (likely no terminal steps — "
                    "value head received no gradient this epoch)", epoch_i
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
                # NOTE: score_loss is NOT included — the metrics DB table has no
                # score_loss column. It is logged below but not persisted to DB.
            }
            try:
                write_metrics(self.db_path, metrics)
            except Exception:
                logger.exception("Failed to write metrics for epoch %d — continuing", epoch_i)

            if hasattr(self.vecenv, "reset_stats"):
                self.vecenv.reset_stats()

            try:
                update_training_progress(self.db_path, epoch_i, self.global_step)
            except Exception:
                logger.exception("Failed to update training progress — continuing")

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
                try:
                    update_training_progress(
                        self.db_path, epoch_i + 1, self.global_step, str(ckpt_path)
                    )
                except Exception:
                    logger.exception("Failed to record checkpoint path in DB — continuing")
                logger.info("Checkpoint saved: %s", ckpt_path)

    def _maybe_update_heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat >= 10.0:
            self._last_heartbeat = now
            update_training_progress(self.db_path, self.epoch, self.global_step)
