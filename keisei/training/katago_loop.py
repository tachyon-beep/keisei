"""KataGo training loop orchestrator."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from keisei.config import AppConfig, load_config

if TYPE_CHECKING:
    from keisei.training.value_adapter import ValueHeadAdapter
from keisei.db import (
    init_db,
    read_training_state,
    update_training_progress,
    write_game_snapshots,
    write_metrics,
    write_training_state,
)
from keisei.training.algorithm_registry import validate_algorithm_params
from keisei.training.checkpoint import load_checkpoint, save_checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from keisei.training.distributed import (
    DistributedContext, get_distributed_context,
    setup_distributed, cleanup_distributed, seed_all_ranks,
)
from keisei.training.katago_ppo import (
    KataGoPPOAlgorithm,
    KataGoPPOParams,
    KataGoRolloutBuffer,
)
# scalar_value is used in split_merge_step to keep value computation
# centralized — the single source of truth is KataGoPPOAlgorithm.scalar_value.
from keisei.training.league import OpponentEntry, OpponentPool, OpponentSampler, compute_elo_update
from keisei.training.model_registry import build_model

logger = logging.getLogger(__name__)


@dataclass
class SplitMergeResult:
    """Result of a split-merge step."""

    actions: torch.Tensor  # (num_envs,) merged actions for all envs
    learner_mask: torch.Tensor  # (num_envs,) bool
    opponent_mask: torch.Tensor  # (num_envs,) bool
    learner_log_probs: torch.Tensor  # (n_learner,)
    learner_values: torch.Tensor  # (n_learner,) scalar values for GAE
    learner_indices: torch.Tensor  # (n_learner,) indices into full env array


def _negate_where(
    values: torch.Tensor,
    condition: np.ndarray,
) -> torch.Tensor:
    """Clone a tensor and negate elements where condition is True.

    Shared implementation for perspective correction functions.
    """
    result = values.clone()
    if result.numel() == 0:
        return result
    mask = torch.tensor(condition, device=values.device, dtype=torch.bool)
    result[mask] = -result[mask]
    return result


def to_learner_perspective(
    rewards: torch.Tensor,
    pre_players: np.ndarray,
    learner_side: int,
) -> torch.Tensor:
    """Convert rewards from last-mover perspective to learner perspective.

    In split-merge mode, rewards from vecenv.step() are relative to
    whoever just moved (last_mover = pre_players). When the opponent
    moved, the reward sign must be flipped for the learner.
    """
    return _negate_where(rewards, pre_players != learner_side)


def sign_correct_bootstrap(
    next_values: torch.Tensor,
    current_players: np.ndarray,
    learner_side: int,
) -> torch.Tensor:
    """Correct bootstrap values for learner-centric GAE.

    The value network outputs from current-player perspective. When the
    opponent is to-move, the bootstrap value represents the opponent's
    advantage and must be negated for learner-centric GAE targets.
    """
    return _negate_where(next_values, current_players != learner_side)


class PendingTransitions:
    """Per-env state for learner transitions awaiting outcome resolution.

    In split-merge mode, a learner transition is "opened" when the learner
    moves, but its reward and done flag depend on what happens next (which
    may be an opponent move). This class holds the transition data until
    the outcome resolves.

    Memory footprint: For num_envs=512, obs_shape=(50,9,9), action_space=11259,
    the total allocation is ~90 MB on the device. This is a persistent per-epoch
    cost alongside the rollout buffer. Kept on GPU to avoid device mismatches
    in the finalize-mask logic (dones, valid, learner_next are all on GPU).
    """

    def __init__(
        self,
        num_envs: int,
        obs_shape: tuple[int, ...],
        action_space: int,
        device: torch.device,
    ) -> None:
        self.num_envs = num_envs
        self.obs = torch.zeros(num_envs, *obs_shape, device=device)
        self.actions = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(num_envs, device=device)
        self.values = torch.zeros(num_envs, device=device)
        self.legal_masks = torch.zeros(num_envs, action_space, dtype=torch.bool, device=device)
        self.rewards = torch.zeros(num_envs, device=device)
        self.score_targets = torch.zeros(num_envs, device=device)
        self.valid = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def create(
        self,
        env_mask: torch.Tensor,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        legal_masks: torch.Tensor,
        rewards: torch.Tensor,
        score_targets: torch.Tensor,
    ) -> None:
        """Open pending transitions for envs where the learner just moved.

        Raises AssertionError if any env in env_mask already has a valid
        pending transition (indicates a protocol bug — finalize must be
        called before create for the same env).
        """
        assert not (env_mask & self.valid).any(), (
            "create() called on env(s) with already-valid pending transition. "
            "finalize() must be called first."
        )
        self.obs[env_mask] = obs[env_mask]
        self.actions[env_mask] = actions[env_mask]
        self.log_probs[env_mask] = log_probs[env_mask]
        self.values[env_mask] = values[env_mask]
        self.legal_masks[env_mask] = legal_masks[env_mask]
        self.rewards[env_mask] = rewards[env_mask]
        self.score_targets[env_mask] = score_targets[env_mask]
        self.valid[env_mask] = True

    def accumulate_reward(self, learner_rewards: torch.Tensor) -> None:
        """Add perspective-corrected rewards to all valid pending transitions.

        Correctness assumption: non-mover envs have reward=0.0 from the engine,
        so accumulating learner_rewards across all valid envs is safe. If the
        engine ever emits shaping rewards for non-movers, this method would
        need to accept a per-env mask of which envs actually stepped.
        """
        self.rewards[self.valid] += learner_rewards[self.valid]

    def finalize(
        self,
        finalize_mask: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        """Close pending transitions and return data for buffer insertion.

        Returns None if no transitions need finalizing. Otherwise returns
        a dict with keys: obs, actions, log_probs, values, rewards, dones,
        legal_masks, score_targets, env_ids. All tensors are indexed by
        the finalized subset (not full num_envs).

        The finalize_mask may include envs where valid=False — these are
        safely skipped via the internal `to_finalize = finalize_mask & self.valid`
        guard.
        """
        to_finalize = finalize_mask & self.valid
        if not to_finalize.any():
            return None

        indices = to_finalize.nonzero(as_tuple=True)[0]
        result = {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "log_probs": self.log_probs[indices],
            "values": self.values[indices],
            "rewards": self.rewards[indices],
            "dones": dones[indices].float(),
            "legal_masks": self.legal_masks[indices],
            "score_targets": self.score_targets[indices],
            "env_ids": indices,
        }

        # Clear finalized state
        self.valid[to_finalize] = False
        self.rewards[to_finalize] = 0.0

        return result


def split_merge_step(
    obs: torch.Tensor,
    legal_masks: torch.Tensor,
    current_players: np.ndarray,
    learner_model: torch.nn.Module,
    opponent_model: torch.nn.Module,
    learner_side: int = 0,
    value_adapter: ValueHeadAdapter | None = None,
) -> SplitMergeResult:
    """Execute one step with split learner/opponent forward passes.

    Returns only learner-side data (log_probs, values, indices). The caller
    stores ONLY learner transitions in the rollout buffer.
    """
    num_envs = obs.shape[0]
    device = obs.device

    learner_mask = torch.tensor(current_players == learner_side, device=device)
    opponent_mask = ~learner_mask
    learner_indices = learner_mask.nonzero(as_tuple=True)[0]
    opponent_indices = opponent_mask.nonzero(as_tuple=True)[0]

    actions = torch.zeros(num_envs, dtype=torch.long, device=device)
    learner_log_probs = torch.zeros(0, device=device)
    learner_values = torch.zeros(0, device=device)

    # Learner forward pass (eval mode, no_grad for rollout collection).
    # The model stays in eval() — the caller (ppo.update) switches to train()
    # only during the backward pass. Toggling back to train() here would
    # corrupt BatchNorm running statistics with rollout-context updates.
    if learner_indices.numel() > 0:
        l_obs = obs[learner_indices]
        l_masks = legal_masks[learner_indices]

        learner_model.eval()
        with torch.no_grad():
            l_output = learner_model(l_obs)

        l_flat = l_output.policy_logits.reshape(l_obs.shape[0], -1)
        l_masked = l_flat.masked_fill(~l_masks, float("-inf"))
        l_probs = F.softmax(l_masked, dim=-1)
        l_dist = torch.distributions.Categorical(l_probs)
        l_actions = l_dist.sample()
        learner_log_probs = l_dist.log_prob(l_actions)

        if value_adapter is not None:
            learner_values = value_adapter.scalar_value_from_output(l_output.value_logits)
        else:
            learner_values = KataGoPPOAlgorithm.scalar_value(l_output.value_logits)

        actions[learner_indices] = l_actions

    # Opponent forward pass (always no_grad, eval mode)
    if opponent_indices.numel() > 0:
        o_obs = obs[opponent_indices]
        o_masks = legal_masks[opponent_indices]

        opponent_model.eval()
        with torch.no_grad():
            o_output = opponent_model(o_obs)

        o_flat = o_output.policy_logits.reshape(o_obs.shape[0], -1)
        o_masked = o_flat.masked_fill(~o_masks, float("-inf"))
        o_probs = F.softmax(o_masked, dim=-1)
        o_dist = torch.distributions.Categorical(o_probs)
        o_actions = o_dist.sample()

        actions[opponent_indices] = o_actions

    return SplitMergeResult(
        actions=actions,
        learner_mask=learner_mask,
        opponent_mask=opponent_mask,
        learner_log_probs=learner_log_probs,
        learner_values=learner_values,
        learner_indices=learner_indices,
    )


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
    def __init__(
        self, config: AppConfig, vecenv: Any = None,
        resume_mode: str = "rl",
        dist_ctx: DistributedContext | None = None,
    ) -> None:
        if resume_mode not in ("rl", "sl"):
            raise ValueError(f"resume_mode must be 'rl' or 'sl', got '{resume_mode}'")
        self._resume_mode = resume_mode
        self.config = config
        self.db_path = config.display.db_path

        self.dist_ctx = dist_ctx or get_distributed_context()

        if self.dist_ctx.is_main:
            init_db(self.db_path)
        self.device = self.dist_ctx.device

        if config.league is not None and self.dist_ctx.is_distributed:
            raise ValueError(
                "League mode is not yet supported with DDP. "
                "League mode uses split-merge rollout collection where buffer sizes "
                "can differ across ranks, causing NCCL allreduce deadlocks. "
                "Run without torchrun or remove [league] config."
            )

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

        if self.dist_ctx.is_distributed:
            if config.distributed.sync_batchnorm:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                logger.info("Converted BatchNorm layers to SyncBatchNorm")
            self.model = DDP(
                self.model,
                device_ids=[self.dist_ctx.local_rank] if self.device.type == "cuda" else None,
                output_device=self.dist_ctx.local_rank if self.device.type == "cuda" else None,
                find_unused_parameters=config.distributed.find_unused_parameters,
                gradient_as_bucket_view=config.distributed.gradient_as_bucket_view,
            )
            logger.info(
                "DDP wrapping complete: rank=%d, world_size=%d",
                self.dist_ctx.rank, self.dist_ctx.world_size,
            )
        else:
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                logger.info(
                    "Found %d GPUs; DataParallel skipped for KataGo (use DDP instead)",
                    gpu_count,
                )

        param_count = sum(p.numel() for p in self._base_model.parameters())
        logger.info(
            "Model: %s (%s), params: %d, device: %s, world_size: %d",
            config.model.display_name,
            config.model.architecture,
            param_count,
            self.device,
            self.dist_ctx.world_size,
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

        rl_warmup_epochs = rl_warmup_config.get("epochs", 0)
        rl_warmup_entropy = rl_warmup_config.get("entropy_bonus", 0.05)
        self._original_warmup_duration = rl_warmup_epochs  # fixed; used in _rotate_seat

        self.ppo = KataGoPPOAlgorithm(
            ppo_params, self._base_model, forward_model=self.model,
            warmup_epochs=rl_warmup_epochs, warmup_entropy=rl_warmup_entropy,
        )

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
        self.latest_values: list[float] = [0.0] * self.num_envs
        self.epoch = 0
        self.global_step = 0
        self._phase = "init"
        self._last_heartbeat = time.monotonic()

        # League setup (optional — only if [league] config is present)
        self.pool: OpponentPool | None = None
        self.sampler: OpponentSampler | None = None
        self._current_opponent: torch.nn.Module | None = None
        self._current_opponent_entry: OpponentEntry | None = None
        self._learner_entry_id: int | None = None

        if config.league is not None:
            league_dir = str(Path(config.training.checkpoint_dir) / "league")
            self.pool = OpponentPool(
                self.db_path, league_dir, max_pool_size=config.league.max_pool_size,
            )
            self.sampler = OpponentSampler(
                self.pool,
                historical_ratio=config.league.historical_ratio,
                current_best_ratio=config.league.current_best_ratio,
                elo_floor=config.league.elo_floor,
            )
            # Bootstrap snapshot so pool is never empty
            bootstrap_entry = self.pool.add_snapshot(
                self._base_model, config.model.architecture,
                dict(config.model.params), epoch=0,
            )
            self._learner_entry_id = bootstrap_entry.id
            logger.info(
                "League initialized: pool_size=%d, snapshot_interval=%d",
                config.league.max_pool_size, config.league.snapshot_interval,
            )

        self._check_resume()

    @property
    def _base_model(self) -> torch.nn.Module:
        """Unwrap DataParallel/DDP wrapper if present."""
        return self.model.module if hasattr(self.model, "module") else self.model

    def _check_resume(self) -> None:
        # NOTE: When resuming from an SL checkpoint into RL training, the SL
        # optimizer state is intentionally discarded. KataGoTrainingLoop creates
        # a fresh Adam optimizer. The SL optimizer has momentum from supervised
        # gradients that would fight the RL gradient signal. The RL warmup
        # elevated entropy (Plan D Task 3) compensates for the overconfident
        # SL policy by encouraging exploration in early RL epochs.

        # Rank 0 reads the DB to find checkpoint path; non-main ranks get None.
        checkpoint_path_str: str | None = None
        current_epoch: int = 0
        if self.dist_ctx.is_main:
            state = read_training_state(self.db_path)
            if state is not None and state.get("checkpoint_path"):
                cp = Path(state["checkpoint_path"])
                if cp.exists():
                    checkpoint_path_str = str(cp)
                    current_epoch = state.get("current_epoch", 0)

        # Broadcast checkpoint path to all ranks so everyone loads the same checkpoint.
        # In non-distributed mode, broadcast_object_list is not called.
        if self.dist_ctx.is_distributed:
            obj_list: list[object] = [checkpoint_path_str]
            dist.broadcast_object_list(obj_list, src=0)
            checkpoint_path_str = obj_list[0]  # type: ignore[assignment]

            # Also broadcast epoch so non-main ranks know where to resume
            meta_list: list[object] = [current_epoch]
            dist.broadcast_object_list(meta_list, src=0)
            current_epoch = meta_list[0]  # type: ignore[assignment]

        # ALL ranks load the checkpoint (critical for DDP weight consistency).
        # DDP does NOT re-broadcast weights after __init__() — if only rank 0
        # loads the checkpoint, other ranks train from random weights and
        # gradient allreduce averages nonsensical gradients.
        if checkpoint_path_str is not None:
            checkpoint_path = Path(checkpoint_path_str)
            skip_opt = self._resume_mode == "sl"
            logger.warning(
                "[rank %d] Resuming from checkpoint: %s (skip_optimizer=%s)",
                self.dist_ctx.rank, checkpoint_path, skip_opt,
            )
            meta = load_checkpoint(
                checkpoint_path,
                self._base_model,
                self.ppo.optimizer,
                expected_architecture=self.config.model.architecture,
                scheduler=self.lr_scheduler,
                grad_scaler=self.ppo.scaler,
                skip_optimizer=skip_opt,
                current_world_size=self.dist_ctx.world_size,
            )
            if skip_opt:
                # SL→RL: start RL from epoch 0 so warmup_epochs and
                # checkpoint numbering are RL-relative.
                self.epoch = 0
                self.global_step = 0
            else:
                self.epoch = meta["epoch"]
                self.global_step = meta["step"]
            return

        # Fresh start — only rank 0 writes training state to DB
        if self.dist_ctx.is_main:
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
        else:
            logger.info(
                "[rank %d] Non-main rank: skipping DB write for fresh training state",
                self.dist_ctx.rank,
            )

    def run(self, num_epochs: int, steps_per_epoch: int) -> None:
        reset_result = self.vecenv.reset()
        obs = torch.from_numpy(np.asarray(reset_result.observations)).to(self.device)
        legal_masks = torch.from_numpy(np.asarray(reset_result.legal_masks)).to(
            self.device
        )
        current_players = np.zeros(self.num_envs, dtype=np.uint8)

        start_epoch = self.epoch
        for epoch_i in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch_i

            # Sample opponent for this epoch (if league enabled)
            if self.sampler is not None:
                self._current_opponent_entry = self.sampler.sample()
                self._current_opponent = self.pool.load_opponent(
                    self._current_opponent_entry, device=str(self.device),
                )

            # Per-epoch win/loss/draw counters for Elo tracking
            win_acc = torch.zeros(1, dtype=torch.long, device=self.device)
            loss_acc = torch.zeros(1, dtype=torch.long, device=self.device)
            draw_acc = torch.zeros(1, dtype=torch.long, device=self.device)
            # Per-color win counters (black=0, white=1)
            black_wins = torch.zeros(1, dtype=torch.long, device=self.device)
            white_wins = torch.zeros(1, dtype=torch.long, device=self.device)

            self._phase = "rollout"
            self._force_heartbeat()
            for step_i in range(steps_per_epoch):
                self.global_step += 1

                if self._current_opponent is not None:
                    # Split-merge: learner vs opponent
                    pre_players = current_players.copy()
                    sm_result = split_merge_step(
                        obs=obs, legal_masks=legal_masks,
                        current_players=current_players,
                        learner_model=self.model,
                        opponent_model=self._current_opponent,
                        learner_side=0,
                    )
                    actions = sm_result.actions
                    action_list = actions.tolist()
                    step_result = self.vecenv.step(action_list)

                    current_players = np.asarray(step_result.current_players)

                    rewards = torch.from_numpy(np.asarray(step_result.rewards)).to(self.device)
                    terminated = torch.from_numpy(np.asarray(step_result.terminated)).to(self.device)
                    truncated = torch.from_numpy(np.asarray(step_result.truncated)).to(self.device)
                    dones = terminated | truncated

                    # Track wins/losses/draws for Elo
                    terminal_mask = dones.bool()
                    if terminal_mask.any():
                        t_rewards = rewards[terminal_mask]
                        win_acc += (t_rewards > 0).sum()
                        loss_acc += (t_rewards < 0).sum()
                        draw_acc += (t_rewards == 0).sum()

                        # Per-color wins: rewards are from last_mover perspective,
                        # pre_players tells us who moved this step.
                        t_players = torch.from_numpy(
                            pre_players[terminal_mask.cpu().numpy()]
                        ).to(self.device)
                        # Winner is last_mover when reward > 0, opponent when reward < 0
                        winner_is_black = (
                            ((t_rewards > 0) & (t_players == 0))
                            | ((t_rewards < 0) & (t_players == 1))
                        )
                        winner_is_white = (
                            ((t_rewards > 0) & (t_players == 1))
                            | ((t_rewards < 0) & (t_players == 0))
                        )
                        black_wins += winner_is_black.sum()
                        white_wins += winner_is_white.sum()

                    # Store ONLY learner transitions in the buffer
                    li = sm_result.learner_indices
                    if li.numel() > 0:
                        # Compute value_cats and score_targets for learner envs
                        li_terminal = dones[li].bool()
                        li_rewards = rewards[li]
                        li_value_cats = torch.full(
                            (li.numel(),), -1, dtype=torch.long, device=self.device,
                        )
                        li_value_cats[li_terminal & (li_rewards > 0)] = 0
                        li_value_cats[li_terminal & (li_rewards == 0)] = 1
                        li_value_cats[li_terminal & (li_rewards < 0)] = 2

                        # Per-step material balance for learner envs
                        material = torch.from_numpy(
                            np.asarray(step_result.step_metadata.material_balance, dtype=np.float32),
                        ).to(self.device)
                        li_score_targets = material[li] / self.score_norm

                        self.buffer.add(
                            obs[li], actions[li], sm_result.learner_log_probs,
                            sm_result.learner_values, li_rewards, dones[li],
                            legal_masks[li], li_value_cats, li_score_targets,
                            env_ids=li,
                        )
                else:
                    # No opponent: all envs are learner (original behavior)
                    actions, log_probs, values = self.ppo.select_actions(obs, legal_masks)
                    self.latest_values = values.tolist()
                    action_list = actions.tolist()
                    pre_players = current_players.copy()
                    step_result = self.vecenv.step(action_list)

                    current_players = np.asarray(step_result.current_players)

                    rewards = torch.from_numpy(np.asarray(step_result.rewards)).to(self.device)
                    terminated = torch.from_numpy(np.asarray(step_result.terminated)).to(self.device)
                    truncated = torch.from_numpy(np.asarray(step_result.truncated)).to(self.device)
                    dones = terminated | truncated

                    terminal_mask = dones.bool()
                    if terminal_mask.any():
                        t_rewards = rewards[terminal_mask]
                        # pre_players = who moved this step (last_mover).
                        # Reward is from last_mover's perspective.
                        t_players = torch.from_numpy(
                            pre_players[terminal_mask.cpu().numpy()]
                        ).to(self.device)
                        winner_is_black = (
                            ((t_rewards > 0) & (t_players == 0))
                            | ((t_rewards < 0) & (t_players == 1))
                        )
                        winner_is_white = (
                            ((t_rewards > 0) & (t_players == 1))
                            | ((t_rewards < 0) & (t_players == 0))
                        )
                        black_wins += winner_is_black.sum()
                        white_wins += winner_is_white.sum()

                    value_cats = torch.full(
                        (self.num_envs,), -1, dtype=torch.long, device=self.device,
                    )
                    value_cats[terminal_mask & (rewards > 0)] = 0
                    value_cats[terminal_mask & (rewards == 0)] = 1
                    value_cats[terminal_mask & (rewards < 0)] = 2

                    # Per-step material balance from the Rust engine, normalized.
                    # Every position gets a real score target — no NaN masking needed.
                    material = torch.from_numpy(
                        np.asarray(step_result.step_metadata.material_balance, dtype=np.float32),
                    ).to(self.device)
                    score_targets = material / self.score_norm

                    self.buffer.add(
                        obs, actions, log_probs, values, rewards, dones,
                        legal_masks, value_cats, score_targets,
                    )

                obs = torch.from_numpy(np.asarray(step_result.observations)).to(self.device)
                legal_masks = torch.from_numpy(np.asarray(step_result.legal_masks)).to(self.device)
                self._maybe_write_snapshots()
                self._maybe_update_heartbeat()

            # Bootstrap value for GAE
            self.ppo.forward_model.eval()
            with torch.no_grad():
                output = self.ppo.forward_model(obs)
                next_values = KataGoPPOAlgorithm.scalar_value(output.value_logits)
            self.ppo.forward_model.train()

            # Set epoch-dependent entropy coefficient
            self.ppo.current_entropy_coeff = self.ppo.get_entropy_coeff(epoch_i)
            if epoch_i == 0 or epoch_i == self.ppo.warmup_epochs:
                logger.info(
                    "Entropy coefficient: %.4f (warmup=%d, epoch=%d)",
                    self.ppo.current_entropy_coeff, self.ppo.warmup_epochs, epoch_i,
                )

            self._phase = "update"
            self._force_heartbeat()
            losses = self.ppo.update(
                self.buffer, next_values,
                heartbeat_fn=self._maybe_update_heartbeat,
            )
            self._phase = "rollout"

            if losses["value_loss"] == 0.0:
                logger.info(
                    "Epoch %d: value_loss=0.0 (likely no terminal steps — "
                    "value head received no gradient this epoch)", epoch_i,
                )

            # LR scheduler logic — ALL ranks must participate with the SAME
            # monitor value to keep LR state synchronized across ranks.
            if epoch_i == self.ppo.warmup_epochs and self.lr_scheduler is not None:
                self.lr_scheduler.best = self.lr_scheduler.mode_worse
                self.lr_scheduler.num_bad_epochs = 0
                if self.dist_ctx.is_main:
                    logger.info("LR scheduler fully reset at warmup boundary (epoch %d)", epoch_i)

            if self.lr_scheduler is not None:
                monitor_value = losses.get("value_loss")
                if monitor_value is not None:
                    # Synchronize monitor value across ranks so all schedulers
                    # step identically and maintain the same LR state.
                    if self.dist_ctx.is_distributed:
                        monitor_tensor = torch.tensor(
                            monitor_value, device=self.device,
                        )
                        dist.all_reduce(monitor_tensor, op=dist.ReduceOp.AVG)
                        monitor_value = monitor_tensor.item()

                    old_lr = self.ppo.optimizer.param_groups[0]["lr"]
                    self.lr_scheduler.step(monitor_value)
                    new_lr = self.ppo.optimizer.param_groups[0]["lr"]
                    if new_lr != old_lr and self.dist_ctx.is_main:
                        logger.info("LR reduced: %.6f -> %.6f (value_loss=%.4f)",
                                    old_lr, new_lr, monitor_value)

            # Materialise GPU counters → CPU once per epoch (all ranks, to release GPU memory)
            win_count = win_acc.item()
            loss_count = loss_acc.item()
            draw_count = draw_acc.item()
            black_win_count = black_wins.item()
            white_win_count = white_wins.item()

            # Elo tracking (league mode, rank 0 only)
            if self.dist_ctx.is_main:
                total_games = win_count + loss_count + draw_count
                if (self.pool is not None and self._current_opponent_entry is not None
                        and total_games > 0):
                    learner_entry = self.pool._get_entry(self._learner_entry_id)
                    if learner_entry is not None:
                        self.pool.record_result(
                            epoch=epoch_i, learner_id=learner_entry.id,
                            opponent_id=self._current_opponent_entry.id,
                            wins=win_count, losses=loss_count, draws=draw_count,
                        )
                        result_score = (win_count + 0.5 * draw_count) / total_games
                        k = self.config.league.elo_k_factor if self.config.league else 32.0
                        new_learner_elo, new_opp_elo = compute_elo_update(
                            learner_entry.elo_rating,
                            self._current_opponent_entry.elo_rating,
                            result=result_score, k=k,
                        )
                        self.pool.update_elo(learner_entry.id, new_learner_elo, epoch=self.epoch)
                        self.pool.update_elo(self._current_opponent_entry.id, new_opp_elo, epoch=self.epoch)
                        logger.info(
                            "Elo: learner %.0f->%.0f, opponent(id=%d) %.0f->%.0f | W=%d L=%d D=%d",
                            learner_entry.elo_rating, new_learner_elo,
                            self._current_opponent_entry.id,
                            self._current_opponent_entry.elo_rating, new_opp_elo,
                            win_count, loss_count, draw_count,
                        )

            if self.dist_ctx.is_main:
                # Seat rotation (takes priority — includes its own snapshot)
                rotating_this_epoch = (
                    self.config.league is not None and self.pool is not None
                    and (epoch_i + 1) % self.config.league.epochs_per_seat == 0
                )
                if rotating_this_epoch:
                    self._rotate_seat(epoch_i)

                # Periodic pool snapshot (skip if rotation already snapshotted)
                if (self.pool is not None and self.config.league is not None
                        and (epoch_i + 1) % self.config.league.snapshot_interval == 0
                        and not rotating_this_epoch):
                    self.pool.add_snapshot(
                        self._base_model, self.config.model.architecture,
                        dict(self.config.model.params), epoch=epoch_i + 1,
                    )

            if self.dist_ctx.is_main:
                # Metrics and logging
                ep_completed = getattr(self.vecenv, "episodes_completed", 0)
                total_games = win_count + loss_count + draw_count
                metrics: dict[str, object] = {
                    "epoch": epoch_i, "step": self.global_step,
                    "policy_loss": losses["policy_loss"],
                    "value_loss": losses["value_loss"],
                    "entropy": losses["entropy"],
                    "gradient_norm": losses["gradient_norm"],
                    "episodes_completed": ep_completed,
                    "avg_episode_length": getattr(self.vecenv, "mean_episode_length", None),
                    "truncation_rate": getattr(self.vecenv, "truncation_rate", None),
                    "draw_rate": getattr(self.vecenv, "draw_rate", None),
                    "win_rate": (
                        (win_count + 0.5 * draw_count) / total_games
                        if total_games > 0 else None
                    ),
                    "black_win_rate": (
                        black_win_count / total_games if total_games > 0 else None
                    ),
                    "white_win_rate": (
                        white_win_count / total_games if total_games > 0 else None
                    ),
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
                    epoch_i, self.global_step, losses["policy_loss"],
                    losses["value_loss"], losses["score_loss"], losses["entropy"],
                )

            if (epoch_i + 1) % self.config.training.checkpoint_interval == 0:
                # Barrier ensures all ranks finish PPO update before checkpoint write
                if self.dist_ctx.is_distributed:
                    dist.barrier()

                if self.dist_ctx.is_main:
                    ckpt_path = Path(self.config.training.checkpoint_dir) / f"epoch_{epoch_i:05d}.pt"
                    try:
                        save_checkpoint(
                            ckpt_path, self._base_model, self.ppo.optimizer,
                            epoch_i + 1, self.global_step,
                            architecture=self.config.model.architecture,
                            scheduler=self.lr_scheduler,
                            grad_scaler=self.ppo.scaler,
                            world_size=self.dist_ctx.world_size,
                        )
                        logger.info("Checkpoint saved: %s", ckpt_path)
                    except Exception:
                        logger.exception("Failed to save checkpoint %s — continuing", ckpt_path)
                    try:
                        update_training_progress(
                            self.db_path, epoch_i + 1, self.global_step, str(ckpt_path),
                        )
                    except Exception:
                        logger.exception("Failed to record checkpoint path in DB — continuing")

                # Barrier after save — all ranks proceed together
                if self.dist_ctx.is_distributed:
                    dist.barrier()

    def _rotate_seat(self, epoch: int) -> None:
        """Save current learner weights and reset optimizer for the next seat."""
        new_entry = self.pool.add_snapshot(
            self._base_model, self.config.model.architecture,
            dict(self.config.model.params), epoch=epoch + 1,
        )

        # B5 fix: update learner entry ID so Elo tracks the current snapshot
        self._learner_entry_id = new_entry.id

        # Reset optimizer (fresh Adam — old momentum fights new gradient signal)
        self.ppo.optimizer = torch.optim.Adam(
            self.ppo.model.parameters(), lr=self.ppo.params.learning_rate,
        )

        # B1 fix: recreate LR scheduler pointing at the NEW optimizer
        if self.lr_scheduler is not None and self.config.league is not None:
            algo_params = dict(self.config.training.algorithm_params)
            lr_config = algo_params.get("lr_schedule", {})
            self.lr_scheduler = create_lr_scheduler(
                self.ppo.optimizer,
                schedule_type=lr_config.get("type", "plateau"),
                factor=lr_config.get("factor", 0.5),
                patience=lr_config.get("patience", 50),
                min_lr=lr_config.get("min_lr", 1e-5),
            )

        # B2 fix: extend warmup relative to the rotation point.
        # Uses the ORIGINAL warmup duration (fixed at init) to avoid
        # unbounded accumulation across multiple rotations.
        self.ppo.warmup_epochs = epoch + 1 + self._original_warmup_duration

        logger.info(
            "Seat rotation at epoch %d: optimizer reset, warmup extended to epoch %d, "
            "learner_entry=%d",
            epoch, self.ppo.warmup_epochs, self._learner_entry_id,
        )

    def _force_heartbeat(self) -> None:
        """Write phase transition to DB immediately (skips timer check)."""
        if not self.dist_ctx.is_main:
            return
        self._last_heartbeat = time.monotonic()
        try:
            update_training_progress(
                self.db_path, self.epoch, self.global_step, phase=self._phase,
            )
        except Exception:
            pass  # best-effort — don't let a heartbeat failure stop training

    def _maybe_update_heartbeat(self) -> None:
        if not self.dist_ctx.is_main:
            return
        now = time.monotonic()
        if now - self._last_heartbeat >= 10.0:
            self._last_heartbeat = now
            update_training_progress(
                self.db_path, self.epoch, self.global_step, phase=self._phase,
            )

    def _maybe_write_snapshots(self) -> None:
        if not self.dist_ctx.is_main:
            return
        if self.moves_per_minute <= 0:
            return
        now = time.monotonic()
        interval = 60.0 / self.moves_per_minute
        if now - self._last_snapshot_time < interval:
            return
        self._last_snapshot_time = now

        if not hasattr(self.vecenv, "get_spectator_data"):
            return
        spectator_data = self.vecenv.get_spectator_data()
        snapshots = []
        for i, game_data in enumerate(spectator_data):
            snapshots.append({
                "game_id": i,
                "board_json": json.dumps(game_data.get("board", [])),
                "hands_json": json.dumps(game_data.get("hands", {})),
                "current_player": game_data.get("current_player", "black"),
                "ply": game_data.get("ply", 0),
                "is_over": int(game_data.get("is_over", False)),
                "result": game_data.get("result", "in_progress"),
                "sfen": game_data.get("sfen", ""),
                "in_check": int(game_data.get("in_check", False)),
                "move_history_json": json.dumps(game_data.get("move_history", [])),
                "value_estimate": (
                    self.latest_values[i]
                    if i < len(self.latest_values)
                    else 0.0
                ),
            })
        write_game_snapshots(self.db_path, snapshots)


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    dist_ctx = get_distributed_context()
    setup_distributed(dist_ctx)

    try:
        parser = argparse.ArgumentParser(description="Keisei training")
        parser.add_argument("config", type=Path, help="Path to TOML config file")
        parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
        parser.add_argument("--steps-per-epoch", type=int, default=None,
                            help="Steps per epoch (default: max_ply from config)")
        parser.add_argument("--seed", type=int, default=42,
                            help="Base random seed (each rank adds its rank index)")
        args = parser.parse_args()

        # Set all RNG seeds: base_seed + rank for different-but-reproducible rollouts.
        # Seeds torch, numpy, and Python stdlib RNG.
        seed_all_ranks(args.seed + dist_ctx.rank)

        config = load_config(args.config)
        steps = args.steps_per_epoch or config.training.max_ply
        loop = KataGoTrainingLoop(config, dist_ctx=dist_ctx)
        loop.run(num_epochs=args.epochs, steps_per_epoch=steps)
    finally:
        cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
