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
from keisei.training.tournament import LeagueTournament

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


def _compute_value_cats(
    rewards: torch.Tensor,
    terminal_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Assign value-head categories from terminal rewards.

    Returns a tensor of {-1=ignore, 0=win, 1=draw, 2=loss}.
    Only genuinely terminal positions (not truncated) receive labels.
    Engine produces exact integer-valued rewards (0.0 for draws, ±1.0
    for wins/losses), so float equality with == 0 is safe here.
    """
    cats = torch.full((rewards.numel(),), -1, dtype=torch.long, device=device)
    cats[terminal_mask & (rewards > 0)] = 0
    cats[terminal_mask & (rewards == 0)] = 1
    cats[terminal_mask & (rewards < 0)] = 2
    return cats


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
    learner_side: int | np.ndarray,
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
    learner_side: int | np.ndarray,
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

        Raises RuntimeError if any env in env_mask already has a valid
        pending transition (indicates a protocol bug — finalize must be
        called before create for the same env).
        """
        if (env_mask & self.valid).any():
            raise RuntimeError(
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
        terminated: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        """Close pending transitions and return data for buffer insertion.

        Returns None if no transitions need finalizing. Otherwise returns
        a dict with keys: obs, actions, log_probs, values, rewards, dones,
        terminated, legal_masks, score_targets, env_ids. All tensors are
        indexed by the finalized subset (not full num_envs).

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
            "terminated": terminated[indices].float(),
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
    opponent_model: torch.nn.Module | None = None,
    opponent_models: dict[int, torch.nn.Module] | None = None,
    env_opponent_ids: np.ndarray | None = None,
    learner_side: int | np.ndarray = 0,
    value_adapter: ValueHeadAdapter | None = None,
) -> SplitMergeResult:
    """Execute one step with split learner/opponent forward passes.

    Supports both single-opponent (legacy) and multi-opponent (cohort) modes:
    - Legacy: pass opponent_model=<model>
    - Cohort: pass opponent_models={id: model, ...} and env_opponent_ids=<array>

    Returns only learner-side data (log_probs, values, indices). The caller
    stores ONLY learner transitions in the rollout buffer.
    """
    # Normalize to multi-opponent path
    if opponent_models is None and opponent_model is not None:
        active_opponents: dict[int, torch.nn.Module] = {0: opponent_model}
        active_env_ids: np.ndarray | None = None
    elif opponent_models is not None:
        active_opponents = opponent_models
        active_env_ids = env_opponent_ids
    else:
        raise ValueError("Must provide either opponent_model or opponent_models")

    num_envs = obs.shape[0]
    device = obs.device

    # Use from_numpy to avoid the extra allocation that torch.tensor() incurs.
    cmp_result = np.ascontiguousarray(current_players == learner_side)
    learner_mask = torch.from_numpy(cmp_result).to(device=device, dtype=torch.bool)
    opponent_mask = ~learner_mask
    learner_indices = learner_mask.nonzero(as_tuple=True)[0]

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
            learner_values = value_adapter.scalar_value_blended(
                l_output.value_logits, l_output.score_lead,
            )
        else:
            learner_values = KataGoPPOAlgorithm.scalar_value(l_output.value_logits)

        actions[learner_indices] = l_actions

    # Opponent forward passes (always no_grad, eval mode).
    # When the opponent lives on a different GPU (e.g. cuda:1), move
    # observations there for inference, then move actions back.
    opponent_mask_np = opponent_mask.cpu().numpy()
    for opp_id, model in active_opponents.items():
        if active_env_ids is not None:
            opp_env_mask = (active_env_ids == opp_id) & opponent_mask_np
        else:
            opp_env_mask = opponent_mask_np

        if not opp_env_mask.any():
            continue

        indices = np.flatnonzero(opp_env_mask)
        idx_tensor = torch.from_numpy(indices.astype(np.int64)).to(device)

        o_obs = obs[idx_tensor]
        o_masks = legal_masks[idx_tensor]

        # Detect cross-device opponent (e.g. learner on cuda:0, opponent on cuda:1)
        try:
            opp_device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            opp_device = device  # fallback: assume same device
        cross_device = isinstance(opp_device, torch.device) and opp_device != device
        if cross_device:
            o_obs = o_obs.to(opp_device)
            o_masks = o_masks.to(opp_device)

        model.eval()
        with torch.no_grad():
            o_output = model(o_obs)

        o_flat = o_output.policy_logits.reshape(o_obs.shape[0], -1)
        o_masked = o_flat.masked_fill(~o_masks, float("-inf"))
        o_probs = F.softmax(o_masked, dim=-1)
        o_dist = torch.distributions.Categorical(o_probs)
        o_actions = o_dist.sample()

        if cross_device:
            o_actions = o_actions.to(device)
        actions[idx_tensor] = o_actions

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

        from keisei.training.value_adapter import get_value_adapter
        _model_contract = "multi_head" if config.model.architecture in _KATAGO_ARCHITECTURES else "scalar"
        self.value_adapter = get_value_adapter(
            model_contract=_model_contract,
            lambda_value=ppo_params.lambda_value,
            lambda_score=ppo_params.lambda_score,
            score_blend_alpha=ppo_params.score_blend_alpha,
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
        self._tournament: LeagueTournament | None = None

        # Per-env opponent state (Change 3) — initialized at epoch start when enabled
        self._opponent_models: dict[int, torch.nn.Module] | None = None
        self._env_opponent_ids: np.ndarray | None = None
        self._cached_entries: list[OpponentEntry] = []
        self._cached_entries_by_id: dict[int, OpponentEntry] = {}
        self._opponent_results: dict[int, list[int]] | None = None

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

            # Background tournament for Elo calibration (optional)
            if config.league.tournament_enabled and self.dist_ctx.is_main:
                tournament_device = (
                    config.league.tournament_device
                    or config.league.opponent_device
                    or str(self.device)
                )
                self._tournament = LeagueTournament(
                    db_path=self.db_path,
                    league_dir=league_dir,
                    device=tournament_device,
                    num_envs=config.league.tournament_num_envs,
                    games_per_match=config.league.tournament_games_per_match,
                    k_factor=config.league.tournament_k_factor,
                    pause_seconds=config.league.tournament_pause_seconds,
                    max_ply=config.training.max_ply,
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
                            "model": {
                                "architecture": self.config.model.architecture,
                                "params": dict(self.config.model.params),
                            },
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
        # Store total planned epochs in DB so the dashboard can show progress
        if self.dist_ctx.is_main:
            from keisei.db import _connect
            conn = _connect(self.db_path)
            try:
                conn.execute(
                    "UPDATE training_state SET total_epochs = ? WHERE id = 1",
                    (self.epoch + num_epochs,),
                )
                conn.commit()
            except Exception:
                pass  # Column may not exist in old DBs
            finally:
                conn.close()

        # Start background tournament if configured
        if self._tournament is not None:
            self._tournament.start()

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
                opp_device_cfg = (
                    self.config.league.opponent_device
                    if self.config.league and self.config.league.opponent_device
                    else None
                )
                # Validate the configured opponent device exists; fall back to
                # learner device if it doesn't (e.g. single-GPU machine).
                if opp_device_cfg and opp_device_cfg.startswith("cuda"):
                    try:
                        idx = int(opp_device_cfg.split(":")[-1]) if ":" in opp_device_cfg else 0
                        if idx >= torch.cuda.device_count():
                            if not getattr(self, "_opp_device_warned", False):
                                logger.warning(
                                    "opponent_device=%s not available (%d GPUs), "
                                    "falling back to %s",
                                    opp_device_cfg, torch.cuda.device_count(), self.device,
                                )
                                self._opp_device_warned = True
                            opp_device_cfg = None
                    except ValueError:
                        opp_device_cfg = None
                opp_device = opp_device_cfg or str(self.device)

                # Cache entries once — reused for sampling AND opponent loading
                self._cached_entries = self.pool.list_entries()
                self._cached_entries_by_id = {e.id: e for e in self._cached_entries}

                use_per_env_opps = (
                    self.config.league is not None
                    and self.config.league.per_env_opponents
                    and len(self._cached_entries) > 0
                )

                if use_per_env_opps:
                    # Memory cleanup: release previous epoch's models.
                    # Safe at epoch start: ppo.update() backward pass is a
                    # CUDA sync point, so all prior kernels have completed.
                    if self._opponent_models is not None:
                        del self._opponent_models
                        self._opponent_models = None
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    self._opponent_models = self.pool.load_all_opponents(device=opp_device)

                    # Filter cached entries to only those whose models loaded.
                    # Without this, sample_from could assign an entry whose
                    # checkpoint was corrupt/missing → silent wrong actions.
                    self._cached_entries = [
                        e for e in self._cached_entries if e.id in self._opponent_models
                    ]
                    self._cached_entries_by_id = {e.id: e for e in self._cached_entries}

                    # Per-env opponent assignment
                    self._env_opponent_ids = np.zeros(self.num_envs, dtype=np.int64)
                    for env_i in range(self.num_envs):
                        entry = self.sampler.sample_from(self._cached_entries)
                        self._env_opponent_ids[env_i] = entry.id

                    # Per-opponent W/L/D tracking for epoch-end Elo update
                    self._opponent_results = {
                        eid: [0, 0, 0] for eid in self._opponent_models
                    }

                    # Reuse an already-loaded model for _current_opponent (used by
                    # PendingTransitions guard, sign_correct_bootstrap, etc.).
                    # Do NOT call load_opponent again — saves 14MB VRAM + disk I/O.
                    opp_entry_id = self._current_opponent_entry.id
                    if opp_entry_id in self._opponent_models:
                        self._current_opponent = self._opponent_models[opp_entry_id]
                    else:
                        # Entry's checkpoint was corrupt/missing — use any loaded model
                        self._current_opponent = next(iter(self._opponent_models.values()))
                else:
                    self._opponent_models = None
                    self._env_opponent_ids = None
                    self._opponent_results = None
                    self._current_opponent = self.pool.load_opponent(
                        self._current_opponent_entry, device=opp_device,
                    )

            # One-time LR scheduler reset for league mode to prevent value-loss
            # spike from triggering premature LR reduction after reward-collection
            # fix. Safe to remove once no pre-fix checkpoints are in use.
            # Uses ReduceLROnPlateau internal attributes (best, mode_worse,
            # num_bad_epochs) — guarded by hasattr for forward compatibility.
            if (self._current_opponent is not None
                    and self.lr_scheduler is not None
                    and epoch_i == start_epoch
                    and start_epoch > 0
                    and hasattr(self.lr_scheduler, "best")):
                self.lr_scheduler.best = self.lr_scheduler.mode_worse
                self.lr_scheduler.num_bad_epochs = 0
                if self.dist_ctx.is_main:
                    logger.info(
                        "LR scheduler reset at epoch %d for post-fix checkpoint continuity",
                        epoch_i,
                    )

            if (self._current_opponent is not None
                    and epoch_i == start_epoch
                    and start_epoch > 0
                    and self.dist_ctx.is_main):
                logger.warning(
                    "Resuming league training with corrected reward collection. "
                    "Elo ratings from epochs before this fix may be inaccurate "
                    "(opponent-turn terminals were previously miscounted). "
                    "Elo will self-correct over subsequent epochs."
                )

            # Per-epoch win/loss/draw counters for Elo tracking
            win_acc = torch.zeros(1, dtype=torch.long, device=self.device)
            loss_acc = torch.zeros(1, dtype=torch.long, device=self.device)
            draw_acc = torch.zeros(1, dtype=torch.long, device=self.device)
            terminated_count = 0
            truncated_count = 0
            # Per-color win counters (black=0, white=1)
            black_wins = torch.zeros(1, dtype=torch.long, device=self.device)
            white_wins = torch.zeros(1, dtype=torch.long, device=self.device)

            # Per-env color randomization (config-gated).
            # Invariant: learner_side[env] always reflects the color for
            # the game CURRENTLY RUNNING in that env. Re-randomization
            # happens exclusively on dones, never mid-game.
            use_color_rand = (
                self.config.league is not None
                and self.config.league.color_randomization
                and self._current_opponent is not None
            )
            if use_color_rand:
                learner_side = np.random.randint(0, 2, size=self.num_envs, dtype=np.uint8)
                learner_side_t = torch.from_numpy(learner_side.copy()).to(self.device)
                if epoch_i == start_epoch and self.dist_ctx.is_main:
                    logger.info(
                        "Color randomization enabled: win_rate now reflects "
                        "both colors (was black-only previously)"
                    )
            else:
                learner_side = 0
                learner_side_t = None  # only used when use_color_rand is True
            pending: PendingTransitions | None = None
            _scratch_log_probs = torch.zeros(self.num_envs, device=self.device)
            _scratch_values = torch.zeros(self.num_envs, device=self.device)
            if self._current_opponent is not None:
                obs_channels = obs.shape[1]
                action_space = self.buffer.action_space
                pending = PendingTransitions(
                    self.num_envs, (obs_channels, 9, 9), action_space, self.device,
                )

            self._phase = "rollout"
            self._force_heartbeat()
            for step_i in range(steps_per_epoch):
                self.global_step += 1

                if self._current_opponent is not None:
                    assert pending is not None
                    pre_players = current_players.copy()
                    # GPU copy of pre_players avoids .cpu().numpy() sync for indexing
                    pre_players_t = torch.from_numpy(pre_players).to(self.device)
                    learner_moved = pre_players_t == (learner_side_t if use_color_rand else learner_side)

                    # Split-merge: learner vs opponent
                    if self._opponent_models and self._env_opponent_ids is not None:
                        sm_result = split_merge_step(
                            obs=obs, legal_masks=legal_masks,
                            current_players=current_players,
                            learner_model=self.model,
                            opponent_models=self._opponent_models,
                            env_opponent_ids=self._env_opponent_ids,
                            learner_side=learner_side,
                            value_adapter=self.value_adapter,
                        )
                    else:
                        sm_result = split_merge_step(
                            obs=obs, legal_masks=legal_masks,
                            current_players=current_players,
                            learner_model=self.model,
                            opponent_model=self._current_opponent,
                            learner_side=learner_side,
                            value_adapter=self.value_adapter,
                        )
                    actions = sm_result.actions
                    action_list = actions.tolist()
                    step_result = self.vecenv.step(action_list)

                    current_players = np.asarray(step_result.current_players)
                    current_players_t = torch.from_numpy(current_players).to(self.device)
                    learner_next = current_players_t == (learner_side_t if use_color_rand else learner_side)

                    rewards = torch.from_numpy(np.asarray(step_result.rewards)).to(self.device)
                    terminated = torch.from_numpy(np.asarray(step_result.terminated)).to(self.device)
                    truncated = torch.from_numpy(np.asarray(step_result.truncated)).to(self.device)
                    dones = terminated | truncated

                    terminated_count += terminated.bool().sum().item()
                    truncated_count += (truncated.bool() & ~terminated.bool()).sum().item()

                    # Convert rewards to learner perspective
                    learner_rewards = to_learner_perspective(rewards, pre_players, learner_side)

                    # Track wins/losses/draws from learner perspective
                    terminal_mask = terminated.bool()
                    if terminal_mask.any():
                        t_rewards = learner_rewards[terminal_mask]
                        # Engine produces exact integer-valued rewards (±1.0 or 0.0),
                        # so float equality with == 0 is safe for draw detection.
                        win_acc += (t_rewards > 0).sum()
                        loss_acc += (t_rewards < 0).sum()
                        draw_acc += (t_rewards == 0).sum()

                        # Per-color wins: use pre_players_t (last_mover) + raw rewards
                        # (from last_mover POV) to determine winner's color.
                        t_raw_rewards = rewards[terminal_mask]
                        t_players = pre_players_t[terminal_mask]
                        winner_is_black = (
                            ((t_raw_rewards > 0) & (t_players == 0))
                            | ((t_raw_rewards < 0) & (t_players == 1))
                        )
                        winner_is_white = (
                            ((t_raw_rewards > 0) & (t_players == 1))
                            | ((t_raw_rewards < 0) & (t_players == 0))
                        )
                        black_wins += winner_is_black.sum()
                        white_wins += winner_is_white.sum()

                    # --- Pending transition protocol ---
                    # Steps 1-2 finalize transitions from PRIOR steps.
                    # Steps 3-4 create and possibly finalize transitions from THIS step.
                    # These populations are disjoint. Do not reorder.

                    # 1. Accumulate rewards into existing pending transitions
                    pending.accumulate_reward(learner_rewards)

                    # 2. Finalize resolved pending transitions:
                    #    - Terminal: game ended (on any player's move)
                    #    - Non-terminal return: opponent moved, turn returns to learner
                    finalize_mask = pending.valid & (dones.bool() | learner_next)
                    finalized = pending.finalize(finalize_mask, dones, terminated)

                    if finalized is not None:
                        f_value_cats = _compute_value_cats(
                            finalized["rewards"], finalized["terminated"].bool(), self.device,
                        )
                        self.buffer.add(
                            finalized["obs"], finalized["actions"],
                            finalized["log_probs"], finalized["values"],
                            finalized["rewards"], finalized["dones"],
                            finalized["terminated"],
                            finalized["legal_masks"], f_value_cats,
                            finalized["score_targets"],
                            env_ids=finalized["env_ids"],
                        )

                    # 3. Create new pending for envs where learner just moved
                    if learner_moved.any():
                        li = sm_result.learner_indices
                        # Scatter compact learner data to pre-allocated scratch tensors
                        _scratch_log_probs.zero_()
                        _scratch_values.zero_()
                        if li.numel() > 0:
                            _scratch_log_probs[li] = sm_result.learner_log_probs
                            _scratch_values[li] = sm_result.learner_values

                        material = torch.from_numpy(
                            np.asarray(step_result.step_metadata.material_balance, dtype=np.float32),
                        ).to(self.device)
                        full_score_targets = material / self.score_norm

                        pending.create(
                            learner_moved, obs, actions, _scratch_log_probs,
                            _scratch_values, legal_masks, learner_rewards,
                            full_score_targets,
                        )

                        # 4. Immediately finalize if learner's own move was terminal
                        imm_terminal = learner_moved & dones.bool()
                        if imm_terminal.any():
                            imm_finalized = pending.finalize(imm_terminal, dones, terminated)
                            if imm_finalized is not None:
                                imm_value_cats = _compute_value_cats(
                                    imm_finalized["rewards"], imm_finalized["terminated"].bool(),
                                    self.device,
                                )
                                self.buffer.add(
                                    imm_finalized["obs"], imm_finalized["actions"],
                                    imm_finalized["log_probs"], imm_finalized["values"],
                                    imm_finalized["rewards"], imm_finalized["dones"],
                                    imm_finalized["terminated"],
                                    imm_finalized["legal_masks"], imm_value_cats,
                                    imm_finalized["score_targets"],
                                    env_ids=imm_finalized["env_ids"],
                                )

                    # --- Dones processing order (Changes 2+3) ---
                    # 1. Finalize pending transitions (above — existing protocol)
                    # 2. Per-opponent Elo result tracking + re-sample opponent
                    # 3. Re-randomize learner color
                    # Steps 2-3 set up state for the NEXT game. Step 1 consumes
                    # state from the COMPLETED game. Do not reorder.

                    # Compute done mask once for both Changes 2+3 blocks below.
                    # Avoids redundant dones.bool() / .cpu().numpy() GPU syncs.
                    done_bool = dones.bool()
                    any_done = done_bool.any()
                    done_np: np.ndarray | None = None
                    done_idx_np: np.ndarray | None = None
                    if any_done:
                        done_np = done_bool.cpu().numpy()
                        done_idx_np = np.flatnonzero(done_np)

                    # Per-env opponent: track results and re-sample on done (Change 3).
                    if (self._opponent_results is not None
                            and self._env_opponent_ids is not None
                            and any_done):
                        assert done_np is not None and done_idx_np is not None

                        # Batch-extract rewards to avoid per-env GPU syncs.
                        # One .cpu() call instead of N .item() calls.
                        done_rewards_np = learner_rewards[done_bool].cpu().numpy()
                        done_terminal_np = terminated.bool().cpu().numpy()[done_idx_np]
                        done_opp_ids = self._env_opponent_ids[done_idx_np]

                        for i, env_i in enumerate(done_idx_np):
                            opp_id = int(done_opp_ids[i])
                            if opp_id not in self._opponent_results:
                                continue
                            if done_terminal_np[i]:
                                lr = done_rewards_np[i]
                                if lr > 0:
                                    self._opponent_results[opp_id][0] += 1  # win
                                elif lr < 0:
                                    self._opponent_results[opp_id][1] += 1  # loss
                                else:
                                    self._opponent_results[opp_id][2] += 1  # draw

                            # Re-sample opponent for next game
                            new_entry = self.sampler.sample_from(self._cached_entries)
                            self._env_opponent_ids[env_i] = new_entry.id

                    # Re-randomize learner color for completed games (Change 2).
                    # This sets up state for the NEXT game — must happen AFTER
                    # pending transition finalization and Elo attribution.
                    if use_color_rand and any_done:
                        assert done_np is not None and done_idx_np is not None
                        new_sides = np.random.randint(
                            0, 2, size=int(done_np.sum()), dtype=np.uint8,
                        )
                        learner_side[done_np] = new_sides
                        done_indices_t = torch.from_numpy(
                            done_idx_np.astype(np.int64),
                        ).to(self.device)
                        # CRITICAL: new_sides must stay uint8 to match learner_side_t dtype.
                        # done_indices_t is int64 (for indexing), but scattered VALUES
                        # must match destination tensor dtype (uint8).
                        learner_side_t[done_indices_t] = torch.from_numpy(
                            new_sides,  # uint8 — matches learner_side_t dtype
                        ).to(self.device)
                else:
                    # No opponent: all envs are learner (original behavior)
                    actions, log_probs, values = self.ppo.select_actions(
                        obs, legal_masks, value_adapter=self.value_adapter,
                    )
                    self.latest_values = values.tolist()
                    action_list = actions.tolist()
                    pre_players = current_players.copy()
                    step_result = self.vecenv.step(action_list)

                    current_players = np.asarray(step_result.current_players)

                    rewards = torch.from_numpy(np.asarray(step_result.rewards)).to(self.device)
                    terminated = torch.from_numpy(np.asarray(step_result.terminated)).to(self.device)
                    truncated = torch.from_numpy(np.asarray(step_result.truncated)).to(self.device)
                    dones = terminated | truncated

                    terminated_count += terminated.bool().sum().item()
                    truncated_count += (truncated.bool() & ~terminated.bool()).sum().item()

                    terminal_mask = terminated.bool()
                    if terminal_mask.any():
                        t_rewards = rewards[terminal_mask]
                        # Engine produces exact integer-valued rewards (±1.0 or 0.0),
                        # so float equality with == 0 is safe for draw detection.
                        win_acc += (t_rewards > 0).sum()
                        loss_acc += (t_rewards < 0).sum()
                        draw_acc += (t_rewards == 0).sum()

                        # pre_players_t = who moved this step (last_mover), on GPU.
                        # Reward is from last_mover's perspective.
                        pre_players_t = torch.from_numpy(pre_players).to(self.device)
                        t_players = pre_players_t[terminal_mask]
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

                    value_cats = _compute_value_cats(rewards, terminal_mask, self.device)

                    # Per-step material balance from the Rust engine, normalized.
                    # Every position gets a real score target — no NaN masking needed.
                    material = torch.from_numpy(
                        np.asarray(step_result.step_metadata.material_balance, dtype=np.float32),
                    ).to(self.device)
                    score_targets = material / self.score_norm

                    self.buffer.add(
                        obs, actions, log_probs, values, rewards, dones,
                        terminated, legal_masks, value_cats, score_targets,
                    )

                obs = torch.from_numpy(np.asarray(step_result.observations)).to(self.device)
                legal_masks = torch.from_numpy(np.asarray(step_result.legal_masks)).to(self.device)
                self._maybe_write_snapshots()
                self._maybe_update_heartbeat()

            # Finalize any remaining pending transitions at epoch end.
            # These are learner moves whose games did not resolve before the epoch
            # ended. They are stored with done=False so GAE bootstraps from next_values.
            if pending is not None and pending.valid.any():
                flush_count = pending.valid.sum().item()
                remaining_mask = pending.valid.clone()
                remaining_dones = torch.zeros(self.num_envs, device=self.device)
                remaining_terminated = torch.zeros(self.num_envs, device=self.device)
                remaining = pending.finalize(remaining_mask, remaining_dones, remaining_terminated)
                if remaining is not None:
                    remaining_value_cats = torch.full(
                        (remaining["env_ids"].numel(),), -1,
                        dtype=torch.long, device=self.device,
                    )
                    self.buffer.add(
                        remaining["obs"], remaining["actions"],
                        remaining["log_probs"], remaining["values"],
                        remaining["rewards"], remaining["dones"],
                        remaining["terminated"],
                        remaining["legal_masks"], remaining_value_cats,
                        remaining["score_targets"],
                        env_ids=remaining["env_ids"],
                    )
                    logger.info(
                        "Epoch %d: flushed %d pending transitions at epoch end",
                        epoch_i, flush_count,
                    )

            # Bootstrap value for GAE
            self.ppo.forward_model.eval()
            with torch.no_grad():
                output = self.ppo.forward_model(obs)
                next_values = self.value_adapter.scalar_value_blended(
                    output.value_logits, output.score_lead,
                )
            self.ppo.forward_model.train()

            # Sign-correct bootstrap for split-merge mode: the value network
            # outputs from current-player perspective. When the opponent is to-move
            # at epoch end, the bootstrap value must be negated for learner-centric GAE.
            if self._current_opponent is not None:
                next_values = sign_correct_bootstrap(
                    next_values, current_players, learner_side,
                )

            logger.info(
                "Epoch %d: %d terminated, %d truncated (bootstrapped)",
                epoch_i, terminated_count, truncated_count,
            )

            # Set epoch-dependent entropy coefficient
            self.ppo.current_entropy_coeff = self.ppo.get_entropy_coeff(epoch_i)
            logger.info("Epoch %d: entropy_coeff=%.4f", epoch_i, self.ppo.current_entropy_coeff)
            if epoch_i == 0 or epoch_i == self.ppo.warmup_epochs:
                logger.info(
                    "Entropy coefficient: %.4f (warmup=%d, epoch=%d)",
                    self.ppo.current_entropy_coeff, self.ppo.warmup_epochs, epoch_i,
                )

            self._phase = "update"
            self._force_heartbeat()
            losses = self.ppo.update(
                self.buffer, next_values,
                value_adapter=self.value_adapter,
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
                monitor_value = losses.get("policy_loss")
                if monitor_value is None:
                    raise RuntimeError(
                        "LR scheduler expects 'policy_loss' in losses dict but it was absent. "
                        "Check that ppo.update() returns 'policy_loss'."
                    )
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
                    logger.info("LR reduced: %.6f -> %.6f (monitor=policy_loss, value=%.4f)",
                                old_lr, new_lr, monitor_value)

            # Materialise GPU counters → CPU once per epoch (all ranks, to release GPU memory)
            win_count = win_acc.item()
            loss_count = loss_acc.item()
            draw_count = draw_acc.item()
            black_win_count = black_wins.item()
            white_win_count = white_wins.item()

            # Elo tracking (league mode, rank 0 only)
            # The main trainer is NOT a league participant — it creates
            # snapshots that become league entries, but its live training
            # matches are not recorded as league results. Only the
            # background tournament produces league_results rows.
            # We still update the opponent's Elo so the pool has signal
            # for sampling, but no result row is written.
            if self.dist_ctx.is_main:
                total_games = win_count + loss_count + draw_count
                k = self.config.league.elo_k_factor if self.config.league else 32.0

                if (self.pool is not None
                        and self._opponent_results is not None
                        and self._cached_entries_by_id
                        and total_games > 0):
                    # Per-opponent Elo updates (Change 3).
                    # Freeze the starting learner Elo — all opponent updates are
                    # computed against this SAME base value, then the cumulative
                    # delta is applied once. This prevents path-dependent Elo drift
                    # from dict iteration order.
                    # K is normalized by active opponent count to prevent cumulative
                    # amplification (20 opponents × K=32 = 640 pts without normalization).
                    learner_entry = self.pool._get_entry(self._learner_entry_id)
                    if learner_entry is not None:
                        base_learner_elo = learner_entry.elo_rating
                        cumulative_learner_delta = 0.0
                        n_active = sum(
                            1 for w, l, d in self._opponent_results.values()
                            if w + l + d > 0
                        )
                        k_per_opp = k / max(1, n_active)

                        for opp_id, (w, l, d) in self._opponent_results.items():
                            opp_total = w + l + d
                            if opp_total == 0:
                                continue
                            if opp_id == self._learner_entry_id:
                                continue
                            opp_entry = self._cached_entries_by_id.get(opp_id)
                            if opp_entry is None:
                                continue
                            result_score = (w + 0.5 * d) / opp_total
                            new_learner_elo, new_opp_elo = compute_elo_update(
                                base_learner_elo, opp_entry.elo_rating,
                                result=result_score, k=k_per_opp,
                            )
                            learner_delta = new_learner_elo - base_learner_elo
                            cumulative_learner_delta += learner_delta
                            # Update opponent Elo immediately (each opponent is independent)
                            self.pool.update_elo(opp_id, new_opp_elo, epoch=self.epoch)
                            logger.info(
                                "Elo: learner base=%.0f delta=%.1f, "
                                "opponent(id=%d) %.0f->%.0f | W=%d L=%d D=%d",
                                base_learner_elo, learner_delta,
                                opp_id, opp_entry.elo_rating, new_opp_elo,
                                w, l, d,
                            )

                        # Apply cumulative learner Elo change once
                        final_learner_elo = base_learner_elo + cumulative_learner_delta
                        self.pool.update_elo(
                            self._learner_entry_id, final_learner_elo, epoch=self.epoch,
                        )
                        logger.info(
                            "Elo: learner %.0f->%.0f (cumulative from %d opponents)",
                            base_learner_elo, final_learner_elo,
                            sum(1 for w, l, d in self._opponent_results.values()
                                if w + l + d > 0),
                        )

                elif (self.pool is not None
                        and self._current_opponent_entry is not None
                        and total_games > 0
                        and self._learner_entry_id != self._current_opponent_entry.id):
                    # Legacy single-opponent Elo update
                    learner_entry = self.pool._get_entry(self._learner_entry_id)
                    if learner_entry is not None:
                        result_score = (win_count + 0.5 * draw_count) / total_games
                        new_learner_elo, new_opp_elo = compute_elo_update(
                            learner_entry.elo_rating,
                            self._current_opponent_entry.elo_rating,
                            result=result_score, k=k,
                        )
                        self.pool.update_elo(
                            learner_entry.id, new_learner_elo, epoch=self.epoch,
                        )
                        self.pool.update_elo(
                            self._current_opponent_entry.id, new_opp_elo,
                            epoch=self.epoch,
                        )
                        logger.info(
                            "Elo: learner %.0f->%.0f, opponent(id=%d) %.0f->%.0f "
                            "| W=%d L=%d D=%d",
                            learner_entry.elo_rating, new_learner_elo,
                            self._current_opponent_entry.id,
                            self._current_opponent_entry.elo_rating, new_opp_elo,
                            win_count, loss_count, draw_count,
                        )

            # Carry forward Elo for entries that didn't play this epoch,
            # so the Elo chart has continuous lines with no gaps.
            if self.dist_ctx.is_main and self.pool is not None:
                played_ids = set()
                if self._learner_entry_id is not None:
                    played_ids.add(self._learner_entry_id)
                if self._current_opponent_entry is not None:
                    played_ids.add(self._current_opponent_entry.id)
                # Change 3: include all opponents that had games this epoch.
                # Without this, the carry-forward loop below would overwrite
                # the per-opponent Elo updates with stale pre-epoch values.
                if self._opponent_results is not None:
                    for opp_id, (w, l, d) in self._opponent_results.items():
                        if w + l + d > 0:
                            played_ids.add(opp_id)
                for entry in self.pool.list_entries():
                    if entry.id not in played_ids:
                        self.pool.update_elo(entry.id, entry.elo_rating, epoch=epoch_i)

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
                    # draw_rate from vecenv counts engine-level draws (stalemate,
                    # repetition). draw_count from win/loss/draw accumulators counts
                    # reward==0 terminals. These may diverge if truncations produce
                    # non-zero rewards. Both are included for cross-validation.
                    "draw_rate": getattr(self.vecenv, "draw_rate", None),
                    "win_rate": (
                        win_count / total_games if total_games > 0 else None
                    ),
                    "loss_rate": (
                        loss_count / total_games if total_games > 0 else None
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

        # Stop background tournament when training ends
        if self._tournament is not None:
            self._tournament.stop()

    def _rotate_seat(self, epoch: int) -> None:
        """Save current learner weights and reset optimizer for the next seat."""
        # NOTE: The chart-continuity carry-forward in the epoch loop (lines
        # 1112-1122) is a SEPARATE mechanism that writes unchanged Elo to
        # elo_history for entries that didn't play, keeping chart lines
        # continuous. That must be preserved. This method only handles
        # rotation — new snapshots enter at the DB default of 1000.0.

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
                "opponent_id": (
                    self._current_opponent_entry.id
                    if self._current_opponent_entry is not None
                    else None
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
