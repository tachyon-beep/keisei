"""DynamicTrainer — small PPO updates for Dynamic entries from league match data."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from keisei.training.katago_ppo import ppo_clip_loss, wdl_cross_entropy_loss

if TYPE_CHECKING:
    from keisei.config import DynamicConfig
    from keisei.training.opponent_store import OpponentEntry, OpponentStore

logger = logging.getLogger(__name__)


@dataclass
class MatchRollout:
    """Replay data from a league match for Dynamic entry training.

    Step dimension is VARIABLE — depends on game length, not a fixed value.
    All tensors stored on CPU to avoid GPU memory pressure.
    """

    observations: torch.Tensor  # (steps, num_envs, obs_channels, 9, 9)
    actions: torch.Tensor  # (steps, num_envs)
    rewards: torch.Tensor  # (steps, num_envs)
    dones: torch.Tensor  # (steps, num_envs)
    legal_masks: torch.Tensor  # (steps, num_envs, action_space)
    perspective: torch.Tensor  # (steps, num_envs) — 0=player_A, 1=player_B


class DynamicTrainer:
    """Small PPO updates for Dynamic entries from league match data.

    Threading: all methods are called from the tournament thread only.
    record_match(), should_update(), is_rate_limited(), and update() are
    called sequentially within _run_concurrent_round/_run_one_match.
    No lock is needed — there is no concurrent access from other threads.
    """

    def __init__(
        self,
        store: OpponentStore,
        config: DynamicConfig,
        learner_lr: float,
    ) -> None:
        self.store = store
        self.config = config
        self.learner_lr = learner_lr

        self._match_counts: dict[int, int] = {}
        self._total_matches: dict[int, int] = {}
        self._update_timestamps: list[float] = []
        self._optimizers: dict[int, torch.optim.Adam] = {}
        self._disabled_entries: set[int] = set()
        self._rollout_buffers: dict[int, deque[tuple[MatchRollout, int]]] = {}
        self._error_counts: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Record & query
    # ------------------------------------------------------------------

    def record_match(self, entry_id: int, rollout: MatchRollout, side: int) -> None:
        """Record a match rollout for a Dynamic entry."""
        if entry_id in self._disabled_entries:
            return
        if entry_id not in self._rollout_buffers:
            self._rollout_buffers[entry_id] = deque(maxlen=self.config.max_buffer_depth)
        self._rollout_buffers[entry_id].append((rollout, side))
        self._match_counts[entry_id] = self._match_counts.get(entry_id, 0) + 1

    def should_update(self, entry_id: int) -> bool:
        """Check if enough matches have accumulated for an update."""
        if entry_id in self._disabled_entries:
            return False
        return self._match_counts.get(entry_id, 0) >= self.config.update_every_matches

    def is_rate_limited(self) -> bool:
        """Check if updates are rate-limited (too many in the last 60 seconds)."""
        now = time.monotonic()
        cutoff = now - 60.0
        self._update_timestamps = [t for t in self._update_timestamps if t >= cutoff]
        return len(self._update_timestamps) >= self.config.max_updates_per_minute

    def get_update_stats(self, entry_id: int) -> tuple[int, str | None]:
        """Return (update_count, last_train_at) from the store entry."""
        entry = self.store.get_entry(entry_id)
        if entry is None:
            return (0, None)
        return (entry.update_count, entry.last_train_at)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def _prepare_batch(
        self, entry_id: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concatenate and filter rollouts by perspective, return flat tensors."""
        buffers = self._rollout_buffers.get(entry_id, [])
        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_legal_masks = []

        for rollout, side in buffers:
            # perspective is (steps, num_envs); boolean mask selects this side's
            # time-steps, flattening (steps, num_envs, ...) → (N, ...).
            assert rollout.perspective.shape == rollout.actions.shape, (
                f"perspective shape {rollout.perspective.shape} must match "
                f"actions shape {rollout.actions.shape}"
            )
            mask = rollout.perspective == side
            all_obs.append(rollout.observations[mask])
            all_actions.append(rollout.actions[mask])
            all_rewards.append(rollout.rewards[mask])
            all_dones.append(rollout.dones[mask])
            all_legal_masks.append(rollout.legal_masks[mask])

        if not all_obs:
            # Shape (0,) is sufficient: the only caller checks shape[0] == 0
            # and returns early.  No downstream code inspects higher dimensions.
            empty = torch.zeros(0)
            return empty, empty, empty, empty, empty

        return (
            torch.cat(all_obs).to(device),
            torch.cat(all_actions).to(device),
            torch.cat(all_rewards).to(device),
            torch.cat(all_dones).to(device),
            torch.cat(all_legal_masks).to(device),
        )

    def _get_or_create_optimizer(
        self, entry_id: int, model: torch.nn.Module
    ) -> torch.optim.Adam:
        """Get cached optimizer or create a new one (loading from store if available)."""
        if entry_id in self._optimizers:
            # Re-attach to new model parameters
            opt = self._optimizers[entry_id]
            # We need a fresh optimizer with the right params but the saved state
            new_opt = torch.optim.Adam(
                model.parameters(), lr=self.learner_lr * self.config.lr_scale
            )
            # Try to load state from the cached optimizer
            try:
                new_opt.load_state_dict(opt.state_dict())
                # Move optimizer state to training device
                device = next(model.parameters()).device
                for state in new_opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            except (ValueError, RuntimeError):
                logger.warning("Optimizer state mismatch for entry %d, resetting momentum", entry_id)
            return new_opt

        # Try loading from store
        saved_state = self.store.load_optimizer(entry_id)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learner_lr * self.config.lr_scale
        )
        if saved_state is not None:
            try:
                optimizer.load_state_dict(saved_state)
                # Move optimizer state to training device
                device = next(model.parameters()).device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            except (ValueError, RuntimeError):
                logger.warning(
                    "Failed to load optimizer state for entry %d, starting fresh",
                    entry_id,
                )
        return optimizer

    def update(self, entry: OpponentEntry, device: str) -> bool:
        """Run a small PPO update on the Dynamic entry using accumulated match data.

        Returns True on success, False if an error was caught and handled.
        Raises on error when config.disable_on_error is False.
        """
        try:
            return self._update_inner(entry, device)
        except Exception:
            if not self.config.disable_on_error:
                raise
            # Clear stale rollout data to prevent training on corrupted buffers
            self._match_counts[entry.id] = 0
            self._rollout_buffers[entry.id] = deque(maxlen=self.config.max_buffer_depth)
            self._error_counts[entry.id] = self._error_counts.get(entry.id, 0) + 1
            logger.warning(
                "DynamicTrainer update failed for entry %d (error %d/%d)",
                entry.id,
                self._error_counts[entry.id],
                self.config.max_consecutive_errors,
                exc_info=True,
            )
            if self._error_counts[entry.id] >= self.config.max_consecutive_errors:
                self._disabled_entries.add(entry.id)
                logger.error(
                    "DynamicTrainer disabled entry %d after %d consecutive errors",
                    entry.id,
                    self.config.max_consecutive_errors,
                )
            return False

    def _update_inner(self, entry: OpponentEntry, device: str) -> bool:
        """Internal update logic — may raise."""
        # load_opponent() builds a fresh model instance each call (no cache),
        # so .train() is safe — it won't affect other callers' eval-mode models.
        model = self.store.load_opponent(entry, device)
        model.train()

        # Concatenate and filter rollouts by perspective
        all_obs, all_actions, all_rewards, all_dones, all_legal_masks = (
            self._prepare_batch(entry.id, device)
        )

        if all_obs.shape[0] == 0:
            return False  # no relevant data

        # Create WDL targets from terminal rewards.
        # all_dones includes both termination and truncation (from
        # ConcurrentMatchPool).  Truncated games at max_ply have reward 0,
        # so they are labeled as draws — correct because no winner was
        # determined.  The advantage calculation also handles this: reward 0 ×
        # done 1.0 = zero advantage, so no policy gradient signal for
        # truncated games.
        value_cats = torch.full(
            (all_obs.shape[0],), -1, dtype=torch.long, device=device
        )
        # Exact float comparison is safe: Rust VecEnv compute_reward() returns
        # literal 1.0 / -1.0 / 0.0 with no arithmetic — no epsilon needed.
        terminal_mask = all_dones.bool()
        value_cats[terminal_mask & (all_rewards > 0)] = 0  # win
        value_cats[terminal_mask & (all_rewards == 0)] = 1  # draw
        value_cats[terminal_mask & (all_rewards < 0)] = 2  # loss

        # Initial forward pass for old_log_probs (baseline).
        # Processes all data in a single batch.  OOM risk is bounded by
        # config.max_buffer_depth (default 8) and update_every_matches (default 4).
        with torch.no_grad():
            output = model(all_obs)
            flat_logits = output.policy_logits.reshape(all_obs.shape[0], -1)
            masked = flat_logits.masked_fill(~all_legal_masks, float("-inf"))
            old_log_probs = (
                F.log_softmax(masked, dim=-1)
                .gather(1, all_actions.unsqueeze(1))
                .squeeze(1)
            )

        # Get or create optimizer.  On failure, optimizer is NOT stored back
        # into self._optimizers (that happens at line 305 on success only).
        # This is intentional: failed updates discard momentum from potentially
        # corrupted gradients, and the old cached state is preserved for retry.
        optimizer = self._get_or_create_optimizer(entry.id, model)

        for _ in range(self.config.update_epochs_per_batch):
            indices = torch.randperm(all_obs.shape[0], device=device)
            output = model(all_obs[indices])
            flat_logits = output.policy_logits.reshape(len(indices), -1)
            masked = flat_logits.masked_fill(
                ~all_legal_masks[indices], float("-inf")
            )
            new_log_probs = (
                F.log_softmax(masked, dim=-1)
                .gather(1, all_actions[indices].unsqueeze(1))
                .squeeze(1)
            )

            # Reward-signed advantage: +1 for wins, -1 for losses, 0 for draws/non-terminal.
            # Zero advantage for draws is intentional — draws don't indicate which move
            # was good or bad. The value head still learns from draws via WDL cross-entropy above.
            advantages = all_rewards[indices] * all_dones[indices].float()

            policy_loss = ppo_clip_loss(
                new_log_probs, old_log_probs[indices], advantages, clip_epsilon=0.2
            )
            value_loss = wdl_cross_entropy_loss(
                output.value_logits, value_cats[indices]
            )

            # Simplified objective: equal weights, no entropy bonus, no score
            # head.  Intentionally different from the main PPO learner — Dynamic
            # entries are short-lived opponents, not the primary agent.  The
            # missing entropy bonus means faster policy sharpening, which is
            # acceptable for opponent diversity but could be revisited for
            # long-lived Dynamic entries.
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            optimizer.step()

        # Save model weights atomically.
        # with_suffix(suffix + ".tmp") works for any extension: .pt → .pt.tmp,
        # .ckpt → .ckpt.tmp, etc.
        ckpt_path = Path(entry.checkpoint_path)
        tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        torch.save(model.state_dict(), tmp_path)
        tmp_path.rename(ckpt_path)

        # Move optimizer state to CPU for storage
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()

        # Track matches since last checkpoint flush.  If any code below raises,
        # the error handler in update() resets _match_counts[entry.id] = 0,
        # preventing double-counting on the next update attempt.
        match_count = self._match_counts.get(entry.id, 0)
        self._total_matches[entry.id] = (
            self._total_matches.get(entry.id, 0) + match_count
        )

        # Checkpoint optimizer periodically
        if self._total_matches[entry.id] >= self.config.checkpoint_flush_every:
            self.store.save_optimizer(entry.id, optimizer.state_dict())
            self._total_matches[entry.id] = 0

        self.store.increment_update_count(entry.id)
        self._match_counts[entry.id] = 0
        self._rollout_buffers[entry.id] = deque(maxlen=self.config.max_buffer_depth)
        self._update_timestamps.append(time.monotonic())
        self._error_counts[entry.id] = 0
        self._optimizers[entry.id] = optimizer

        return True
