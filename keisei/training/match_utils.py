"""Shared match-execution utilities for tournament and gauntlet."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from keisei.training.dynamic_trainer import MatchRollout


def _combine_rollouts(rollouts: list[MatchRollout]) -> MatchRollout:
    """Concatenate multiple batch rollouts along step dimension."""
    from keisei.training.dynamic_trainer import MatchRollout

    return MatchRollout(
        observations=torch.cat([r.observations for r in rollouts], dim=0),
        actions=torch.cat([r.actions for r in rollouts], dim=0),
        rewards=torch.cat([r.rewards for r in rollouts], dim=0),
        dones=torch.cat([r.dones for r in rollouts], dim=0),
        legal_masks=torch.cat([r.legal_masks for r in rollouts], dim=0),
        perspective=torch.cat([r.perspective for r in rollouts], dim=0),
    )


def play_match(
    vecenv: Any,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    *,
    device: torch.device,
    num_envs: int,
    max_ply: int,
    games_target: int,
    stop_event: threading.Event | None = None,
    collect_rollout: bool = False,
) -> tuple[int, int, int] | tuple[int, int, int, Any]:
    """Play a set of games between two frozen models.

    Returns (a_wins, b_wins, draws). Includes a batch-count ceiling
    to prevent infinite loops if VecEnv never terminates games.

    When collect_rollout=True, returns (a_wins, b_wins, draws, MatchRollout).
    """
    total_a_wins = 0
    total_b_wins = 0
    total_draws = 0

    all_rollouts: list[Any] | None = [] if collect_rollout else None

    games_remaining = games_target
    max_batches = games_target * 3  # ceiling to prevent infinite loop
    batches_run = 0

    while games_remaining > 0 and batches_run < max_batches:
        if stop_event is not None and stop_event.is_set():
            break
        result = play_batch(
            vecenv, model_a, model_b,
            device=device, num_envs=num_envs, max_ply=max_ply,
            stop_event=stop_event, collect_rollout=collect_rollout,
        )
        if collect_rollout:
            a_wins, b_wins, draws, rollout = result
            assert all_rollouts is not None
            all_rollouts.append(rollout)
        else:
            a_wins, b_wins, draws = result
        total_a_wins += a_wins
        total_b_wins += b_wins
        total_draws += draws
        games_remaining -= (a_wins + b_wins + draws)
        batches_run += 1

    if games_remaining > 0 and batches_run >= max_batches:
        logger.warning(
            "play_match hit batch ceiling (%d batches) with %d/%d games remaining",
            max_batches, games_remaining, games_target,
        )

    if collect_rollout:
        assert all_rollouts is not None
        # Filter out empty rollouts from interrupted batches (shape (0,)
        # tensors would crash _combine_rollouts due to rank mismatch).
        valid_rollouts = [r for r in all_rollouts if r.observations.dim() > 1]
        if not valid_rollouts:
            return total_a_wins, total_b_wins, total_draws, None
        combined = _combine_rollouts(valid_rollouts)
        return total_a_wins, total_b_wins, total_draws, combined
    return total_a_wins, total_b_wins, total_draws


def play_batch(
    vecenv: Any,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    *,
    device: torch.device,
    num_envs: int,
    max_ply: int,
    stop_event: threading.Event | None = None,
    collect_rollout: bool = False,
) -> tuple[int, int, int] | tuple[int, int, int, Any]:
    """Play one batch of games (num_envs concurrent).

    Returns (a_wins, b_wins, draws). Player A = side 0 (black),
    Player B = side 1 (white).

    When collect_rollout=True, returns (a_wins, b_wins, draws, MatchRollout).
    """
    # Defensive: ensure both models are in eval mode so BatchNorm running
    # statistics are not corrupted during inference.  ConcurrentMatchPool
    # already does this (concurrent_matches.py _assign_pairing), but
    # play_batch is a public API and callers may pass training-mode models.
    model_a.eval()
    model_b.eval()

    reset_result = vecenv.reset()
    obs = torch.from_numpy(np.asarray(reset_result.observations)).to(device)
    legal_masks = torch.from_numpy(np.asarray(reset_result.legal_masks)).to(device)
    current_players = np.zeros(num_envs, dtype=np.uint8)

    a_wins = 0
    b_wins = 0
    draws = 0

    # Rollout collection buffers (CPU)
    if collect_rollout:
        step_obs: list[torch.Tensor] = []
        step_actions: list[torch.Tensor] = []
        step_rewards: list[torch.Tensor] = []
        step_dones: list[torch.Tensor] = []
        step_legal_masks: list[torch.Tensor] = []
        step_perspective: list[torch.Tensor] = []

    for _ply in range(max_ply):
        if stop_event is not None and stop_event.is_set():
            break

        # Save pre-step perspective (who is about to move).
        # VecEnv rewards are from the last-mover's perspective, so we
        # need this to attribute wins to the correct player.
        pre_step_players = current_players.copy()

        if collect_rollout:
            step_perspective.append(torch.from_numpy(pre_step_players))
            step_obs.append(obs.cpu())
            step_legal_masks.append(legal_masks.cpu())

        player_a_mask = torch.from_numpy(current_players == 0).to(device)
        player_b_mask = ~player_a_mask

        actions = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Model A forward (full no_grad scope covers softmax + sample)
        a_indices = player_a_mask.nonzero(as_tuple=True)[0]
        if a_indices.numel() > 0:
            with torch.no_grad():
                a_out = model_a(obs[a_indices])
                a_logits = a_out.policy_logits.reshape(a_indices.numel(), -1)
                a_masked = a_logits.masked_fill(~legal_masks[a_indices], float("-inf"))
                a_probs = F.softmax(a_masked, dim=-1)
                actions[a_indices] = torch.distributions.Categorical(a_probs).sample()

        # Model B forward
        b_indices = player_b_mask.nonzero(as_tuple=True)[0]
        if b_indices.numel() > 0:
            with torch.no_grad():
                b_out = model_b(obs[b_indices])
                b_logits = b_out.policy_logits.reshape(b_indices.numel(), -1)
                b_masked = b_logits.masked_fill(~legal_masks[b_indices], float("-inf"))
                b_probs = F.softmax(b_masked, dim=-1)
                actions[b_indices] = torch.distributions.Categorical(b_probs).sample()

        if collect_rollout:
            step_actions.append(actions.cpu())

        step_result = vecenv.step(actions.cpu().numpy())
        obs = torch.from_numpy(np.asarray(step_result.observations)).to(device)
        legal_masks = torch.from_numpy(np.asarray(step_result.legal_masks)).to(device)
        current_players = np.asarray(step_result.current_players, dtype=np.uint8)
        rewards = np.asarray(step_result.rewards)
        terminated = np.asarray(step_result.terminated)
        truncated = np.asarray(step_result.truncated)

        if collect_rollout:
            step_rewards.append(
                torch.from_numpy(np.asarray(rewards, dtype=np.float32))
            )
            step_dones.append(
                torch.from_numpy((terminated | truncated).astype(np.float32))
            )

        # Vectorized result counting.
        # Rewards are from the last-mover's perspective:
        #   reward > 0 means the mover won, reward < 0 means the mover lost.
        # pre_step_players tells us who moved (0=A/black, 1=B/white).
        done = terminated | truncated
        a_moved = pre_step_players == 0
        b_moved = pre_step_players == 1
        a_wins += int(((rewards > 0) & done & a_moved).sum())
        a_wins += int(((rewards < 0) & done & b_moved).sum())
        b_wins += int(((rewards > 0) & done & b_moved).sum())
        b_wins += int(((rewards < 0) & done & a_moved).sum())
        draws += int(((rewards == 0) & done).sum())

        # Heuristic: stop once enough games finish. All games completing in
        # this step are counted ABOVE before this check — no games are dropped.
        # May exceed num_envs on VecEnv auto-reset (a fast game can finish and
        # restart within one batch), but that's fine — the outer play_match()
        # manages games_remaining exactly.
        if a_wins + b_wins + draws >= num_envs:
            break

    if collect_rollout:
        from keisei.training.dynamic_trainer import MatchRollout

        rollout = MatchRollout(
            observations=torch.stack(step_obs) if step_obs else torch.empty(0),
            actions=torch.stack(step_actions) if step_actions else torch.empty(0),
            rewards=torch.stack(step_rewards) if step_rewards else torch.empty(0),
            dones=torch.stack(step_dones) if step_dones else torch.empty(0),
            legal_masks=torch.stack(step_legal_masks) if step_legal_masks else torch.empty(0),
            perspective=torch.stack(step_perspective) if step_perspective else torch.empty(0),
        )
        return a_wins, b_wins, draws, rollout

    return a_wins, b_wins, draws


def release_models(*models: torch.nn.Module, device_type: str = "cpu") -> None:
    """Reclaim cached CUDA allocator blocks if on GPU.

    Note: callers still hold references to the model objects — this function
    cannot free them.  The value is ``empty_cache()``, which returns unused
    cached blocks to the CUDA allocator (useful even with live tensors).
    """
    if device_type == "cuda":
        torch.cuda.empty_cache()
