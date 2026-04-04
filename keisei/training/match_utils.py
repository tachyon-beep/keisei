"""Shared match-execution utilities for tournament and gauntlet."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


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
) -> tuple[int, int, int]:
    """Play a set of games between two frozen models.

    Returns (a_wins, b_wins, draws). Includes a batch-count ceiling
    to prevent infinite loops if VecEnv never terminates games.
    """
    total_a_wins = 0
    total_b_wins = 0
    total_draws = 0

    games_remaining = games_target
    max_batches = games_target * 3  # ceiling to prevent infinite loop
    batches_run = 0

    while games_remaining > 0 and batches_run < max_batches:
        if stop_event is not None and stop_event.is_set():
            break
        a_wins, b_wins, draws = play_batch(
            vecenv, model_a, model_b,
            device=device, num_envs=num_envs, max_ply=max_ply,
            stop_event=stop_event,
        )
        total_a_wins += a_wins
        total_b_wins += b_wins
        total_draws += draws
        games_remaining -= (a_wins + b_wins + draws)
        batches_run += 1

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
) -> tuple[int, int, int]:
    """Play one batch of games (num_envs concurrent).

    Returns (a_wins, b_wins, draws). Player A = side 0 (black),
    Player B = side 1 (white).
    """
    reset_result = vecenv.reset()
    obs = torch.from_numpy(np.asarray(reset_result.observations)).to(device)
    legal_masks = torch.from_numpy(np.asarray(reset_result.legal_masks)).to(device)
    current_players = np.zeros(num_envs, dtype=np.uint8)

    a_wins = 0
    b_wins = 0
    draws = 0

    for _ply in range(max_ply):
        if stop_event is not None and stop_event.is_set():
            break

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

        step_result = vecenv.step(actions.cpu().numpy())
        obs = torch.from_numpy(np.asarray(step_result.observations)).to(device)
        legal_masks = torch.from_numpy(np.asarray(step_result.legal_masks)).to(device)
        current_players = np.asarray(step_result.current_players, dtype=np.uint8)
        rewards = np.asarray(step_result.rewards)
        terminated = np.asarray(step_result.terminated)
        truncated = np.asarray(step_result.truncated)

        # Vectorized result counting
        done = terminated | truncated
        a_wins += int(((rewards > 0) & done).sum())
        b_wins += int(((rewards < 0) & done).sum())
        draws += int(((rewards == 0) & done).sum())

        if a_wins + b_wins + draws >= num_envs:
            break

    return a_wins, b_wins, draws


def release_models(*models: torch.nn.Module, device_type: str = "cpu") -> None:
    """Delete models and clear CUDA cache if on GPU."""
    for m in models:
        del m
    if device_type == "cuda":
        torch.cuda.empty_cache()
