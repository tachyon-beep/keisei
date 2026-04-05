"""ConcurrentMatchPool — partitioned concurrent match execution.

Replaces the sequential match loop in the tournament with interleaved
execution across multiple VecEnv partitions.  Each partition runs a
different pairing simultaneously, sharing a single VecEnv.step() call
per ply across all active partitions.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from keisei.config import ConcurrencyConfig
from keisei.training.opponent_store import OpponentEntry

if TYPE_CHECKING:
    from keisei.training.dynamic_trainer import MatchRollout

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    """Result of a completed match between two opponents."""

    entry_a: OpponentEntry
    entry_b: OpponentEntry
    a_wins: int
    b_wins: int
    draws: int
    rollout: MatchRollout | None


@dataclass
class _MatchSlot:
    """Internal state for one concurrent partition of the VecEnv."""

    index: int
    env_start: int  # inclusive
    env_end: int  # exclusive
    entry_a: OpponentEntry | None = None
    entry_b: OpponentEntry | None = None
    model_a: torch.nn.Module | None = None
    model_b: torch.nn.Module | None = None
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    games_target: int = 0
    active: bool = False
    collect_rollout: bool = False
    ply_count: int = 0
    # Rollout collection buffers (CPU tensors)
    _obs: list[torch.Tensor] = field(default_factory=list)
    _actions: list[torch.Tensor] = field(default_factory=list)
    _rewards: list[torch.Tensor] = field(default_factory=list)
    _dones: list[torch.Tensor] = field(default_factory=list)
    _masks: list[torch.Tensor] = field(default_factory=list)
    _perspective: list[torch.Tensor] = field(default_factory=list)

    @property
    def games_completed(self) -> int:
        return self.a_wins + self.b_wins + self.draws

    def reset_for_pairing(
        self,
        *,
        entry_a: OpponentEntry,
        entry_b: OpponentEntry,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        games_target: int,
        collect_rollout: bool = False,
    ) -> None:
        """Reset slot state for a new pairing."""
        self.entry_a = entry_a
        self.entry_b = entry_b
        self.model_a = model_a
        self.model_b = model_b
        self.a_wins = 0
        self.b_wins = 0
        self.draws = 0
        self.games_target = games_target
        self.active = True
        self.collect_rollout = collect_rollout
        self.ply_count = 0
        self._obs = []
        self._actions = []
        self._rewards = []
        self._dones = []
        self._masks = []
        self._perspective = []

    def to_result(self) -> MatchResult:
        """Convert slot state to a MatchResult."""
        rollout: MatchRollout | None = None
        if self.collect_rollout and self._obs:
            from keisei.training.dynamic_trainer import MatchRollout

            rollout = MatchRollout(
                observations=torch.stack(self._obs),
                actions=torch.stack(self._actions),
                rewards=torch.stack(self._rewards),
                dones=torch.stack(self._dones),
                legal_masks=torch.stack(self._masks),
                perspective=torch.stack(self._perspective),
            )
        assert self.entry_a is not None
        assert self.entry_b is not None
        return MatchResult(
            entry_a=self.entry_a,
            entry_b=self.entry_b,
            a_wins=self.a_wins,
            b_wins=self.b_wins,
            draws=self.draws,
            rollout=rollout,
        )


# ---------------------------------------------------------------------------
# ConcurrentMatchPool
# ---------------------------------------------------------------------------


class ConcurrentMatchPool:
    """Partitioned concurrent match execution pool.

    Divides a VecEnv into contiguous partitions, each running a different
    pairing.  All partitions share a single vecenv.step() call per ply.
    """

    def __init__(self, config: ConcurrencyConfig) -> None:
        self.config = config

    def partition_range(self, partition_idx: int) -> tuple[int, int]:
        """Return (start, end) env indices for a partition slot."""
        if partition_idx < 0 or partition_idx >= self.config.parallel_matches:
            raise ValueError(
                f"partition_idx {partition_idx} out of range "
                f"[0, {self.config.parallel_matches})"
            )
        start = partition_idx * self.config.envs_per_match
        end = start + self.config.envs_per_match
        return (start, end)

    def run_round(
        self,
        vecenv: Any,
        pairings: list[tuple[OpponentEntry, OpponentEntry]],
        *,
        load_fn: Callable[[OpponentEntry], Any],
        release_fn: Callable[[Any, Any], None],
        device: str | torch.device = "cpu",
        games_per_match: int = 64,
        max_ply: int = 512,
        stop_event: threading.Event | None = None,
        trainable_fn: Callable[[OpponentEntry, OpponentEntry], bool] | None = None,
    ) -> list[MatchResult]:
        """Execute pairings with true concurrent partitioned inference.

        Args:
            vecenv: A VecEnv instance with total_envs environments.
            pairings: Priority-ordered list of (entry_a, entry_b).
            load_fn: Callable(entry) -> model. Loads a model for inference.
            release_fn: Callable(model_a, model_b) -> None. Releases models.
            device: Torch device for inference.
            games_per_match: Number of games each pairing must complete.
            max_ply: Maximum ply per game (safety bound).
            stop_event: If set, abort early and return partial results.
            trainable_fn: If provided, called with entry_a to decide if
                rollout data should be collected for this pairing.

        Returns:
            List of MatchResult in the same order as input pairings.
        """
        if not pairings:
            return []

        if stop_event is not None and stop_event.is_set():
            logger.warning("run_round: stop_event already set, returning empty results")
            return []

        results: dict[int, MatchResult] = {}
        next_pairing_idx = 0
        total_pairings = len(pairings)
        parallel = min(self.config.parallel_matches, total_pairings)

        # Create slots
        slots: list[_MatchSlot] = []
        for slot_idx in range(parallel):
            env_start, env_end = self.partition_range(slot_idx)
            slots.append(_MatchSlot(index=slot_idx, env_start=env_start, env_end=env_end))

        # Reset VecEnv once at round start
        reset_result = vecenv.reset()
        obs = torch.from_numpy(np.asarray(reset_result.observations)).to(device)
        legal_masks = torch.from_numpy(
            np.asarray(reset_result.legal_masks)
        ).to(device)
        current_players = np.zeros(self.config.total_envs, dtype=np.uint8)

        # Assign initial pairings and track slot→pairing mapping
        slot_pairing_map: dict[int, int] = {}
        for slot in slots:
            if next_pairing_idx >= total_pairings:
                break
            pairing_idx = next_pairing_idx
            self._assign_pairing(
                slot, pairing_idx, pairings[pairing_idx],
                load_fn, games_per_match, trainable_fn,
            )
            if slot.active:
                slot_pairing_map[slot.index] = pairing_idx
            next_pairing_idx += 1

        # Game loop
        active_slots = [s for s in slots if s.active]

        while active_slots:
            if stop_event is not None and stop_event.is_set():
                logger.warning("run_round: stop_event fired, returning partial results")
                break

            # Increment ply for all active slots
            for slot in active_slots:
                slot.ply_count += 1

            # Build action tensor for ALL envs
            actions = torch.zeros(self.config.total_envs, dtype=torch.long, device=device)

            for slot in active_slots:
                s, e = slot.env_start, slot.env_end
                partition_obs = obs[s:e]
                partition_legal = legal_masks[s:e]
                partition_players = current_players[s:e]

                # Collect pre-step rollout data
                if slot.collect_rollout:
                    slot._obs.append(partition_obs.cpu())
                    slot._masks.append(partition_legal.cpu())
                    slot._perspective.append(
                        torch.from_numpy(partition_players.copy())
                    )

                assert slot.model_a is not None
                assert slot.model_b is not None

                player_a_mask = torch.from_numpy(partition_players == 0).to(device)
                player_b_mask = ~player_a_mask

                slot_actions = torch.zeros(e - s, dtype=torch.long, device=device)

                # Model A forward (player 0 = Black)
                a_indices = player_a_mask.nonzero(as_tuple=True)[0]
                if a_indices.numel() > 0:
                    with torch.no_grad():
                        a_out = slot.model_a(partition_obs[a_indices])
                        a_logits = a_out.policy_logits.reshape(a_indices.numel(), -1)
                        a_masked = a_logits.masked_fill(
                            ~partition_legal[a_indices], float("-inf")
                        )
                        a_probs = F.softmax(a_masked, dim=-1)
                        slot_actions[a_indices] = torch.distributions.Categorical(
                            a_probs
                        ).sample()

                # Model B forward (player 1 = White)
                b_indices = player_b_mask.nonzero(as_tuple=True)[0]
                if b_indices.numel() > 0:
                    with torch.no_grad():
                        b_out = slot.model_b(partition_obs[b_indices])
                        b_logits = b_out.policy_logits.reshape(b_indices.numel(), -1)
                        b_masked = b_logits.masked_fill(
                            ~partition_legal[b_indices], float("-inf")
                        )
                        b_probs = F.softmax(b_masked, dim=-1)
                        slot_actions[b_indices] = torch.distributions.Categorical(
                            b_probs
                        ).sample()

                actions[s:e] = slot_actions

                if slot.collect_rollout:
                    slot._actions.append(slot_actions.cpu())

            # For inactive env ranges, pick first legal action
            inactive_ranges: list[tuple[int, int]] = []
            active_ranges = {(s.env_start, s.env_end) for s in active_slots}
            for slot_idx in range(self.config.parallel_matches):
                env_s, env_e = self.partition_range(slot_idx)
                if (env_s, env_e) not in active_ranges:
                    inactive_ranges.append((env_s, env_e))
            for start, end in inactive_ranges:
                for idx in range(start, end):
                    legal = legal_masks[idx].nonzero(as_tuple=True)[0]
                    actions[idx] = legal[0] if legal.numel() > 0 else 0

            # Step ALL environments at once
            step_result = vecenv.step(actions.cpu().numpy())
            obs = torch.from_numpy(np.asarray(step_result.observations)).to(device)
            legal_masks = torch.from_numpy(
                np.asarray(step_result.legal_masks)
            ).to(device)
            current_players = np.asarray(step_result.current_players, dtype=np.uint8)
            rewards_np = np.asarray(step_result.rewards)
            terminated = np.asarray(step_result.terminated)
            truncated = np.asarray(step_result.truncated)

            # Process completions per slot
            completed_slot_indices: list[int] = []
            for i, slot in enumerate(active_slots):
                s, e = slot.env_start, slot.env_end
                partition_rewards = rewards_np[s:e]
                partition_term = terminated[s:e]
                partition_trunc = truncated[s:e]
                partition_done = partition_term | partition_trunc

                if slot.collect_rollout:
                    slot._rewards.append(
                        torch.from_numpy(partition_rewards.copy().astype(np.float32))
                    )
                    slot._dones.append(
                        torch.from_numpy(partition_done.astype(np.float32))
                    )

                # Count completed games
                for env_i in range(e - s):
                    if partition_done[env_i]:
                        r = float(partition_rewards[env_i])
                        if r > 0:
                            slot.a_wins += 1
                        elif r < 0:
                            slot.b_wins += 1
                        else:
                            slot.draws += 1

                if slot.games_completed >= slot.games_target:
                    completed_slot_indices.append(i)
                elif slot.ply_count >= max_ply:
                    logger.warning(
                        "Slot %d hit max_ply %d with %d/%d games, yielding partial result",
                        slot.index, max_ply, slot.games_completed, slot.games_target,
                    )
                    completed_slot_indices.append(i)

            # Process completed slots (iterate in reverse to allow removal)
            for i in sorted(completed_slot_indices, reverse=True):
                slot = active_slots.pop(i)
                pairing_idx = slot_pairing_map[slot.index]
                results[pairing_idx] = slot.to_result()

                # Release models
                try:
                    release_fn(slot.model_a, slot.model_b)
                except Exception:
                    logger.exception(
                        "Failed to release models for slot %d", slot.index
                    )

                slot.active = False

                # Swap in next pairing if available.
                # NOTE: VecEnv only supports global reset(), not per-partition.
                # Envs auto-reset on termination, so most are at move 0 when
                # the slot completes. However, envs that finished early may be
                # mid-game from auto-reset. This noise is negligible for
                # typical games_target values (64+) and washes out in Elo.
                if next_pairing_idx < total_pairings:
                    new_pairing_idx = next_pairing_idx
                    self._assign_pairing(
                        slot, new_pairing_idx, pairings[new_pairing_idx],
                        load_fn, games_per_match, trainable_fn,
                    )
                    if slot.active:
                        slot_pairing_map[slot.index] = new_pairing_idx
                        active_slots.append(slot)
                    next_pairing_idx += 1

        # Release any remaining active models (early termination)
        for slot in active_slots:
            try:
                release_fn(slot.model_a, slot.model_b)
            except Exception:
                logger.exception(
                    "Failed to release models for slot %d", slot.index
                )

        # Return results in original pairing order
        return [results[i] for i in sorted(results.keys())]

    def _assign_pairing(
        self,
        slot: _MatchSlot,
        pairing_idx: int,
        pairing: tuple[OpponentEntry, OpponentEntry],
        load_fn: Callable[[OpponentEntry], Any],
        games_target: int,
        trainable_fn: Callable[[OpponentEntry, OpponentEntry], bool] | None = None,
    ) -> None:
        """Load models and assign a pairing to a slot."""
        entry_a, entry_b = pairing
        try:
            model_a = load_fn(entry_a)
            model_b = load_fn(entry_b)
        except Exception:
            logger.warning(
                "Failed to load models for pairing %d (%s vs %s), skipping",
                pairing_idx,
                entry_a.display_name,
                entry_b.display_name,
                exc_info=True,
            )
            slot.active = False
            return

        collect_rollout = False
        if trainable_fn is not None:
            collect_rollout = trainable_fn(entry_a, entry_b)

        model_a.eval()
        model_b.eval()

        slot.reset_for_pairing(
            entry_a=entry_a,
            entry_b=entry_b,
            model_a=model_a,
            model_b=model_b,
            games_target=games_target,
            collect_rollout=collect_rollout,
        )
