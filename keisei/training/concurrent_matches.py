"""ConcurrentMatchPool — partitioned concurrent match execution.

Replaces the sequential match loop in the tournament with interleaved
execution across multiple VecEnv partitions.  Each partition runs a
different pairing simultaneously, sharing a single VecEnv.step() call
per ply across all active partitions.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from keisei.config import ConcurrencyConfig
from keisei.training.game_feature_tracker import GameFeatureTracker
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
    feature_tracker: GameFeatureTracker | None = None


@dataclass
class RoundStats:
    """Monitoring metrics for a single run_round execution."""

    round_duration_s: float = 0.0
    pairings_requested: int = 0
    pairings_completed: int = 0
    total_games: int = 0
    total_plies: int = 0
    active_slots: int = 0
    model_load_time_s: float = 0.0
    model_load_count: int = 0
    cache_hits: int = 0


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
    feature_tracker: GameFeatureTracker | None = None

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
        epoch: int = 0,
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
        num_envs = self.env_end - self.env_start
        self.feature_tracker = GameFeatureTracker(
            num_envs=num_envs,
            entry_a_id=entry_a.id,
            entry_b_id=entry_b.id,
            epoch=epoch,
        )

    def to_result(self) -> MatchResult:
        """Convert slot state to a MatchResult.

        Only called on completed slots.  Buffer length consistency is guaranteed
        by the loop structure: obs/masks/perspective, actions, and rewards/dones
        are all appended within the same loop iteration, and stop_event is only
        checked between iterations.  If a forward pass raises mid-iteration,
        the slot is never completed, so to_result is never called.
        """
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
            feature_tracker=self.feature_tracker,
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
        epoch: int = 0,
    ) -> tuple[list[MatchResult], RoundStats]:
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
            trainable_fn: If provided, called with (entry_a, entry_b) to decide
                if rollout data should be collected for this pairing.

        Returns:
            Tuple of (results, stats). Results is a list of MatchResult for
            successfully completed pairings, ordered by their original index.
            Stats contains timing and throughput metrics for the round.
        """
        stats = RoundStats(pairings_requested=len(pairings))

        if not pairings:
            return [], stats

        if stop_event is not None and stop_event.is_set():
            logger.warning("run_round: stop_event already set, returning empty results")
            return [], stats

        round_start = time.monotonic()
        results: dict[int, MatchResult] = {}
        next_pairing_idx = 0
        total_pairings = len(pairings)
        parallel = min(self.config.effective_parallel, total_pairings)
        stats.active_slots = parallel

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
        # Black (player 0) always moves first in shogi; no colour
        # randomisation exists.  Matches match_utils.play_batch init.
        current_players = np.zeros(self.config.total_envs, dtype=np.uint8)

        # Assign initial pairings and track slot→pairing mapping.
        # If a pairing fails to load, the slot stays idle for this round
        # (unlike swap-in which retries with subsequent pairings).  This is
        # acceptable: idle slots just reduce concurrency slightly.
        slot_pairing_map: dict[int, int] = {}
        for slot in slots:
            if next_pairing_idx >= total_pairings:
                break
            pairing_idx = next_pairing_idx
            self._assign_pairing(
                slot, pairing_idx, pairings[pairing_idx],
                load_fn, games_per_match, trainable_fn, stats=stats, epoch=epoch,
            )
            if slot.active:
                slot_pairing_map[slot.index] = pairing_idx
            next_pairing_idx += 1

        # Game loop
        active_slots = [s for s in slots if s.active]

        try:
            while active_slots:
                if stop_event is not None and stop_event.is_set():
                    logger.warning("run_round: stop_event fired, returning partial results")
                    break

                # Increment ply for all active slots
                for slot in active_slots:
                    slot.ply_count += 1

                # Build action tensor for ALL envs.
                # Save pre-step player state per slot — needed for correct reward
                # attribution (VecEnv rewards are from last-mover perspective).
                actions = torch.zeros(self.config.total_envs, dtype=torch.long, device=device)
                pre_step_players: dict[int, np.ndarray] = {}

                for slot in active_slots:
                    s, e = slot.env_start, slot.env_end
                    partition_obs = obs[s:e]
                    partition_legal = legal_masks[s:e]
                    partition_players = current_players[s:e]
                    pre_step_players[slot.index] = partition_players.copy()

                    # Collect pre-step rollout data.  Exactly one append per
                    # slot per loop iteration — the loop structure guarantees
                    # the single-pass-per-ply invariant that rollout buffers
                    # depend on.
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

                # For inactive env ranges, pick first legal action.
                # Iterates ALL parallel_matches partitions (not just effective_parallel)
                # because the VecEnv has total_envs = parallel_matches * envs_per_match
                # and every env needs a valid action for step().
                inactive_ranges: list[tuple[int, int]] = []
                active_ranges = {(s.env_start, s.env_end) for s in active_slots}
                for slot_idx in range(self.config.parallel_matches):
                    env_s, env_e = self.partition_range(slot_idx)
                    if (env_s, env_e) not in active_ranges:
                        inactive_ranges.append((env_s, env_e))
                if inactive_ranges:
                    # Batched: argmax over legal_masks gives first legal action per
                    # env in one GPU kernel — no per-env GPU-CPU sync.
                    # If no legal actions (shouldn't occur), argmax returns 0.
                    inactive_idx = torch.cat([
                        torch.arange(s, e, device=device) for s, e in inactive_ranges
                    ])
                    actions[inactive_idx] = legal_masks[inactive_idx].to(torch.long).argmax(dim=-1)

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

                    # Post-step rollout: rewards and done flags.  After VecEnv
                    # auto-reset, the obs for the next iteration is from the NEW
                    # episode, not the terminal state.  This is correct: the done
                    # mask tells the training code to zero-bootstrap at terminal
                    # steps, making the post-reset obs irrelevant at boundaries.
                    if slot.collect_rollout:
                        slot._rewards.append(
                            torch.from_numpy(partition_rewards.copy().astype(np.float32))
                        )
                        slot._dones.append(
                            torch.from_numpy(partition_done.astype(np.float32))
                        )

                    # Count completed games.  Rewards are from the LAST-MOVER's
                    # perspective: +1 = last mover won, -1 = last mover lost.
                    # pre_step_players tells us who moved (0=A/Black, 1=B/White).
                    # Same convention as match_utils.play_batch.
                    slot_pre_players = pre_step_players[slot.index]

                    # Feature tracking for this slot's partition
                    if slot.feature_tracker is not None and hasattr(step_result, 'step_metadata'):
                        meta = step_result.step_metadata
                        slot.feature_tracker.record_step(
                            actions=actions[s:e].cpu().numpy(),
                            captured_piece=np.asarray(meta.captured_piece)[s:e],
                            termination_reason=np.asarray(meta.termination_reason)[s:e],
                            ply_count=np.asarray(meta.ply_count)[s:e],
                            pre_step_players=slot_pre_players,
                            terminated=partition_term,
                            truncated=partition_trunc,
                            rewards=partition_rewards,
                        )

                    for env_i in range(e - s):
                        if partition_done[env_i]:
                            r = float(partition_rewards[env_i])
                            a_moved = slot_pre_players[env_i] == 0
                            if r > 0:
                                if a_moved:
                                    slot.a_wins += 1
                                else:
                                    slot.b_wins += 1
                            elif r < 0:
                                if a_moved:
                                    slot.b_wins += 1
                                else:
                                    slot.a_wins += 1
                            else:
                                slot.draws += 1

                    if slot.games_completed >= slot.games_target:
                        completed_slot_indices.append(i)
                    else:
                        # Safety bound: allow enough plies for all game waves.
                        # Each wave fills envs_per_match envs; with auto-reset
                        # we need ceil(games_target / envs_per_match) waves.
                        envs_in_slot = slot.env_end - slot.env_start
                        waves_needed = -(-slot.games_target // max(1, envs_in_slot))
                        ply_ceiling = max_ply * (waves_needed + 1)
                        if slot.ply_count >= ply_ceiling:
                            logger.warning(
                                "Slot %d hit ply ceiling %d (max_ply=%d × %d waves) "
                                "with %d/%d games, yielding partial result",
                                slot.index, ply_ceiling, max_ply, waves_needed + 1,
                                slot.games_completed, slot.games_target,
                            )
                            completed_slot_indices.append(i)

                # Process completed slots (iterate in reverse to allow removal)
                for i in sorted(completed_slot_indices, reverse=True):
                    slot = active_slots.pop(i)
                    pairing_idx = slot_pairing_map[slot.index]
                    results[pairing_idx] = slot.to_result()

                    # Release models and clear refs so Python can GC them even if
                    # the next _assign_pairing fails (no swap-in overwrites the refs).
                    try:
                        release_fn(slot.model_a, slot.model_b)
                    except Exception:
                        logger.exception(
                            "Failed to release models for slot %d", slot.index
                        )
                    slot.model_a = None
                    slot.model_b = None

                    slot.active = False

                    # Swap in next pairing if available.
                    # NOTE: VecEnv only supports global reset(), not per-partition.
                    # Envs auto-reset on termination, so most are at move 0 when
                    # the slot completes. However, envs that finished early may be
                    # mid-game from auto-reset. This noise is negligible for
                    # typical games_target values (64+) and washes out in Elo.
                    while next_pairing_idx < total_pairings:
                        new_pairing_idx = next_pairing_idx
                        next_pairing_idx += 1
                        self._assign_pairing(
                            slot, new_pairing_idx, pairings[new_pairing_idx],
                            load_fn, games_per_match, trainable_fn, stats=stats, epoch=epoch,
                        )
                        if slot.active:
                            slot_pairing_map[slot.index] = new_pairing_idx
                            active_slots.append(slot)
                            break
                        # Load failed — try next pairing instead of skipping

        finally:
            # Release any remaining active models (stop_event or unexpected exception).
            # No double-release risk: completed slots are pop()-ed from active_slots
            # and their models released above.  If swap-in loads new models, the slot
            # is re-appended with those new refs.  If swap-in fails, the slot is NOT
            # re-appended.  So active_slots only contains unreleased models.
            for slot in active_slots:
                try:
                    release_fn(slot.model_a, slot.model_b)
                except Exception:
                    logger.exception(
                        "Failed to release models for slot %d", slot.index
                    )
                slot.model_a = None
                slot.model_b = None

        # Collect stats
        stats.round_duration_s = time.monotonic() - round_start
        ordered = [results[i] for i in sorted(results.keys())]
        stats.pairings_completed = len(ordered)
        stats.total_games = sum(r.a_wins + r.b_wins + r.draws for r in ordered)
        stats.total_plies = sum(s.ply_count for s in slots)

        if stats.round_duration_s > 0:
            gpm = stats.total_games / stats.round_duration_s * 60
            logger.info(
                "Round complete: %d/%d pairings, %d games, %d plies in %.1fs "
                "(%.0f games/min, %d loads in %.2fs)",
                stats.pairings_completed, stats.pairings_requested,
                stats.total_games, stats.total_plies, stats.round_duration_s,
                gpm, stats.model_load_count, stats.model_load_time_s,
            )

        return ordered, stats

    def _assign_pairing(
        self,
        slot: _MatchSlot,
        pairing_idx: int,
        pairing: tuple[OpponentEntry, OpponentEntry],
        load_fn: Callable[[OpponentEntry], Any],
        games_target: int,
        trainable_fn: Callable[[OpponentEntry, OpponentEntry], bool] | None = None,
        stats: RoundStats | None = None,
        epoch: int = 0,
    ) -> None:
        """Load models and assign a pairing to a slot."""
        entry_a, entry_b = pairing
        try:
            t0 = time.monotonic()
            model_a = load_fn(entry_a)
            try:
                model_b = load_fn(entry_b)
            except Exception:
                # model_a loaded successfully but model_b failed — move model_a
                # off GPU and reclaim CUDA cache to avoid a memory leak.
                # del only drops the local binding; .cpu() actually frees VRAM.
                try:
                    model_a.cpu()
                    del model_a
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                raise
        except Exception:
            logger.warning(
                "Failed to load models for pairing %d (%s vs %s), skipping",
                pairing_idx,
                entry_a.display_name,
                entry_b.display_name,
                exc_info=True,
            )
            slot.model_a = None
            slot.model_b = None
            slot.active = False
            return

        load_elapsed = time.monotonic() - t0
        if stats is not None:
            stats.model_load_time_s += load_elapsed
            stats.model_load_count += 2

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
            epoch=epoch,
        )
