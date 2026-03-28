"""Continuous match scheduler for the Elo ladder spectator system.

Runs model-vs-model games forever, updates Elo ratings, and broadcasts
live game state for the dashboard. Maintains N concurrent game slots:
the first K are spectated (paced, state published), the rest are
background (full speed, Elo only).
"""

import asyncio
import enum
import logging
import math
import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from keisei.utils.agent_loading import load_evaluation_agent

logger = logging.getLogger(__name__)

_ELO_PROXIMITY_SCALE = 200.0
_UNCERTAINTY_MIN_GAMES = 1  # floor to prevent division by zero in 1/sqrt(games)


class MatchOutcome(enum.Enum):
    """Winner of a completed game."""

    BLACK_WIN = "black_win"
    WHITE_WIN = "white_win"
    DRAW = "draw"


@dataclass
class MatchResult:
    """Result from a completed game."""

    winner: MatchOutcome
    move_count: int
    reason: str


@dataclass
class ActiveMatchState:
    """Typed state for an in-progress match slot."""

    match_id: str
    model_a: Dict[str, Any]
    model_b: Dict[str, Any]
    sfen: str
    move_count: int
    move_log: List[str]
    status: str
    spectated: bool


class SchedulerConfig(BaseModel):
    """Configuration for ContinuousMatchScheduler.

    Co-located with the scheduler for cohesion. The ladder subcommand in
    train.py constructs this from AppConfig defaults + CLI overrides.

    Note: model_type, tower_depth, tower_width, se_ratio MUST match the
    architecture used during training. A mismatch will cause state_dict
    load failures caught as RuntimeError in _run_match. These are
    duplicated from TrainingConfig until checkpoint metadata includes
    architecture info (tracked as a follow-up).
    """

    checkpoint_dir: Path
    elo_registry_path: Path
    device: str = "cuda"
    num_concurrent: int = Field(6, ge=1, le=32)
    num_spectated: int = Field(3, ge=0)
    move_delay: float = Field(1.5, ge=0.0)
    poll_interval: float = Field(30.0, ge=1.0)
    max_moves_per_game: int = Field(500, ge=1)
    pool_size: int = Field(50, ge=2, le=200)  # Capped: _pick_matchup is O(n²)
    input_channels: int = 46
    input_features: str = "core46"
    model_type: str = "resnet"
    tower_depth: int = Field(9, ge=1)
    tower_width: int = Field(256, ge=1)
    se_ratio: Optional[float] = 0.25
    state_path: Optional[Path] = None
    max_consecutive_failures: int = Field(5, ge=1, le=100)
    move_timeout: float = Field(30.0, ge=1.0, description="Per-move timeout in seconds")
    fast_track_games: int = Field(
        8, ge=0, le=50,
        description="Games to play immediately when a new checkpoint is discovered",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("device")
    @classmethod
    def validate_device_string(cls, v: str) -> str:
        """Reject invalid device strings early instead of failing in PyTorch."""
        import re

        if not re.match(r"^(cpu|cuda(:\d+)?)$", v):
            raise ValueError(
                f"Invalid device '{v}': must be 'cpu', 'cuda', or 'cuda:N'"
            )
        return v

    @model_validator(mode="after")
    def validate_spectated_within_concurrent(self) -> "SchedulerConfig":
        """Ensure num_spectated does not exceed num_concurrent."""
        if self.num_spectated > self.num_concurrent:
            raise ValueError(
                f"num_spectated ({self.num_spectated}) must be <= "
                f"num_concurrent ({self.num_concurrent})"
            )
        return self


class ContinuousMatchScheduler:
    """Continuous Elo ladder match scheduler."""

    def __init__(self, config: SchedulerConfig):
        from keisei.evaluation.opponents.elo_registry import EloRegistry
        from keisei.evaluation.opponents.opponent_pool import OpponentPool
        from keisei.utils.utils import PolicyOutputMapper

        self._config = config

        # Shared policy mapper (one instance, reused across all matches)
        self._policy_mapper = PolicyOutputMapper()

        # State — scheduler owns the EloRegistry exclusively.
        # OpponentPool gets elo_registry_path=None to prevent a second
        # registry instance from racing on the same file.
        self._pool = OpponentPool(pool_size=config.pool_size, elo_registry_path=None)
        self._elo_registry = EloRegistry(config.elo_registry_path)
        self._pool_paths: List[Path] = []
        # Games played is persisted in EloRegistry — survives restarts
        self._games_played = Counter(self._elo_registry.get_all_games_played())
        # Wins persisted in EloRegistry — survives restarts
        self._wins: Counter = Counter(self._elo_registry.get_all_wins())
        # Fast-track queue: new models get N dedicated games for initial rating
        self._fast_track_queue: deque = deque()
        self._active_matches: Dict[int, ActiveMatchState] = {}
        self._match_tasks: Dict[int, asyncio.Task] = {}
        self._recent_results: List[Dict[str, Any]] = []
        self._elo_lock = asyncio.Lock()
        # Serialize GPU inference across concurrent slots — PyTorch's
        # internal state (cuBLAS workspace, allocator) is not thread-safe.
        self._inference_semaphore = asyncio.Semaphore(1)
        # Blacklist for checkpoints that fail to load — cleared on pool refresh
        self._failed_checkpoints: Set[Path] = set()
        # Circuit breaker: consecutive match failures across all slots
        self._consecutive_failures: int = 0
        self._state_path = config.state_path or (Path(".keisei_ladder") / "state.json")

    def _get_rating(self, name: str) -> float:
        """Get Elo rating for a model by checkpoint filename."""
        return self._elo_registry.get_rating(name)

    def _refresh_pool(self) -> int:
        """Scan checkpoint directory and update internal pool paths."""
        old_paths = set(self._pool_paths)
        added = self._pool.scan_directory(self._config.checkpoint_dir)
        self._pool_paths = list(self._pool.get_all())
        # Clear blacklist on refresh — files may have been restored or
        # the issue may have been transient (NFS, incomplete write).
        if added > 0:
            self._failed_checkpoints.clear()
            # Fast-track new models for initial rating
            if self._config.fast_track_games > 0:
                new_paths = set(self._pool_paths) - old_paths
                for path in new_paths:
                    self._fast_track_queue.append(
                        (path, self._config.fast_track_games)
                    )
                    logger.info(
                        "Fast-tracking %s for %d initial games",
                        path.name, self._config.fast_track_games,
                    )
        return added

    def _pick_fast_track_matchup(self) -> Optional[Tuple[Path, Path]]:
        """Pick a match for a fast-tracked model, or None if queue is empty.

        Pairs the fast-tracked model with a random opponent from the pool.
        Decrements the remaining games counter; removes from queue when done.
        """
        while self._fast_track_queue:
            model_path, remaining = self._fast_track_queue[0]
            # Drop entries that are blacklisted, evicted from pool, or deleted
            if (
                model_path in self._failed_checkpoints
                or model_path not in self._pool_paths
            ):
                self._fast_track_queue.popleft()
                continue
            # Pick a random opponent that isn't the same model or blacklisted
            candidates = [
                p for p in self._pool_paths
                if p != model_path and p not in self._failed_checkpoints
            ]
            if not candidates:
                break
            opponent = random.choice(candidates)
            # Decrement or remove from queue
            if remaining <= 1:
                self._fast_track_queue.popleft()
            else:
                self._fast_track_queue[0] = (model_path, remaining - 1)
            # Randomize who plays Black/White
            if random.random() < 0.5:
                return model_path, opponent
            return opponent, model_path
        return None

    def _pick_matchup(self) -> Tuple[Path, Path]:
        """Select two models for a match using weighted random by Elo proximity.

        Weight = proximity * uncertainty_a * uncertainty_b, where:
        - proximity = 1 / (1 + |elo_a - elo_b| / 200)
        - uncertainty = 1 / sqrt(max(1, games_played))

        A continuous uncertainty weight is used instead of a binary threshold
        (e.g., a fixed N-game boost for new models). This avoids the "Success
        to the Successful" archetype where models with poor initial ratings
        get locked out of future matches.
        """
        paths = [p for p in self._pool_paths if p not in self._failed_checkpoints]
        if len(paths) < 2:
            raise ValueError(
                f"Need at least 2 models for a match, have {len(paths)}"
            )

        # Build per-model uncertainty weights: 1/sqrt(games) — naturally
        # decays as confidence grows, no abrupt threshold cutoff.
        model_weights = {}
        for p in paths:
            games = max(_UNCERTAINTY_MIN_GAMES, self._games_played.get(p.name, 0))
            model_weights[p] = 1.0 / math.sqrt(games)

        # Build pair weights: proximity * uncertainty_a * uncertainty_b
        pairs = []
        weights = []
        for i, a in enumerate(paths):
            for b in paths[i + 1 :]:
                elo_a = self._get_rating(a.name)
                elo_b = self._get_rating(b.name)
                proximity = 1.0 / (
                    1.0 + abs(elo_a - elo_b) / _ELO_PROXIMITY_SCALE
                )
                pair_weight = proximity * model_weights[a] * model_weights[b]
                pairs.append((a, b))
                weights.append(pair_weight)

        # Weighted random selection
        (selected,) = random.choices(pairs, weights=weights, k=1)
        # Randomize who plays Sente/Gote
        if random.random() < 0.5:
            return selected[0], selected[1]
        return selected[1], selected[0]

    def _build_state_snapshot(self) -> Dict[str, Any]:
        """Build the state dict for publishing (pure computation, no I/O)."""
        leaderboard = []
        for name, elo in sorted(
            self._elo_registry.get_all_ratings().items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            games = self._games_played.get(name, 0)
            wins = self._wins.get(name, 0)
            leaderboard.append({
                "name": name,
                "elo": round(elo, 1),
                "games_played": games,
                "win_rate": round(wins / games, 3) if games > 0 else 0.0,
            })

        matches = []
        for slot, match in self._active_matches.items():
            entry = {
                "slot": slot,
                "spectated": match.spectated,
                "match_id": match.match_id,
                "model_a": match.model_a,
                "model_b": match.model_b,
                "move_count": match.move_count,
                "status": match.status,
            }
            if match.spectated:
                entry["sfen"] = match.sfen
                entry["move_log"] = match.move_log
            matches.append(entry)

        return {
            "schema_version": "ladder-v1",
            "timestamp": time.time(),
            "matches": matches,
            "leaderboard": leaderboard,
            "recent_results": self._recent_results[-20:],
        }

    def _write_state_sync(self, state: Dict[str, Any]) -> None:
        """Synchronous file write — called via asyncio.to_thread."""
        from keisei.webui.state_snapshot import write_snapshot_atomic

        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        write_snapshot_atomic(state, self._state_path)

    async def _publish_state(self) -> None:
        """Write atomic JSON state file for the spectator dashboard.

        Schema: "ladder-v1" — consumed by the future spectator dashboard
        (see filigree keisei-8f408d3360). NOT compatible with the training
        dashboard's BroadcastStateEnvelope ("v1.0.0") format. This is a
        separate file at a separate path for the ladder subsystem.

        File I/O is offloaded to a thread to avoid blocking the event loop.
        """
        state = self._build_state_snapshot()
        await asyncio.to_thread(self._write_state_sync, state)

    async def _run_game_loop(
        self,
        game: Any,
        agent_a: Any,
        agent_b: Any,
        spectated: bool,
        slot: int,
    ) -> MatchResult:
        """Play a game to completion. Pace and publish if spectated."""
        import torch
        from keisei.shogi.shogi_core_definitions import Color

        move_count = 0
        agents = {Color.BLACK: agent_a, Color.WHITE: agent_b}
        move_log: List[str] = []

        def _opponent_wins(agent_idx: int) -> MatchOutcome:
            return MatchOutcome.WHITE_WIN if agent_idx == 0 else MatchOutcome.BLACK_WIN

        while move_count < self._config.max_moves_per_game:
            current_color = game.current_player
            current_agent_idx = 0 if current_color == Color.BLACK else 1
            agent = agents[current_color]

            legal_moves = game.get_legal_moves()
            if not legal_moves:
                logger.warning(
                    "Slot %d: no legal moves at move %d (SFEN: %s)",
                    slot, move_count, game.to_sfen(),
                )
                return MatchResult(
                    winner=_opponent_wins(current_agent_idx),
                    move_count=move_count,
                    reason="no_legal_moves",
                )

            obs = game.get_observation()
            agent_device = getattr(agent, "device", torch.device("cpu"))
            if isinstance(agent_device, str):
                agent_device = torch.device(agent_device)
            legal_mask = self._policy_mapper.get_legal_mask(
                legal_moves, device=agent_device
            )

            # Offload synchronous PyTorch inference to a thread so the
            # asyncio event loop stays responsive. Timeout prevents hung
            # inference from permanently consuming a slot.
            try:
                async with self._inference_semaphore:
                    selected_move, _, _, _ = await asyncio.wait_for(
                        asyncio.to_thread(
                            agent.select_action, obs, legal_mask, is_training=False
                        ),
                        timeout=self._config.move_timeout,
                    )
            except asyncio.TimeoutError:
                logger.error(
                    "Slot %d: inference timed out at move %d (>%.1fs)",
                    slot, move_count, self._config.move_timeout,
                )
                return MatchResult(
                    winner=_opponent_wins(current_agent_idx),
                    move_count=move_count,
                    reason="inference_timeout",
                )

            if selected_move is None:
                logger.warning(
                    "Slot %d: agent returned None action at move %d",
                    slot, move_count,
                )
                return MatchResult(
                    winner=_opponent_wins(current_agent_idx),
                    move_count=move_count,
                    reason="no_action_selected",
                )

            _, reward, done, info = game.make_move(selected_move)
            move_count += 1
            move_log.append(str(selected_move))

            if spectated:
                match_state = self._active_matches[slot]
                match_state.sfen = game.to_sfen()
                match_state.move_count = move_count
                match_state.move_log = move_log[-20:]
                match_state.status = "in_progress"
                try:
                    await self._publish_state()
                except Exception:
                    logger.exception("Slot %d: failed to publish state mid-game", slot)
                await asyncio.sleep(self._config.move_delay)

            if done:
                winner_raw = info.get("winner")
                if winner_raw == "black" or winner_raw == 0:
                    outcome = MatchOutcome.BLACK_WIN
                elif winner_raw == "white" or winner_raw == 1:
                    outcome = MatchOutcome.WHITE_WIN
                else:
                    outcome = MatchOutcome.DRAW
                return MatchResult(
                    winner=outcome,
                    move_count=move_count,
                    reason=info.get("terminal_reason", "game_over"),
                )

        return MatchResult(
            winner=MatchOutcome.DRAW,
            move_count=move_count,
            reason="max_moves",
        )

    async def _run_match(
        self, slot: int, model_a_path: Path, model_b_path: Path
    ) -> None:
        """Run a single match: load models, play game, update Elo."""
        import torch
        from keisei.shogi.shogi_game import ShogiGame

        cfg = self._config
        spectated = slot < cfg.num_spectated
        name_a = model_a_path.name
        name_b = model_b_path.name

        logger.info(
            "Slot %d: %s vs %s (%s)",
            slot, name_a, name_b, "spectated" if spectated else "background",
        )

        agent_a = None
        agent_b = None
        try:
            # Log GPU memory before loading models for concurrent slots.
            if cfg.device.startswith("cuda") and torch.cuda.is_available():
                mem_gb = torch.cuda.memory_allocated() / 1e9
                logger.info(
                    "Slot %d: GPU memory before load: %.2f GB", slot, mem_gb
                )

            agent_a = load_evaluation_agent(
                str(model_a_path), cfg.device, self._policy_mapper,
                cfg.input_channels, cfg.input_features,
                model_type=cfg.model_type,
                tower_depth=cfg.tower_depth,
                tower_width=cfg.tower_width,
                se_ratio=cfg.se_ratio,
            )
            agent_b = load_evaluation_agent(
                str(model_b_path), cfg.device, self._policy_mapper,
                cfg.input_channels, cfg.input_features,
                model_type=cfg.model_type,
                tower_depth=cfg.tower_depth,
                tower_width=cfg.tower_width,
                se_ratio=cfg.se_ratio,
            )

            game = ShogiGame(max_moves_per_game=cfg.max_moves_per_game)
            game.reset()

            self._active_matches[slot] = ActiveMatchState(
                match_id=f"{name_a}_vs_{name_b}_{int(time.time())}",
                model_a={"name": name_a, "elo": self._get_rating(name_a)},
                model_b={"name": name_b, "elo": self._get_rating(name_b)},
                sfen=game.to_sfen(),
                move_count=0,
                move_log=[],
                status="in_progress",
                spectated=spectated,
            )

            result = await self._run_game_loop(
                game, agent_a, agent_b, spectated, slot
            )

            elo_result_map = {
                MatchOutcome.BLACK_WIN: "agent_win",
                MatchOutcome.WHITE_WIN: "opponent_win",
                MatchOutcome.DRAW: "draw",
            }
            elo_result = elo_result_map[result.winner]

            # Serialize Elo + win updates so concurrent _run_match tasks
            # don't interleave reads/writes across await boundaries.
            async with self._elo_lock:
                old_elo_a = self._get_rating(name_a)
                old_elo_b = self._get_rating(name_b)
                self._elo_registry.update_ratings(name_a, name_b, [elo_result])
                if result.winner == MatchOutcome.BLACK_WIN:
                    self._elo_registry.record_win(name_a)
                elif result.winner == MatchOutcome.WHITE_WIN:
                    self._elo_registry.record_win(name_b)
                await asyncio.to_thread(self._elo_registry.save)
                new_elo_a = self._get_rating(name_a)
                new_elo_b = self._get_rating(name_b)

                # Sync local counters from registry
                self._games_played = Counter(self._elo_registry.get_all_games_played())
                self._wins = Counter(self._elo_registry.get_all_wins())

            winner_name_map = {
                MatchOutcome.BLACK_WIN: name_a,
                MatchOutcome.WHITE_WIN: name_b,
                MatchOutcome.DRAW: "draw",
            }
            match_result = {
                "model_a": name_a,
                "model_b": name_b,
                "winner": winner_name_map[result.winner],
                "elo_delta_a": round(new_elo_a - old_elo_a, 1),
                "elo_delta_b": round(new_elo_b - old_elo_b, 1),
                "move_count": result.move_count,
                "reason": result.reason,
                "timestamp": time.time(),
            }
            self._recent_results.append(match_result)
            self._recent_results = self._recent_results[-50:]
            self._consecutive_failures = 0  # Reset circuit breaker on success

            logger.info(
                "Slot %d: %s (%+.1f) vs %s (%+.1f) — %d moves, %s",
                slot, name_a, new_elo_a - old_elo_a,
                name_b, new_elo_b - old_elo_b,
                result.move_count, result.reason,
            )

        except FileNotFoundError as e:
            logger.error("Slot %d: checkpoint not found: %s", slot, e)
            # Blacklist only the checkpoint(s) that are actually missing
            if not model_a_path.exists():
                self._failed_checkpoints.add(model_a_path)
            if not model_b_path.exists():
                self._failed_checkpoints.add(model_b_path)
            self._consecutive_failures += 1
        except ValueError as e:
            logger.error("Slot %d: invalid configuration: %s", slot, e)
            self._consecutive_failures += 1
        except RuntimeError as e:
            logger.error(
                "Slot %d: model load/inference failed for %s vs %s: %s",
                slot, name_a, name_b, e,
            )
            # Architecture mismatch or corrupt weights — blacklist both
            self._failed_checkpoints.add(model_a_path)
            self._failed_checkpoints.add(model_b_path)
            self._consecutive_failures += 1
        except Exception:
            logger.exception("Slot %d: unexpected match failure (%s vs %s)", slot, name_a, name_b)
            self._consecutive_failures += 1
        finally:
            # Explicitly free model GPU memory
            del agent_a, agent_b
            if cfg.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._active_matches.pop(slot, None)
            try:
                await self._publish_state()
            except Exception:
                logger.exception("Slot %d: failed to publish state", slot)

    async def run(self) -> None:
        """Run the scheduler forever. Cancel to stop."""
        logger.info("Starting ContinuousMatchScheduler")
        self._refresh_pool()
        logger.info("Pool has %d models", len(self._pool_paths))

        if len(self._pool_paths) < 2:
            logger.warning(
                "Need at least 2 checkpoints to start. Waiting for models..."
            )

        poll_task = asyncio.create_task(self._poll_checkpoints_loop())
        slot_task = asyncio.create_task(self._manage_game_slots())

        try:
            await asyncio.gather(poll_task, slot_task)
        except asyncio.CancelledError:
            poll_task.cancel()
            slot_task.cancel()
            # Await in-flight match tasks so they can finish cleanly
            pending = [
                t for t in self._match_tasks.values() if not t.done()
            ]
            if pending:
                logger.info("Awaiting %d in-flight matches...", len(pending))
                await asyncio.gather(*pending, return_exceptions=True)
            logger.info("Scheduler stopped")
            raise

    async def _manage_game_slots(self) -> None:
        """Keep all N game slots filled with matches."""
        while True:
            if len(self._pool_paths) < 2:
                await asyncio.sleep(1.0)
                continue

            # Circuit breaker: stop scheduling if too many consecutive failures
            if self._consecutive_failures >= self._config.max_consecutive_failures:
                logger.error(
                    "Circuit breaker: %d consecutive failures, pausing for %ds. "
                    "Blacklisted checkpoints: %s",
                    self._consecutive_failures, 30,
                    [str(p) for p in self._failed_checkpoints],
                )
                await asyncio.sleep(30.0)
                # Reset counter but preserve blacklist — blacklist is only
                # cleared when _refresh_pool discovers new checkpoints.
                self._consecutive_failures = 0
                continue

            for slot_id in range(self._config.num_concurrent):
                if slot_id not in self._match_tasks or self._match_tasks[slot_id].done():
                    # Fast-track new models first, then normal weighted selection
                    ft_pair = self._pick_fast_track_matchup()
                    if ft_pair is not None:
                        model_a, model_b = ft_pair
                    else:
                        try:
                            model_a, model_b = self._pick_matchup()
                        except ValueError as e:
                            logger.warning("Cannot schedule match for slot %d: %s", slot_id, e)
                            break
                    task = asyncio.create_task(
                        self._run_match(slot_id, model_a, model_b)
                    )
                    self._match_tasks[slot_id] = task

            await asyncio.sleep(0.1)

    async def _poll_checkpoints_loop(self) -> None:
        """Periodically scan for new checkpoints."""
        while True:
            await asyncio.sleep(self._config.poll_interval)
            added = self._refresh_pool()
            if added > 0:
                logger.info("Found %d new checkpoints", added)
