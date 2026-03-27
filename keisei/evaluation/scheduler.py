"""Continuous match scheduler for the Elo ladder spectator system.

Runs model-vs-model games forever, updates Elo ratings, and broadcasts
live game state for the dashboard. Maintains N concurrent game slots:
the first K are spectated (paced, state published), the rest are
background (full speed, Elo only).
"""

import asyncio
import json
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_NEWMODEL_GAME_THRESHOLD = 5
_NEWMODEL_WEIGHT_BOOST = 3.0
_ELO_PROXIMITY_SCALE = 200.0


@dataclass
class MatchResult:
    """Typed result from a completed game."""

    done: bool
    winner: Optional[int]  # 0=Black, 1=White, None=Draw
    move_count: int
    reason: str


class SchedulerConfig(BaseModel):
    """Configuration for ContinuousMatchScheduler.

    Defined here for now; move to keisei/config_schema.py when the scheduler
    is wired into the train.py ladder subcommand.
    """

    checkpoint_dir: Path
    elo_registry_path: Path
    device: str = "cuda"
    num_concurrent: int = Field(6, ge=1, le=32)
    num_spectated: int = Field(3, ge=0)
    move_delay: float = Field(1.5, ge=0.0)
    poll_interval: float = Field(30.0, ge=1.0)
    max_moves_per_game: int = Field(500, ge=1)
    pool_size: int = Field(50, ge=2, le=1000)
    input_channels: int = 46
    input_features: str = "core46"
    model_type: str = "resnet"
    tower_depth: int = Field(9, ge=1)
    tower_width: int = Field(256, ge=1)
    se_ratio: Optional[float] = 0.25
    state_path: Optional[Path] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ContinuousMatchScheduler:
    """Continuous Elo ladder match scheduler."""

    def __init__(self, config: SchedulerConfig):
        from keisei.evaluation.opponents.elo_registry import EloRegistry
        from keisei.evaluation.opponents.opponent_pool import OpponentPool
        from keisei.utils.utils import PolicyOutputMapper

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.device = config.device
        self.num_concurrent = config.num_concurrent
        self.num_spectated = config.num_spectated
        self.move_delay = config.move_delay
        self.poll_interval = config.poll_interval
        self.max_moves_per_game = config.max_moves_per_game
        self.input_channels = config.input_channels
        self.input_features = config.input_features
        self.model_type = config.model_type
        self.tower_depth = config.tower_depth
        self.tower_width = config.tower_width
        self.se_ratio = config.se_ratio

        # Shared policy mapper (one instance, reused across all matches)
        self._policy_mapper = PolicyOutputMapper()

        # State — scheduler owns the EloRegistry exclusively.
        # OpponentPool gets elo_registry_path=None to prevent a second
        # registry instance from racing on the same file (B1 fix).
        self._pool = OpponentPool(pool_size=config.pool_size, elo_registry_path=None)
        self._elo_registry = EloRegistry(config.elo_registry_path)
        self._pool_paths: List[Path] = []
        self._games_played: Counter = Counter()
        self._active_matches: Dict[int, Dict[str, Any]] = {}
        self._match_tasks: Dict[int, asyncio.Task] = {}
        self._recent_results: List[Dict[str, Any]] = []
        self._state_path = config.state_path or (Path(".keisei_ladder") / "state.json")

    def _get_rating(self, name: str) -> float:
        """Get Elo rating for a model by checkpoint filename."""
        return self._elo_registry.get_rating(name)

    def _refresh_pool(self) -> int:
        """Scan checkpoint directory and update internal pool paths."""
        added = self._pool.scan_directory(self.checkpoint_dir)
        self._pool_paths = list(self._pool.get_all())
        return added

    def _pick_matchup(self) -> Tuple[Path, Path]:
        """Select two models for a match using weighted random by Elo proximity.

        Weight = 1 / (1 + |elo_a - elo_b| / 200).
        Models with <5 games get a 3x weight boost.
        """
        paths = self._pool_paths
        if len(paths) < 2:
            raise ValueError(
                f"Need at least 2 models for a match, have {len(paths)}"
            )

        # Build per-model weights (boost new models)
        model_weights = {}
        for p in paths:
            games = self._games_played.get(p.name, 0)
            boost = _NEWMODEL_WEIGHT_BOOST if games < _NEWMODEL_GAME_THRESHOLD else 1.0
            model_weights[p] = boost

        # Build pair weights
        pairs = []
        weights = []
        for i, a in enumerate(paths):
            for b in paths[i + 1 :]:
                elo_a = self._get_rating(a.name)
                elo_b = self._get_rating(b.name)
                proximity_weight = 1.0 / (
                    1.0 + abs(elo_a - elo_b) / _ELO_PROXIMITY_SCALE
                )
                pair_weight = proximity_weight * model_weights[a] * model_weights[b]
                pairs.append((a, b))
                weights.append(pair_weight)

        # Weighted random selection
        (selected,) = random.choices(pairs, weights=weights, k=1)
        # Randomize who plays Sente/Gote
        if random.random() < 0.5:
            return selected[0], selected[1]
        return selected[1], selected[0]

    def _publish_state(self) -> None:
        """Write atomic JSON state file for the spectator dashboard.

        Schema: "ladder-v1" — consumed by the future spectator dashboard
        (see filigree keisei-8f408d3360). NOT compatible with the training
        dashboard's BroadcastStateEnvelope ("v1.0.0") format. This is a
        separate file at a separate path for the ladder subsystem.
        """
        # Build leaderboard from Elo registry
        leaderboard = []
        for name, elo in sorted(
            self._elo_registry.ratings.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            games = self._games_played.get(name, 0)
            leaderboard.append({
                "name": name,
                "elo": round(elo, 1),
                "games_played": games,
            })

        # Build matches list (spectated only get full state)
        matches = []
        for slot, match in self._active_matches.items():
            entry = {
                "slot": slot,
                "spectated": match.get("spectated", False),
                "match_id": match.get("match_id", ""),
                "model_a": match.get("model_a", {}),
                "model_b": match.get("model_b", {}),
                "move_count": match.get("move_count", 0),
                "status": match.get("status", "unknown"),
            }
            if match.get("spectated"):
                entry["sfen"] = match.get("sfen")
                entry["move_log"] = match.get("move_log", [])
            matches.append(entry)

        state = {
            "schema_version": "ladder-v1",
            "timestamp": time.time(),
            "matches": matches,
            "leaderboard": leaderboard,
            "recent_results": self._recent_results[-20:],
        }

        # Use existing atomic write utility — handles cleanup on failure,
        # uses tempfile.mkstemp + os.replace.
        from keisei.webui.state_snapshot import write_snapshot_atomic

        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        write_snapshot_atomic(state, self._state_path)

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

        while move_count < self.max_moves_per_game:
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
                    done=True,
                    winner=1 - current_agent_idx,
                    move_count=move_count,
                    reason="no_legal_moves",
                )

            obs = game.get_observation()
            legal_mask = self._policy_mapper.get_legal_mask(
                legal_moves, device=torch.device("cpu")
            )

            # Offload synchronous PyTorch inference to a thread so the
            # asyncio event loop stays responsive (B3 fix).
            # Note: is_training is keyword-only in PPOAgent.select_action.
            selected_move, _, _, _ = await asyncio.to_thread(
                lambda: agent.select_action(obs, legal_mask, is_training=False)
            )

            if selected_move is None:
                logger.warning(
                    "Slot %d: agent returned None action at move %d",
                    slot, move_count,
                )
                return MatchResult(
                    done=True,
                    winner=1 - current_agent_idx,
                    move_count=move_count,
                    reason="no_action_selected",
                )

            _, reward, done, info = game.make_move(selected_move)
            move_count += 1
            move_log.append(str(selected_move))

            if spectated:
                self._active_matches[slot].update({
                    "sfen": game.to_sfen(),
                    "move_count": move_count,
                    "move_log": move_log[-20:],
                    "status": "in_progress",
                })
                self._publish_state()
                await asyncio.sleep(self.move_delay)

            if done:
                winner = info.get("winner")
                winner_idx = None
                if winner == "black" or winner == 0:
                    winner_idx = 0
                elif winner == "white" or winner == 1:
                    winner_idx = 1
                return MatchResult(
                    done=True,
                    winner=winner_idx,
                    move_count=move_count,
                    reason=info.get("terminal_reason", "game_over"),
                )

        return MatchResult(
            done=True,
            winner=None,
            move_count=move_count,
            reason="max_moves",
        )

    async def _run_match(
        self, slot: int, model_a_path: Path, model_b_path: Path
    ) -> None:
        """Run a single match: load models, play game, update Elo."""
        import torch
        from keisei.shogi.shogi_game import ShogiGame
        from keisei.utils.agent_loading import load_evaluation_agent

        spectated = slot < self.num_spectated
        name_a = model_a_path.name
        name_b = model_b_path.name

        logger.info(
            "Slot %d: %s vs %s (%s)",
            slot, name_a, name_b, "spectated" if spectated else "background",
        )

        try:
            # W1: Log GPU memory usage when loading models for concurrent slots.
            if self.device.startswith("cuda"):
                import torch as _torch
                if _torch.cuda.is_available():
                    mem_gb = _torch.cuda.memory_allocated() / 1e9
                    logger.info(
                        "Slot %d: GPU memory before load: %.2f GB", slot, mem_gb
                    )

            agent_a = load_evaluation_agent(
                str(model_a_path), self.device, self._policy_mapper,
                self.input_channels, self.input_features,
                model_type=self.model_type,
                tower_depth=self.tower_depth,
                tower_width=self.tower_width,
                se_ratio=self.se_ratio,
            )
            agent_b = load_evaluation_agent(
                str(model_b_path), self.device, self._policy_mapper,
                self.input_channels, self.input_features,
                model_type=self.model_type,
                tower_depth=self.tower_depth,
                tower_width=self.tower_width,
                se_ratio=self.se_ratio,
            )

            game = ShogiGame(max_moves_per_game=self.max_moves_per_game)
            game.reset()

            self._active_matches[slot] = {
                "match_id": f"{name_a}_vs_{name_b}_{int(time.time())}",
                "model_a": {"name": name_a, "elo": self._get_rating(name_a)},
                "model_b": {"name": name_b, "elo": self._get_rating(name_b)},
                "sfen": game.to_sfen(),
                "move_count": 0,
                "move_log": [],
                "status": "in_progress",
                "spectated": spectated,
            }

            result = await self._run_game_loop(
                game, agent_a, agent_b, spectated, slot
            )

            if result.winner == 0:
                elo_result = "agent_win"
            elif result.winner == 1:
                elo_result = "opponent_win"
            else:
                elo_result = "draw"

            old_elo_a = self._get_rating(name_a)
            old_elo_b = self._get_rating(name_b)
            self._elo_registry.update_ratings(name_a, name_b, [elo_result])
            self._elo_registry.save()
            new_elo_a = self._get_rating(name_a)
            new_elo_b = self._get_rating(name_b)

            self._games_played[name_a] += 1
            self._games_played[name_b] += 1

            match_result = {
                "model_a": name_a,
                "model_b": name_b,
                "winner": name_a if result.winner == 0 else (name_b if result.winner == 1 else "draw"),
                "elo_delta_a": round(new_elo_a - old_elo_a, 1),
                "elo_delta_b": round(new_elo_b - old_elo_b, 1),
                "move_count": result.move_count,
                "reason": result.reason,
                "timestamp": time.time(),
            }
            self._recent_results.append(match_result)
            self._recent_results = self._recent_results[-50:]

            logger.info(
                "Slot %d: %s (%+.1f) vs %s (%+.1f) — %d moves, %s",
                slot, name_a, new_elo_a - old_elo_a,
                name_b, new_elo_b - old_elo_b,
                result.move_count, result.reason,
            )

        except FileNotFoundError as e:
            logger.error("Slot %d: checkpoint not found: %s", slot, e)
        except ValueError as e:
            logger.error("Slot %d: invalid configuration: %s", slot, e)
        except Exception:
            logger.exception("Slot %d: unexpected match failure", slot)
        finally:
            self._active_matches.pop(slot, None)
            try:
                self._publish_state()
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
            # W8 fix: await in-flight match tasks so they can finish
            pending = [
                t for t in self._match_tasks.values() if not t.done()
            ]
            if pending:
                logger.info("Awaiting %d in-flight matches...", len(pending))
                await asyncio.gather(*pending, return_exceptions=True)
            logger.info("Scheduler stopped")

    async def _manage_game_slots(self) -> None:
        """Keep all N game slots filled with matches."""
        while True:
            if len(self._pool_paths) < 2:
                await asyncio.sleep(1.0)
                continue

            for slot_id in range(self.num_concurrent):
                if slot_id not in self._match_tasks or self._match_tasks[slot_id].done():
                    try:
                        model_a, model_b = self._pick_matchup()
                        task = asyncio.create_task(
                            self._run_match(slot_id, model_a, model_b)
                        )
                        self._match_tasks[slot_id] = task
                    except ValueError:
                        break

            await asyncio.sleep(0.1)

    async def _poll_checkpoints_loop(self) -> None:
        """Periodically scan for new checkpoints."""
        while True:
            await asyncio.sleep(self.poll_interval)
            added = self._refresh_pool()
            if added > 0:
                logger.info("Found %d new checkpoints", added)
