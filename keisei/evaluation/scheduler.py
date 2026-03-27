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
