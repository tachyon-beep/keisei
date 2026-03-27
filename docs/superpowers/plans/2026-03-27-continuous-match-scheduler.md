# ContinuousMatchScheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a continuous match scheduler that runs model-vs-model games forever, updates Elo ratings, and broadcasts live game state for the spectator dashboard.

**Architecture:** A single async class `ContinuousMatchScheduler` maintains N concurrent game coroutines. When a game finishes, it picks a new matchup (weighted random by Elo proximity), runs the game, updates ratings, and publishes state to an atomic JSON file. The first K games are "spectated" (paced with delays, board state broadcast); the rest run at full speed for Elo grinding.

**Tech Stack:** Python 3.13, asyncio, PyTorch, existing `ShogiGame`, `PolicyOutputMapper`, `load_evaluation_agent`, `OpponentPool`, `EloRegistry`

**Spec:** `docs/superpowers/specs/2026-03-27-continuous-match-scheduler-design.md`
**Filigree:** keisei-a488fa2419

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `keisei/evaluation/scheduler.py` | `ContinuousMatchScheduler` class (~250 lines) |
| `tests/unit/test_scheduler.py` | Unit tests for match selection, state publishing, slot management |

### Modified Files

| File | Change |
|------|--------|
| `keisei/evaluation/__init__.py` | Export `ContinuousMatchScheduler` |
| `keisei/utils/agent_loading.py` | Fix `_load_model_from_checkpoint` to use `model_factory` instead of hardcoded `ActorCritic` |
| `keisei/evaluation/opponents/elo_registry.py` | Make `save()` atomic with tempfile + os.replace (W2 fix) |

### Reference Files (read-only — use their APIs)

| File | API Used |
|------|----------|
| `keisei/evaluation/opponents/opponent_pool.py` | `OpponentPool`, `scan_directory()`, `get_all()` |
| `keisei/evaluation/opponents/elo_registry.py` | `EloRegistry`, `get_rating()`, `update_ratings()`, `save()` |
| `keisei/shogi/shogi_game.py` | `ShogiGame`, `reset()`, `get_legal_moves()`, `make_move()`, `get_observation()`, `to_sfen()`, `current_player` |
| `keisei/utils/utils.py` | `PolicyOutputMapper` |
| `keisei/webui/state_snapshot.py` | `write_snapshot_atomic()` |
| `keisei/training/models/__init__.py` | `model_factory()` |
| `keisei/core/ppo_agent.py` | `PPOAgent` (via `load_evaluation_agent`) |

---

## Task 1: Match Selection Logic

**Files:**
- Create: `keisei/evaluation/scheduler.py`
- Test: `tests/unit/test_scheduler.py` (new)

Build the `_pick_matchup` method and the class skeleton.

- [ ] **Step 1: Write failing tests for match selection**

Create `tests/unit/test_scheduler.py`:

```python
"""Unit tests for ContinuousMatchScheduler."""

import pytest
from pathlib import Path
from collections import Counter

pytestmark = pytest.mark.unit


class TestMatchSelection:
    """Weighted random matchup selection by Elo proximity."""

    def _make_scheduler(self, checkpoints, ratings=None):
        """Create a scheduler with mock pool and ratings."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler

        scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
        scheduler._pool_paths = [Path(c) for c in checkpoints]
        scheduler._games_played = Counter()
        scheduler._elo_registry = None

        # Mock ratings
        scheduler._get_rating = lambda name: (ratings or {}).get(name, 1500.0)
        return scheduler

    def test_pick_returns_two_distinct_paths(self):
        scheduler = self._make_scheduler(["a.pth", "b.pth", "c.pth"])
        a, b = scheduler._pick_matchup()
        assert a != b
        assert a in scheduler._pool_paths
        assert b in scheduler._pool_paths

    def test_pick_raises_with_fewer_than_two_models(self):
        scheduler = self._make_scheduler(["a.pth"])
        with pytest.raises(ValueError, match="at least 2"):
            scheduler._pick_matchup()

    def test_pick_favors_close_ratings(self):
        """Models with close Elo should be matched more often."""
        ratings = {"a.pth": 1500.0, "b.pth": 1510.0, "c.pth": 1900.0}
        scheduler = self._make_scheduler(["a.pth", "b.pth", "c.pth"], ratings)

        pair_counts = Counter()
        for _ in range(1000):
            a, b = scheduler._pick_matchup()
            pair = tuple(sorted([a.name, b.name]))
            pair_counts[pair] += 1

        # a vs b (10 Elo apart) should be picked much more than a vs c (400 apart)
        ab_count = pair_counts.get(("a.pth", "b.pth"), 0)
        ac_count = pair_counts.get(("a.pth", "c.pth"), 0)
        assert ab_count > ac_count * 2, f"Close-rated pair {ab_count} should dominate distant {ac_count}"

    def test_new_models_get_weight_boost(self):
        """Models with <5 games get 3x weight boost."""
        ratings = {"a.pth": 1500.0, "b.pth": 1500.0, "c.pth": 1500.0}
        scheduler = self._make_scheduler(["a.pth", "b.pth", "c.pth"], ratings)
        # a and b have many games, c is new
        scheduler._games_played["a.pth"] = 20
        scheduler._games_played["b.pth"] = 20
        scheduler._games_played["c.pth"] = 1

        pair_counts = Counter()
        for _ in range(1000):
            a, b = scheduler._pick_matchup()
            pair = tuple(sorted([a.name, b.name]))
            pair_counts[pair] += 1

        # c should appear in more pairs than expected (boosted)
        c_appearances = sum(v for k, v in pair_counts.items() if "c.pth" in k)
        assert c_appearances > 400, f"New model should appear often, got {c_appearances}/1000"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_scheduler.py -v`
Expected: FAIL — `keisei.evaluation.scheduler` module not found

- [ ] **Step 3: Implement the skeleton and _pick_matchup**

Create `keisei/evaluation/scheduler.py`:

```python
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
    """Typed result from a completed game (W10 fix)."""

    done: bool
    winner: Optional[int]  # 0=Black, 1=White, None=Draw
    move_count: int
    reason: str


class SchedulerConfig(BaseModel):
    """Configuration for ContinuousMatchScheduler (W7 fix).

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
        self._match_tasks: Dict[int, asyncio.Task] = {}  # W8: managed by _manage_game_slots, read by run()
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_scheduler.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/evaluation/scheduler.py tests/unit/test_scheduler.py
git commit -m "feat(scheduler): add ContinuousMatchScheduler skeleton with weighted matchup selection"
```

---

## Task 2: Game Execution

**Files:**
- Modify: `keisei/evaluation/scheduler.py`
- Test: `tests/unit/test_scheduler.py`

Add `_run_match` and `_run_game_loop` methods. Fix `_load_model_from_checkpoint` in `agent_loading.py` to use `model_factory`.

- [ ] **Step 1: Write failing test for game execution**

Add to `tests/unit/test_scheduler.py`:

```python
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import numpy as np
import torch


class TestGameExecution:
    """Scheduler runs games between two models."""

    @pytest.mark.asyncio
    async def test_run_game_loop_returns_result(self):
        """Game loop runs until done and returns winner info."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler
        from keisei.shogi.shogi_core_definitions import Color

        scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
        scheduler.max_moves_per_game = 10
        scheduler.move_delay = 0
        scheduler.num_spectated = 0
        scheduler._active_matches = {}
        scheduler._publish_state = MagicMock()
        scheduler._policy_mapper = MagicMock()
        scheduler._policy_mapper.get_legal_mask.return_value = torch.zeros(13527)

        # Mock game — alternates current_player like real ShogiGame
        game = MagicMock()
        game.get_legal_moves.return_value = [(0, 0, 1, 1, False)]
        game.get_observation.return_value = np.zeros((46, 9, 9))
        game.to_sfen.return_value = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

        # Alternate current_player between BLACK and WHITE each move
        player_cycle = [Color.BLACK, Color.WHITE] * 5
        type(game).current_player = property(
            lambda self, _cycle=iter(player_cycle): next(_cycle)
        )

        # After 10 moves, game ends
        obs = np.zeros((46, 9, 9))
        game.make_move.side_effect = [
            (obs, 0.0, False, {}),
        ] * 9 + [(obs, 0.0, True, {"winner": "black"})]

        # Mock agents
        agent_a = MagicMock()
        agent_a.select_action.return_value = ((0, 0, 1, 1, False), 0, 0.0, 0.0)
        agent_b = MagicMock()
        agent_b.select_action.return_value = ((0, 0, 1, 1, False), 0, 0.0, 0.0)

        result = await scheduler._run_game_loop(
            game, agent_a, agent_b, spectated=False, slot=0
        )
        assert result.done is True
        assert result.winner == 0  # Black wins

    @pytest.mark.asyncio
    async def test_run_game_loop_draw_on_max_moves(self):
        """Game returns draw when max_moves reached without winner (W6 fix)."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler, MatchResult
        from keisei.shogi.shogi_core_definitions import Color

        scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
        scheduler.max_moves_per_game = 3  # Very short game
        scheduler.move_delay = 0
        scheduler.num_spectated = 0
        scheduler._active_matches = {}
        scheduler._publish_state = MagicMock()
        scheduler._policy_mapper = MagicMock()
        scheduler._policy_mapper.get_legal_mask.return_value = torch.zeros(13527)

        game = MagicMock()
        game.get_legal_moves.return_value = [(0, 0, 1, 1, False)]
        game.get_observation.return_value = np.zeros((46, 9, 9))
        game.to_sfen.return_value = "startpos"

        player_cycle = [Color.BLACK, Color.WHITE] * 5
        type(game).current_player = property(
            lambda self, _cycle=iter(player_cycle): next(_cycle)
        )

        # Never return done=True — game should hit max_moves
        obs = np.zeros((46, 9, 9))
        game.make_move.return_value = (obs, 0.0, False, {})

        agent_a = MagicMock()
        agent_a.select_action.return_value = ((0, 0, 1, 1, False), 0, 0.0, 0.0)
        agent_b = MagicMock()
        agent_b.select_action.return_value = ((0, 0, 1, 1, False), 0, 0.0, 0.0)

        result = await scheduler._run_game_loop(
            game, agent_a, agent_b, spectated=False, slot=0
        )
        assert result.done is True
        assert result.winner is None  # Draw
        assert result.move_count == 3
        assert result.reason == "max_moves"

    @pytest.mark.asyncio
    async def test_run_match_handles_missing_checkpoint(self):
        """_run_match logs error and doesn't crash on missing file (W4 fix)."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler, SchedulerConfig
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SchedulerConfig(
                checkpoint_dir=Path(tmpdir),
                elo_registry_path=Path(tmpdir) / "elo.json",
                device="cpu",
                num_concurrent=1,
                state_path=Path(tmpdir) / "state.json",
            )
            scheduler = ContinuousMatchScheduler(config)

            # Call with non-existent checkpoint paths
            await scheduler._run_match(
                0, Path("/nonexistent/a.pth"), Path("/nonexistent/b.pth")
            )
            # Should not raise — error is caught and logged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_scheduler.py::TestGameExecution -v`
Expected: FAIL — `_run_game_loop` doesn't exist

- [ ] **Step 3a: Fix `_load_model_from_checkpoint` in agent_loading.py (B2 prerequisite)**

Modify `keisei/utils/agent_loading.py` to use `model_factory` instead of hardcoded `ActorCritic`:

```python
# Replace _load_model_from_checkpoint (lines 159-176) with:

def _load_model_from_checkpoint(
    checkpoint_path: str,
    device_str: str,
    policy_mapper,
    input_channels: int,
    model_type: str = "resnet",
    tower_depth: int = 9,
    tower_width: int = 256,
    se_ratio: Optional[float] = 0.25,
) -> Any:
    """Create a model using model_factory and return it with the device."""
    import torch  # pylint: disable=import-outside-toplevel

    from keisei.training.models import (  # pylint: disable=import-outside-toplevel
        model_factory,
    )

    device = torch.device(device_str)
    num_actions = policy_mapper.get_total_actions()
    obs_shape = (input_channels, 9, 9)

    temp_model = model_factory(
        model_type=model_type,
        obs_shape=obs_shape,
        num_actions=num_actions,
        tower_depth=tower_depth,
        tower_width=tower_width,
        se_ratio=se_ratio,
    ).to(device)
    return temp_model, device
```

Also update `load_evaluation_agent` to accept and forward the architecture params:

```python
def load_evaluation_agent(
    checkpoint_path: str,
    device_str: str,
    policy_mapper,
    input_channels: int,
    input_features: Optional[str] = "core46",
    model_type: str = "resnet",
    tower_depth: int = 9,
    tower_width: int = 256,
    se_ratio: Optional[float] = 0.25,
) -> Any:
    # ... (existing validation unchanged) ...

    config = _build_evaluation_config(
        device_str, policy_mapper, input_channels, input_features
    )

    temp_model, device = _load_model_from_checkpoint(
        checkpoint_path, device_str, policy_mapper, input_channels,
        model_type=model_type,
        tower_depth=tower_depth,
        tower_width=tower_width,
        se_ratio=se_ratio,
    )

    agent = _create_ppo_agent(temp_model, config, device, checkpoint_path)
    # ... (rest unchanged) ...
```

Existing callers that pass no architecture params get the defaults (`resnet`, depth=9, width=256, se_ratio=0.25) — backward compatible.

- [ ] **Step 3b: Make EloRegistry.save() atomic (W2 fix)**

Modify `keisei/evaluation/opponents/elo_registry.py` — replace the non-atomic `save()` with tempfile + os.replace:

```python
    def save(self) -> None:
        """Persist ratings to disk atomically."""
        import os
        import tempfile

        data = {
            "ratings": self.ratings,
            "metadata": {
                "initial_rating": self.initial_rating,
                "k_factor": self.k_factor,
            },
        }
        dir_path = str(self.file_path.parent)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, str(self.file_path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
```

This prevents a crash mid-write from corrupting the ratings file (previously a truncate-and-write that could leave empty/partial JSON).

- [ ] **Step 3c: Implement _run_game_loop and _run_match**

Add to `keisei/evaluation/scheduler.py`:

```python
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
            move_log.append(str(selected_move))  # W3 fix: populate move_log

            if spectated:
                # Update active match state for dashboard
                self._active_matches[slot].update({
                    "sfen": game.to_sfen(),
                    "move_count": move_count,
                    "move_log": move_log[-20:],  # Last 20 moves
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
            winner=None,  # Draw
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

            # load_evaluation_agent now uses model_factory internally (B2 fix —
            # see Task 2 Step 3a for the agent_loading.py patch).
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

            # Set initial match state
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

            # Update Elo — EloRegistry expects "agent_win"/"opponent_win"/"draw"
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

            # Track games played
            self._games_played[name_a] += 1
            self._games_played[name_b] += 1

            # Record result
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
            self._recent_results = self._recent_results[-50:]  # Keep last 50

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
            # Protected publish — exception here must not mask the original
            # error from the try block (B4 fix).
            try:
                self._publish_state()
            except Exception:
                logger.exception("Slot %d: failed to publish state", slot)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_scheduler.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/evaluation/scheduler.py keisei/utils/agent_loading.py keisei/evaluation/opponents/elo_registry.py tests/unit/test_scheduler.py
git commit -m "feat(scheduler): add game execution with spectated pacing and Elo updates

Also fixes load_evaluation_agent to use model_factory for correct
architecture detection (ResNet/CNN) instead of hardcoded ActorCritic.
Also makes EloRegistry.save() atomic to prevent data loss on crash."
```

---

## Task 3: State Publishing

**Files:**
- Modify: `keisei/evaluation/scheduler.py`
- Test: `tests/unit/test_scheduler.py`

- [ ] **Step 1: Write failing test for state publishing**

Add to `tests/unit/test_scheduler.py`:

```python
import json
import tempfile


class TestStatePublishing:
    """Scheduler writes atomic JSON state for dashboard."""

    def test_publish_state_creates_json(self):
        from keisei.evaluation.scheduler import ContinuousMatchScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
            scheduler._state_path = Path(tmpdir) / "state.json"
            scheduler._active_matches = {
                0: {"match_id": "test", "status": "in_progress", "spectated": True,
                    "sfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
                    "model_a": {}, "model_b": {}, "move_count": 0}
            }
            scheduler._recent_results = [{"winner": "model_a"}]
            scheduler._elo_registry = MagicMock()
            scheduler._elo_registry.ratings = {"model_a": 1520.0, "model_b": 1480.0}
            scheduler._games_played = Counter({"model_a": 10, "model_b": 5})

            scheduler._publish_state()

            assert scheduler._state_path.exists()
            data = json.loads(scheduler._state_path.read_text())
            assert "matches" in data
            assert "leaderboard" in data
            assert "recent_results" in data
            assert data["schema_version"] == "ladder-v1"

    def test_leaderboard_sorted_by_elo(self):
        from keisei.evaluation.scheduler import ContinuousMatchScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = ContinuousMatchScheduler.__new__(ContinuousMatchScheduler)
            scheduler._state_path = Path(tmpdir) / "state.json"
            scheduler._active_matches = {}
            scheduler._recent_results = []
            scheduler._elo_registry = MagicMock()
            scheduler._elo_registry.ratings = {
                "weak": 1400.0, "strong": 1600.0, "mid": 1500.0
            }
            scheduler._games_played = Counter({"weak": 5, "strong": 5, "mid": 5})

            scheduler._publish_state()

            data = json.loads(scheduler._state_path.read_text())
            elos = [entry["elo"] for entry in data["leaderboard"]]
            assert elos == sorted(elos, reverse=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_scheduler.py::TestStatePublishing -v`
Expected: FAIL — `_publish_state` not fully implemented

- [ ] **Step 3: Implement _publish_state**

Add to `ContinuousMatchScheduler`:

```python
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
        # uses tempfile.mkstemp + os.replace (replaces inline reimplementation).
        from keisei.webui.state_snapshot import write_snapshot_atomic

        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        write_snapshot_atomic(state, self._state_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_scheduler.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/evaluation/scheduler.py tests/unit/test_scheduler.py
git commit -m "feat(scheduler): add atomic JSON state publishing for dashboard"
```

---

## Task 4: Main Run Loop

**Files:**
- Modify: `keisei/evaluation/scheduler.py`
- Test: `tests/unit/test_scheduler.py`

Wire everything together: the `run()` coroutine that manages game slots and polls for new checkpoints.

- [ ] **Step 1: Write failing test for the run loop**

Add to `tests/unit/test_scheduler.py`:

```python
class TestRunLoop:
    """Scheduler run loop manages concurrent game slots."""

    @pytest.mark.asyncio
    async def test_run_starts_and_can_be_cancelled(self):
        """Scheduler starts, runs briefly, and stops on cancellation."""
        from keisei.evaluation.scheduler import ContinuousMatchScheduler
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            ckpt_dir.mkdir()
            # Create 2 fake checkpoints
            (ckpt_dir / "checkpoint_ts1000.pth").write_bytes(b"fake")
            (ckpt_dir / "checkpoint_ts2000.pth").write_bytes(b"fake")

            elo_path = Path(tmpdir) / "elo.json"
            state_path = Path(tmpdir) / "state.json"

            from keisei.evaluation.scheduler import SchedulerConfig

            config = SchedulerConfig(
                checkpoint_dir=ckpt_dir,
                elo_registry_path=elo_path,
                device="cpu",
                num_concurrent=1,
                num_spectated=0,
                move_delay=0,
                poll_interval=1.0,
                max_moves_per_game=5,
                state_path=state_path,
            )
            scheduler = ContinuousMatchScheduler(config)

            # Mock _run_match to avoid loading real models from fake files.
            # Track call count to verify slot management works.
            match_count = 0

            async def fake_run_match(slot, a, b):
                nonlocal match_count
                match_count += 1
                await asyncio.sleep(0.05)

            scheduler._run_match = fake_run_match

            # run() should not raise when cancelled
            task = asyncio.create_task(scheduler.run())
            await asyncio.sleep(0.5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass  # Expected

            # Verify slot management actually dispatched matches
            assert match_count > 0, "Scheduler should have dispatched at least one match"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_scheduler.py::TestRunLoop -v`
Expected: FAIL — `run()` not implemented

- [ ] **Step 3: Implement run() and _manage_game_slots**

Add to `ContinuousMatchScheduler`:

```python
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
            # their current Elo save before we exit.
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
            # Wait until we have enough models
            if len(self._pool_paths) < 2:
                await asyncio.sleep(1.0)
                continue

            # Fill empty slots
            for slot_id in range(self.num_concurrent):
                if slot_id not in self._match_tasks or self._match_tasks[slot_id].done():
                    try:
                        model_a, model_b = self._pick_matchup()
                        task = asyncio.create_task(
                            self._run_match(slot_id, model_a, model_b)
                        )
                        self._match_tasks[slot_id] = task
                    except ValueError:
                        break  # Not enough models

            # Brief sleep to avoid busy-loop
            await asyncio.sleep(0.1)

    async def _poll_checkpoints_loop(self) -> None:
        """Periodically scan for new checkpoints."""
        while True:
            await asyncio.sleep(self.poll_interval)
            added = self._refresh_pool()
            if added > 0:
                logger.info("Found %d new checkpoints", added)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_scheduler.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite for regressions**

Run: `uv run pytest tests/unit/test_scheduler.py tests/unit/test_ladder_elo_integration.py tests/unit/test_opponent_pool_scanning.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add keisei/evaluation/scheduler.py tests/unit/test_scheduler.py
git commit -m "feat(scheduler): add main run loop with concurrent game slots and checkpoint polling"
```

---

## Task 5: Export and Integration Smoke Test

**Files:**
- Modify: `keisei/evaluation/__init__.py`
- Test: `tests/unit/test_scheduler.py`

- [ ] **Step 1: Add export**

In `keisei/evaluation/__init__.py`, add:

```python
from .scheduler import ContinuousMatchScheduler, MatchResult, SchedulerConfig
```

(Check existing exports first — add alongside them.)

- [ ] **Step 2: Write import test**

Add to `tests/unit/test_scheduler.py`:

```python
class TestSchedulerExport:
    """Scheduler is importable from the evaluation package."""

    def test_importable_from_package(self):
        from keisei.evaluation import ContinuousMatchScheduler
        assert ContinuousMatchScheduler is not None
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/unit/test_scheduler.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add keisei/evaluation/__init__.py keisei/evaluation/scheduler.py tests/unit/test_scheduler.py
git commit -m "feat(scheduler): export ContinuousMatchScheduler from evaluation package"
```
