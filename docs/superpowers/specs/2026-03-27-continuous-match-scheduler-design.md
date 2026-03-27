# Continuous Match Scheduler

**Status**: Approved
**Date**: 2026-03-27
**Filigree**: keisei-a488fa2419

## Problem

The ladder spectator system needs a continuous loop that runs model-vs-model games, updates Elo ratings, and broadcasts live game state to the dashboard. The existing `BackgroundTournamentManager` runs a fixed set of opponents once and completes — it's not designed for continuous operation.

## Solution

A new `ContinuousMatchScheduler` class that maintains N concurrent games forever. When a game finishes, it picks a new matchup and starts another immediately. The first K games are "spectated" (paced for human viewing, state broadcast to dashboard), the rest are background (full speed, Elo updates only).

## Architecture

```
ContinuousMatchScheduler
├── OpponentPool (checkpoint discovery + selection)
├── EloRegistry (persistent ratings)
├── Game slots [0..N-1]
│   ├── Slots 0..K-1: spectated (paced, state published)
│   └── Slots K..N-1: background (full speed)
└── State publisher (atomic JSON for dashboard)
```

## Match Selection

Weighted random by Elo proximity:

```python
weight = 1 / (1 + abs(elo_a - elo_b) / 200)
```

Models with fewer than 5 games played get a 3x weight boost for fast initial rating convergence.

Selection draws two distinct models from the pool using these weights. If the pool has fewer than 2 models, the scheduler waits and polls for new checkpoints.

## Game Execution

Each game slot runs a coroutine:

1. Load two models from checkpoints (on assigned GPU device)
2. Create a `ShogiGame` instance
3. Alternate moves: `model.forward(obs)` → `game.make_move(action)`
4. For spectated games: `await asyncio.sleep(move_delay)` between moves
5. For spectated games: call `_publish_state()` after each move
6. On game end: update Elo, save registry, log result

Models are loaded fresh per game (not cached) to keep memory bounded. Each game creates its own `ShogiGame` — no shared state between slots.

## State Broadcast

The scheduler writes `.keisei_ladder/state.json` atomically (same pattern as `StreamlitManager`):

```json
{
  "schema_version": "ladder-v1",
  "timestamp": "2026-03-27T15:30:00Z",
  "matches": [
    {
      "slot": 0,
      "spectated": true,
      "match_id": "uuid",
      "model_a": {"name": "checkpoint_ts50000", "elo": 1523},
      "model_b": {"name": "checkpoint_ts100000", "elo": 1487},
      "board_state": { "board": [...], "hands": {...}, ... },
      "move_log": ["P76", "P34", ...],
      "move_count": 42,
      "current_player": "black",
      "status": "in_progress"
    }
  ],
  "leaderboard": [
    {"name": "checkpoint_ts100000", "elo": 1523, "games_played": 47, "win_rate": 0.62}
  ],
  "recent_results": [
    {
      "model_a": "checkpoint_ts50000",
      "model_b": "checkpoint_ts100000",
      "winner": "model_a",
      "elo_delta_a": 12,
      "elo_delta_b": -12,
      "move_count": 156,
      "timestamp": "2026-03-27T15:29:30Z"
    }
  ]
}
```

Only spectated matches include `board_state` and `move_log`. Background matches appear only in `recent_results` after completion.

## New File

`keisei/evaluation/scheduler.py` (~250 lines)

```python
class ContinuousMatchScheduler:
    def __init__(
        self,
        checkpoint_dir: Path,
        elo_registry_path: Path,
        device: str = "cuda",
        num_concurrent: int = 6,
        num_spectated: int = 3,
        move_delay: float = 1.5,
        poll_interval: float = 30.0,
        max_moves_per_game: int = 500,
        pool_size: int = 50,
    ): ...

    async def run(self) -> None:
        """Run the scheduler forever. Cancel to stop."""

    async def _manage_game_slots(self) -> None:
        """Keep all N game slots filled."""

    def _pick_matchup(self) -> tuple[Path, Path]:
        """Weighted random pair selection from pool."""

    async def _run_match(
        self, slot: int, model_a_path: Path, model_b_path: Path
    ) -> None:
        """Run a single game, update Elo, publish state."""

    async def _run_game_loop(
        self, game: ShogiGame, model_a, model_b, spectated: bool, slot: int
    ) -> dict:
        """Execute moves until game ends. Pace and publish if spectated."""

    def _publish_state(self) -> None:
        """Write atomic JSON state file for dashboard."""

    async def _poll_checkpoints(self) -> None:
        """Periodic directory scan for new models."""
```

## Config

New fields in `EvaluationConfig` or passed directly to the scheduler:

```python
scheduler_num_concurrent: int = 6
scheduler_num_spectated: int = 3
scheduler_move_delay: float = 1.5
scheduler_poll_interval: float = 30.0
```

These are NOT added to `ParallelConfig` — the scheduler is an evaluation feature, not a training feature.

## Dependencies

- `OpponentPool` with `scan_directory()` — done (keisei-e96f54e960)
- `EloRegistry` with JSON persistence — done (keisei-e35abcd9c4)
- `evaluate_checkpoint` with opponent support — done (keisei-c09cbac641)
- `ShogiGame` — existing, no changes needed
- Model loading — `PPOAgent.load_model()` or direct `torch.load` + `model.load_state_dict()`

## What We're NOT Doing

- No training integration in v1 — scheduler only plays existing checkpoints
- No model caching — load fresh per game (memory-bounded, simple)
- No game replay/recording — just live state broadcast
- No WebSocket/streaming — dashboard polls the JSON file on its refresh cycle
- No authentication — local dashboard only

## Lifecycle

```
CLI: python train.py ladder --checkpoint-dir models/my-run/
  │
  ├── Scan directory → populate OpponentPool
  ├── Load Elo ratings from registry
  ├── Start N game coroutines
  │
  └── Forever:
        ├── Game completes → update Elo → publish → start new game
        ├── Every 30s → scan for new checkpoints
        └── Ctrl+C → graceful shutdown (finish active games, save Elo)
```

## Testing

1. Unit: `_pick_matchup` weight distribution, state JSON schema
2. Unit: spectated vs background slot assignment
3. Integration: start scheduler with 2 fake checkpoints, verify game completes and Elo updates
4. Integration: add checkpoint mid-run, verify it enters pool and gets matched
