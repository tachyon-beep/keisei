# Showcase Tab: Model-vs-Model at Watchable Speed

**Date:** 2026-04-08
**Status:** Reviewed (Architecture, Systems, Python, QA, PyTorch)
**Scope:** Phase B — Model vs Model (watchable). Phase A (Human vs Model) deferred to future spec.

## Problem

The webui has two tabs (Training, League) that serve training monitoring. There is no way to watch two models play a full game at human-comprehensible speed. Students and observers want to pick favourites from the league, watch them compete, and see commentary-style information (top candidate moves, win probability, eval bar).

## Constraints

- **Zero GPU impact.** Both GPUs are fully committed to training. Showcase inference runs on CPU only. Enforced at process level via `CUDA_VISIBLE_DEVICES=""` — see [GPU Isolation](#gpu-isolation).
- **No training *process* coupling.** The showcase system must not share memory, state, or a process with the training loop. It MAY import pure utility functions from `keisei.training` (e.g., `build_model()`, `load_opponent()`) — these are stateless and have no GPU side effects. The constraint is about runtime isolation, not import boundaries.
- **Spectator-first.** Phase B is model-vs-model only. No human move input (that's Phase A).
- **League pool only.** Checkpoint selection is limited to entries currently in the live league pool (provides Elo context and competitive framing).

## Architecture: Sidecar Process + Shared DB

```
Showcase sidecar (CPU)         Training loop (GPU)
    | writes moves                 | writes metrics/games
         \                        /
           SQLite (shared DB, WAL mode)
                    | polls
             FastAPI server
                    | WebSocket
              Svelte frontend
           +--------+---------+----------+
           |Training| League  |Showcase  |
           +--------+---------+----------+
```

### Component Overview

| Component | Location | Responsibility |
|-----------|----------|----------------|
| Showcase Runner | `keisei/showcase/runner.py` | Loads checkpoints, runs games on CPU, writes to DB |
| Showcase DB Schema | `keisei/db.py` (extended) | `showcase_queue` + `showcase_games` + `showcase_moves` tables |
| Showcase API | `keisei/server/app.py` (extended) | Polls showcase tables, receives client commands, pushes via WebSocket |
| Showcase Tab | `webui/src/lib/ShowcaseView.svelte` | New tab rendering board + commentary |
| Showcase Store | `webui/src/stores/showcase.js` | Client-side state for showcase games |

### Why a Sidecar

Process isolation is a *feature*, not a workaround. If the showcase crashes, training continues unaffected. The sidecar can be stopped/restarted/`nice`d independently. It creates no CUDA context. The worst case is a lost demo game.

## Showcase Runner (Sidecar Process)

### Execution Model

The runner uses a **synchronous loop with `threading.Event`**, matching the existing `LeagueTournament._run_loop()` pattern in `keisei/training/tournament.py`. It does NOT use asyncio.

```python
# Pseudocode for the main loop
stop_event = threading.Event()
speed_event = threading.Event()  # wakes sleep early on speed change

while not stop_event.is_set():
    match = claim_next_match(db)  # atomic UPDATE...RETURNING
    if match is None:
        maybe_auto_showcase(db)
        stop_event.wait(timeout=5.0)  # poll for new matches
        continue
    run_game(match, db, stop_event, speed_event)
```

Move pacing uses `speed_event.wait(timeout=delay)` instead of `time.sleep()`, so speed changes take effect on the next move (not after the current sleep completes).

### Responsibilities

1. Poll `showcase_queue` for pending match requests (atomic claim — see [Queue Claiming](#atomic-queue-claiming))
2. Load the corresponding model checkpoints onto CPU
3. Run the game move-by-move using `SpectatorEnv` from shogi-gym
4. For each move: run inference, extract commentary data, write to DB, wait for pacing delay
5. Report game result when complete
6. On startup: clean up orphaned `in_progress` games from previous crashes (mark as `abandoned`)
7. Write a heartbeat timestamp to `showcase_heartbeat` every 10 seconds

### Match Request Flow

**On-demand (primary):** The FastAPI server writes a row to `showcase_queue`. The sidecar polls this table for pending requests using atomic claiming.

**Auto-showcase:** When the sidecar is idle and no manual matches are pending OR running:

1. Query the league for the top-2 entries by Elo
2. If they haven't played a showcase match in the last 30 minutes, queue one at Normal speed
3. Configurable: `--auto-showcase-interval 1800` (seconds), `--no-auto-showcase` to disable

Key: auto-showcase is suppressed when ANY manual match is pending or running (not just running). This prevents auto-matches from blocking the queue when students arrive.

### GPU Isolation

The runner enforces CPU-only execution at multiple levels:

```python
# 1. Environment variable — MUST be set before any torch import
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 2. Thread limits — avoid starving training data loaders
import torch
torch.set_num_threads(2)       # configurable via --cpu-threads
torch.set_num_interop_threads(1)

# 3. map_location on every load (belt-and-suspenders)
state_dict = torch.load(path, map_location="cpu", weights_only=True)

# 4. Runtime assertion after load
for name, param in model.named_parameters():
    assert param.device == torch.device("cpu"), f"Parameter {name} on {param.device}"
```

`CUDA_VISIBLE_DEVICES=""` is the hard firewall. Even if downstream code calls `torch.cuda.is_available()`, it returns `False`. No CUDA context can be created.

### Inference Details

- **Device:** CPU only (see above).
- **Model loading:** Reuse the pattern from `opponent_store.py:load_opponent()`:
  - `torch.load(path, map_location="cpu", weights_only=True)` — prevents GPU allocation during unpickle
  - `model.load_state_dict(state_dict, strict=True)`
  - `model.eval()` — **mandatory** (BatchNorm2d uses batch statistics in train mode; at batch_size=1, this produces completely wrong normalization)
  - `model.configure_amp(enabled=False)` if the model has this method (KataGo-style models)
  - Cache the two most recent models in memory (LRU keyed on `(entry_id, checkpoint_path)`)
- **Inference context:** Use `torch.inference_mode()` (not `torch.no_grad()`). Strictly safer — tensors cannot accidentally be used in gradient computation.
- **Do NOT use `torch.compile()`.** The JIT cost is not amortized for one-game-at-a-time inference.

**Model output contracts — the runner MUST handle both:**

| Architecture | Output type | Policy | Value |
|-------------|-------------|--------|-------|
| `resnet`, `mlp`, `transformer` | `tuple[Tensor, Tensor]` | `(batch, 11259)` logits | Scalar tanh in `[-1, 1]` |
| `se_resnet` (KataGo) | `KataGoOutput` | `(batch, 9, 9, 139)` logits → reshape to flat | 3-element logits → softmax → `[W, D, L]` |

For the "win probability" displayed in commentary:
- ResNet-style: `(value + 1) / 2` to map `[-1, 1]` → `[0, 1]`
- KataGo-style: `softmax(value_logits)[0]` (win probability)

**Observation dtype:** Always call `.float()` on the tensor from `torch.from_numpy(obs)`. The shogi-gym observation may return float64 or uint8; the model expects float32.

### Latency Estimates (Architecture-Dependent)

| Architecture | Approx params | CPU latency (single obs) | Notes |
|-------------|--------------|-------------------------|-------|
| `resnet` (hidden=128, layers=10) | ~5M | 10-50ms | Well within all presets |
| `se_resnet` (channels=256, blocks=40) | ~50-80M | 200-500ms | May exceed Fast preset budget |

**The implementing agent must benchmark against the actual deployed league architecture before finalizing speed presets.** If the league uses large SE-ResNet models, the Fast preset (0.5s) may need to be increased to 1.0s, or removed.

### Move Pacing

Pacing is controlled in-memory via a `threading.Event` and a speed variable. Speed changes from the server are communicated via `showcase_queue.speed` column (updated by the server, read by the sidecar on each move). No separate config table needed.

| Preset | Delay | Approx game length (100 moves) |
|--------|-------|---------------------------------|
| Slow   | 4s    | ~7 minutes                      |
| Normal | 2s    | ~3.5 minutes                    |
| Fast   | 0.5s  | ~50 seconds                     |

Default: Normal (2s). The sidecar reads speed from the `showcase_queue` row for the current match on each move iteration. The server updates this column when it receives a `change_showcase_speed` message from the client.

### Max-Ply Cutoff

Games are forcibly ended at **512 ply** (matches the Rust engine's `MAX_GAME_LENGTH`). The result is recorded as `draw` with a note in the move log. This prevents unbounded game duration — a 512-ply game at Slow speed would take ~34 minutes.

### Process Lifecycle

- Started as: `python -m keisei.showcase.runner --db-path <path> [--cpu-threads 2] [--auto-showcase-interval 1800] [--no-auto-showcase]`
- On startup: mark any `in_progress` showcase games as `abandoned` (crash recovery)
- On startup: mark any `running` queue entries as `cancelled` (crash recovery)
- Heartbeat: write current timestamp to `showcase_heartbeat` table every 10 seconds
- Graceful shutdown (SIGTERM/SIGINT): finish current move, write game result as `abandoned`, exit
- Hard crash: orphaned games cleaned up on next startup (see above)

### Checkpoint File Safety

The training loop saves checkpoints atomically (`.tmp` + `rename()`), so the runner will never see a half-written file. However, there is a TOCTOU window: an entry could be retired and its checkpoint deleted between the DB read and `torch.load()`. The runner must handle this:

```python
try:
    model = load_model_for_showcase(checkpoint_path, architecture, model_params)
except FileNotFoundError:
    logger.warning("Checkpoint %s missing (entry may have been retired)", checkpoint_path)
    mark_game_abandoned(db, game_id, reason="checkpoint_missing")
    continue
```

### SQLite Write Contention

Three processes write to the same WAL-mode database: training loop (high frequency), showcase sidecar (low frequency), and FastAPI server (very low frequency — only queue inserts from client requests). WAL mode serializes writes.

The sidecar's writes (one move row every 0.5-4s) are infrequent and unlikely to contend. However, during training bursts (tournament rounds), contention is possible. The sidecar must use a retry strategy:

```python
MAX_RETRIES = 3
RETRY_BASE_DELAY = 0.1  # seconds

for attempt in range(MAX_RETRIES):
    try:
        conn.execute(insert_sql, params)
        conn.commit()
        break
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e) and attempt < MAX_RETRIES - 1:
            delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.05)
            time.sleep(delay)
        else:
            raise
```

The `busy_timeout = 5000` in `_connect()` provides the first line of defense; this retry handles the case where even that times out.

## Database Schema Extensions

### Schema Version

Increment `SCHEMA_VERSION` from 2 to 3. Add `_migrate_v2_to_v3()` to the `_MIGRATIONS` registry. Since all showcase tables are new (no column additions to existing tables), the migration function is a no-op — the new tables are created by `CREATE TABLE IF NOT EXISTS` in `init_db()`. The migration function should be defined for completeness and to follow the established pattern:

```python
def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """v2 -> v3: Add showcase tables (created by init_db IF NOT EXISTS)."""
    pass  # New tables only; no ALTER TABLE needed.

_MIGRATIONS: dict[int, callable] = {
    2: _migrate_v1_to_v2,
    3: _migrate_v2_to_v3,
}
```

### `showcase_queue` table

```sql
CREATE TABLE IF NOT EXISTS showcase_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id_1  TEXT NOT NULL,       -- league entry ID (black)
    entry_id_2  TEXT NOT NULL,       -- league entry ID (white)
    speed       TEXT NOT NULL DEFAULT 'normal',  -- slow/normal/fast
    status      TEXT NOT NULL DEFAULT 'pending', -- pending/running/completed/cancelled
    requested_at TEXT NOT NULL,
    started_at  TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_showcase_queue_status ON showcase_queue(status);

-- Enforce at most one running match at a time
CREATE UNIQUE INDEX IF NOT EXISTS idx_showcase_queue_one_running
    ON showcase_queue(status) WHERE status = 'running';
```

### Atomic Queue Claiming

To prevent double-claiming if two sidecar instances start accidentally:

```sql
UPDATE showcase_queue
SET status = 'running', started_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
WHERE id = (
    SELECT id FROM showcase_queue
    WHERE status = 'pending'
    ORDER BY id ASC
    LIMIT 1
)
RETURNING id, entry_id_1, entry_id_2, speed;
```

If `rowcount == 0`, another process claimed it. The partial unique index on `status = 'running'` provides a DB-level invariant that at most one match runs at a time.

### `showcase_games` table

```sql
CREATE TABLE IF NOT EXISTS showcase_games (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    queue_id        INTEGER NOT NULL REFERENCES showcase_queue(id),
    entry_id_black  TEXT NOT NULL,
    entry_id_white  TEXT NOT NULL,
    elo_black       REAL,           -- snapshot at game start
    elo_white       REAL,
    name_black      TEXT,
    name_white      TEXT,
    status          TEXT NOT NULL DEFAULT 'in_progress',
        -- in_progress/black_win/white_win/draw/abandoned
    abandon_reason  TEXT,           -- null unless abandoned: 'crash_recovery'/'checkpoint_missing'/'shutdown'
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    total_ply       INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_showcase_games_status ON showcase_games(status);
```

### `showcase_moves` table

```sql
CREATE TABLE IF NOT EXISTS showcase_moves (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         INTEGER NOT NULL REFERENCES showcase_games(id),
    ply             INTEGER NOT NULL,
    action_index    INTEGER NOT NULL,
    usi_notation    TEXT NOT NULL,
    board_json      TEXT NOT NULL,       -- full board state after move (see note below)
    hands_json      TEXT NOT NULL,       -- hand pieces after move
    current_player  TEXT NOT NULL,       -- whose turn it is AFTER this move
    in_check        INTEGER NOT NULL DEFAULT 0,
    value_estimate  REAL,                -- model's win probability after this move (0-1 scale)
    top_candidates  TEXT,                -- JSON: [{"usi": "7g7f", "probability": 0.34}, ...] top-3
    move_time_ms    INTEGER,             -- actual inference time in milliseconds
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_showcase_moves_game_ply ON showcase_moves(game_id, ply);
```

**Note on `board_json` redundancy:** Storing full board state per move (~1-2KB per row, ~200KB per game) is a deliberate tradeoff for simpler client code. The alternative (reconstruct from move sequence) would require client-side Shogi logic or a server endpoint. This redundancy enables Phase C replay without architectural changes. For long-running deployments, consider a retention policy (e.g., delete completed games older than 7 days).

### `showcase_heartbeat` table

```sql
CREATE TABLE IF NOT EXISTS showcase_heartbeat (
    id              INTEGER PRIMARY KEY CHECK (id = 1),  -- singleton row
    last_heartbeat  TEXT NOT NULL,
    runner_pid      INTEGER
);
```

The sidecar upserts this row every 10 seconds. The server checks staleness to report sidecar health.

## WebSocket Protocol Extension

### Server Architecture Change

The existing WebSocket handler uses a `TaskGroup` with two coroutines (`_poll_and_push` and `_keepalive`). Add a **third coroutine**: `_receive_commands()`.

```python
async with asyncio.TaskGroup() as tg:
    tg.create_task(_poll_and_push(websocket, db_path))
    tg.create_task(_keepalive(websocket))
    tg.create_task(_receive_commands(websocket, db_path))  # NEW
```

`_receive_commands()` calls `websocket.receive_json()` in a loop, validates messages, and writes to DB. It runs concurrently with the send loop — no blocking.

**Refactoring `_poll_and_push`:** The existing function is ~150 lines handling 4 data domains (metrics, games, training state, league). Adding showcase polling would push it past 200 lines. Refactor into a dispatcher with separate coroutines per domain, each with its own poll interval and cursor:

```python
async def _poll_and_push(ws, db_path):
    async with asyncio.TaskGroup() as tg:
        tg.create_task(_poll_metrics(ws, db_path))
        tg.create_task(_poll_games(ws, db_path))
        tg.create_task(_poll_league(ws, db_path))
        tg.create_task(_poll_showcase(ws, db_path))  # NEW
```

### New message types (server → client)

**`showcase_init`** — Included in the WebSocket `init` message payload when a showcase game is in progress. Solves the cold-start problem (student opens tab mid-game):

```json
{
  "type": "init",
  "...existing fields...",
  "showcase": {
    "game": { "id": 1, "status": "in_progress", "...game fields..." },
    "moves": [ "...all moves so far..." ],
    "queue": [ "...pending/running queue entries..." ],
    "sidecar_alive": true
  }
}
```

**`showcase_update`** — Sent when new showcase moves are written (polled every 0.5s):

```json
{
  "type": "showcase_update",
  "game": {
    "id": 1,
    "status": "in_progress",
    "entry_id_black": "...",
    "entry_id_white": "...",
    "elo_black": 1523.4,
    "elo_white": 1487.2,
    "name_black": "Epoch-42-v3",
    "name_white": "Epoch-38-v1",
    "total_ply": 47
  },
  "new_moves": [
    {
      "ply": 47,
      "usi_notation": "7g7f",
      "board_json": ["..."],
      "hands_json": {"..."},
      "current_player": "white",
      "in_check": false,
      "value_estimate": 0.62,
      "top_candidates": [
        {"usi": "7g7f", "probability": 0.34},
        {"usi": "2g2f", "probability": 0.21},
        {"usi": "6i7h", "probability": 0.11}
      ]
    }
  ]
}
```

**`showcase_status`** — Sent when game starts, ends, queue changes, or periodically (every 10s):

```json
{
  "type": "showcase_status",
  "queue": [
    {"id": 1, "entry_id_1": "...", "entry_id_2": "...", "status": "running"},
    {"id": 2, "entry_id_1": "...", "entry_id_2": "...", "status": "pending"}
  ],
  "active_game_id": 1,
  "sidecar_alive": true,
  "queue_depth": 2
}
```

`sidecar_alive` is derived from the `showcase_heartbeat` table: `true` if last heartbeat is within 30 seconds.

### New message types (client → server)

This is the first bidirectional message flow in the webui. The `_receive_commands()` coroutine handles these.

**`request_showcase_match`**:

```json
{
  "type": "request_showcase_match",
  "entry_id_1": "...",
  "entry_id_2": "...",
  "speed": "normal"
}
```

**`change_showcase_speed`**:

```json
{
  "type": "change_showcase_speed",
  "speed": "fast"
}
```

**`cancel_showcase_match`**:

```json
{
  "type": "cancel_showcase_match",
  "queue_id": 1
}
```

### Input Validation Rules

The `_receive_commands()` coroutine MUST validate all incoming messages. Invalid messages receive an error response over WebSocket:

```json
{ "type": "showcase_error", "message": "Entry not found in active league pool", "request_type": "request_showcase_match" }
```

**Validation rules for `request_showcase_match`:**

| Field | Rule |
|-------|------|
| `entry_id_1` | Must exist in `league_entries` with `status = 'active'` |
| `entry_id_2` | Must exist in `league_entries` with `status = 'active'` |
| `entry_id_1` vs `entry_id_2` | Must be different |
| `speed` | Must be one of: `slow`, `normal`, `fast` |
| Queue depth | Total `pending` rows in `showcase_queue` must be < 5 |
| Rate limit | Max 1 match request per 10 seconds per WebSocket connection (tracked in-memory) |

**Validation rules for `change_showcase_speed`:**

| Field | Rule |
|-------|------|
| `speed` | Must be one of: `slow`, `normal`, `fast` |
| Active game | Must have a `running` queue entry to change speed on |

**Validation rules for `cancel_showcase_match`:**

| Field | Rule |
|-------|------|
| `queue_id` | Must exist in `showcase_queue` |
| `status` | Queue entry must be `pending` (cannot cancel a running match) |

## Frontend: Showcase Tab

### Layout

```
+------------------------------------------------------+
|  [Match Controls]                                     |
|  [ Entry A v ] vs [ Entry B v ]  [Speed: Normal v]   |
|  [Start Match]    [Cancel]                            |
+------------------------------------------------------+
|                        |                              |
|    9x9 Shogi Board     |   Commentary Panel           |
|    (Board.svelte)      |   - Current eval bar         |
|    (PieceTray.svelte)  |   - Win probability graph    |
|                        |   - Top-3 candidate moves    |
|                        |   - "Model chose X (34%)"    |
|                        |   - Move played highlight    |
+------------------------------------------------------+
|  Move Log (MoveLog.svelte)                            |
|  - Notation switching (Western/Japanese/USI)          |
|  - Probability annotations on each move               |
+------------------------------------------------------+
|  Match Queue (if multiple queued)                     |
|  Sidecar status indicator (alive/stale/down)          |
+------------------------------------------------------+
```

### Error States

The frontend must handle these states gracefully:

| State | Display |
|-------|---------|
| Sidecar not running (`sidecar_alive: false`) | "Showcase engine is offline" banner, controls disabled |
| No active game, sidecar running | "No match in progress. Start one above!" prompt |
| WebSocket disconnected mid-game | Standard reconnection indicator (reuses existing `StatusIndicator.svelte`) + stale board with "Reconnecting..." overlay |
| Queue full (5 pending) | "Start Match" button disabled, tooltip "Queue full (5 pending)" |
| Student joins mid-game | Full game state loaded from `showcase_init` — board, all prior moves, win probability history |

### Components (new)

| Component | Purpose |
|-----------|---------|
| `ShowcaseView.svelte` | Top-level tab layout, orchestrates sub-components |
| `MatchControls.svelte` | Entry selection dropdowns (from league pool), speed selector, start/cancel buttons |
| `CommentaryPanel.svelte` | Top-3 candidates, value estimate, move annotation |
| `WinProbGraph.svelte` | Win probability over time (reuse uPlot from MetricsChart) |
| `MatchQueue.svelte` | Shows pending/active matches in queue + sidecar health |
| `SidecarStatus.svelte` | Small indicator showing sidecar alive/stale/down |

### Components (reused from Training tab)

- `Board.svelte` — 9x9 board rendering (no modification needed)
- `PieceTray.svelte` — Hand pieces display (no modification needed)
- `MoveLog.svelte` — Move history with notation switching (minor extension: show probability annotations)
- `EvalBar.svelte` — Value estimate visualization (if it exists; otherwise a new thin component)

### Svelte Store: `showcase.js`

```javascript
// Core state
export const showcaseGame = writable(null);       // active game metadata
export const showcaseMoves = writable([]);         // move history with commentary
export const showcaseQueue = writable([]);         // pending/active matches
export const showcaseSpeed = writable('normal');   // current speed setting
export const sidecarAlive = writable(false);       // heartbeat-derived health

// Derived
export const showcaseBoard = derived(showcaseMoves, moves => {
    // Latest board state from most recent move
    if (moves.length === 0) return null;
    return moves[moves.length - 1];
});

export const winProbHistory = derived(showcaseMoves, moves => {
    // Array of {ply, value_estimate} for the graph
    return moves.map(m => ({ ply: m.ply, value: m.value_estimate }));
});

export const queueDepth = derived(showcaseQueue, q =>
    q.filter(e => e.status === 'pending').length
);
```

### WebSocket Message Handling

Extend `handleMessage()` in `ws.js` to process new message types:

- `showcase_update` → update `showcaseGame`, append to `showcaseMoves`
- `showcase_status` → update `showcaseQueue`, `sidecarAlive`
- `showcase_error` → display toast/notification to user
- `init` (extended) → populate `showcaseGame`, `showcaseMoves`, `showcaseQueue`, `sidecarAlive` from `data.showcase`

Add `ws.send()` capability for the first time:

```javascript
export function sendShowcaseCommand(message) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
    }
}
```

## Auto-Showcase

When no manual match is pending or running and the sidecar is idle:

1. Query the league for the top-2 entries by Elo
2. If they haven't played a showcase match in the last 30 minutes, queue one at Normal speed
3. Configurable: `--auto-showcase-interval 1800` (seconds), `--no-auto-showcase` to disable
4. If the league pool has fewer than 2 entries, auto-showcase is silently disabled

Auto-showcase is suppressed when ANY manual match is **pending or running** (not just running). This prevents auto-matches from blocking the queue when students arrive.

## Model Loading Reference

The implementing agent should follow the pattern from `opponent_store.py:load_opponent()`. Here is the complete recommended pattern for the showcase runner:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # MUST be before any torch import

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

from keisei.training.model_registry import build_model

def load_model_for_showcase(
    checkpoint_path: Path,
    architecture: str,
    model_params: dict[str, Any],
) -> nn.Module:
    """Load a model checkpoint for CPU-only showcase inference."""
    model = build_model(architecture, model_params)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()  # MANDATORY — BatchNorm2d is wrong at batch_size=1 in train mode
    if hasattr(model, "configure_amp"):
        model.configure_amp(enabled=False)  # Explicit — CPU float16 is slow/unreliable
    # Runtime assertion
    for name, param in model.named_parameters():
        assert param.device == torch.device("cpu"), f"Param {name} on {param.device}"
    return model


def run_inference(
    model: nn.Module,
    obs: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run a single forward pass. Returns (policy_probs, win_probability)."""
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()  # (1, 50, 9, 9)
    with torch.inference_mode():
        output = model(obs_tensor)
    # Handle both model contracts
    if hasattr(output, "policy_logits"):  # KataGoOutput (se_resnet)
        policy_logits = output.policy_logits.squeeze(0).reshape(-1)
        win_prob = torch.softmax(output.value_logits.squeeze(0), dim=0)[0].item()
    else:  # tuple (resnet, mlp, transformer)
        policy_logits, value_tensor = output
        policy_logits = policy_logits.squeeze(0)
        win_prob = (value_tensor.squeeze(0).item() + 1.0) / 2.0  # [-1,1] -> [0,1]
    # Mask illegal moves, compute softmax for top-K extraction
    # (action masking handled by caller using SpectatorEnv.legal_actions())
    return policy_logits.numpy(), win_prob
```

## Gap Analysis: Phase B → Phase C (Full Analysis Mode)

Phase C adds pause/rewind/step and richer visualization. The incremental work beyond Phase B:

### Additional Data (backend)

| Feature | What's needed | Effort |
|---------|---------------|--------|
| Full policy distribution | Store all 13,527 action probabilities per move (or top-20) | Low — already computed during inference, just serialize more |
| Board heatmap data | Map action probabilities back to destination squares | Medium — need `ActionMapper.decode()` to group by square |

### Additional Components (frontend)

| Component | Purpose | Effort |
|-----------|---------|--------|
| `PolicyHeatmap.svelte` | Overlay move probabilities on board squares (color intensity) | Medium — new rendering layer on Board.svelte |
| `VCRControls.svelte` | Play/pause/step-forward/step-back/speed-slider | Medium — client-side navigation through stored move history |
| `MoveExplorer.svelte` | Click a move in the log to jump to that position | Low — index into showcaseMoves array |

### Architectural changes

- **Client-side game state navigation:** Phase B streams moves in real-time. Phase C needs to also navigate backward through history. The `showcaseMoves` array already stores full board state per move, so this is a client-side index pointer — no backend changes.
- **Pause/resume:** The sidecar keeps generating moves at its own pace. The client just stops advancing its display pointer. Moves accumulate in the store and the user catches up when they unpause. No server-side pause needed.
- **Policy heatmap:** Requires `ActionMapper` inverse mapping (action index → destination square). shogi-gym's `DefaultActionMapper` has `decode_action()` in Rust but it's not currently exposed via PyO3. Would need a small PyO3 addition.

### Estimated gap: ~3-4 components, 1 small Rust/PyO3 addition, 0 architectural changes.

Phase C is a natural extension of Phase B with no architectural pivots required.

## Gap Analysis: Phase B → Phase A (Human vs Model)

Phase A adds interactive play. The incremental work beyond Phase B:

### New backend requirements

| Feature | What's needed | Effort |
|---------|---------------|--------|
| Move input validation | Server-side `SpectatorEnv.legal_actions()` check | Low — already available |
| Bidirectional WebSocket | Client sends moves, server validates and applies | Medium — Phase B introduces client→server messages for match requests, so the pattern exists |
| Human game session | New table/state for human-vs-model games (separate from showcase) | Medium |
| Inference on demand | Model responds to human move within ~1s | Low — same CPU inference as showcase |

### New frontend requirements

| Component | Purpose | Effort |
|-----------|---------|--------|
| Interactive Board | Click-to-select piece, click-to-place, promotion dialog | High — biggest new component; Board.svelte needs click handlers, legal move highlighting, drag-or-click UX |
| Promotion Dialog | Ask user whether to promote when entering promotion zone | Low |
| Game Controls | Resign, offer draw, new game | Low |

### Architectural changes

- **Bidirectional WebSocket is the main evolution.** Phase B starts this pattern (match requests). Phase A extends it with per-move commands.
- **Session management:** Human games need a concept of "whose turn is it" enforced server-side, timeouts if the human walks away, etc.

### Estimated gap: ~1 high-effort component (interactive board), ~2 medium backend features, builds on Phase B's bidirectional WebSocket.

## Testing Strategy

### Sidecar Runner

**Unit tests:**
- Mock `SpectatorEnv` and model, verify move-by-move DB writes with correct schema
- Test atomic queue claiming: simulate two concurrent claims, verify only one succeeds
- Test crash recovery: create orphaned `in_progress` games, verify startup cleanup marks them `abandoned`
- Test heartbeat writes
- Test speed change: verify `speed_event.wait(timeout)` wakes early when speed changes
- Test max-ply cutoff: verify game ends at 512 ply with `draw` result
- Test model output contract handling: both tuple and `KataGoOutput` paths

**Integration tests:**
- Run a full game with a tiny model (e.g., `mlp` with minimal params), verify complete DB state
- Test CPU-only constraint: assert `CUDA_VISIBLE_DEVICES` is empty, assert all model parameters on CPU device, assert `torch.cuda.is_available()` returns `False`
- Test checkpoint TOCTOU: delete checkpoint file after queue claim, verify graceful `abandoned` result

**SQLite concurrency tests:**
- Run sidecar writes and simulated training writes concurrently against the same WAL-mode database
- Verify no unhandled `sqlite3.OperationalError` propagates
- Verify move data is not lost or duplicated under contention
- Verify the retry strategy (exponential backoff) handles `SQLITE_BUSY`

### WebSocket Extensions

**Server-side:**
- Test `_receive_commands()` coroutine processes all three message types correctly
- Test validation: invalid entry IDs, self-match, invalid speed, queue full, rate limit exceeded
- Test `showcase_error` response messages for each validation failure
- Test `showcase_init` data included in WebSocket init payload when game is in progress
- Test `sidecar_alive` derivation from heartbeat staleness
- Test `_poll_showcase()` as an independent coroutine (after refactoring `_poll_and_push`)

**Client-to-server round-trip:**
- Test: send `request_showcase_match` → verify row appears in `showcase_queue`
- Test: send `change_showcase_speed` → verify queue row speed column updated
- Test: send `cancel_showcase_match` → verify queue row status changes to `cancelled`
- Test: send malformed JSON → verify connection is not dropped (graceful error handling)

### Frontend

- Component tests for new Svelte components (ShowcaseView, CommentaryPanel, MatchControls, etc.)
- Store tests for `showcase.js` derived state (board, winProbHistory, queueDepth)
- Error state rendering: sidecar down banner, queue full button state, reconnection overlay
- Cold-start: verify mid-game join renders full board + move history from `showcase_init`
- E2E: request a match, verify board updates appear, verify game completion renders result

### Validation (User Acceptance)

Before declaring the feature complete:
- At least 2 actual students/observers use the Showcase tab with real league data
- Verify speed presets feel right (is 2s/move comfortable? too slow? too fast?)
- Verify commentary panel information is comprehensible to non-expert observers
- Verify auto-showcase behavior is not confusing when games start with no user action
- Benchmark CPU inference latency against actual deployed model architecture

## Non-Goals (Phase B)

- Human move input (Phase A)
- Checkpoint selection outside the live league pool
- GPU inference
- Elo rating changes from showcase matches (these are exhibition only)
- Game recording/replay from disk (moves are in DB but no explicit replay UI — that's Phase C)
- Audio/sound effects
- Authentication/authorization for match requests (acceptable for student-facing tool; revisit if exposed to untrusted networks)
