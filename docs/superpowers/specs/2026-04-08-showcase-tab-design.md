# Showcase Tab: Model-vs-Model at Watchable Speed

**Date:** 2026-04-08
**Status:** Draft
**Scope:** Phase B — Model vs Model (watchable). Phase A (Human vs Model) deferred to future spec.

## Problem

The webui has two tabs (Training, League) that serve training monitoring. There is no way to watch two models play a full game at human-comprehensible speed. Students and observers want to pick favourites from the league, watch them compete, and see commentary-style information (top candidate moves, win probability, eval bar).

## Constraints

- **Zero GPU impact.** Both GPUs are fully committed to training. Showcase inference runs on CPU only.
- **No training loop coupling.** The showcase system must not import from, call into, or share memory with the training process.
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
| Showcase DB Schema | `keisei/db.py` (extended) | `showcase_games` + `showcase_moves` tables |
| Showcase API | `keisei/server/app.py` (extended) | Polls showcase tables, pushes via WebSocket |
| Showcase Tab | `webui/src/lib/ShowcaseView.svelte` | New tab rendering board + commentary |
| Showcase Store | `webui/src/stores/showcase.js` | Client-side state for showcase games |

## Showcase Runner (Sidecar Process)

### Responsibilities

1. Accept match requests (which two league entries to pit against each other)
2. Load the corresponding model checkpoints onto CPU
3. Run the game move-by-move using `SpectatorEnv` from shogi-gym
4. For each move: run inference, extract commentary data, write to DB, sleep for pacing
5. Report game result when complete

### Match Request Flow

**On-demand (primary):** The FastAPI server writes a row to a `showcase_queue` table. The sidecar polls this table for pending requests.

**Auto-showcase:** An optional timer in the sidecar picks the top-2 Elo entries from the league pool and queues a match automatically (e.g., every N minutes if no manual match is running).

### Inference Details

- **Device:** CPU only. `torch.device('cpu')` — no CUDA context created.
- **Model loading:** Load `state_dict` from the checkpoint file used by the league system. Cache the two most recent models in memory to avoid repeated disk reads.
- **Forward pass:** Single observation through the policy+value network. Expected latency: 10-50ms on CPU for a ResNet-style model. Well within the 0.5-4s move pacing budget.
- **Commentary extraction:** After the forward pass, extract:
  - Top-3 candidate moves with softmax probabilities
  - Value head output (win probability estimate)
  - Selected move and whether it was in the top-3

### Move Pacing

Server-side `asyncio.sleep()` between moves. Three speed presets:

| Preset | Delay | Approx game length (100 moves) |
|--------|-------|---------------------------------|
| Slow   | 4s    | ~7 minutes                      |
| Normal | 2s    | ~3.5 minutes                    |
| Fast   | 0.5s  | ~50 seconds                     |

Default: Normal (2s). Speed can be changed mid-game via a frontend control that writes to a `showcase_config` table (or a simple shared-state mechanism).

### Process Lifecycle

- Started as a separate Python process: `python -m keisei.showcase.runner --db-path <path>`
- Reads the same SQLite database as training and the server
- Can be stopped/restarted independently without affecting training or the dashboard
- Graceful shutdown: finishes the current move, writes game result as `abandoned` if mid-game

## Database Schema Extensions

### `showcase_queue` table

```sql
CREATE TABLE showcase_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id_1  TEXT NOT NULL,       -- league entry ID (black)
    entry_id_2  TEXT NOT NULL,       -- league entry ID (white)
    speed       TEXT NOT NULL DEFAULT 'normal',  -- slow/normal/fast
    status      TEXT NOT NULL DEFAULT 'pending', -- pending/running/completed/cancelled
    requested_at TEXT NOT NULL,
    started_at  TEXT,
    completed_at TEXT
);
```

### `showcase_games` table

```sql
CREATE TABLE showcase_games (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    queue_id        INTEGER NOT NULL REFERENCES showcase_queue(id),
    entry_id_black  TEXT NOT NULL,
    entry_id_white  TEXT NOT NULL,
    elo_black       REAL,           -- snapshot at game start
    elo_white       REAL,
    name_black      TEXT,
    name_white      TEXT,
    status          TEXT NOT NULL DEFAULT 'in_progress', -- in_progress/black_win/white_win/draw/abandoned
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    total_ply       INTEGER DEFAULT 0
);
```

### `showcase_moves` table

```sql
CREATE TABLE showcase_moves (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         INTEGER NOT NULL REFERENCES showcase_games(id),
    ply             INTEGER NOT NULL,
    action_index    INTEGER NOT NULL,
    usi_notation    TEXT NOT NULL,
    board_json      TEXT NOT NULL,       -- full board state after move
    hands_json      TEXT NOT NULL,       -- hand pieces after move
    current_player  TEXT NOT NULL,       -- whose turn it is AFTER this move
    in_check        INTEGER NOT NULL DEFAULT 0,
    value_estimate  REAL,                -- model's win probability after this move
    top_candidates  TEXT,                -- JSON: [{action, usi, probability}, ...] top-3
    move_time_ms    INTEGER,             -- actual inference time
    created_at      TEXT NOT NULL
);
```

### Index

```sql
CREATE INDEX idx_showcase_moves_game_ply ON showcase_moves(game_id, ply);
CREATE INDEX idx_showcase_games_status ON showcase_games(status);
CREATE INDEX idx_showcase_queue_status ON showcase_queue(status);
```

## WebSocket Protocol Extension

### New message types (server → client)

**`showcase_update`** — Sent when a new showcase move is written (polled every 0.5s):

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
      "board_json": [...],
      "hands_json": {...},
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

**`showcase_status`** — Sent when game starts, ends, or queue changes:

```json
{
  "type": "showcase_status",
  "queue": [
    {"id": 1, "entry_id_1": "...", "entry_id_2": "...", "status": "running"},
    {"id": 2, "entry_id_1": "...", "entry_id_2": "...", "status": "pending"}
  ],
  "active_game_id": 1
}
```

### New message types (client → server)

This is the first bidirectional message flow in the webui. The server needs a new handler for incoming WebSocket messages.

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

The server validates these and writes to the `showcase_queue` / `showcase_config` tables. The sidecar picks them up on its next poll.

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
+------------------------------------------------------+
```

### Components (new)

| Component | Purpose |
|-----------|---------|
| `ShowcaseView.svelte` | Top-level tab layout, orchestrates sub-components |
| `MatchControls.svelte` | Entry selection dropdowns, speed selector, start/cancel buttons |
| `CommentaryPanel.svelte` | Top-3 candidates, value estimate, move annotation |
| `WinProbGraph.svelte` | Win probability over time (reuse uPlot from MetricsChart) |
| `MatchQueue.svelte` | Shows pending/active matches in queue |

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
```

## Auto-Showcase

When no manual match is requested and the sidecar is idle:

1. Query the league for the top-2 entries by Elo
2. If they haven't played a showcase match in the last 30 minutes, queue one at Normal speed
3. Configurable: `--auto-showcase-interval 1800` (seconds), `--no-auto-showcase` to disable

This ensures the Showcase tab always has something to show, even if no one is actively requesting matches.

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

- Unit tests: mock `SpectatorEnv` and model, verify move-by-move DB writes
- Integration test: run a full game with a tiny model, verify DB state
- Test CPU-only constraint: assert no CUDA tensors are created

### WebSocket Extensions

- Test new message types (showcase_update, showcase_status)
- Test client→server messages (request_showcase_match, change_speed, cancel)
- Test polling of showcase tables

### Frontend

- Component tests for new Svelte components (ShowcaseView, CommentaryPanel, etc.)
- Store tests for showcase.js derived state
- E2E: request a match, verify board updates appear

## Non-Goals (Phase B)

- Human move input (Phase A)
- Checkpoint selection outside the live league pool
- GPU inference
- Elo rating changes from showcase matches (these are exhibition only)
- Game recording/replay from disk (moves are in DB but no explicit replay UI yet — that's Phase C)
- Audio/sound effects
