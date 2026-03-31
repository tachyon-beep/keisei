# Training Harness & Spectator WebUI — Design Spec

**Date:** 2026-04-01
**Status:** Draft (revised after architecture, systems, Python, QE, and neural architecture reviews)

## Purpose

A training harness for DRL Shogi self-play and a read-only spectator dashboard for watching training in real time. The system is **pedagogical** — designed to teach how RL training works, not just run it. Multiple model architectures and training algorithms will be supported, feeding into a future Elo ladder.

## Architecture Overview

Three independent processes communicating through a single SQLite file:

```
┌─────────────────┐       ┌──────────────┐       ┌─────────────────┐
│  Training Loop   │──────▶│   SQLite DB   │◀──────│  FastAPI Server  │
│  (Python + Rust) │ write │ (single file) │  read │  + WebSocket     │
└─────────────────┘       └──────────────┘       └────────┬────────┘
                                                          │ WS push
                                                 ┌────────▼────────┐
                                                 │  Svelte SPA      │
                                                 │  (browser)       │
                                                 └─────────────────┘
```

- **Training loop** — uses `VecEnv` (Rust/PyO3) for batched self-play. Calls `VecEnv.get_spectator_data()` for display-ready game snapshots. Writes snapshots + metrics to SQLite.
- **FastAPI server** — polls SQLite via `asyncio.to_thread`, pushes updates to browser clients via WebSocket.
- **Svelte dashboard** — renders game boards, move logs, piece trays, and live training charts (uPlot).

### Key properties

- **Decoupled**: Training runs without the dashboard. Dashboard connects/disconnects freely. Neither blocks the other.
- **Resumable**: Training writes periodic checkpoints (model weights + optimizer state + epoch). On restart, detects existing `training_state` in the DB and resumes from last checkpoint.
- **Durable**: Training metrics persist in SQLite. Dashboard restarts don't lose history.
- **Single dependency for IPC**: No Redis, no message queue — just one SQLite file with WAL mode for concurrent read/write.

### Rust engine requirements

The following additions to the Rust `shogi-gym` crate are required (tracked separately):

1. **`VecEnv.get_spectator_data() -> list[dict]`** — returns spectator-format dicts (board, hands, SFEN, ply, result, in_check) for all N games directly from VecEnv's internal `GameState` objects. Move history is **not** included (VecEnv doesn't track it); the Python training loop maintains per-game move history and merges it before the SQLite write. Eliminates the need for Python-side SpectatorEnv mirroring and the entire class of sync bugs that would entail.
2. **`SpectatorEnv.from_sfen(sfen: str, max_ply: int = 500)`** — static constructor, loads a position from SFEN string for recovery/debugging.
3. **`VecEnv.get_sfens() -> list[str]`** — returns current SFEN for all games, enabling resume capability.
4. **`VecEnv.mean_episode_length() -> float`** and **`VecEnv.truncation_rate() -> float`** — computed properties from running Rust-side stats, reset via `reset_stats()`.

## Configuration

All training parameters live in a TOML config file, loaded with `tomllib` (stdlib, Python 3.11+). The dashboard is a pure spectator — no runtime controls.

```toml
[training]
num_games = 8             # Concurrent games in VecEnv (1–10)
max_ply = 500             # Maximum plies per game before truncation
algorithm = "ppo"         # Training algorithm name (extensible)
checkpoint_interval = 50  # Save checkpoint every N epochs
checkpoint_dir = "checkpoints/"

[training.algorithm_params]
# Algorithm-specific — validated by dataclass at registry boundary
learning_rate = 3e-4
gamma = 0.99
clip_epsilon = 0.2
epochs_per_batch = 4
batch_size = 256

[display]
moves_per_minute = 30     # Spectator pacing (0 = headless / no snapshot writes)
db_path = "keisei.db"     # SQLite database location (resolved to absolute path at load)

[model]
display_name = "Hikaru"   # Human-facing name shown in dashboard (hides architecture)
architecture = "resnet"   # Internal architecture name — "resnet", "transformer", "mlp", etc.

[model.params]
# Architecture-specific — validated by per-architecture dataclass
hidden_size = 256
num_layers = 8
# Examples for other architectures:
#   Transformer: d_model = 256, nhead = 8, num_layers = 6
#   MLP:         hidden_sizes = [2048, 512, 256]
```

### Config loading and validation

1. Load TOML with `tomllib` (binary mode: `open("keisei.toml", "rb")`).
2. Resolve `db_path` to an absolute path immediately — prevents split-brain if training and server start from different directories.
3. Validate all fields against expected types and ranges (e.g., `1 <= num_games <= 10`, `max_ply > 0`).
4. Look up `architecture` in the model registry; validate `[model.params]` against the architecture's parameter dataclass. Typos surface immediately with clear error messages, not deep in PyTorch constructors.
5. Same validation for `algorithm` + `[training.algorithm_params]`.

### Model registry

A model factory maps `architecture` names to model classes. Each architecture declares a params dataclass:

```python
@dataclass
class ResNetParams:
    hidden_size: int   # channels per residual block
    num_layers: int    # number of residual blocks

PARAM_SCHEMAS: dict[str, type] = {"resnet": ResNetParams, "transformer": TransformerParams, ...}
MODEL_REGISTRY: dict[str, type[BaseModel]] = {"resnet": ResNetModel, ...}
```

The registry validates `[model.params]` against the dataclass at startup. Unknown keys, missing keys, and wrong types produce clear `ValueError` messages pointing at the config, not at model internals.

### Anonymous competition model

Each model gets a `display_name` (e.g., "Hikaru", "Sakura", "Tetsuo") — a human-friendly name shown in the dashboard. The architecture and params are hidden from viewers during training. This creates a "season" format:

- Viewers watch multiple named models train and compete, pick favorites.
- At season end, the architectures behind each name are revealed.
- The dashboard only ever shows `display_name` — never architecture details.

### Algorithm registry

Same pattern as models — `algorithm` maps to an algorithm class, `[training.algorithm_params]` validated against a per-algorithm params dataclass.

### `moves_per_minute` semantics

- **`> 0`**: Training runs at full speed. The DB writer throttles game snapshot writes via a simple timestamp check (`if time_since_last_write < 60 / moves_per_minute: skip`). This is in the DB writer, not the training hot path.
- **`= 0`**: Headless mode. No game snapshot writes at all. Training metrics are still written. Use this for pure training without spectating. This avoids SQLite WAL contention under high write pressure.

## SQLite Schema

Single database file. WAL mode enabled with explicit tuning:

```sql
PRAGMA journal_mode = WAL;
PRAGMA wal_autocheckpoint = 1000;  -- checkpoint every 1000 pages
PRAGMA busy_timeout = 5000;         -- 5s timeout on lock contention
```

All tables use `STRICT` typing (SQLite 3.37+) for type enforcement at the storage boundary.

### `schema_version` table

```sql
CREATE TABLE schema_version (
    version  INTEGER NOT NULL
) STRICT;
-- Initial insert: INSERT INTO schema_version VALUES (1);
```

Checked on startup by both training loop and server. Migrations applied sequentially. This is the first table checked; if the version is unknown, the process exits with a clear error.

### `metrics` table — append-only time-series

```sql
CREATE TABLE metrics (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch              INTEGER NOT NULL,
    step               INTEGER NOT NULL,
    policy_loss        REAL,
    value_loss         REAL,
    entropy            REAL,
    win_rate           REAL,
    draw_rate          REAL,
    truncation_rate    REAL,
    avg_episode_length REAL,
    gradient_norm      REAL,
    episodes_completed INTEGER,
    timestamp          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
) STRICT;
CREATE INDEX idx_metrics_epoch ON metrics(epoch);
CREATE INDEX idx_metrics_id ON metrics(id);
```

**Retention**: The training loop does not prune metrics (append-only for full history). The server handles retention at the query level — see WebSocket protocol below.

### `game_snapshots` table — overwritten per game each step

```sql
CREATE TABLE game_snapshots (
    game_id           INTEGER PRIMARY KEY,
    board_json        TEXT NOT NULL,
    hands_json        TEXT NOT NULL,
    current_player    TEXT NOT NULL,
    ply               INTEGER NOT NULL,
    is_over           INTEGER NOT NULL,
    result            TEXT NOT NULL,
    sfen              TEXT NOT NULL,
    in_check          INTEGER NOT NULL,
    move_history_json TEXT NOT NULL,
    updated_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
) STRICT;
```

**Transaction boundary**: All game snapshot writes for a single training step are wrapped in one SQLite transaction. This ensures the dashboard never reads a partially-updated set of games.

Board, hands, SFEN, and game state data comes from `VecEnv.get_spectator_data()` (Rust). Move history is tracked Python-side by the training loop (which already knows the actions taken) and merged into the dict before the SQLite write. No Python-side SpectatorEnv mirroring — eliminates an entire class of state-sync bugs.

### `training_state` table — singleton row for run-level info

```sql
CREATE TABLE training_state (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    config_json      TEXT NOT NULL,
    display_name     TEXT NOT NULL,
    model_arch       TEXT NOT NULL,
    algorithm_name   TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    current_epoch    INTEGER NOT NULL DEFAULT 0,
    current_step     INTEGER NOT NULL DEFAULT 0,
    checkpoint_path  TEXT,
    status           TEXT NOT NULL DEFAULT 'running',
    heartbeat_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
) STRICT;
```

**Resume semantics**: On startup, the training loop checks for an existing row:
- If no row exists → fresh run, `INSERT` new row.
- If row exists with `status = 'running'` and stale `heartbeat_at` (>60s old) → previous run crashed. Load checkpoint from `checkpoint_path`, resume from `current_epoch`. Log a warning.
- If row exists with `status = 'completed'` → previous run finished. Start fresh (new DB or explicit `--resume` flag).
- Uses `INSERT OR REPLACE` to update the singleton row.

**Heartbeat**: The training loop updates `heartbeat_at` every 10 seconds. The dashboard compares `heartbeat_at` to current time and shows a warning if stale (>30s), indicating the training process may have crashed.

## Training Loop

### Components

1. **Config loader** (`config.py`) — reads TOML with `tomllib`, validates all fields, resolves paths, validates model/algorithm params via registry dataclasses.
2. **VecEnv wrapper** — manages `num_games` concurrent games via the Rust `VecEnv`. Each step returns observations, rewards, legal masks, terminal info.
3. **Model** — PyTorch `nn.Module` subclass of `BaseModel` (ABC). See Model Architecture section below.
4. **Algorithm** — implements the training update (e.g., PPO collects rollouts, computes GAE, runs optimization epochs). Responsible for applying legal mask before computing action probabilities.
5. **DB writer** — after each training epoch, writes metrics row. On a `moves_per_minute` timer, calls `VecEnv.get_spectator_data()` and writes game snapshots inside a single transaction. Updates `heartbeat_at` every 10 seconds.
6. **Checkpoint writer** — every `checkpoint_interval` epochs, saves model weights, optimizer state, epoch counter, and step counter to `checkpoint_dir`. Updates `checkpoint_path` in `training_state`.

### Data flow per training step

```
VecEnv.step(actions)
  → observations, rewards, legal_masks, terminated, truncated, step_metadata
  → Model.forward(observations) → raw policy logits + value estimate
  → Algorithm applies legal_mask, samples actions
  → Store transition in rollout buffer
  → (If epoch boundary) Algorithm.update(rollout_buffer) → write metrics to SQLite
  → (If snapshot timer elapsed) VecEnv.get_spectator_data() → write game_snapshots
  → (If checkpoint interval) Save model + optimizer state
  → (Every 10s) Update heartbeat_at
```

### Metrics computation

- `win_rate` — derived from `VecEnv.episodes_completed` and win count tracked by the training loop (not from raw rewards, which are perspective-dependent).
- `draw_rate` — from `VecEnv.draw_rate` (computed in Rust).
- `truncation_rate` — from `VecEnv.truncation_rate()` (computed in Rust from running stats).
- `avg_episode_length` — from `VecEnv.mean_episode_length()` (computed in Rust from running `total_episode_ply / episodes_completed`).
- `gradient_norm` — `torch.nn.utils.clip_grad_norm_` return value, logged per update.
- `entropy` — policy entropy from the algorithm's loss computation.

All VecEnv stats are reset via `VecEnv.reset_stats()` at epoch boundaries.

### Structured logging

All training loop output uses Python `logging` module with structured format:

```
%(asctime)s %(levelname)s %(name)s %(message)s
```

No bare `print()` statements. Log levels: `INFO` for epoch summaries and checkpoints, `WARNING` for stale heartbeat / resume detection, `ERROR` for recoverable failures, `CRITICAL` for unrecoverable (NaN detected, CUDA OOM).

## Model Architecture

### Base class (`base.py`)

```python
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseModel(ABC, nn.Module):
    """Abstract base for all Keisei model architectures.

    Contract:
    - Input: observation tensor (batch, 46, 9, 9)
    - Output: (policy_logits, value) where:
        - policy_logits: (batch, 13527) — RAW, UNMASKED logits
        - value: (batch, 1) — scalar value estimate, tanh-activated
    - The algorithm (not the model) applies the legal mask before softmax.
    - Models must not depend on or consume the legal mask.
    """

    OBS_CHANNELS = 46
    BOARD_SIZE = 9
    ACTION_SPACE = 13527

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (policy_logits, value)."""
        ...
```

**Key design decisions:**
- **Raw unmasked logits**: The model outputs logits over all 13527 actions. The algorithm applies `legal_mask` before softmax. This keeps the model stateless w.r.t. legal moves and makes testing simpler.
- **`tanh` value head**: Value output is `tanh`-activated, bounding it to [-1, 1]. This matches the reward structure (-1/0/+1) and prevents unbounded value estimates from destabilizing policy gradients.
- **Legal mask in training loss**: During PPO updates, log-probabilities of taken actions must be computed on the masked distribution (softmax over legal actions only). Computing on unmasked softmax produces incorrect probability estimates — a common board-game RL bug.

### ResNet

- Residual blocks with Conv2d → BatchNorm2d → ReLU. BatchNorm is required for stable training.
- `hidden_size` = number of channels per residual block (e.g., 256).
- `num_layers` = number of residual blocks (e.g., 8).
- **Policy head**: 1×1 conv to reduce channels → flatten → Linear(reduced × 81, 13527).
- **Value head**: 1×1 conv to reduce channels → flatten → Linear → ReLU → Linear(1) → tanh.

### Transformer

- Each of the 81 board squares treated as a token with 46-dimensional features.
- Input projection: Linear(46, d_model).
- **2D positional encoding required**: Row embeddings (9) + column embeddings (9), added to token features. Not 1D sinusoidal — the spatial structure of the board must be preserved.
- Standard TransformerEncoder layers with `d_model`, `nhead`, `num_layers`.
- **Policy head**: Linear projection per token → flatten to 13527, or spatial decode.
- **Value head**: Mean-pool over 81 tokens → MLP → tanh.

### MLP

- Flattens (batch, 46, 9, 9) → (batch, 3726). This intentionally discards spatial structure.
- Staged reduction layers with LayerNorm after each Linear (required to prevent gradient pathologies on high-dimensional input).
- `hidden_sizes` config param (e.g., `[2048, 512, 256]`) rather than uniform `hidden_size`.
- **Pedagogical baseline**: The MLP is expected to underperform architectures with spatial inductive bias (ResNet, Transformer). This is intentional — it demonstrates that architecture choice matters. Dashboard viewers watching the anonymous season will see the MLP-backed player learn slower.

### LSTM — deferred

LSTM requires hidden state `(h, c)` threading through the rollout buffer across steps. The current `VecEnv` returns only the current observation with no history. Implementing LSTM properly requires:
- `forward(obs, hidden_state) -> (policy_logits, value, new_hidden_state)` — different signature from other architectures.
- Rollout buffer must store and restore hidden states per environment.
- Hidden state must be reset on episode boundaries (VecEnv auto-reset).

This is a substantial design change to the rollout buffer and algorithm interface. LSTM is deferred to a future spec iteration rather than implemented incorrectly.

### Policy head size note

The final Linear layer mapping `hidden_size → 13527` is 3.5M parameters at `hidden_size=256`. This may dominate small models. For architectures where the backbone is small (MLP, shallow Transformer), this is expected and acceptable.

## FastAPI Server

### Endpoints

- `GET /` — serves the Svelte SPA (static files).
- `GET /healthz` — returns `{"status": "ok", "db_accessible": true/false, "training_alive": true/false}`. Checks DB readability and `heartbeat_at` freshness.
- `WS /ws` — WebSocket endpoint. On connect, sends current state. Then pushes incremental updates.

### Async SQLite access

All SQLite reads are executed via `asyncio.to_thread` to avoid blocking the uvicorn event loop:

```python
async def poll_db(db_path: str, since_id: int) -> list[dict]:
    return await asyncio.to_thread(_sync_read_metrics, db_path, since_id)
```

Connections use `sqlite3.connect(db_path, check_same_thread=False)`. Read queries are wrapped in explicit read transactions (`BEGIN DEFERRED`) for snapshot isolation under WAL mode.

### WebSocket protocol

Server → Client messages:

```json
{"type": "init", "games": [...], "metrics": [...], "training_state": {"display_name": "Hikaru", "epoch": 47, "status": "running", "heartbeat_at": "...", ...}}
{"type": "game_update", "snapshots": [...]}
{"type": "metrics_update", "rows": [...]}
{"type": "training_status", "status": "running", "heartbeat_at": "...", "epoch": 48}
```

**Metrics pagination on init**: The `init` message sends only the most recent 500 metrics rows (configurable via `MAX_METRICS_IN_INIT`). The client can request older data if needed. Streaming updates append new rows as they arrive. This prevents multi-second init payloads on long training runs.

**High-water mark polling**: The server tracks `last_seen_metrics_id` per connection. Poll query: `SELECT * FROM metrics WHERE id > ? ORDER BY id LIMIT 100`. Only new rows are sent — never the full history.

**Reconnect behavior**: On reconnect, the client receives a fresh `init` message and **replaces** (not merges) its local state. This prevents duplicate data points at reconnect boundaries.

### WebSocket lifecycle management

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # startup/shutdown hooks

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(poll_and_push(websocket, db_path))
            tg.create_task(keepalive(websocket))  # ping/pong, detects dead connections
    except* WebSocketDisconnect:
        pass
```

- **`asyncio.TaskGroup`** (Python 3.11+) ensures all tasks are cancelled if any raises (e.g., disconnect). No leaked background tasks.
- **Ping/pong heartbeat** every 15 seconds via `keepalive()`. Detects dead connections (browser tab killed, network drop) within ~30s.
- **Slow consumer policy**: `websocket.send_json()` with a 5-second timeout. If the client can't keep up, the connection is closed. This prevents the server from buffering unbounded data for a stalled client.

### No client → server messages needed

The dashboard is read-only. No controls, no commands. The `keepalive` task drains any client pong frames but ignores other messages.

## Svelte Dashboard

### Layout (Layout B from brainstorming)

```
┌────────────────────────────────────────────────────────────┐
│  Keisei Training Dashboard                    ☗ Hikaru    │
│  ● Training alive (epoch 47)    [or ⚠ Training stale]    │
├───────────┬────────────────────────────────────────────────┤
│           │                                                │
│  G1  G2   │   ┌─────────────────┐  ┌───────────────────┐  │
│  [thumb]  │   │  ☖ White hand   │  │  Game Info         │  │
│  [thumb]  │   ├─────────────────┤  │  Player: Hikaru    │  │
│           │   │                 │  │  Result: In Prog   │  │
│  G3  G4   │   │   9×9 Board    │  ├───────────────────┤  │
│  [thumb]  │   │   (large,      │  │  Move Log          │  │
│  [thumb]  │   │    kanji)      │  │  # ☗Black ☖White  │  │
│           │   │                 │  │  1 7g→7f  3c→3d   │  │
│  G5  G6   │   ├─────────────────┤  │  2 2g→2f  8c→8d   │  │
│  [thumb]  │   │  ☗ Black hand   │  │  ...               │  │
│  [thumb]  │   └─────────────────┘  └───────────────────┘  │
│           │                                                │
├───────────┴────────────────────────────────────────────────┤
│  Training Metrics — Epoch 47                               │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Policy/Value Loss │  │ Win Rate          │                │
│  │ (uPlot chart)     │  │ (uPlot chart)     │                │
│  └──────────────────┘  └──────────────────┘                │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Avg Episode Len   │  │ Policy Entropy    │                │
│  │ (uPlot chart)     │  │ (uPlot chart)     │                │
│  └──────────────────┘  └──────────────────┘                │
│  💡 Annotations: "Falling entropy = more decisive agent"   │
└────────────────────────────────────────────────────────────┘
```

### Board rendering

- 9×9 grid, CSS Grid, kanji piece characters.
- Coordinate labels: columns 9–1 (top), rows a–i (right).
- Last-move highlight (green border/background on destination square).
- Piece orientation: White's pieces rendered upside-down (CSS `transform: rotate(180deg)`) per shogi convention.

### Piece trays

- Displayed above (White) and below (Black) the board.
- Each captured piece type shown as a kanji tile with a count badge when >1.
- Ordered by value: rook, bishop, gold, silver, knight, lance, pawn.

### Move log

- Two-column table: move number, Black's move, White's move.
- Scrollable, auto-scrolls to latest move.
- Current move highlighted in green.

### Training status indicator

- Green dot + "Training alive (epoch N)" when `heartbeat_at` is fresh.
- Yellow warning + "Training stale" when `heartbeat_at` is >30s old.
- Red + "Training stopped" when `status = 'completed'` or `'paused'`.

### Charts (uPlot)

Four charts in a 2×2 grid:

1. **Policy & Value Loss** — dual y-axis, lines with gradient fill. X-axis: training steps.
2. **Win Rate** — single line with 50% reference line. X-axis: episodes.
3. **Average Episode Length** — single line. Annotation: "Longer games = more strategic play".
4. **Policy Entropy** — single line. Annotation: "Falling entropy = agent becoming more decisive".

All charts: dark theme, streaming data (append new points as metrics arrive), hover for exact values. Client-side metrics store capped at 10,000 points; older data downsampled for rendering.

## Packaging and Dependencies

### pyproject.toml

```toml
[project]
name = "keisei"
version = "0.1.0"
description = "Deep RL Shogi training system with Rust core"
requires-python = ">=3.12"
dependencies = [
    "torch",
    "numpy",
    "fastapi",
    "uvicorn[standard]",
    "websockets",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "httpx", "ruff", "mypy"]

[project.scripts]
keisei-train = "keisei.training.loop:main"
keisei-serve = "keisei.server.app:main"

[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.build_meta"
```

**`shogi-gym` dependency**: The Rust engine must be built and installed separately via `maturin develop` in the `shogi-engine/` directory. This is documented in the README and enforced by a startup check in the training loop (`import shogi_gym` with a clear error message if missing).

## Project Structure

```
keisei/
├── pyproject.toml
├── keisei.toml                 # Example training config
├── shogi-engine/               # Existing Rust crates
│   ├── crates/shogi-core/
│   └── crates/shogi-gym/
├── keisei/                     # Python package
│   ├── __init__.py
│   ├── config.py               # TOML loader + dataclass validation
│   ├── db.py                   # SQLite schema, migrations, read/write helpers
│   ├── training/
│   │   ├── __init__.py
│   │   ├── loop.py             # Main training loop orchestrator
│   │   ├── checkpoint.py       # Save/load model + optimizer state
│   │   ├── model_registry.py   # Architecture name → (model class, params dataclass)
│   │   ├── algorithm_registry.py
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── base.py         # BaseModel ABC (forward signature, value tanh)
│   │       ├── resnet.py       # ResNet with BatchNorm
│   │       ├── mlp.py          # MLP baseline with LayerNorm
│   │       └── transformer.py  # Transformer with 2D positional encoding
│   └── server/
│       ├── __init__.py
│       ├── app.py              # FastAPI app, lifespan, /healthz, /ws
│       └── static/             # Built Svelte SPA
└── webui/                      # Svelte source
    ├── package.json
    ├── src/
    │   ├── App.svelte
    │   ├── lib/
    │   │   ├── Board.svelte
    │   │   ├── PieceTray.svelte
    │   │   ├── MoveLog.svelte
    │   │   ├── GameThumbnail.svelte
    │   │   ├── MetricsChart.svelte
    │   │   └── ws.js
    │   └── stores/
    │       ├── games.js
    │       └── metrics.js
    └── static/
```

## Entry Points

```bash
# Build the Rust engine (required once, or after Rust changes)
cd shogi-engine && maturin develop --release && cd ..

# Start training (writes to SQLite, no UI needed)
uv run keisei-train --config keisei.toml

# Start dashboard (reads from SQLite, serves WebUI)
uv run keisei-serve --config keisei.toml

# Both can run independently, start/stop in any order
```

## Quality Gates

These are hard gates — the system must not ship without them.

1. **Schema contract test**: Create DB, insert one row per table with realistic game state, read back, assert JSON round-trip fidelity for `board_json`, `hands_json`, `move_history_json`.
2. **VecEnv spectator data correctness**: Run 20 random steps on `VecEnv(num_envs=4)`, call `get_spectator_data()` after each step, verify `ply` and `sfen` match the game state.
3. **WebSocket protocol conformance**: Start FastAPI with pre-seeded SQLite. Connect test client → assert `init` within 1s. Write metrics row externally → assert `metrics_update` within 500ms.
4. **Config validation coverage**: Parametrized test with invalid values for every config field — `num_games=0`, `architecture="unknown"`, missing required keys, typos in `[model.params]`.
5. **GAE determinism**: Synthetic rollout with known rewards and constant value function. Assert advantage estimates match closed-form result.
6. **WAL write pressure**: Write 10,000 rows with concurrent reader holding a read transaction. Assert WAL file stays below 10MB threshold.

## Future: Ladder System

The config + SQLite design naturally extends to a ladder:

- Each training run produces a model checkpoint + its config.
- A ladder runner loads pairs of checkpoints, runs `SpectatorEnv` matches, writes results to a `ladder_results` table.
- The dashboard adds a ladder view: Elo ratings over time, head-to-head matrix, recent match results.
- The `training_state` singleton constraint will need to be relaxed for multi-run ladder scenarios.

This is explicitly **out of scope** for this spec but the architecture accommodates it.

## Non-Goals

- No interactive game controls (play moves manually, pause/resume from UI).
- No distributed training — single machine, single process.
- No authentication or multi-user support on the dashboard.
- No LSTM architecture in initial implementation (deferred — see Model Architecture section).
