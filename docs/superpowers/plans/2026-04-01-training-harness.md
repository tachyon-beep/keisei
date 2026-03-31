# Training Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Python training harness for DRL Shogi self-play — config loading, SQLite persistence, model architectures (ResNet, Transformer, MLP), PPO training algorithm, checkpointing, and a training loop that writes game snapshots + metrics to SQLite.

**Architecture:** Three-layer stack: config → db → training components (models, algorithm, loop). The training loop uses VecEnv (Rust/PyO3) for batched self-play and writes to SQLite for the spectator dashboard (built separately in Plan 2). All Python, all `uv run`.

**Tech Stack:** Python 3.12+, PyTorch, SQLite (WAL mode), tomllib (stdlib), shogi-gym (Rust/PyO3)

**Spec:** `docs/superpowers/specs/2026-04-01-training-harness-webui-design.md`

**Depends on:** Rust API extensions from `docs/superpowers/specs/2026-04-01-vecenv-spectator-api-design.md` — specifically `VecEnv.get_spectator_data()`, `VecEnv.get_sfens()`, `VecEnv.mean_episode_length()`, `VecEnv.truncation_rate()`. Tasks that need these are marked with [RUST-DEP]. They can be stubbed/mocked until the Rust side lands.

---

## File Map

| File | Responsibility | Created/Modified |
|------|---------------|-----------------|
| `pyproject.toml` | Dependencies + entry points | Modify |
| `keisei.toml` | Example training config | Create |
| `keisei/__init__.py` | Package marker | Create |
| `keisei/config.py` | TOML loading + dataclass validation | Create |
| `keisei/db.py` | SQLite schema, migrations, read/write | Create |
| `keisei/training/__init__.py` | Package marker | Create |
| `keisei/training/models/__init__.py` | Package marker | Create |
| `keisei/training/models/base.py` | BaseModel ABC | Create |
| `keisei/training/models/resnet.py` | ResNet architecture | Create |
| `keisei/training/models/mlp.py` | MLP baseline | Create |
| `keisei/training/models/transformer.py` | Transformer architecture | Create |
| `keisei/training/model_registry.py` | Name → (class, params dataclass) | Create |
| `keisei/training/algorithm_registry.py` | Name → (class, params dataclass) | Create |
| `keisei/training/ppo.py` | PPO algorithm (rollout buffer, GAE, update) | Create |
| `keisei/training/checkpoint.py` | Save/load model + optimizer + epoch | Create |
| `keisei/training/loop.py` | Main training loop orchestrator | Create |
| `tests/test_config.py` | Config loading + validation tests | Create |
| `tests/test_db.py` | Schema + read/write round-trip tests | Create |
| `tests/test_models.py` | Model forward pass shape tests | Create |
| `tests/test_ppo.py` | GAE + PPO update tests | Create |
| `tests/test_checkpoint.py` | Save/load round-trip tests | Create |
| `tests/test_loop.py` | Integration: training loop smoke test | Create |

---

### Task 1: Project Scaffolding and Dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `keisei/__init__.py`
- Create: `keisei/training/__init__.py`
- Create: `keisei/training/models/__init__.py`

- [ ] **Step 1: Update pyproject.toml with dependencies and entry points**

```toml
[project]
name = "keisei"
version = "0.1.0"
description = "Deep RL Shogi training system with Rust core"
license = {text = "MIT"}
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

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.mypy]
python_version = "3.12"
strict = true
```

- [ ] **Step 2: Create package __init__.py files**

`keisei/__init__.py`:
```python
"""Keisei — Deep RL training system for Shogi."""
```

`keisei/training/__init__.py`:
```python
"""Training components: models, algorithms, and loop orchestration."""
```

`keisei/training/models/__init__.py`:
```python
"""Neural network architectures for Shogi policy+value networks."""
```

- [ ] **Step 3: Install dependencies**

Run: `uv pip install -e ".[dev]"`
Expected: Successful install, all deps resolved.

- [ ] **Step 4: Verify import works**

Run: `uv run python -c "import keisei; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml keisei/__init__.py keisei/training/__init__.py keisei/training/models/__init__.py
git commit -m "scaffold: keisei Python package with deps and entry points"
```

---

### Task 2: Config Loading and Validation

**Files:**
- Create: `keisei/config.py`
- Create: `keisei.toml`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the example config file**

`keisei.toml`:
```toml
[training]
num_games = 8
max_ply = 500
algorithm = "ppo"
checkpoint_interval = 50
checkpoint_dir = "checkpoints/"

[training.algorithm_params]
learning_rate = 3e-4
gamma = 0.99
clip_epsilon = 0.2
epochs_per_batch = 4
batch_size = 256

[display]
moves_per_minute = 30
db_path = "keisei.db"

[model]
display_name = "Hikaru"
architecture = "resnet"

[model.params]
hidden_size = 256
num_layers = 8
```

- [ ] **Step 2: Write failing tests for config loading**

`tests/test_config.py`:
```python
import pytest
from pathlib import Path
from keisei.config import load_config, AppConfig, TrainingConfig, DisplayConfig, ModelConfig


@pytest.fixture
def sample_toml(tmp_path: Path) -> Path:
    config_file = tmp_path / "test.toml"
    config_file.write_text("""\
[training]
num_games = 4
max_ply = 300
algorithm = "ppo"
checkpoint_interval = 10
checkpoint_dir = "ckpt/"

[training.algorithm_params]
learning_rate = 1e-3
gamma = 0.99
clip_epsilon = 0.2
epochs_per_batch = 4
batch_size = 128

[display]
moves_per_minute = 60
db_path = "test.db"

[model]
display_name = "TestBot"
architecture = "resnet"

[model.params]
hidden_size = 64
num_layers = 4
""")
    return config_file


def test_load_config_basic(sample_toml: Path) -> None:
    config = load_config(sample_toml)
    assert isinstance(config, AppConfig)
    assert config.training.num_games == 4
    assert config.training.max_ply == 300
    assert config.training.algorithm == "ppo"
    assert config.display.moves_per_minute == 60
    assert config.model.display_name == "TestBot"
    assert config.model.architecture == "resnet"


def test_db_path_resolved_to_absolute(sample_toml: Path) -> None:
    config = load_config(sample_toml)
    assert Path(config.display.db_path).is_absolute()


def test_checkpoint_dir_resolved_to_absolute(sample_toml: Path) -> None:
    config = load_config(sample_toml)
    assert Path(config.training.checkpoint_dir).is_absolute()


def test_num_games_out_of_range(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text("""\
[training]
num_games = 0
max_ply = 500
algorithm = "ppo"
checkpoint_interval = 50
checkpoint_dir = "ckpt/"
[training.algorithm_params]
[display]
moves_per_minute = 30
db_path = "test.db"
[model]
display_name = "X"
architecture = "resnet"
[model.params]
hidden_size = 64
num_layers = 4
""")
    with pytest.raises(ValueError, match="num_games"):
        load_config(config_file)


def test_num_games_too_high(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text("""\
[training]
num_games = 11
max_ply = 500
algorithm = "ppo"
checkpoint_interval = 50
checkpoint_dir = "ckpt/"
[training.algorithm_params]
[display]
moves_per_minute = 30
db_path = "test.db"
[model]
display_name = "X"
architecture = "resnet"
[model.params]
hidden_size = 64
num_layers = 4
""")
    with pytest.raises(ValueError, match="num_games"):
        load_config(config_file)


def test_unknown_architecture(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text("""\
[training]
num_games = 4
max_ply = 500
algorithm = "ppo"
checkpoint_interval = 50
checkpoint_dir = "ckpt/"
[training.algorithm_params]
[display]
moves_per_minute = 30
db_path = "test.db"
[model]
display_name = "X"
architecture = "nonexistent"
[model.params]
hidden_size = 64
num_layers = 4
""")
    with pytest.raises(ValueError, match="architecture"):
        load_config(config_file)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'keisei.config'`

- [ ] **Step 4: Implement config.py**

`keisei/config.py`:
```python
"""TOML config loading with dataclass validation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

# Architecture names that are valid — must match model_registry.py
VALID_ARCHITECTURES = {"resnet", "mlp", "transformer"}
VALID_ALGORITHMS = {"ppo"}


@dataclass(frozen=True)
class TrainingConfig:
    num_games: int
    max_ply: int
    algorithm: str
    checkpoint_interval: int
    checkpoint_dir: str
    algorithm_params: dict[str, object]


@dataclass(frozen=True)
class DisplayConfig:
    moves_per_minute: int
    db_path: str


@dataclass(frozen=True)
class ModelConfig:
    display_name: str
    architecture: str
    params: dict[str, object]


@dataclass(frozen=True)
class AppConfig:
    training: TrainingConfig
    display: DisplayConfig
    model: ModelConfig


def load_config(path: Path) -> AppConfig:
    """Load and validate a TOML config file.

    Resolves relative paths (db_path, checkpoint_dir) to absolute
    paths relative to the config file's parent directory.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    config_dir = path.parent.resolve()

    # --- Training section ---
    t = raw.get("training", {})
    num_games = t.get("num_games", 8)
    if not (1 <= num_games <= 10):
        raise ValueError(f"num_games must be 1–10, got {num_games}")

    max_ply = t.get("max_ply", 500)
    if max_ply <= 0:
        raise ValueError(f"max_ply must be positive, got {max_ply}")

    algorithm = t.get("algorithm", "ppo")
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Valid: {sorted(VALID_ALGORITHMS)}"
        )

    checkpoint_interval = t.get("checkpoint_interval", 50)
    if checkpoint_interval <= 0:
        raise ValueError(f"checkpoint_interval must be positive, got {checkpoint_interval}")

    checkpoint_dir = str((config_dir / t.get("checkpoint_dir", "checkpoints/")).resolve())
    algorithm_params = t.get("algorithm_params", {})

    training = TrainingConfig(
        num_games=num_games,
        max_ply=max_ply,
        algorithm=algorithm,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        algorithm_params=algorithm_params,
    )

    # --- Display section ---
    d = raw.get("display", {})
    moves_per_minute = d.get("moves_per_minute", 30)
    if moves_per_minute < 0:
        raise ValueError(f"moves_per_minute must be >= 0, got {moves_per_minute}")

    db_path = str((config_dir / d.get("db_path", "keisei.db")).resolve())

    display = DisplayConfig(moves_per_minute=moves_per_minute, db_path=db_path)

    # --- Model section ---
    m = raw.get("model", {})
    display_name = m.get("display_name", "Player")
    architecture = m.get("architecture", "resnet")
    if architecture not in VALID_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Valid: {sorted(VALID_ARCHITECTURES)}"
        )

    model_params = m.get("params", {})

    model = ModelConfig(
        display_name=display_name,
        architecture=architecture,
        params=model_params,
    )

    return AppConfig(training=training, display=display, model=model)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add keisei/config.py keisei.toml tests/test_config.py
git commit -m "feat: config loading with TOML + dataclass validation"
```

---

### Task 3: SQLite Database Layer

**Files:**
- Create: `keisei/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write failing tests for DB schema and round-trip**

`tests/test_db.py`:
```python
import json
import sqlite3
import pytest
from pathlib import Path
from keisei.db import (
    init_db,
    write_metrics,
    read_metrics_since,
    write_game_snapshots,
    read_game_snapshots,
    write_training_state,
    read_training_state,
    update_heartbeat,
    update_training_progress,
    SCHEMA_VERSION,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def db(db_path: Path) -> Path:
    init_db(str(db_path))
    return db_path


def test_init_creates_tables(db: Path) -> None:
    conn = sqlite3.connect(str(db))
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "schema_version" in tables
    assert "metrics" in tables
    assert "game_snapshots" in tables
    assert "training_state" in tables
    conn.close()


def test_init_is_idempotent(db_path: Path) -> None:
    init_db(str(db_path))
    init_db(str(db_path))  # should not raise


def test_schema_version(db: Path) -> None:
    conn = sqlite3.connect(str(db))
    version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    assert version == SCHEMA_VERSION
    conn.close()


def test_wal_mode_enabled(db: Path) -> None:
    conn = sqlite3.connect(str(db))
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    conn.close()


def test_metrics_round_trip(db: Path) -> None:
    write_metrics(str(db), {
        "epoch": 1,
        "step": 100,
        "policy_loss": 2.5,
        "value_loss": 0.8,
        "entropy": 5.1,
        "win_rate": 0.52,
        "draw_rate": 0.1,
        "truncation_rate": 0.05,
        "avg_episode_length": 120.5,
        "gradient_norm": 1.2,
        "episodes_completed": 50,
    })
    rows = read_metrics_since(str(db), since_id=0)
    assert len(rows) == 1
    row = rows[0]
    assert row["epoch"] == 1
    assert row["step"] == 100
    assert abs(row["policy_loss"] - 2.5) < 1e-6
    assert abs(row["win_rate"] - 0.52) < 1e-6
    assert row["episodes_completed"] == 50
    assert "id" in row
    assert "timestamp" in row


def test_metrics_since_filters(db: Path) -> None:
    for i in range(5):
        write_metrics(str(db), {"epoch": i, "step": i * 10})
    rows = read_metrics_since(str(db), since_id=3)
    assert len(rows) == 2
    assert rows[0]["epoch"] == 3
    assert rows[1]["epoch"] == 4


def test_game_snapshots_round_trip(db: Path) -> None:
    board = [None] * 81
    board[0] = {"type": "king", "color": "black", "promoted": False, "row": 0, "col": 0}
    hands = {"black": {"pawn": 2}, "white": {"pawn": 0}}
    history = [{"action": 42, "notation": "7g-7f"}]

    snapshots = [
        {
            "game_id": 0,
            "board_json": json.dumps(board),
            "hands_json": json.dumps(hands),
            "current_player": "black",
            "ply": 10,
            "is_over": 0,
            "result": "in_progress",
            "sfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "in_check": 0,
            "move_history_json": json.dumps(history),
        }
    ]
    write_game_snapshots(str(db), snapshots)
    result = read_game_snapshots(str(db))
    assert len(result) == 1
    assert result[0]["game_id"] == 0
    assert result[0]["ply"] == 10
    assert json.loads(result[0]["board_json"])[0]["type"] == "king"
    assert json.loads(result[0]["move_history_json"]) == history


def test_game_snapshots_overwrite(db: Path) -> None:
    snap = {
        "game_id": 0, "board_json": "[]", "hands_json": "{}",
        "current_player": "black", "ply": 1, "is_over": 0,
        "result": "in_progress", "sfen": "startpos", "in_check": 0,
        "move_history_json": "[]",
    }
    write_game_snapshots(str(db), [snap])
    snap["ply"] = 99
    write_game_snapshots(str(db), [snap])
    result = read_game_snapshots(str(db))
    assert len(result) == 1
    assert result[0]["ply"] == 99


def test_training_state_write_and_read(db: Path) -> None:
    write_training_state(str(db), {
        "config_json": '{"test": true}',
        "display_name": "Hikaru",
        "model_arch": "resnet",
        "algorithm_name": "ppo",
        "started_at": "2026-04-01T00:00:00Z",
    })
    state = read_training_state(str(db))
    assert state is not None
    assert state["display_name"] == "Hikaru"
    assert state["status"] == "running"
    assert state["current_epoch"] == 0


def test_update_heartbeat(db: Path) -> None:
    write_training_state(str(db), {
        "config_json": "{}", "display_name": "X", "model_arch": "resnet",
        "algorithm_name": "ppo", "started_at": "2026-04-01T00:00:00Z",
    })
    old_state = read_training_state(str(db))
    update_heartbeat(str(db))
    new_state = read_training_state(str(db))
    assert new_state["heartbeat_at"] >= old_state["heartbeat_at"]


def test_update_training_progress(db: Path) -> None:
    write_training_state(str(db), {
        "config_json": "{}", "display_name": "X", "model_arch": "resnet",
        "algorithm_name": "ppo", "started_at": "2026-04-01T00:00:00Z",
    })
    update_training_progress(str(db), epoch=5, step=500, checkpoint_path="/tmp/ckpt.pt")
    state = read_training_state(str(db))
    assert state["current_epoch"] == 5
    assert state["current_step"] == 500
    assert state["checkpoint_path"] == "/tmp/ckpt.pt"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'keisei.db'`

- [ ] **Step 3: Implement db.py**

`keisei/db.py`:
```python
"""SQLite database layer — schema, migrations, read/write helpers."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

SCHEMA_VERSION = 1


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA wal_autocheckpoint = 1000")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db(db_path: str) -> None:
    """Create tables if they don't exist. Idempotent."""
    conn = _connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS metrics (
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
        );

        CREATE INDEX IF NOT EXISTS idx_metrics_epoch ON metrics(epoch);
        CREATE INDEX IF NOT EXISTS idx_metrics_id ON metrics(id);

        CREATE TABLE IF NOT EXISTS game_snapshots (
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
        );

        CREATE TABLE IF NOT EXISTS training_state (
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
        );
    """)

    # Insert schema version if not present
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
    conn.commit()
    conn.close()


def write_metrics(db_path: str, metrics: dict[str, Any]) -> None:
    """Append one metrics row."""
    conn = _connect(db_path)
    conn.execute(
        """INSERT INTO metrics (epoch, step, policy_loss, value_loss, entropy,
           win_rate, draw_rate, truncation_rate, avg_episode_length,
           gradient_norm, episodes_completed)
           VALUES (:epoch, :step, :policy_loss, :value_loss, :entropy,
           :win_rate, :draw_rate, :truncation_rate, :avg_episode_length,
           :gradient_norm, :episodes_completed)""",
        {
            "epoch": metrics.get("epoch", 0),
            "step": metrics.get("step", 0),
            "policy_loss": metrics.get("policy_loss"),
            "value_loss": metrics.get("value_loss"),
            "entropy": metrics.get("entropy"),
            "win_rate": metrics.get("win_rate"),
            "draw_rate": metrics.get("draw_rate"),
            "truncation_rate": metrics.get("truncation_rate"),
            "avg_episode_length": metrics.get("avg_episode_length"),
            "gradient_norm": metrics.get("gradient_norm"),
            "episodes_completed": metrics.get("episodes_completed"),
        },
    )
    conn.commit()
    conn.close()


def read_metrics_since(db_path: str, since_id: int, limit: int = 500) -> list[dict[str, Any]]:
    """Read metrics rows with id > since_id, up to limit rows."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM metrics WHERE id > ? ORDER BY id LIMIT ?",
        (since_id, limit),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def write_game_snapshots(db_path: str, snapshots: list[dict[str, Any]]) -> None:
    """Write game snapshots inside a single transaction."""
    conn = _connect(db_path)
    conn.execute("BEGIN")
    for snap in snapshots:
        conn.execute(
            """INSERT OR REPLACE INTO game_snapshots
               (game_id, board_json, hands_json, current_player, ply,
                is_over, result, sfen, in_check, move_history_json)
               VALUES (:game_id, :board_json, :hands_json, :current_player,
                :ply, :is_over, :result, :sfen, :in_check, :move_history_json)""",
            snap,
        )
    conn.commit()
    conn.close()


def read_game_snapshots(db_path: str) -> list[dict[str, Any]]:
    """Read all game snapshot rows."""
    conn = _connect(db_path)
    rows = conn.execute("SELECT * FROM game_snapshots ORDER BY game_id").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def write_training_state(db_path: str, state: dict[str, Any]) -> None:
    """Insert or replace the singleton training state row."""
    conn = _connect(db_path)
    conn.execute(
        """INSERT OR REPLACE INTO training_state
           (id, config_json, display_name, model_arch, algorithm_name,
            started_at, current_epoch, current_step, checkpoint_path, status)
           VALUES (1, :config_json, :display_name, :model_arch, :algorithm_name,
            :started_at, :current_epoch, :current_step, :checkpoint_path, :status)""",
        {
            "config_json": state["config_json"],
            "display_name": state["display_name"],
            "model_arch": state["model_arch"],
            "algorithm_name": state["algorithm_name"],
            "started_at": state["started_at"],
            "current_epoch": state.get("current_epoch", 0),
            "current_step": state.get("current_step", 0),
            "checkpoint_path": state.get("checkpoint_path"),
            "status": state.get("status", "running"),
        },
    )
    conn.commit()
    conn.close()


def read_training_state(db_path: str) -> dict[str, Any] | None:
    """Read the singleton training state row, or None if not present."""
    conn = _connect(db_path)
    row = conn.execute("SELECT * FROM training_state WHERE id = 1").fetchone()
    conn.close()
    return dict(row) if row else None


def update_heartbeat(db_path: str) -> None:
    """Update the heartbeat timestamp on the training state row."""
    conn = _connect(db_path)
    conn.execute(
        "UPDATE training_state SET heartbeat_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = 1"
    )
    conn.commit()
    conn.close()


def update_training_progress(
    db_path: str, epoch: int, step: int, checkpoint_path: str | None = None
) -> None:
    """Update epoch, step, and optionally checkpoint_path."""
    conn = _connect(db_path)
    if checkpoint_path is not None:
        conn.execute(
            "UPDATE training_state SET current_epoch = ?, current_step = ?, checkpoint_path = ?, heartbeat_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = 1",
            (epoch, step, checkpoint_path),
        )
    else:
        conn.execute(
            "UPDATE training_state SET current_epoch = ?, current_step = ?, heartbeat_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = 1",
            (epoch, step),
        )
    conn.commit()
    conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add keisei/db.py tests/test_db.py
git commit -m "feat: SQLite database layer with WAL mode and schema v1"
```

---

### Task 4: BaseModel ABC and ResNet

**Files:**
- Create: `keisei/training/models/base.py`
- Create: `keisei/training/models/resnet.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for base model contract and ResNet shapes**

`tests/test_models.py`:
```python
import pytest
import torch
from keisei.training.models.base import BaseModel
from keisei.training.models.resnet import ResNetModel, ResNetParams


def test_base_model_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseModel()  # type: ignore[abstract]


class TestResNet:
    def test_forward_shapes(self) -> None:
        params = ResNetParams(hidden_size=32, num_layers=2)
        model = ResNetModel(params)
        obs = torch.randn(4, 46, 9, 9)
        policy_logits, value = model(obs)
        assert policy_logits.shape == (4, 13527)
        assert value.shape == (4, 1)

    def test_value_bounded(self) -> None:
        params = ResNetParams(hidden_size=32, num_layers=2)
        model = ResNetModel(params)
        obs = torch.randn(8, 46, 9, 9)
        _, value = model(obs)
        assert value.min() >= -1.0
        assert value.max() <= 1.0

    def test_single_sample(self) -> None:
        params = ResNetParams(hidden_size=32, num_layers=2)
        model = ResNetModel(params)
        obs = torch.randn(1, 46, 9, 9)
        policy_logits, value = model(obs)
        assert policy_logits.shape == (1, 13527)
        assert value.shape == (1, 1)

    def test_has_batchnorm(self) -> None:
        params = ResNetParams(hidden_size=32, num_layers=2)
        model = ResNetModel(params)
        bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        assert len(bn_layers) > 0, "ResNet must use BatchNorm2d"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement base.py**

`keisei/training/models/base.py`:
```python
"""Abstract base model for all Keisei architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base for all Keisei model architectures.

    Contract:
    - Input: observation tensor (batch, 46, 9, 9)
    - Output: (policy_logits, value) where:
        - policy_logits: (batch, 13527) -- RAW, UNMASKED logits
        - value: (batch, 1) -- scalar value estimate, tanh-activated
    - The algorithm (not the model) applies the legal mask before softmax.
    """

    OBS_CHANNELS = 46
    BOARD_SIZE = 9
    ACTION_SPACE = 13527

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

- [ ] **Step 4: Implement resnet.py**

`keisei/training/models/resnet.py`:
```python
"""ResNet architecture for Shogi policy+value network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .base import BaseModel


@dataclass(frozen=True)
class ResNetParams:
    hidden_size: int  # channels per residual block
    num_layers: int   # number of residual blocks


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class ResNetModel(BaseModel):
    def __init__(self, params: ResNetParams) -> None:
        super().__init__()
        ch = params.hidden_size

        # Input projection: 46 channels → hidden_size
        self.input_conv = nn.Conv2d(self.OBS_CHANNELS, ch, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(ch)

        # Residual tower
        self.blocks = nn.Sequential(
            *[ResidualBlock(ch) for _ in range(params.num_layers)]
        )

        # Policy head: 1x1 conv → flatten → linear → 13527
        policy_channels = 2
        self.policy_conv = nn.Conv2d(ch, policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc = nn.Linear(
            policy_channels * self.BOARD_SIZE * self.BOARD_SIZE,
            self.ACTION_SPACE,
        )

        # Value head: 1x1 conv → flatten → fc → relu → fc(1) → tanh
        value_channels = 1
        self.value_conv = nn.Conv2d(ch, value_channels, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(
            value_channels * self.BOARD_SIZE * self.BOARD_SIZE,
            ch,
        )
        self.value_fc2 = nn.Linear(ch, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.input_bn(self.input_conv(obs)))
        x = self.blocks(x)

        # Policy head
        p = torch.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = torch.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add keisei/training/models/base.py keisei/training/models/resnet.py tests/test_models.py
git commit -m "feat: BaseModel ABC and ResNet architecture with BatchNorm"
```

---

### Task 5: MLP and Transformer Architectures

**Files:**
- Create: `keisei/training/models/mlp.py`
- Create: `keisei/training/models/transformer.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Add failing tests for MLP and Transformer**

Append to `tests/test_models.py`:
```python
from keisei.training.models.mlp import MLPModel, MLPParams
from keisei.training.models.transformer import TransformerModel, TransformerParams


class TestMLP:
    def test_forward_shapes(self) -> None:
        params = MLPParams(hidden_sizes=[128, 64])
        model = MLPModel(params)
        obs = torch.randn(4, 46, 9, 9)
        policy_logits, value = model(obs)
        assert policy_logits.shape == (4, 13527)
        assert value.shape == (4, 1)

    def test_value_bounded(self) -> None:
        params = MLPParams(hidden_sizes=[128, 64])
        model = MLPModel(params)
        obs = torch.randn(8, 46, 9, 9)
        _, value = model(obs)
        assert value.min() >= -1.0
        assert value.max() <= 1.0

    def test_has_layernorm(self) -> None:
        params = MLPParams(hidden_sizes=[128, 64])
        model = MLPModel(params)
        ln_layers = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
        assert len(ln_layers) > 0, "MLP must use LayerNorm"


class TestTransformer:
    def test_forward_shapes(self) -> None:
        params = TransformerParams(d_model=32, nhead=4, num_layers=2)
        model = TransformerModel(params)
        obs = torch.randn(4, 46, 9, 9)
        policy_logits, value = model(obs)
        assert policy_logits.shape == (4, 13527)
        assert value.shape == (4, 1)

    def test_value_bounded(self) -> None:
        params = TransformerParams(d_model=32, nhead=4, num_layers=2)
        model = TransformerModel(params)
        obs = torch.randn(8, 46, 9, 9)
        _, value = model(obs)
        assert value.min() >= -1.0
        assert value.max() <= 1.0

    def test_has_positional_encoding(self) -> None:
        params = TransformerParams(d_model=32, nhead=4, num_layers=2)
        model = TransformerModel(params)
        assert hasattr(model, "row_embed"), "Transformer must have 2D row embeddings"
        assert hasattr(model, "col_embed"), "Transformer must have 2D column embeddings"
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `uv run pytest tests/test_models.py -v -k "MLP or Transformer"`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement mlp.py**

`keisei/training/models/mlp.py`:
```python
"""MLP baseline architecture -- intentionally lacks spatial inductive bias."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .base import BaseModel


@dataclass(frozen=True)
class MLPParams:
    hidden_sizes: list[int]  # e.g. [2048, 512, 256] — staged reduction


class MLPModel(BaseModel):
    def __init__(self, params: MLPParams) -> None:
        super().__init__()
        input_size = self.OBS_CHANNELS * self.BOARD_SIZE * self.BOARD_SIZE  # 3726

        # Shared trunk: staged reduction with LayerNorm
        layers: list[nn.Module] = []
        prev_size = input_size
        for size in params.hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.LayerNorm(size))
            layers.append(nn.ReLU())
            prev_size = size
        self.trunk = nn.Sequential(*layers)

        # Policy head
        self.policy_fc = nn.Linear(prev_size, self.ACTION_SPACE)

        # Value head
        self.value_fc = nn.Linear(prev_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = obs.flatten(1)  # (batch, 3726)
        x = self.trunk(x)
        policy_logits = self.policy_fc(x)
        value = torch.tanh(self.value_fc(x))
        return policy_logits, value
```

- [ ] **Step 4: Implement transformer.py**

`keisei/training/models/transformer.py`:
```python
"""Transformer architecture with 2D positional encoding for Shogi."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .base import BaseModel


@dataclass(frozen=True)
class TransformerParams:
    d_model: int    # model dimension
    nhead: int      # number of attention heads
    num_layers: int # number of transformer encoder layers


class TransformerModel(BaseModel):
    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        d = params.d_model

        # Input projection: 46 features per square → d_model
        self.input_proj = nn.Linear(self.OBS_CHANNELS, d)

        # 2D positional encoding: row (9) + column (9) embeddings
        self.row_embed = nn.Embedding(self.BOARD_SIZE, d)
        self.col_embed = nn.Embedding(self.BOARD_SIZE, d)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=params.nhead, dim_feedforward=d * 4,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=params.num_layers)

        # Policy head: per-token projection → flatten
        self.policy_fc = nn.Linear(d * self.BOARD_SIZE * self.BOARD_SIZE, self.ACTION_SPACE)

        # Value head: mean-pool → MLP → tanh
        self.value_fc1 = nn.Linear(d, d)
        self.value_fc2 = nn.Linear(d, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = obs.shape[0]

        # Reshape: (batch, 46, 9, 9) → (batch, 81, 46)
        x = obs.permute(0, 2, 3, 1).reshape(batch, 81, self.OBS_CHANNELS)

        # Project to d_model
        x = self.input_proj(x)  # (batch, 81, d_model)

        # Add 2D positional encoding
        rows = torch.arange(self.BOARD_SIZE, device=obs.device)
        cols = torch.arange(self.BOARD_SIZE, device=obs.device)
        row_emb = self.row_embed(rows)  # (9, d)
        col_emb = self.col_embed(cols)  # (9, d)
        # Broadcast: each square gets row_embed[r] + col_embed[c]
        pos = (row_emb.unsqueeze(1) + col_emb.unsqueeze(0)).reshape(81, -1)  # (81, d)
        x = x + pos.unsqueeze(0)  # (batch, 81, d)

        # Transformer encoder
        x = self.encoder(x)  # (batch, 81, d)

        # Policy head
        policy_logits = self.policy_fc(x.reshape(batch, -1))  # (batch, 13527)

        # Value head: mean pool → MLP → tanh
        pooled = x.mean(dim=1)  # (batch, d)
        v = torch.relu(self.value_fc1(pooled))
        value = torch.tanh(self.value_fc2(v))  # (batch, 1)

        return policy_logits, value
```

- [ ] **Step 5: Run all model tests**

Run: `uv run pytest tests/test_models.py -v`
Expected: All 10 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add keisei/training/models/mlp.py keisei/training/models/transformer.py tests/test_models.py
git commit -m "feat: MLP baseline and Transformer with 2D positional encoding"
```

---

### Task 6: Model and Algorithm Registries

**Files:**
- Create: `keisei/training/model_registry.py`
- Create: `keisei/training/algorithm_registry.py`
- Create: `tests/test_registries.py`

- [ ] **Step 1: Write failing tests**

`tests/test_registries.py`:
```python
import pytest
import torch
from keisei.training.model_registry import build_model, VALID_ARCHITECTURES, validate_model_params
from keisei.training.algorithm_registry import VALID_ALGORITHMS, validate_algorithm_params


class TestModelRegistry:
    def test_valid_architectures_match_config(self) -> None:
        assert "resnet" in VALID_ARCHITECTURES
        assert "mlp" in VALID_ARCHITECTURES
        assert "transformer" in VALID_ARCHITECTURES

    def test_build_resnet(self) -> None:
        model = build_model("resnet", {"hidden_size": 32, "num_layers": 2})
        obs = torch.randn(2, 46, 9, 9)
        p, v = model(obs)
        assert p.shape == (2, 13527)
        assert v.shape == (2, 1)

    def test_build_mlp(self) -> None:
        model = build_model("mlp", {"hidden_sizes": [128, 64]})
        obs = torch.randn(2, 46, 9, 9)
        p, v = model(obs)
        assert p.shape == (2, 13527)

    def test_build_transformer(self) -> None:
        model = build_model("transformer", {"d_model": 32, "nhead": 4, "num_layers": 2})
        obs = torch.randn(2, 46, 9, 9)
        p, v = model(obs)
        assert p.shape == (2, 13527)

    def test_unknown_architecture_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model("nonexistent", {})

    def test_bad_params_raises(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            validate_model_params("resnet", {"bad_key": 99})

    def test_missing_params_raises(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            validate_model_params("resnet", {})


class TestAlgorithmRegistry:
    def test_ppo_in_registry(self) -> None:
        assert "ppo" in VALID_ALGORITHMS

    def test_unknown_algorithm_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown algorithm"):
            validate_algorithm_params("nonexistent", {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_registries.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement model_registry.py**

`keisei/training/model_registry.py`:
```python
"""Model registry: architecture name -> (model class, params dataclass)."""

from __future__ import annotations

from typing import Any

from keisei.training.models.base import BaseModel
from keisei.training.models.resnet import ResNetModel, ResNetParams
from keisei.training.models.mlp import MLPModel, MLPParams
from keisei.training.models.transformer import TransformerModel, TransformerParams

_REGISTRY: dict[str, tuple[type[BaseModel], type]] = {
    "resnet": (ResNetModel, ResNetParams),
    "mlp": (MLPModel, MLPParams),
    "transformer": (TransformerModel, TransformerParams),
}

VALID_ARCHITECTURES = set(_REGISTRY.keys())


def validate_model_params(architecture: str, params: dict[str, Any]) -> object:
    """Validate params dict against the architecture's dataclass. Returns the dataclass instance."""
    if architecture not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Valid: {sorted(VALID_ARCHITECTURES)}"
        )
    _, params_cls = _REGISTRY[architecture]
    try:
        return params_cls(**params)
    except TypeError as e:
        raise TypeError(f"Invalid params for '{architecture}': {e}") from e


def build_model(architecture: str, params: dict[str, Any]) -> BaseModel:
    """Build a model from architecture name and params dict."""
    validated_params = validate_model_params(architecture, params)
    model_cls, _ = _REGISTRY[architecture]
    return model_cls(validated_params)
```

- [ ] **Step 4: Implement algorithm_registry.py**

`keisei/training/algorithm_registry.py`:
```python
"""Algorithm registry: algorithm name -> params dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class PPOParams:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    epochs_per_batch: int = 4
    batch_size: int = 256


_PARAM_SCHEMAS: dict[str, type] = {
    "ppo": PPOParams,
}

VALID_ALGORITHMS = set(_PARAM_SCHEMAS.keys())


def validate_algorithm_params(algorithm: str, params: dict[str, Any]) -> object:
    """Validate params dict against the algorithm's dataclass."""
    if algorithm not in _PARAM_SCHEMAS:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Valid: {sorted(VALID_ALGORITHMS)}"
        )
    params_cls = _PARAM_SCHEMAS[algorithm]
    try:
        return params_cls(**params)
    except TypeError as e:
        raise TypeError(f"Invalid params for '{algorithm}': {e}") from e
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_registries.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add keisei/training/model_registry.py keisei/training/algorithm_registry.py tests/test_registries.py
git commit -m "feat: model and algorithm registries with dataclass validation"
```

---

### Task 7: PPO Algorithm (Rollout Buffer + GAE + Update)

**Files:**
- Create: `keisei/training/ppo.py`
- Create: `tests/test_ppo.py`

- [ ] **Step 1: Write failing tests for GAE and PPO update**

`tests/test_ppo.py`:
```python
import pytest
import torch
import numpy as np
from keisei.training.ppo import compute_gae, RolloutBuffer, PPOAlgorithm
from keisei.training.algorithm_registry import PPOParams


class TestGAE:
    def test_single_step_terminal(self) -> None:
        """Single step ending in terminal reward of +1. GAE = reward - value."""
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.5])
        dones = torch.tensor([True])
        next_value = torch.tensor(0.0)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        # A = r + gamma * V(s') * (1 - done) - V(s) = 1.0 + 0 - 0.5 = 0.5
        assert abs(advantages[0].item() - 0.5) < 1e-5

    def test_two_steps_no_terminal(self) -> None:
        """Two steps, not terminal. Known closed-form GAE."""
        rewards = torch.tensor([0.0, 0.0])
        values = torch.tensor([0.5, 0.6])
        dones = torch.tensor([False, False])
        next_value = torch.tensor(0.7)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        # delta_1 = 0 + 0.99*0.7 - 0.6 = 0.093
        # delta_0 = 0 + 0.99*0.6 - 0.5 = 0.094
        # A_1 = delta_1 = 0.093
        # A_0 = delta_0 + 0.99*0.95*A_1 = 0.094 + 0.9405*0.093 = 0.1815...
        assert advantages.shape == (2,)
        assert abs(advantages[1].item() - 0.093) < 1e-3
        assert abs(advantages[0].item() - 0.1815) < 1e-2

    def test_terminal_resets_bootstrap(self) -> None:
        """Done at step 0 should zero the bootstrap for that step."""
        rewards = torch.tensor([1.0, 0.0])
        values = torch.tensor([0.3, 0.4])
        dones = torch.tensor([True, False])
        next_value = torch.tensor(0.5)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        # delta_0 = 1.0 + 0 - 0.3 = 0.7 (done=True zeroes next value)
        assert abs(advantages[0].item() - 0.7) < 0.1  # approximate due to GAE recursion


class TestRolloutBuffer:
    def test_add_and_get(self) -> None:
        buf = RolloutBuffer(num_envs=2, obs_shape=(46, 9, 9), action_space=13527)
        obs = torch.randn(2, 46, 9, 9)
        actions = torch.tensor([0, 1])
        log_probs = torch.tensor([-1.0, -2.0])
        values = torch.tensor([0.5, 0.6])
        rewards = torch.tensor([0.0, 0.0])
        dones = torch.tensor([False, False])
        legal_masks = torch.ones(2, 13527, dtype=torch.bool)

        buf.add(obs, actions, log_probs, values, rewards, dones, legal_masks)
        assert buf.size == 1

        buf.add(obs, actions, log_probs, values, rewards, dones, legal_masks)
        assert buf.size == 2

    def test_clear(self) -> None:
        buf = RolloutBuffer(num_envs=2, obs_shape=(46, 9, 9), action_space=13527)
        obs = torch.randn(2, 46, 9, 9)
        buf.add(obs, torch.zeros(2, dtype=torch.long), torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2, dtype=torch.bool), torch.ones(2, 13527, dtype=torch.bool))
        buf.clear()
        assert buf.size == 0


class TestPPOAlgorithm:
    def test_select_actions(self) -> None:
        from keisei.training.models.resnet import ResNetModel, ResNetParams
        params = PPOParams(learning_rate=1e-3, batch_size=4, epochs_per_batch=1)
        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
        ppo = PPOAlgorithm(params, model)

        obs = torch.randn(4, 46, 9, 9)
        legal_masks = torch.ones(4, 13527, dtype=torch.bool)
        actions, log_probs, values = ppo.select_actions(obs, legal_masks)
        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)
        assert (actions >= 0).all()
        assert (actions < 13527).all()

    def test_update_returns_losses(self) -> None:
        from keisei.training.models.resnet import ResNetModel, ResNetParams
        params = PPOParams(learning_rate=1e-3, batch_size=4, epochs_per_batch=1)
        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
        ppo = PPOAlgorithm(params, model)

        buf = RolloutBuffer(num_envs=2, obs_shape=(46, 9, 9), action_space=13527)
        for _ in range(8):
            obs = torch.randn(2, 46, 9, 9)
            legal_masks = torch.ones(2, 13527, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(obs, actions, log_probs, values, torch.zeros(2), torch.zeros(2, dtype=torch.bool), legal_masks)

        next_values = torch.zeros(2)
        losses = ppo.update(buf, next_values)
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "entropy" in losses
        assert "gradient_norm" in losses
        assert isinstance(losses["policy_loss"], float)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ppo.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement ppo.py**

`keisei/training/ppo.py`:
```python
"""PPO algorithm: rollout buffer, GAE, clipped policy update."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from keisei.training.models.base import BaseModel
from keisei.training.algorithm_registry import PPOParams


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: (T,) rewards at each timestep
        values: (T,) value estimates at each timestep
        dones: (T,) boolean terminal flags
        next_value: scalar value estimate for the state after the last step
        gamma: discount factor
        lam: GAE lambda

    Returns:
        advantages: (T,) GAE advantage estimates
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    last_gae = torch.tensor(0.0, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_val * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae

    return advantages


class RolloutBuffer:
    """Fixed-size buffer for collecting PPO rollouts across N environments."""

    def __init__(self, num_envs: int, obs_shape: tuple[int, ...], action_space: int) -> None:
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.clear()

    def clear(self) -> None:
        self.observations: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []
        self.legal_masks: list[torch.Tensor] = []

    @property
    def size(self) -> int:
        return len(self.observations)

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        legal_masks: torch.Tensor,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.legal_masks.append(legal_masks)

    def flatten(self) -> dict[str, torch.Tensor]:
        """Stack all steps into flat tensors for training."""
        return {
            "observations": torch.stack(self.observations).reshape(-1, *self.obs_shape),
            "actions": torch.stack(self.actions).reshape(-1),
            "log_probs": torch.stack(self.log_probs).reshape(-1),
            "values": torch.stack(self.values).reshape(-1),
            "rewards": torch.stack(self.rewards).reshape(-1),
            "dones": torch.stack(self.dones).reshape(-1),
            "legal_masks": torch.stack(self.legal_masks).reshape(-1, self.action_space),
        }


class PPOAlgorithm:
    """PPO with clipped objective, GAE, and masked action probabilities."""

    def __init__(self, params: PPOParams, model: BaseModel) -> None:
        self.params = params
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    @torch.no_grad()
    def select_actions(
        self, obs: torch.Tensor, legal_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select actions using the current policy with legal mask applied.

        Returns: (actions, log_probs, values) — all shape (batch,).
        """
        policy_logits, values = self.model(obs)

        # Mask illegal actions: set logits to -inf where mask is False
        masked_logits = policy_logits.masked_fill(~legal_masks, float("-inf"))
        probs = F.softmax(masked_logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions, log_probs, values.squeeze(-1)

    def update(
        self, buffer: RolloutBuffer, next_values: torch.Tensor
    ) -> dict[str, float]:
        """Run PPO update on the collected rollout buffer.

        Returns dict of loss values for logging.
        """
        data = buffer.flatten()
        T = buffer.size
        N = buffer.num_envs

        # Compute GAE per environment
        rewards_2d = data["rewards"].reshape(T, N)
        values_2d = data["values"].reshape(T, N)
        dones_2d = data["dones"].reshape(T, N)

        all_advantages = torch.zeros(T, N)
        for env_i in range(N):
            all_advantages[:, env_i] = compute_gae(
                rewards_2d[:, env_i],
                values_2d[:, env_i],
                dones_2d[:, env_i],
                next_values[env_i],
                gamma=self.params.gamma,
                lam=0.95,
            )

        advantages = all_advantages.reshape(-1)
        returns = advantages + data["values"]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_samples = T * N
        batch_size = min(self.params.batch_size, total_samples)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_grad_norm = 0.0
        num_updates = 0

        for _ in range(self.params.epochs_per_batch):
            indices = torch.randperm(total_samples)
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                idx = indices[start:end]

                batch_obs = data["observations"][idx]
                batch_actions = data["actions"][idx]
                batch_old_log_probs = data["log_probs"][idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                batch_legal_masks = data["legal_masks"][idx]

                # Forward pass
                policy_logits, values = self.model(batch_obs)

                # Masked log probs
                masked_logits = policy_logits.masked_fill(~batch_legal_masks, float("-inf"))
                log_probs_all = F.log_softmax(masked_logits, dim=-1)
                new_log_probs = log_probs_all.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                # Entropy (over legal actions only)
                probs = F.softmax(masked_logits, dim=-1)
                entropy = -(probs * log_probs_all).sum(dim=-1).mean()

                # PPO clipped objective
                ratio = (new_log_probs - batch_old_log_probs).exp()
                clip = self.params.clip_epsilon
                surr1 = ratio * batch_advantages
                surr2 = ratio.clamp(1 - clip, 1 + clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_grad_norm += float(grad_norm)
                num_updates += 1

        buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "gradient_norm": total_grad_norm / max(num_updates, 1),
        }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ppo.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add keisei/training/ppo.py tests/test_ppo.py
git commit -m "feat: PPO algorithm with GAE, rollout buffer, masked action probabilities"
```

---

### Task 8: Checkpointing

**Files:**
- Create: `keisei/training/checkpoint.py`
- Create: `tests/test_checkpoint.py`

- [ ] **Step 1: Write failing tests**

`tests/test_checkpoint.py`:
```python
import pytest
import torch
from pathlib import Path
from keisei.training.checkpoint import save_checkpoint, load_checkpoint
from keisei.training.models.resnet import ResNetModel, ResNetParams


@pytest.fixture
def model() -> ResNetModel:
    return ResNetModel(ResNetParams(hidden_size=16, num_layers=1))


def test_save_and_load_round_trip(tmp_path: Path, model: ResNetModel) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(path, model, optimizer, epoch=10, step=1000)
    assert path.exists()

    loaded = load_checkpoint(path, model, optimizer)
    assert loaded["epoch"] == 10
    assert loaded["step"] == 1000


def test_model_weights_preserved(tmp_path: Path, model: ResNetModel) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"

    # Get original output
    obs = torch.randn(1, 46, 9, 9)
    with torch.no_grad():
        original_policy, original_value = model(obs)

    save_checkpoint(path, model, optimizer, epoch=1, step=100)

    # Perturb model
    for p in model.parameters():
        p.data.add_(torch.randn_like(p))

    # Load checkpoint restores original weights
    load_checkpoint(path, model, optimizer)

    with torch.no_grad():
        restored_policy, restored_value = model(obs)

    assert torch.allclose(original_policy, restored_policy, atol=1e-6)
    assert torch.allclose(original_value, restored_value, atol=1e-6)


def test_load_nonexistent_raises(tmp_path: Path, model: ResNetModel) -> None:
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", model, optimizer)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_checkpoint.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement checkpoint.py**

`keisei/training/checkpoint.py`:
```python
"""Model checkpointing: save and load model + optimizer + training state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
) -> None:
    """Save model weights, optimizer state, and training progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Load checkpoint into model and optimizer. Returns metadata dict.

    Raises FileNotFoundError if path doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {"epoch": checkpoint["epoch"], "step": checkpoint["step"]}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_checkpoint.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add keisei/training/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: model checkpointing with save/load round-trip"
```

---

### Task 9: Training Loop Orchestrator

**Files:**
- Create: `keisei/training/loop.py`
- Create: `tests/test_loop.py`

This is the integration task. It ties config → model → PPO → VecEnv → DB together. Tests use mocked VecEnv since the Rust engine may not be available in CI.

- [ ] **Step 1: Write failing integration test**

`tests/test_loop.py`:
```python
"""Integration tests for the training loop.

Uses a mock VecEnv to avoid requiring the Rust engine in CI.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from keisei.training.loop import TrainingLoop
from keisei.config import load_config
from keisei.db import init_db, read_metrics_since, read_training_state


def _make_mock_vecenv(num_envs: int = 4) -> MagicMock:
    """Create a mock VecEnv that returns realistic-shaped data."""
    mock = MagicMock()
    mock.num_envs = num_envs
    mock.action_space_size = 13527
    mock.observation_channels = 46

    reset_result = MagicMock()
    reset_result.observations = np.zeros((num_envs, 46, 9, 9), dtype=np.float32)
    reset_result.legal_masks = np.ones((num_envs, 13527), dtype=bool)
    mock.reset.return_value = reset_result

    step_result = MagicMock()
    step_result.observations = np.zeros((num_envs, 46, 9, 9), dtype=np.float32)
    step_result.legal_masks = np.ones((num_envs, 13527), dtype=bool)
    step_result.rewards = np.zeros(num_envs, dtype=np.float32)
    step_result.terminated = np.zeros(num_envs, dtype=bool)
    step_result.truncated = np.zeros(num_envs, dtype=bool)
    step_result.current_players = np.zeros(num_envs, dtype=np.uint8)
    step_metadata = MagicMock()
    step_metadata.ply_count = np.ones(num_envs, dtype=np.uint16) * 10
    step_result.step_metadata = step_metadata
    mock.step.return_value = step_result

    mock.episodes_completed = 0
    mock.episodes_drawn = 0
    mock.episodes_truncated = 0
    mock.draw_rate = 0.0
    mock.mean_episode_length.return_value = 0.0
    mock.truncation_rate.return_value = 0.0

    mock.get_spectator_data.return_value = [
        {
            "board": [None] * 81,
            "hands": {"black": {}, "white": {}},
            "current_player": "black",
            "ply": 0,
            "is_over": False,
            "result": "in_progress",
            "sfen": "startpos",
            "in_check": False,
        }
        for _ in range(num_envs)
    ]
    mock.get_sfens.return_value = ["startpos"] * num_envs

    return mock


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    config = tmp_path / "test.toml"
    config.write_text(f"""\
[training]
num_games = 4
max_ply = 100
algorithm = "ppo"
checkpoint_interval = 2
checkpoint_dir = "{tmp_path / 'ckpt'}"

[training.algorithm_params]
learning_rate = 1e-3
gamma = 0.99
clip_epsilon = 0.2
epochs_per_batch = 1
batch_size = 8

[display]
moves_per_minute = 0
db_path = "{tmp_path / 'test.db'}"

[model]
display_name = "TestBot"
architecture = "resnet"

[model.params]
hidden_size = 16
num_layers = 1
""")
    return config


def test_training_loop_runs_one_epoch(config_file: Path) -> None:
    config = load_config(config_file)
    mock_vecenv = _make_mock_vecenv(num_envs=config.training.num_games)

    loop = TrainingLoop(config, vecenv=mock_vecenv)
    loop.run(num_epochs=1, steps_per_epoch=8)

    # Verify metrics were written to DB
    rows = read_metrics_since(config.display.db_path, since_id=0)
    assert len(rows) == 1
    assert rows[0]["epoch"] == 0

    # Verify training state exists
    state = read_training_state(config.display.db_path)
    assert state is not None
    assert state["display_name"] == "TestBot"
    assert state["status"] == "running"


def test_training_loop_creates_checkpoint(config_file: Path) -> None:
    config = load_config(config_file)
    mock_vecenv = _make_mock_vecenv(num_envs=config.training.num_games)

    loop = TrainingLoop(config, vecenv=mock_vecenv)
    loop.run(num_epochs=3, steps_per_epoch=8)

    # checkpoint_interval=2, so we should have checkpoints at epoch 1 and 2 (0-indexed)
    ckpt_dir = Path(config.training.checkpoint_dir)
    assert ckpt_dir.exists()
    checkpoints = list(ckpt_dir.glob("*.pt"))
    assert len(checkpoints) >= 1


def test_training_loop_writes_metrics_each_epoch(config_file: Path) -> None:
    config = load_config(config_file)
    mock_vecenv = _make_mock_vecenv(num_envs=config.training.num_games)

    loop = TrainingLoop(config, vecenv=mock_vecenv)
    loop.run(num_epochs=3, steps_per_epoch=8)

    rows = read_metrics_since(config.display.db_path, since_id=0)
    assert len(rows) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_loop.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement loop.py**

`keisei/training/loop.py`:
```python
"""Training loop orchestrator."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from keisei.config import AppConfig
from keisei.db import (
    init_db,
    write_metrics,
    write_game_snapshots,
    write_training_state,
    update_heartbeat,
    update_training_progress,
    read_training_state,
)
from keisei.training.model_registry import build_model
from keisei.training.algorithm_registry import validate_algorithm_params, PPOParams
from keisei.training.ppo import PPOAlgorithm, RolloutBuffer
from keisei.training.checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class TrainingLoop:
    """Main training loop: VecEnv -> Model -> PPO -> SQLite."""

    def __init__(self, config: AppConfig, vecenv: Any = None) -> None:
        self.config = config
        self.db_path = config.display.db_path

        # Init DB
        init_db(self.db_path)

        # Build model
        self.model = build_model(config.model.architecture, config.model.params)
        logger.info(
            "Model: %s (%s), params: %d",
            config.model.display_name,
            config.model.architecture,
            sum(p.numel() for p in self.model.parameters()),
        )

        # Build algorithm
        ppo_params = validate_algorithm_params(config.training.algorithm, config.training.algorithm_params)
        assert isinstance(ppo_params, PPOParams)
        self.ppo = PPOAlgorithm(ppo_params, self.model)

        # VecEnv (injected or created)
        if vecenv is not None:
            self.vecenv = vecenv
        else:
            from shogi_gym import VecEnv
            self.vecenv = VecEnv(
                num_envs=config.training.num_games,
                max_ply=config.training.max_ply,
            )

        self.num_envs = config.training.num_games

        # Rollout buffer
        self.buffer = RolloutBuffer(
            num_envs=self.num_envs,
            obs_shape=(46, 9, 9),
            action_space=13527,
        )

        # Move history tracking (Python-side)
        self.move_histories: list[list[dict[str, Any]]] = [[] for _ in range(self.num_envs)]

        # Snapshot pacing
        self.moves_per_minute = config.display.moves_per_minute
        self._last_snapshot_time = 0.0

        # Training state
        self.epoch = 0
        self.global_step = 0
        self._last_heartbeat = time.monotonic()

        # Check for resume
        self._check_resume()

    def _check_resume(self) -> None:
        """Check if we should resume from a previous run."""
        state = read_training_state(self.db_path)
        if state is not None and state["checkpoint_path"]:
            checkpoint_path = Path(state["checkpoint_path"])
            if checkpoint_path.exists():
                logger.warning("Resuming from checkpoint: %s (epoch %d)", checkpoint_path, state["current_epoch"])
                meta = load_checkpoint(checkpoint_path, self.model, self.ppo.optimizer)
                self.epoch = meta["epoch"]
                self.global_step = meta["step"]
                return

        # Fresh run — write initial training state
        write_training_state(self.db_path, {
            "config_json": json.dumps({
                "training": {
                    "num_games": self.config.training.num_games,
                    "algorithm": self.config.training.algorithm,
                },
                "model": {
                    "architecture": self.config.model.architecture,
                },
            }),
            "display_name": self.config.model.display_name,
            "model_arch": self.config.model.architecture,
            "algorithm_name": self.config.training.algorithm,
            "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

    def run(self, num_epochs: int, steps_per_epoch: int) -> None:
        """Run the training loop for a fixed number of epochs."""
        reset_result = self.vecenv.reset()
        obs = torch.from_numpy(np.array(reset_result.observations))
        legal_masks = torch.from_numpy(np.array(reset_result.legal_masks))

        start_epoch = self.epoch
        for epoch_i in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch_i
            win_count = 0

            for step_i in range(steps_per_epoch):
                self.global_step += 1

                # Select actions
                actions, log_probs, values = self.ppo.select_actions(obs, legal_masks)

                # Step environment
                action_list = actions.tolist()
                step_result = self.vecenv.step(action_list)

                rewards = torch.from_numpy(np.array(step_result.rewards))
                terminated = torch.from_numpy(np.array(step_result.terminated))
                truncated = torch.from_numpy(np.array(step_result.truncated))
                dones = terminated | truncated

                # Track wins (reward > 0 means the mover won)
                win_count += int((rewards > 0).sum().item())

                # Update move histories
                for env_i in range(self.num_envs):
                    self.move_histories[env_i].append({
                        "action": action_list[env_i],
                        "notation": f"a{action_list[env_i]}",
                    })
                    if dones[env_i]:
                        self.move_histories[env_i] = []

                # Store transition
                self.buffer.add(obs, actions, log_probs, values, rewards, dones, legal_masks)

                # Next observation
                obs = torch.from_numpy(np.array(step_result.observations))
                legal_masks = torch.from_numpy(np.array(step_result.legal_masks))

                # Snapshot writing (paced)
                self._maybe_write_snapshots()

                # Heartbeat
                self._maybe_update_heartbeat()

            # End of epoch: PPO update
            with torch.no_grad():
                _, next_values = self.model(obs)
                next_values = next_values.squeeze(-1)

            losses = self.ppo.update(self.buffer, next_values)

            # Compute metrics
            ep_completed = getattr(self.vecenv, "episodes_completed", 0)
            metrics = {
                "epoch": epoch_i,
                "step": self.global_step,
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "entropy": losses["entropy"],
                "gradient_norm": losses["gradient_norm"],
                "win_rate": win_count / max(ep_completed, 1) if ep_completed else None,
                "draw_rate": getattr(self.vecenv, "draw_rate", None),
                "truncation_rate": self.vecenv.truncation_rate() if hasattr(self.vecenv, "truncation_rate") else None,
                "avg_episode_length": self.vecenv.mean_episode_length() if hasattr(self.vecenv, "mean_episode_length") else None,
                "episodes_completed": ep_completed,
            }
            write_metrics(self.db_path, metrics)

            # Reset VecEnv stats for next epoch
            if hasattr(self.vecenv, "reset_stats"):
                self.vecenv.reset_stats()

            logger.info(
                "Epoch %d | step %d | policy_loss=%.4f value_loss=%.4f entropy=%.4f",
                epoch_i, self.global_step,
                losses["policy_loss"], losses["value_loss"], losses["entropy"],
            )

            # Checkpoint
            if (epoch_i + 1) % self.config.training.checkpoint_interval == 0:
                ckpt_path = Path(self.config.training.checkpoint_dir) / f"epoch_{epoch_i:05d}.pt"
                save_checkpoint(ckpt_path, self.model, self.ppo.optimizer, epoch_i + 1, self.global_step)
                update_training_progress(self.db_path, epoch_i + 1, self.global_step, str(ckpt_path))
                logger.info("Checkpoint saved: %s", ckpt_path)

    def _maybe_write_snapshots(self) -> None:
        """Write game snapshots if enough time has elapsed per moves_per_minute."""
        if self.moves_per_minute <= 0:
            return

        now = time.monotonic()
        interval = 60.0 / self.moves_per_minute
        if now - self._last_snapshot_time < interval:
            return
        self._last_snapshot_time = now

        if hasattr(self.vecenv, "get_spectator_data"):
            spectator_data = self.vecenv.get_spectator_data()
            snapshots = []
            for i, game_data in enumerate(spectator_data):
                snapshots.append({
                    "game_id": i,
                    "board_json": json.dumps(game_data.get("board", [])),
                    "hands_json": json.dumps(game_data.get("hands", {})),
                    "current_player": game_data.get("current_player", "black"),
                    "ply": game_data.get("ply", 0),
                    "is_over": int(game_data.get("is_over", False)),
                    "result": game_data.get("result", "in_progress"),
                    "sfen": game_data.get("sfen", ""),
                    "in_check": int(game_data.get("in_check", False)),
                    "move_history_json": json.dumps(self.move_histories[i]),
                })
            write_game_snapshots(self.db_path, snapshots)

    def _maybe_update_heartbeat(self) -> None:
        """Update heartbeat every 10 seconds."""
        now = time.monotonic()
        if now - self._last_heartbeat >= 10.0:
            self._last_heartbeat = now
            update_heartbeat(self.db_path)


def main() -> None:
    """CLI entry point: keisei-train."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Keisei training loop")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=256, help="Steps per epoch")
    args = parser.parse_args()

    config = load_config(args.config)
    loop = TrainingLoop(config)
    loop.run(num_epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_loop.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS across all test files.

- [ ] **Step 6: Commit**

```bash
git add keisei/training/loop.py tests/test_loop.py
git commit -m "feat: training loop orchestrator with VecEnv, PPO, SQLite, checkpointing"
```

---

### Task 10: Final Integration Verification

**Files:** None new — this task verifies everything works together.

- [ ] **Step 1: Run full test suite with coverage**

Run: `uv run pytest -v --tb=short`
Expected: All tests pass. Count should be approximately: 6 (config) + 11 (db) + 10 (models) + 9 (registries) + 7 (ppo) + 3 (checkpoint) + 3 (loop) = ~49 tests.

- [ ] **Step 2: Run ruff lint**

Run: `uv run ruff check keisei/ tests/`
Expected: No errors (or only minor style issues to fix).

- [ ] **Step 3: Verify CLI entry point**

Run: `uv run keisei-train --help`
Expected: Shows argparse help with `--config`, `--epochs`, `--steps-per-epoch`.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore: lint fixes and final integration verification"
```

---

## Summary

| Task | Component | Tests | Dependencies |
|------|-----------|-------|-------------|
| 1 | Scaffolding + deps | 0 | None |
| 2 | Config loading | 6 | Task 1 |
| 3 | SQLite DB layer | 11 | Task 1 |
| 4 | BaseModel + ResNet | 4 | Task 1 |
| 5 | MLP + Transformer | 6 | Task 4 |
| 6 | Model + Algorithm registries | 9 | Tasks 4, 5 |
| 7 | PPO (GAE, buffer, update) | 7 | Task 4 |
| 8 | Checkpointing | 3 | Task 4 |
| 9 | Training loop | 3 | Tasks 2, 3, 6, 7, 8 |
| 10 | Integration verification | 0 | All |

Tasks 2-5 can run in parallel after Task 1. Tasks 6-8 have light dependencies. Task 9 brings everything together.
