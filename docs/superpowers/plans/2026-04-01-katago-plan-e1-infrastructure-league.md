# KataGo Plan E-1: Infrastructure & League Core

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundational infrastructure for pipeline consolidation and the opponent league: extract shared utilities, extend config/DB, create the value-head adapter pattern, add contract types to the model registry, and implement OpponentPool with Elo tracking.

**Architecture:** New standalone modules (`gae.py`, `league.py`) plus extensions to existing config, DB, and registry. No modifications to the training loop itself — that's Plan E-2. All new code is independently testable.

**Tech Stack:** Python 3.13, PyTorch, SQLite, dataclasses. Tests via `uv run pytest`.

**Dependencies:** Requires Plans A-D complete. Verify before starting:

```bash
uv run python -c "from keisei.training.katago_loop import KataGoTrainingLoop; from keisei.training.katago_ppo import KataGoPPOAlgorithm; from keisei.training.models.se_resnet import SEResNetModel; print('Plans A-D ready')"
```

**Spec reference:** `docs/superpowers/specs/2026-04-01-plan-e-league-consolidation-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `keisei/training/gae.py` | `compute_gae()` extracted from `ppo.py` |
| Create | `keisei/training/value_adapter.py` | `ValueHeadAdapter` ABC, `ScalarValueAdapter`, `MultiHeadValueAdapter` |
| Create | `keisei/training/league.py` | `OpponentEntry`, `OpponentPool`, `OpponentSampler`, `compute_elo_update()` |
| Modify | `keisei/training/katago_ppo.py` | Import `compute_gae` from new `gae.py` location |
| Modify | `keisei/training/model_registry.py` | Add contract type (`"scalar"` / `"multi_head"`) and `obs_channels` per architecture |
| Modify | `keisei/config.py` | Add `LeagueConfig`, `DemonstratorConfig` dataclasses and `[league]`/`[demonstrator]` TOML sections |
| Modify | `keisei/db.py` | Schema v1→v2 migration, `league_entries`/`league_results` tables, `game_type`/`demo_slot` columns |
| Create | `tests/test_gae.py` | Regression tests for extracted `compute_gae()` |
| Create | `tests/test_value_adapter.py` | Value-head adapter unit tests |
| Create | `tests/test_league.py` | OpponentPool, OpponentSampler, Elo tests |
| Create | `tests/test_league_config.py` | Config extension tests |
| Create | `tests/test_db_migration.py` | Schema migration tests |

---

### Task 1: Extract `compute_gae` to `gae.py`

**Files:**
- Create: `keisei/training/gae.py`
- Modify: `keisei/training/katago_ppo.py`
- Create: `tests/test_gae.py`

- [ ] **Step 1: Write the regression test**

```python
# tests/test_gae.py
"""Regression tests for compute_gae — extracted from ppo.py."""

import torch
import pytest

from keisei.training.gae import compute_gae


class TestComputeGAE:
    def test_single_step_no_done(self):
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.5])
        dones = torch.tensor([False])
        next_value = torch.tensor(0.3)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        assert advantages.shape == (1,)
        # delta = reward + gamma * next_value * (1-done) - value
        # delta = 1.0 + 0.99 * 0.3 * 1.0 - 0.5 = 0.797
        assert abs(advantages[0].item() - 0.797) < 1e-3

    def test_episode_boundary_resets(self):
        rewards = torch.tensor([1.0, 2.0])
        values = torch.tensor([0.5, 0.5])
        dones = torch.tensor([True, False])
        next_value = torch.tensor(0.3)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        assert advantages.shape == (2,)
        # Step 0 is terminal: delta = 1.0 + 0 - 0.5 = 0.5
        assert abs(advantages[0].item() - 0.5) < 1e-3

    def test_multi_step_accumulation(self):
        """Verify GAE recursive accumulation over a non-terminal trajectory.

        Hand-computed reference for 3 steps, gamma=0.99, lam=0.95:
        Step 2: delta_2 = r2 + gamma*next_val*(1-d2) - v2 = 3.0 + 0.99*0.0*1.0 - 0.5 = 2.5
                gae_2 = 2.5
        Step 1: delta_1 = r1 + gamma*v2*(1-d1) - v1 = 2.0 + 0.99*0.5*1.0 - 0.5 = 1.995
                gae_1 = 1.995 + 0.99*0.95*1.0*2.5 = 1.995 + 2.35125 = 4.34625
        Step 0: delta_0 = r0 + gamma*v1*(1-d0) - v0 = 1.0 + 0.99*0.5*1.0 - 0.5 = 0.995
                gae_0 = 0.995 + 0.99*0.95*1.0*4.34625 = 0.995 + 4.086... = 5.081...
        """
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        dones = torch.zeros(3, dtype=torch.bool)
        next_value = torch.tensor(0.0)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        assert advantages.shape == (3,)
        assert abs(advantages[2].item() - 2.5) < 1e-3
        assert abs(advantages[1].item() - 4.34625) < 1e-3
        assert abs(advantages[0].item() - 5.081) < 1e-2  # accumulated error tolerance

    def test_output_dtype_and_device(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        dones = torch.zeros(3, dtype=torch.bool)
        next_value = torch.tensor(0.0)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        assert advantages.dtype == torch.float32
        assert advantages.shape == (3,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gae.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'keisei.training.gae'`

- [ ] **Step 3: Write the implementation**

Copy `compute_gae` from `keisei/training/ppo.py` (lines 12-32) to the new file:

```python
# keisei/training/gae.py
"""Generalized Advantage Estimation — shared utility for PPO variants."""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute GAE advantages for a single environment's trajectory.

    Args:
        rewards: (T,) per-step rewards
        values: (T,) value estimates at each step
        dones: (T,) episode termination flags
        next_value: scalar value estimate for the state after the last step
        gamma: discount factor
        lam: GAE lambda (bias-variance tradeoff)

    Returns:
        (T,) advantage estimates
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
```

- [ ] **Step 4: Update `katago_ppo.py` import**

In `keisei/training/katago_ppo.py`, change:
```python
from keisei.training.ppo import compute_gae
```
to:
```python
from keisei.training.gae import compute_gae
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_gae.py -v`
Expected: PASS

Note: `tests/test_katago_ppo.py` (from Plan B) should also pass after the import update.
If Plan B tests exist, verify: `uv run pytest tests/test_katago_ppo.py -v`

- [ ] **Step 6: Commit**

```bash
git add keisei/training/gae.py tests/test_gae.py keisei/training/katago_ppo.py
git commit -m "refactor: extract compute_gae to shared gae.py module"
```

---

### Task 2: Config Extensions

**Files:**
- Modify: `keisei/config.py`
- Create: `tests/test_league_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_league_config.py
"""Tests for league and demonstrator config extensions."""

import tempfile
from pathlib import Path

import pytest

from keisei.config import load_config, LeagueConfig, DemonstratorConfig


LEAGUE_TOML = """
[model]
display_name = "Test"
architecture = "se_resnet"

[model.params]
num_blocks = 2
channels = 32
se_reduction = 8
global_pool_channels = 16
policy_channels = 8
value_fc_size = 32
score_fc_size = 16
obs_channels = 50

[training]
algorithm = "katago_ppo"
num_games = 2
max_ply = 50
checkpoint_interval = 10
checkpoint_dir = "checkpoints/"

[training.algorithm_params]
learning_rate = 0.0002
score_normalization = 76.0
grad_clip = 1.0

[display]
moves_per_minute = 0
db_path = "test.db"

[league]
max_pool_size = 20
snapshot_interval = 10
epochs_per_seat = 50
historical_ratio = 0.8
current_best_ratio = 0.2
initial_elo = 1000
elo_k_factor = 32
elo_floor = 500

[demonstrator]
num_games = 3
auto_matchup = true
moves_per_minute = 60
device = "cpu"
"""


def test_load_config_with_league(tmp_path):
    toml_file = tmp_path / "league.toml"
    toml_file.write_text(LEAGUE_TOML)
    config = load_config(toml_file)
    assert config.league is not None
    assert config.league.max_pool_size == 20
    assert config.league.elo_floor == 500
    assert config.league.historical_ratio + config.league.current_best_ratio == 1.0


def test_load_config_with_demonstrator(tmp_path):
    toml_file = tmp_path / "demo.toml"
    toml_file.write_text(LEAGUE_TOML)
    config = load_config(toml_file)
    assert config.demonstrator is not None
    assert config.demonstrator.num_games == 3
    assert config.demonstrator.device == "cpu"


def test_league_config_defaults():
    lc = LeagueConfig()
    assert lc.max_pool_size == 20
    assert lc.snapshot_interval == 10
    assert lc.epochs_per_seat == 50
    assert lc.elo_floor == 500


def test_demonstrator_config_defaults():
    dc = DemonstratorConfig()
    assert dc.num_games == 3
    assert dc.device == "cuda"


def test_load_config_without_league_section(tmp_path):
    """Config without [league] should get None."""
    toml = LEAGUE_TOML.split("[league]")[0] + "\n"
    toml = toml.split("[demonstrator]")[0] + "\n"
    toml_file = tmp_path / "noleague.toml"
    toml_file.write_text(toml)
    config = load_config(toml_file)
    assert config.league is None
    assert config.demonstrator is None


def test_league_ratio_validation(tmp_path):
    """historical_ratio + current_best_ratio must equal 1.0."""
    bad_toml = LEAGUE_TOML.replace("historical_ratio = 0.8", "historical_ratio = 0.6")
    toml_file = tmp_path / "badratio.toml"
    toml_file.write_text(bad_toml)
    with pytest.raises(ValueError, match="ratio"):
        load_config(toml_file)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_league_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'LeagueConfig'`

- [ ] **Step 3: Write the implementation**

Add to `keisei/config.py`:

```python
@dataclass(frozen=True)
class LeagueConfig:
    max_pool_size: int = 20
    snapshot_interval: int = 10
    epochs_per_seat: int = 50
    historical_ratio: float = 0.8
    current_best_ratio: float = 0.2
    initial_elo: float = 1000.0
    elo_k_factor: float = 32.0
    elo_floor: float = 500.0


@dataclass(frozen=True)
class DemonstratorConfig:
    num_games: int = 3
    auto_matchup: bool = True
    moves_per_minute: int = 60
    device: str = "cuda"
```

Update `AppConfig`:

```python
@dataclass(frozen=True)
class AppConfig:
    training: TrainingConfig
    display: DisplayConfig
    model: ModelConfig
    league: LeagueConfig | None = None
    demonstrator: DemonstratorConfig | None = None
```

**Also update the architecture and algorithm whitelists** (Plans B/C added these to their registries, but `config.py` maintains a separate validation set):

```python
VALID_ARCHITECTURES = {"resnet", "mlp", "transformer", "se_resnet"}
VALID_ALGORITHMS = {"ppo", "katago_ppo"}
```

Update `load_config()` to parse `[league]` and `[demonstrator]` sections:

```python
# After existing parsing, before return:
league_config = None
if "league" in raw:
    league_data = raw["league"]
    league_config = LeagueConfig(**league_data)
    if abs(league_config.historical_ratio + league_config.current_best_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"League ratio sum must be 1.0, got "
            f"{league_config.historical_ratio} + {league_config.current_best_ratio} = "
            f"{league_config.historical_ratio + league_config.current_best_ratio}"
        )

demo_config = None
if "demonstrator" in raw:
    demo_config = DemonstratorConfig(**raw["demonstrator"])

return AppConfig(
    training=training_config,
    display=display_config,
    model=model_config,
    league=league_config,
    demonstrator=demo_config,
)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_league_config.py -v`
Expected: PASS

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (no regressions — old tests don't pass [league] section)

- [ ] **Step 5: Commit**

```bash
git add keisei/config.py tests/test_league_config.py
git commit -m "feat: add LeagueConfig and DemonstratorConfig to config system"
```

---

### Task 3: DB Schema Migration

**Files:**
- Modify: `keisei/db.py`
- Create: `tests/test_db_migration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db_migration.py
"""Tests for DB schema migration v1 → v2."""

import sqlite3
from pathlib import Path

import pytest

from keisei.db import init_db


def _get_schema_version(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def _get_table_columns(db_path: str, table: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cursor.fetchall()]
    finally:
        conn.close()


def _table_exists(db_path: str, table: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


class TestFreshDB:
    def test_creates_league_tables(self, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        assert _table_exists(db_path, "league_entries")
        assert _table_exists(db_path, "league_results")

    def test_game_snapshots_has_new_columns(self, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        cols = _get_table_columns(db_path, "game_snapshots")
        assert "game_type" in cols
        assert "demo_slot" in cols

    def test_schema_version_is_2(self, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        assert _get_schema_version(db_path) == 2


class TestMigrationV1ToV2:
    def _create_v1_db(self, db_path: str) -> None:
        """Create a minimal v1 database to test migration."""
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version VALUES (1)")
        conn.execute("""
            CREATE TABLE game_snapshots (
                game_id INTEGER PRIMARY KEY,
                board_json TEXT, hands_json TEXT, current_player TEXT,
                ply INTEGER, is_over INTEGER, result TEXT, sfen TEXT,
                in_check INTEGER, move_history_json TEXT,
                value_estimate REAL DEFAULT 0.0,
                updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            )
        """)
        conn.execute("""
            CREATE TABLE metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_json TEXT, recorded_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE training_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                state_json TEXT, updated_at TEXT
            )
        """)
        # Insert a v1 game snapshot to verify migration preserves data
        conn.execute(
            "INSERT INTO game_snapshots (game_id, ply, is_over) VALUES (1, 10, 0)"
        )
        conn.commit()
        conn.close()

    def test_migration_adds_columns(self, tmp_path):
        db_path = str(tmp_path / "v1.db")
        self._create_v1_db(db_path)
        init_db(db_path)
        cols = _get_table_columns(db_path, "game_snapshots")
        assert "game_type" in cols
        assert "demo_slot" in cols

    def test_migration_creates_league_tables(self, tmp_path):
        db_path = str(tmp_path / "v1.db")
        self._create_v1_db(db_path)
        init_db(db_path)
        assert _table_exists(db_path, "league_entries")
        assert _table_exists(db_path, "league_results")

    def test_migration_preserves_existing_data(self, tmp_path):
        db_path = str(tmp_path / "v1.db")
        self._create_v1_db(db_path)
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT ply FROM game_snapshots WHERE game_id=1").fetchone()
        conn.close()
        assert row[0] == 10

    def test_migration_sets_default_game_type(self, tmp_path):
        db_path = str(tmp_path / "v1.db")
        self._create_v1_db(db_path)
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT game_type FROM game_snapshots WHERE game_id=1"
        ).fetchone()
        conn.close()
        assert row[0] == "live"

    def test_schema_version_updated_to_2(self, tmp_path):
        db_path = str(tmp_path / "v1.db")
        self._create_v1_db(db_path)
        init_db(db_path)
        assert _get_schema_version(db_path) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db_migration.py -v`
Expected: FAIL — `league_entries` table not created, `game_type` column missing

- [ ] **Step 3: Write the implementation**

In `keisei/db.py`, update `SCHEMA_VERSION` and add migration logic:

```python
SCHEMA_VERSION = 2
```

Add the new table DDL strings (alongside existing ones):

```python
_LEAGUE_ENTRIES_DDL = """
CREATE TABLE IF NOT EXISTS league_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    architecture    TEXT NOT NULL,
    model_params    TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL,
    elo_rating      REAL NOT NULL DEFAULT 1000.0,
    created_epoch   INTEGER NOT NULL,
    games_played    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
)
"""

_LEAGUE_RESULTS_DDL = """
CREATE TABLE IF NOT EXISTS league_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch           INTEGER NOT NULL,
    learner_id      INTEGER NOT NULL REFERENCES league_entries(id),
    opponent_id     INTEGER NOT NULL REFERENCES league_entries(id),
    wins            INTEGER NOT NULL,
    losses          INTEGER NOT NULL,
    draws           INTEGER NOT NULL,
    recorded_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
)
"""
```

Update `init_db()` to handle migration:

```python
def init_db(db_path: str) -> None:
    conn = _connect(db_path)
    try:
        # Check current version
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        current_version = row[0] if row else 0

        if current_version == 0:
            # Fresh database — create all tables at v2
            conn.execute(_METRICS_DDL)
            conn.execute(_GAME_SNAPSHOTS_V2_DDL)  # includes game_type, demo_slot
            conn.execute(_TRAINING_STATE_DDL)
            conn.execute(_LEAGUE_ENTRIES_DDL)
            conn.execute(_LEAGUE_RESULTS_DDL)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_league_results_epoch ON league_results(epoch)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_league_entries_elo ON league_entries(elo_rating)")
            conn.execute("INSERT INTO schema_version VALUES (2)")
        elif current_version == 1:
            # Migrate v1 → v2
            _migrate_v1_to_v2(conn)
        # else: already at v2, nothing to do

        conn.commit()
    finally:
        conn.close()


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Migrate schema from v1 to v2: add league tables and game_snapshots columns."""
    # Add new columns — guarded against re-execution (idempotent).
    # SQLite has no ADD COLUMN IF NOT EXISTS, so check PRAGMA first.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(game_snapshots)")}
    if "game_type" not in existing_cols:
        conn.execute("ALTER TABLE game_snapshots ADD COLUMN game_type TEXT NOT NULL DEFAULT 'live'")
    if "demo_slot" not in existing_cols:
        conn.execute("ALTER TABLE game_snapshots ADD COLUMN demo_slot INTEGER")

    # Create league tables
    conn.execute(_LEAGUE_ENTRIES_DDL)
    conn.execute(_LEAGUE_RESULTS_DDL)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_league_results_epoch ON league_results(epoch)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_league_entries_elo ON league_entries(elo_rating)")

    # Update version
    conn.execute("UPDATE schema_version SET version = 2")
```

The `_GAME_SNAPSHOTS_V2_DDL` is the original DDL with the two new columns added:

```python
_GAME_SNAPSHOTS_V2_DDL = """
CREATE TABLE IF NOT EXISTS game_snapshots (
    game_id           INTEGER PRIMARY KEY,
    board_json        TEXT,
    hands_json        TEXT,
    current_player    TEXT,
    ply               INTEGER,
    is_over           INTEGER,
    result            TEXT,
    sfen              TEXT,
    in_check          INTEGER,
    move_history_json TEXT,
    value_estimate    REAL DEFAULT 0.0,
    game_type         TEXT NOT NULL DEFAULT 'live',
    demo_slot         INTEGER,
    updated_at        TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
)
"""
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_db_migration.py -v`
Expected: PASS

Run: `uv run pytest tests/test_db.py -v`
Expected: PASS (no regressions)

- [ ] **Step 5: Commit**

```bash
git add keisei/db.py tests/test_db_migration.py
git commit -m "feat: DB schema migration v1→v2 with league tables and game_type column"
```

---

### Task 4: Value-Head Adapter

**Files:**
- Create: `keisei/training/value_adapter.py`
- Create: `tests/test_value_adapter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_value_adapter.py
"""Tests for value-head adapters (scalar vs multi-head)."""

import torch
import torch.nn.functional as F
import pytest

from keisei.training.value_adapter import (
    ScalarValueAdapter,
    MultiHeadValueAdapter,
    get_value_adapter,
)


class TestScalarValueAdapter:
    def test_scalar_value_output(self):
        adapter = ScalarValueAdapter()
        # Simulate BaseModel output: (policy_logits, value)
        value = torch.tensor([[0.5], [-0.3], [0.8]])  # (3, 1)
        scalar = adapter.scalar_value_from_output(value)
        assert scalar.shape == (3,)
        assert torch.allclose(scalar, torch.tensor([0.5, -0.3, 0.8]))

    def test_scalar_value_loss(self):
        adapter = ScalarValueAdapter()
        value = torch.tensor([[0.5], [-0.3]], requires_grad=True)
        returns = torch.tensor([0.8, -0.1])
        loss = adapter.compute_value_loss(value, returns, value_cats=None, score_targets=None)
        assert loss.item() > 0
        loss.backward()
        assert value.grad is not None


class TestMultiHeadValueAdapter:
    def test_scalar_value_output(self):
        adapter = MultiHeadValueAdapter()
        # Simulate KataGoOutput.value_logits: (batch, 3) — W/D/L
        value_logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        scalar = adapter.scalar_value_from_output(value_logits)
        assert scalar.shape == (2,)
        # P(W) - P(L): first sample should be positive, second negative
        assert scalar[0] > 0
        assert scalar[1] < 0

    def test_multi_head_value_loss(self):
        adapter = MultiHeadValueAdapter()
        value_logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]], requires_grad=True)
        value_cats = torch.tensor([0, 2])  # W, L
        score_pred = torch.tensor([[0.5], [-0.3]], requires_grad=True)
        score_targets = torch.tensor([0.013, -0.013])
        loss = adapter.compute_value_loss(
            value_logits, returns=None,
            value_cats=value_cats, score_targets=score_targets,
            score_pred=score_pred,
        )
        assert loss.item() > 0
        loss.backward()
        assert value_logits.grad is not None

    def test_ignore_index_for_non_terminal(self):
        adapter = MultiHeadValueAdapter()
        value_logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
        # -1 = non-terminal, should be ignored
        value_cats = torch.tensor([-1, 1])
        score_pred = torch.tensor([[0.0], [0.0]])
        score_targets = torch.tensor([0.0, 0.0])
        loss = adapter.compute_value_loss(
            value_logits, returns=None,
            value_cats=value_cats, score_targets=score_targets,
            score_pred=score_pred,
        )
        # Only sample 1 contributes to loss (sample 0 has ignore_index=-1)
        assert loss.item() > 0


class TestGetValueAdapter:
    def test_returns_scalar_for_base_model(self):
        from keisei.training.models.base import BaseModel
        # Can't instantiate ABC, so test via isinstance check
        adapter = get_value_adapter(model_contract="scalar")
        assert isinstance(adapter, ScalarValueAdapter)

    def test_returns_multi_head_for_katago(self):
        adapter = get_value_adapter(model_contract="multi_head")
        assert isinstance(adapter, MultiHeadValueAdapter)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_value_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# keisei/training/value_adapter.py
"""Value-head adapters for dual-contract model support.

Encapsulates loss computation differences between scalar-value models
(BaseModel) and multi-head W/D/L models (KataGoBaseModel) so the
unified training loop never branches on model type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class ValueHeadAdapter(ABC):
    """Interface for value-head loss computation and scalar projection."""

    @abstractmethod
    def scalar_value_from_output(self, value_output: torch.Tensor) -> torch.Tensor:
        """Project model's value output to a scalar (batch,) for GAE."""
        ...

    @abstractmethod
    def compute_value_loss(
        self,
        value_output: torch.Tensor,
        returns: torch.Tensor | None,
        value_cats: torch.Tensor | None,
        score_targets: torch.Tensor | None,
        score_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute value loss appropriate for the model contract."""
        ...


class ScalarValueAdapter(ValueHeadAdapter):
    """For BaseModel: tanh-activated scalar value, MSE loss vs returns."""

    def scalar_value_from_output(self, value_output: torch.Tensor) -> torch.Tensor:
        # value_output is (batch, 1), squeeze to (batch,)
        return value_output.squeeze(-1)

    def compute_value_loss(
        self,
        value_output: torch.Tensor,
        returns: torch.Tensor | None,
        value_cats: torch.Tensor | None = None,
        score_targets: torch.Tensor | None = None,
        score_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if returns is None:
            raise ValueError("ScalarValueAdapter requires returns")
        return F.mse_loss(value_output.squeeze(-1), returns)


class MultiHeadValueAdapter(ValueHeadAdapter):
    """For KataGoBaseModel: W/D/L cross-entropy + score MSE."""

    def __init__(self, lambda_value: float = 1.5, lambda_score: float = 0.02) -> None:
        self.lambda_value = lambda_value
        self.lambda_score = lambda_score

    def scalar_value_from_output(self, value_output: torch.Tensor) -> torch.Tensor:
        # value_output is value_logits (batch, 3) — W/D/L
        value_probs = F.softmax(value_output, dim=-1)
        return value_probs[:, 0] - value_probs[:, 2]  # P(W) - P(L)

    def compute_value_loss(
        self,
        value_output: torch.Tensor,
        returns: torch.Tensor | None = None,
        value_cats: torch.Tensor | None = None,
        score_targets: torch.Tensor | None = None,
        score_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if value_cats is None:
            raise ValueError("MultiHeadValueAdapter requires value_cats")
        if score_targets is None:
            raise ValueError("MultiHeadValueAdapter requires score_targets")
        if score_pred is None:
            raise ValueError("MultiHeadValueAdapter requires score_pred")

        value_loss = F.cross_entropy(value_output, value_cats, ignore_index=-1)
        score_loss = F.mse_loss(score_pred.squeeze(-1), score_targets)
        return self.lambda_value * value_loss + self.lambda_score * score_loss


def get_value_adapter(model_contract: str) -> ValueHeadAdapter:
    """Return the appropriate adapter for a model contract type."""
    if model_contract == "scalar":
        return ScalarValueAdapter()
    elif model_contract == "multi_head":
        return MultiHeadValueAdapter()
    else:
        raise ValueError(f"Unknown model contract: {model_contract}")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_value_adapter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/value_adapter.py tests/test_value_adapter.py
git commit -m "feat: add ValueHeadAdapter for dual-contract model support"
```

---

### Task 5: Model Registry Contract Types

**Files:**
- Modify: `keisei/training/model_registry.py`
- Modify: `tests/test_registries.py` (if existing tests need updating)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_registries.py` (or create a new section):

```python
# Add to tests/test_registries.py or create tests/test_registry_contracts.py
from keisei.training.model_registry import get_model_contract, get_obs_channels


class TestModelContractTypes:
    def test_resnet_is_scalar(self):
        assert get_model_contract("resnet") == "scalar"

    def test_mlp_is_scalar(self):
        assert get_model_contract("mlp") == "scalar"

    def test_transformer_is_scalar(self):
        assert get_model_contract("transformer") == "scalar"

    def test_se_resnet_is_multi_head(self):
        assert get_model_contract("se_resnet") == "multi_head"

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError):
            get_model_contract("nonexistent")

    def test_obs_channels_scalar(self):
        assert get_obs_channels("resnet") == 50  # post-Plan-A: all models accept 50

    def test_obs_channels_multi_head(self):
        assert get_obs_channels("se_resnet") == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registries.py::TestModelContractTypes -v`
Expected: FAIL — `get_model_contract` not found

- [ ] **Step 3: Write the implementation**

Update `keisei/training/model_registry.py` to extend the registry with contract and obs_channels:

```python
# Extended registry: (model_cls, params_cls, contract, obs_channels)
_REGISTRY: dict[str, tuple[type, type, str, int]] = {
    "resnet": (ResNetModel, ResNetParams, "scalar", 50),
    "mlp": (MLPModel, MLPParams, "scalar", 50),
    "transformer": (TransformerModel, TransformerParams, "scalar", 50),
    "se_resnet": (SEResNetModel, SEResNetParams, "multi_head", 50),
}


def get_model_contract(architecture: str) -> str:
    """Return the value-head contract type for an architecture."""
    if architecture not in _REGISTRY:
        raise ValueError(f"Unknown architecture '{architecture}'")
    return _REGISTRY[architecture][2]


def get_obs_channels(architecture: str) -> int:
    """Return the expected observation channels for an architecture."""
    if architecture not in _REGISTRY:
        raise ValueError(f"Unknown architecture '{architecture}'")
    return _REGISTRY[architecture][3]
```

Update `build_model` and `validate_model_params` to use the new tuple structure (index [0] for model_cls, [1] for params_cls).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_registries.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/model_registry.py tests/test_registries.py
git commit -m "feat: add contract type and obs_channels to model registry"
```

---

### Task 6: OpponentPool and OpponentEntry

**Files:**
- Create: `keisei/training/league.py`
- Create: `tests/test_league.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_league.py
"""Tests for the opponent league: pool, sampler, Elo."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from keisei.db import init_db
from keisei.training.league import (
    OpponentEntry,
    OpponentPool,
    OpponentSampler,
    compute_elo_update,
)


@pytest.fixture
def league_db(tmp_path):
    db_path = str(tmp_path / "league.db")
    init_db(db_path)
    return db_path


@pytest.fixture
def league_dir(tmp_path):
    d = tmp_path / "checkpoints" / "league"
    d.mkdir(parents=True)
    return d


class TestOpponentEntry:
    def test_from_db_row(self):
        row = (1, "resnet", '{"hidden_size": 16}', "/path/to/ckpt.pt",
               1000.0, 10, 5, "2026-04-01T00:00:00Z")
        entry = OpponentEntry.from_db_row(row)
        assert entry.id == 1
        assert entry.architecture == "resnet"
        assert entry.model_params == {"hidden_size": 16}
        assert entry.elo_rating == 1000.0


class TestOpponentPool:
    def test_add_snapshot(self, league_db, league_dir, tmp_path):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=5)
        # Create a fake model and save a checkpoint
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(
            model=model,
            architecture="resnet",
            model_params={"hidden_size": 16},
            epoch=10,
        )
        entries = pool.list_entries()
        assert len(entries) == 1
        assert entries[0].architecture == "resnet"
        assert entries[0].created_epoch == 10
        assert Path(entries[0].checkpoint_path).exists()

    def test_eviction_respects_max_pool_size(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=3)
        model = torch.nn.Linear(10, 10)
        for epoch in range(5):
            pool.add_snapshot(model, "resnet", {"hidden_size": 16}, epoch=epoch)
        entries = pool.list_entries()
        assert len(entries) == 3
        # Oldest entries (epoch 0, 1) should have been evicted
        epochs = [e.created_epoch for e in entries]
        assert 0 not in epochs
        assert 1 not in epochs

    def test_eviction_skips_pinned_entries(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=2)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)
        entry_0 = pool.list_entries()[0]

        # Pin the first entry
        pool.pin(entry_0.id)
        pool.add_snapshot(model, "resnet", {}, epoch=1)
        pool.add_snapshot(model, "resnet", {}, epoch=2)

        entries = pool.list_entries()
        # Entry 0 should survive because it's pinned; entry 1 evicted instead
        epochs = [e.created_epoch for e in entries]
        assert 0 in epochs
        assert 2 in epochs

        pool.unpin(entry_0.id)

    def test_load_opponent(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=5)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {"hidden_size": 16}, epoch=5)
        entry = pool.list_entries()[0]

        # Mock build_model to return a fresh Linear
        with patch("keisei.training.league.build_model") as mock_build:
            mock_model = torch.nn.Linear(10, 10)
            mock_build.return_value = mock_model
            loaded = pool.load_opponent(entry)
            mock_build.assert_called_once_with("resnet", {"hidden_size": 16})
            # Model should be in eval mode
            assert not loaded.training

    def test_empty_pool_list(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=5)
        assert pool.list_entries() == []

    def test_update_elo(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=5)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)
        entry = pool.list_entries()[0]
        pool.update_elo(entry.id, 1050.0)
        updated = pool.list_entries()[0]
        assert updated.elo_rating == 1050.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_league.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# keisei/training/league.py
"""Opponent league: pool management, sampling, and Elo tracking."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from keisei.training.model_registry import build_model

logger = logging.getLogger(__name__)


@dataclass
class OpponentEntry:
    """A snapshot in the opponent pool."""

    id: int
    architecture: str
    model_params: dict[str, Any]
    checkpoint_path: str
    elo_rating: float
    created_epoch: int
    games_played: int
    created_at: str

    @classmethod
    def from_db_row(cls, row: tuple) -> OpponentEntry:
        return cls(
            id=row[0],
            architecture=row[1],
            model_params=json.loads(row[2]),
            checkpoint_path=row[3],
            elo_rating=row[4],
            created_epoch=row[5],
            games_played=row[6],
            created_at=row[7],
        )


class OpponentPool:
    """Manages the collection of checkpoint snapshots available as opponents."""

    def __init__(self, db_path: str, league_dir: str, max_pool_size: int = 20) -> None:
        self.db_path = db_path
        self.league_dir = Path(league_dir)
        self.league_dir.mkdir(parents=True, exist_ok=True)
        self.max_pool_size = max_pool_size
        self._pinned: set[int] = set()

    def _connect(self) -> sqlite3.Connection:
        # Thread-safe: check_same_thread=False allows access from demonstrator thread.
        # WAL mode + busy_timeout match db._connect() pragmas for consistency.
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def add_snapshot(
        self,
        model: torch.nn.Module,
        architecture: str,
        model_params: dict[str, Any],
        epoch: int,
    ) -> OpponentEntry:
        """Save a checkpoint snapshot and add it to the pool."""
        # Unwrap DataParallel to avoid "module." key prefix in state_dict
        raw_model = model.module if hasattr(model, "module") else model
        ckpt_path = self.league_dir / f"{architecture}_ep{epoch:05d}.pt"
        torch.save(raw_model.state_dict(), ckpt_path)

        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT INTO league_entries
                   (architecture, model_params, checkpoint_path, created_epoch)
                   VALUES (?, ?, ?, ?)""",
                (architecture, json.dumps(model_params), str(ckpt_path), epoch),
            )
            entry_id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()

        logger.info("Pool snapshot: %s epoch %d → %s (id=%d)",
                     architecture, epoch, ckpt_path.name, entry_id)

        self._evict_if_needed()

        entry = self._get_entry(entry_id)
        assert entry is not None
        return entry

    def _get_entry(self, entry_id: int) -> OpponentEntry | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM league_entries WHERE id = ?", (entry_id,)
            ).fetchone()
            return OpponentEntry.from_db_row(row) if row else None
        finally:
            conn.close()

    def list_entries(self) -> list[OpponentEntry]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM league_entries ORDER BY created_epoch ASC"
            ).fetchall()
            return [OpponentEntry.from_db_row(r) for r in rows]
        finally:
            conn.close()

    def _evict_if_needed(self) -> None:
        entries = self.list_entries()
        while len(entries) > self.max_pool_size:
            # Find oldest non-pinned entry
            evicted = False
            for entry in entries:
                if entry.id not in self._pinned:
                    self._delete_entry(entry)
                    entries = self.list_entries()
                    evicted = True
                    break
            if not evicted:
                logger.warning("All entries pinned, cannot evict to reach max_pool_size=%d",
                               self.max_pool_size)
                break

    def _delete_entry(self, entry: OpponentEntry) -> None:
        ckpt = Path(entry.checkpoint_path)
        if ckpt.exists():
            ckpt.unlink()
        conn = self._connect()
        try:
            conn.execute("DELETE FROM league_entries WHERE id = ?", (entry.id,))
            conn.commit()
        finally:
            conn.close()
        logger.info("Evicted pool entry id=%d (epoch %d)", entry.id, entry.created_epoch)

    def pin(self, entry_id: int) -> None:
        """Pin an entry to prevent eviction (used by DemonstratorRunner)."""
        self._pinned.add(entry_id)

    def unpin(self, entry_id: int) -> None:
        """Release a pin."""
        self._pinned.discard(entry_id)

    def load_opponent(self, entry: OpponentEntry, device: str = "cpu") -> torch.nn.Module:
        """Load an opponent model from a pool entry."""
        ckpt = Path(entry.checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint missing for pool entry id={entry.id} "
                f"(arch={entry.architecture}, epoch={entry.created_epoch}): {ckpt}"
            )
        model = build_model(entry.architecture, entry.model_params)
        state_dict = torch.load(str(ckpt), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    def update_elo(self, entry_id: int, new_elo: float) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE league_entries SET elo_rating = ? WHERE id = ?",
                (new_elo, entry_id),
            )
            conn.commit()
        finally:
            conn.close()

    def record_result(
        self, epoch: int, learner_id: int, opponent_id: int,
        wins: int, losses: int, draws: int,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO league_results
                   (epoch, learner_id, opponent_id, wins, losses, draws)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (epoch, learner_id, opponent_id, wins, losses, draws),
            )
            conn.execute(
                "UPDATE league_entries SET games_played = games_played + ? WHERE id = ?",
                (wins + losses + draws, opponent_id),
            )
            conn.commit()
        finally:
            conn.close()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_league.py -v -k "Entry or Pool"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/league.py tests/test_league.py
git commit -m "feat: add OpponentPool with snapshot, eviction, pinning, and DB persistence"
```

---

### Task 7: OpponentSampler and Elo Calculations

**Files:**
- Modify: `keisei/training/league.py`
- Modify: `tests/test_league.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_league.py`:

```python
class TestEloCalculation:
    def test_equal_elo_expected_is_half(self):
        new_a, new_b = compute_elo_update(1000.0, 1000.0, result=1.0, k=32)
        assert abs(new_a - 1016.0) < 0.1  # won: 1000 + 32*(1.0-0.5)
        assert abs(new_b - 984.0) < 0.1   # lost: 1000 + 32*(0.0-0.5)

    def test_draw_against_equal(self):
        new_a, new_b = compute_elo_update(1000.0, 1000.0, result=0.5, k=32)
        assert abs(new_a - 1000.0) < 0.1
        assert abs(new_b - 1000.0) < 0.1

    def test_upset_gives_more_elo(self):
        # Weak player (800) beats strong player (1200)
        new_a, new_b = compute_elo_update(800.0, 1200.0, result=1.0, k=32)
        # Expected ≈ 0.09 for 800-rated, so gain ≈ 32*(1.0-0.09) ≈ 29
        assert new_a > 825
        assert new_b < 1175


class TestOpponentSampler:
    def test_sample_from_pool(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        for i in range(5):
            pool.add_snapshot(model, "resnet", {}, epoch=i)

        sampler = OpponentSampler(
            pool, historical_ratio=0.8, current_best_ratio=0.2, elo_floor=500.0
        )
        entry = sampler.sample()
        assert isinstance(entry, OpponentEntry)

    def test_current_best_is_most_recent(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        for i in range(5):
            pool.add_snapshot(model, "resnet", {}, epoch=i)

        sampler = OpponentSampler(pool, historical_ratio=0.0, current_best_ratio=1.0)
        # With 100% current_best, should always return the most recent
        for _ in range(10):
            entry = sampler.sample()
            assert entry.created_epoch == 4

    def test_single_entry_returns_it(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)

        sampler = OpponentSampler(pool, historical_ratio=0.8, current_best_ratio=0.2)
        entry = sampler.sample()
        assert entry.created_epoch == 0

    def test_elo_floor_excludes_weak_from_historical(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)
        pool.add_snapshot(model, "resnet", {}, epoch=1)

        # Set epoch-0 entry below floor
        entries = pool.list_entries()
        pool.update_elo(entries[0].id, 400.0)  # below floor of 500

        sampler = OpponentSampler(
            pool, historical_ratio=1.0, current_best_ratio=0.0, elo_floor=500.0
        )
        # Historical sampling should only return epoch-1 (above floor)
        for _ in range(10):
            entry = sampler.sample()
            assert entry.created_epoch == 1

    def test_all_below_floor_falls_back(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)

        entries = pool.list_entries()
        pool.update_elo(entries[0].id, 400.0)

        sampler = OpponentSampler(
            pool, historical_ratio=1.0, current_best_ratio=0.0, elo_floor=500.0
        )
        # All below floor — should still return something (fallback to any entry)
        entry = sampler.sample()
        assert entry is not None

    def test_pool_health(self, league_db, league_dir):
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)
        pool.add_snapshot(model, "resnet", {}, epoch=1)

        entries = pool.list_entries()
        pool.update_elo(entries[0].id, 400.0)  # below floor

        sampler = OpponentSampler(pool, elo_floor=500.0)
        health = sampler.pool_health()
        assert health == 0.5  # 1 of 2 entries above floor
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_league.py -v -k "Elo or Sampler"`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

Add to `keisei/training/league.py`:

```python
import random


def compute_elo_update(
    elo_a: float, elo_b: float, result: float, k: float = 32.0,
) -> tuple[float, float]:
    """Compute updated Elo ratings after a match.

    Args:
        elo_a: Player A's current rating
        elo_b: Player B's current rating
        result: A's result (1.0 = win, 0.5 = draw, 0.0 = loss)
        k: K-factor controlling rating volatility

    Returns:
        (new_elo_a, new_elo_b)
    """
    expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    expected_b = 1.0 - expected_a
    result_b = 1.0 - result

    new_a = elo_a + k * (result - expected_a)
    new_b = elo_b + k * (result_b - expected_b)
    return new_a, new_b


class OpponentSampler:
    """Selects which opponent the learner faces each epoch."""

    def __init__(
        self,
        pool: OpponentPool,
        historical_ratio: float = 0.8,
        current_best_ratio: float = 0.2,
        elo_floor: float = 500.0,
    ) -> None:
        self.pool = pool
        self.historical_ratio = historical_ratio
        self.current_best_ratio = current_best_ratio
        self.elo_floor = elo_floor

    def sample(self) -> OpponentEntry:
        """Sample an opponent from the pool using the two-tier strategy."""
        entries = self.pool.list_entries()
        if not entries:
            raise RuntimeError("Cannot sample from an empty opponent pool")

        if len(entries) == 1:
            return entries[0]

        # Current best = most recently created
        current_best = entries[-1]

        if random.random() < self.current_best_ratio:
            return current_best

        # Historical: sample from entries above Elo floor, EXCLUDING current_best
        # to avoid double-counting (current_best is already reachable via its tier).
        historical = [
            e for e in entries[:-1] if e.elo_rating >= self.elo_floor
        ]
        if not historical:
            # Fallback: if all historical entries below floor, include all (minus current_best)
            historical = entries[:-1] if len(entries) > 1 else entries
            if not historical:
                historical = entries  # absolute fallback: single entry
            logger.warning("All historical entries below Elo floor %.0f — sampling from full pool",
                           self.elo_floor)

        return random.choice(historical)

    def pool_health(self) -> float:
        """Fraction of pool entries above the Elo floor."""
        entries = self.pool.list_entries()
        if not entries:
            return 0.0
        above = sum(1 for e in entries if e.elo_rating >= self.elo_floor)
        return above / len(entries)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_league.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/league.py tests/test_league.py
git commit -m "feat: add OpponentSampler with Elo floor and compute_elo_update"
```

---

### Task 8: Full Verification

**Files:** None (verification only)

- [ ] **Step 1: Run all Python tests**

Run: `uv run pytest -v`
Expected: All tests PASS — existing + new. No regressions.

- [ ] **Step 2: Verify imports are clean**

Run: `uv run python -c "
from keisei.training.gae import compute_gae
from keisei.training.value_adapter import get_value_adapter, ScalarValueAdapter, MultiHeadValueAdapter
from keisei.training.league import OpponentPool, OpponentSampler, compute_elo_update
from keisei.training.model_registry import get_model_contract, get_obs_channels
from keisei.config import LeagueConfig, DemonstratorConfig
print('Plan E-1 imports OK')
"`
Expected: `Plan E-1 imports OK`

- [ ] **Step 3: Commit if any fixes were needed**

```bash
git add -u
git commit -m "fix: address issues found in Plan E-1 verification"
```
