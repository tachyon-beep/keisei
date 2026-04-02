# Test Pyramid Remediation Design

**Date:** 2026-04-02
**Status:** Approved
**Scope:** Restructure flat `tests/` directory into `tests/unit/` and `tests/integration/` with shared factories, markers, and CI-friendly filtering.

## Problem

The Python test suite (525 tests, 36 files) lives in a flat `tests/` directory with no structural separation between unit and integration tests. Integration tests (313, 60%) outnumber unit tests (212, 40%), creating a **diamond anti-pattern**. There are no pytest markers configured and no way to run fast tests independently.

Current state:
- 525 Python tests in 36 files, all in `tests/`
- 135 WebUI JS tests (healthy, not in scope)
- Full suite runs in ~13s (acceptable now, won't scale)
- No markers, no directory structure, no test-level filtering

## Goals

1. **Fast local feedback** — `pytest tests/unit/` or `pytest -m "not slow"` in <5s
2. **Structural clarity** — directory membership communicates test level at a glance
3. **Shared test infrastructure** — factories for common test objects, eliminating duplication
4. **CI flexibility** — markers enable matrix splits (unit-only on PR, full on merge)

## Non-Goals

- Rewriting existing tests (only moving and splitting)
- Adding new tests
- Changing production code
- Restructuring WebUI tests (already healthy)

## Classification Rule

- **Unit**: no filesystem, no DB, no subprocess, no network. Mock-heavy is fine. PyTorch tensor math is unit. A string literal like `db_path="/tmp/test.db"` passed to a fully-mocked constructor is unit.
- **Integration**: uses `tmp_path`, `db_path`, real SQLite, writes files, starts servers.
- **Slow**: integration tests that run 1+ training epochs (model forward + backward through the full loop). Applied manually via `@pytest.mark.slow`.

## File Classification

### Pure Unit → `tests/unit/` (16 files, 233 tests)

| File | Tests | Notes |
|------|-------|-------|
| `test_katago_ppo.py` | 43 | Pure tensor math, buffer ops, PPO params |
| `test_models.py` | 26 | Model forward passes, shape checks |
| `test_katago_model.py` | 26 | SE-ResNet model tests |
| `test_evaluate.py` | 20 | Evaluation logic |
| `test_spatial_action_mapper.py` | 17 | Action space mapping |
| `test_demonstrator.py` | 17 | Fully mocked, no real IO |
| `test_registries.py` | 16 | Registry lookups |
| `test_katago_observation.py` | 13 | Observation encoding |
| `test_value_adapter.py` | 11 | Value adapter returns |
| `test_gae.py` | 9 | GAE computation |
| `test_registry_gaps.py` | 8 | Registry edge cases |
| `test_lr_scheduler.py` | 8 | LR scheduler logic |
| `test_model_gaps.py` | 7 | Model edge cases |
| `test_split_merge.py` | 4 | Split-merge step |
| `test_gae_padded.py` | 4 | Padded GAE |
| `test_katago_checkpoint.py` | 4 | Checkpoint logic (no tmp_path) |

### Pure Integration → `tests/integration/` (18 files, 252 tests)

| File | Tests | Notes |
|------|-------|-------|
| `test_katago_loop.py` | 47 | Training loop — all tests use DB + tmp_path |
| `test_sl_pipeline.py` | 46 | Parsers write/read files, dataset shards |
| `test_league.py` | 29 | OpponentPool — SQLite + filesystem |
| `test_prepare_sl.py` | 17 | SL data preparation — file IO |
| `test_config.py` | 17 | Config loading from files |
| `test_server.py` | 4 | Server lifecycle |
| `test_server_gaps.py` | 16 | Server edge cases |
| `test_server_edge_cases.py` | 14 | Server error handling |
| `test_checkpoint.py` | 8 | Checkpoint save/load |
| `test_checkpoint_gaps.py` | 2 | Checkpoint edge cases |
| `test_katago_config.py` | 4 | KataGo config file handling |
| `test_db.py` | 13 | Database operations |
| `test_db_gaps.py` | 11 | DB edge cases |
| `test_db_schema_v2.py` | 5 | Schema migration |
| `test_amp.py` | 8 | AMP with checkpointing |
| `test_sl_amp.py` | 3 | SL AMP training |
| `test_sl_observation_canary.py` | 2 | SL observation with real data |
| `test_league_config.py` | 6 | League config with DB |

### Files to Split (2 files → 4 files)

**`test_pytorch_audit_gaps.py`** (26 tests):
- Unit classes (22 tests) → `tests/unit/test_ppo_gradient_flow.py`: `TestPPOGradientFlow`, `TestNumericalStability`, `TestSplitMergeValueConsistency`, `TestBufferMemoryLifecycle`, `TestPerEnvGAEFallback`, `TestValueLossZeroEdgeCase`, `TestComputeValueMetricsEdgeCases`
- Integration classes (4 tests) → `tests/integration/test_checkpoint_scheduler_rng.py`: `TestCheckpointSchedulerRNG`

**`test_pytorch_hot_path_gaps.py`** (14 tests):
- Unit classes (12 tests) → `tests/unit/test_ppo_hot_path.py`: `TestPPOWithAMP`, `TestPPOWithValueAdapter`, `TestSplitMergeEdgeCases`, `TestSingleElementAdvantageNormalization`
- Integration classes (2 tests) → `tests/integration/test_grad_scaler_checkpoint.py`: `TestGradScalerCheckpointRoundTrip`

**After split: Unit = 267 (51%), Integration = 258 (49%)**

## Directory Structure

```
tests/
├── conftest.py                              # shared: db_path, db fixtures
├── unit/
│   ├── __init__.py
│   ├── conftest.py                          # unit factories: small_model, filled_buffer, ppo_algorithm, mock_vecenv
│   ├── test_katago_ppo.py
│   ├── test_models.py
│   ├── test_katago_model.py
│   ├── test_evaluate.py
│   ├── test_spatial_action_mapper.py
│   ├── test_demonstrator.py
│   ├── test_registries.py
│   ├── test_katago_observation.py
│   ├── test_value_adapter.py
│   ├── test_gae.py
│   ├── test_registry_gaps.py
│   ├── test_lr_scheduler.py
│   ├── test_model_gaps.py
│   ├── test_split_merge.py
│   ├── test_gae_padded.py
│   ├── test_katago_checkpoint.py
│   ├── test_ppo_gradient_flow.py            # split from test_pytorch_audit_gaps.py
│   └── test_ppo_hot_path.py                 # split from test_pytorch_hot_path_gaps.py
├── integration/
│   ├── __init__.py
│   ├── conftest.py                          # integration factories: katago_config, league_db, league_dir, sample_sfen_dir
│   ├── test_katago_loop.py
│   ├── test_sl_pipeline.py
│   ├── test_league.py
│   ├── test_prepare_sl.py
│   ├── test_server.py
│   ├── test_server_gaps.py
│   ├── test_server_edge_cases.py
│   ├── test_config.py
│   ├── test_checkpoint.py
│   ├── test_checkpoint_gaps.py
│   ├── test_katago_config.py
│   ├── test_db.py
│   ├── test_db_gaps.py
│   ├── test_db_schema_v2.py
│   ├── test_amp.py
│   ├── test_sl_amp.py
│   ├── test_sl_observation_canary.py
│   ├── test_league_config.py
│   ├── test_checkpoint_scheduler_rng.py     # split from test_pytorch_audit_gaps.py
│   └── test_grad_scaler_checkpoint.py       # split from test_pytorch_hot_path_gaps.py
```

## Shared Test Infrastructure (Factories)

### `tests/unit/conftest.py`

```python
"""Shared factories for unit tests — no filesystem or DB access."""

import pytest
import torch
from unittest.mock import MagicMock

import numpy as np

from keisei.training.katago_ppo import (
    KataGoPPOAlgorithm,
    KataGoPPOParams,
    KataGoRolloutBuffer,
)
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


@pytest.fixture
def small_model():
    """Minimal SE-ResNet for fast unit tests."""
    return SEResNetModel(SEResNetParams(
        num_blocks=2, channels=32, se_reduction=8,
        global_pool_channels=16, policy_channels=8,
        value_fc_size=32, score_fc_size=16, obs_channels=50,
    ))


@pytest.fixture
def filled_buffer():
    """Factory: returns a function that creates pre-loaded rollout buffers.

    Usage: buf = filled_buffer(num_envs=4, steps=3)
    """
    def _make(num_envs=4, steps=3, action_space=11259):
        buf = KataGoRolloutBuffer(
            num_envs=num_envs, obs_shape=(50, 9, 9), action_space=action_space,
        )
        for t in range(steps):
            is_last = t == steps - 1
            buf.add(
                obs=torch.randn(num_envs, 50, 9, 9),
                actions=torch.randint(0, action_space, (num_envs,)),
                log_probs=torch.randn(num_envs),
                values=torch.randn(num_envs),
                rewards=torch.where(
                    torch.tensor([is_last] * num_envs),
                    torch.tensor([1.0, -1.0, 0.0, 1.0][:num_envs]),
                    torch.zeros(num_envs),
                ),
                dones=torch.tensor([is_last] * num_envs),
                legal_masks=torch.ones(num_envs, action_space, dtype=torch.bool),
                value_categories=torch.where(
                    torch.tensor([is_last] * num_envs),
                    torch.tensor([0, 2, 1, 0][:num_envs]),
                    torch.full((num_envs,), -1),
                ),
                score_targets=torch.randn(num_envs).clamp(-1.5, 1.5),
            )
        return buf
    return _make


@pytest.fixture
def ppo_algorithm(small_model):
    """A KataGoPPOAlgorithm with small model, ready for loss/gradient tests."""
    return KataGoPPOAlgorithm(model=small_model, params=KataGoPPOParams())


@pytest.fixture
def mock_vecenv():
    """Factory: returns a function that creates mock VecEnvs with correct shapes.

    Usage: env = mock_vecenv(num_envs=2, terminate_at_step=3)
    """
    def _make(num_envs=2, *, terminate_at_step=None):
        rng = np.random.default_rng(42)
        mock = MagicMock()
        mock.observation_channels = 50
        mock.action_space_size = 11259
        mock.episodes_completed = 0
        mock.mean_episode_length = 0.0
        mock.truncation_rate = 0.0
        step_count = [0]

        def make_reset_result():
            result = MagicMock()
            result.observations = rng.standard_normal(
                (num_envs, 50, 9, 9)
            ).astype(np.float32)
            result.legal_masks = np.ones((num_envs, 11259), dtype=bool)
            return result

        def make_step_result(actions):
            step_count[0] += 1
            result = MagicMock()
            result.observations = rng.standard_normal(
                (num_envs, 50, 9, 9)
            ).astype(np.float32)
            result.legal_masks = np.ones((num_envs, 11259), dtype=bool)
            result.rewards = np.zeros(num_envs, dtype=np.float32)
            result.terminated = np.zeros(num_envs, dtype=bool)
            result.truncated = np.zeros(num_envs, dtype=bool)
            if terminate_at_step and step_count[0] == terminate_at_step:
                result.rewards[0] = 1.0
                result.terminated[0] = True
            result.current_players = np.zeros(num_envs, dtype=np.uint8)
            result.material_balances = np.zeros(num_envs, dtype=np.int32)
            return result

        mock.reset.side_effect = lambda: make_reset_result()
        mock.step.side_effect = make_step_result
        return mock
    return _make
```

### `tests/integration/conftest.py`

```python
"""Shared factories for integration tests + auto-marking."""

from pathlib import Path

import pytest

from keisei.config import AppConfig, DisplayConfig, LeagueConfig, ModelConfig, TrainingConfig
from keisei.db import init_db


# ---------------------------------------------------------------------------
# Auto-mark all tests in this directory as integration
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(items):
    integration_dir = str(Path(__file__).parent)
    for item in items:
        if str(item.fspath).startswith(integration_dir):
            item.add_marker(pytest.mark.integration)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

@pytest.fixture
def katago_config(tmp_path):
    """Full AppConfig pointing at temp directories, suitable for training loop tests."""
    return AppConfig(
        training=TrainingConfig(
            num_envs=2,
            steps_per_epoch=4,
            ppo_epochs=1,
            max_epochs=1,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            checkpoint_interval=999,
            warmup_epochs=0,
            architecture="se_resnet",
            algorithm="katago_ppo",
        ),
        model=ModelConfig(
            num_blocks=2,
            channels=32,
            se_reduction=8,
            global_pool_channels=16,
            policy_channels=8,
            value_fc_size=32,
            score_fc_size=16,
        ),
        display=DisplayConfig(
            db_path=str(tmp_path / "test.db"),
        ),
    )


@pytest.fixture
def league_db(tmp_path):
    """Initialised temporary SQLite database for league tests."""
    db_path = str(tmp_path / "league.db")
    init_db(db_path)
    return db_path


@pytest.fixture
def league_dir(tmp_path):
    """Temporary directory for league checkpoint files."""
    d = tmp_path / "checkpoints" / "league"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def sample_sfen_dir(tmp_path):
    """Directory containing a sample SFEN game file for SL pipeline tests."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    sfen_content = (
        "result:win_black\n"
        "startpos\n"
        "7g7f\n"
        "3c3d\n"
        "2g2f\n"
        "8c8d\n"
    )
    sfen_file = games_dir / "test.sfen"
    sfen_file.write_text(sfen_content)
    return games_dir
```

## Markers

### `pyproject.toml` changes

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "integration: tests that use filesystem, DB, or external resources",
    "slow: tests that take >1s individually (training loops, full epochs)",
]
```

### `@pytest.mark.slow` targets

Applied manually to test classes that run training epochs:

| File | Classes/functions |
|------|-------------------|
| `tests/integration/test_katago_loop.py` | `TestKataGoTrainingLoopRun`, `TestLeagueIntegration`, `TestSplitMergeIntegration`, `TestSeatRotation`, `TestCheckResume` |
| `tests/integration/test_sl_pipeline.py` | `TestSLTrainer` |
| `tests/integration/test_amp.py` | All tests (class-level mark) |
| `tests/integration/test_sl_amp.py` | All tests (class-level mark) |

## Running Tests

```bash
uv run pytest tests/unit/                       # unit only (~3-5s)
uv run pytest tests/integration/                # all integration (~8-10s)
uv run pytest tests/integration/ -m "not slow"  # fast integration (~4-6s)
uv run pytest -m "not slow"                     # everything except slow (~8-10s)
uv run pytest                                   # full suite (~13s)
```

## Verification

After migration, confirm:

```bash
uv run pytest --collect-only -q | tail -1                    # 525 tests collected
uv run pytest tests/unit/ --collect-only -q | tail -1        # 267 tests
uv run pytest tests/integration/ --collect-only -q | tail -1 # 258 tests
uv run pytest                                                # 524 passed, 1 skipped
```

## Migration Steps

1. Create directory structure (`tests/unit/`, `tests/integration/`, `__init__.py`)
2. Write `tests/unit/conftest.py` and `tests/integration/conftest.py` with factories
3. Move 16 pure-unit files to `tests/unit/`
4. Move 18 pure-integration files to `tests/integration/`
5. Split `test_pytorch_audit_gaps.py` → `test_ppo_gradient_flow.py` (unit) + `test_checkpoint_scheduler_rng.py` (integration)
6. Split `test_pytorch_hot_path_gaps.py` → `test_ppo_hot_path.py` (unit) + `test_grad_scaler_checkpoint.py` (integration)
7. Remove duplicate helper functions from test files, update to use conftest fixtures
8. Add `@pytest.mark.slow` to identified test classes
9. Update `pyproject.toml` with marker definitions
10. Run full verification suite
