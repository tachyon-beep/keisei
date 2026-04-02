# Test Pyramid Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure flat `tests/` into `tests/unit/` and `tests/integration/` with shared factories, pytest markers, and CI-friendly filtering.

**Architecture:** Move 36 test files into two subdirectories based on whether they touch filesystem/DB. Split 2 mixed files. Consolidate duplicated test helpers into shared conftest factories. Add `integration` and `slow` markers.

**Tech Stack:** pytest, Python 3.13, uv

**Spec:** `docs/superpowers/specs/2026-04-02-test-pyramid-remediation-design.md`

---

### Task 1: Create directory structure and __init__.py files

**Files:**
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`

- [ ] **Step 1: Create unit and integration directories with init files**

```bash
mkdir -p tests/unit tests/integration
touch tests/unit/__init__.py tests/integration/__init__.py
```

- [ ] **Step 2: Verify structure**

```bash
ls tests/unit/__init__.py tests/integration/__init__.py
```

Expected: both files exist, no errors.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/__init__.py tests/integration/__init__.py
git commit -m "test: create tests/unit/ and tests/integration/ directory structure"
```

---

### Task 2: Write unit conftest with shared factories

**Files:**
- Create: `tests/unit/conftest.py`

The following helper functions are currently duplicated across test files and will be consolidated here:
- `_small_model()` — duplicated in `test_pytorch_audit_gaps.py` and `test_pytorch_hot_path_gaps.py`
- `_filled_buffer()` — duplicated in `test_pytorch_audit_gaps.py` and `test_pytorch_hot_path_gaps.py`

Note: `test_sl_amp.py` has its own `_small_model()` with different params (1 block, 8 channels) — that stays local since it's an integration file with deliberately smaller config.

- [ ] **Step 1: Write tests/unit/conftest.py**

```python
"""Shared factories for unit tests — no filesystem or DB access."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

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

- [ ] **Step 2: Verify conftest loads**

Run: `uv run pytest tests/unit/ --collect-only -q`
Expected: `no tests ran` (no test files yet), but no import errors.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/conftest.py
git commit -m "test: add unit conftest with shared factories (small_model, filled_buffer, ppo_algorithm, mock_vecenv)"
```

---

### Task 3: Write integration conftest with shared factories and auto-marking

**Files:**
- Create: `tests/integration/conftest.py`

Consolidates fixtures currently defined locally in:
- `test_katago_loop.py:82` — `katago_config`
- `test_league.py:20-30` — `league_db`, `league_dir`
- `test_prepare_sl.py:12-25` — `sample_sfen_dir`

**Important:** The `katago_config` fixture must match the exact field names and structure used in `test_katago_loop.py` — it uses `AppConfig(training=TrainingConfig(...), display=DisplayConfig(...), model=ModelConfig(...))` with `algorithm_params` dict, `display_name`, and `params` dict.

- [ ] **Step 1: Write tests/integration/conftest.py**

Copy the exact `katago_config` fixture from `tests/test_katago_loop.py:82-119`, the `league_db` and `league_dir` fixtures from `tests/test_league.py:19-30`, and the `sample_sfen_dir` fixture from `tests/test_prepare_sl.py:11-25`. Read those files first to get the exact current code. Add the `pytest_collection_modifyitems` hook for auto-marking.

The structure:

```python
"""Shared factories for integration tests + auto-marking."""

from pathlib import Path

import pytest

from keisei.config import AppConfig, DisplayConfig, ModelConfig, TrainingConfig
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
# Factories — copied from their original test files
# ---------------------------------------------------------------------------

@pytest.fixture
def katago_config(tmp_path):
    """Full AppConfig for training loop tests.

    Copied from tests/test_katago_loop.py:82-119 — read that file
    for the exact current field names and structure.
    """
    # PASTE EXACT FIXTURE FROM test_katago_loop.py HERE
    ...


@pytest.fixture
def league_db(tmp_path):
    """Initialised temporary SQLite database for league tests.

    Copied from tests/test_league.py:20-23.
    """
    db_path = str(tmp_path / "league.db")
    init_db(db_path)
    return db_path


@pytest.fixture
def league_dir(tmp_path):
    """Temporary directory for league checkpoint files.

    Copied from tests/test_league.py:27-30.
    """
    d = tmp_path / "checkpoints" / "league"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def sample_sfen_dir(tmp_path):
    """Directory with a sample SFEN game file.

    Copied from tests/test_prepare_sl.py:12-25 — read that file
    for the exact current content.
    """
    # PASTE EXACT FIXTURE FROM test_prepare_sl.py HERE
    ...
```

**Critical:** Do NOT simplify or rewrite the `katago_config` or `sample_sfen_dir` fixtures. Copy them verbatim from the source files, then delete the originals from those files in later tasks.

- [ ] **Step 2: Verify conftest loads**

Run: `uv run pytest tests/integration/ --collect-only -q`
Expected: `no tests ran` (no test files yet), but no import errors.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/conftest.py
git commit -m "test: add integration conftest with shared factories and auto-marking hook"
```

---

### Task 4: Move 16 pure-unit test files to tests/unit/

**Files:**
- Move: `tests/test_katago_ppo.py` → `tests/unit/test_katago_ppo.py`
- Move: `tests/test_models.py` → `tests/unit/test_models.py`
- Move: `tests/test_katago_model.py` → `tests/unit/test_katago_model.py`
- Move: `tests/test_evaluate.py` → `tests/unit/test_evaluate.py`
- Move: `tests/test_spatial_action_mapper.py` → `tests/unit/test_spatial_action_mapper.py`
- Move: `tests/test_demonstrator.py` → `tests/unit/test_demonstrator.py`
- Move: `tests/test_registries.py` → `tests/unit/test_registries.py`
- Move: `tests/test_katago_observation.py` → `tests/unit/test_katago_observation.py`
- Move: `tests/test_value_adapter.py` → `tests/unit/test_value_adapter.py`
- Move: `tests/test_gae.py` → `tests/unit/test_gae.py`
- Move: `tests/test_registry_gaps.py` → `tests/unit/test_registry_gaps.py`
- Move: `tests/test_lr_scheduler.py` → `tests/unit/test_lr_scheduler.py`
- Move: `tests/test_model_gaps.py` → `tests/unit/test_model_gaps.py`
- Move: `tests/test_split_merge.py` → `tests/unit/test_split_merge.py`
- Move: `tests/test_gae_padded.py` → `tests/unit/test_gae_padded.py`
- Move: `tests/test_katago_checkpoint.py` → `tests/unit/test_katago_checkpoint.py`

- [ ] **Step 1: Move all 16 files using git mv**

```bash
git mv tests/test_katago_ppo.py tests/unit/test_katago_ppo.py
git mv tests/test_models.py tests/unit/test_models.py
git mv tests/test_katago_model.py tests/unit/test_katago_model.py
git mv tests/test_evaluate.py tests/unit/test_evaluate.py
git mv tests/test_spatial_action_mapper.py tests/unit/test_spatial_action_mapper.py
git mv tests/test_demonstrator.py tests/unit/test_demonstrator.py
git mv tests/test_registries.py tests/unit/test_registries.py
git mv tests/test_katago_observation.py tests/unit/test_katago_observation.py
git mv tests/test_value_adapter.py tests/unit/test_value_adapter.py
git mv tests/test_gae.py tests/unit/test_gae.py
git mv tests/test_registry_gaps.py tests/unit/test_registry_gaps.py
git mv tests/test_lr_scheduler.py tests/unit/test_lr_scheduler.py
git mv tests/test_model_gaps.py tests/unit/test_model_gaps.py
git mv tests/test_split_merge.py tests/unit/test_split_merge.py
git mv tests/test_gae_padded.py tests/unit/test_gae_padded.py
git mv tests/test_katago_checkpoint.py tests/unit/test_katago_checkpoint.py
```

- [ ] **Step 2: Run unit tests to verify they pass**

Run: `uv run pytest tests/unit/ -q`
Expected: 233 passed (all unit tests pass from new location).

- [ ] **Step 3: Commit**

```bash
git add -A tests/
git commit -m "test: move 16 pure-unit test files to tests/unit/"
```

---

### Task 5: Move 18 pure-integration test files to tests/integration/

**Files:**
- Move: `tests/test_katago_loop.py` → `tests/integration/test_katago_loop.py`
- Move: `tests/test_sl_pipeline.py` → `tests/integration/test_sl_pipeline.py`
- Move: `tests/test_league.py` → `tests/integration/test_league.py`
- Move: `tests/test_prepare_sl.py` → `tests/integration/test_prepare_sl.py`
- Move: `tests/test_config.py` → `tests/integration/test_config.py`
- Move: `tests/test_server.py` → `tests/integration/test_server.py`
- Move: `tests/test_server_gaps.py` → `tests/integration/test_server_gaps.py`
- Move: `tests/test_server_edge_cases.py` → `tests/integration/test_server_edge_cases.py`
- Move: `tests/test_checkpoint.py` → `tests/integration/test_checkpoint.py`
- Move: `tests/test_checkpoint_gaps.py` → `tests/integration/test_checkpoint_gaps.py`
- Move: `tests/test_katago_config.py` → `tests/integration/test_katago_config.py`
- Move: `tests/test_db.py` → `tests/integration/test_db.py`
- Move: `tests/test_db_gaps.py` → `tests/integration/test_db_gaps.py`
- Move: `tests/test_db_schema_v2.py` → `tests/integration/test_db_schema_v2.py`
- Move: `tests/test_amp.py` → `tests/integration/test_amp.py`
- Move: `tests/test_sl_amp.py` → `tests/integration/test_sl_amp.py`
- Move: `tests/test_sl_observation_canary.py` → `tests/integration/test_sl_observation_canary.py`
- Move: `tests/test_league_config.py` → `tests/integration/test_league_config.py`

- [ ] **Step 1: Move all 18 files using git mv**

```bash
git mv tests/test_katago_loop.py tests/integration/test_katago_loop.py
git mv tests/test_sl_pipeline.py tests/integration/test_sl_pipeline.py
git mv tests/test_league.py tests/integration/test_league.py
git mv tests/test_prepare_sl.py tests/integration/test_prepare_sl.py
git mv tests/test_config.py tests/integration/test_config.py
git mv tests/test_server.py tests/integration/test_server.py
git mv tests/test_server_gaps.py tests/integration/test_server_gaps.py
git mv tests/test_server_edge_cases.py tests/integration/test_server_edge_cases.py
git mv tests/test_checkpoint.py tests/integration/test_checkpoint.py
git mv tests/test_checkpoint_gaps.py tests/integration/test_checkpoint_gaps.py
git mv tests/test_katago_config.py tests/integration/test_katago_config.py
git mv tests/test_db.py tests/integration/test_db.py
git mv tests/test_db_gaps.py tests/integration/test_db_gaps.py
git mv tests/test_db_schema_v2.py tests/integration/test_db_schema_v2.py
git mv tests/test_amp.py tests/integration/test_amp.py
git mv tests/test_sl_amp.py tests/integration/test_sl_amp.py
git mv tests/test_sl_observation_canary.py tests/integration/test_sl_observation_canary.py
git mv tests/test_league_config.py tests/integration/test_league_config.py
```

- [ ] **Step 2: Run integration tests to verify they pass**

Run: `uv run pytest tests/integration/ -q`
Expected: 252 passed (all integration tests pass from new location).

Note: some integration tests define their own `katago_config`/`league_db`/`league_dir`/`sample_sfen_dir` fixtures locally. These will shadow the conftest versions — that's fine for now. We'll deduplicate in Task 7.

- [ ] **Step 3: Run full suite to verify total count**

Run: `uv run pytest --collect-only -q 2>/dev/null | tail -1`
Expected: Should show 485 tests collected (233 unit + 252 integration). The 2 mixed files (`test_pytorch_audit_gaps.py`, `test_pytorch_hot_path_gaps.py`) still sit in `tests/` and will be split in Tasks 6a/6b. Total after split will be 525.

- [ ] **Step 4: Commit**

```bash
git add -A tests/
git commit -m "test: move 18 pure-integration test files to tests/integration/"
```

---

### Task 6a: Split test_pytorch_audit_gaps.py into unit and integration parts

**Files:**
- Read: `tests/test_pytorch_audit_gaps.py` (source — will be deleted)
- Create: `tests/unit/test_ppo_gradient_flow.py` (22 unit tests)
- Create: `tests/integration/test_checkpoint_scheduler_rng.py` (4 integration tests)
- Delete: `tests/test_pytorch_audit_gaps.py`

The split:
- **Unit** (→ `test_ppo_gradient_flow.py`): `TestPPOGradientFlow`, `TestNumericalStability`, `TestSplitMergeValueConsistency`, `TestBufferMemoryLifecycle`, `TestPerEnvGAEFallback`, `TestValueLossZeroEdgeCase`, `TestComputeValueMetricsEdgeCases`, `TestTransformerKataGoIncompatibility`
- **Integration** (→ `test_checkpoint_scheduler_rng.py`): `TestCheckpointSchedulerRNG`

- [ ] **Step 1: Create tests/unit/test_ppo_gradient_flow.py**

Read `tests/test_pytorch_audit_gaps.py` and copy everything EXCEPT `TestCheckpointSchedulerRNG` and the checkpoint-related imports (`load_checkpoint`, `save_checkpoint`). Replace inline `_small_model()` and `_filled_buffer()` calls with the `small_model` and `filled_buffer` conftest fixtures where tests use them directly. Where test classes define their own `ppo` fixture using `_small_model()`, replace with the `small_model` conftest fixture.

Specifically:
- Remove the `_small_model()` and `_filled_buffer()` function definitions
- In `TestPPOGradientFlow`: change the `ppo` fixture to use the `small_model` conftest fixture
- In `TestNumericalStability`: same
- In `TestSplitMergeValueConsistency.test_values_match_scalar_value_method`: use `small_model` fixture
- In `TestBufferMemoryLifecycle.test_clear_releases_tensor_references`: use `filled_buffer` fixture
- In `TestBufferMemoryLifecycle.test_update_calls_clear`: use `small_model` and `filled_buffer`
- In `TestBufferMemoryLifecycle.test_flatten_then_clear_no_shared_references`: use `filled_buffer`
- Tests that create custom buffers (non-default args) should call `filled_buffer(num_envs=..., steps=...)` since it's a factory fixture
- Keep `import random` only if `TestCheckpointSchedulerRNG` needed it (it does — for rng_state test) — since we're removing that class, check if any remaining class uses `random`. Answer: no. Remove `import random`.

- [ ] **Step 2: Create tests/integration/test_checkpoint_scheduler_rng.py**

Read `tests/test_pytorch_audit_gaps.py` and copy only `TestCheckpointSchedulerRNG` plus its imports (`load_checkpoint`, `save_checkpoint`, `random`, etc.) and the `_small_model()` helper (this class's `model` fixture uses it, and since it needs `tmp_path` it can't use the unit conftest). Actually: use the `small_model` fixture from `tests/unit/conftest.py`? No — integration conftest doesn't have that. Instead, define `_small_model()` locally in this file (it's only 5 lines) or add a `small_model` fixture to the integration conftest too.

Decision: add a `small_model` fixture to `tests/integration/conftest.py` as well — identical to the unit version. Checkpoint tests need it, and this avoids duplicating the helper in every integration file.

- [ ] **Step 2b: Add small_model fixture to integration conftest**

Add the same `small_model` fixture from `tests/unit/conftest.py` to `tests/integration/conftest.py`:

```python
@pytest.fixture
def small_model():
    """Minimal SE-ResNet for tests that need a model but aren't testing model behavior."""
    return SEResNetModel(SEResNetParams(
        num_blocks=2, channels=32, se_reduction=8,
        global_pool_channels=16, policy_channels=8,
        value_fc_size=32, score_fc_size=16, obs_channels=50,
    ))
```

Add the required imports (`SEResNetModel`, `SEResNetParams`) to the integration conftest if not already present.

- [ ] **Step 3: Delete original file**

```bash
git rm tests/test_pytorch_audit_gaps.py
```

- [ ] **Step 4: Verify test counts**

Run: `uv run pytest tests/unit/test_ppo_gradient_flow.py --collect-only -q 2>/dev/null | tail -1`
Expected: 22 tests collected

Run: `uv run pytest tests/integration/test_checkpoint_scheduler_rng.py --collect-only -q 2>/dev/null | tail -1`
Expected: 4 tests collected

- [ ] **Step 5: Run both files to verify they pass**

Run: `uv run pytest tests/unit/test_ppo_gradient_flow.py tests/integration/test_checkpoint_scheduler_rng.py -q`
Expected: 26 passed

- [ ] **Step 6: Commit**

```bash
git add -A tests/
git commit -m "test: split test_pytorch_audit_gaps.py into unit/test_ppo_gradient_flow.py and integration/test_checkpoint_scheduler_rng.py"
```

---

### Task 6b: Split test_pytorch_hot_path_gaps.py into unit and integration parts

**Files:**
- Read: `tests/test_pytorch_hot_path_gaps.py` (source — will be deleted)
- Create: `tests/unit/test_ppo_hot_path.py` (12 unit tests)
- Create: `tests/integration/test_grad_scaler_checkpoint.py` (2 integration tests)
- Delete: `tests/test_pytorch_hot_path_gaps.py`

The split:
- **Unit** (→ `test_ppo_hot_path.py`): `TestPPOWithAMP`, `TestPPOWithValueAdapter`, `TestSplitMergeEdgeCases`, `TestSingleElementAdvantageNormalization`
- **Integration** (→ `test_grad_scaler_checkpoint.py`): `TestGradScalerCheckpointRoundTrip`

- [ ] **Step 1: Create tests/unit/test_ppo_hot_path.py**

Read `tests/test_pytorch_hot_path_gaps.py` and copy everything EXCEPT `TestGradScalerCheckpointRoundTrip` and the checkpoint-related imports. Replace `_small_model()` and `_filled_buffer()` with conftest fixtures as in Task 6a.

Specifically:
- Remove `_small_model()` and `_filled_buffer()` definitions
- Remove `from keisei.training.checkpoint import load_checkpoint, save_checkpoint`
- Remove `from torch.amp import GradScaler`
- Each test that calls `_small_model()` inline should receive `small_model` as a fixture parameter
- Each test that calls `_filled_buffer(...)` should receive `filled_buffer` as a fixture parameter and call `filled_buffer(num_envs=4, steps=3)`

- [ ] **Step 2: Create tests/integration/test_grad_scaler_checkpoint.py**

Copy `TestGradScalerCheckpointRoundTrip` with its imports. Use `small_model` fixture from integration conftest (added in Task 6a step 2).

- [ ] **Step 3: Delete original file**

```bash
git rm tests/test_pytorch_hot_path_gaps.py
```

- [ ] **Step 4: Verify test counts**

Run: `uv run pytest tests/unit/test_ppo_hot_path.py --collect-only -q 2>/dev/null | tail -1`
Expected: 12 tests collected

Run: `uv run pytest tests/integration/test_grad_scaler_checkpoint.py --collect-only -q 2>/dev/null | tail -1`
Expected: 2 tests collected

- [ ] **Step 5: Run both files to verify they pass**

Run: `uv run pytest tests/unit/test_ppo_hot_path.py tests/integration/test_grad_scaler_checkpoint.py -q`
Expected: 14 passed

- [ ] **Step 6: Commit**

```bash
git add -A tests/
git commit -m "test: split test_pytorch_hot_path_gaps.py into unit/test_ppo_hot_path.py and integration/test_grad_scaler_checkpoint.py"
```

---

### Task 7: Deduplicate fixtures — remove local definitions that now live in conftest

**Files:**
- Modify: `tests/integration/test_katago_loop.py` — remove local `katago_config` fixture (lines ~82-119) and `_make_mock_katago_vecenv` if moved to conftest
- Modify: `tests/integration/test_league.py` — remove local `league_db` fixture (lines ~19-23) and `league_dir` fixture (lines ~26-30)
- Modify: `tests/integration/test_prepare_sl.py` — remove local `sample_sfen_dir` fixture (lines ~11-25)

**Important considerations:**
- `_make_mock_katago_vecenv` in `test_katago_loop.py` has complex options (`terminate_at_step`, `alternate_players`, `material_balance`) that are used by many tests. This stays as a local helper in `test_katago_loop.py` — it's too specialized for the shared conftest. The unit conftest's `mock_vecenv` is simpler and serves different tests.
- Before removing any local fixture, verify it has the exact same signature and behavior as the conftest version. If they differ, keep the local version.
- `test_league.py` also uses a local `league_db` and `league_dir` — these should match the conftest versions exactly. Verify by reading both before removing.

- [ ] **Step 1: Read the conftest versions and the local versions side by side**

Read `tests/integration/conftest.py` and compare each fixture against:
- `tests/integration/test_katago_loop.py` (katago_config)
- `tests/integration/test_league.py` (league_db, league_dir)
- `tests/integration/test_prepare_sl.py` (sample_sfen_dir)

Only remove a local fixture if the conftest version is identical in behavior.

- [ ] **Step 2: Remove deduplicated local fixtures**

For each file where the local fixture matches the conftest version, delete the local fixture definition and its associated imports (if now unused).

- [ ] **Step 3: Run full suite to verify nothing broke**

Run: `uv run pytest -q`
Expected: 524 passed, 1 skipped (same as before migration)

- [ ] **Step 4: Commit**

```bash
git add -A tests/
git commit -m "test: deduplicate fixtures — use shared conftest versions"
```

---

### Task 8: Add @pytest.mark.slow to training epoch tests

**Files:**
- Modify: `tests/integration/test_katago_loop.py`
- Modify: `tests/integration/test_sl_pipeline.py`
- Modify: `tests/integration/test_amp.py`
- Modify: `tests/integration/test_sl_amp.py`

- [ ] **Step 1: Add slow marks to test_katago_loop.py**

Add `import pytest` if not already imported. Add `@pytest.mark.slow` to these classes:
- `TestKataGoTrainingLoopRun`
- `TestLeagueIntegration`
- `TestSplitMergeIntegration`
- `TestSeatRotation`
- `TestCheckResume`
- `TestRotateSeatIsolation`
- `TestRotateSeat`
- `TestSLToRLCheckpointHandoff`
- `TestCheckpointResumeRoundTrip`
- `TestCheckpointWrittenToDisk`

Do NOT mark these (they're fast init/config tests):
- `TestKataGoTrainingLoopInit`
- `TestCreateLrSchedulerUnknownType`
- `TestArchitectureAlgorithmMismatchGuard`
- `TestMaybeUpdateHeartbeat`
- `TestValueCategoryNoLeague`
- `TestSwallowedExceptions`

Read each class to verify: if it calls `loop.run(...)` or does model forward+backward, it's slow. If it only constructs objects or checks config, it's fast.

- [ ] **Step 2: Add slow marks to test_sl_pipeline.py**

Add `@pytest.mark.slow` to these classes (they run training epochs):
- `TestSLTrainer`
- `TestSLTrainerCheckpointRoundTrip`
- `TestSLTrainerSchedulerAndClipping`
- `TestSLTrainerWithBinaryShards`
- `TestSLTrainerExtended`
- `TestWriteShardPerformance`

Do NOT mark parser/dataset tests (they're fast file I/O):
- `TestSFENParser`, `TestCSAParser`, `TestCSAParserHardening`, `TestCSAPromotionDetection`, `TestCSAParserEdgeCases`
- `TestSLDataset`, `TestSLDatasetPartialShard`, `TestSLDatasetMultiShard`
- `TestGameFilterRatingKeys`

Read each class to verify before marking.

- [ ] **Step 3: Add slow marks to test_amp.py**

Add `@pytest.mark.slow` to the entire module or to each class. Read the file first — if all 4 classes run training steps, mark them all.

- [ ] **Step 4: Add slow marks to test_sl_amp.py**

Add `@pytest.mark.slow` to `TestSLAmp` (the only class — all 3 tests run SL training epochs).

- [ ] **Step 5: Verify slow marker works**

Run: `uv run pytest -m slow --collect-only -q 2>/dev/null | tail -1`
Expected: should collect the marked tests (exact count depends on how many classes were marked).

Run: `uv run pytest -m "not slow" -q`
Expected: passes, with fewer tests than the full suite.

- [ ] **Step 6: Commit**

```bash
git add -A tests/
git commit -m "test: add @pytest.mark.slow to training epoch tests"
```

---

### Task 9: Update pyproject.toml with marker definitions

**Files:**
- Modify: `pyproject.toml:54-56`

- [ ] **Step 1: Add markers to pytest config**

In `pyproject.toml`, change:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

to:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "integration: tests that use filesystem, DB, or external resources",
    "slow: tests that take >1s individually (training loops, full epochs)",
]
```

- [ ] **Step 2: Verify no marker warnings**

Run: `uv run pytest --strict-markers -q 2>&1 | head -5`
Expected: no `PytestUnknownMarkWarning` warnings.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "config: register integration and slow pytest markers"
```

---

### Task 10: Final verification

**Files:** None (verification only)

- [ ] **Step 1: Verify total test count**

Run: `uv run pytest --collect-only -q 2>/dev/null | tail -1`
Expected: 525 tests collected

- [ ] **Step 2: Verify unit test count**

Run: `uv run pytest tests/unit/ --collect-only -q 2>/dev/null | tail -1`
Expected: 267 tests collected (233 original + 22 from audit_gaps split + 12 from hot_path split)

- [ ] **Step 3: Verify integration test count**

Run: `uv run pytest tests/integration/ --collect-only -q 2>/dev/null | tail -1`
Expected: 258 tests collected (252 original + 4 from audit_gaps split + 2 from hot_path split)

- [ ] **Step 4: Verify no orphaned test files in tests/ root**

```bash
ls tests/test_*.py 2>/dev/null
```
Expected: no output (all test files moved to subdirectories). `tests/conftest.py` should still exist.

- [ ] **Step 5: Run full suite**

Run: `uv run pytest -q`
Expected: 524 passed, 1 skipped, with warnings (same as pre-migration baseline)

- [ ] **Step 6: Verify marker filtering**

Run: `uv run pytest -m "not integration" --collect-only -q 2>/dev/null | tail -1`
Expected: 267 tests (unit tests only)

Run: `uv run pytest -m "not slow" -q`
Expected: passes, fewer tests than full suite

- [ ] **Step 7: Verify unit tests are fast**

Run: `uv run pytest tests/unit/ -q --tb=no`
Expected: completes in <5s

- [ ] **Step 8: Commit verification results as comment**

No commit needed — this is a verification-only task.
