# Test Pyramid Markers & Renames — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify the test suite into unit/integration via pytest markers, split the `test_katago_loop.py` monolith, and rename 24 test files for clarity.

**Architecture:** Add a pytest `integration` marker to `pyproject.toml`, apply module-level `pytestmark` to integration test files, split `test_katago_loop.py` by extracting integration classes into a new file, then `git mv` 24 test files. No test logic changes — markers and renames only.

**Tech Stack:** pytest markers, git mv

---

### Task 1: Add `integration` marker to pyproject.toml

**Files:**
- Modify: `pyproject.toml:57-59`

- [ ] **Step 1: Add the marker definition**

In `pyproject.toml`, replace:

```toml
markers = [
    "slow: marks tests as slow-running (deselect with '-m \"not slow\"')",
]
```

with:

```toml
markers = [
    "slow: marks tests as slow-running (deselect with '-m \"not slow\"')",
    "integration: marks tests requiring DB, filesystem, or subprocess resources (deselect with '-m \"not integration\"')",
]
```

- [ ] **Step 2: Verify marker is recognized**

Run: `uv run pytest --markers | grep integration`

Expected: `@pytest.mark.integration: marks tests requiring DB, filesystem, or subprocess resources`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pytest integration marker to pyproject.toml"
```

---

### Task 2: Split `test_katago_loop.py` into unit and integration files

This is the most complex task. The 1857-line file has 27 test classes. 16 are unit tests (mocked I/O), 11 are integration (real DB/checkpoints).

**Files:**
- Modify: `tests/test_katago_loop.py` (keep unit tests)
- Create: `tests/test_katago_loop_integration.py` (move integration tests here)

**Classification:**

UNIT classes (stay in `test_katago_loop.py`):
- `TestDDPInit` (lines 133-149)
- `TestRankGating` (lines 152-209)
- `TestCreateLrSchedulerUnknownType` (lines 703-717)
- `TestLrSchedulerPrivateInternals` (lines 719-754)
- `TestArchitectureAlgorithmMismatchGuard` (lines 756-809)
- `TestMaybeUpdateHeartbeat` (lines 812-841)
- `TestValueCategoryNoLeague` (lines 843-923)
- `TestSwallowedExceptions` (lines 961-1011)
- `TestDDPDBInit` (lines 1357-1381) — init_db is mocked in both tests
- `TestPendingTransitionsCreate` (lines 1389-1462)
- `TestPendingTransitionsAccumulateReward` (lines 1465-1493)
- `TestPendingTransitionsFinalize` (lines 1496-1558)
- `TestComputeValueCats` (lines 1566-1605)
- `TestToLearnerPerspective` (lines 1608-1647)
- `TestSignCorrectBootstrap` (lines 1650-1690)
- `TestMainEntryPoint` (lines 1693-1713)
- `TestDDPLeagueGuard` (lines 1716-1733)

INTEGRATION classes (move to `test_katago_loop_integration.py`):
- `katago_config` fixture (lines 211-249) — needed by integration tests
- `_with_league` helper (lines 326-341) — needed by integration tests
- `_make_league_config` helper (lines 440-446) — needed by integration tests
- `TestKataGoTrainingLoopInit` (lines 252-286)
- `TestKataGoTrainingLoopRun` (lines 287-324)
- `TestLeagueIntegration` (lines 344-381)
- `TestSplitMergeIntegration` (lines 383-438)
- `TestSeatRotation` (lines 449-540)
- `TestCheckResume` (lines 541-632)
- `TestRotateSeatIsolation` (lines 634-701)
- `TestCheckpointWrittenToDisk` (lines 925-959)
- `TestRotateSeat` (lines 1013-1191) — uses `league_config` fixture with real DB
- `TestSLToRLCheckpointHandoff` (lines 1193-1261)
- `TestCheckpointResumeRoundTrip` (lines 1264-1355)
- `TestFairnessInteractions` (lines 1736-1857)

- [ ] **Step 1: Create `tests/test_katago_loop_integration.py`**

Create the new file with:

1. The module docstring: `"""Integration tests for KataGoTrainingLoop (real DB, checkpoints, pool store)."""`
2. A module-level marker: `pytestmark = pytest.mark.integration`
3. All imports from the original file (copy the full import block from lines 1-16)
4. Import `_make_mock_katago_vecenv` and `_make_config` from the original: `from tests.test_katago_loop import _make_mock_katago_vecenv, _make_config`
5. The `katago_config` fixture (lines 211-249)
6. The `_with_league` helper (lines 326-341)
7. The `_make_league_config` helper (lines 440-446)
8. All 13 integration test classes listed above, in their original order

The file header should look like:

```python
"""Integration tests for KataGoTrainingLoop (real DB, checkpoints, pool store)."""

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from keisei.config import AppConfig, DisplayConfig, LeagueConfig, ModelConfig, TrainingConfig
from keisei.db import (
    init_db,
    read_metrics_since,
    read_training_state,
    update_training_progress,
    write_training_state,
)
from keisei.training.checkpoint import save_checkpoint
from keisei.training.distributed import DistributedContext
from keisei.training.katago_loop import KataGoTrainingLoop

from tests.test_katago_loop import _make_config, _make_mock_katago_vecenv

pytestmark = pytest.mark.integration
```

Note: check the integration test classes for any additional imports they use (e.g., `init_db`, `read_training_state`, `write_training_state`, `save_checkpoint`, `read_metrics_since`) and include them in the import block. Read the full file to confirm all needed imports.

Then paste all integration classes and their helper fixtures/functions.

- [ ] **Step 2: Remove integration classes from `test_katago_loop.py`**

Remove from `test_katago_loop.py`:
- The `katago_config` fixture (lines 211-249)
- The `_with_league` helper (lines 326-341)
- The `_make_league_config` helper (lines 440-446)
- All 13 integration test classes listed above

Keep:
- The module docstring (update to: `"""Unit tests for KataGoTrainingLoop (mocked I/O)."""`)
- All imports
- `_make_mock_katago_vecenv` function
- `_make_config` function
- All 17 unit test classes listed above

Clean up any imports that are no longer needed in the unit file (e.g., `LeagueConfig` may still be needed by some unit tests — check before removing).

- [ ] **Step 3: Verify both files parse correctly**

Run: `uv run python -c "import py_compile; py_compile.compile('tests/test_katago_loop.py', doraise=True); py_compile.compile('tests/test_katago_loop_integration.py', doraise=True); print('OK')"`

Expected: `OK`

- [ ] **Step 4: Run the unit tests**

Run: `uv run pytest tests/test_katago_loop.py -v --tb=short`

Expected: All ~47 unit tests pass.

- [ ] **Step 5: Run the integration tests**

Run: `uv run pytest tests/test_katago_loop_integration.py -v --tb=short`

Expected: All ~36 integration tests pass.

- [ ] **Step 6: Verify total test count is preserved**

Run: `uv run pytest tests/test_katago_loop.py tests/test_katago_loop_integration.py --co -q --no-header 2>&1 | tail -1`

Expected: `83 tests collected` (same as before the split)

- [ ] **Step 7: Commit**

```bash
git add tests/test_katago_loop.py tests/test_katago_loop_integration.py
git commit -m "refactor: split test_katago_loop into unit and integration files"
```

---

### Task 3: Tag integration test files with `pytestmark`

Apply `pytestmark = pytest.mark.integration` to 25 test files (24 existing + the new `test_katago_loop_integration.py` which was tagged in Task 2).

**Files to modify** (24 files — `test_katago_loop_integration.py` already done in Task 2):

For each file listed below, add two lines after the module docstring and before the first import:

```python
import pytest

pytestmark = pytest.mark.integration
```

If the file already imports `pytest`, just add the `pytestmark` line after all imports.

The pattern: find the last import line, add `pytestmark = pytest.mark.integration` after a blank line.

- [ ] **Step 1: Tag DB test files**

Add `pytestmark = pytest.mark.integration` to:
- `tests/test_db.py`
- `tests/test_db_schema_v2.py`
- `tests/test_db_gaps.py`

- [ ] **Step 2: Tag server test files**

Add `pytestmark = pytest.mark.integration` to:
- `tests/test_server.py`
- `tests/test_server_edge_cases.py`
- `tests/test_server_factory.py`
- `tests/test_server_gaps.py`

- [ ] **Step 3: Tag opponent/tournament test files**

Add `pytestmark = pytest.mark.integration` to:
- `tests/test_tournament.py`
- `tests/test_opponent_store.py`
- `tests/test_phase3_store.py`
- `tests/test_phase3_integration.py`

- [ ] **Step 4: Tag checkpoint test files**

Add `pytestmark = pytest.mark.integration` to:
- `tests/test_checkpoint.py`
- `tests/test_katago_checkpoint.py`

- [ ] **Step 5: Tag tiered pool test files**

Add `pytestmark = pytest.mark.integration` to:
- `tests/test_tiered_pool.py`
- `tests/test_tiered_pool_wiring.py`
- `tests/test_tier_managers.py`

- [ ] **Step 6: Tag historical/trainer test files**

Add `pytestmark = pytest.mark.integration` to:
- `tests/test_historical_gauntlet.py`
- `tests/test_historical_library.py`
- `tests/test_dynamic_trainer.py`
- `tests/test_phase3_managers.py`

- [ ] **Step 7: Tag Elo and transition test files**

Add `pytestmark = pytest.mark.integration` to:
- `tests/test_role_elo.py`
- `tests/test_transition.py`
- `tests/test_transition_gaps.py`

- [ ] **Step 8: Tag DDP integration test**

Add `pytestmark = pytest.mark.integration` to:
- `tests/integration/test_ddp_training.py`

- [ ] **Step 9: Verify marker filtering works**

Run: `uv run pytest -m "not integration" --co -q --no-header 2>&1 | tail -1`

Expected: approximately 685 tests collected (unit tests only).

Run: `uv run pytest -m integration --co -q --no-header 2>&1 | tail -1`

Expected: approximately 406 tests collected (integration tests only).

Run: `uv run pytest --co -q --no-header 2>&1 | tail -1`

Expected: 1091 tests collected (all tests — unchanged from before).

- [ ] **Step 10: Commit**

```bash
git add tests/test_db.py tests/test_db_schema_v2.py tests/test_db_gaps.py \
  tests/test_server.py tests/test_server_edge_cases.py tests/test_server_factory.py tests/test_server_gaps.py \
  tests/test_tournament.py tests/test_opponent_store.py tests/test_phase3_store.py tests/test_phase3_integration.py \
  tests/test_checkpoint.py tests/test_katago_checkpoint.py \
  tests/test_tiered_pool.py tests/test_tiered_pool_wiring.py tests/test_tier_managers.py \
  tests/test_historical_gauntlet.py tests/test_historical_library.py tests/test_dynamic_trainer.py tests/test_phase3_managers.py \
  tests/test_role_elo.py tests/test_transition.py tests/test_transition_gaps.py \
  tests/integration/test_ddp_training.py
git commit -m "chore: tag 24 integration test files with pytest.mark.integration"
```

---

### Task 4: Rename test files (batch 1 — no cross-references)

All renames use `git mv` to preserve history. These files are standalone — no other test file imports from them.

**Important:** After Task 3 committed the marker changes to these files under their old names, the renames here are pure `git mv` operations. No content changes needed.

- [ ] **Step 1: Rename DB test files**

```bash
git mv tests/test_db_schema_v2.py tests/test_db_league_schema.py
git mv tests/test_db_gaps.py tests/test_db_edge_cases.py
```

- [ ] **Step 2: Rename server test files**

```bash
git mv tests/test_server_edge_cases.py tests/test_server_diagnostics.py
git mv tests/test_server_gaps.py tests/test_server_websocket.py
```

- [ ] **Step 3: Rename model test files**

```bash
git mv tests/test_katago_model.py tests/test_se_resnet.py
git mv tests/test_katago_observation.py tests/test_katago_obs_channels.py
git mv tests/test_model_architecture_gaps.py tests/test_model_variants.py
git mv tests/test_model_gaps.py tests/test_model_degenerate_configs.py
```

- [ ] **Step 4: Rename checkpoint/compile test files**

```bash
git mv tests/test_checkpoint_gaps.py tests/test_checkpoint_optimizer_state.py
git mv tests/test_katago_checkpoint.py tests/test_checkpoint_architecture.py
git mv tests/test_compile.py tests/test_torch_compile.py
```

- [ ] **Step 5: Rename training/PPO test files**

```bash
git mv tests/test_concurrent_matches.py tests/test_match_pool.py
git mv tests/test_evaluate.py tests/test_evaluation.py
git mv tests/test_gae_padded.py tests/test_gae_batched.py
git mv tests/test_entropy_annealing.py tests/test_entropy_warmup_decay.py  
```

Wait — `test_entropy_annealing.py` was in the "keep" list in the design. Let me recheck. The design says to keep `test_entropy_annealing`. Remove that rename. Corrected step:

```bash
git mv tests/test_concurrent_matches.py tests/test_match_pool.py
git mv tests/test_evaluate.py tests/test_evaluation.py
git mv tests/test_gae_padded.py tests/test_gae_batched.py
```

- [ ] **Step 6: Rename phase3/opponent test files**

```bash
git mv tests/test_phase3_integration.py tests/test_phase3_rollout_wiring.py
git mv tests/test_phase3_managers.py tests/test_dynamic_manager.py
git mv tests/test_phase3_store.py tests/test_opponent_store_phase3.py
```

- [ ] **Step 7: Rename transition test files**

```bash
git mv tests/test_pending_transitions.py tests/test_split_merge_transitions.py
git mv tests/test_transition.py tests/test_sl_to_rl.py
git mv tests/test_transition_gaps.py tests/test_sl_to_rl_error_paths.py
```

- [ ] **Step 8: Rename remaining test files**

```bash
git mv tests/test_pytorch_audit_gaps.py tests/test_pytorch_training_gaps.py
git mv tests/test_pytorch_hot_path_gaps.py tests/test_pytorch_amp_pipeline.py
git mv tests/test_registry_gaps.py tests/test_registry_validation.py
git mv tests/test_tiered_pool_wiring.py tests/test_tiered_pool_phase3.py
git mv tests/test_tournament.py tests/test_league_tournament.py
```

- [ ] **Step 9: Verify all tests still collect**

Run: `uv run pytest --co -q --no-header 2>&1 | tail -1`

Expected: 1091 tests collected (same total as before renames).

- [ ] **Step 10: Verify no broken imports**

Run: `uv run pytest -x --tb=short -q 2>&1 | tail -5`

Expected: All tests pass. If any fail due to import errors from cross-file references, fix them.

- [ ] **Step 11: Commit**

```bash
git add -A tests/
git commit -m "refactor: rename 24 test files for clarity

Renames process-artifact names (_gaps) to content-descriptive names,
disambiguates generic names (test_compile → test_torch_compile), and
aligns file names with the classes/modules they actually test."
```

---

### Task 5: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v --tb=short 2>&1 | tail -20`

Expected: All 1091 tests pass.

- [ ] **Step 2: Verify marker split**

Run: `uv run pytest -m "not integration" --co -q --no-header 2>&1 | tail -1`

Expected: ~685 tests collected.

Run: `uv run pytest -m integration --co -q --no-header 2>&1 | tail -1`

Expected: ~406 tests collected.

- [ ] **Step 3: Verify unit + integration = total**

The two counts from step 2 should sum to 1091. If they don't, some test file is missing a marker or has a marker it shouldn't.

- [ ] **Step 4: Run unit tests only and check speed**

Run: `uv run pytest -m "not integration" -q --tb=short 2>&1 | tail -5`

Expected: Fast completion (these are all mocked/pure-logic tests).

- [ ] **Step 5: Commit spec as done**

No code change — just verify everything is clean:

Run: `git status`

Expected: Clean working tree, all changes committed.
