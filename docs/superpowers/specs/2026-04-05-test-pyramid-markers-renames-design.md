# Test Pyramid Markers & Renames

**Date:** 2026-04-05
**Status:** Approved

## Problem

The test suite has 1,091 tests across 63 files with no structural classification. 99% live in a flat `tests/` directory. There's no way to run "just the fast tests" during development — every `pytest` invocation runs DB-touching integration tests alongside pure unit tests.

The effective distribution (~63% unit, ~37% integration) is reasonable, but invisible. Additionally, many test file names are misleading or describe process artifacts (`_gaps`) rather than content.

## Design

### Part 1: Integration marker infrastructure

Add `integration` marker to `pyproject.toml` alongside the existing `slow` marker:

```toml
markers = [
    "slow: marks tests as slow-running (deselect with '-m \"not slow\"')",
    "integration: marks tests requiring DB, filesystem, or subprocess resources (deselect with '-m \"not integration\"')",
]
```

### Part 2: Tag integration tests

Apply module-level `pytestmark = pytest.mark.integration` to files that create real SQLite databases, spawn subprocesses, or perform significant filesystem I/O.

**Integration files (25):**
- DB: `test_db.py`, `test_db_league_schema.py` (renamed), `test_db_edge_cases.py` (renamed)
- Server: `test_server.py`, `test_server_diagnostics.py` (renamed), `test_server_factory.py`, `test_server_websocket.py` (renamed)
- Opponent/tournament: `test_league_tournament.py` (renamed), `test_opponent_store.py`, `test_opponent_store_phase3.py` (renamed), `test_phase3_rollout_wiring.py` (renamed)
- Training loop: `test_katago_loop_integration.py` (new, split from monolith), `test_katago_checkpoint.py` (renamed to `test_checkpoint_architecture.py`)
- Checkpoint: `test_checkpoint.py`
- Tiered pool: `test_tiered_pool.py`, `test_tiered_pool_phase3.py` (renamed), `test_tier_managers.py`
- Historical: `test_historical_gauntlet.py`, `test_historical_library.py`
- Trainer: `test_dynamic_trainer.py`, `test_dynamic_manager.py` (renamed)
- Elo: `test_role_elo.py`
- Transition: `test_sl_to_rl.py` (renamed), `test_sl_to_rl_error_paths.py` (renamed)
- DDP: `integration/test_ddp_training.py`

### Part 3: Split `test_katago_loop.py`

The 1,857-line monolith contains 83 tests mixing mocked unit tests with DB-touching integration tests. Split into:
- `test_katago_loop.py` — unit tests with mocked VecEnv, config validation, no DB
- `test_katago_loop_integration.py` — tests creating real DB, checkpoints, full training steps

### Part 4: Rename test files

24 renames for clarity:

| Current | New |
|---------|-----|
| `test_checkpoint_gaps.py` | `test_checkpoint_optimizer_state.py` |
| `test_compile.py` | `test_torch_compile.py` |
| `test_concurrent_matches.py` | `test_match_pool.py` |
| `test_db_schema_v2.py` | `test_db_league_schema.py` |
| `test_db_gaps.py` | `test_db_edge_cases.py` |
| `test_evaluate.py` | `test_evaluation.py` |
| `test_gae_padded.py` | `test_gae_batched.py` |
| `test_katago_checkpoint.py` | `test_checkpoint_architecture.py` |
| `test_katago_model.py` | `test_se_resnet.py` |
| `test_katago_observation.py` | `test_katago_obs_channels.py` |
| `test_model_architecture_gaps.py` | `test_model_variants.py` |
| `test_model_gaps.py` | `test_model_degenerate_configs.py` |
| `test_pending_transitions.py` | `test_split_merge_transitions.py` |
| `test_phase3_integration.py` | `test_phase3_rollout_wiring.py` |
| `test_phase3_managers.py` | `test_dynamic_manager.py` |
| `test_phase3_store.py` | `test_opponent_store_phase3.py` |
| `test_pytorch_audit_gaps.py` | `test_pytorch_training_gaps.py` |
| `test_pytorch_hot_path_gaps.py` | `test_pytorch_amp_pipeline.py` |
| `test_registry_gaps.py` | `test_registry_validation.py` |
| `test_server_edge_cases.py` | `test_server_diagnostics.py` |
| `test_server_gaps.py` | `test_server_websocket.py` |
| `test_tiered_pool_wiring.py` | `test_tiered_pool_phase3.py` |
| `test_tournament.py` | `test_league_tournament.py` |
| `test_transition.py` | `test_sl_to_rl.py` |
| `test_transition_gaps.py` | `test_sl_to_rl_error_paths.py` |

## Out of Scope

- E2E tests (deferred until training pipeline stabilizes)
- Directory reorganization (markers are sufficient)
- Test logic changes (markers and renames only, no behavioral changes)

## Verification

1. `pytest -m "not integration"` runs ~685 unit tests cleanly
2. `pytest -m integration` runs ~406 integration tests cleanly
3. `pytest` (no filter) runs all 1,091 tests — no regressions
4. No test is accidentally lost during renames or the katago_loop split
