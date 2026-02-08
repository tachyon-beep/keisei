# Moves 1-3 Execution Readiness Spikes

Date: 2026-02-08  
Scope: `keisei-eva` (Move 1), `keisei-tw5` (Move 2), `keisei-35e` (Move 3), and program gate `keisei-qhg`.

## Objective

Reduce implementation risk and ambiguity before coding by freezing critical defaults, mapping cross-epic dependencies, and defining bead-level ready checks.

## Method

Deep-dive pass on current integration points:

- WebUI snapshot and renderer:
  - `keisei/webui/state_snapshot.py`
  - `keisei/webui/streamlit_app.py`
  - `keisei/webui/streamlit_manager.py`
- Training callback and execution boundary:
  - `keisei/training/callbacks.py`
  - `keisei/training/callback_manager.py`
  - `keisei/training/training_loop_manager.py`
- Evaluation lineage-adjacent components:
  - `keisei/evaluation/core_manager.py`
  - `keisei/evaluation/opponents/opponent_pool.py`
  - `keisei/evaluation/opponents/elo_registry.py`

## Current-State Findings

### Move 1 (`keisei-eva`) baseline

- Snapshot contract is currently ad hoc and unversioned (`state_snapshot.py` emits free-form keys).
- Renderer reads raw keys directly (`streamlit_app.py`), with no parser or schema gate.
- Stale detection exists only in UI (`streamlit_app.py`) and is not part of payload health semantics.

### Move 2 (`keisei-tw5`) baseline

- Checkpoint creation and evaluation events exist as runtime behavior but are not persisted as append-only lineage events.
- Opponent/Elo state is persisted, but lineage ancestry/provenance is not event-backed.

### Move 3 (`keisei-35e`) baseline

- Callback path has full `Trainer` reach-through (9 distinct attributes accessed); this maximizes convenience but increases coupling risk.
- Callbacks are **bidirectional**: they both read from and write to trainer state. Specifically, `EvaluationCallback` and `AsyncEvaluationCallback` write `trainer.evaluation_elo_snapshot` (a `Dict[str, Any] | None` declared at `trainer.py:158`). The `CallbackContext` migration must account for this output channel, not just input reads.
- Async and sync callback paths are both active; compatibility must be preserved while narrowing interfaces.

## Decision Freezes (Risk Reduction)

These defaults should be treated as frozen for initial implementation unless explicitly changed in bead notes.

1. Broadcast schema version
- `schema_version` format: `"v1.0.0"` string (semver-like).
- Patch-level additions allowed only for optional fields.
- Required-field or semantic-breaking changes require minor/major bump and migration note.

2. Envelope health semantics
- Health map keys: `training`, `league`, `lineage`, `skill_differential`, `model_profile`.
- Status enum: `ok | stale | missing | error`.
- Producer-side stale threshold default: 30 seconds, matching current UI behavior.
- Note: current snapshot sections `board_state`, `metrics`, `step_info`, `buffer_info`, and `model_info` all fold into the `training` view. They are sub-keys of the `training` payload, not independent health-tracked views. Only top-level view categories appear in the health map.

3. Pending updates policy
- `pending_updates` remains scalar-only (`int|float|str|bool|null`) in v1.
- Invalid values are dropped, never fatal to snapshot writes.

4. Lineage event persistence
- Append-only JSONL as v1 storage contract.
- Event ids are monotonic with timestamp + UUID suffix to avoid collisions.
- No inferred parent links from filenames; parentage requires explicit event field.

5. Callback boundary migration
- Introduce `CallbackContext` + adapter first.
- Keep temporary trainer-backed shim during transition for parity.
- Remove shim only after parity tests pass for sync and async paths.
- `CallbackContext` must include a structured output slot (e.g. `outputs: Dict[str, Any]`) to replace the current pattern where callbacks write directly to `trainer.evaluation_elo_snapshot`. Callbacks must not mutate trainer state directly under the new interface.

6. Broadcast mode values
- `mode` field in the v1 envelope reflects the active evaluation strategy configuration.
- Frozen v1 values: `"training_only"` (no evaluation), `"single_opponent"`, `"tournament"`, `"ladder"`, `"benchmark"`, `"custom"`.
- These align with `EvaluationStrategy` enum in `config_schema.py`. `"training_only"` covers the case where evaluation is disabled.
- Unknown or future mode values must be treated as opaque strings by the renderer (no crash on unrecognized mode).

## Proposed v1 Contract Shape (Move 1)

```json
{
  "schema_version": "v1.0.0",
  "timestamp": 0.0,
  "speed": 0.0,
  "mode": "single_opponent",
  "active_views": ["training"],
  "health": {
    "training": "ok",
    "league": "missing",
    "lineage": "missing",
    "skill_differential": "missing",
    "model_profile": "missing"
  },
  "training": {
    "board_state": {},
    "metrics": {},
    "step_info": {},
    "buffer_info": {},
    "model_info": {}
  },
  "league": null,
  "lineage": null,
  "skill_differential": null,
  "model_profile": null,
  "pending_updates": {}
}
```

### Current-to-v1 Key Mapping (Frozen)

The existing `build_snapshot()` output maps into the v1 envelope as follows:

| Current top-level key | v1 location | Notes |
|---|---|---|
| `timestamp` | `envelope.timestamp` | Unchanged |
| `speed` | `envelope.speed` | Moved to envelope top-level (required) |
| `board_state` | `envelope.training.board_state` | Nullable (null when no game active) |
| `metrics` | `envelope.training.metrics` | Always present |
| `step_info` | `envelope.training.step_info` | Nullable (null before first episode) |
| `buffer_info` | `envelope.training.buffer_info` | Nullable (null before buffer init) |
| `model_info` | `envelope.training.model_info` | Always present |
| `pending_updates` | `envelope.pending_updates` | Scalar-only, unchanged |
| *(new)* | `envelope.schema_version` | Required `"v1.0.0"` string |
| *(new)* | `envelope.mode` | From evaluation config strategy |
| *(new)* | `envelope.active_views` | List of non-null view keys |
| *(new)* | `envelope.health` | Status map for all view categories |

The `training` view is the only populated view in v1. Other views (`league`, `lineage`, `skill_differential`, `model_profile`) are `null` with health status `"missing"` until their respective Moves deliver producers.

## Bead-Level Readiness Checks

### `keisei-eva.1.*`

Ready when:

- `view_contracts.py` defines envelope and all view types.
- Required/optional fields documented in module docstring.
- Validation helpers cover required key presence and scalar sanitization.

Primary risk:

- Contract shape drifts from existing snapshot keys and breaks renderer.

Mitigation:

- Add canonical fixture set (valid + invalid) before snapshot refactor.

### `keisei-eva.2.*`

Ready when:

- `build_snapshot()` assembles via contract helper end-to-end.
- Health map is always present and populated.
- Atomic writes are preserved unchanged.
- Existing `tests/integration/test_webui_state_snapshot.py` tests pass without modification (regression gate).

Primary risk:

- Hidden callsites expect legacy keys.
- Existing integration tests assert on current key structure and would fail silently if renamed without updating tests.

Mitigation:

- Temporary adapter for legacy consumers if any are discovered during integration.
- Run `uv run pytest -q tests/integration/test_webui_state_snapshot.py` as a mandatory pre-merge gate for any `state_snapshot.py` change.

### `keisei-eva.3.*`

Ready when:

- Renderer reads through contract parser layer only.
- Missing optional views render placeholder states without exceptions.
- Stale/missing data warnings are parser-driven, not ad hoc.

Primary risk:

- Renderer crash loops on partial snapshots.

Mitigation:

- Parser defaults + stale-safe rerun path tests.

### `keisei-eva.4.*`

Ready when:

- Required-field and schema-version drift tests fail correctly on breakage.
- Renderer fallback tests cover absent optional views and stale envelope states.
- Contract reference doc includes migration template.

Primary risk:

- No dedicated `tests/webui/` currently exists.

Mitigation:

- Add `tests/webui/` and keep integration tests in `tests/integration/` where needed.

### `keisei-tw5.1.*` to `keisei-tw5.2.*`

Ready when:

- Lineage event schema frozen for checkpoint and parent linkage.
- Registry append/read API defined with idempotency behavior.

Primary risk:

- Duplicate or reordered events under retries.

Mitigation:

- Deterministic event id and replay order tests.

### `keisei-35e.1.*` to `keisei-35e.2.*`

Ready when:

- `CallbackContext` fields/hook protocol frozen.
- `CallbackContext` includes a structured output slot for callback results (replacing direct `trainer.evaluation_elo_snapshot` writes).
- CallbackManager can execute callbacks via context adapter without behavior regressions.
- Bidirectional data flow is tested: inputs (timestep, model handles, etc.) AND outputs (evaluation snapshots, metrics) match legacy behavior.

Primary risk:

- Callback behavior changes silently during migration.
- Callback write paths (`trainer.evaluation_elo_snapshot = snapshot`) are missed during context migration, causing evaluation Elo data to silently stop propagating.

Mitigation:

- Parity tests comparing legacy trainer-access path vs context path for both read and write operations.
- Explicit test that `evaluation_elo_snapshot` data produced via context path reaches the same consumer as the legacy direct-write path.

## Cross-Epic Dependency Truths

1. `keisei-eva.3.2` and `keisei-eva.4.2` are correctly blocked by `keisei-tw5.4.3` (lineage snapshot helper).
2. `keisei-tw5.3.*` should remain deferred until `keisei-35e.3.2` to avoid callback API rework.
3. Critical first implementation lane remains:
- `keisei-eva.1.1`, `keisei-eva.1.2`
- `keisei-tw5.1.1`, `keisei-tw5.1.2`
- `keisei-35e.1.1`

## Test Gate Plan (Minimum)

1. Contract and snapshot:
- `uv run pytest -q tests/integration/test_webui_state_snapshot.py`
- Add new: `uv run pytest -q tests/webui`

2. Callback boundary:
- `uv run pytest -q tests/unit/test_callbacks.py tests/unit/test_callback_manager.py`

3. Regression safety:
- `uv run pytest -q tests/unit/test_review_regressions.py`

## Spike Output Summary

This pass reduces risk by replacing unspecified behavior with frozen defaults for:

- schema versioning policy
- health/status semantics and current-key-to-v1 mapping
- broadcast mode values (aligned to EvaluationStrategy enum)
- pending update sanitization
- lineage event persistence contract
- callback migration strategy (including bidirectional data flow)

The bead graph is implementable now with fewer unknowns and explicit gating conditions for each lane.

---

*Revision 2 (2026-02-08): Added current-to-v1 key mapping table, frozen mode values, `speed` field in contract shape, callback write-path analysis, and integration test regression gate for keisei-eva.2.*
