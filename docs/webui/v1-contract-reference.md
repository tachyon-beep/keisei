# v1 Broadcast State Contract Reference

**Schema version:** `v1.0.0`
**Status:** Frozen (Move 1 complete)

## Envelope Structure

The broadcast state file is a JSON document conforming to `BroadcastStateEnvelope`
(defined in `keisei/webui/view_contracts.py`).

```
BroadcastStateEnvelope
├── schema_version: str          "v1.0.0"
├── timestamp: float             UNIX epoch seconds
├── speed: float                 Timesteps/second
├── mode: str                    Evaluation strategy or "training_only"
├── active_views: list[str]      Currently populated views
├── health: dict[str, str]       Per-view health status
├── training: TrainingViewState  Training view payload (always present in v1)
├── pending_updates: dict        Scalar-only ephemeral updates
│
└── (optional, null in v1)
    ├── league: LeagueViewState | null
    ├── lineage: LineageViewState | null
    ├── skill_differential: SkillDifferentialViewState | null
    └── model_profile: ModelProfileViewState | null
```

### Required Fields

| Field | Type | Description |
|---|---|---|
| `schema_version` | `str` | Semver-like version string, currently `"v1.0.0"` |
| `timestamp` | `float` | UNIX epoch of snapshot creation |
| `speed` | `float` | Training speed in timesteps/second |
| `mode` | `str` | Active evaluation strategy (`"training_only"`, `"single_opponent"`, `"tournament"`, `"ladder"`, `"benchmark"`, `"custom"`) |
| `active_views` | `list[str]` | View names with populated data (v1: `["training"]`) |
| `health` | `dict[str, str]` | Health status for each of the 5 view categories |
| `training` | `dict` | Training view payload (see below) |
| `pending_updates` | `dict[str, scalar]` | Ephemeral scalar values (int, float, str, bool, None only) |

### Health Statuses

| Status | Meaning |
|---|---|
| `ok` | Data is fresh and valid |
| `stale` | Data exists but is older than 30 seconds |
| `missing` | View has no producer yet |
| `error` | Producer encountered an error |

### View Categories

Health is tracked for 5 view categories:
`training`, `league`, `lineage`, `skill_differential`, `model_profile`

## Training View Payload

```
TrainingViewState
├── board_state: BoardState | null     (null between episodes)
├── metrics: MetricsState              (always present)
├── step_info: StepInfo | null         (null before first episode)
├── buffer_info: BufferInfo | null     (null before buffer creation)
└── model_info: ModelInfo              (always present)
```

### MetricsState

| Field | Type |
|---|---|
| `global_timestep` | `int` |
| `total_episodes` | `int` |
| `black_wins` | `int` |
| `white_wins` | `int` |
| `draws` | `int` |
| `processing` | `bool` |
| `learning_curves` | `LearningCurves` |
| `win_rates_history` | `list[dict]` |
| `hot_squares` | `Any` |

### LearningCurves (last 50 values)

`policy_losses`, `value_losses`, `entropies`, `kl_divergences`,
`clip_fractions`, `learning_rates`, `episode_lengths`, `episode_rewards`

### BoardState

9x9 grid of `PieceInfo | null`, plus `current_player`, `move_count`,
`game_over`, `winner`, `black_hand`, `white_hand`.

### StepInfo

`move_log` (last 20), capture/drop/promotion counts per side.

### BufferInfo

`size`, `capacity` (experience buffer fill level).

### ModelInfo

`gradient_norm` (float).

## Example Payload

```json
{
  "schema_version": "v1.0.0",
  "timestamp": 1707400000.0,
  "speed": 42.5,
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
    "board_state": null,
    "metrics": {
      "global_timestep": 100,
      "total_episodes": 5,
      "black_wins": 3,
      "white_wins": 1,
      "draws": 1,
      "processing": false,
      "learning_curves": {
        "policy_losses": [0.5, 0.4],
        "value_losses": [0.3, 0.2],
        "entropies": [1.0, 0.9],
        "kl_divergences": [0.01, 0.02],
        "clip_fractions": [0.1, 0.08],
        "learning_rates": [0.0003, 0.0003],
        "episode_lengths": [120.0, 130.0],
        "episode_rewards": [0.5, 0.6]
      },
      "win_rates_history": [],
      "hot_squares": []
    },
    "step_info": null,
    "buffer_info": {"size": 50, "capacity": 2048},
    "model_info": {"gradient_norm": 1.23}
  },
  "pending_updates": {"epoch": 3, "lr": 0.0003}
}
```

## Schema Evolution Policy

**Decision Freeze #1 — Versioning:**

- `schema_version` uses semver-like strings: `v{major}.{minor}.{patch}`
- **Patch** (e.g., v1.0.1): Optional field additions only. No migration required.
- **Minor** (e.g., v1.1.0): New required fields or semantic changes. Requires migration note.
- **Major** (e.g., v2.0.0): Breaking structural changes. Requires full migration guide.

**Renderers must treat unknown fields and mode values as opaque** (forward-compatible).

## Migration Note Template

When bumping the schema version, add an entry below:

```
### v1.0.0 → v1.x.x (date)

**Change:** [describe what changed]
**Migration:** [what consumers need to do]
**Affected views:** [which views changed]
**Breaking:** yes/no
```

## Decision Freezes

| DF# | Title | Rule |
|---|---|---|
| #1 | Schema Versioning | Semver-like strings; patch = optional only |
| #2 | Health Semantics | 5 view categories; 30s stale threshold |
| #3 | pending_updates | Scalar values only (int, float, str, bool, None) |
| #4 | Lineage Persistence | Backed by append-only JSONL events |
| #6 | Mode Handling | Unknown modes treated as opaque strings |

## Source Files

| File | Role |
|---|---|
| `keisei/webui/view_contracts.py` | Contract type definitions and validation |
| `keisei/webui/state_snapshot.py` | Producer (builds envelope from trainer) |
| `keisei/webui/envelope_parser.py` | Consumer access layer (parser) |
| `keisei/webui/streamlit_app.py` | Renderer (reads through parser) |
| `tests/webui/test_contract_schema_lock.py` | Schema drift detection tests |
| `tests/webui/test_view_contracts.py` | Contract validation tests |
| `tests/webui/test_snapshot_envelope.py` | Producer envelope tests |
| `tests/webui/test_envelope_parser.py` | Parser access layer tests |
| `tests/webui/test_renderer_fallbacks.py` | Renderer fallback tests |
