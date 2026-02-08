# Three-View Broadcast Architecture

**Date:** 2026-02-08  
**Decision Anchor:** `docs/architecture/decisions/ADR-001-intent-hierarchy-and-league-first-architecture.md`

## Required Views

1. **Training View**
   - A model learning through self-play.
   - Existing base: `keisei/webui/state_snapshot.py`, `keisei/webui/streamlit_app.py`.

2. **League Match View**
   - Two ranked league models playing each other.
   - Includes promotion/relegation context and match stakes.

3. **Lineage View**
   - Graph of model ancestry and relationships ("who learned from who").
   - Includes parent/child edges, branch points, promotions, and major milestones.

## Recommended Additional View

4. **Skill Differential View** (recommended)
   - Shows skill spread across active models and expected/observed match differentials.
   - Makes the progression claim legible: stronger models should generally beat weaker ones, with visible upset rates and promotion pressure.

5. **Model Profile View** (recommended)
   - Per-model "how it prefers to play" profile.
   - Includes opening repertoire tendencies, tactical/positional bias, and board heatmaps.

## Delivery Modes

The presentation layer must support:

1. **Multi-channel**
   - One view per channel/process (e.g., stream A/B/C).
2. **Split-screen**
   - One composite output with 2-4 simultaneous panes.
3. **Rotating**
   - Timed rotation across views for a single output channel.

## Proposed Presentation Architecture

### 1) View Producers

- `TrainingViewProducer`: emits training state snapshots.
- `LeagueViewProducer`: emits active ranked/exhibition match state and league table deltas.
- `LineageViewProducer`: emits ancestry graph state and update events.
- `SkillDifferentialViewProducer`: emits rating distribution, expected win probabilities, and upset metrics.
- `ModelProfileViewProducer`: emits per-model style fingerprints and heatmaps.

Suggested location:
- `keisei/webui/views/` (new package)

### 2) Broadcast State Hub

Central aggregator that stores latest state for all views and exposes one unified payload.

Suggested location:
- `keisei/webui/broadcast_state.py`

Responsibilities:
- Merge producer outputs.
- Timestamp/version each view payload.
- Expose health flags per producer.

### 3) Composer

Mode-aware composition engine:

- `single(view=<one of: training, league, lineage, skill_differential, model_profile>)`
- `split(layout=2up|3up|4up)`
- `rotate(order=[...], interval_seconds=N)`

Suggested location:
- `keisei/webui/composer.py`

### 4) Renderer

Streamlit UI consumes composed payload and renders:

- Training panel(s)
- League board + standings panel(s)
- Lineage graph panel
- Skill differential panel
- Model profile panel

Suggested updates:
- `keisei/webui/streamlit_app.py`

## Data Contracts

Use explicit JSON-serializable contracts:

1. `TrainingViewState`
2. `LeagueMatchViewState`
3. `LineageViewState`
4. `SkillDifferentialViewState` (recommended)
5. `ModelProfileViewState` (recommended)
6. `BroadcastStateEnvelope`

Minimal envelope shape:

```json
{
  "timestamp": 0,
  "mode": "split",
  "active_views": ["training", "league", "lineage", "skill_differential", "model_profile"],
  "training": {},
  "league": {},
  "lineage": {},
  "skill_differential": {},
  "model_profile": {},
  "health": {
    "training": "ok",
    "league": "ok",
    "lineage": "ok",
    "skill_differential": "ok",
    "model_profile": "ok"
  }
}
```

## Skill Differential View Requirements

1. Display current rating distribution (histogram or ordered ladder).
2. Show expected win probability from rating gap versus observed outcomes.
3. Highlight upset frequency by gap bucket (e.g., 0-50, 51-100 Elo).
4. Show promotion pressure indicators:
   - candidate model momentum
   - risk of demotion
   - confidence/uncertainty for current ranking.
5. Link differential metrics to recent league matches for explainability.

## Model Profile View Requirements

1. Selectable model identity (ranked model, checkpoint, lineage node).
2. Opening preference distribution:
   - most frequent first-move patterns
   - repertoire entropy (narrow vs broad opening pool).
3. Tactical/positional indicators:
   - capture rate, drop rate, promotion rate
   - average game length and termination profile.
4. Board activity heatmaps:
   - origin-square heatmap
   - destination-square heatmap
   - optional phase-specific heatmaps (opening/mid/endgame).
5. Style drift over lineage:
   - compare child profile against parent profile with divergence score.
6. Matchup fingerprint:
   - how this profile performs vs specific style clusters.

## League View Requirements

1. Identify both competitors by model ID/checkpoint lineage node.
2. Show rank, Elo delta, and promotion/relegation implications.
3. Publish match lifecycle state:
   - scheduled, in_progress, completed, failed.
4. Keep result provenance for replay and audit.

## Lineage View Requirements

Lineage must be event-backed, not inferred from filenames.

### Required lineage events

1. `checkpoint_created`
2. `checkpoint_promoted`
3. `league_match_completed`
4. `branch_created` (optional but recommended)

### Minimum lineage node fields

1. `model_id`
2. `checkpoint_path`
3. `parent_model_id` (nullable)
4. `created_at`
5. `source_run`
6. `training_step`
7. `promotion_status`

Suggested location:
- `keisei/evaluation/lineage/registry.py`

Persistence options:
- JSONL event log (simple)
- SQLite graph/event store (recommended for growth)

## Config Additions

Add to `WebUIConfig` or nested broadcast config:

1. `broadcast_mode: "single" | "split" | "rotate"`
2. `broadcast_views: list[str]`
3. `rotate_interval_seconds: int`
4. `default_single_view: "training" | "league" | "lineage"`
5. `broadcast_state_path: str`
6. `enable_skill_differential_view: bool`
7. `enable_model_profile_view: bool`
8. `model_profile_default_id: str | null`

Add to `EvaluationConfig` or nested league config:

1. `enable_continuous_exhibition: bool`
2. `promotion_policy: str`
3. `lineage_registry_path: str`

## Operational SLOs

1. Exhibition view update freshness: <= 5 seconds.
2. Rotation switch delay: <= 1 second.
3. Split-screen render degradation: no complete panel blackout on single-view failure.
4. Lineage consistency: no orphan node without explicit provenance event.
5. Skill-differential freshness: <= 1 match behind current league state.
6. Model-profile recompute lag: <= 2 completed matches for tracked models.

## Incremental Rollout

1. **Slice A:** Add lineage registry events at checkpoint save/promotion boundaries.
2. **Slice B:** Add league producer and league panel rendering.
3. **Slice C:** Add lineage producer and graph rendering.
4. **Slice D:** Add skill differential producer and panel.
5. **Slice E:** Add model profile producer and profile/heatmap panel.
6. **Slice F:** Add composer modes (single/split/rotate) and multi-channel process templates.

This order gives immediate value (league + lineage visibility) while preserving existing training dashboard behavior.
