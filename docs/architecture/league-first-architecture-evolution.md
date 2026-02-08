# League-First Architecture Evolution Plan

**Date:** 2026-02-08  
**Inputs:** `README.md`, `docs/DESIGN.md`, `docs/architecture-assessment.md`, `docs/architecture-consensus-dissenting-viewpoint-paper.md`  
**Decision Anchor:** `docs/architecture/decisions/ADR-001-intent-hierarchy-and-league-first-architecture.md`

## Objective

Evolve the current training-centric integration into a league-first runtime where:

1. Shogi is continuously watchable (exhibition lane).
2. DRL evaluation is rich and multi-strategy (formal lane).
3. LLM-coding evaluation remains process telemetry, not runtime control logic.

## Current Constraints (Code-Observed)

1. `keisei/training/callbacks.py` callbacks receive full `Trainer`, permitting broad cross-manager side effects.
2. `keisei/training/callback_manager.py` schedules periodic evaluation but not an explicit exhibition runtime lane.
3. `keisei/evaluation/core_manager.py` can run multiple strategies but is mostly called as periodic point-in-time evaluation.
4. `keisei/webui/streamlit_manager.py` publishes trainer snapshot state but not a dedicated league/exhibition state contract.

## Target Runtime Shape

### Lane A: Training-Critical Evaluation

- Purpose: policy quality gating, checkpoint promotion, Elo updates.
- Trigger: deterministic schedule aligned with `steps_per_epoch`.
- Failure mode: degrade gracefully, training continues with clear warning.

### Lane B: Continuous Exhibition

- Purpose: always-available spectator match stream.
- Trigger: independent recurring schedule (wall clock and/or timesteps).
- Failure mode: auto-restart with backoff; never silently dead.

### Shared Services

1. **League Orchestrator**
   - Owns scheduling, queueing, and lifecycle of evaluation/exhibition jobs.
2. **Strategy Router**
   - Selects from existing strategy implementations (`single_opponent`, `tournament`, `ladder`, `benchmark`, `custom`) using explicit policy.
3. **Result Publisher**
   - Emits league snapshot artifacts for WebUI and logs.
4. **Rating/Registry Service**
   - Centralized Elo/ranking updates and provenance metadata.
5. **Skill Differential Service**
   - Computes expected-vs-observed performance across rating gaps and promotion pressure indicators.
6. **Model Profiling Service**
   - Builds per-model style fingerprints and board heatmaps from league/training games.

## Architectural Changes

### 1) Introduce Typed Contracts at Training/Evaluation Boundary

Add explicit DTOs:

- `EvaluationRequest` (lane, strategy, trigger metadata, model reference)
- `EvaluationOutcome` (summary stats, rating updates, artifact refs)
- `LeagueSnapshot` (active exhibition match, standings, recent results, health)
- `SkillDifferentialSnapshot` (rating spread, expected win rates, upset rates, momentum)
- `ModelProfileSnapshot` (opening tendencies, style indicators, heatmaps, lineage drift)

Suggested location:
- `keisei/evaluation/league/contracts.py`

Why:
- Replaces implicit full-object coupling.
- Makes callback behavior testable and auditable.

### 2) Replace Full-`Trainer` Callback Reach-Through with Narrow Context

Create a callback context object containing only needed fields:

- `global_timestep`
- checkpoint save hook
- evaluation enqueue hook
- logging hook
- run metadata

Suggested updates:
- `keisei/training/callbacks.py`
- `keisei/training/callback_manager.py`
- `keisei/training/training_loop_manager.py`

Why:
- Keeps training manager boundaries intact.
- Prevents callback accretion into architecture bypasses.

### 3) Add League Orchestration Layer

Add a runtime manager that coordinates two lanes and reuses existing evaluators/background tournament support.

Suggested location:
- `keisei/evaluation/league/runtime.py`

Integration points:
- Invoked from callback manager (for scheduled triggers).
- Reads from `EvaluationConfig`.
- Writes league snapshot for WebUI.

Why:
- Makes exhibition “always running” a first-class operational behavior.

### 4) Extend Config for League and Exhibition Controls

Add explicit settings to `EvaluationConfig` (or nested `LeagueConfig`):

- `enable_league_runtime: bool`
- `enable_continuous_exhibition: bool`
- `exhibition_strategy: Literal[...]`
- `exhibition_interval_seconds: int`
- `exhibition_games_per_cycle: int`
- `exhibition_max_concurrent_games: int`
- `league_snapshot_path: str`

Add cross-field validation in `AppConfig`:

- interval alignment where required
- strategy compatibility checks
- exhibition enabled implies league runtime enabled

Why:
- Avoids hidden behavior and runtime misconfiguration.

### 5) Upgrade WebUI Data Contract for League View

Add league fields to snapshot generation:

- active exhibition pairing/status
- current standings/rating deltas
- recent game outcomes
- league runtime health

Suggested files:
- `keisei/webui/state_snapshot.py`
- `keisei/webui/streamlit_app.py`
- `keisei/webui/streamlit_manager.py`

Why:
- Supports the “watchable Shogi + DRL in motion” objective directly.

### 5b) Add Skill Differential View Contract

Add fields that explicitly represent cross-skill dynamics:

- rating distribution across active models
- expected win probability by Elo gap bucket
- observed win probability by Elo gap bucket
- upset rate and trend
- promotion/demotion pressure score

Suggested files:
- `keisei/webui/state_snapshot.py`
- `keisei/webui/streamlit_app.py`
- `keisei/evaluation/analytics/` (reuse/extend existing analytics modules)

Why:
- Directly demonstrates that models improve against stronger competition and that ranking gaps translate into visible match differentials.

### 5c) Add Model Profile and Heatmap Contract

Add fields that describe model play preference:

- opening move distribution and repertoire entropy
- capture/drop/promotion behavior rates
- average game length and termination reason mix
- origin/destination square heatmaps
- lineage style-drift metrics (child vs parent profile divergence)

Suggested files:
- `keisei/evaluation/analytics/` (extend existing analytics pipeline)
- `keisei/webui/state_snapshot.py`
- `keisei/webui/streamlit_app.py`
- `keisei/evaluation/lineage/registry.py` (for parent-child linkage)

Why:
- Supports the narrative that models at different skill levels exhibit distinct styles and that style evolves as models learn against stronger opponents.

### 6) Keep LLM Goal Subordinate and Measurable

Add process telemetry and reporting, not runtime coupling:

- Tag architecture/code changes with source provenance (human, LLM-assisted, mixed).
- Track defect escape rate and lead time by provenance.
- Publish findings in docs or CI reports.

Suggested location:
- `docs/experiments/llm-coding-effectiveness.md`

Why:
- Evaluates the third leg without distorting product architecture.

## Migration Phases

1. **Phase 1 (Boundary hardening):**
   - Callback context, typed request/outcome contracts.
2. **Phase 2 (League runtime MVP):**
   - Continuous exhibition scheduler + snapshot publishing.
3. **Phase 3 (Strategy expansion):**
   - Promote tournament/ladder/benchmark/custom into supported league policies.
4. **Phase 4 (Skill differential telemetry):**
   - Compute and publish expected-vs-observed performance by rating gap.
5. **Phase 5 (Model profile telemetry):**
   - Compute and publish per-model style profiles and heatmaps.
6. **Phase 6 (Operational maturity):**
   - Health checks, restart/backoff, SLOs, and CI quality gates.

## Required Test Gates

1. Unit tests for contracts, scheduler logic, and callback context behavior.
2. Integration tests for:
   - training lane evaluation trigger
   - continuous exhibition lane persistence/recovery
   - snapshot publication consumed by WebUI
   - skill-differential metric consistency with league outcomes
   - model-profile metric consistency (same game logs produce stable profile outputs)
3. Strategy tests for all retained strategies.
4. CI gate re-enabled for lint/type/test and coverage trend tracking.

## Minimal First Slice (Recommended)

If we want momentum without broad refactor risk:

1. Add `LeagueSnapshot` and exhibition scheduler only.
2. Publish snapshot to WebUI.
3. Keep existing callback signature temporarily.
4. Then refactor callback context once exhibition loop is stable.

This delivers visible league behavior quickly while preserving an incremental path to cleaner boundaries.
