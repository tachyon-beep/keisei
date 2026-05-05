# Keisei Architecture Analysis — Final Report

**Workspace:** `docs/arch-analysis-2026-05-05-1007/`
**Date:** 2026-05-05
**Inputs:** `00-coordination.md`, `01-discovery-findings.md`, `02-subsystem-catalog.md` (20 entries), `temp/validation-catalog.md` (PASS-WITH-FIXES).

## Executive Summary

Keisei is a deep-RL training system for Shogi, structured as a Rust core with Python orchestration and a Svelte spectator dashboard. The Rust workspace (`shogi-core` ~8.4 KLoC, `shogi-gym` ~6 KLoC) implements rules, move generation, and a vectorised RL environment exposed to Python via PyO3. The Python harness (~16 KLoC across `keisei/training`, `keisei/sl`, `keisei/server`, `keisei/showcase`, plus `db.py` and `config.py`) drives training, runs an in-process league, ingests SL data, serves a FastAPI dashboard, and operates a headless showcase sidecar. The WebUI is a Svelte 4 SPA (29 components, 10 stores) consuming a single WebSocket. Eight subsystems (A–H, see §1) sit behind three hard integration boundaries: a 9-PyClass FFI surface, a 21-table SQLite database in WAL mode, and a `/ws` WebSocket with 12 server-to-client message types.

The repository is in a **post-2026-04-01 rebuild** following migration of the engine from pure Python to Rust. Per project memory, the Rust engine plan is currently a standalone burn-in harness; full Keisei integration is queued as a later phase. Treat this report as a snapshot of an evolving system, not a steady-state architecture.

The headline strengths are unusually clean for an early-rebuild codebase: the Rust core has zero runtime deps and is conservatively isolated from PyO3 (gym crate only); configuration is centralised in 17 frozen dataclasses with cross-registry validation; the DB-as-message-bus discipline is real (WebUI never opens SQLite, training never opens a socket); and the WebSocket layer has had concrete correctness fixes (per-connection `asyncio.Lock`, host allow-list re-checked for WS scope).

The headline risks cluster in the league/tournament subsystem (C3) and the single-file scale of the foundational layer. The validation gate flagged no fabrications and no boundary leaks; three numeric off-by-N edits were applied. The substantive risks — non-atomic match-result recording across separate transactions (`tournament.py:352–460`, P1 `keisei-fa604bad63`), two parallel result-recording paths (in-process `LeagueTournament` vs sidecar `tournament_runner`), and the unsafe-soundness contract in `vec_env.rs` resting on doc-only invariants — are real but well-bounded. Subsequent sections give the architect joining the project a calibrated map of where to spend attention first.

## 1. Architecture at a Glance

Eight subsystems, grouped into four user-selected buckets:

- **A. `shogi-core`** (Rust, ~8,400 LoC, stable). Pure-Rust rules engine — position, move generation, attack maps, special-rule enforcement, Zobrist, SFEN. Zero deps. *Ref: catalog A.*
- **B. `shogi-gym`** (Rust + PyO3, ~6,000 LoC, stable). Vectorised RL environment exposing 9 PyClasses; rayon-parallel `VecEnv.step` with `catch_unwind` per env. *Ref: catalog B.*
- **C. `keisei.training`** (Python, ~11,500 LoC, dense / hot-spot). PPO/KataGo trainer + multi-tier league + concurrent matches + sidecar tournament. Split in catalog into C1a (PPO core), C1b (loop), C2 (models), C3 (league — the densest cluster), C4 (style/features), C5 (eval CLI). *Ref: catalog C1a–C5.*
- **D. `keisei.sl`** (Python, ~1,070 LoC, clean one-way dependency). SL data prep + `SLTrainer`, fed into RL via `transition.sl_to_rl`. *Ref: catalog D.*
- **E. `keisei.config` + `keisei.db`** (Python, 759 + 1,158 LoC, foundational). 17 typed config dataclasses; SQLite with v8 schema and chained migrations. *Ref: catalog E1, E2.*
- **F. `keisei.server`** (Python, 658 LoC, stable). FastAPI app: 4 routes, `/ws` WebSocket protocol, 12 server→client messages, 4 client→server commands. *Ref: catalog F.*
- **G. `keisei.showcase`** (Python, ~790 LoC, stable). Headless sidecar runner + showcase DB ops + CPU inference. *Ref: catalog G.*
- **H. `webui`** (Svelte 4, 29 components / 10 stores). Four-tab SPA: Training / League / Showcase / About; single chart library (`uplot`), single WS client, one outbound command surface. *Ref: catalog H1–H7.*

Diagram 1 (C4 L1 system context) and Diagram 2 (C4 L2 container view) in `03-diagrams.md` render the eight subsystems and the three integration boundaries explicitly.

## 2. How the System Fits Together

### 2.1 The PyO3 FFI surface

`shogi_gym._native` exposes nine PyClasses: `VecEnv`, `SpectatorEnv`, `DefaultActionMapper`, `SpatialActionMapper`, `DefaultObservationGenerator`, `KataGoObservationGenerator`, `StepResult`, `ResetResult`, `StepMetadata` (catalog B "FFI Exported Surface"). Seven Python files import `shogi_gym`: `katago_loop.py`, `tournament.py`, `tournament_runner.py`, `historical_gauntlet.py`, `demonstrator.py`, `evaluate.py`, `showcase/runner.py` — verified by grep in the validation report. All consumers use `observation_mode="katago"` + `action_mode="spatial"`, giving a 50-channel observation tensor and an 11,259-dim action space.

The interesting contract is on the Rust side. `vec_env.rs` runs the per-env update inside a rayon `par_iter` above `PARALLEL_THRESHOLD = 64` envs, releasing the GIL via `py.allow_threads`. The disjoint-index write invariant — each rayon worker writes a unique slice of the pre-allocated flat buffers — is enforced by **convention plus `debug_assert`**, with `unsafe impl Send/Sync for SendPtr<T>` at `vec_env.rs:66-67`. Two large `unsafe` blocks (happy path `:348-458`, panic recovery `:480-539`) write through these pointers. Per-env `catch_unwind` (`vec_env.rs:462`) contains panics with a sentinel reset; closed P2 `keisei-cdf80418a1` confirms this was added 2026-04-03 (catalog B Concerns). The four mode-tag dispatch sites must stay in sync when adding a new observation/action mode — documented at `vec_env.rs:128-137`, but not enforced beyond exhaustive `match`.

### 2.2 SQLite as the message bus

`keisei.db` (catalog E2) creates 21 tables in `init_db()` (`db.py:186-542`, verified by `grep -c "CREATE TABLE IF NOT EXISTS"`), and chains v1→v8 migrations via the `_MIGRATIONS` registry at `db.py:175`. Notable schema discipline: single-row tables enforced by `CHECK (id = 1)` on `training_state`, `tournament_stats`, `showcase_heartbeat`; partial UNIQUE INDEX `idx_showcase_queue_one_running` (`db.py:427`) gives at-most-one running showcase match at the schema level; canonical H2H ordering via `CHECK (entry_a_id < entry_b_id)` on `head_to_head`; v4→v5 migration filters self-play rows so the CHECK doesn't abort the entire `INSERT...SELECT` (`db.py:99-103`).

Concurrency mode: WAL with `busy_timeout=5000`, `wal_autocheckpoint=1000`, `foreign_keys=ON`, `check_same_thread=False` (`db.py:16-22`). Each helper opens, transacts, commits, and closes its own connection — there is no pool. Multi-statement writes use explicit `BEGIN` / `BEGIN IMMEDIATE` (`db.py:592, 652, 1016`); `write_epoch_summary` ends with `PRAGMA wal_checkpoint(TRUNCATE)` (`db.py:633`) to bound WAL growth.

The role split is clean: training writes; the FastAPI server reads; the showcase sidecar both reads and writes (but only its own four `showcase_*` tables); the WebUI never touches SQLite (catalog cross-cutting note line 397). The one out-of-bucket Python import from training is `showcase/inference.py` consuming `model_registry.{build_model, get_model_contract, get_obs_channels}` for inference-time architecture lookup — not for training control flow.

### 2.3 WebSocket protocol

`keisei/server/app.py` exposes one WebSocket at `/ws`. Twelve server→client message types (validation: `grep -E '"type"' server/app.py` returns exactly twelve): `init`, `metrics_update`, `game_update`, `training_status`, `league_update`, `showcase_update`, `showcase_status`, `showcase_error`, `showcase_match_queued`, `showcase_speed_changed`, `showcase_match_cancelled`, `ping` (catalog F WS subsection; the discovery doc undercounted by three command-acks). Four client→server commands: `request_showcase_match`, `change_showcase_speed`, `cancel_showcase_match`, `pong`.

Operational hardening: per-connection `asyncio.Lock` wraps every `_send_json` (`app.py:84-102`, motivated by an explicit `websockets/legacy/protocol.py:308` AssertionError comment); 5 s send timeout; 15 s ping interval; `HostFilterMiddleware` for HTTP plus a manual host re-check inside `ws_endpoint` (`app.py:226`) because `BaseHTTPMiddleware` doesn't filter WS scope; `except*` PEP 654 exception-group flattening (`app.py:74, 250-263`) so per-task exceptions log with full traceback.

Polling cadences: 200 ms for metrics/games/training_state; 5 s for league; 500 ms for showcase. All DB calls are wrapped in `await asyncio.to_thread`. Fingerprint-based change detection (`_style_fingerprint` at `app.py:47`, showcase tuple at `app.py:583`) suppresses redundant pushes. Diagram 6 (showcase-match sequence) and Diagram 7 (DB-as-message-bus dataflow) in `03-diagrams.md` make the producer/consumer roles explicit.

## 3. Where the Complexity Lives

### 3.1 The league/tournament system (C3)

C3 is the densest module cluster in the codebase: ~14 in-bucket modules including `opponent_store.py` (1,324 LoC), `tournament.py` (658), `tournament_runner.py` (421), `concurrent_matches.py` (625), `match_scheduler.py` (463), `tier_managers.py` (511), `historical_gauntlet.py`, `historical_library.py`, `frontier_promoter.py`, `role_elo.py`, `priority_scorer.py`, `match_utils.py`, `tournament_dispatcher.py`, `tournament_queue.py`, plus `demonstrator.py` for exhibition matches (catalog C3 Key Components).

`OpponentStore` is the universal anchor — referenced or imported by 16 in-bucket files (validation report). It composes three tiers (Frontier Static / Recent Fixed / Dynamic) plus a Historical Library, owns the on-disk league directory under `<checkpoint_dir>/league`, manages an in-memory model LRU plus pin-set, and mediates all role-Elo updates. Manager methods rely on `threading.RLock` reentrance so callbacks like `list_by_role` work inside an open transaction (`tier_managers.py:102-104`).

The league offers two execution modes selected by `config.league.tournament_mode` (`katago_loop.py:672`): in-process thread (`LeagueTournament` in `tournament.py`) or sidecar worker (`tournament_runner.py` + `tournament_dispatcher.py` + `tournament_queue.py`). Both modes are wired into the loop today; both replicate the Elo bookkeeping path. Diagram 3 (C4 L3 training components) in `03-diagrams.md` renders the dual paths side-by-side. This is the single biggest structural debt in the codebase (see §5).

### 3.2 The training loop integration

`katago_loop.py` (1,989 LoC, one class) is the single integration point binding C1, C2, and C3. It directly imports **14 in-bucket modules** (validation report; catalog cross-cutting line 394 was corrected from 11 to 14): `algorithm_registry`, `checkpoint`, `concurrent_matches`, `distributed`, `historical_gauntlet`, `katago_ppo`, `match_scheduler`, `model_registry`, `opponent_store`, `priority_scorer`, `role_elo`, `tiered_pool`, `tournament`, `tournament_dispatcher`. Transitive coverage via `TieredPool` and `LeagueTournament` is broader. Inside the file: the loop, opponent rotation, league bookkeeping, snapshotting, DB writes, and the `keisei-train` CLI all share one class (catalog C1b Concerns). One direct SQL UPDATE on `training_state.total_epochs` at `katago_loop.py:843-848` bypasses the typed `db.py` helper layer.

### 3.3 The single-file scale of `db.py` and `config.py`

`db.py` is 1,158 LoC: 21 `CREATE TABLE` statements, eight migration functions, and read/write helpers for ≥12 entity families (catalog E2). `config.py` is 759 LoC composing 17 frozen dataclasses with extensive `__post_init__` validation, legacy-key rejection, deprecation warnings, and path-relative-to-config resolution (catalog E1). Both files are still readable end-to-end but are at the top of the maintenance-threshold bracket — natural decomposition seams already exist (per-table modules in `db.py`; per-bucket dataclass groups in `config.py`).

## 4. Architectural Strengths

- **Centralised typed configuration with cross-registry validation.** All 17 dataclasses are `frozen=True`; `algorithm` and `architecture` are validated against `VALID_ALGORITHMS`/`VALID_ARCHITECTURES` imported from `keisei.training.{algorithm_registry,model_registry}` (`config.py:14-18, 582, 639`). Single source of truth, no scattered globals.

- **Pure-Rust core with zero external runtime dependencies.** `shogi-core/Cargo.toml` declares only `criterion` as `[dev-dependencies]`; PyO3 lives only in `shogi-gym`. This isolates UB risk and keeps `cargo test -p shogi-core` fast (catalog A).

- **DB-as-message-bus discipline holds in practice, not just on paper.** Out-of-bucket imports from training to server/showcase are limited to the FFI surface and `model_registry` for inference-time architecture lookup. WebUI never opens SQLite; server never opens a socket to training. The boundary is enforced by import graph, not by access control (catalog cross-cutting note 397).

- **Cross-bucket boundaries are minimal and explicit.** FFI = 9 PyClasses; DB = 21 tables; WS = 12 + 4 messages. Every consumer site references the boundary's owner rather than redocumenting the surface (validation Boundary Ownership Audit). No ownership leaks observed across 20 entries.

- **WebUI single-chart-library + WS-only data flow.** `uplot ^1.6.31` is the only chart dep (`webui/package.json:18`); two call sites (`MetricsChart`, `WinProbGraph`); no fetch / no localStorage-as-state-of-truth — only as user prefs (catalog H5, H7).

- **Operational hardening is concrete, not abstract.** Per-connection WS lock with cited library line (`app.py:84-102`); manual host re-check for WS scope (`app.py:226`); `BaseExceptionGroup` flattening for clean tracebacks (`app.py:74`); rayon parallel `VecEnv` step with per-env `catch_unwind` panic isolation (`vec_env.rs:462`).

## 5. Architectural Risks

Each item: file:line / filigree ID + impact level.

- **Non-atomic match-result recording.** `LeagueTournament._record_match_result` (`tournament.py:352-460`) issues `store.record_result` plus two `store.update_elo` calls plus `role_elo_tracker.update_from_result` as **separate transactions on `OpponentStore`**, not one. Filigree `keisei-fa604bad63` (P1). Partial Elo state on crash. **Critical.**

- **Two parallel match-recording paths.** `tournament._record_match_result` and `tournament_runner._record_result` re-implement the same Elo + DB write sequence; sidecar lost-update race tracked as `keisei-ea85c3d5b5` (P2); the dispatcher's `PriorityScorer` never sees sidecar completions, `keisei-6cb0990f53`. Drift risk plus correctness risk. **High.**

- **Slot-reuse and partial-load bugs in `concurrent_matches.py`.** `keisei-53eb4eb1f8` (P1) slot reuse without partition reset leaks games; `keisei-4b6c36cd2b` (P1) partial-load `.cpu()` cleanup poisons the shared LRU at `opponent_store.py:344`; `keisei-bc58948f9f`/`f2189813df`/`08ccd20240` (P2 cluster). **High.**

- **`vec_env.rs` Send/Sync soundness rests on doc-only invariants.** `unsafe impl Send/Sync for SendPtr<T>` (`vec_env.rs:66-67`) and two large `unsafe` blocks (`:348-458`, `:480-539`) depend on disjoint-index writes; only `debug_assert` enforces it. Closed P2 `keisei-cdf80418a1` for catch_unwind (now in place); open P4 stylistic bucket `keisei-1883589523`. **Medium** (production-stable, but adding a new mode requires touching 4 sites correctly).

- **`historical_gauntlet.run_gauntlet` tuple-unpacks a dataclass.** `MatchOutcome` is a dataclass (`match_utils.py:21`), not a tuple — every gauntlet match treated as failure. `keisei-4509042dd1` (P1). **High** (silent regression in milestone evaluation).

- **`FrontierManager.review` retires Static incumbent then aborts.** `keisei-959d0eebe7` (P1) silent shrink when Dynamic candidate became inactive. **High.**

- **Server-side init undercounts and silent fingerprint misses.** `total_episodes` truncated to first 500 metrics rows then advances cursor (`app.py:354`, `keisei-f11f3179f7`); style profile fingerprint only captures `(checkpoint_id, status, primary_style)` so raw_metrics/percentile recomputes slip past (`keisei-8108b42644`); showcase status fingerprint misses speed-change diffs (`keisei-d26b243465`); `_handle_match_request` skips entry-id existence check (`keisei-bcc71be6ef`); `gpus` key missing from system_stats on non-zero `nvidia-smi` exit (`keisei-974d6aba11`). Roughly 7 open server triage items in catalog F. **Medium** in aggregate.

- **Showcase sidecar liveness is on the same thread as the game loop.** Heartbeat refreshes only between games (`runner.py:286-292`); long games silently mark sidecar offline. `keisei-6045789532`. **Medium.**

- **Showcase `ModelCache` key omits checkpoint mtime** (`inference.py:127`, `keisei-8c55b48bcc`) — updated checkpoint at the same path serves stale weights. **Medium.**

- **`katago_ppo.py` accepts NaN/negative hyperparameters and SLConfig doesn't validate lambdas.** `keisei-cb7008ac73`, `keisei-bef32b64a8`, `keisei-678359b7aa` — gradient ascent can occur silently with negative `lambda_*`. **Medium.**

- **`KataGoPPOAlgorithm.flush_timings()` is never called.** `keisei-ca5e280cae` (P2, in-progress at session start) — `select_actions` CUDA events accumulate unboundedly on the GPU. Catalog C1a flags this in Concerns alongside the hyperparameter validation issues. Memory growth on long training runs. **Medium.**

- **Svelte 4 → 5 migration ahead.** Every component uses `export let`; reactive `$:` blocks pervasive; `createEventDispatcher` in `MoveLog`; `before/afterUpdate` in `MoveLog`/`MetricsChart`/`PlayerCard`. Tracked as P4 cluster `keisei-9b1171d032`/`a5fe9f710e`/`975949c0b3`/`a1622bc4cf`. **Low** but planned (catalog cross-cutting Svelte 5 section).

- **DB writer concurrency model is convention-based.** WAL plus single-writer-by-convention works today, but `check_same_thread=False` and the absence of a connection pool mean cross-process write ordering depends entirely on caller behaviour (training loop vs showcase sidecar). Not formally documented in `db.py` (catalog E2 Confidence). **Medium** as the league sidecar is wired in.

## 6. Rebuild Context

The repo was reset on 2026-04-01 after migrating the engine from pure Python to Rust (per project memory). The Rust engine's current plan is a standalone burn-in harness; full Keisei integration is queued as a later phase. Several of the densest C3 risks above (sidecar tournament, concurrent matches, dispatcher) are recent additions running alongside the legacy in-process tournament thread — both are wired into `katago_loop.py` today.

What this means for an architect:
- The C3 hot-spot reflects an **incomplete migration** (sidecar partially landed; in-process path still active), not steady-state debt. The right intervention is finishing the migration, not refactoring both paths in parallel.
- The 20-entry catalog is a snapshot of an **evolving system**. The boundaries (FFI, DB, WS) are stable; the C3 internals are not.
- The Rust side is conservative and well-isolated — the appropriate place to focus engineering attention is the league system, not the engine.

## 7. Recommended Next Steps

### For an architect

`axiom-system-architect` (assess + prioritise + catalog-debt skills) would be high-leverage on three concrete targets:
- **The league system (C3).** Dual match-recording paths, transaction boundaries in `_record_match_result`, slot-reuse fault paths, dispatcher idempotency. The densest concentration of P1/P2 bugs in the codebase.
- **`db.py` decomposition.** 1,158 LoC with natural per-entity seams already visible; v8 migrations stable enough to split.
- **`katago_loop.py` decomposition.** 1,989 LoC in one class; the loop, league bookkeeping, snapshotting, DB writes, and CLI are all colocated.

### For security

`ordis-security-architect` threat-model would be useful given:
- The FastAPI WS surface with allow-listed Hosts (`ALLOWED_HOSTS = {"keisei.foundryside.dev", "192.168.1.240", "127.0.0.1", "localhost"}`, `app.py:71`) is the only network entry point. Worth modelling against host-header-manipulation, WebSocket origin validation, and resource exhaustion (the unbounded `read_game_snapshots` in `_poll_and_push` `app.py:284`, `keisei-5f12aa7362`).
- Operator-controlled SQLite writes from training. The DB is the only message bus; a malformed `config_json` blob, a corrupt JSON-typed column, or a runaway training row count can degrade the dashboard.
- `SpectatorEnv` runs untrusted-ish models in the showcase sidecar (`runner.py:113`); CPU-only enforcement via `enforce_cpu_only` is asserted but the threat model isn't documented.

### For quality

`ordis-quality-engineering` test-gap analysis (`analyze-test-gaps`) would be valuable. The league/concurrent-matches cluster has dense filed bugs (~13 open issues in catalog C3 Concerns) and likely under-tested paths — slot reuse, partition reset, two-path Elo bookkeeping, dispatcher round completion idempotency, sidecar heartbeat under long matches. A coverage map against C3 modules would identify which paths are exercised by `tests/` versus which are filed-bug-only.

## 8. Limitations of This Analysis

- **LoC measured for Rust + Python; not for Svelte.** WebUI sizing is by file count and per-component reads, not aggregate LoC (catalog H1 et seq.).
- **Six large Python files were skimmed, not exhaustively read.** Per the Confidence sections of the catalog: `attack.rs` (read 80/1091), `rules.rs` (80/1947), `game.rs` (100/2247), `KataGoPPOAlgorithm` body (`katago_ppo.py:391-991`), `LeagueTournament` (read 1-460/658), `OpponentStore` (1-360/1324), `ConcurrentMatchPool` (1-120/625), `tournament_runner._record_result` (not read), `concurrent_matches.py` slot-reuse fault paths (relied on filigree).
- **WebUI files skimmed only superficially:** `LeagueTable` (40/474), `MatchupMatrix` (40/398), `RecentMatches` (40/290), `MatchScorecard` (40/322), `MatchControls` (60/242), `EntryDetail` (40/≥), `MetricsChart` (40/201), `ShowcaseView` (200/566), `CommentaryPanel` (60/≥130), `AboutView` (60/≥), `ShogiLegend` (30/228) — per catalog H7 "Files skimmed only superficially".
- **Static analysis only.** No profiling, no runtime trace, no fuzzing, no end-to-end migration test execution.
- **Filigree priority labels weren't returning P1/P2 in the standard query format** at the time of analysis (catalog cross-reference line 1006 — `--label=bug --label=P1 --json` returned `[]`); bugs were located via title and file matches instead, so the priority signal is weaker than ideal. Several IDs cited above carry the priority assigned by the catalog agent based on the issue body, not via label query.
- **Subsystem H7 ("cross-cutting infrastructure") aggregates several small concerns.** The synthesiser may want to revisit if shrinking the WebUI surface area (e.g. extracting `createPersistedStore`, repeated 9× per H7 Concerns) is on the roadmap.

## Appendix: File Map

- `00-coordination.md` — strategy, parallel orchestration plan, execution log.
- `01-discovery-findings.md` — top-level structure, 8-subsystem identification, integration boundaries.
- `02-subsystem-catalog.md` — 20 catalogued subsystems (merged from four parallel passes; numeric fixes applied per validation).
- `03-diagrams.md` — Diagrams 1–7: C4 L1 system context, C4 L2 container, C4 L3 training components, C4 L3 webui components, training-rollout sequence, showcase-match sequence, DB-as-message-bus dataflow.
- `04-final-report.md` — this document.
- `temp/catalog-rust.md`, `temp/catalog-training.md`, `temp/catalog-server-data.md`, `temp/catalog-webui.md` — per-bucket parallel cataloguing outputs prior to merge.
- `temp/validation-catalog.md` — independent sample-grep validation report (verdict PASS-WITH-FIXES).
