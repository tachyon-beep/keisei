# Server / Data Catalog

Subsystems E1 (config), E2 (SQLite persistence), F (FastAPI server), G (showcase runner + DB ops).

---

## E1. Config Layer

**Location:** `keisei/config.py`

**Responsibility:** Define typed, frozen dataclasses for every Keisei configuration knob and load+validate a single TOML file into a composed `AppConfig` tree.

**Key Components:**
- `keisei/config.py` — 759 LoC, 17 frozen dataclasses + `load_config(Path) -> AppConfig`.

**Class inventory (17 dataclasses):**

Top-level group (3): `TrainingConfig` (config.py:21), `DisplayConfig` (config.py:32 — `db_path`, `moves_per_minute`), `ModelConfig` (config.py:38).

League group (10) composed inside `LeagueConfig` (config.py:414): `FrontierStaticConfig` (45), `RecentFixedConfig` (82), `DynamicConfig` (102), `MatchSchedulerConfig` (188), `HistoricalLibraryConfig` (261), `GauntletConfig` (288), `RoleEloConfig` (305), `PriorityScorerConfig` (325), `ConcurrencyConfig` (364), `StorageConfig` (394).

Auxiliary (3): `DemonstratorConfig` (526), `DistributedConfig` (534), `AppConfig` (548 — root: training/display/model + optional league + optional demonstrator + distributed).

**Composition:** `AppConfig` (config.py:548) holds `training`, `display`, `model` (required), plus `league: LeagueConfig | None`, `demonstrator: DemonstratorConfig | None`, and `distributed: DistributedConfig`. Sub-config defaults are concrete instances of frozen dataclasses (config.py:441–451) — explicit comment notes the mutable-default-argument pitfall doesn't apply because every sub-config is `frozen=True`.

**`load_config(path)` flow** (config.py:558–759):
1. `tomllib.load(open(path, "rb"))`; `config_dir = path.parent.resolve()`.
2. Per section (`training`, `display`, `model`, `league`, `demonstrator`, `distributed`): reject unknown keys via `set(raw) - {f.name for f in fields(...)}` (config.py:566–571 etc.), then construct the dataclass.
3. **Path resolution:** `checkpoint_dir` and `db_path` resolved relative to TOML location, not CWD (config.py:594, 625).
4. **Cross-registry validation:** `algorithm` against `VALID_ALGORITHMS` (config.py:582), `architecture` against `VALID_ARCHITECTURES` (config.py:639) — both imported from training registries (single source of truth, comment at config.py:14).
5. **Legacy section rejection:** raises on old `[league.frontier_static]`, `[league.recent_fixed]`, `[league.sampling]`, `[league.role_elo]`, `[league.matchmaking]` with the new name (config.py:654–666).
6. **Convenience sugar:** `[league.active]` (`frontier_static_slots` etc.) is folded into `frontier`/`recent`/`dynamic` sub-configs (config.py:680–688) using `setdefault` so explicit sub-section keys win.
7. **Deprecated key warnings:** `max_pool_size`, `historical_ratio`, `current_best_ratio` emit `DeprecationWarning` and are dropped (config.py:691–701).
8. **Validation in `__post_init__`:** every dataclass has bounds checks raising `ValueError` (e.g. learner mix ratios sum to 1.0 (config.py:222), envs ≤ total_envs (config.py:373), elo_floor ≤ initial_elo (config.py:473)).
9. **Cross-config warning** (not error): `max_resident_models < max_active_entries` warns about LRU thrash (config.py:513–523).
10. **`league.enabled = false`** collapses to `league_config = None` so consumers' `if config.league is not None` checks short-circuit (config.py:730–733).

**Dependencies:**
- Inbound: every entry-point loads TOML through this (`server/app.py:613, 635`, training entry points per discovery doc).
- Outbound: `tomllib`, `dataclasses`, `keisei.training.{algorithm_registry,model_registry}` (for VALID_* sets).

**Patterns Observed:**
- Frozen dataclasses + `__post_init__` validation: every config class enforces invariants at construction (e.g. config.py:212–258 `MatchSchedulerConfig`).
- "Single source of truth" cross-registry import to avoid drift (config.py:14–18).
- Legacy/deprecated key handling distinguishes hard errors (renamed sections) from warnings (removed scalars).
- Path-relative-to-config resolution prevents CWD-dependent behaviour.

**Concerns (from open filigree triage on these files):**
- `keisei-392fa79ffb` — multiple config dataclasses lack domain validation.
- `keisei-1f74d42285` — `algorithm_params` and `model.params` not validated as mappings; malformed TOML crashes late.
- `keisei-fcb32efacb` — `DemonstratorConfig` (config.py:526) lacks bounds validation.

**Confidence:** High for class inventory and `load_config` flow (read 100% of file). Medium for "every consumer's behaviour" — only the boundary into `load_config` was validated; consumer-side semantics owned by other agents.

---

## E2. SQLite Persistence Layer

**Location:** `keisei/db.py`

**Responsibility:** Single-file SQLite DDL, idempotent migrations, and read/write helpers for every persisted entity (metrics, snapshots, training state, league, head-to-head, game features, style profiles, tournament queue + workers, showcase pre-creates).

**Key Components:**
- `keisei/db.py` — 1158 LoC. Module constants: `SCHEMA_VERSION = 8` (db.py:13), `_MIGRATIONS` registry (db.py:175–183).

### Tables created in `init_db()` (db.py:186–542)

| Table | Purpose |
|---|---|
| `schema_version` (db.py:194) | Single-row int holding current schema version. |
| `metrics` (db.py:197) | Per-step training metrics (policy/value loss, win rates, episode length, gradient norm, episodes_completed, timestamp). Indexed on `epoch` and `id`. |
| `game_snapshots` (db.py:217) | Live learner game state — board/hands JSON, ply, sfen, in_check, move_history, value_estimate, game_type, demo_slot, opponent_id. PK = `game_id`. |
| `training_state` (db.py:234) | Single-row training process state — config_json, display_name, model_arch, algorithm, started_at, current_epoch/step, checkpoint_path, status, phase, heartbeat_at, learner_entry_id. CHECK id=1. |
| `league_entries` (db.py:250) | League members — display_name, flavour_facts (JSON), architecture, model_params (JSON), checkpoint_path, elo_rating, created_epoch/at, role, status, parent_entry_id, lineage_group, protection_remaining, last_match_at, four role-specific elo columns, optimizer_path, update_count, last_train_at, retired_at, training_enabled, three games_vs_* counters, dynamic_update_worker. Indexed on elo_rating. |
| `league_results` (db.py:281) | Per-match record — epoch, both entries, match_type, both roles, num_games, wins/draws, before/after Elo, training_updates, recorded_at. Indexed on epoch + each entry side. |
| `elo_history` (db.py:303) | Per-epoch Elo points for charting; indexed on entry_id and (entry_id, epoch). |
| `league_transitions` (db.py:312) | Role/status change audit log with reason; indexed on entry_id. |
| `head_to_head` (db.py:327) | Aggregate H2H matrix maintained incrementally by `record_result()`; canonical PK `(entry_a_id, entry_b_id)` with CHECK `entry_a_id < entry_b_id`. Indexed on each side. |
| `league_meta` (db.py:341) | Single-row bootstrap flag; default-seeded. |
| `historical_library` (db.py:346) | Slot table for log-spaced historical opponents — slot_index PK, target_epoch, entry_id, actual_epoch, selected_at, selection_mode. |
| `gauntlet_results` (db.py:354) | Periodic gauntlet outcomes vs historical slots — wins/losses/draws + Elo before/after. Indexed on epoch. |
| `tournament_stats` (db.py:368) | Single-row latest tournament round counters — duration, pairings_requested/completed, total_games/plies, active_slots, model_load_time/count, games_per_min. CHECK id=1. |
| `game_features` (db.py:381) | Per-game feature vector for style profiling — opening (first_action, opening_seq_3/6, rook_moved_ply, king_displacement_20), tempo (first_capture/check/drop_ply, num_checks/captures), drops/promotions, positional (rook_moves_in_20, king_moves_in_30, num_repetitions), termination_reason. Indexed on checkpoint_id, opponent_id, epoch. |
| `showcase_queue` (db.py:416) | Showcase match request queue — entry_id_1/2 (TEXT), speed, status, requested_at, started/completed_at. Partial UNIQUE INDEX enforcing at-most-one running entry (db.py:427). |
| `showcase_games` (db.py:430) | Per-showcase-game record — queue_id FK, both sides, Elo + names snapshot, status, abandon_reason, started/completed_at, total_ply. Indexed on status. |
| `showcase_moves` (db.py:447) | Per-ply showcase data — game_id FK, ply, action_index, usi_notation (legacy: actually Hodges), board/hands JSON, current_player, in_check, value_estimate, top_candidates JSON, move_heatmap_json, move_usi (true USI), move_time_ms, created_at. UNIQUE(game_id, ply); indexed on (game_id, ply). |
| `showcase_heartbeat` (db.py:467) | Single-row sidecar liveness — last_heartbeat, runner_pid. CHECK id=1. |
| `style_profiles` (db.py:472) | One row per checkpoint — recomputed_at, profile_status, games_sampled, raw_metrics_json, percentile_json, primary_style, secondary_traits JSON, commentary_json, updated_at. PK on checkpoint_id. |
| `tournament_pairing_queue` (db.py:485) | Sidecar tournament pairings — round_id, both entries, games_target, status, worker_id, claimed/completed_at, enqueued_epoch, priority. Three indexes: pending (status, priority DESC, id), round (round_id), staleness (status, enqueued_epoch). |
| `tournament_worker_heartbeat` (db.py:505) | Per-worker sidecar liveness — pid, device, last_seen, pairings_done. PK = worker_id. |

### Migration chain (`_MIGRATIONS` db.py:175)

| Version | Function | Effect |
|---|---|---|
| → v2 | `_migrate_v1_to_v2` (db.py:38) | `ALTER TABLE` add 19 columns to `league_entries` (role/status/lineage/protection/last_match_at/four role Elos/optimizer/update_count/retired_at/training_enabled/three games_vs_*) and `learner_entry_id` to `training_state`. Constant defaults (SQLite ALTER lacks expression defaults). |
| → v3 | `_migrate_v2_to_v3` (db.py:74) | No-op — new showcase tables created via `CREATE TABLE IF NOT EXISTS` in `init_db`. |
| → v4 | `_migrate_v3_to_v4` (db.py:79) | `ALTER TABLE league_entries ADD dynamic_update_worker TEXT`; tournament sidecar tables also handled by `init_db`. |
| → v5 | `_migrate_v4_to_v5` (db.py:89) | Backfill `head_to_head` from `league_results` inside `BEGIN IMMEDIATE`/`COMMIT` (rolls back on failure). Filters self-play rows (`entry_a_id != entry_b_id`) so the CHECK constraint doesn't abort the INSERT...SELECT. |
| → v6 | `_migrate_v5_to_v6` (db.py:132) | `CREATE INDEX idx_pairing_queue_staleness ON tournament_pairing_queue(status, enqueued_epoch)` — unblocks staleness UPDATE which was scanning every pending row. |
| → v7 | `_migrate_v6_to_v7` (db.py:147) | `ALTER TABLE showcase_moves ADD move_heatmap_json TEXT` — nullable; older rows render no heatmap. |
| → v8 | `_migrate_v7_to_v8` (db.py:158) | `ALTER TABLE showcase_moves ADD move_usi TEXT` — true USI alongside legacy Hodges-as-`usi_notation`; nullable. |

**Migration runner** (db.py:513–540): If `schema_version` row missing, treats as v0 and runs all migrations (idempotent). If db_version > SCHEMA_VERSION, raises `RuntimeError`. Else loops `target in range(db_version+1, SCHEMA_VERSION+1)` running registered functions. Inserts/updates the version row at end.

**Read/write API families:**
- **Connection management** — `_connect()` (db.py:16): `journal_mode=WAL`, `busy_timeout=5000`, `wal_autocheckpoint=1000`, `Row` factory, `foreign_keys=ON`. `check_same_thread=False`. Module convenience `wal_checkpoint(db_path)` (db.py:1143) forces TRUNCATE checkpoint.
- **Metrics** — `write_metrics` (db.py:545), `write_epoch_summary` (db.py:576, batches metrics+training_state+`PRAGMA wal_checkpoint(TRUNCATE)` in one txn), `read_metrics_since(since_id, limit)` (db.py:638).
- **Game snapshots** — `write_game_snapshots` (db.py:649, INSERT OR REPLACE in BEGIN txn), `read_game_snapshots` (db.py:685), `read_game_snapshots_since(since_ts, since_game_id)` (db.py:694) using composite cursor `(updated_at, game_id)` to avoid skipping rows with equal timestamps.
- **Training state** — `write_training_state` (db.py:726, INSERT OR REPLACE), `read_training_state` (db.py:755), `update_heartbeat` (db.py:764), `update_training_progress(epoch, step, checkpoint_path?, phase?, learner_entry_id?)` (db.py:776).
- **League** — `read_league_data(max_results=500)` (db.py:803) returns dict with `entries` (parses flavour_facts/model_params JSON), `results`, `historical_library` (joined to entry_name/elo), `gauntlet_results` (last 50 epochs), `transitions` (last 200). `read_elo_history(max_epochs=0)` (db.py:876) — 0 = full, else recent N.
- **Tournament stats** — `write_tournament_stats(stats)` (db.py:901, computes games_per_min and UPSERTs single row). `read_tournament_stats` (db.py:936).
- **Head-to-head** — `read_head_to_head` (db.py:948) tolerates missing table with warning. `backfill_head_to_head` (db.py:972) used for repair, returns inserted-pair count, runs in `BEGIN IMMEDIATE`. (Note: incremental write path lives in `keisei.training.opponent_store` per imports — outside scope.)
- **Game features** — `write_game_features(features)` (db.py:1010, batch INSERT), `read_game_features_for_checkpoint(id)` (db.py:1048), `read_all_game_features(min_epoch=None)` (db.py:1063).
- **Style profiles** — `write_style_profile` (db.py:1085, UPSERT, dumps raw_metrics/percentiles/secondary_traits/commentary as JSON), `read_style_profiles` (db.py:1123, parses JSON columns back).

**Concurrency discipline:**
- WAL mode + `busy_timeout=5000` (db.py:18–20) — readers don't block writers.
- `foreign_keys = ON` (db.py:22) on every connection.
- Each helper opens, transacts, commits, closes — no shared connection pool.
- Multi-statement writes wrapped explicitly in `BEGIN`/`BEGIN IMMEDIATE` (db.py:592, 652, 1016); the head-to-head migration and backfill use `BEGIN IMMEDIATE` and explicit `ROLLBACK` on exception.
- `check_same_thread=False` (db.py:17) — connections are short-lived per call, but the flag lets callers shuttle a connection across threads if needed (showcase runner does pass conns through `await asyncio.to_thread`).
- WAL hygiene: `write_epoch_summary` ends with `PRAGMA wal_checkpoint(TRUNCATE)` (db.py:633) to bound WAL growth.

**Dependencies:**
- Inbound: `keisei.server.app` (read side), `keisei.showcase.db_ops` + `keisei.showcase.runner` (own writer, imports `_connect`), `keisei.training.*` (write side — owned by training agent).
- Outbound: stdlib `sqlite3`, `json`, `logging`. No third-party.

**Patterns Observed:**
- "DB is the message bus" (per discovery §8). Reader/writer separation enforced by convention, not access control.
- Idempotent DDL via `CREATE TABLE/INDEX IF NOT EXISTS` + version-gated `ALTER TABLE` migrations. Comment at db.py:42–48 calls out the SQLite `ALTER` constant-default limitation.
- Composite cursor `(updated_at, game_id)` (db.py:694) prevents row-loss when timestamps tie.
- Single-row tables use `CHECK (id = 1)` + UPSERT (db.py:235, 369, 468).
- Partial UNIQUE INDEX `idx_showcase_queue_one_running` (db.py:427) enforces at-most-one running showcase match at the schema level.
- v4→v5 migration includes filter `WHERE entry_a_id != entry_b_id` because a single self-play row would abort the entire INSERT and leave the table empty — rationale comment at db.py:99–103.

**Concerns (from open filigree triage):**
- `keisei-44945464ac` — `read_tournament_stats` (db.py:942) silent-swallows all `Exception` → `None`; same anti-pattern as historical `read_head_to_head`.
- `keisei-25569d556d` — `read_elo_history(max_epochs=N)` (db.py:885) off-by-one returns N+1 epochs.
- `keisei-33964dec44` — although v6 added the staleness index (db.py:142), there's still a triage about expiry walking pending rows; verify against current consumer.
- Code-observed: `read_training_state` (db.py:755) returns `None` on missing row but several callers in app.py (e.g. `_training_alive`, `_poll_and_push`) handle that path; behavioural ok. Multiple writes per epoch from `write_metrics`/`write_training_state`/`update_heartbeat` separately — `write_epoch_summary` (db.py:583) batches but isn't the only write site.
- `read_game_snapshots` (db.py:685) is unbounded — full table scan; callers expected to be one-shot init only.

**Confidence:** High — read 100% of db.py, every table and migration confirmed against DDL. Medium for the "WAL discipline holds under concurrent writers" claim — db.py provides the primitives, but cross-process write ordering depends on caller behaviour (e.g. training loop vs showcase sidecar both writing). High for migration chain — every `_migrate_*` function read.

---

## F. FastAPI Server

**Location:** `keisei/server/app.py`, `keisei/server/static/`

**Responsibility:** Serve the SPA bundle and the `/ws` WebSocket that streams DB-derived state to the dashboard; accept showcase-control commands; reject unknown Hosts.

**Key Components:**
- `keisei/server/app.py` — 658 LoC. Factory `create_app(db_path, allowed_hosts=None) -> FastAPI` (app.py:202), `create_app_from_env()` (app.py:613), `main()` CLI (app.py:629).
- `keisei/server/static/` — built UI: `index.html`, `favicon.svg`, `assets/index-*.{css,js}` (Vite hashed bundle).
- `HostFilterMiddleware(BaseHTTPMiddleware)` (app.py:185) — rejects HTTP requests outside allowed Hosts.

**Module constants** (app.py:59–71):
- `MAX_METRICS_IN_INIT = 500`
- `POLL_INTERVAL_S = 0.2` (training/metrics/games loop)
- `LEAGUE_POLL_INTERVAL_S = 5.0`
- `SHOWCASE_POLL_INTERVAL_S = 0.5`
- `POLL_BATCH_SIZE = 100`
- `HEARTBEAT_STALE_S = 30`
- `WS_SEND_TIMEOUT_S = 5.0`
- `WS_PING_INTERVAL_S = 15.0`
- `MAX_SHOWCASE_QUEUE_DEPTH = 5`
- `VALID_SPEEDS = {"slow","normal","fast"}`
- `ALLOWED_HOSTS = {"keisei.foundryside.dev","192.168.1.240","127.0.0.1","localhost"}`; `TEST_ALLOWED_HOSTS` adds `testserver`/`test`.

### HTTP routes

| Method | Path | Handler | Description |
|---|---|---|---|
| GET | `/healthz` | `healthz()` (app.py:217) | Returns `{status, db_accessible, training_alive}`. `db_accessible` opens the DB and runs `SELECT 1 FROM schema_version`; `training_alive` checks `training_state.heartbeat_at` age < 30 s. |
| WS  | `/ws`     | `ws_endpoint()` (app.py:237) | Spectator stream — see WS subsection below. |
| GET | `/audio/*` | `StaticFiles` mount (app.py:268–270) | Serves `audio/` from repo root; range-supported for `<audio>` streaming. Conditional on dir existence. Mounted before `/`. |
| GET | `/*`      | `StaticFiles(directory=static, html=True)` (app.py:274) | SPA bundle — falls through to `index.html` for client-side routes. Conditional on dir existence. |

### WebSocket protocol (`/ws`)

Per-connection state on accept (app.py:242–249):
- `await websocket.accept()`
- `send_lock = asyncio.Lock()` — every send goes through `_send_json(ws, send_lock, msg, timeout=WS_SEND_TIMEOUT_S)` (app.py:84) which acquires the lock then `asyncio.wait_for(ws.send_json(msg), timeout=5.0)`.
- `asyncio.TaskGroup` spawns four background tasks: `_poll_and_push`, `_keepalive`, `_receive_commands`, `_poll_showcase`.
- Host-allowlist re-checked manually for WS scope (app.py:226) since `BaseHTTPMiddleware` only handles HTTP; rejects with close code `1008`.
- `except*` handlers (app.py:250–263) flatten `BaseExceptionGroup` (`_flatten_exception_group`, app.py:74) so per-task exceptions log with full traceback rather than the bare empty-message form.

**Server → client messages (8 types):**

| Type | Sender | Payload (server-side) |
|---|---|---|
| `init` | `_poll_and_push` (app.py:324) | One-shot snapshot on connect: `games`, `metrics` (≤500), `training_state`, `league_entries`, `league_results`, `historical_library`, `gauntlet_results`, `transitions`, `elo_history` (capped at 500 epochs to fit `WS_SEND_TIMEOUT_S`), `tournament_stats`, `style_profiles`, `head_to_head`, `showcase: {game, moves, queue, sidecar_alive}`. |
| `metrics_update` | `_poll_and_push` (app.py:368) | `{rows: [...]}` — incremental via `read_metrics_since(last_metrics_id)`. |
| `game_update` | `_poll_and_push` (app.py:376) | `{snapshots: [...]}` — incremental via `read_game_snapshots_since(last_ts, last_game_id)` composite cursor. |
| `training_status` | `_poll_and_push` (app.py:387) | `{status, phase, heartbeat_at, epoch, step, episodes, config_json, display_name, model_arch, total_epochs, system_stats, learner_entry_id}`. `system_stats` from `_get_system_stats` (app.py:117) — psutil CPU/RAM + nvidia-smi GPU. Sent only when status/epoch/heartbeat changed. |
| `league_update` | `_poll_and_push` (app.py:433) | Same shape as init's league fields. Sent every 5 s if entry_ids/result_id/transition_id changed; `style_profiles` only included when fingerprint changed (`_style_fingerprint`, app.py:47). |
| `showcase_status` | `_poll_showcase` (app.py:587) | `{queue, active_game_id, sidecar_alive}`. Fingerprinted `(game_id, len(queue), alive)` — sent on change only. |
| `showcase_update` | `_poll_showcase` (app.py:604) | `{game, new_moves}` — incremental via `read_showcase_moves_since(game_id, last_sent_ply)`. Cursor resets when `game_id` changes. |
| `showcase_error` | `_handle_match_request` / `_handle_speed_change` / `_handle_cancel` (app.py:498, 526, 542) | `{error: str}` — validation failure responses. |
| `ping` | `_keepalive` (app.py:454) | Empty heartbeat every 15 s. On send timeout/disconnect, raises `WebSocketDisconnect` to collapse the TaskGroup. |
| `showcase_match_queued` / `showcase_speed_changed` / `showcase_match_cancelled` | command handlers (app.py:517, 534, 546) | Acks for client commands; not in the discovery doc's nine-type taxonomy but they are server→client message types. |

(Discovery doc lists 9 types — the catalog above lists 9 plus the three command-ack messages observed in code.)

**Client → server commands** (`_receive_commands` app.py:459):
- `request_showcase_match` — `{entry_id_1, entry_id_2, speed}` → validates speed in `VALID_SPEEDS`, both ids non-empty and distinct, queue depth < `MAX_SHOWCASE_QUEUE_DEPTH=5`; on success calls `showcase_queue_match`.
- `change_showcase_speed` — `{queue_id, speed}` → `showcase_update_speed`.
- `cancel_showcase_match` — `{queue_id}` → `showcase_cancel_match` (only marks `pending` rows cancelled — db_ops.py:74).
- `pong` — keepalive response, no-op.
- Unknown types logged at DEBUG. Non-JSON payloads logged WARN and skipped.

**Polling architecture:** `_poll_and_push` runs at 200 ms cadence for metrics/games/training_state and accumulates a `league_poll_elapsed` counter to fire league reads every 5 s; `_poll_showcase` runs at 500 ms; `_keepalive` at 15 s. All DB calls go through `await asyncio.to_thread(...)` since `sqlite3` is synchronous.

**Lifespan** (app.py:204): `init_db(db_path)` runs on startup via `asyncio.to_thread`, applying any pending migrations before the dashboard reads.

**`main()` CLI** (app.py:629): `argparse` with `--config` (required), `--host` default `127.0.0.1`, `--port` default `8741`, `--socket` (mutually overrides host/port via `uvicorn.run(uds=...)`). `create_app_from_env` reads `KEISEI_CONFIG` env var (default `keisei-league.toml`).

**Dependencies:**
- Inbound: WebUI WebSocket client, deployment scripts (entry point `keisei-serve` per pyproject.toml).
- Outbound: `keisei.db` (read API + `_connect` for `_db_accessible`), `keisei.showcase.db_ops` (showcase read+write), `keisei.config.load_config` (CLI), `fastapi`, `starlette.middleware.base`, `uvicorn`, optional `psutil`, `nvidia-smi` subprocess.

**Patterns Observed:**
- Per-connection asyncio.Lock pattern for WS sends (app.py:84–102) — explicit comment cites the `websockets/legacy/protocol.py:308` `_drain_helper` AssertionError that motivates it.
- Fingerprint-based change detection avoids redundant sends: `_style_fingerprint` (app.py:47), showcase status tuple (app.py:583).
- Composite cursor `(updated_at, game_id)` matches the db.py read API.
- Manual Host re-check for WebSocket scope because `BaseHTTPMiddleware` only filters HTTP (app.py:226–234).
- IPv6 bracketed-host parsing in `_extract_hostname` (app.py:170).
- `except* (WebSocketDisconnect, CancelledError, Exception)` PEP 654 exception-group clauses (app.py:250–263) with manual flattening for nested groups.

**Concerns (from open filigree triage):**
- `keisei-bd56f91623` — concurrent WS producers send without serialisation. Code as committed *does* hold `send_lock`; commit `f608594` landed the lock. Issue may pre-date the fix or refer to a regression scope. Verify status.
- `keisei-31bb8cd791` — open task: regression test for concurrent ws.send_json serialisation.
- `keisei-f11f3179f7` — `total_episodes` undercounted when DB has >500 rows at WS init: only sums first 500 (app.py:354) but the metrics cursor advances past them, so future increments add to the truncated base.
- `keisei-974d6aba11` — `gpus` key missing from system_stats when `nvidia-smi` returncode is non-zero (app.py:138 sets `gpus` only inside the `returncode == 0` branch — no else; the outer `except Exception` does set `gpus = []` but a non-zero exit doesn't raise).
- `keisei-bcc71be6ef` — `_handle_match_request` skips entry validation (app.py:491) — only checks string non-emptiness; doesn't verify `entry_id_1`/`_2` exist in `league_entries`.
- `keisei-d26b243465` — showcase status fingerprint (`(game_id, len(queue), alive)`, app.py:583) misses speed-change diffs; speed updates in queue won't trigger a status push.
- `keisei-8108b42644` — style profile fingerprint misses recompute-without-status-change: `_style_fingerprint` only captures `(checkpoint_id, status, primary_style)` (app.py:53) so changes to raw_metrics/percentiles silently slip past.
- `keisei-5f12aa7362` — open: audit init reads for unbounded queries that have bounded counterparts (e.g. `read_game_snapshots` in `_poll_and_push` app.py:284 is unbounded).
- Code-observed: `_get_system_stats` (app.py:117) does `psutil.cpu_percent(interval=0.1)` synchronously inside `await asyncio.to_thread`, but the 100 ms blocking sleep is inside the worker thread — fine for the event loop but it does pace the training_status push.

**Confidence:** High for routes, message types, constants, lifespan (read 100% of file). High for the lock/serialisation rationale (commit reference + cited library line). Medium for the consumer-side semantics of every message — webui agent owns that side.

---

## G. Showcase Runner + DB Ops

**Location:** `keisei/showcase/`

**Responsibility:** Headless sidecar that drains the `showcase_queue`, plays model-vs-model games on CPU at watchable speed, and writes per-move state into `showcase_*` tables for the WS layer to stream.

**Key Components:**
- `keisei/showcase/runner.py` — 344 LoC. `ShowcaseRunner` class + `main()` CLI; signal handling (SIGTERM/SIGINT → `runner.stop()`).
- `keisei/showcase/db_ops.py` — 246 LoC. Showcase-specific writers + retry helper `_retry_write` (db_ops.py:21) that wraps SQLITE_BUSY with exponential backoff (`MAX_RETRIES=3`, `RETRY_BASE_DELAY=0.1` s).
- `keisei/showcase/inference.py` — 144 LoC. CPU-only loader + LRU `ModelCache(max_size=2)` (inference.py:104, thread-safe via `threading.Lock`); `enforce_cpu_only` sets `CUDA_VISIBLE_DEVICES=""` and torch threads; `run_inference` zero-pads observations from 46→50 channels for KataGo models.
- `keisei/showcase/heatmap.py` — 49 LoC. `build_heatmap(chosen_usi, legal_with_usi, probs)` filters legal moves sharing the chosen move's first 2 chars (from-square or drop prefix) and pairs with policy probability.
- `keisei/showcase/__main__.py` — 4 LoC. `from keisei.showcase.runner import main; main()` (no `if __name__ == "__main__"` guard).

**Runner constants** (runner.py:48–53): `MAX_PLY=512`, `SPEED_DELAYS={"slow":4.0,"normal":2.0,"fast":0.5}` (seconds per ply), `HEARTBEAT_INTERVAL=10.0`, `POLL_INTERVAL=5.0` (idle-loop wait), `SAMPLING_TEMPERATURE=0.5`, `SPEED_POLL_INTERVAL=5` (re-read DB speed every N plies).

**Lifecycle (`ShowcaseRunner.run`, runner.py:281):**
1. `enforce_cpu_only(cpu_threads)` — sets `CUDA_VISIBLE_DEVICES=""`, `torch.set_num_threads`, `set_num_interop_threads(1)`.
2. `_startup_cleanup` → `cleanup_orphaned_games(db_path, stale_after_s=60)` (db_ops.py:224) — if the heartbeat is stale, marks all `in_progress` showcase_games as `abandoned`/`crash_recovery` and cancels any `running` queue entries.
3. Initial `write_heartbeat`.
4. Main loop until `_stop_event.is_set()`:
   - Periodic `write_heartbeat` every 10 s.
   - `claim_next_match` (db_ops.py:49) — atomic `UPDATE ... RETURNING` of one pending row, sets status='running' + started_at. If a row returned, runs `_run_game(match)`.
   - Else `_maybe_auto_showcase` (runner.py:262) — if interval expired (`auto_showcase_interval` default 1800 s) and queue empty, picks top-2 active league entries by elo_rating and queues them.
   - Idle wait `_stop_event.wait(timeout=POLL_INTERVAL)`.
5. `stop()` sets `_stop_event` and `_speed_event` (latter is the per-ply pacing wake-up).

**Per-game flow (`_run_game`, runner.py:113):**
1. `_load_models(match)` (runner.py:90) reads both `league_entries` rows and uses `ModelCache.get_or_load(entry_id, checkpoint_path, arch, params)` for each side. Cache key omits checkpoint mtime.
2. `create_showcase_game` writes a row, returns `game_id`.
3. `_create_env()` constructs `SpectatorEnv(max_ply=MAX_PLY, action_mode="spatial")` (imports `from shogi_gym import SpectatorEnv` — Rust FFI consumer; FFI surface owned by Rust agent).
4. Loop until `_stop_event` / `env.is_over` / `ply >= MAX_PLY`:
   - `env.get_observation()` → `run_inference(model, obs, arch)` returning `(policy_logits, win_prob)`.
   - Mask illegal moves to −1e9, temperature-scale, softmax over legal-only, NaN guard (uniform fallback if total < 1e-10).
   - Sample `np.random.choice` over legal probs only.
   - `env.step(action)` returns new state.
   - Build heatmap from `legal_with_usi` captured *before* `env.step` (so position matches the policy distribution).
   - `write_showcase_move(game_id, ply, action_index, usi_notation, board/hands JSON, current_player, in_check, value_estimate=win_prob, top_candidates JSON, move_time_ms, move_heatmap_json, move_usi)` — atomic INSERT OR IGNORE move + UPDATE `showcase_games.total_ply` in `BEGIN IMMEDIATE` with retry on lock (db_ops.py:124).
   - Every 5 plies, re-read `showcase_queue.speed` for live speed control; pace via `self._speed_event.wait(timeout=delay)`.
5. Resolution: shutdown → `mark_game_abandoned("shutdown")`; max_ply → `mark_game_completed("draw")`; checkmate → `mark_game_completed("black_win"|"white_win")`; repetition/perpetual_check/impasse/max_moves → `"draw"`.
6. `finally:` always calls `complete_queue_entry(match["id"])` (db_ops.py:104) which only acts on `status='running'` rows.

**db_ops API surface (15 functions):** `_now_iso`, `_retry_write`, `queue_match`, `claim_next_match`, `read_queue` (pending+running only), `cancel_match` (pending-only guard), `update_queue_speed`, `create_showcase_game`, `complete_queue_entry`, `read_active_showcase_game`, `write_showcase_move` (atomic with retry), `read_showcase_moves_since(game_id, since_ply)`, `read_all_showcase_moves`, `mark_game_completed`, `mark_game_abandoned`, `write_heartbeat`, `read_heartbeat`, `cleanup_orphaned_games`.

**Interaction with db.py vs own tables:**
- Imports `_connect` from `keisei.db` — same SQLite connection pragmas, same WAL mode.
- *Owns* writes to `showcase_queue`, `showcase_games`, `showcase_moves`, `showcase_heartbeat` — the runner is the single writer to these tables (modulo the server's queue/cancel/speed control commands, which go through the same `db_ops` retry path).
- *Reads* `league_entries` directly via `_connect` (runner.py:91, 270) — not through a `keisei.db` helper, just raw SELECT.
- *Does not* write to any non-showcase table.

**Dependencies:**
- Inbound: `keisei.server.app` imports the `db_ops` read/write API; `keisei.showcase.__main__` invokes runner.main.
- Outbound: `keisei.db._connect`; `keisei.training.model_registry.{build_model, get_model_contract, get_obs_channels}` (in inference.py); `shogi_gym.SpectatorEnv` (FFI consumer per discovery §7.1 — surface owned by Rust agent); `torch`, `numpy`, stdlib `signal`/`threading`/`os`.

**Patterns Observed:**
- Sidecar process pattern: separate from training, owns its own write path, signalled via SIGTERM/SIGINT, heartbeat-tracked.
- Crash-recovery on startup gated by heartbeat age (db_ops.py:229) — fast restart skips recovery.
- LRU model cache with double-check after slow I/O (inference.py:128–143) avoids holding the lock during `torch.load`.
- Atomic write contract: `INSERT OR IGNORE move` + `UPDATE total_ply` in a single `BEGIN IMMEDIATE` (db_ops.py:136) — keeps `total_ply` consistent with the highest committed move.
- Retry-with-jitter on SQLITE_BUSY (`_retry_write`, db_ops.py:21) — exponential 0.1, 0.2 s plus 0–50 ms jitter.
- Pre-step USI capture (runner.py:150) — `legal_moves_with_usi()` must match the position the policy was computed against.

**Concerns (from open filigree triage):**
- `keisei-2f4b87732b` — `cleanup_orphaned_games` skips recovery when heartbeat is fresh (db_ops.py:234), allowing fast-restart to leave games stuck `in_progress`.
- `keisei-f1e5004e86` — `claim_next_match` (db_ops.py:49) can raise `IntegrityError` when partial UNIQUE INDEX `idx_showcase_queue_one_running` (db.py:427) collides with an existing 'running' row.
- `keisei-6895e438ca` — `total_ply` regresses on out-of-order writes: `write_showcase_move` (db_ops.py:154) unconditionally `UPDATE total_ply = ?` instead of `MAX(total_ply, ?)`.
- `keisei-8c55b48bcc` — `ModelCache` key (inference.py:127) is `(entry_id, checkpoint_path)` — no mtime, so an updated checkpoint at the same path serves stale weights.
- `keisei-d9882a0664` — `run_inference` zero-pads 46→50 channels (inference.py:78) — fabricates KataGo planes with zeros instead of computing them.
- `keisei-6045789532` — `_run_game` (runner.py:113) blocks the heartbeat thread; long games silently mark the sidecar offline. Heartbeat is on the same thread as the game loop (runner.py:286–292) — only refreshed between games.
- `keisei-94c618781b` — NaN/Inf logits crash `np.random.choice`; the existing guard at runner.py:160 only handles `total < 1e-10` but not non-finite logits feeding `np.exp`.
- `keisei-1b408ed6dc` — `__main__.py` (showcase/__main__.py:1–4) calls `main()` at module import, missing `if __name__ == "__main__"` guard. Confirmed by reading the 4-line file.
- Code-observed: `_load_entry` (runner.py:102) uses raw SELECT `*` but accesses columns by name — fragile if schema diverges; tightly coupled to `league_entries` shape.

**Confidence:** High for runner control flow, db_ops API and showcase-table writer ownership (read 100% of all 5 modules). High for the linked filigree concerns — each cites the exact file:line evidence. Medium for the FFI behaviour of `SpectatorEnv` — relied on FFI ownership comment in the brief; not re-verified against shogi_gym source.

---

## Information Gaps & Caveats

- The webui agent owns the consumer side of every WS message — server-side payloads listed above describe what is *sent*, not what is *expected* by clients.
- Training-side writers (e.g. `record_result` for `head_to_head`, `write_game_features` callers, `write_style_profile` callers) live outside `keisei/db.py` itself in `keisei.training.*` and are owned by the training agent.
- The shogi_gym FFI surface (PyClasses returned by `SpectatorEnv.legal_moves_with_usi`, `step`, `get_observation`) is owned by the Rust agent.
- I did not run the test suite to verify migration idempotency end-to-end; claim is based on `IF NOT EXISTS`/`_migrate_add_column`'s duplicate-column tolerance (db.py:30–35).
- I did not enumerate every consumer of `AppConfig`; that boundary belongs to the training agent per scope guard.
