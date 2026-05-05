# Technical Debt Register — Keisei

**Date:** 2026-05-05
**Sources:** `04-final-report.md` §5, `02-subsystem-catalog.md` Concerns sections, `temp/validation-catalog.md`, filed Filigree bugs.
**Companion:** `05-architecture-critique.md` (prose); this is the data table.

## Conventions

- **Severity** matches `04-final-report.md` §5 risk ranking. Critical = correctness/data-loss risk live in the hot path; High = silent regression or known bug filed P1; Medium = bounded risk, filed P2 cluster, or maintainability threshold; Low = stylistic / planned migration / cosmetic. Where this register disagrees with the final report it is called out in the item's Impact field.
- **Effort** is calibrated against current code shape: **S** ≤1 day (single-helper edit), **M** 1–5 days (cross-module change with new tests), **L** 1–3 weeks (path deletion / decomposition / migration), **XL** >3 weeks (deep refactor of foundational file).
- **Filigree IDs** use the form `keisei-<hash>`; references match the catalogue. Where an ID was closed/in-progress at session start it is noted inline.

## Summary by Severity

| Severity | Count |
|----------|-------|
| Critical | 2 |
| High     | 6 |
| Medium   | 9 |
| Low      | 4 |
| **Total**| **21** |

---

## Debt Items (ranked Critical → Low)

### DEBT-001 — Non-atomic match-result recording in `LeagueTournament`
- **Severity:** Critical
- **Effort:** M (1–3 days; wrap in single `OpponentStore` transaction, add regression test for crash-mid-record)
- **Subsystem(s):** C3
- **Evidence:** `keisei/training/tournament.py:352–460` (`_record_match_result`); store calls at `:413`, `:430`, `:431`, `:433`. Catalog C3 Concerns line 276. Filigree `keisei-fa604bad63` (P1).
- **Impact:** `record_result` + two `update_elo` + `role_elo_tracker.update_from_result` are issued as separate transactions. Crash, OS kill, or DB busy-error between them leaves Elo, role-Elo, and result rows inconsistent — the league's primary source of truth diverges silently. This is the only Critical correctness bug in the in-process tournament path.
- **Proposed fix:** Open one `OpponentStore.transaction()` context spanning all four writes (the RLock already supports reentrance from `tier_managers`). Introduce a single `record_match_atomic(result, role_updates, elo_updates)` helper on `OpponentStore` so both recording paths can call it.
- **Dependencies:** Done before DEBT-002 (which deletes one of the two paths) so that whichever path survives is correct.

### DEBT-002 — Two parallel match-recording paths drift
- **Severity:** Critical
- **Effort:** L (1–3 weeks; pick one path, delete the other, re-route `katago_loop.py` mode switch, migrate any sidecar-only consumers)
- **Subsystem(s):** C3
- **Evidence:** `tournament.py:_record_match_result` vs `tournament_runner.py:_record_result`; mode switch at `katago_loop.py:672`. Catalog C3 Concerns line 290 + cross-cutting line 395. Filigree `keisei-ea85c3d5b5` (P2 lost-update race), `keisei-6cb0990f53` (PriorityScorer never sees sidecar completions), `keisei-3067a55ab0` (broad except → terminal failed batch), `keisei-9cae4c09b4` (`check_round_completion` not idempotent), `keisei-90d4e0c1b2` (epoch skew).
- **Impact:** The sidecar tournament re-implements Elo bookkeeping with no shared code with the in-process path; PriorityScorer is starved of completions when sidecar mode is active, breaking pairing quality. Final report flags this as **High**; this register escalates to **Critical** because the bug surface (5 filed P2s on the sidecar side alone) and the structural divergence risk compound. Per §6 of the final report this is incomplete migration debt — the right fix is finishing the migration, not maintaining both.
- **Proposed fix:** Decide the canonical path (sidecar appears to be the strategic direction per the rebuild context). Extract Elo/result/style write into a single shared module callable by both, ship behind the existing config flag, run both in shadow for a sprint, then delete the loser. Wire dispatcher's `PriorityScorer.observe()` into the sidecar completion handler before deletion.
- **Dependencies:** DEBT-001 (so the surviving path is transactional).

### DEBT-003 — Slot-reuse and partial-load bugs in `concurrent_matches.py`
- **Severity:** High
- **Effort:** M (3–5 days; partition reset on slot acquire + cleanup contract on shared `OpponentStore._model_cache`)
- **Subsystem(s):** C3
- **Evidence:** `keisei/training/concurrent_matches.py` (slot lifecycle ~`:64,95`); `opponent_store.py:344` shared LRU. Catalog C3 Concerns line 277–282. Filigree `keisei-53eb4eb1f8` (P1, slot reuse leaks games), `keisei-4b6c36cd2b` (P1, partial-load `.cpu()` poisons shared LRU).
- **Impact:** Slot reuse without partition reset leaks games across pairings — silent contamination of league results. Partial-load cleanup `.cpu()`-mutates a model that is still in the shared `OpponentStore._model_cache`, poisoning subsequent matches that draw the same opponent.
- **Proposed fix:** Make `_MatchSlot.reset_for_pairing` reset the env partition unconditionally; replace in-place `.cpu()` cleanup with cache-aware drop via `OpponentStore.release(entry_id)` that bumps refcount and only mutates a fresh load. Add a partition-reset assertion in tests.
- **Dependencies:** None.

### DEBT-004 — `historical_gauntlet.run_gauntlet` tuple-unpacks a dataclass
- **Severity:** High
- **Effort:** S (≤1 day; one site, one test)
- **Subsystem(s):** C3
- **Evidence:** `keisei/training/historical_gauntlet.py:run_gauntlet`; `MatchOutcome` dataclass at `match_utils.py:21`. Catalog C3 Concerns line 283. Filigree `keisei-4509042dd1` (P1).
- **Impact:** Every gauntlet match is silently treated as a failure because the function unpacks `(a, b)` from a dataclass instance. Milestone evaluation against the historical library has been wrong since the dataclass refactor; gauntlet-derived Elo deltas in production are noise.
- **Proposed fix:** Replace tuple-unpack with attribute access. Add a `mypy --strict`-style annotation on the call boundary so the regression cannot recur silently.
- **Dependencies:** None.

### DEBT-005 — `FrontierManager.review` retires Static incumbent then aborts
- **Severity:** High
- **Effort:** S (≤1 day; reorder side-effects, add invariant test)
- **Subsystem(s):** C3
- **Evidence:** `keisei/training/tier_managers.py` (`FrontierManager.review` at `:89`). Catalog C3 Concerns line 284. Filigree `keisei-959d0eebe7` (P1).
- **Impact:** When the Dynamic candidate becomes inactive between selection and promotion, the Static incumbent is already retired, then the promotion aborts — silent shrink of the Frontier tier. Capacity drifts down across runs without operator visibility.
- **Proposed fix:** Reorder to validate Dynamic candidate availability before any Static-side mutation; or wrap both mutations in a single `OpponentStore` transaction with rollback on abort.
- **Dependencies:** None.

### DEBT-006 — `katago_loop.py` is 1,989 LoC in one class
- **Severity:** High
- **Effort:** XL (3–6 weeks; staged extraction)
- **Subsystem(s):** C1b
- **Evidence:** `keisei/training/katago_loop.py`; class `KataGoTrainingLoop` at `:454`; 14 in-bucket imports verified by `validation-catalog.md`. Catalog C1b Concerns line 189; final-report §3.2 + §7 explicit recommendation.
- **Impact:** The loop, opponent rotation, league bookkeeping, snapshotting, DB writes, and `keisei-train` CLI all share one class. Single biggest maintainability risk in C1b; every league-shape change touches this file. Direct SQL UPDATE on `training_state.total_epochs` at `:843–848` already bypasses the typed helper layer — the file is past the threshold where convention holds.
- **Proposed fix:** Stage 1 — extract CLI / `main()` to `train_cli.py`. Stage 2 — extract `LeagueIntegration` (rotation + match dispatch + bookkeeping) into a collaborator. Stage 3 — extract `CheckpointAndStateWriter` so DB writes go through one seam. Stage 4 — replace `:843` direct SQL with a `db.update_total_epochs` helper.
- **Dependencies:** DEBT-002 (collapsing dual tournament paths reduces league-integration surface area before extraction).

### DEBT-007 — `db.py` is 1,158 LoC with 21 tables and 12 entity helper families
- **Severity:** High
- **Effort:** L (2–3 weeks; per-entity module split, no schema change)
- **Subsystem(s):** E2
- **Evidence:** `keisei/db.py`; final-report §3.3; catalog E2 (table inventory, migration registry at `:175`). Recommended target in §7.
- **Impact:** Foundational file at the maintenance threshold. Natural per-entity seams already visible (metrics, snapshots, training_state, league, head_to_head, game_features, style_profiles, showcase_*, tournament_queue). Migration registry stable at v8 — safe moment to split.
- **Proposed fix:** Split into `keisei/db/{__init__.py,_connect.py,migrations.py,metrics.py,snapshots.py,training_state.py,league.py,head_to_head.py,game_features.py,style_profiles.py,showcase.py,tournament_queue.py}`. Keep public symbols re-exported from `keisei.db` to preserve all current import sites. Migration registry stays in one file.
- **Dependencies:** None — pure decomposition with no behavioural change.

### DEBT-008 — `KataGoPPOAlgorithm.flush_timings()` is never called
- **Severity:** High
- **Effort:** S (≤1 day; wire into epoch boundary)
- **Subsystem(s):** C1a
- **Evidence:** `keisei/training/katago_ppo.py` (KataGoPPOAlgorithm body `:391–991`, not exhaustively read). Filigree `keisei-ca5e280cae` (P2, in-progress at session start). Final report §5 risk bullet.
- **Impact:** `select_actions` allocates CUDA timing events that are never released — unbounded GPU memory growth on long runs. Final report grades **Medium**; this register grades **High** because it manifests as long-run training failure, not a contained risk.
- **Proposed fix:** Call `flush_timings()` from the loop's epoch boundary (or from `train_epoch` end) inside `katago_loop.py`. Add a metric counter so the operator sees event count over time.
- **Dependencies:** None.

### DEBT-009 — Hyperparameter validation gaps allow silent gradient ascent
- **Severity:** High
- **Effort:** M (2–3 days; tighten `__post_init__` across dataclasses)
- **Subsystem(s):** C1a, D
- **Evidence:** `katago_ppo.py:102–116` (`KataGoPPOParams.__post_init__`); `sl/trainer.py:32–42` (`SLConfig`). Filigree `keisei-cb7008ac73` (PPO NaN/negative), `keisei-bef32b64a8` (`0.0 * NaN` poisoning), `keisei-678359b7aa` (negative SL lambdas), `keisei-b2324bf429` (rollout buffer leading-dim broadcast), `keisei-02c52bae2d` (sticky env_ids), `keisei-d43009528f` (gae scalar broadcast).
- **Impact:** Negative `lambda_*` causes silent gradient ascent (the loss term flips sign). NaN slips through `__post_init__`. Buffer add() doesn't validate leading dims so a misshaped tensor is broadcast into adjacent envs. Six related filings; all silent corruption modes.
- **Proposed fix:** Add `_require_positive`, `_require_finite`, `_require_in_range` helpers; apply across `KataGoPPOParams`, `SLConfig`, and the rollout buffer's `add()` site. Reject NaN explicitly. Add a unit test grid covering negative/NaN/zero per parameter.
- **Dependencies:** None.

### DEBT-010 — `vec_env.rs` Send/Sync soundness rests on doc-only invariants
- **Severity:** Medium
- **Effort:** M (3–5 days; wrap in safe abstraction or replace with `&mut [T]` chunks via `par_chunks_mut`)
- **Subsystem(s):** B
- **Evidence:** `shogi-engine/crates/shogi-gym/src/vec_env.rs:66-67` (`unsafe impl Send/Sync for SendPtr<T>`); large unsafe blocks `:348–458` (happy path) and `:480–539` (panic recovery). Catalog B Concerns line 102. Filigree `keisei-1883589523` (P4 stylistic).
- **Impact:** Production-stable today (`catch_unwind` contains panics; `debug_assert` checks disjoint indices in dev). But the disjoint-index invariant is convention-enforced across four mode-tag dispatch sites that must stay in sync (`vec_env.rs:128–137`). Adding a new observation/action mode requires touching all four correctly. Soundness regression risk on any Rust contributor without context.
- **Proposed fix:** Replace `SendPtr<T>` writes with `par_chunks_mut(env_stride)` so each rayon worker owns a `&mut` slice — borrow checker enforces disjointness. Where shape mismatches make chunking impossible, encapsulate the unsafe in a single private struct with a typed safe API. Either path collapses 4 dispatch sites to 1.
- **Dependencies:** None.

### DEBT-011 — Server fingerprint-based change detection misses diffs
- **Severity:** Medium
- **Effort:** S (≤1 day; widen tuples, add tests)
- **Subsystem(s):** F
- **Evidence:** `keisei/server/app.py:47` (`_style_fingerprint`), `:583` (showcase status fingerprint). Filigree `keisei-8108b42644` (style: raw_metrics/percentile recomputes slip past), `keisei-d26b243465` (showcase: speed-change diffs missed).
- **Impact:** Style profile recomputes that don't change `(checkpoint_id, status, primary_style)` never reach the WebUI; speed-change updates in the queue don't trigger a status push. Both are silent UI staleness — the dashboard shows old data while the DB is current.
- **Proposed fix:** Hash a content digest (e.g. `hashlib.blake2b` of the JSON payload) instead of a hand-picked tuple. One implementation, applied at both fingerprint sites. Add a test that mutates each downstream field and asserts a push fires.
- **Dependencies:** None.

### DEBT-012 — Server init undercounts and command-validation gaps
- **Severity:** Medium
- **Effort:** S (≤1 day; per-issue, several small edits)
- **Subsystem(s):** F
- **Evidence:** `app.py:354` (total_episodes truncated to first 500), `:138` (gpus key conditional), `:284` (unbounded `read_game_snapshots`), `:491` (`_handle_match_request`). Filigree `keisei-f11f3179f7`, `keisei-974d6aba11`, `keisei-5f12aa7362`, `keisei-bcc71be6ef`.
- **Impact:** `total_episodes` truncates to first 500 metrics rows then advances the cursor — the running total is permanently behind. `gpus` key absent on non-zero `nvidia-smi` exit (UI breaks instead of degrading). Init reads are unbounded for `read_game_snapshots`. Match request doesn't verify entry IDs exist before queueing.
- **Proposed fix:** Sum `total_episodes` via `SELECT SUM(...)` not by iterating the truncated init slice. Move `gpus = []` to the outer scope so all paths set it. Add `read_game_snapshots(limit=N)` overload and use it in init. Validate `entry_id_1/_2` against `league_entries` in `_handle_match_request` before enqueue.
- **Dependencies:** None.

### DEBT-013 — `ModelCache` key omits checkpoint mtime → stale weights
- **Severity:** Medium
- **Effort:** S (≤1 day; one-line key change + invalidation test)
- **Subsystem(s):** G
- **Evidence:** `keisei/showcase/inference.py:127` (`ModelCache` key is `(entry_id, checkpoint_path)`). Catalog G Concerns line 717. Filigree `keisei-8c55b48bcc`.
- **Impact:** When a checkpoint is overwritten in place at the same path (a routine training operation), the showcase serves stale weights for the cached entry indefinitely. WebUI shows old policy behaviour while training has moved on.
- **Proposed fix:** Include `os.stat(path).st_mtime_ns` in the cache key. Invalidate on mtime change. Add a test that reuses a path with new bytes and asserts a reload.
- **Dependencies:** None.

### DEBT-014 — Showcase sidecar heartbeat blocks on game loop
- **Severity:** Medium
- **Effort:** M (2–3 days; separate heartbeat thread + shutdown discipline)
- **Subsystem(s):** G
- **Evidence:** `keisei/showcase/runner.py:286–292` (heartbeat refreshed only between games); `_run_game` at `:113`. Filigree `keisei-6045789532`.
- **Impact:** Long showcase games (deep PVs, slow speed) silently mark the sidecar offline because heartbeat only refreshes between games. The dashboard then renders the sidecar as dead while it is in fact running, which triggers `cleanup_orphaned_games` on the next restart.
- **Proposed fix:** Run heartbeat on a dedicated daemon thread with `threading.Event`-based shutdown signalling; refresh on a 10s tick regardless of game state. Add an integration test using a slow mock game.
- **Dependencies:** None.

### DEBT-015 — Showcase write-ordering and cleanup gaps
- **Severity:** Medium
- **Effort:** S (≤1 day each; cluster of three small edits)
- **Subsystem(s):** G
- **Evidence:** `keisei/showcase/db_ops.py:154` (`UPDATE total_ply = ?` unconditional), `:234` (`cleanup_orphaned_games` skip on fresh heartbeat), `:49` (`claim_next_match` IntegrityError). Filigree `keisei-6895e438ca`, `keisei-2f4b87732b`, `keisei-f1e5004e86`.
- **Impact:** `total_ply` regresses on out-of-order writes (should be `MAX`). Fast-restart leaves orphaned in-progress games stuck because heartbeat is fresh. Partial UNIQUE INDEX on `running` showcase entry can collide on claim under contention.
- **Proposed fix:** `UPDATE total_ply = MAX(total_ply, ?)`. Decouple orphan-cleanup from heartbeat freshness — clean up on every startup, not conditionally. Wrap claim in `INSERT ON CONFLICT DO NOTHING`-equivalent or retry-on-IntegrityError.
- **Dependencies:** None.

### DEBT-016 — `read_tournament_stats` swallows all exceptions as `None`
- **Severity:** Medium
- **Effort:** S (≤1 day; narrow the except, add structured logging)
- **Subsystem(s):** E2 (consumed by C3)
- **Evidence:** `keisei/db.py:942`; same anti-pattern existed historically in `read_head_to_head`. Filigree `keisei-44945464ac`.
- **Impact:** Any DB-level error is hidden as a missing-row condition. The dashboard shows a benign empty state when the actual cause is a malformed JSON column or schema mismatch — debugging time amplified.
- **Proposed fix:** Catch only `sqlite3.OperationalError` for "no such table" and `sqlite3.DatabaseError`; let everything else propagate. Log the exception with table name. Same edit applies to `read_elo_history` and any other broad-except read helpers (audit while in-flight).
- **Dependencies:** None — but combine with DEBT-007 split to do this per-module.

### DEBT-017 — DB writer concurrency is convention-based
- **Severity:** Medium
- **Effort:** M (2–4 days; document + add `BEGIN IMMEDIATE` audit + connection pool decision)
- **Subsystem(s):** E2
- **Evidence:** `db.py:16–22` (`check_same_thread=False`, no pool); cross-process writers are training loop + showcase sidecar + (now) tournament sidecar. Final-report §5 final bullet; catalog E2 Confidence.
- **Impact:** WAL + `busy_timeout=5000` works today, but cross-process write ordering between trainer / showcase sidecar / tournament sidecar depends entirely on caller behaviour. Not formally documented in `db.py`. As the league sidecar is wired in (DEBT-002 outcome), the absence of explicit single-writer-per-table contracts becomes a correctness risk, not just a performance one.
- **Proposed fix:** Document the writer-per-table contract in `db.py` module docstring (training writes most; sidecar owns `showcase_*`; tournament_runner owns `tournament_pairing_queue` claim/complete). Audit all multi-statement writes for `BEGIN IMMEDIATE`. Decide explicitly whether a connection pool is warranted; if no, document the per-call open-close discipline.
- **Dependencies:** Best done alongside DEBT-007 split.

### DEBT-018 — Style-profile constants duplicated between Rust and Python
- **Severity:** Medium
- **Effort:** S (≤1 day; codegen or shared constants module)
- **Subsystem(s):** C4, B
- **Evidence:** `keisei/training/game_feature_tracker.py:19–28` hard-codes move-type ranges that mirror `spatial_action_mapper.rs`. Catalog C4 Concerns line 317.
- **Impact:** Drift risk if the Rust spatial encoding changes — Python feature tracker silently miscategorises drops/promotions. No test cross-checks the constants. Style profiling becomes wrong without any signal.
- **Proposed fix:** Generate the Python constants from a Rust-side declarative table at build time (or expose the constants on the PyO3 module surface, e.g. `shogi_gym._native.SPATIAL_MOVE_TYPE_PROMOTE_START`). Add a startup assertion that compares Python constants to the FFI-exposed values.
- **Dependencies:** None.

### DEBT-019 — SL `prepare.py` writes outcome-proxy score targets
- **Severity:** Medium
- **Effort:** M (3–5 days; compute material balance per position; reshard)
- **Subsystem(s):** D
- **Evidence:** `keisei/sl/prepare.py:171` placeholder comment; outcome-proxy used in lieu of per-position material balance. Filigree `keisei-25cb7bb826` (P1).
- **Impact:** SL's score head trains on a constant-per-game label, not a position-specific signal — the score head learns the trivial outcome distribution rather than positional evaluation. Reduces SL→RL transfer quality. Final report does not list this; this register surfaces it from catalog D.
- **Proposed fix:** Compute per-position material balance using `shogi_core::material_balance` (already used in `shogi-gym`); rewrite shards. Backfill or rebuild the SL data lake. Note the layout/version bump in `_SHARD_DTYPE`.
- **Dependencies:** None.

### DEBT-020 — `evaluate.py` ignores model hyperparameters
- **Severity:** Low
- **Effort:** S (≤1 day; thread params through CLI + load)
- **Subsystem(s):** C5
- **Evidence:** `keisei/training/evaluate.py:75` (`params_a or {}`); imports private `_get_policy_flat` from `demonstrator.py`. Filigree `keisei-9a5ac20307` (P2).
- **Impact:** `keisei-evaluate` fails on non-default architectures because `build_model(arch, {})` constructs default-shape weights that don't match the checkpoint. CLI is unusable for the architectures most worth evaluating.
- **Proposed fix:** Read params from a side-car `*.params.json` next to the checkpoint, or thread `--params-a/--params-b` flags. Promote `_get_policy_flat` to a public helper.
- **Dependencies:** None.

### DEBT-021 — WebUI "createPersistedStore" repeated 9× verbatim
- **Severity:** Low
- **Effort:** S (≤1 day; extract helper, replace call sites)
- **Subsystem(s):** H7, H4
- **Evidence:** Catalog H7 Concerns line 989 (corrected to 9 per validation report fix). LocalStorage keys: `aboutLevel, activeTab, audioEnabled, keisei_league_event_run_marker, keisei_league_events, keisei-theme, notationStyle, showcaseHeatmapEnabled, showcaseSpeed`.
- **Impact:** ~40 lines of duplicate logic across 7 stores; SSR-guard pattern repeated; validators inline. Low risk of divergent behaviour as the pattern is small, but Svelte 5 migration (DEBT-022) will multiply the cost of any later refactor.
- **Proposed fix:** Extract `createPersistedStore(key, default, validator?)` to `webui/src/stores/_persisted.js`. Replace nine call sites. Tests already exist per-store.
- **Dependencies:** Should ship before DEBT-022 (Svelte 5 migration) so the migration touches fewer call sites.

---

## Theme Index

- **Atomicity / non-transactional writes:** DEBT-001, DEBT-002, DEBT-005, DEBT-015, DEBT-017.
- **Single-file scale (>1k LoC):** DEBT-006 (`katago_loop.py`), DEBT-007 (`db.py`).
- **Convention-not-mechanism (compile-time invariants depending on doc/`debug_assert`):** DEBT-003 (slot/cache cleanup), DEBT-009 (hyperparameter validation), DEBT-010 (Send/Sync), DEBT-017 (writer-per-table), DEBT-018 (Rust↔Python constants).
- **Fingerprint-based change detection misses:** DEBT-011.
- **Silent error swallowing / undercounting:** DEBT-004 (gauntlet failure), DEBT-012 (init undercount), DEBT-016 (broad except), DEBT-020 (evaluate CLI).
- **Resource leaks / staleness:** DEBT-008 (CUDA timing events), DEBT-013 (ModelCache mtime), DEBT-014 (sidecar heartbeat).
- **Cross-language contract surfaces:** DEBT-010, DEBT-018.
- **Data-quality / training-correctness:** DEBT-009, DEBT-019.

Batch-fix campaigns implied by the index:
1. **One-day server pass:** DEBT-011 + DEBT-012 + DEBT-016 — same file, same review, all S-effort.
2. **One-day showcase pass:** DEBT-013 + DEBT-015 — same module, all S-effort.
3. **Hyperparameter hardening sprint:** DEBT-009 alone covers six filed bugs.
4. **League migration completion:** DEBT-001 → DEBT-002 sequenced; unlocks DEBT-006 stage 2.
5. **Foundational decomposition:** DEBT-007 then DEBT-006; both touch import paths broadly so do them adjacent in time.

---

## Excluded Items

- **Svelte 4 → 5 migration cluster** (`keisei-9b1171d032`, `keisei-a5fe9f710e`, `keisei-975949c0b3`, `keisei-a1622bc4cf`). Tracked as planned migration, not debt. Final report §5 grades Low and notes "but planned" — register treats as scheduled work. DEBT-021 (createPersistedStore) is included because it is a same-shape refactor that pays off both before *and* without the migration.
- **`MoveList::new` `uninitialized_array` lint** (catalog A Concerns line 54). Sound idiom, stylistic-only, no behavioural impact.
- **Showcase `__main__.py` missing `if __name__ == "__main__"` guard** (`keisei-1b408ed6dc`). 4-line file, trivial. Fix at next touch.
- **Closed P2 `keisei-cdf80418a1` (catch_unwind)** — verified landed 2026-04-03; not debt.
- **`OpponentStore._pinned` in-memory only** (`keisei-76cc7fdc85`). Acknowledged as known limitation by the catalog; no current consequence; not debt today.
- **Two CSAParser bugs (`keisei-7a3316c590`, `keisei-295086b4cc`, `keisei-7af7ec2b8a`)** — SL data quality issues confined to one parser; bundled under DEBT-019's reshard if/when SL data is regenerated. Surface as standalone if a sponsor wants them individually tracked.
- **`KataGoRolloutBuffer` sticky `_has_*` flags (`keisei-02c52bae2d`)** — folded into DEBT-009 hyperparameter/state hardening sprint.
- **WebUI `MetricsGrid` re-reads CSS-vars on every metrics tick** (catalog H5 Concerns line 927). Performance cost is sub-millisecond; not a register-worthy item.
