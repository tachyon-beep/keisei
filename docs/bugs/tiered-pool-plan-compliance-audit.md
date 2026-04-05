# Tiered Opponent Pool — Plan Compliance Audit

**Date:** 2026-04-05
**Plan:** `docs/concepts/tiered-opponent-pool.md` (Kenshi Mixed League Full Design)
**Scope:** All 44 Python source + test files reviewed by independent Opus agents
**Assumption:** All phases (0-4) fully implemented — this is the final state

---

## Executive Summary

- **44 files reviewed** (23 source, 21 test files + conftest/_helpers)
- **20 files clean** (no plan-relevant findings)
- **24 files with findings** (total ~85 raw findings, deduplicated to categories below)

The core league logic is solid — `tiered_pool.py`, `tier_managers.py`, `role_elo.py`, `match_scheduler.py`, `frontier_promoter.py`, `historical_gauntlet.py`, and `concurrent_matches.py` all pass compliance checks. The issues cluster in three areas:

1. **Config shape gaps** — missing keys, name mismatches, and absent sections vs the plan's section 17 TOML
2. **Dead data model fields** — columns exist in the schema but are never written
3. **Test coverage holes** — core lifecycle paths (promote, evict, frontier clone) lack integration tests

---

## PLAN_DEVIATION Findings (Implementation)

### D1. Config: Missing top-level `enabled` and `mode` fields
- **Plan section:** section 17 `[league]`
- **File:** `config.py:275-331`
- **Issue:** `LeagueConfig` has no `enabled: bool` or `mode: str` field. League enablement is presence-based (`league: LeagueConfig | None`).
- **Impact:** Users cannot temporarily disable league without removing the entire `[league]` TOML section.

### D2. Config: Missing `[league.storage]` section entirely
- **Plan section:** section 17 `[league.storage]`
- **File:** `config.py:459` — section is `pop()`'d and discarded
- **Issue:** `clone_on_promotion = true` and `persist_optimizer_for_dynamic = true` have no config representation. Behaviors are hardcoded (correctly), but the config contract is missing.

### D3. Config: Missing config keys across multiple sections
- **Plan section:** section 17
- **Files:** `config.py` various locations
- **Missing keys:**
  - `[league.frontier_static]`: `replace_policy`, `span_selection`
  - `[league.recent_fixed]`: `retire_if_below_dynamic_floor`
  - `[league.history]`: `enabled`, `selection`, `active_league_participation`, `benchmark_gauntlet_interval_epochs`
  - `[league.matchmaking]`: `pairing_policy`, five match-class weight ratios (`dynamic_dynamic_weight` etc.)
  - `[league.elo]`: `track_role_specific`
  - `[league.dynamic]`: `batch_reuse` (deferred to Phase 4 per implementation plan)
- **Note:** Many of these are hardcoded behaviors with only one supported value. The plan specifies them as config for forward-compatibility.

### D4. Config: Field name mismatches vs plan TOML
- **Plan section:** section 17
- **Files:** `config.py`
- **Mismatches:**
  - `lr_scale` vs plan's `learning_rate_scale`
  - `frontier_k`/`dynamic_k`/`recent_k` vs plan's `frontier_benchmark_k`/`dynamic_league_k`/`recent_initial_k`
- **Impact:** Low — functionally equivalent, but TOML files written to match the plan document won't load without alias mapping.

### D5. Config: `[league.matchmaking]` alias maps to wrong config class
- **Plan section:** section 17
- **File:** `config.py:444-448`
- **Issue:** `[league.matchmaking]` TOML is aliased to `PriorityScorerConfig`, but the plan's `[league.matchmaking]` contains concurrency keys + pairing policy + weight ratios that don't exist on `PriorityScorerConfig`.

### D6. DB: Column name mismatches vs plan data model
- **Plan section:** section 13
- **File:** `db.py`
- **Mismatches:**
  - `source_epoch` -> `created_epoch`
  - `games_total` -> `games_played`
  - Table `league_matches` -> `league_results`
  - `created_at` -> `recorded_at` (in league_results)
- **Impact:** Low — semantic equivalents, but creates friction when cross-referencing plan vs code.

### D7. `games_vs_frontier`/`games_vs_dynamic`/`games_vs_recent` never updated
- **Plan section:** section 13.1
- **File:** `opponent_store.py:717-761`
- **Issue:** `record_result()` increments `games_played` but never updates per-role game counters. The columns exist in the DB and dataclass but are permanently zero.
- **Impact:** Medium — tier review logic that needs per-role exposure data (section 6.2, 7.1) cannot use these counters. The Frontier exposure requirement (section 12 item 2) is unenforceable.

### D8. `training_enabled` not set correctly for frozen roles on clone
- **Plan section:** section 6.1, 6.2, 14
- **File:** `opponent_store.py:386-393`
- **Issue:** `clone_entry()` doesn't set `training_enabled` in INSERT. DB default is `1` (True), so Frontier Static and Recent Fixed clones have `training_enabled = True`. Currently harmless because `get_trainable()` filters by role, but the data model is wrong.

### D9. `match_type` always "calibration" — training matches mislabeled
- **Plan section:** section 8.2, 13.3
- **File:** `tournament.py:327, 432`
- **Issue:** Both `_run_concurrent_round` and `_run_one_match` hardcode `match_type="calibration"` for ALL matches. Training matches (D-vs-D, D-vs-RF) should be `"train"`.
- **Impact:** Medium — analytics/dashboard cannot distinguish training from calibration matches. The `is_training_match()` function exists but isn't used for labeling.

### D10. Frontier exposure requirement not enforced
- **Plan section:** section 12 item 2
- **Files:** `tournament.py`, `priority_scorer.py`, `match_scheduler.py`
- **Issue:** `games_vs_frontier` is never written (see D7). No scheduling bonus for Dynamic entries under-exposed to Frontier Static. The 20% D-vs-FS match weight provides statistical coverage but not per-entry guarantees.

### D11. No GPU back-pressure or inference-only fallback in Dynamic trainer
- **Plan section:** section 10.4
- **File:** `dynamic_trainer.py`
- **Issue:** Rate limiting (updates/minute) and per-entry disable-on-error exist. Missing: GPU utilization monitoring, queue depth back-pressure, and global inference-only fallback when training becomes broadly unstable.

### D12. `HistoricalSlot` dataclass omits `selected_at` field
- **Plan section:** section 13.2
- **File:** `historical_library.py:16-27`
- **Issue:** DB stores `selected_at`, `get_historical_slots()` fetches it, but `HistoricalSlot` drops it when constructing objects.

### D13. `metadata.json` not updated for Dynamic optimizer state
- **Plan section:** section 14
- **File:** `opponent_store.py:830-850`
- **Issue:** `save_optimizer()` writes `optimizer.pt` and updates DB, but never updates `metadata.json`. The sidecar is not self-describing for Dynamic entries.

### D14. Server: `league_transitions` data not sent to frontend
- **Plan section:** section 15.2
- **File:** `server/app.py:200-216`
- **Issue:** `read_league_data()` never queries `league_transitions`. Dashboard cannot show admission queue, evictions, or promotions.

---

## P1 Findings (Non-plan bugs)

### P1-1. Server: No role-specific win-rate aggregation
- **File:** `server/app.py`
- **Issue:** All `league_results` are sent raw with no limit. No pre-aggregated role-specific win rates. Unbounded data transfer to client.

### P1-2. Eviction tests use `elo_rating` not `elo_dynamic`
- **File:** `test_tier_managers.py:281-315`
- **Issue:** Eviction tests pass by coincidence because `elo_rating` and `elo_dynamic` start equal. If initialization changes, tests become silently wrong.

### P1-3. `test_opponent_store_phase3.py`: SCHEMA_VERSION assertion brittle
- **File:** `test_opponent_store_phase3.py:59-60`
- **Issue:** `assert SCHEMA_VERSION == 1` contradicts the docstring ("schema v6") and will break on any schema bump.

---

## Test Coverage Gaps (Grouped by Theme)

### T1. Core lifecycle paths untested in integration
- **Missing:** Recent Fixed -> Dynamic PROMOTE path (`test_tiered_pool.py`)
- **Missing:** Dynamic -> Frontier Static clone via `on_epoch_end` (`test_tiered_pool.py`)
- **Missing:** Dynamic eviction when tier is full (`test_tiered_pool.py`)
- **Missing:** `protected_candidate_ids` (Frontier promotion candidates exempt from eviction) — zero tests anywhere
- **Missing:** End-to-end lifecycle through training loop (`test_katago_loop_integration.py`)

### T2. Weighted tournament mode completely untested
- **Missing:** `TournamentMode.WEIGHTED` / `_weighted_sample()` — zero tests anywhere
- **Missing:** `MATCH_CLASS_WEIGHTS` constant value assertions
- **Missing:** `is_training_match()` / `TRAINING_CLASSES` — zero tests anywhere
- **Missing:** `classify_match()` for all role combinations

### T3. Config defaults not validated against plan
- **Missing:** `FrontierStaticConfig`, `RecentFixedConfig`, `DynamicConfig` default assertions in `test_league_config.py`
- **Missing:** `HistoricalLibraryConfig`, `GauntletConfig`, `RoleEloConfig` defaults
- **Missing:** `ConcurrencyConfig`, `PriorityScorerConfig` defaults
- **Missing:** Full section-17 TOML round-trip integration test
- **Missing:** `[league.storage]` section entirely

### T4. DB schema coverage incomplete
- **Missing:** `historical_library` table column assertions (`test_db.py`)
- **Missing:** `league_results` Elo before/after and training_updates columns (`test_db_league_schema.py`)
- **Missing:** `league_transitions` column and content tests
- **Missing:** Role enum value insertion tests
- **Missing:** FK constraint enforcement test (`PRAGMA foreign_keys`)

### T5. Opponent store: clone and optimizer paths
- **Missing:** Optimizer save/load round-trip test (`test_opponent_store.py`)
- **Missing:** Dynamic-to-Frontier-Static clone test
- **Missing:** Filesystem layout (section 14) validation test
- **Missing:** `training_enabled`, `games_vs_*`, `retired_at` field coverage (`test_opponent_store_phase3.py`)

### T6. Tier manager rejection criteria untested
- **Missing:** Volatility/spread gate for Recent Fixed review (`test_tier_managers.py`)
- **Missing:** Streak-based rejection for Frontier promotion (tests use `streak_epochs=0`)
- **Missing:** Lineage overlap rejection (tests use `max_lineage_overlap=10`, effectively disabled)

### T7. Server/dashboard plan compliance
- **Missing:** Historical library and gauntlet_results in WebSocket init (`test_server.py`)
- **Missing:** Role field propagation in entries
- **Missing:** Multi-view Elo metrics in payload
- **Missing:** `league_update` message content tests

### T8. Dynamic training via concurrent path untested
- **Missing:** `_run_concurrent_round` Dynamic training trigger (`test_phase3_rollout_wiring.py`)
- **Missing:** `RoleEloTracker` integration in match result path
- **Missing:** `match_type` classification assertions in result recording

### T9. Historical library exclusion from active league untested
- **Missing:** No test anywhere asserts historical entries are excluded from active-league matchmaking
- **Missing:** No test for `active_league_participation = false` config

---

## Clean Files (No Findings)

### Source (14 clean):
`tiered_pool.py`, `tier_managers.py`, `frontier_promoter.py`, `historical_gauntlet.py`, `role_elo.py`, `match_scheduler.py`, `match_utils.py`, `concurrent_matches.py`, `transition.py`, `evaluate.py`, `demonstrator.py`, `checkpoint.py`, `katago_ppo.py`, `algorithm_registry.py`

### Tests (12 clean):
`test_historical_gauntlet.py`, `test_role_elo.py`, `test_match_utils.py`, `test_match_pool.py`, `test_priority_scorer.py`, `test_db_edge_cases.py`, `test_config.py`, `test_tiered_pool_phase3.py`, `test_split_merge.py`, `test_split_merge_transitions.py`, `test_katago_loop.py`, `test_katago_ppo.py`, `conftest.py`/`_helpers.py`

---

## Recommended Priority Order

1. **D9** (match_type mislabeling) — one-line fix, high data integrity impact
2. **D7 + D10** (games_vs_* counters + frontier exposure) — enables diversity controls
3. **D8** (training_enabled on frozen clones) — data model correctness
4. **T1** (lifecycle integration tests) — most critical coverage gap
5. **T2** (weighted tournament mode tests) — zero coverage for a core feature
6. **D11** (GPU back-pressure / inference fallback) — safety rail for production
7. **D14** (transitions in dashboard) — observability gap
8. **D1-D5** (config shape) — forward-compatibility and plan alignment
9. **T3-T9** (remaining test gaps) — breadth coverage
