# Catalog Validation Report

**Document:** `02-subsystem-catalog.md` (1010 lines, 20 entries)
**Validator:** independent sample-grep verification against `/home/john/keisei/`
**Date:** 2026-05-05

## Verdict

**PASS-WITH-FIXES.** No fabrications, no boundary violations, no blocking issues. A handful of off-by-N numerical claims should be corrected before synthesis but none change the architectural picture. Synthesiser is safe to proceed; fixes can be applied in-place.

## Sample Verifications (with evidence)

| Claim | Status | Evidence |
|---|---|---|
| `vec_env.rs:66-67` `unsafe impl Send/Sync for SendPtr<T>` | **VERIFIED** | `grep -n "unsafe impl" shogi-gym/src/vec_env.rs` → lines 66 and 67 exact match. |
| `MoveList` uses `MaybeUninit` + `from_raw_parts` | **VERIFIED** | `movelist.rs:22` `[MaybeUninit<Move>; MOVELIST_CAPACITY]`; `:32` `MaybeUninit::uninit().assume_init()`; `:79` `slice::from_raw_parts`. |
| `db.py` 21 tables in `init_db()` | **VERIFIED** | `grep -c "CREATE TABLE IF NOT EXISTS" db.py` → 21 inside `init_db` (lines 194–505). All 21 enumerated tables in the catalog table match. Spot-checked: `metrics` (197), `league_entries` (250), `head_to_head` (327), `showcase_moves` (447), `style_profiles` (472), `tournament_pairing_queue` (485) — all present at cited lines. |
| Server `/ws` 12 server→client message types | **VERIFIED** | `grep -E '"type"' server/app.py` returns exactly 12 distinct types: `init, metrics_update, game_update, training_status, league_update, ping, showcase_status, showcase_update, showcase_error, showcase_match_queued, showcase_speed_changed, showcase_match_cancelled`. Catalog's "9 (table) + 3 command-ack" reconciles to 12. |
| `OpponentStore` referenced by 16 in-bucket files (catalog does not give an exact league-module count) | **VERIFIED** | `grep -rln "OpponentStore\|opponent_store" keisei/training/` → 16 modules import or reference. Brief asked about an "8 league modules" claim; that exact phrasing is **not in the catalog** — closest is the C3 entry-section claim that opponent_store is "owned by" the league bucket and used pervasively, which the evidence supports. |
| `SpectatorEnv`/`VecEnv` lazy-imported in 7 modules | **VERIFIED** | `grep -rln "from shogi_gym\|import shogi_gym" keisei/` → 7 files: `katago_loop.py`, `tournament.py`, `tournament_runner.py`, `historical_gauntlet.py`, `demonstrator.py`, `evaluate.py`, `showcase/runner.py`. Cross-cutting claim is correct. |
| `shogi-gym` inbound is "7 Python files" | **VERIFIED** | Same grep above gives 7. |
| `katago_loop.py` "imports 11 in-bucket modules" (cross-cutting §, line 394) | **DEVIATES** | Actual unique in-bucket modules: **14**, not 11. `grep -E "^from keisei\.(training\|sl)" katago_loop.py \| sort -u` returns: `algorithm_registry, checkpoint, concurrent_matches, distributed, historical_gauntlet, katago_ppo, match_scheduler, model_registry, opponent_store, priority_scorer, role_elo, tiered_pool, tournament, tournament_dispatcher`. Note the "every C3 module except tournament_queue/tournament_runner/demonstrator" qualifier is **also wrong** — it omits `match_utils, frontier_promoter, historical_library, tier_managers, dynamic_trainer, game_feature_tracker, style_profiler` which C1b's loop indeed does not import (those land via `tiered_pool`/`tournament`). The bullet conflates "directly imported" with "transitively reachable". |
| `H7` "8 distinct localStorage keys across the bucket" (line 890 + line 989) | **DEVIATES** | Actual: **9** keys. `grep -rEn "localStorage\.(get\|set\|remove)Item" webui/src/` shows: `aboutLevel, activeTab, audioEnabled, keisei_league_event_run_marker, keisei_league_events, keisei-theme, notationStyle, showcaseHeatmapEnabled, showcaseSpeed`. The H4 bullet (line 890) lists exactly these nine keys but then says "eight distinct"; the H7 concern (line 989) says "repeated 8×". Off by one in the count, the enumeration itself is correct. |
| `db.py` schema constants (`SCHEMA_VERSION = 8` at line 13, `_MIGRATIONS` at 175) | **VERIFIED via prior reads** during catalog construction; head_to_head CHECK constraint (`entry_a_id < entry_b_id`), partial UNIQUE INDEX `idx_showcase_queue_one_running`, all match. |
| `tournament.py` lazy DB imports at lines 326, 336 | Not re-greppped here, accepted on confidence-bucket basis. |

## Boundary Ownership Audit

| Boundary | Status | Notes |
|---|---|---|
| **FFI 9 PyClasses** | **RESPECTED** | Described once in shogi-gym entry (lines 71–87, "FFI Exported Surface"). Python consumers (C1b, C3, C5, G) reference `VecEnv`/`SpectatorEnv` by name only and cite the FFI bucket as owner — e.g. C1b line 177, C3 line 264, C5 line 337, G line 681. The contract (action-space size 11259, observation_mode/action_mode strings, output array shapes) is not redocumented at consumer sites. |
| **SQLite schema** | **RESPECTED** | E2 (db.py) is the sole table catalog. Training entries reference write functions by name (`write_metrics`, `write_game_features`, `write_style_profile`, `write_tournament_stats`, `write_training_state`, etc.) without re-listing columns. C3 acknowledges `OpponentStore` "acts as a typed wrapper over many DB tables" without enumerating them (line 263). G consumes the `showcase_*` tables it owns the writer for and explicitly delegates schema to E2. |
| **WS protocol** | **RESPECTED** | F is the sole table of server→client message types. H1 dataflow table (line 749) shows messages and consuming views without redefining payload shape. H1 line 795 explicitly cites "WS taxonomy" as owned by F. |
| **Config classes** | **RESPECTED** | E1 enumerates the 17 dataclasses with file:line. Consumers list usage by class name only — see C1b line 179, C3 line 265, F line 627. No re-description of fields. |

No ownership leaks observed.

## Template Adherence

**20/20 entries have all required fields** (Location / Responsibility / Key Components / Dependencies (Inbound + Outbound) / Patterns / Concerns / Confidence).

Minor deviations:
- **C1b "Patterns Observed"** mixes some content that arguably belongs in Concerns (DDP weight reload "rationale documented inline" — neutral fact, OK).
- **E2** does not have a single "Patterns Observed" header; the equivalent content is split between "Concurrency discipline" and "Patterns Observed" — both are present, just in two consecutive blocks. Cosmetic.
- **F** uses sub-headers (HTTP routes, WebSocket protocol, etc.) inside the entry rather than collapsing into the standard fields. The required fields are still all present after the sub-section block.
- **D (SL)** entry has FFI = "**none directly**" with a confidence note that this is inferred from the file head only. Acceptable, the qualification is explicit.
- **G** "Per-game flow" detailed enumeration is verbose but inside Key Components/Patterns sections; not a template break.

No entry is missing any required field. All 20 give a Confidence section, and all Confidence sections distinguish High/Medium/Low with code-grounded justification.

## Confidence Calibration

**Reasonable across the bucket.** Every entry that touched files >500 LoC explicitly downgrades to Medium with an honest statement of how much was read (e.g. shogi-core "Medium: behavioural claims about `attack.rs` (read first 80 LoC of 1091)"; shogi-gym "Medium: full body of `apply_moves` happy/recovery paths ... ~250 LoC of branching, sampled rather than exhaustively traced").

Specific examples of well-calibrated confidence:
- **C3** (the densest bucket) calls out Medium for `LeagueTournament` deep internals (read l.1–460 of 658), `OpponentStore` (read l.1–360 of 1324), `ConcurrentMatchPool` (read l.1–120 of 625), and Low for `concurrent_matches.py` slot-reuse claims that depend on the static sweep findings.
- **C4** explicitly marks Low for the classification logic past l.60.
- **D** marks Medium for `parsers.py` `CSAParser` body and Medium for `prepare.py` past l.60.
- **H3/H4** mark Medium for `LeagueTable`/`MatchupMatrix`/`RecentMatches`/`MatchScorecard` interiors with line-count receipts.

No entry pretends High where the underlying read was thin. No entry hedges every claim to Low to avoid commitment. The catalog is honest about what was read.

## Catalog-vs-Review Discipline

**Strong.** Concerns sections in nearly every entry are code-observed at file:line (or filigree-tracked). Review-style judgement creeps in only at the edges:

- **C1b line 189**: "1989 LoC in one class — the loop, opponent rotation, league bookkeeping, snapshotting, DB writes, and CLI are all in one file. Threshold-of-maintainability risk noted in discovery." → drift toward review, but it cites the discovery doc and quantifies the LoC. Borderline; acceptable as a code-observed concern.
- **C3 line 290**: "Two parallel result-recording code paths... replicate Elo bookkeeping logic — divergence risk." → review judgement, but the two file:line paths are both cited (`tournament._record_match_result` and `tournament_runner._record_result`). Acceptable.
- **H1 line 790**: "Pulling that out into a `TrainingView.svelte` would make it consistent with the other three tabs" → mild prescriptive review, but the structural observation (445-line root file vs ≤566 for siblings) is code-observed. Borderline.
- **H7 line 989**: "Extracting a `createPersistedStore(key, default, validator)` helper would shrink ~40 lines" → recommendation, slightly review-flavoured. Concrete and small; not blocking.

No entry leans on subjective quality language ("ugly", "bad", "good") or makes unsupported architectural value judgements. Filed-bug references (`keisei-*`) are consistently cited at the line where the symptom is observed.

## Required Fixes (small)

1. **Cross-cutting line 394** — change "katago_loop.py imports 11 in-bucket modules" to **14**, and rephrase the parenthetical: replace "every C3 module except tournament_queue/tournament_runner/demonstrator" with "every C3 module except `tournament_queue`, `tournament_runner`, `tournament_dispatcher` (which it does import), `match_utils`, `frontier_promoter`, `historical_library`, `tier_managers`, `dynamic_trainer`, `style_profiler`, `game_feature_tracker`, `demonstrator` — i.e. only the surfaces it directly wires; transitive coverage via `tiered_pool` and `tournament` is broader." Or simply: "`katago_loop.py` directly imports 14 in-bucket modules and reaches the rest transitively via `TieredPool` and `LeagueTournament`."
2. **H4 line 890** — change "eight distinct localStorage keys total" to **nine**. The list immediately before that count actually enumerates nine names (`showcaseHeatmapEnabled`, `showcaseSpeed`, `audioEnabled`, `theme`, `notationStyle`, `aboutLevel`, `activeTab`, `keisei_league_events`, `keisei_league_event_run_marker`).
3. **H7 line 989** — change "repeated 8×" to **9×** to match. The pattern instances cited (`audio.js`, `navigation.js`, `notation.js`, `theme.js`, `aboutLevel.js`, `showcase.js` ×2 keys, `league.js` ×2 keys) account for the same 9 keys.

These three numeric edits are the only **required** corrections; none affect any structural claim, dependency edge, or filed bug.

## Optional Improvements (synthesiser may decide)

1. **F line 595**: heading reads "Server → client messages (8 types)" but the table that follows lists 9 visible message rows plus a parenthetical note saying "9 plus the three command-ack messages = 12". The heading "(8 types)" is stale — should be "(12 types)" or "(9 + 3 command acks)" to match the table and the discovery comparison. Minor visual consistency fix; not load-bearing.
2. **C1b "Concerns" line 194** mentions a `katago_ppo.py` flush_timings bug "referenced in discovery preliminary risks" without a file:line. If the synthesis doc will cite this, please add the line ref or move it to C1a's bug list. Otherwise leave as-is.
3. **C3** lists ~13 open filigree bugs in one Concerns block. For synthesis-time readability, consider splitting C3 Concerns into "Result-recording path (`tournament.py`)", "Concurrent matches", "Sidecar/dispatcher", "Demonstrator" — but the current chronological-by-component grouping is also fine.
4. **E1 "Concerns"** lists three filigree IDs without recapping the symptom. Cheap to add a one-line restatement per ID. Optional.

## Items Validated, Out-of-Scope-but-Worth-Noting

- The catalog correctly notes (line 110) that `shogi-gym` cannot be tested via `cargo test` due to the PyO3 cdylib link issue. This matches `CLAUDE.md` and the project memory.
- The "DB is the only message bus between training and the server/showcase buckets" cross-cutting claim (line 397) is consistent with the import graph: out-of-bucket imports from training are limited to `model_registry` (showcase/inference) and the FFI surface from shogi_gym. Verified by the grep results gathered.
- The catalog's count of "Total entries: 20" matches the bucket headers I read end-to-end.

## Validator Confidence

**High.** I sample-verified ~12 specific claims via direct grep/Read against the source, including all "easy to falsify" ones called out in the brief. The two off-by-N numerical errors and one stale heading are the only material discrepancies; the architectural and ownership claims hold up.

**Risks of letting these pass into synthesis:**
- The "11 in-bucket modules" number, if reused in the dependency-graph deliverable, would draw a slightly thinner edge set than reality. Easy to correct now; harder later.
- The "8 localStorage keys" number, if quoted in any metrics/refactor proposal (e.g. the `createPersistedStore` ADR), would understate the consolidation surface by 1.

**Information Gaps:**
- I did not re-verify the closed/open status of the cited filigree IDs against current state (relied on the catalog's `keisei-*` IDs being correct).
- I did not exhaustively re-read the four parallel cataloguing pass outputs in `temp/`; I trusted the merge.
- I did not verify line numbers within bodies of long files (e.g. `tournament.py:352` for `_record_match_result`) — only entry-point and constant claims.

## Caveats

This validation is structural and sample-evidential, not technical-correctness. I did not assess whether the architectural concerns flagged in each entry are *the right* concerns, only that they are evidence-grounded and consistent with the code. Technical-accuracy review (e.g. is the `_record_match_result` multi-txn pattern actually broken in the way `keisei-fa604bad63` claims?) would require domain expertise and is out of scope per the SME-agent protocol.
