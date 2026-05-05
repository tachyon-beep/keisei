# Coordination Plan — Keisei Architecture Analysis

## Analysis Configuration
- **Workspace:** `docs/arch-analysis-2026-05-05-1007/`
- **Scope:** Whole repository (Rust engine + Python training/server/SL/showcase + WebUI)
- **Deliverables:** Option A — Full Analysis
  - `01-discovery-findings.md`
  - `02-subsystem-catalog.md`
  - `03-diagrams.md`
  - `04-final-report.md`
- **User-prioritised focus areas (all four):**
  1. Rust engine (`shogi-core` + `shogi-gym`)
  2. Python training harness (`keisei/training`, `keisei/sl`)
  3. WebUI (`webui/`)
  4. Server / data layer (`keisei/server`, `keisei/showcase`, `keisei/db.py`, `keisei/config.py`)
- **Strategy:** Parallel subsystem analysis after holistic discovery — bounded by 4 user-selected areas with loose coupling at well-defined boundaries (FFI, SQLite, WebSocket).
- **Complexity estimate:** Medium-high. ~31K LOC across two languages plus Svelte UI; clear module boundaries but training subsystem has dense internal coupling (~30 internal modules).
- **Time constraint:** None declared.
- **Project state caveat (memory-derived):** Repo is in a post-reset rebuild after migrating from a pure-Python engine to Rust. Treat findings as a snapshot of an evolving codebase. The Rust engine plan is currently a standalone burn-in harness; Keisei integration is a future phase.

## Orchestration Strategy

**Decision:** Parallel for subsystem cataloguing, sequential for synthesis.

**Why parallel is safe here:**
- Boundaries are explicit and narrow:
  - **FFI boundary:** PyO3 `_native` module exposes 9 Python classes; Python consumers import `shogi_gym.*` only.
  - **DB boundary:** SQLite (`keisei.db`) is the integration substrate. Training writes; server/showcase read; webui never touches it directly.
  - **Network boundary:** Server WebSocket protocol with 9 enumerated message types (`init`, `game_update`, `metrics_update`, `training_status`, `league_update`, `showcase_update`, `showcase_status`, `showcase_error`, `ping`).
- The four focus buckets touch mostly disjoint files (~80%+ no overlap).
- Foundations (`config.py`, `db.py`) are read-only inputs to all other subsystems and can be characterised once in the server/data bucket and referenced.

**Why not fully parallel inside training:** ~30 modules with high internal coupling (opponent_store, role_elo, tournament*, gauntlet, dynamic_trainer all interlinked). Cataloguing within the bucket is sequential by one agent.

## Execution Log
- **2026-05-05 10:07** Workspace created at `docs/arch-analysis-2026-05-05-1007/`.
- **2026-05-05 10:07** User selected Option A (Full Analysis) and all four subsystem focus areas via AskUserQuestion.
- **2026-05-05 10:08** Holistic discovery scan: directory tree, manifests (Cargo workspace, pyproject, package.json), LOC counts per crate/package, internal coupling sample, FFI consumer map, WS message taxonomy.
- **2026-05-05 10:09** Discovery findings written.
- **2026-05-05 10:10** Subsystem catalog: dispatched 4 parallel analysis agents (one per focus bucket).
- **2026-05-05 10:14** Rust agent returned: 2 entries (shogi-core, shogi-gym), ~1640 words. Surfaced 3 unsafe blocks in `vec_env.rs` carrying parallel-step soundness; documented `unsafe impl Send/Sync for SendPtr<T>` and 4-site coupling discipline for adding new modes.
- **2026-05-05 10:17** Server/data agent returned: 4 entries (E1 Config, E2 SQLite, F Server, G Showcase), 21 tables enumerated, v1→v8 migrations summarised, **3 additional WS server→client message types found beyond the discovery doc's 9 (`showcase_match_queued`, `showcase_speed_changed`, `showcase_match_cancelled`)**, plus 4 client→server messages.
- **2026-05-05 10:17** WebUI agent returned: 7 entries (H1–H7), 29 components mapped, full WS→store→view table at top of doc, no open P1/P2 webui bugs.
- **2026-05-05 10:18** Training agent returned: 7 entries (C1a/C1b/C2/C3/C4/C5/D), ~3,600 words. Justified split of C1 into PPO core (`katago_ppo`+`gae`+`value_adapter`) vs loop+orchestration. Flagged dense coupling: `tournament.py` imports 9 league modules, `OpponentStore` imported by 8 modules. Confirmed P1 bug `keisei-fa604bad63` (non-atomic `_record_match_result`) by direct code reading.
- **2026-05-05 10:19** Catalog merged into `02-subsystem-catalog.md` (1010 lines, 20 entries).
- **2026-05-05 10:19** Validation gate dispatched (mandatory for ≥3 subsystems).
- **2026-05-05 10:22** Validation returned: PASS-WITH-FIXES. Three numeric fixes applied inline (line 394 11→14, line 890 eight→nine, line 989 8×→9×, line 595 8→12 server messages). Boundary ownership and template adherence: clean. 20/20 entries.
- **2026-05-05 10:22** Diagrams + final report dispatched in parallel.
- **2026-05-05 10:27** Diagrams returned: 7 Mermaid diagrams (L1 system context, L2 container, L3 training, L3 webui, training-rollout sequence, showcase sequence, DB-as-message-bus dataflow). 537 lines.
- **2026-05-05 10:27** Final report returned: 2,838 words. Top 3 risks elevated: non-atomic `_record_match_result`, dual match-recording paths, concurrent_matches slot-reuse cluster.
- **2026-05-05 10:28** Phase 1 (archaeology) complete. Deliverables 00–04 in place.
- **2026-05-05 10:30** Phase 2 (architect handover) initiated. Dispatched `architecture-critic` and `debt-cataloger` from `axiom-system-architect` pack in parallel against the analysis workspace.
- **2026-05-05 10:35** `architecture-critic` returned: `05-architecture-critique.md`. Verdict — structurally sound with one well-localised abscess (C3 league); top theme "Convention-not-mechanism"; recommends consolidating league to sidecar path, fixing `_record_match_result` atomicity first, splitting `db.py` before `katago_loop.py`.
- **2026-05-05 10:35** `debt-cataloger` returned: `06-debt-register.md`. 21 items total: **2 Critical / 6 High / 9 Medium / 4 Low**. Top 5 by severity-then-effort: DEBT-001 (atomicity), DEBT-002 (dual league paths), DEBT-004 (gauntlet tuple-unpack), DEBT-005 (FrontierManager retire-then-abort), DEBT-008 (flush_timings). Two severity disagreements with final-report ranking surfaced and called out in-line.
- **2026-05-05 10:35** Architect handover complete.
