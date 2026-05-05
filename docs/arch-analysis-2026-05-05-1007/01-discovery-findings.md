# Discovery Findings — Keisei

## 1. Project Identity

**Keisei** is a deep-RL training system for Shogi (Japanese chess), built around a Rust core (`shogi-engine`) with PyO3 bindings (`shogi-gym`) consumed by a Python training harness, plus a FastAPI/Svelte spectator dashboard. Currently in a post-reset rebuild phase (see `CHANGELOG.md`), with the engine recently migrated from pure Python to Rust.

- **Languages:** Rust (engine), Python ≥3.12 (training/server/SL), JavaScript (Svelte 4 UI).
- **Build / dep mgmt:** `uv` (Python), Cargo workspace (Rust), maturin (Rust→Python wheel), npm + Vite (UI).
- **Repo reset date:** 2026-04-01 (per project memory).

## 2. Directory Layout (top-level, code only)

```
keisei/                         # Python training harness (root package)
├── __init__.py                 # version 0.2.0
├── config.py                   # 17 dataclass-style config classes
├── db.py                       # SQLite layer with v8 schema + migrations
├── server/app.py               # FastAPI dashboard + WebSocket
├── showcase/                   # Showcase match runner (5 modules)
├── sl/                         # Supervised-learning data prep + trainer
└── training/                   # PPO/KataGo trainer + league system (~30 modules)
    └── models/                 # 6 model architectures (mlp, resnet, se_resnet, transformer, katago_base, base)

shogi-engine/                   # Cargo workspace
├── Cargo.toml                  # workspace = [shogi-core, shogi-gym]
└── crates/
    ├── shogi-core/             # Pure Rust engine (no deps beyond std)
    │   └── src/                # 11 modules (~8.4K LoC)
    └── shogi-gym/              # PyO3 cdylib + rlib (~6K LoC)
        └── src/                # 9 modules, exposes 9 PyClass

webui/                          # Svelte 4 + Vite SPA
├── package.json                # uplot dep; svelte ^4, vite ^5, vitest ^4
├── public/
└── src/
    ├── App.svelte
    ├── lib/                    # 29 .svelte components + helpers
    └── stores/                 # 10 stores (games, training, league, showcase, metrics, theme, audio, navigation, notation, aboutLevel)

tests/                          # Python tests (unit + integration)
configs/                        # Top-level *.toml training configs
checkpoints/, logs/, data/      # Runtime artefacts
keisei.db                       # SQLite DB (training state, snapshots, metrics, league)
```

## 3. Entry Points

| Entry point | Source | Role |
|---|---|---|
| `keisei-train` | `keisei.training.katago_loop:main` | Main training loop (PPO + league) |
| `keisei-serve` | `keisei.server.app:main` | FastAPI spectator dashboard |
| `keisei-prepare-sl` | `keisei.sl.prepare:main` | SL data preparation |
| `keisei-evaluate` | `keisei.training.evaluate:main` | Standalone model evaluation CLI |
| `python -m keisei.showcase` | `keisei/showcase/__main__.py` | Showcase match runner |
| `npm run dev` (in `webui/`) | `vite` | UI dev server |
| `cargo test -p shogi-core` | — | Pure Rust tests |
| `maturin develop` (in `shogi-gym`) | — | Build PyO3 extension into local venv |

`run.sh` orchestrates common tasks (15.8K bytes — substantial).

## 4. Codebase Sizing

| Area | Files | LOC (approx) |
|---|---|---|
| `shogi-core/` (Rust) | 11 src + 1 bench | ~8,400 |
| `shogi-gym/` (Rust + PyO3) | 9 src | ~6,000 |
| `keisei/training/` + models | 33 .py | ~11,500 |
| `keisei/{config,db,__init__}.py` | 3 | ~1,920 |
| `keisei/server/` | 1 | (large; not yet counted in detail) |
| `keisei/showcase/` | 5 | ~790 |
| `keisei/sl/` | 4 | ~2,000 (estimated from total 16,283 minus accounted) |
| `webui/src/lib/` (.svelte) | 29 | (counted only; LOC not aggregated yet) |
| `webui/src/stores/` (.js, prod) | 10 | — |
| **Total identified** | — | **~31,000 LOC across Rust + Python (UI not measured)** |

## 5. Technology Stack

- **Rust:** edition 2024, `pyo3 = "0.23"`, `numpy = "0.23"`, `rayon = "1.10"`. Core has zero external deps.
- **Python deps (runtime):** `torch`, `numpy`, `fastapi`, `uvicorn[standard]`. Optional `chardet` for SL.
- **Python tooling:** ruff (line-length 140, E/F/I/W), mypy near-strict (extends-strict minus untyped-call; excludes `shogi-engine/`), pytest with `asyncio_mode=auto`, pytest-xdist, pytest-asyncio, httpx.
- **Distributed training:** `torch.distributed` + DDP, with NCCL/Gloo support inferred (see `training/distributed.py`).
- **UI deps:** `svelte ^4`, `vite ^5`, `vitest ^4`, `jsdom ^29`, `uplot ^1.6` (single chart lib).
- **Persistence:** SQLite with WAL mode (visible in repo: `keisei.db`, `keisei.db-shm`, `keisei.db-wal`); migration schema currently at v8.

## 6. Subsystem Identification (8 subsystems, grouped into 4 user-selected buckets)

| # | Subsystem | Bucket | Purpose |
|---|---|---|---|
| **A** | `shogi-core` | Rust engine | Pure-Rust Shogi rules, position, move-gen, zobrist, SFEN |
| **B** | `shogi-gym` | Rust engine | PyO3 RL environment: VecEnv, observations (default + KataGo), action mappers (1D + spatial), spectator env, step results |
| **C** | `keisei.training` | Python training | PPO/KataGo trainer, league/tournament infrastructure, model architectures, Elo, opponent store, gauntlet, frontier promotion, style profiling |
| **D** | `keisei.sl` | Python training | Supervised-learning data prep + trainer (preparing pretraining datasets) |
| **E** | `keisei.config` + `keisei.db` | Server/data | Foundational shared layer — 17 typed config classes; SQLite schema with 8 migrations |
| **F** | `keisei.server` | Server/data | FastAPI app + WebSocket dashboard, with allow-listed hosts and per-connection write locks |
| **G** | `keisei.showcase` | Server/data | Headless showcase match runner; queues matches, writes results to DB for the UI to consume |
| **H** | `webui` | WebUI | Svelte 4 SPA — board, eval bar, league table, showcase scrubber, metrics charts, commentary panel |

## 7. Cross-Subsystem Boundaries (the integration surfaces)

### 7.1 FFI boundary (Python ↔ Rust)
- Module: `shogi_gym._native` (`shogi-gym` PyO3 cdylib).
- Exported PyClasses: `DefaultActionMapper`, `KataGoObservationGenerator`, `SpatialActionMapper`, `DefaultObservationGenerator`, `VecEnv`, `SpectatorEnv`, `StepResult`, `ResetResult`, `StepMetadata`.
- Python consumers (grep `shogi_gym`): `training/demonstrator.py`, `training/evaluate.py`, `training/tournament_runner.py`, `training/historical_gauntlet.py`, `training/katago_loop.py`, `training/tournament.py`, `showcase/runner.py`. **(7 files)**
- Tested through Python (`maturin develop` + pytest), not via `cargo test -p shogi-gym` (linker fails without Python symbols — documented in `CLAUDE.md`).

### 7.2 SQLite boundary (training ↔ server/showcase ↔ ?)
- Single DB file: `keisei.db` with `keisei.db-shm` / `keisei.db-wal` (WAL mode).
- Schema layers (from `db.py` function names): metrics, epoch summaries, game snapshots (regular + showcase), training state, heartbeats, league + role-Elo, tournament stats, head-to-head, game features (per-game), style profiles. Migrations v1→v8 chained.
- Showcase has its own ops module (`showcase/db_ops.py`) for queue + heartbeat + per-move state.
- Concurrency mode: WAL + single-writer-by-convention; not yet inspected for explicit locking strategies.

### 7.3 WebSocket boundary (server ↔ webui)
Message types observed in `webui/src/lib/ws.js`:
- `init` — initial snapshot
- `game_update` — current learner game state
- `metrics_update` — training metrics (uplot-compatible)
- `training_status` — heartbeat / progress
- `league_update` — league standings/Elo
- `showcase_update`, `showcase_status`, `showcase_error` — showcase channel
- `ping` — keepalive
Server-side: per-connection `asyncio.Lock` to serialise sends (see commit `f608594`); allow-listed Host header enforcement; 5 s send timeout, 15 s ping interval.

### 7.4 HTTP / static boundary
- `keisei/server/static/` — built UI is served by FastAPI (StaticFiles).
- JSON endpoints exist (`PlainTextResponse`, `JSONResponse` imports + `Request` usage); REST surface to be enumerated in subsystem entry.

### 7.5 Filesystem boundaries
- `checkpoints/` (per-config dirs): PyTorch checkpoint files.
- `logs/`: many log files (training/run logs).
- `configs/` + top-level `keisei-*.toml`: training run profiles.

## 8. Key Cross-Cutting Patterns Observed

1. **Configuration is centralised and typed.** `keisei/config.py` exposes 17 dataclass-style config classes assembling into `AppConfig`, loaded from TOML via `load_config(Path)`. No scattered globals.
2. **DB is the message bus.** Training writes; server reads. Showcase has its own writer. Webui consumes via WS only — never hits SQLite. This is a deliberate boundary, not coincidence (server's imports from `db.py` and `showcase/db_ops.py` are extensive read-side).
3. **The Rust core is conservative.** Pure std, zero deps, single workspace. PyO3 lives only in the gym crate. This isolates UB risk and keeps `cargo test -p shogi-core` fast.
4. **Heavy use of async + Python typing strictness.** `pyproject.toml` enables most strict-mypy options except `disallow_untyped_call` (PyTorch stub limitation). `asyncio_mode=auto` for tests.
5. **Distributed training is first-class.** `keisei/training/distributed.py` plus DDP imports in the loop suggest single-machine multi-GPU is supported, but multi-node setup needs investigation.
6. **Supervised-learning is a sibling track.** `keisei/sl/` has its own dataset/parser/trainer (probably PSN/KIF parsing → tensor pipelines → SL pretraining), distinct from the RL loop.
7. **League/tournament is the largest sub-area.** Roughly half of `keisei/training/` is league-related (tournament, gauntlet, frontier promoter, opponent store, role Elo, tier managers, priority scorer, match scheduler, concurrent matches, dispatcher, queue, runner, demonstrator). This indicates a substantial multi-agent training programme.

## 9. Identified Risks / Smells (preliminary, not yet evidenced in catalog)

These are observations to validate in the subsystem catalog, not conclusions:

- **`keisei/db.py` is 1158 LOC** — single-file DB layer is approaching a maintenance threshold; migration logic + read/write functions for ≥ 12 entity families in one module.
- **`keisei/training/` has ~30 sibling modules at the same level** — flat structure may obscure intended layering (core trainer vs league vs scheduling vs models). The `models/` subfolder is the only sub-grouping.
- **Open filigree bug list** is rich with concurrency and silent-failure issues in `concurrent_matches.py`, `tournament.py`, `historical_gauntlet.py`, `opponent_store.py`. Several of these (`P1`/`P2`) hint at race-conditions and partial-failure-mode bugs in the league system. Worth flagging as a hot-spot.
- **WebUI uses Svelte 4 with a P4 task to migrate to Svelte 5** (per session-context critical path) — UI reactivity will need rework once that lands.
- **CUDA event-handling bug open against `KataGoPPOAlgorithm`** (P2 in-progress: `flush_timings` never called) — implies the timing/profiling instrumentation in the PPO algorithm is leaky.

## 10. Confidence

**Confidence: High** for structural facts (file layout, manifests, entry points, FFI surface, WS message types, LOC).
**Confidence: Medium** for cross-subsystem semantics (e.g. exact concurrency contract on the SQLite DB, exact REST surface) — these are read at greater depth in the per-subsystem catalog.
**Confidence: Low** for runtime behaviour, performance, and correctness — to be evidenced in subsystem deep dives or deferred to architect handover.
