# Subsystem Catalog — Keisei

Merged from four parallel cataloguing passes (Rust engine, Python training, server/data, WebUI). Each parallel pass owned its bucket; cross-bucket references point to the owning entry rather than re-describing the surface.

**Total entries:** 20
- Rust engine: 2 (shogi-core, shogi-gym)
- Python training: 7 (C1a/C1b/C2/C3/C4/C5/D)
- Server/data: 4 (E1, E2, F, G)
- WebUI: 7 (H1–H7)

**Boundary ownership applied during cataloguing:**
- FFI exported surface owned by Rust agent; FFI consumers owned by training agent.
- SQLite schema owned by server/data agent (E2); writers reference but do not redocument.
- WS protocol message taxonomy owned by server/data agent (F); UI consumers owned by webui agent (H1).
- Config classes catalogued in E1; consumers list usage by class name only.

---

# Bucket A/B — Rust Engine

## shogi-core

**Location:** `shogi-engine/crates/shogi-core/`

**Responsibility:** Pure-Rust Shogi rules engine — board state, move generation, attack maps, special-rule enforcement (uchi-fu-zume, sennichite, impasse), Zobrist hashing, and SFEN serialization. Zero external runtime dependencies (`std` only); criterion only as `[dev-dependencies]`.

**Key Components:**
- `lib.rs` (16 LoC) — declares the 11 public modules and re-exports `Move`, `Piece`, `Position`, `GameState`, `MoveList`.
- `types.rs` (619 LoC) — `Color`, `PieceType` (1–8 niche), `HandPieceType` (1–7), `Square` (`new_unchecked` debug-only check, `flip()`, `offset()`), `Move` (`Board`/`Drop` enum), `GameResult` (with `is_terminal`/`is_truncation`), `ShogiError`.
- `piece.rs` (255 LoC) — `Piece(NonZeroU8)` with [promoted|color|piece_type] bit layout; `Option<Piece>` is one byte (verified by test `test_option_piece_is_one_byte`).
- `position.rs` (593 LoC) — `Position { board: [u8; 81], hands: [[u8; 7]; 2], current_player, hash: u64 }` plus `startpos()`, `compute_hash()`, `set_piece()`, `find_king()`.
- `zobrist.rs` (409 LoC) — deterministic xoshiro256** seeded `0xDEAD_BEEF_CAFE_BABE` (`zobrist.rs:78`); table indexed `[square][piece.to_u8()]`, `[color][hpt.index()][count]`, plus `side_to_move` word.
- `attack.rs` (1091 LoC) — `compute_attack_map(&Position) -> [[u8; 81]; 2]`, ground-truth ray-casting oracle with `would_wrap_file` guard (`attack.rs:42`); 8 direction constants defined at top.
- `movegen.rs` (1245 LoC) — pseudo-legal generator (`generate_pseudo_legal_board_moves`, `generate_pseudo_legal_drops`); promotion-zone helpers `must_promote`, `is_dead_drop`, `in_promotion_zone`.
- `rules.rs` (1947 LoC) — uchi-fu-zume detection via temporary mutation+rollback (`rules.rs:19`), sennichite (repetition / perpetual check), impasse (24-point), plus `material_balance()` consumed by `shogi-gym`.
- `game.rs` (2247 LoC) — `GameState` (position + attack_map + pawn_columns + repetition_map + ply/max_ply + result); `make_move`/`unmake_move`/`UndoInfo`, `legal_moves()`, `generate_legal_moves_into(&mut MoveList)`, `check_termination()`. Default max_ply = 500.
- `movelist.rs` (219 LoC) — fixed-capacity (`MOVELIST_CAPACITY = 1024`) stack-allocated buffer using `[MaybeUninit<Move>; 1024]` for zero-allocation move generation in the hot path.
- `sfen.rs` (736 LoC) — SFEN parse/serialize with `STARTPOS_SFEN` constant.
- `benches/movegen.rs` (53 LoC) — criterion benches: `legal_moves_opening`, `legal_moves_opening_hot_path`, `make_unmake_cycle`, `attack_map_from_scratch`.

**Dependencies:**
- Inbound: `shogi-gym` crate (path dep, all 9 modules import from `shogi_core`).
- Outbound: Rust `std` only. `criterion 0.5` as `[dev-dependencies]` for benches.

**Patterns Observed:**
- Niche-optimised piece encoding via `NonZeroU8` (`piece.rs:12`) so `Option<Piece>` is 1 byte.
- Stack-allocated zero-alloc movelist (`movelist.rs:21`) with `MaybeUninit` and `as_slice()` returning a raw `from_raw_parts` slice.
- Make/unmake with explicit `UndoInfo` carrying `prev_hash`, `prev_attack_map`, `was_in_check` (`game.rs:19`).
- Ground-truth oracle pattern: `attack.rs::compute_attack_map` is the regression baseline that incremental updates elsewhere are validated against (per `attack.rs:1` doc).
- Deterministic Zobrist with hard-coded seed → reproducible hashes across runs.
- Edition 2024; `if let Some(...) && ...` chained let-else syntax used (`game.rs:35`).

**Concerns:**
- `MoveList::new` calls `unsafe { MaybeUninit::uninit().assume_init() }` on `[MaybeUninit<Move>; 1024]` (`movelist.rs:32`). This is the documented idiom and sound (the wrapper is `MaybeUninit`, not `Move`), but rustc's `uninitialized_array` lint nominally prefers `MaybeUninit::<[_; N]>::uninit().assume_init()` for clarity — minor.
- `Position::set_piece` and direct mutation of `position.board` / `position.hash` are exposed as `pub` fields rather than encapsulated (`position.rs:22`). `rules.rs:33` and `vec_env.rs` simulation paths rely on this; mutation contract is doc-only.
- `attack.rs` (1091 LoC) and `rules.rs` (1947 LoC) are the two largest modules; I sampled headers + key functions but did not read every branch. Claims about behaviour at this depth are Medium confidence.
- No open P1/P2 filigree bugs match these files (verified via `filigree list --label=bug`).

**Confidence:**
- High: file inventory, LoC, public API of `lib.rs`/`types.rs`/`piece.rs`/`movelist.rs` (read 100%), Cargo manifest claims (zero deps, criterion bench harness), bench list, public field layout of `Position`/`GameState`.
- Medium: behavioural claims about `attack.rs` (read first 80 LoC of 1091), `rules.rs` (first 80 LoC of 1947), `game.rs` (first 100 LoC of 2247), and `sfen.rs` (first 60 LoC of 736). Inferred from documented headers, function signatures, and call sites in `shogi-gym`.

---

## shogi-gym

**Location:** `shogi-engine/crates/shogi-gym/`

**Responsibility:** PyO3 cdylib that wraps `shogi-core` as a vectorized RL environment for Python — exposes batched `VecEnv`, single-game `SpectatorEnv`, two observation generators, two action mappers, and three result/metadata classes through the `_native` extension module.

**Key Components:**
- `lib.rs` (25 LoC) — `#[pymodule] _native` registers the 9 PyClasses.
- `vec_env.rs` (1996 LoC) — `VecEnv` PyClass; pre-allocated flat buffers (`obs_buffer`, `legal_mask_buffer`, `reward_buffer`, `terminated_buffer`, `truncated_buffer`, `captured_buffer`, `term_reason_buffer`, `ply_buffer`, `material_balance_buffer`, `terminal_obs_buffer`, `current_players_buffer`); two-phase `step()` (decode+validate under GIL → apply under `py.allow_threads`); rayon `par_iter` above `PARALLEL_THRESHOLD = 64` envs (`vec_env.rs:51`); per-env `catch_unwind` panic isolation (`vec_env.rs:462`).
- `observation.rs` (757 LoC) — `DefaultObservationGenerator` (ZST), 46-channel layout (`NUM_CHANNELS=46`, `BUFFER_LEN=46*81`); shared `generate_base_channels` (channels 0-43) used by KataGo too.
- `katago_observation.rs` (669 LoC) — `KataGoObservationGenerator` (ZST), 50 channels (44 shared + 4 repetition + 1 check + 1 reserved).
- `action_mapper.rs` (549 LoC) — `ActionMapper` trait (`Send + Sync`), `DefaultActionMapper` (ZST); `ACTION_SPACE_SIZE = 13_527` (12,960 board + 567 drop); perspective flip via `Square::flip()` for White.
- `spatial_action_mapper.rs` (726 LoC) — `SpatialActionMapper` (ZST); `SPATIAL_ACTION_SPACE_SIZE = 11_259` (81 squares × 139 move types: 64 slide + 64 slide-promote + 4 knight + 7 drop).
- `spectator.rs` (428 LoC) — `SpectatorEnv` PyClass; single-game, no auto-reset, returns rich `PyDict` (Streamlit/dashboard) with `from_sfen` constructor, `to_dict`, `legal_actions`.
- `spectator_data.rs` (729 LoC) — Hodges + USI move notation helpers (`move_notation`, `move_usi`, `square_notation`, `build_spectator_dict`), shared by `vec_env` and `spectator`.
- `step_result.rs` (150 LoC) — `StepResult { observations: PyArray4<f32>, legal_masks: PyArray2<bool>, rewards: PyArray1<f32>, terminated/truncated: PyArray1<bool>, terminal_observations: PyArray4<f32>, current_players: PyArray1<u8>, step_metadata: Py<StepMetadata> }`; `StepMetadata { captured_piece: u8, termination_reason: u8, ply_count: u16, material_balance: i32 }`; `ResetResult { observations, legal_masks }`; `TerminationReason` enum (0–5).

**FFI Exported Surface (the 9 PyClasses):**
- `VecEnv(num_envs=512, max_ply=500, observation_mode="default"|"katago", action_mode="default"|"spatial")` → `.reset() -> ResetResult`, `.step(actions: list[int]) -> StepResult`. Output arrays shape: obs `(N, C, 9, 9)`, mask `(N, A)`, scalars `(N,)`. Decode raises `ValueError` on negative actions, `RuntimeError` on illegal action.
- `SpectatorEnv(max_ply=500, action_mode="default")` plus static `from_sfen(sfen, max_ply=None, action_mode="default")`.
- `DefaultActionMapper`, `SpatialActionMapper`, `DefaultObservationGenerator`, `KataGoObservationGenerator` — exposed as ZST PyClasses (likely for direct Python-side encode/decode introspection).
- `StepResult`, `ResetResult`, `StepMetadata` — read-only via `#[pyo3(get)]` on every field.

**Dependencies:**
- Inbound: 7 Python files in `keisei/` per `01-discovery-findings.md` §7.1 (training agent owns that side).
- Outbound: `shogi-core` (path), `pyo3 = "0.23"`, `numpy = "0.23"`, `rayon = "1.10"`. `crate-type = ["cdylib", "rlib"]`; `extension-module` is a feature gate.

**Patterns Observed:**
- ZST-enforced obs/action generators: 4 `const _: () = assert!(size_of::<...>() == 0)` checks at `vec_env.rs:45–48` block stateful generators at compile time.
- Tag-enum dispatch inside rayon closure (`ObsModeTag`/`ActionModeTag`, `vec_env.rs:200`) so that `Copy + Send` is preserved across thread boundaries; `From<&Mode> for Tag` impls force exhaustive matches (`vec_env.rs:208`) to keep tags in sync at compile time.
- Two-phase step contract: validate under GIL, then `py.allow_threads(|| self.apply_moves(...))` (`vec_env.rs:705`) — Python threads can run during the heavy phase.
- `SendPtr<T>` wrapper (`vec_env.rs:62`) carrying `len` for debug bounds-checks; manually `unsafe impl Send/Sync`. Each rayon worker writes a disjoint index into pre-allocated flat buffers (no allocation per step, no aliasing).
- Per-env `catch_unwind` (`vec_env.rs:462`) with auto-reset to startpos and sentinel buffer values (`terminated=true`, `reward=0.0`); panic count returned, batch continues. Closed bug `keisei-cdf80418a1` confirms this was added 2026-04-03.
- Output buffers materialized as numpy via `.to_pyarray(py).reshape(...)` after `allow_threads` block (`vec_env.rs:718`).
- `PARALLEL_THRESHOLD = 64`: small N falls back to sequential `for_each`.

**Concerns:**
- Three `unsafe` regions in `vec_env.rs`: the `unsafe impl Send/Sync for SendPtr<T>` (`vec_env.rs:66-67`), and two large `unsafe` blocks (`:348-458` happy path, `:480-539` panic-recovery path) doing pointer offset writes plus `slice::from_raw_parts_mut`. Soundness depends on the disjoint-index invariant; this is documented in comments but not enforced beyond `debug_assert`. Low-priority filigree task `keisei-1883589523` lists related stylistic items.
- The two `match obs_tag/act_tag` blocks inside the rayon closure (lines ~409, 427, 442, 499, 521) are duplicated four times across happy-path and panic-recovery; any new mode requires touching all four sites (the doc comment at `vec_env.rs:128–137` lists this 4-place coupling).
- `vec_env.rs:285` and `:445` use `.expect("legal move must be encodable")` inside the rayon closure on `mapper.encode()`; an encoder bug would panic, but `catch_unwind` now contains it.
- Move-history bookkeeping is single-threaded: the `move_histories` push at `vec_env.rs:696-701` runs before `allow_threads`, holding GIL — fine for correctness, but proportional to `N`.
- `shogi-gym` cannot be tested via `cargo test -p shogi-gym` — PyO3 cdylib needs Python symbols at link time (per `CLAUDE.md`); Rust `#[cfg(test)]` blocks (e.g. `step_result.rs:99–149`, `vec_env.rs:33–39` test-only mutexes) are compiled and run via `maturin develop` + pytest in the gym's own `.venv`.
- No open P1/P2 filigree bugs match these files (verified). One closed P2 (`keisei-cdf80418a1`) for the catch_unwind gap; one closed P2 (`keisei-61c392ec93`) for per-env generator reconstruction (resolved by ZST static-asserts); one open P4 stylistic bucket (`keisei-1883589523`).

**Confidence:**
- High: PyClass surface, output array shapes/dtypes, action-space sizes, observation channel counts, Cargo dependency declarations, rayon threshold, GIL release point, `unsafe` block locations, `SendPtr` design, `catch_unwind` recovery semantics, ZST asserts, two-phase step contract — all read directly.
- Medium: full body of `apply_moves` happy/recovery paths (read but ~250 LoC of branching, sampled rather than exhaustively traced), all of `katago_observation.rs` past line 60, `spatial_action_mapper.rs` past line 60, `spectator.rs` past line 120, and `spectator_data.rs` past line 60. Inferred behaviour from headers and the corresponding constants/types.
- Low: detailed correctness of `move_notation` disambiguation logic in `spectator_data.rs` — not read.


---

# Bucket C/D — Python Training & SL

# Subsystem Catalog — Python Training Bucket (`keisei/training/` + `keisei/sl/`)

This document covers the Python training harness only. Bucket scope: `keisei/training/` (~30 modules, ~11.5K LoC) and `keisei/sl/` (4 modules, ~1.07K LoC). Server, db, config, and showcase are documented elsewhere. The Rust FFI surface (9 PyClasses in `shogi_gym._native`) is documented in the engine bucket; this file only catalogs the **consumer side**.

Grouping diverges slightly from the suggested layout: I split the **PPO core** (`katago_ppo.py` + `gae.py` + `value_adapter.py`) from the **loop/orchestrator** (`katago_loop.py` + `checkpoint.py` + `distributed.py` + `transition.py` + registries) because the PPO module is a self-contained library used by both the loop and `dynamic_trainer.py`. That gives 7 entries: C1a (PPO core), C1b (Loop & orchestration), C2 (Models), C3 (League), C4 (Style/features), C5 (Evaluation CLI), D (SL).

---

## C1a. KataGo PPO Core (algorithm + GAE + value adapter)

**Location:** `keisei/training/{katago_ppo.py, gae.py, value_adapter.py}`

**Responsibility:** Pure PPO/KataGo algorithm primitives — multi-head loss functions, rollout buffer, PPO step, GAE advantage computation, and the scalar/multi-head value-loss adapter — independent of environment, league, or DB.

**Key Components:**
- `katago_ppo.py` (991 LoC) — `KataGoPPOParams` (frozen dataclass with `__post_init__` validation, l.102), `KataGoRolloutBuffer` (CPU-side step buffer with shape/cat/score-NaN guards at l.235–268), `KataGoPPOAlgorithm` (l.391; AMP-aware optimizer, scaler, entropy decay, warmup, optional `torch.compile`), and loss helpers `ppo_clip_loss` / `wdl_cross_entropy_loss`.
- `gae.py` (296 LoC) — `compute_gae` (1D/2D, supports per-step `next_value_override` for truncation/perspective bootstrap; wrapped in `torch.no_grad()` at l.45) and `compute_gae_gpu` consumed by `KataGoPPOAlgorithm`.
- `value_adapter.py` (144 LoC) — `ScalarValueAdapter` and `MultiHeadValueAdapter` implement a common `ValueHeadAdapter` interface so the loop never branches on model contract; `get_value_adapter(model_contract, ...)` factory at l.129.

**Dependencies:**
- Inbound: `keisei.training.katago_loop` (instantiates `KataGoPPOAlgorithm`/`KataGoRolloutBuffer`); `keisei.training.dynamic_trainer` (reuses `ppo_clip_loss`, `wdl_cross_entropy_loss`); `keisei.training.algorithm_registry` (registers `KataGoPPOParams` schema).
- Outbound: `keisei.sl.dataset.SCORE_NORMALIZATION` (single shared constant at `katago_ppo.py:14`); `keisei.training.models.katago_base.KataGoBaseModel` (type only); `torch`, `torch.nn.functional`, `torch.amp.{GradScaler,autocast}`. **No DB, no shogi_gym, no league imports.**
- Config consumed: none directly; params injected via `KataGoPPOParams` constructor.

**Patterns Observed:**
- Frozen dataclass + `__post_init__` validation (`katago_ppo.py:102–116`).
- Pre-CPU-detach for buffer writes to avoid CUDA syncs (`katago_ppo.py:225–232`).
- Adapter pattern keeps loop code contract-agnostic (`value_adapter.py:16,43,62`).
- Autograd-safe zero-loss sentinel (`value_logits.sum() * 0.0`) when no valid value targets (`katago_ppo.py:55`, `value_adapter.py:119`).

**Concerns:**
- `KataGoPPOParams.__post_init__` validates lower bounds but not upper bounds and does not reject NaN (open issue **keisei-cb7008ac73**: katago_ppo accepts NaN/negative hyperparameters; partially mitigated by checks at `katago_ppo.py:103–116` but still incomplete).
- `KataGoRolloutBuffer.add()` does not validate leading dims of `actions/log_probs/values/dones/...` against `obs_cpu.shape[0]` — open issue **keisei-b2324bf429** (broadcasting can silently corrupt rollouts).
- `KataGoRolloutBuffer._has_env_ids` / `_has_next_value_override` are sticky once set on first add and never cleared on `clear()` (`katago_ppo.py:189–193`); open issue **keisei-02c52bae2d** (stale env_ids across cycles).
- `value_adapter.py` `MultiHeadValueAdapter.compute_value_loss` returns `lambda_value * value_loss + lambda_score * score_loss` (l.126); when `lambda_value/lambda_score` are negative, gradient ascent occurs silently — open issue **keisei-bef32b64a8** (`0.0 * NaN` poisoning) and **keisei-678359b7aa** (negative lambdas at the SL config level, same pattern).
- `gae.py:compute_gae` silently broadcasts a scalar `next_value` across envs for 2D rollouts — open issue **keisei-d43009528f**.

**Confidence:** High for module boundaries, public API, and pattern claims (read entry sections). Medium for `KataGoPPOAlgorithm` internals (l.391–991 not fully read; only the constructor, loss helpers, and signatures verified).

---

## C1b. Training loop & orchestration

**Location:** `keisei/training/{katago_loop.py, checkpoint.py, distributed.py, transition.py, algorithm_registry.py, dynamic_trainer.py}`

**Responsibility:** The end-to-end RL training driver — wires PPO core to `shogi_gym.VecEnv`, manages DDP, checkpoint persistence, opponent rotation, league hookup, and the SL→RL transition. Owns the `keisei-train` entry point.

**Key Components:**
- `katago_loop.py` (1989 LoC) — `KataGoTrainingLoop` (l.454), `split_merge_step` (l.284, the per-step learner/opponent dispatch), `to_learner_perspective`/`sign_correct_bootstrap` (l.111/125, two-player reward sign correction), `PendingTransitions` (l.139), `main()` (l.1955; the `keisei-train` console script).
- `dynamic_trainer.py` (418 LoC) — `DynamicTrainer` (l.41) does small PPO updates on Dynamic-tier league entries from match rollouts; reuses `ppo_clip_loss`/`wdl_cross_entropy_loss` from C1a; has its own `_update_lock` (l.72) for serialisation; `MatchRollout` dataclass at l.26.
- `checkpoint.py` (179 LoC) — `save_checkpoint`/`load_checkpoint`; numpy-RNG-state-to-tensor encoder (`_numpy_rng_to_safe`, l.17) so `torch.load(weights_only=True)` works.
- `distributed.py` (159 LoC) — `DistributedContext` (frozen dataclass with eager device resolution at l.39) + setup/cleanup helpers; called once at process start.
- `transition.py` (180 LoC) — `sl_to_rl()` (l.31) bridges `SLTrainer` output into `KataGoTrainingLoop(resume_mode="sl")`; cross-validates SL and RL architecture/params before running SL to fail fast.
- `algorithm_registry.py` (40 LoC) — schema lookup `algorithm name -> Params dataclass` (only `katago_ppo` registered).

**Dependencies:**
- Inbound (in-bucket): `transition.py` instantiates the loop; nothing else imports it. Out-of-bucket: tests only.
- Outbound (in-bucket): C1a (PPO core), C2 (`model_registry.build_model`), C3 (entire league surface — `OpponentStore`, `OpponentEntry`, `Role`, `RoleEloTracker`, `TieredPool`, `MatchScheduler`, `ConcurrentMatchPool`, `PriorityScorer`, `LeagueTournament`, `TournamentDispatcher`, `HistoricalGauntlet`).
- FFI: `katago_loop.py:578` lazy-imports `shogi_gym.VecEnv` and instantiates with `observation_mode="katago"`, `action_mode="spatial"`; uses `vecenv.observation_channels` and `vecenv.action_space_size` (must be 11259); `split_merge_step` calls `vecenv.step()`/`vecenv.reset()` and consumes the StepResult/StepMetadata fields. `transition.py` passes `vecenv` through to the loop.
- `db.py`: `init_db`, `read_training_state`, `update_training_progress`, `write_epoch_summary`, `write_game_snapshots`, `write_training_state` (`katago_loop.py:25`); also `_connect` for ad-hoc UPDATE on `training_state.total_epochs` (`katago_loop.py:841` — direct SQL, not via a public helper). `transition.py` calls `init_db`, `write_training_state`.
- Config consumed: `AppConfig`, `TrainingConfig`, `ModelConfig`, `DisplayConfig`, plus `config.distributed.{sync_batchnorm, find_unused_parameters, gradient_as_bucket_view}` and the entire `config.league.*` tree.

**Patterns Observed:**
- Lazy FFI import inside the constructor so the module loads without `shogi_gym` (`katago_loop.py:577`); pattern repeated in `tournament.py:197`, `historical_gauntlet.py:94`, `tournament_runner.py:129`, `evaluate.py:101`, `demonstrator.py:187`.
- DDP weights re-loaded on every rank, not broadcast (`katago_loop.py:756–786`); rationale documented inline.
- Hard fail-fast architecture/observation channel checks (`katago_loop.py:480–498, 587–598`) — survive `python -O`.
- `split_merge_step` rolls per-env learner/opponent dispatch into a single VecEnv batch step.
- `DynamicTrainer` rate-limits checkpoint writes via update timestamps (`dynamic_trainer.py:95`); writes weights via `store.save_weights` (`dynamic_trainer.py:381`).

**Concerns:**
- `katago_loop.py` is **1989 LoC in one class** — the loop, opponent rotation, league bookkeeping, snapshotting, DB writes, and CLI are all in one file. Threshold-of-maintainability risk noted in discovery.
- Direct SQL UPDATE on `training_state.total_epochs` at `katago_loop.py:843–848` (caller catches plain `Exception` and falls through; bypasses the typed `db.py` helper layer).
- Open issue **keisei-dac195136d** (P1): checkpoint save failures still record the failed path in the DB — would land in `katago_loop.py`'s checkpoint path.
- Open issue **keisei-00d62cc231** (P2): `dynamic_trainer.py` has no finite-value gate before `store.save_weights` at l.381 — NaN/Inf weights can be persisted.
- Open issue **keisei-9a5ac20307** (P2): `keisei-evaluate` CLI ignores hyperparameters (lives in C5 but referenced from this loop's `_get_policy_flat` helper indirectly — see C5).
- `katago_ppo.py` flush_timings event-handling bug (referenced in discovery preliminary risks; CUDA timing instrumentation incomplete).
- `transition.py:80–86` cross-checks SL and RL params via `validate_model_params` but only when `rl_config_path` is provided.

**Confidence:** High for the entry/constructor (l.1–720 read directly) and module boundaries. Medium for inner methods of `KataGoTrainingLoop` (`run`, `train_epoch`, `_rotate_seat`, `_run_league_match`, etc., l.838–1950 not read). Medium for `dynamic_trainer.py` (entry + lock semantics + `update()` skeleton verified at l.247/381; mid-file logic skimmed).

---

## C2. Model architectures & registry

**Location:** `keisei/training/models/` + `keisei/training/model_registry.py`

**Responsibility:** Concrete neural network architectures and a typed registry mapping architecture name → (model class, params dataclass, value-head contract, observation channels). All architectures share a 50-channel 9×9 input and 11259-action output.

**Key Components:**
- `models/base.py` (27 LoC) — `BaseModel` ABC: scalar value contract, `(policy, value)` tuple output.
- `models/katago_base.py` (78 LoC) — `KataGoBaseModel` ABC with `KataGoOutput` (policy_logits, value_logits, score_lead); `configure_amp()` (l.52) freezes after `torch.compile()` to prevent silent recompile.
- `models/se_resnet.py` (159 LoC) — `SEResNetModel` + `GlobalPoolBiasBlock`; the only `multi_head` architecture currently registered. Default params: 40 blocks, 256 channels, SE-reduction 16.
- `models/resnet.py` (83 LoC) — `ResNetModel` (scalar contract).
- `models/mlp.py` (54 LoC) — `MLPModel` (scalar contract).
- `models/transformer.py` (95 LoC) — `TransformerModel` (scalar contract).
- `model_registry.py` (100 LoC) — `_REGISTRY` (l.24) maps architecture string to `ArchitectureSpec`; `validate_model_params` (l.43) does dataclass + semantic validation; `build_model` (l.86); `get_model_contract`, `get_obs_channels` accessors.

**Dependencies:**
- Inbound (in-bucket): `katago_loop.py`, `dynamic_trainer.py` (via `KataGoBaseModel`), `transition.py`, `evaluate.py`, `opponent_store.py:22` (rebuilds models from stored arch+params), `tournament.py` (indirectly via `opponent_store.load_opponent_cached`), `katago_ppo.py:16` (KataGoBaseModel type), `sl/trainer.py:15`.
- Inbound (out-of-bucket): `keisei.showcase.inference` imports `build_model`, `get_model_contract`, `get_obs_channels` (`showcase/inference.py:16`).
- Outbound: `torch.nn` only. **No DB, no shogi_gym, no config.**
- Config consumed: none directly; `params: dict` passed by callers.

**Patterns Observed:**
- Frozen-dataclass per-architecture params with `__post_init__` validation (e.g. `se_resnet.py:26–37`).
- Single source of truth for architecture → contract mapping (`model_registry.py:24`); the only place that decides which architectures are `multi_head` vs `scalar`.
- `KataGoBaseModel.configure_amp` raises if called after compile (`katago_base.py:59`) — a tripwire for AMP/compile interaction bugs.

**Concerns:**
- Only `se_resnet` is `multi_head`; `katago_loop.py:484` enforces `katago_ppo` requires `se_resnet`. The other three (resnet, mlp, transformer) are scalar-contract and have no registered algorithm — effectively dead with respect to the production training entry point until a scalar PPO algorithm is registered in `algorithm_registry.py`.
- `_REGISTRY` (`model_registry.py:24`) hard-codes 50 obs channels for all architectures. Discovery flagged the loop also hard-codes 50 (`katago_loop.py:588`) — duplicate magic number.
- None observed in models themselves (read 100% of `base.py`, `katago_base.py`; sampled `se_resnet.py` head; verified registry).

**Confidence:** High for ABCs and registry (read in full). Medium for the four concrete architectures' forward implementations (only `se_resnet.py` head sampled; `resnet/mlp/transformer` not read).

---

## C3. League / tournament infrastructure

**Location:** `keisei/training/{tournament*.py, opponent_store.py, tiered_pool.py, tier_managers.py, frontier_promoter.py, historical_gauntlet.py, historical_library.py, role_elo.py, match_scheduler.py, match_utils.py, priority_scorer.py, concurrent_matches.py, demonstrator.py}`

**Responsibility:** The multi-agent league: persistent opponent pool with three tiers (Frontier Static, Recent Fixed, Dynamic) plus a Historical Library, role-aware Elo, priority-scored pairing, concurrent match execution, in-process or sidecar-worker tournament drivers, and a separate exhibition demonstrator.

**Key Components:**
- `opponent_store.py` (1324 LoC) — `OpponentStore` (l.334; SQLite-backed pool with thread-local connections, in-memory model LRU, in-memory pin set, transaction context manager); `OpponentEntry` dataclass with composite + 4 role-specific Elo columns (l.240); `Role`/`EntryStatus`/`EloColumn` `StrEnum`s (l.27/42/48); `compute_elo_update` (l.308). Owns league filesystem under `<checkpoint_dir>/league`.
- `tiered_pool.py` (328 LoC) — `TieredPool` (l.25) high-level orchestrator that wires `FrontierManager` + `RecentFixedManager` + `DynamicManager` + `HistoricalLibrary` + `RoleEloTracker` + `MatchScheduler` + optional `HistoricalGauntlet` + optional `DynamicTrainer`. `snapshot_learner` (l.109) admits the learner's current weights into Recent Fixed.
- `tier_managers.py` (511 LoC) — `FrontierManager` (l.36; `review()` at l.89 promotes Dynamic→Frontier with one-retirement-per-call cap), `RecentFixedManager`, `DynamicManager`, `ReviewOutcome` enum.
- `frontier_promoter.py` (129 LoC) — `FrontierPromoter` (l.15) evaluates promotion candidates (5 criteria; in-memory streak tracking lost on restart by design, l.30).
- `historical_library.py` (251 LoC) — `HistoricalLibrary` (l.29) maintains 5 log-spaced milestone slots; two-pass assignment (l.65–80).
- `historical_gauntlet.py` (219 LoC) — `HistoricalGauntlet` (l.20) periodic learner-vs-history benchmark; runs synchronously on the tournament thread.
- `role_elo.py` (187 LoC) — `RoleEloTracker` (l.19) writes role-specific Elo deltas atomically (`update_from_result` l.31); `historical` context only updates side A.
- `match_scheduler.py` (463 LoC) — `MatchScheduler` (l.99); `MatchClass` enum (l.25); `classify_match` (l.63); `MATCH_CLASS_WEIGHTS` (l.47); `is_training_match` (l.81).
- `match_utils.py` (335 LoC) — `play_match` (l.49), `play_batch`, `_combine_rollouts` (l.35), `release_models`, `MatchOutcome` dataclass (l.21).
- `priority_scorer.py` (130 LoC) — `PriorityScorer` (l.12); 8-term weighted score (l.104) with sliding-window repeat tracking.
- `concurrent_matches.py` (625 LoC) — `ConcurrentMatchPool` running interleaved partitions of one shared `VecEnv`; `MatchResult`/`RoundStats`/`_MatchSlot` dataclasses (l.36/49/64).
- `tournament.py` (658 LoC) — `LeagueTournament` (l.53; background thread, `_record_match_result` at l.352, `_run_concurrent_round` at l.464, `_run_one_match` at l.574); `majority_wins_result` (l.40).
- `tournament_dispatcher.py` (142 LoC) — trainer-side `TournamentDispatcher` that enqueues round-robin pairings into `tournament_pairing_queue`. Owns single-writer recompute of `style_profiles` (l.113).
- `tournament_queue.py` (369 LoC) — pairing queue + worker heartbeat DB ops; uses raw `_connect` and atomic SQL.
- `tournament_runner.py` (421 LoC) — `python -m keisei.training.tournament_runner` sidecar worker (`main` at l.383); claims pairings via `claim_next_pairings_batch`, runs them via `ConcurrentMatchPool`, records Elo through its own path (no LeagueTournament instance).
- `demonstrator.py` (239 LoC) — `DemonstratorRunner` (l.45; daemon thread for inference-only exhibition matches; non-fatal crash policy at l.76).

**Dependencies:**
- Inbound (in-bucket): C1b (`katago_loop.py` builds the entire league stack); `style_profiler.py` (C4) is invoked by `tournament_dispatcher` and `tournament_runner` (C3 module that owns the dispatch); `transition.py` does NOT touch the league.
- Inbound (out-of-bucket): `tests/`. **Server/showcase do NOT directly import the league** — they consume via DB only.
- Outbound: C2 (`model_registry.build_model` from `opponent_store.py:22`); `db.py` (`_connect` from `tournament_dispatcher.py:16`, `tournament_queue.py:13`; `write_tournament_stats` and `write_game_features` lazy-imported in `tournament.py:326,336` and `tournament_runner.py:315`); plus `OpponentStore` itself acts as a typed wrapper over many DB tables (entry rows, league_results, elo_history, historical slots, role-elo updates, weights/optimizer paths).
- FFI: `tournament.py:198` → `VecEnv(observation_mode="katago", action_mode="spatial")`; `tournament_runner.py:129` → same; `historical_gauntlet.py:94` → same; `demonstrator.py:187` → same.
- Config consumed: `LeagueConfig`, `MatchSchedulerConfig`, `PriorityScorerConfig`, `RoleEloConfig`, `HistoricalLibraryConfig`, `GauntletConfig`, `FrontierStaticConfig`, `RecentFixedConfig`, `DynamicConfig`, `ConcurrencyConfig`.

**Patterns Observed:**
- Heavy use of `threading.RLock` on `OpponentStore` so manager methods can call `list_by_role` inside an open transaction (`tier_managers.py:102–104` documents the reentrant requirement).
- Two execution modes for tournaments: in-process thread (`tournament.py`) and sidecar worker (`tournament_runner.py` + `tournament_queue.py` + `tournament_dispatcher.py`); selected by `config.league.tournament_mode` (`katago_loop.py:672`).
- Lazy DB imports inside methods to keep import-time cycles small (`tournament.py:326,336`).
- Worker-claim queue uses `BEGIN IMMEDIATE` + conditional UPDATE for atomic claim (`tournament_queue.py` docstring l.3).
- Match classification table-driven: `MATCH_CLASS_WEIGHTS` shared between scheduler and scorer (`priority_scorer.py:8`).
- `_MatchSlot.reset_for_pairing` clears per-slot state but keeps env partition assignment fixed (`concurrent_matches.py:95`).

**Concerns (densest hotspot in the bucket):**
- `_record_match_result` (`tournament.py:352–460`) issues `store.record_result` (l.413) + two `store.update_elo` calls (l.430,431) + `role_elo_tracker.update_from_result` (l.433) as **separate transactions on `OpponentStore`**, not one. Open issue **keisei-fa604bad63** (P1): "splits one logical match across multiple txns — partial Elo state".
- `concurrent_matches.py` slot reuse / partition reset bugs:
  - **keisei-53eb4eb1f8** (P1): slot reuse without partition reset leaks games across pairings.
  - **keisei-4b6c36cd2b** (P1): partial-load cleanup `.cpu()` poisons shared LRU-cached models in `OpponentStore._model_cache` (`opponent_store.py:344`).
  - **keisei-bc58948f9f** (P2): initial-fill load failures abort entire batch.
  - **keisei-f2189813df** (P2): matches overshoot `games_per_match` when `envs_per_match > games_per_match`.
  - **keisei-08ccd20240** (P2): zero-legal-action guard still steps VecEnv with invalid action 0.
- `historical_gauntlet.py:run_gauntlet` tuple-unpacks `MatchOutcome` — every gauntlet match treated as failure (open **keisei-4509042dd1** P1). `MatchOutcome` is a dataclass (`match_utils.py:21`), not a tuple — confirms the bug is real.
- `frontier_promoter` / `tier_managers.FrontierManager.review` retires Static incumbent then aborts when Dynamic candidate became inactive (open **keisei-959d0eebe7** P1): silent shrink.
- `tournament_runner.py` Elo race: `_record_match_result` here is a separate path from the in-process `LeagueTournament`. Open issues **keisei-ea85c3d5b5** (P2: cross-worker lost-update race), **keisei-3067a55ab0** (P2: broad except sends entire batch to terminal `failed`), **keisei-90d4e0c1b2** (P2: `game_features.epoch` vs enqueued-epoch skew).
- `tournament_dispatcher.py` issues: **keisei-6cb0990f53** (PriorityScorer never sees sidecar match completions), **keisei-9cae4c09b4** (`check_round_completion` not idempotent — advances PriorityScorer repeatedly; visible at `tournament_dispatcher.py:140` where `advance_round` is called unconditionally if no pending/playing), **keisei-b8126dc1e5** (`enqueue_round` can exceed `dispatcher_max_queue_depth`).
- `demonstrator.py:CUDA action tensor not synchronized` — open **keisei-e4d01b2ce2** (P2): illegal-action race on private stream (`demonstrator.py:71` opens a `cuda.Stream` but tensor sync not enforced before `vecenv.step`).
- `read_tournament_stats` swallows all exceptions as None — **keisei-44945464ac** (P2; lives in db.py but consumed by this bucket via the lazy `write_tournament_stats` peer at `tournament.py:326`).
- `OpponentStore._pinned` is in-memory only (`opponent_store.py:341`) — known limitation tracked in `keisei-76cc7fdc85`.
- Two parallel result-recording code paths (`tournament._record_match_result` and `tournament_runner._record_result`) replicate Elo bookkeeping logic — divergence risk.

**Confidence:** High for module boundaries, public APIs, and dependency edges (entry sections of every module read; constructors verified). Medium for `LeagueTournament` deep internals (read l.1–460 of 658), `ConcurrentMatchPool` (read l.1–120 of 625), `OpponentStore` (read l.1–360 of 1324 — class structure verified, method bodies skimmed only). Medium for `tournament_runner.py` (entry section + main verified; `_record_result` body not read). Low for `concurrent_matches.py` slot-reuse claims beyond the filed bugs (I did not personally verify the partition-reset fault path; relying on the static sweep's findings which I cross-referenced to the file structure).

---

## C4. Style profiling & game features

**Location:** `keisei/training/{game_feature_tracker.py, style_profiler.py}`

**Responsibility:** Inline per-game behavioural feature extraction during match play (drops, captures, promotions, opening sequences) and a batch aggregator that writes `style_profiles` rows used by the WebUI commentary panel.

**Key Components:**
- `game_feature_tracker.py` (356 LoC) — `GameFeatureTracker` accumulates features from action ids and `StepMetadata`; `classify_action(action_id)` (l.52) decodes the spatial 9×9×139 encoding without replaying through `SpectatorEnv`. Constants for promotion/drop move-type ranges (l.22–28) and opening-window thresholds (l.42–47).
- `style_profiler.py` (466 LoC) — `StyleProfiler` reads from `read_all_game_features` and writes `write_style_profile`; sample thresholds `THRESHOLD_INSUFFICIENT/PROVISIONAL/TREND` (l.26); rule-based label table `_STYLE_RULES` (l.64); `_percentile_rank` via `bisect` (l.31).

**Dependencies:**
- Inbound (in-bucket): `concurrent_matches.py:22` (creates `GameFeatureTracker` per slot); `tournament.py:28` (passes tracker into `play_match`); `match_utils.py:18` (TYPE_CHECKING only). `style_profiler.StyleProfiler` is constructed by `tournament_dispatcher._recompute_style_profiles` (`tournament_dispatcher.py:121`).
- Outbound: `keisei.db.read_all_game_features`, `write_style_profile` (`style_profiler.py:18`); `numpy` only in `game_feature_tracker.py`. **No shogi_gym imports** — it consumes already-decoded action ids and metadata.
- Config consumed: none.

**Patterns Observed:**
- Move-type decoding hard-coded to match `spatial_action_mapper.rs` constants (`game_feature_tracker.py:19–28`); this is a Rust↔Python contract surface that lives outside the FFI types.
- Rule-table-driven style labelling (`style_profiler.py:64`) — readable, extensible.
- Single-writer enforcement on `style_profiles` documented at `tournament_dispatcher.py:113–119`.

**Concerns:**
- The `spatial_action_mapper` constants (move types 64–131 = with promotion, 132–138 = drops, 255 = NO_CAPTURE) are duplicated between Rust and `game_feature_tracker.py:19–28`. Drift risk if Rust encoding changes — no test cross-checks the constants.
- BLACK_ROOK_SQUARE / BLACK_KING_SQUARE indices (`game_feature_tracker.py:37–39`) hand-encode perspective rotation rules; comment notes "perspective-relative" handling but no validation.
- Style profile writes are not transactional with match results — they happen on a separate trainer-only path (`tournament_dispatcher._recompute_style_profiles`) and fail silently (`tournament_dispatcher.py:126–127`).

**Confidence:** Medium-High for the tracker constants and entry surface (read l.1–60). Low for the actual classification logic past l.60 of `game_feature_tracker.py` and aggregation in `style_profiler.py:80–466`.

---

## C5. Standalone evaluation CLI

**Location:** `keisei/training/evaluate.py`

**Responsibility:** `keisei-evaluate` console-script — head-to-head match between two checkpoints with Wilson-CI win-rate and Elo-delta reporting. Independent of the training loop and league.

**Key Components:**
- `evaluate.py` (196 LoC) — `EvalResult` dataclass with `win_rate`, `elo_delta`, `win_rate_ci` (l.21); `run_evaluation` (l.59); `_play_evaluation_games` (l.79; loads each model, instantiates one `VecEnv`); `main` (l.166).

**Dependencies:**
- Inbound: console_scripts entry (per discovery doc §3); not imported by any other in-bucket module.
- Outbound: `keisei.training.demonstrator._get_policy_flat` (`evaluate.py:15`) — bridges scalar/multi-head model output; `keisei.training.model_registry.build_model` (`evaluate.py:16`).
- FFI: `evaluate.py:101` → `VecEnv(observation_mode="katago", action_mode="spatial")`.
- DB: none. **No DB writes — purely stdout/CLI.**
- Config consumed: none (CLI argparse).

**Patterns Observed:**
- Wilson score CI (`evaluate.py:46`) — proper statistical reporting.
- `torch.load(weights_only=True)` for safety (`evaluate.py:88,95`).
- Cross-architecture matches supported via per-side `arch_a`/`arch_b` parameters.

**Concerns:**
- Open issue **keisei-9a5ac20307** (P2): `keisei-evaluate` ignores model hyperparameters (`params_a or {}` at `evaluate.py:75`) — `load_state_dict` fails for non-default architectures because `build_model(arch, {})` constructs default-shape weights that don't match the checkpoint.
- Imports private `_get_policy_flat` from `demonstrator.py:22` — leaking module-private helper.

**Confidence:** High — file is short (196 LoC), entry section read in full.

---

## D. Supervised learning pipeline

**Location:** `keisei/sl/`

**Responsibility:** Distinct lifecycle from RL training — parses external Shogi game records (SFEN/CSA), encodes positions to the same 50-channel observation, writes binary shards, and trains a `KataGoBaseModel` via supervised cross-entropy + MSE losses. Owns the `keisei-prepare-sl` CLI and `SLTrainer` (which is invoked by `transition.sl_to_rl`).

**Key Components:**
- `parsers.py` (405 LoC) — `GameParser` ABC, `SFENParser`, `CSAParser`; `GameRecord`/`ParsedMove`/`GameOutcome` dataclasses (l.15–32); `GameFilter` for ply/rating filtering (l.34).
- `prepare.py` (278 LoC) — `prepare_sl_data` (l.54); `_iter_records_safe` (l.37) per-record exception isolation; parser registry (l.25); `keisei-prepare-sl` `main`.
- `dataset.py` (208 LoC) — `SLDataset` (l.72; mmap-backed shard reader); `write_shard` (l.46) with structured numpy dtype `_SHARD_DTYPE` (l.37) asserting `RECORD_SIZE` matches; `OBS_SIZE = 50*81 = 4050`, `RECORD_SIZE = 16220`, `SCORE_NORMALIZATION = 76.0` (l.32, **shared with RL via `katago_ppo.py:14`**).
- `trainer.py` (179 LoC) — `SLConfig` (l.18; `__post_init__` validation but lambdas are NOT validated for sign — see Concerns); `SLTrainer` (l.61) trains one epoch at a time; AMP + cosine LR scheduler.

**Dependencies:**
- Inbound (in-bucket): `keisei.training.transition` imports `SLTrainer`, `SLConfig`; `keisei.training.katago_ppo` imports `SCORE_NORMALIZATION` from `sl.dataset`. **`sl/` does NOT import from `keisei.training` except `models.katago_base.KataGoBaseModel` (`trainer.py:15`)** — clean one-way dependency for the model class only.
- Inbound (out-of-bucket): `tests/`. The `keisei-prepare-sl` console script entry is the public CLI surface.
- Outbound: `keisei.training.models.katago_base.KataGoBaseModel`; `numpy`, `torch.utils.data.{Dataset,DataLoader}`.
- DB: **none.** SL uses files only; `transition.sl_to_rl` is the writer of `training_state` after SL completes.
- FFI: **none directly.** Position encoding in `prepare.py` is pure Python/numpy — does not call `shogi_gym` (per the file head; see confidence note).
- Config consumed: none from `keisei.config`; `SLConfig` is its own dataclass (`trainer.py:18`).

**Patterns Observed:**
- Per-record exception isolation in `_iter_records_safe` (`prepare.py:37–51`) so one corrupt record cannot kill a whole shard.
- Structured dtype + `assert _SHARD_DTYPE.itemsize == RECORD_SIZE` (`dataset.py:43`) prevents binary-layout drift.
- `_sl_worker_init` clears mmap cache after fork (`trainer.py:48`) — correct DataLoader-worker discipline.
- Empty-dataset defensive paths: `shuffle=has_data`, `num_workers=... if has_data else 0`, `pin_memory=...` only when there is data (`trainer.py:101–110`).

**Concerns:**
- `SLConfig.__post_init__` (`trainer.py:32–42`) validates `grad_clip > 0`, `total_epochs >= 0`, `batch_size > 0`, `learning_rate > 0`, `num_workers >= 0`, **but does not validate `lambda_policy/lambda_value/lambda_score`** — open issue **keisei-678359b7aa** (P1): negative lambdas cause silent gradient ascent.
- `prepare_sl_data()` writes outcome-proxy score targets instead of per-position material balance — open issue **keisei-25cb7bb826** (P1; `prepare.py:171` placeholder comment confirms).
- `SLDataset.write_shard` uses `assert` for shape validation (`dataset.py:59–62`) — open issue **keisei-4048efd9e0** (P2; stripped under `python -O`).
- `SLDataset` accepts truncated/trailing-garbage shards by flooring record count — open issue **keisei-1bc29e72e8** (P2).
- `CSAParser` issues: hard-codes %KACHI/%JISHOGI as last-mover-wins (**keisei-7a3316c590**), drops valid V2.2 games ending in `%+ILLEGAL_ACTION` (**keisei-295086b4cc**), no comma multi-statement tokenizer (**keisei-7af7ec2b8a**) — three open P2 bugs in `parsers.py`.
- Score-loss balance: open observation **keisei-1d50558dda** (P2): `lambda_score=0.02` may be too small now that score loss is dense (this is a SL vs RL tuning mismatch).

**Confidence:** High for `trainer.py` (read in full), `dataset.py` (entry section + structured-dtype machinery read). Medium for `parsers.py` (l.1–60 read; `CSAParser` body not read — relying on the four cross-referenced bugs to characterise its concerns). Medium for `prepare.py` (l.1–60 read; the outcome-proxy bug confirmed via grep at l.171).

---

## Cross-cutting observations

- **Densest internal coupling:** `katago_loop.py` imports 14 in-bucket modules — it is the single integration point that binds C1, C2, and C3 together (verified by direct grep of the loop's `from keisei.training.*` imports).
- **Two parallel match-recording paths**: `LeagueTournament._record_match_result` (`tournament.py:352`) and `tournament_runner._record_result` re-implement the same Elo + DB write sequence. The dispatcher/queue/worker arc is a newer code path running alongside the legacy in-process tournament thread; both are still wired into the loop.
- **Lazy `shogi_gym.VecEnv` imports in 7 modules** are a deliberate pattern to keep the modules importable in environments without the Rust extension built — useful for tests but creates an ad-hoc convention rather than a single-import-point seam.
- **DB is the only message bus between training and the server/showcase buckets.** The only out-of-bucket Python import is `showcase/inference.py` consuming `model_registry.{build_model, get_model_contract, get_obs_channels}` — i.e. for inference-time architecture lookup, not for training control flow.
- **SL is cleanly separable.** `keisei/sl/` only imports `KataGoBaseModel` from `keisei/training/models/`. The reverse direction is tighter (`katago_ppo.py` and `transition.py` import from `sl.{dataset,trainer}`), but the cycle is one-way modulo a shared constant (`SCORE_NORMALIZATION`).
- **Filed-bug density** is overwhelmingly concentrated in C3 (league system, ~13 open bugs) and C1a/SL (~7 bugs). C2 (models) and C5 (evaluate) each have one filed concern. C1b (loop) has two. This matches the discovery doc's risk preliminary.


---

# Bucket E/F/G — Server / Data Layer

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

**Server → client messages (12 types):**

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


---

# Bucket H — WebUI

# WebUI Subsystem Catalog (bucket H)

Scope: `webui/` — Svelte 4 SPA (29 `.svelte` components, 10 production stores, helpers, package.json, public/). Tests live alongside sources (`*.test.js`); coverage exists for every store and most pure helpers but is not catalogued here. The single chart library is `uplot ^1.6.31` (`webui/package.json:18`).

## Data flow: WebSocket message → store(s) updated → consuming view(s)

Source of truth: `webui/src/lib/ws.js:95–225`. Showcase outbound commands are dispatched via `sendShowcaseCommand` (`ws.js:37`) from `MatchControls.svelte` and `MatchQueue.svelte`.

| WS message (in) | Stores written | Views / components consuming |
|---|---|---|
| `init` (`ws.js:97–121`) | `games`, `selectedGameId`, `metrics`, `trainingState`, `leagueEntries` (+ `diffLeagueEntries`), `leagueResults`, `eloHistory`, `historicalLibrary`, `gauntletResults`, `leagueTransitions`, `headToHeadRaw`, `tournamentStats`, `styleProfilesRaw`, `showcaseGame`, `showcaseMoves`, `showcaseQueue`, `sidecarAlive` | All views (cold-start snapshot) |
| `game_update` (`ws.js:123–145`) | `games` (delta-merge by `game_id`), `selectedGameId` (auto-switch off ended games) | Training tab: `App.svelte` board area, `GameThumbnail`, `MoveLog`, `EvalBar`, `PieceTray`, `PlayerCard` (via `selectedGame`/`selectedOpponent`) |
| `metrics_update` (`ws.js:147–149`) | `metrics` (append + cap to MAX_POINTS=10000 in `metrics.js:4`) | Training tab: `MetricsGrid`, `MetricsChart`; Training-status surface: `App.svelte`'s `learnerStats` via `latestMetrics` |
| `training_status` (`ws.js:151–167`) | `trainingState` (merge-update of status, phase, heartbeat, epoch, step, episodes, config_json, display_name, model_arch, total_epochs, system_stats, learner_entry_id) | `StatusIndicator`, `App.svelte` (PlayerCard learner side), `HistoricalLibrary`, `learnerEntry` derived |
| `league_update` (`ws.js:169–180`) | Same league stores as `init` (entries, results, eloHistory, historicalLibrary, gauntletResults, transitions, headToHeadRaw, tournamentStats, styleProfilesRaw) | League tab: `LeagueView`, `LeagueTable`, `LeagueEventLog`, `MatchupMatrix`, `RecentMatches`, `EntryDetail`, `HistoricalLibrary`; cross-tab: `selectedOpponent` derived |
| `showcase_update` (`ws.js:182–205`) | `showcaseGame`, `showcaseMoves` (game-aware reset+append), `sidecarAlive=true`; calls `resetShowcaseSelectionOnGameChange` on game change | Showcase tab: `ShowcaseView`, `MatchScorecard`, `WinProbGraph`, `CommentaryPanel`, `Board`/`PieceTray`/`MoveLog` (showcase mode) |
| `showcase_status` (`ws.js:207–216`) | `showcaseQueue`, `sidecarAlive`; clears `showcaseGame`/`showcaseMoves` when no `active_game_id` | `MatchControls`, `MatchQueue`, `ShowcaseStatsBanner`, `ShowcaseView` offline banner |
| `showcase_error` (`ws.js:218–220`) | None — `console.warn` only | None (silent on UI) |
| `ping` (`ws.js:222–223`) | None (keepalive) | None |
| `connectionState` (internal store, `ws.js:30`) | `connectionState` ∈ {connecting, connected, reconnecting} with 3s grace before flipping to `reconnecting` (`ws.js:27, 67–76`) | `StatusIndicator` banners (`StatusIndicator.svelte:107–115`) |
| Outbound: `request_showcase_match`, `cancel_showcase_match`, `change_showcase_speed` | n/a (sent via `sendShowcaseCommand`) | `MatchControls.svelte:41–53`, `MatchQueue.svelte:23` |

---

## H1. App shell and WebSocket client

**Location:** `webui/src/App.svelte`, `webui/src/main.js`, `webui/src/app.css`, `webui/src/lib/ws.js`, `webui/src/lib/StatusIndicator.svelte`, `webui/src/lib/TabBar.svelte`

**Responsibility:** Bootstraps the SPA, owns the WS connection lifecycle (connect/reconnect/disconnect), routes between four tabs (Training / League / Showcase / About), and renders the Training-tab layout inline (thumbnails, player cards, board, metrics).

**Key Components:**
- `App.svelte` (445 lines) — Root. Mounts `connect()` (`App.svelte:24-27`), renders `StatusIndicator`, dispatches on `$activeTab` to `LeagueView`/`ShowcaseView`/`AboutView` or the inline Training layout, hosts the `<audio>` element for lofi (`App.svelte:162`).
- `main.js` (8 lines) — Standard Svelte 4 mount: `new App({ target: ... })`.
- `lib/ws.js` (240 lines) — WS client. Auto-reconnect with exponential backoff + 50–150% jitter (`ws.js:84–93`); 3 s `DISCONNECT_GRACE_MS` (`ws.js:27`) defers the `'reconnecting'` banner so brief drops are invisible.
- `lib/StatusIndicator.svelte` (236 lines) — Top header showing learner identity, alive/stale phase badge, epoch/step/games counters, wall/train clocks (`StatusIndicator.svelte:34–48`), CPU/GPU stats, plus connection banners.
- `lib/TabBar.svelte` (120 lines) — Four tab buttons (training, league, showcase, about) with arrow/Home/End keyboard nav (`TabBar.svelte:17–29`), audio toggle, theme toggle.
- `app.css` — Global tokens (not read line-by-line).

**Dependencies:**
- Inbound: `main.js` instantiates `App.svelte`; `App.svelte` is the root and is not imported elsewhere.
- Outbound: stores `games`, `training`, `league`, `metrics`, `navigation`, `audio`; helpers `safeParse`. WS messages consumed: ALL (this is the dispatcher entry point).

**Patterns Observed:**
- Tab routing by `{#if $activeTab === '…'}` ladder (`App.svelte:165–253`). No SPA router — there are exactly four mutually exclusive top-level views.
- Reactive `$:` blocks compute derived view-model fields from stores (`App.svelte:49–157`). Heavy use of IIFE-style reactives for multi-step derivations.
- Local `setInterval` for tick-driven freshness (`StatusIndicator.svelte:44–45`, `training.js:8`); cleaned up in `onDestroy`.
- `sendShowcaseCommand` (`ws.js:37`) is the only outbound surface — guarded by `readyState === OPEN`.

**Concerns:**
- `App.svelte` doubles as the Training tab layout (boards, PlayerCards, MoveLog wired inline at `App.svelte:187–246`). Pulling that out into a `TrainingView.svelte` would make it consistent with the other three tabs (`LeagueView`/`ShowcaseView`/`AboutView`). Code-observed: 445 lines vs. the other tab containers each at ≤566.
- Svelte-5 incompatibility surface area is concentrated here: 18 `export let` props across `App.svelte`/`StatusIndicator.svelte`, plus heavy reactive `$:` (will need `$state`/`$derived`/`$effect` rewrite). Filigree task `keisei-a5fe9f710e` (P4) tracks the migration.
- `ws.js` swallows `JSON.parse` failures with `console.warn` only (`ws.js:62–64`); no observability hook for malformed-message rate.
- `connect()` early-returns if `readyState <= OPEN` (`ws.js:44`) — i.e. if `CONNECTING` or `OPEN` — but a stuck `CONNECTING` socket cannot be cancelled by re-calling `connect()`.

**Confidence:** High for routing/lifecycle (read 100% of `App.svelte`, `main.js`, `ws.js`, `StatusIndicator.svelte`, `TabBar.svelte`). High for WS taxonomy.

---

## H2. Live game viewer (Training tab board)

**Location:** `webui/src/lib/Board.svelte`, `PieceTray.svelte`, `MoveLog.svelte`, `EvalBar.svelte`, `WinProbGraph.svelte`, `CommentaryPanel.svelte`, `NotationToggle.svelte`, `ShogiLegend.svelte`, `MoveDots.svelte`, `GameThumbnail.svelte`; helpers `pieces.js`, `handPieces.js`, `moveRows.js`, `usiCoords.js`, `evalCalc.js`, `movePatterns.js`, `gameThumbnail.js`.

**Responsibility:** Render the current learner game position (board grid, hands, eval bar, move log, last-move highlights, optional policy heatmap). Same components are reused by Showcase (H4) — the only delta is showcase passes `lastMoveFromIdx`/`lastMoveToIdx`/`heatmap` props that the Training tab leaves at defaults.

**Key Components:**
- `Board.svelte` (167 lines) — 9×9 grid, props for `board`, `inCheck`, `currentPlayer`, `lastMoveFromIdx/ToIdx`, `heatmap`. Read-only/aria-img.
- `PieceTray.svelte` (108 lines) — Captured-piece tray, uses `getHandPieces()` from `handPieces.js`.
- `MoveLog.svelte` (264 lines) — Notation list, supports interactive scrubbing (`selectedIdx`, `dispatch('select')`); preserves user scroll via `before/afterUpdate` (`MoveLog.svelte:53–72`); embeds `NotationToggle`.
- `EvalBar.svelte` (92 lines) — Vertical W/B percentage bar driven by `computeEval` (`evalCalc.js`).
- `WinProbGraph.svelte` (169 lines) — uplot line chart of value over plies, with vertical scrub marker plugin (`WinProbGraph.svelte:25–39`).
- `CommentaryPanel.svelte` (≥60 lines read) — Top candidates list, win-prob bar, REPLAY badge when scrubbing (`CommentaryPanel.svelte:30`).
- `NotationToggle.svelte` (55 lines) — Cycles `notationStyle` store across western/japanese/usi.
- `ShogiLegend.svelte` (228 lines) — Static piece guide with `MoveDots` for each piece.
- `GameThumbnail.svelte` (134 lines) — Mini board in the Training thumbnail panel; click sets `selectedGameId`.
- Helpers (`pieces.js` 29, `handPieces.js` 16, `moveRows.js` 55, `usiCoords.js` 40, `evalCalc.js` 16, `movePatterns.js` 100, `gameThumbnail.js` 32) — pure functions, all individually unit-tested.

**Dependencies:**
- Inbound: `App.svelte` (Training tab), `ShowcaseView.svelte` (Showcase tab).
- Outbound: stores `games` (`selectedGame`, `selectedOpponent`), `notation` (notationStyle); helpers above. WS messages consumed: `game_update` (Training), `showcase_update` (Showcase).

**Patterns Observed:**
- All board/eval components are pure presentational — no store imports inside `Board`/`PieceTray`/`EvalBar`. Data flows via props from view containers (`App.svelte`/`ShowcaseView.svelte`).
- `moveHistory` is passed as a serialised JSON string (`App.svelte:52`, `ShowcaseView.svelte:29`) and parsed in `MoveLog` via `parseMoves`. Tradeoff documented implicitly: stable prop identity for reactivity.
- Shared `notationStyle` store keeps multiple panels (`MoveLog`, `CommentaryPanel`) in lockstep across tabs (`notation.js:7–11`).
- `safeParse` (`safeParse.js`) used for JSON-typed columns from the backend (`App.svelte:50–52`).

**Concerns:**
- `Board.svelte` accepts board both as `[]` of piece objects but does not validate — a malformed `board_json` payload silently renders an empty grid. Mitigated by `safeParse` falling back, but no sentinel for "parse failed vs. genuinely empty".
- Svelte-5 migration surface: every component uses `export let` props (Board has 5; MoveLog 3; EvalBar 2; CommentaryPanel uses store subscription). All will become `$props()`.
- `MoveLog`'s `dispatch('select')` event pattern (`MoveLog.svelte:21,38`) is `createEventDispatcher` — Svelte 5 prefers callback props.

**Confidence:** High for Board/PieceTray/EvalBar/MoveLog/NotationToggle (read in full or near-full). Medium for `CommentaryPanel` (read 60/≥130 — only the head, but data plumbing is all there). Helpers are tiny and have tests.

---

## H3. League view

**Location:** `webui/src/lib/LeagueView.svelte`, `LeagueTable.svelte`, `LeagueEventLog.svelte`, `MatchupMatrix.svelte`, `RecentMatches.svelte`, `EntryDetail.svelte`, `MatchHistory.svelte`, `HistoricalLibrary.svelte`. Helpers `roleIcons.js`, `collapseEvents.js`, `eloChartData.js`.

**Responsibility:** Render the league leaderboard, head-to-head matrix, event log, recent matches, and a per-entry drill-down. Bound entirely to the league stores; reads from showcase/training only for the `learner_entry_id` linkage.

**Key Components:**
- `LeagueView.svelte` (354 lines) — Layout shell. Stats banner from `leagueStats`/`tournamentStats` (`LeagueView.svelte:46–87`), grid containing `LeagueTable`, optional `EntryDetail` panel (toggled by `focusedEntryId`), `MatchupMatrix`, `LeagueEventLog`, `RecentMatches`. Esc-to-close detail (`:23–28`).
- `LeagueTable.svelte` (474 lines) — Sortable leaderboard with flat/grouped views, role capacity columns; uses `leagueRanked`, `entryWLD`, `eloDelta`, `leagueByRole`, `styleProfiles`, `displayElo`.
- `LeagueEventLog.svelte` (148 lines) — Persistent event log driven by `leagueEvents` (localStorage-backed) plus `transitionCounts`; collapses adjacent same-type events via `collapseEvents.js`.
- `MatchupMatrix.svelte` (398 lines) — 20-slot symmetric grid (active league cap), padded with placeholder slots. Builds an aggregate "trainer" row across all learner snapshots (`:24–38`). Uses `headToHead` derived map.
- `RecentMatches.svelte` (290 lines) — Last 30 results from `leagueResults`, with epoch separators and clash counts.
- `EntryDetail.svelte` (>40 lines read) — Per-entry drill-down: primary/secondary Elo columns, last-round results, style profile, embedded `MetricsChart` for Elo over time.
- `MatchHistory.svelte` (83 lines) — Match list for one entry.
- `HistoricalLibrary.svelte` (114 lines) — Library slot table + gauntlet staleness indicator.

**Dependencies:**
- Inbound: `App.svelte` (loads `LeagueView` when `$activeTab === 'league'`).
- Outbound: stores `league` (most named exports — `leagueEntries`, `leagueResults`, `leagueRanked`, `leagueStats`, `learnerEntry`, `tournamentStats`, `headToHead`, `eloDelta`, `entryWLD`, `styleProfiles`, `leagueByRole`, `eloHistory`, `historicalLibrary`, `gauntletResults`, `leagueTransitions`, `leagueEvents`, `transitionCounts`, `focusedEntryId`); `training` (for `learner_entry_id`); helpers `roleIcons`, `collapseEvents`, `eloChartData`. WS messages consumed: `init`, `league_update`, `training_status` (for learner linkage).

**Patterns Observed:**
- Heavy use of `derived` stores in `league.js` (lines 201, 220, 227, 249, 278, 302, 318, 350, 367) — the view layer is mostly thin and the data shaping is centralised.
- LocalStorage persistence for `leagueEvents` with a (id, display_name) "run marker" fingerprint to invalidate stale events across DB resets (`league.js:30–60, 87–115`).
- `focusedEntryId` is a global writable used as a mini-router for the detail panel (`league.js:14`).
- "Trainer aggregate" row in `MatchupMatrix` is computed from snapshot grouping by `display_name` (`MatchupMatrix.svelte:16–18`).

**Concerns:**
- `league.js` is 383 lines of cross-cutting derived logic — by far the largest store. `diffLeagueEntries` is called explicitly from `ws.js` rather than reactively; this is documented at `league.js:82–86` but creates two paths to keep in sync (init at `ws.js:102`, league_update at `ws.js:171`).
- `LeagueTable.svelte` (474) and `MatchupMatrix.svelte` (398) are the largest components in the bucket; both will be high-effort Svelte 5 migration targets — many `$:` blocks (`LeagueTable.svelte:34–39` and similar) and `export let` props.
- `RecentMatches`'s `clashCounts` (`RecentMatches.svelte:9–17`) recomputes the full map on every reactive run — no memoisation, scaled by full `$leagueResults`.

**Confidence:** High for `LeagueView`, `LeagueEventLog`, `HistoricalLibrary` (read in full or near-full), and the `league.js` store API. Medium for `LeagueTable`/`MatchupMatrix`/`RecentMatches` interior layout (read first ~40 lines of each — data plumbing verified, but did not exhaustively trace all sort/render branches).

---

## H4. Showcase view

**Location:** `webui/src/lib/ShowcaseView.svelte`, `MatchControls.svelte`, `MatchQueue.svelte`, `MatchScorecard.svelte`, `ShowcaseStatsBanner.svelte`, `CommentaryPanel.svelte`, `WinProbGraph.svelte`. Showcase reuses H2 board components.

**Responsibility:** Render queued and live showcase matches with replay scrubbing, optional policy heatmap overlay, commentary, and outbound queue/cancel/speed controls.

**Key Components:**
- `ShowcaseView.svelte` (566 lines — largest in bucket) — Owns scrub keyboard handler (Arrow/Home/End/Space/h, `ShowcaseView.svelte:91–114`), heatmap aggregation across promotion variants (`:48–59`), live-move ARIA announcer (`:139–153`), assembles all child panels.
- `MatchControls.svelte` (242 lines) — Setup form (entry pickers + speed selector) collapsible while a match runs. Outbound: `request_showcase_match`, `change_showcase_speed`.
- `MatchQueue.svelte` (170 lines) — Queue list with two-step cancel confirmation. Outbound: `cancel_showcase_match`.
- `MatchScorecard.svelte` (322 lines) — Black/white player banners with tier badges, role icons, head-to-head stats, progress-bar by ply.
- `ShowcaseStatsBanner.svelte` (135 lines) — 3-card glanceable banner: engine status, live ply, queue depth.

**Dependencies:**
- Inbound: `App.svelte` (when `$activeTab === 'showcase'`).
- Outbound: stores `showcase` (all exports — `showcaseGame`, `showcaseMoves`, `showcaseQueue`, `sidecarAlive`, `showcaseSelectedPly`, `showcaseDisplayedMove`, `isScrubbing`, `winProbHistory`, `queueDepth`, `showcaseHeatmapEnabled`, `showcaseSpeed`, `resetShowcaseSelectionOnGameChange`); `league` (`leagueEntries`, `headToHead`, `displayElo`); helpers `safeParse`, `usiCoords`, `roleIcons`. WS messages consumed: `init` (showcase fields), `showcase_update`, `showcase_status`, `showcase_error`. Outbound WS: `request_showcase_match`, `cancel_showcase_match`, `change_showcase_speed` via `sendShowcaseCommand`.

**Patterns Observed:**
- `showcaseSelectedPly === null` represents "live tail" everywhere (`showcase.js:20–47`); landing on the last ply auto-flips back to live (`ShowcaseView.svelte:71`). This single sentinel keeps the scrub state machine simple.
- Persistent UI prefs in localStorage via store-subscribe pattern: `showcaseHeatmapEnabled` (`showcase.js:60–73`), `showcaseSpeed` (`showcase.js:75–91`), `audioEnabled`, `theme`, `notationStyle`, `aboutLevel`, `activeTab`, `keisei_league_events`, `keisei_league_event_run_marker` — nine distinct localStorage keys total.
- WS `showcase_update` clears moves on game change (`ws.js:188–193`) and only appends moves with `ply > maxPly` (`ws.js:198`) — server can safely re-send overlapping windows.
- Showcase keyboard handler skips form controls (`ShowcaseView.svelte:95–96`) so MatchControls' selects keep arrow keys.
- `disabledReason` pattern in `MatchControls.svelte:30–36` produces a per-state hint instead of a binary `disabled` flag.

**Concerns:**
- `ShowcaseView.svelte` at 566 lines mixes layout, scrub state machine, keyboard, ARIA announcer, and heatmap math. Splitting the scrub controller into a helper module would reduce the future Svelte 5 migration surface.
- `showcase_error` is logged only (`ws.js:218–220`); there's no UI surface for the message — silent on the user experience even when actionable.
- `MatchControls.requestMatch` does not retain optimistic queue state — it sends and waits for the next `showcase_status`. Acceptable, but a slow round-trip will show no feedback in the meantime.

**Confidence:** High for `ShowcaseView` (read 200/566 incl. the entire scrub/keyboard/announcer logic and all data flow), `MatchControls`, `MatchQueue`, `ShowcaseStatsBanner`, the showcase store. Medium for `MatchScorecard` (read first 40/322 — wiring confirmed, full layout not exhaustively traced).

---

## H5. Metrics and charts

**Location:** `webui/src/lib/MetricsGrid.svelte`, `MetricsChart.svelte`. Helpers `chartHelpers.js`, `eloChartData.js`, `metricsColumns.js`. Store `stores/metrics.js`.

**Responsibility:** Render the four training-metrics charts (policy/value loss, win rate, episode length, entropy) in a click-to-expand grid driven by the `metrics` time-series store. Charts are reused by `EntryDetail` (H3) for Elo history.

**Key Components:**
- `MetricsGrid.svelte` (246 lines) — Composes four `MetricsChart` panels (`MetricsGrid.svelte:28–45`); resolves theme colors from CSS variables and re-reads on theme change (`:18–26`); click-to-expand UI.
- `MetricsChart.svelte` (201 lines) — uplot wrapper. Accepts `xData`, `series`, optional side legend mode; rebuilds on prop/theme change; uses `ResizeObserver`.
- `chartHelpers.js` (99 lines) — `buildChartOpts`, `buildChartData`, `resolveThemeColors` (CSS-var driven).
- `eloChartData.js` (42 lines) — Aggregates `eloHistory` rows for `EntryDetail`'s line chart.
- `metricsColumns.js` (49 lines) — `extractColumns`: pivots flat metric rows into per-key arrays (steps, epochs, policyLoss, valueLoss, pvRatio, winRates, etc.).

**Dependencies:**
- Inbound: `MetricsGrid` is rendered by `App.svelte` (Training tab footer). `MetricsChart` is also imported by `EntryDetail.svelte` (`EntryDetail.svelte:4`) and `WinProbGraph.svelte` shares uplot logic via `chartHelpers`.
- Outbound: stores `metrics` (writable + `latestMetrics` derived), `theme`. WS messages consumed: `metrics_update` (append) and `init` (initial load).

**Patterns Observed:**
- The `metrics` store is a custom factory that auto-prunes to 10000 points (`metrics.js:9–16`) — bounds memory growth on long runs.
- Chart colors flow through CSS custom properties so themes (light/dark) re-tint without re-rendering data (`MetricsGrid.svelte:18–26`, `chartHelpers.js`).
- `MetricsChart` is the only uplot consumer outside `WinProbGraph.svelte` — single chart library, two call sites (and `EntryDetail` reuses `MetricsChart`).

**Concerns:**
- `MetricsGrid` re-reads CSS-var colors via `getComputedStyle(document.documentElement).getPropertyValue` on every reactive run (`MetricsGrid.svelte:13`), and `chartColors` is recomputed when either `$theme` or `$metrics` changes (`:18`). The rebuild on `$metrics` change is unnecessary and would cause a getComputedStyle hit per metrics tick.
- `import 'uplot/dist/uPlot.min.css'` happens inside `MetricsChart.svelte:4` only. `WinProbGraph.svelte` imports `uPlot` but not its CSS — relies on `MetricsChart` being mounted first. If a user lands on Showcase before Training, the WinProbGraph might render unstyled.
- Svelte-5 surface: `MetricsChart` exports 8 props with `export let`; lifecycle uses `onMount`/`onDestroy`/`afterUpdate` — `afterUpdate` is being soft-deprecated in Svelte 5.

**Confidence:** High for store + grid + helper plumbing (read in full). Medium for `MetricsChart` interior (read 40/201; uplot lifecycle assumed-correct since tests exist for `chartHelpers`).

---

## H6. About / static views

**Location:** `webui/src/lib/AboutView.svelte`. Store `stores/aboutLevel.js`.

**Responsibility:** Static educational content with a 5-level progressive-disclosure slider ("The Big Idea" → "Research View"); persisted in localStorage.

**Key Components:**
- `AboutView.svelte` (>60 lines read of static content) — Hardcoded data tables for observation planes, model configs, head architectures, training knobs (`AboutView.svelte:5–60`); content gated by `$aboutLevel`.
- `aboutLevel.js` (32 lines) — Five-level enum with localStorage persistence; clamp to 1–5 (`aboutLevel.js:19–24`).

**Dependencies:**
- Inbound: `App.svelte` (when `$activeTab === 'about'`).
- Outbound: store `aboutLevel`. No WS messages consumed.

**Patterns Observed:**
- All content is hardcoded — there is no backend feed for the About tab. This makes it a "pure documentation" surface.
- Same localStorage-clamp pattern as other persistent prefs (`aboutLevel.js:19–24`).

**Concerns:**
- Numeric constants in the About tab (e.g. `lr = 2e-4`, GAE γ/λ, entropy schedule, plane indices at `AboutView.svelte:7–61`) are hardcoded copies of training-side config. If config drifts, About becomes silently wrong. No test asserts agreement with `keisei/config.py` defaults.

**Confidence:** Medium — read first 60/≥? lines of content tables and the full store; did not read the gating/render markup tail.

---

## H7. Cross-cutting infrastructure

**Location:** Stores `theme.js`, `audio.js`, `navigation.js`, `notation.js`, `training.js`, `aboutLevel.js`. Helpers `safeParse.js`, `collapseEvents.js`, `timeFormat.js`, `indicator.js`, `configTooltip.js`, `roleIcons.js`. Components `StatusIndicator.svelte` (banner + clocks; principally H1), audio element (in `App.svelte`).

**Responsibility:** Stateless cross-cutting concerns — persisted UI prefs, time formatting, status iconography, JSON parsing. Each is small and individually unit-tested.

**Key Components:**
- `theme.js` (17 lines) — Dark/light toggle, sets `data-theme` on `<html>`, persists via localStorage.
- `audio.js` (28 lines) — `audioEnabled` flag (default false); App.svelte reconciles the `<audio>` element on store change (`App.svelte:36–47`), gracefully handles `NotAllowedError` autoplay rejection without flipping the flag.
- `navigation.js` (10 lines) — `activeTab` (default 'training'), persisted.
- `notation.js` (35 lines) — Three-style cycle with shared labels.
- `training.js` (19 lines) — `trainingState` plus a derived `trainingAlive` ticking every 10s on heartbeat freshness threshold of 30s (`training.js:13–18`).
- `safeParse.js` (11 lines) — JSON.parse with fallback; pass-through for non-strings.
- `collapseEvents.js` (27 lines) — Adjacent-event run-length compaction for `LeagueEventLog`.
- `timeFormat.js` (35 lines) — UTC parser + elapsed formatter for the wall/train clocks.
- `indicator.js` (14 lines) — Maps `(alive, status)` → display indicator for `StatusIndicator`.
- `configTooltip.js` (24 lines) — Builds tooltip text from the training state's `config_json`.
- `roleIcons.js` (42 lines) — Tier role → emoji/label/colour.

**Dependencies:**
- Inbound: every other bucket consumes one or more of these.
- Outbound: only `localStorage`, `document.documentElement` (theme), `Intl`-style date formatting. No WS coupling except indirectly via the `trainingAlive` derived store.

**Patterns Observed:**
- Universal "pref store" idiom: `loadInitial()` from localStorage with defensive `typeof localStorage !== 'undefined'` guards (SSR-safe even though the app is SPA-only) — repeated in `audio.js:15–22`, `navigation.js:1–10`, `notation.js:14–22`, `theme.js:1–13`, `aboutLevel.js:17–32`, `showcase.js:62–91`, `league.js:35–48`.
- Pure helper modules are exhaustively tested (`pieces.test.js`, `usiCoords.test.js`, `evalCalc.test.js`, `handPieces.test.js`, `safeParse.test.js`, `timeFormat.test.js`, `chartHelpers.test.js`, `configTooltip.test.js`, `metricsColumns.test.js`, `moveRows.test.js`, `eloChartData.test.js`, `gameThumbnail.test.js`, `collapseEvents.test.js`, `movePatterns.test.js`, `indicator.test.js`).

**Concerns:**
- `training.js`'s 10s tick (`training.js:8`) runs unconditionally for the lifetime of the page even when the Training tab is not active — minor, but adds wakeups and CPU work.
- The "pref store + localStorage" idiom is repeated 9× verbatim. Extracting a `createPersistedStore(key, default, validator)` helper would shrink ~40 lines and centralise the SSR guard.

**Confidence:** High — all eight stores and most helpers read in full or are <30 lines. Test coverage is broad.

---

## Cross-cutting Svelte 5 migration risk (informational)

Code patterns observed across this bucket that filigree task `keisei-a5fe9f710e` will need to address:
- `export let` props: every `.svelte` file in `lib/` (29 files). Will become `$props()`.
- Reactive `$:` blocks: most components, often nested IIFEs (`App.svelte:76–123` for `learnerFlavour`/`learnerStats`). Will become `$derived` or `$effect`.
- `createEventDispatcher` + `dispatch('select')`: at least `MoveLog.svelte:21,38`. Will become callback props.
- `<slot/>`-based composition: did not see explicit slot usage in the files read; LeagueView/ShowcaseView/AboutView are routed by `{#if}` from `App.svelte`. Slots are likely used in card/panel patterns not exhaustively read — not flagged with confidence.
- `beforeUpdate`/`afterUpdate`: `MoveLog.svelte:53,59`, `MetricsChart.svelte`, `PlayerCard.svelte:36`. Soft-deprecated in Svelte 5.

## Cross-reference: filigree open issues

`filigree list --label=bug --label=P1 --json` and `--label=P2 --json` both returned `[]` at time of analysis (2026-05-05). Searching `webui` returned only the four-issue Svelte 5 migration cluster: `keisei-9b1171d032` (deps), `keisei-a5fe9f710e` (migrate), `keisei-975949c0b3` (component APIs), `keisei-a1622bc4cf` (verify tests). All P4. No P1/P2 webui-specific bugs are open.

## Files skimmed only superficially

`LeagueTable.svelte` (read 40/474), `MatchupMatrix.svelte` (40/398), `RecentMatches.svelte` (40/290), `MatchScorecard.svelte` (40/322), `MatchControls.svelte` (60/242 — verified outbound surface), `EntryDetail.svelte` (40/≥), `MetricsChart.svelte` (40/201), `ShowcaseView.svelte` (200/566 — read scrub/keyboard/announcer/data plumbing in full, skimmed the layout markup tail), `CommentaryPanel.svelte` (60/≥130), `AboutView.svelte` (60/≥), `ShogiLegend.svelte` (30/228 — content-only). All lighter components (`HistoricalLibrary`, `LeagueEventLog`, `MatchHistory`, `MatchQueue`, `ShowcaseStatsBanner`, `GameThumbnail`, `Board`, `PieceTray`, `MoveLog`, `EvalBar`, `WinProbGraph`, `NotationToggle`, `MoveDots`, `TabBar`, `StatusIndicator`) and ALL stores + ALL non-test JS helpers in `lib/` were read in full or near-full for the data-flow-relevant sections.
