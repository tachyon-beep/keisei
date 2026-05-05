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
