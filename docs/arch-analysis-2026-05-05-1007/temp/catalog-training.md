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

- **Densest internal coupling:** `katago_loop.py` imports 11 in-bucket modules — every C3 module except `tournament_queue`/`tournament_runner`/`demonstrator` (sidecar/exhibition). It is the single integration point that binds C1, C2, and C3 together.
- **Two parallel match-recording paths**: `LeagueTournament._record_match_result` (`tournament.py:352`) and `tournament_runner._record_result` re-implement the same Elo + DB write sequence. The dispatcher/queue/worker arc is a newer code path running alongside the legacy in-process tournament thread; both are still wired into the loop.
- **Lazy `shogi_gym.VecEnv` imports in 7 modules** are a deliberate pattern to keep the modules importable in environments without the Rust extension built — useful for tests but creates an ad-hoc convention rather than a single-import-point seam.
- **DB is the only message bus between training and the server/showcase buckets.** The only out-of-bucket Python import is `showcase/inference.py` consuming `model_registry.{build_model, get_model_contract, get_obs_channels}` — i.e. for inference-time architecture lookup, not for training control flow.
- **SL is cleanly separable.** `keisei/sl/` only imports `KataGoBaseModel` from `keisei/training/models/`. The reverse direction is tighter (`katago_ppo.py` and `transition.py` import from `sl.{dataset,trainer}`), but the cycle is one-way modulo a shared constant (`SCORE_NORMALIZATION`).
- **Filed-bug density** is overwhelmingly concentrated in C3 (league system, ~13 open bugs) and C1a/SL (~7 bugs). C2 (models) and C5 (evaluate) each have one filed concern. C1b (loop) has two. This matches the discovery doc's risk preliminary.
