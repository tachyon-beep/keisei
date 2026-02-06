# Keisei Deep Code Analysis — Repair Manifest

**Date:** 2026-02-07
**Scope:** 87 Python source files, 27,222 lines across 7 subsystems
**Packages analyzed:** 25/25
**Analysis documents:** 87/87

---

## Executive Summary

| Verdict | Count | % |
|---------|-------|---|
| CRITICAL | 11 | 12.6% |
| NEEDS_ATTENTION | 50 | 57.5% |
| SOUND | 26 | 29.9% |

**By subsystem:**

| Subsystem | Files | CRITICAL | NEEDS_ATTENTION | SOUND |
|-----------|-------|----------|-----------------|-------|
| Config/Root | 3 | 0 | 3 | 0 |
| Core | 7 | 2 | 4 | 1 |
| Shogi | 8 | 1 | 4 | 3 |
| Training | 28 | 1 | 20 | 7 |
| Evaluation | 24 | 4 | 12 | 8 |
| Utils | 10 | 2 | 5 | 3 |
| WebUI | 3 | 1 | 1 | 1 |

The codebase has a functioning training pipeline but suffers from: (1) a security vulnerability in the WebUI HTTP server, (2) several interface contract mismatches that cause silent data corruption or runtime crashes, (3) pervasive dead code and incomplete implementations in the evaluation subsystem, and (4) multiple instances of silent failure that make debugging difficult. The training core (PPO, experience buffer, game engine) is fundamentally sound but has correctness issues at critical boundaries.

---

## P0 — Must Fix (Active bugs, security vulnerabilities, data corruption)

These findings represent code that is actively broken, produces incorrect results, or exposes security vulnerabilities in production use.

### P0-01: Directory Traversal in WebUI HTTP Server
- **File:** `keisei/webui/web_server.py` lines 22-29
- **Impact:** SECURITY — Remote file disclosure from any network-reachable client
- **Details:** `translate_path()` bypasses `SimpleHTTPRequestHandler`'s built-in path sanitization by directly joining user-supplied URL paths with the static directory via `os.path.join`, without normalizing or rejecting `..` path components. Combined with default binding to `0.0.0.0`, an attacker can read arbitrary files on the host filesystem.
- **Fix:** Use `os.path.normpath` + prefix check, or delegate to the parent class's `translate_path()`.

### P0-02: evaluate_actions Return Order Mismatch
- **Files:** `keisei/core/actor_critic_protocol.py`, `keisei/core/base_actor_critic.py`, `keisei/evaluation/core/model_manager.py`
- **Impact:** CORRECTNESS — Silent training corruption or runtime crash
- **Details:** The `ActorCriticProtocol` docstring specifies `evaluate_actions` returns `(log_probs, values, entropy)`. The actual `BaseActorCriticModel` implementation returns `(log_probs, entropy, value)` — values and entropy are swapped. `DynamicActorCritic` in `evaluation/core/model_manager.py` follows the docstring order, meaning evaluation unpacks values where entropy should be and vice versa.
- **Fix:** Align the protocol, all implementations, and all callers to a single return order.

### P0-03: Batch-Wide NaN Fallback Destroys Valid Distributions
- **File:** `keisei/core/base_actor_critic.py`
- **Impact:** CORRECTNESS — Training quality silently degraded
- **Details:** In `get_action_and_value`, when any single sample in a batch has all legal actions masked, the NaN detection triggers a batch-wide uniform fallback that replaces the policy distribution for ALL samples, including those with perfectly valid action masks. This means one malformed sample silently corrupts the entire batch's policy.
- **Fix:** Apply NaN fallback per-sample, not batch-wide.

### P0-04: callbacks.py save_checkpoint Called With Wrong Arguments
- **File:** `keisei/training/callbacks.py` lines 147-152
- **Impact:** CRASH — TypeError on first evaluation attempt
- **Details:** `EvaluationCallback.on_step_end()` bootstrap path calls `save_checkpoint(agent, global_timestep, run_artifact_dir, "initial_eval_checkpoint")` — passing 4 arguments in the wrong order. `ModelManager.save_checkpoint` requires 7 positional arguments `(agent, model_dir, timestep, episode_count, stats, run_name, is_wandb_active)`. This will raise `TypeError` at runtime. Additionally, the return value is `Tuple[bool, Optional[str]]` but is treated as a path string.
- **Fix:** Pass all 7 required arguments in correct order; unpack the return tuple.

### P0-05: CustomEvaluator Returns Random Fabricated Results
- **File:** `keisei/evaluation/strategies/custom.py` lines 271-286
- **Impact:** CORRECTNESS — Evaluation metrics are entirely fabricated
- **Details:** `evaluate_step()` is a placeholder that returns `random.choice([0, 1, None])` as game results. The entire `CustomEvaluator` — all 414 lines of evaluation orchestration including round-robin, single-elimination, and custom sequence modes — produces fabricated metrics with no actual Shogi games being played. No warning, no `NotImplementedError`, no logging.
- **Fix:** Implement actual game execution or raise `NotImplementedError`.

### P0-06: torch.load Without weights_only=True (3 locations)
- **Files:** `keisei/core/ppo_agent.py`, `keisei/training/utils.py` line 32, `keisei/evaluation/core/model_manager.py`
- **Impact:** SECURITY — Arbitrary code execution from untrusted checkpoint files
- **Details:** `torch.load` without `weights_only=True` allows pickle deserialization, enabling arbitrary code execution when loading malicious checkpoint files. PyTorch 2.7.0 (installed) defaults to `weights_only=True`, so these calls may also break with legitimate checkpoints containing non-tensor metadata.
- **Fix:** Add `weights_only=True` to all `torch.load` calls; use `torch.serialization.add_safe_globals()` for non-tensor metadata.

### P0-07: Escaped Newlines in Log File Writes
- **File:** `keisei/utils/utils.py` lines 514, 561
- **Impact:** CORRECTNESS — All log files are single unbroken lines
- **Details:** Both `TrainingLogger.log()` and `EvaluationLogger.log()` write `"\\n"` (literal backslash-n) instead of `"\n"` (actual newline) to log files. Every entry is appended without line separation, making log files unreadable.
- **Fix:** Change `"\\n"` to `"\n"` in both write calls.

### P0-08: Silent Checkpoint Load Failure
- **File:** `keisei/training/model_manager.py` lines 346-349, 362-365
- **Impact:** DATA LOSS — Training silently continues from random initialization
- **Details:** `PPOAgent.load_model()` returns a dict with an `"error"` key on failure instead of raising. `ModelManager` stores this dict as `self.checkpoint_data` and returns `True` (success). Training continues from a randomly-initialized model while believing it has resumed from a checkpoint, silently losing all prior training progress.
- **Fix:** Raise an exception on checkpoint load failure, or validate the returned dict structure.

### P0-09: Hand Piece Normalization Uses Wrong Divisor
- **File:** `keisei/shogi/shogi_game_io.py`
- **Impact:** TRAINING QUALITY — Observation encoding degrades learning
- **Details:** Hand piece channel normalization divides by 18.0 for ALL piece types. The correct per-type maximums are: 18 for Pawns, 4 for Lances/Knights/Silvers/Golds, 2 for Bishops/Rooks. Using `/18.0` for Bishops means having 2 Bishops in hand produces a channel value of 0.11 instead of 1.0, significantly compressing the dynamic range and degrading the neural network's ability to distinguish hand compositions.
- **Fix:** Use per-piece-type maximum counts for normalization.

---

## P1 — High Priority (Correctness issues, significant data quality impact)

These findings represent logic errors, architectural defects, or incorrect behavior that doesn't immediately crash but produces wrong results or limits functionality.

### P1-01: Async Callbacks Architecturally Broken
- **Files:** `keisei/training/train.py`, `keisei/training/training_loop_manager.py`
- **Impact:** The `--enable-async-evaluation` flag effectively does nothing
- **Details:** `main()` in `train.py` is async (called via `asyncio.run()`), so an event loop is always active. `_run_async_callbacks_sync()` detects the running loop and skips all async callbacks with a warning. The async evaluation path is structurally dead.

### P1-02: Ladder Evaluator Config Access Bugs (3 locations)
- **File:** `keisei/evaluation/strategies/ladder.py`
- **Impact:** Configuration is always ignored; hardcoded defaults used
- **Details:** (1) Line 512: `getattr(self.config, "num_games_per_match", 2)` always returns 2 — value lives in `strategy_params`. (2) Line 726: `getattr(self.config, "num_opponents_to_select", 5)` always returns 5. (3) Line 715: `opp.name != agent_rating` compares string to float — always True — making opponent self-exclusion a no-op.

### P1-03: Benchmark Evaluator Config Access Bug
- **File:** `keisei/evaluation/strategies/benchmark.py` line 158
- **Impact:** Multi-game-per-case configurations silently ignored
- **Details:** `getattr(self.config, "num_games_per_benchmark_case", 1)` always returns 1 because this parameter is in `strategy_params`, not a direct attribute.

### P1-04: Inverted Recency Bonus in Opponent Selection
- **File:** `keisei/evaluation/opponents/enhanced_manager.py` line 351
- **Impact:** Opponent diversity degraded — recently played opponents preferred
- **Details:** Comment says "Bonus for less recently played opponents" but the math awards a higher bonus to more recently played opponents, inverting the intended curriculum behavior.

### P1-05: EvaluationResult.from_dict() Calls Undefined Function
- **File:** `keisei/evaluation/core/evaluation_result.py` lines 330-333
- **Impact:** Deserialization is broken for saved evaluation results
- **Details:** `from_dict()` calls `get_config_class()` which is never defined or imported anywhere. The fallback calls `EvaluationConfig.from_dict()` which doesn't exist on the Pydantic model.

### P1-06: features.py build_core46 Incompatible With ShogiGame
- **File:** `keisei/shogi/features.py`
- **Impact:** Dead code maintenance trap; would crash if ever called
- **Details:** Uses wrong attribute access patterns (`piece.is_promoted()` method call vs `piece.is_promoted` property), wrong method calls, no perspective flip. Completely diverges from the production observation generator in `shogi_game_io.py`.

### P1-07: Redundant Attack-Check Computation Doubles Legal Move Cost
- **File:** `keisei/shogi/shogi_rules_logic.py` lines 527-543
- **Impact:** Performance — legal move generation approximately 2x slower than necessary
- **Details:** `generate_all_legal_moves` computes a full attack check that is immediately discarded. The subsequent `generate_raw_moves` + `is_legal_move` filter already handles check detection.

### P1-08: check_for_uchi_fu_zume Mutates Game State Without Exception Safety
- **File:** `keisei/shogi/shogi_rules_logic.py`
- **Impact:** Game state corruption if exception occurs during check
- **Details:** The method makes speculative moves to test for uchi-fu-zume but uses no try/finally to ensure state is restored on exception.

### P1-09: Processing State Leak in Training Loop
- **File:** `keisei/training/training_loop_manager.py` lines 138-149
- **Impact:** Metrics manager permanently stuck in "processing" state
- **Details:** If `perform_ppo_update` raises, `set_processing(False)` is never called, permanently leaving the metrics manager in a stale state.

### P1-10: Model eval/train Mode Not Protected by try/finally
- **File:** `keisei/training/callbacks.py`
- **Impact:** Neural network stuck in eval mode after evaluation exception
- **Details:** The sync `EvaluationCallback` sets model to eval mode before evaluation but has no try/finally to restore train mode if evaluation raises, silently degrading subsequent training.

### P1-11: Blocking async def in parallel_executor
- **File:** `keisei/evaluation/core/parallel_executor.py`
- **Impact:** Event loop frozen during evaluation
- **Details:** `execute_games_parallel` is declared `async` but blocks the event loop with synchronous `concurrent.futures.as_completed` iteration.

### P1-12: Override File Merge Replaces Entire Config Sections
- **File:** `keisei/utils/utils.py` line 143
- **Impact:** Override files silently discard all non-overridden defaults
- **Details:** When a structured override config is detected, the merge does `config_data[k] = v` (shallow replace) instead of deep merge. An override specifying `{"training": {"learning_rate": 0.001}}` discards all other training defaults.

### P1-13: Elo Tracking Gated by Log Availability
- **File:** `keisei/training/callbacks.py`
- **Impact:** Elo tracking silently stops if logging is unavailable
- **Details:** Elo registry update is gated by `log_both is not None`. If logging is unavailable, Elo tracking stops, which is a logic error — Elo tracking should be independent of logging availability.

### P1-14: core_manager.py RuntimeError Self-Catch
- **File:** `keisei/evaluation/core_manager.py`
- **Impact:** Intentionally raised RuntimeErrors are silently swallowed
- **Details:** The `except RuntimeError` handler meant to catch "no running event loop" also catches RuntimeErrors intentionally raised within the try block.

---

## P2 — Medium Priority (Robustness, maintainability, data integrity concerns)

### P2-01: __deepcopy__ Drops move_history
- **File:** `keisei/shogi/shogi_game.py`
- Simulations and tree searches lose all game history.

### P2-02: set_piece Allows Out-of-Bounds Writes
- **File:** `keisei/shogi/shogi_game.py`
- No bounds checking on row/column coordinates.

### P2-03: board_history Grows Without Bound
- **File:** `keisei/shogi/shogi_game.py`
- Every move appends a 9x9 board copy with no pruning mechanism.

### P2-04: KIF Export Multiple Bugs
- **File:** `keisei/shogi/shogi_game_io.py`
- Uses current hands instead of initial position hands; drops drop moves; non-standard notation format; termination_map case mismatch.

### P2-05: Duplicate TEST_BATCH_SIZE Constants
- **File:** `keisei/constants.py` lines 75 and 152
- Defined as 16 at line 75, redefined as 32 at line 152. Second silently overwrites first.

### P2-06: Observation Constants Duplicated
- **Files:** `keisei/shogi/shogi_core_definitions.py` and `keisei/constants.py`
- Same constants defined in two places with different names; drift risk.

### P2-07: Silent Pass When All Legal Actions Masked
- **File:** `keisei/core/base_actor_critic.py`
- Produces uniform distribution over all 13,527 actions including illegal ones.

### P2-08: se_ratio Falsy-Value Override Bug
- **File:** `keisei/training/model_manager.py` line 83
- `getattr(...) or config_default` pattern means `se_ratio=0` falls through to config default.

### P2-09: Unconditional import wandb (Multiple Files)
- **Files:** `keisei/training/model_manager.py`, `keisei/training/utils.py`
- Makes entire modules fail to import if wandb not installed, violating optional-integration pattern.

### P2-10: Non-Standard Elo Batch Normalization
- **File:** `keisei/evaluation/opponents/elo_registry.py`
- Divides scores by `len(results)` before K-factor; 100-game batch same update as 1-game.

### P2-11: Dead Elo Tracking in Enhanced Opponent Manager
- **File:** `keisei/evaluation/opponents/enhanced_manager.py`
- `OpponentPerformanceData.elo_rating` never updated; ELO_BASED strategy always uses default 1200.0.

### P2-12: Display Refresh Iterates Model Parameters at 4Hz
- **File:** `keisei/training/display.py`
- GPU-to-CPU transfer of all model parameters on every dashboard refresh.

### P2-13: Incomplete Checkpoint Restore
- **File:** `keisei/training/metrics_manager.py`
- Trend data, Elo ratings, and enhanced metrics lost on resume.

### P2-14: Non-Thread-Safe Processing Flag
- **File:** `keisei/training/metrics_manager.py`
- Boolean flag read/written from training and WebUI threads without synchronization.

### P2-15: Hard-Coded Tensor Dimensions in Experience Buffer
- **File:** `keisei/core/experience_buffer.py`
- Dimensions `(buffer_size, 46, 9, 9)` and `(buffer_size, 13527)` hardcoded instead of parameterized.

### P2-16: Weight Cloning for UI Doubles Peak Memory
- **File:** `keisei/training/trainer.py` lines 266-284
- Full model weight clone on every PPO update for cosmetic dashboard feature.

### P2-17: Hardcoded Dummy Config in agent_loading.py
- **File:** `keisei/utils/agent_loading.py` lines 48-170
- 120-line hardcoded `AppConfig` for evaluation agent loading; architecture mismatch risk.

### P2-18: CLI Defaults Override Config
- **File:** `keisei/training/train.py`
- argparse defaults for `strategy`, `num_games`, `opponent_type` always override config file values.

### P2-19: Profiling Unbounded Memory Growth
- **File:** `keisei/utils/profiling.py`
- Timing lists accumulate without windowing; ~40MB for 1M-step run.

### P2-20: SO_REUSEADDR Set After Bind
- **File:** `keisei/webui/web_server.py` line 97
- Socket option set after binding; has no effect.

### P2-21: WebUI Max Connections Never Enforced
- **File:** `keisei/webui/webui_manager.py`
- `max_connections=10` configured but never checked; unlimited connections accepted.

### P2-22: WebUI Panels Display Fabricated Data
- **File:** `keisei/webui/webui_manager.py` lines 517, 534, 592
- References to nonexistent attributes (`history.win_rates`, `step_manager.recent_episodes`) cause fallback to random/default data with no user indication.

### P2-23: Parallel Worker Silent Data Loss
- **File:** `keisei/training/parallel/self_play_worker.py` lines 301-304
- Worker experience batches silently dropped when queues are full; no backpressure, no metric.

### P2-24: No Worker Restart Mechanism
- **File:** `keisei/training/parallel/parallel_manager.py`
- Dead worker processes are detected but never restarted; system permanently degrades.

### P2-25: Evaluation core/model_manager.py Staged as Deleted
- **File:** `keisei/evaluation/core/model_manager.py`
- Git status shows file as deleted; `keisei/evaluation/core/__init__.py` still imports from it.

### P2-26: run_async() / run() Duplication in Training Loop
- **File:** `keisei/training/training_loop_manager.py`
- ~100 lines of duplicated logic between sync and async paths; bug fixes in one will be missed in other.

### P2-27: User-Controlled Run Names Unsanitized in Paths
- **File:** `keisei/training/session_manager.py` line 120
- Run names used directly in `os.path.join` without directory traversal sanitization.

### P2-28: Dangerous Decompression Fallback
- **File:** `keisei/training/parallel/utils.py`
- `decompress_array` can return 0-dimensional numpy array containing raw bytes if decompression fails.

---

## P3 — Low Priority (Code quality, dead code, minor issues)

### Dead Code
- `keisei/constants.py`: Dead `GameTerminationReason` class, dead `OBS_*` constants
- `keisei/shogi/features.py`: `build_core46` function (dangerous — see P1-06)
- `keisei/shogi/features.py`: Test/dummy `FeatureSpec` entries in production registry
- `keisei/training/display_components.py`: `ShogiBoard._pad_symbol` no-op method
- `keisei/training/model_manager.py`: `_handle_checkpoint_resume` never called
- `keisei/training/step_manager.py`: `_prepare_demo_info` never called
- `keisei/training/previous_model_selector.py`: Entire module unused in production
- `keisei/training/parallel/model_sync.py`: `prepare_model_for_sync` / `restore_model_from_sync` never called
- `keisei/training/parallel/communication.py`: `_shared_model_data` initialized to None, never used
- `keisei/evaluation/core/base_evaluator.py`: `validate_config()` always returns True
- `keisei/evaluation/performance_manager.py`: `PerformanceGuard` class never used
- `keisei/evaluation/strategies/tournament.py`: `evaluate_step_in_memory` is a no-op
- `keisei/utils/compilation_validator.py`: `create_compilation_decorator` creates but never invokes validator
- `keisei/utils/profiling.py`: `_timer_context` method unused

### Code Duplication
- `_format_hand` duplicated verbatim in `display_components.py` (2 classes)
- ~260 lines of game-loop code duplicated across 4 evaluation strategies
- Eval interval alignment logic tripled in `callback_manager.py`
- `run()` / `run_async()` ~80% identical in `training_loop_manager.py`
- `save_checkpoint` / `save_final_checkpoint` / `save_final_model` ~60 lines duplicated
- Move formatting code ~80% duplicated between `format_move_with_description` variants

### Triple Elo Implementation
- `keisei/training/elo_rating.py` — training self-play Elo
- `keisei/evaluation/analytics/elo_tracker.py` — evaluation analytics Elo
- `keisei/evaluation/opponents/elo_registry.py` — opponent tracking Elo
- Three independent implementations with slightly different math

### Unused Imports (partial list)
- `asyncio` in `callback_manager.py` and `callbacks.py`
- `sys` in `env_manager.py`, `session_manager.py`, `train_wandb_sweep.py`, `utils.py`
- `signal` in `web_server.py`
- `torch.profiler` in `performance_benchmarker.py`
- `ActorCriticProtocol` in `setup_manager.py`

### Missing Type Annotations
- `setup_manager.py`: All method parameters untyped
- `session_manager.py`: `args`, `eval_config`, `result` parameters duck-typed with `hasattr`
- `utils.py`: `setup_directories`, `setup_seeding`, `setup_wandb` untyped

### Minor Issues
- `DisplayComponent` protocol defined but never used as type constraint
- `MultiMetricSparkline.data` lists grow without bounds
- `TerminalInfo.unicode_ok` computed but never consumed
- Debug border colors in `display.py` lines 343-344
- Inconsistent rate units: `get_win_rates()` returns 0-100, `get_win_loss_draw_rates()` returns 0-1
- `format_ppo_metrics()` has side effect of recording to history
- `log_error_to_stderr` used for success messages in `display_manager.py`
- Error games counted as draws in `base_evaluator.py`
- `sys.path.insert` at module level in `shogi_game_io.py`
- `is_in_check` returns True (masks corruption) when king not found

---

## Recommended Remediation Order

**Sprint 1 — Security & Crashes (P0-01, P0-04, P0-06, P0-07)**
Fix the directory traversal vulnerability, the save_checkpoint crash, torch.load security, and escaped newlines. These are straightforward fixes with immediate impact.

**Sprint 2 — Training Correctness (P0-02, P0-03, P0-08, P0-09)**
Fix the evaluate_actions return order mismatch, batch-wide NaN fallback, silent checkpoint failure, and hand piece normalization. These directly affect training quality.

**Sprint 3 — Evaluation Pipeline (P0-05, P1-01 through P1-05)**
Fix or remove the CustomEvaluator placeholder, unbreak async callbacks, fix ladder/benchmark config access bugs, fix inverted recency bonus, and fix EvaluationResult deserialization.

**Sprint 4 — Robustness (P1-06 through P1-14, P2-01 through P2-10)**
Address exception safety, state management, dead code cleanup, and game engine robustness issues.

**Sprint 5 — Code Quality (P2-11 through P2-28, P3)**
Address code duplication, dead code, missing type annotations, and minor issues.

---

## Appendix: Analysis Coverage

All 87 Python source files in `keisei/` were analyzed across 25 packages.
Per-file analysis documents are at `docs/code_analysis/{path_with_underscores}.analysis.md`.
Per-file verdicts are in `docs/code_analysis/_verdicts.csv`.
The analysis plan is at `docs/code_analysis/_plan.md`.
