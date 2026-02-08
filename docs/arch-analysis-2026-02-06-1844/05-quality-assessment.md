# Architecture Quality Assessment

**Source:** `docs/arch-analysis-2026-02-06-1844/`
**Assessed:** 2026-02-08
**Assessor:** architecture-critic (6 parallel assessment agents)
**Prior Assessment:** 2026-02-06 (overwritten -- prior was letter-grade based without sufficient evidence depth)

## Executive Summary

Keisei has solid foundations -- a correct Shogi domain layer, a working PPO implementation, a well-designed WebUI, and comprehensive test fixtures. But three systemic problems undermine the architecture:

1. **Non-atomic checkpoint saves** risk losing entire training runs on crash
2. **The evaluation subsystem is 10x larger than the RL engine it evaluates** and is mostly unused infrastructure
3. **82 broad exception catches** suppress errors that should be loud, including in the game engine where silent failures corrupt training data

**Critical issues: 3 | High issues: 16 | Overall score: 2.8/5**

The prior assessment graded this B+ with no critical issues. That was wrong. The evidence below explains why.

---

## Subsystem Scores

| Subsystem | Score | Critical | High | Key Problem |
|-----------|-------|----------|------|-------------|
| Core RL Engine | 3/5 | 0 | 4 | `scaler` overloaded for 2 purposes; silent failure modes |
| Shogi Engine | 3/5 | 0 | 3 | O(N^4) move generation; sys.path mutation |
| Training System | 2.5/5 | 1 | 3 | Non-atomic checkpoints; manager pattern is cosmetic |
| Evaluation System | 2/5 | 1 | 6 | 10x over-engineered; 500+ lines of copy-paste |
| Utilities | 3/5 | 0 | 1 | Circular dep confirmed; grab-bag organization |
| Configuration | 3/5 | 0 | 3 | No cross-field validation; webui key dropped |
| WebUI (Streamlit) | 4/5 | 0 | 1 | Well-isolated; 0.0.0.0 default binding |
| Cross-cutting | 2.6/5 | 0 | 4 | No CI; 82 broad catches; unused test markers |

---

## Critical Issues (Must Fix)

### C1. Checkpoints Are NOT Saved Atomically

**Evidence:** `core/ppo_agent.py:483` -- `torch.save(save_dict, file_path)` writes directly to the target path. Called from `training/model_manager.py:639`.

**Impact:** If the process crashes or is killed during `torch.save`, the checkpoint file is partially written and corrupted. `model_manager.py:631-635` skips re-saving if the file already exists, so a corrupted checkpoint persists permanently. A long training run's progress is unrecoverable.

**Fix:** Save to `file_path + '.tmp'`, then `os.replace(file_path + '.tmp', file_path)`. `os.replace` is atomic on POSIX.

### C2. Deprecated Evaluation Config Emits Warnings at Import Time

**Evidence:** `evaluation/core/evaluation_config.py:26-30` -- The entire file is marked DEPRECATED but emits a `DeprecationWarning` at module import level, not inside a function. Every import triggers the warning.

**Fix:** Delete the file. Fix any remaining imports.

### C3. Silent Game Engine Exception Swallowing Corrupts Training Data

**Evidence:** `shogi/shogi_game.py:900` and `:956` -- `except Exception: return False` in move validation methods. Any bug in move validation logic is silently treated as "invalid move."

**Impact:** If a rules logic bug causes an unexpected exception during `test_move_validity`, the move is rejected without any error signal. Training data quality degrades silently -- the agent learns from incorrectly filtered move sets.

**Fix:** Log and re-raise. A bug in move validation should crash training, not silently corrupt it.

---

## High-Severity Issues

### Training System

**H1. Manager Pattern is Cosmetic (God Object via Back-Reference)**

`TrainingLoopManager.__init__` takes `trainer: "Trainer"` and makes 65+ references to `self.trainer.*`. All callbacks receive the full Trainer. SetupManager is pure forwarding (every method wraps another manager's method). No manager defines a protocol or interface. The 9-manager architecture provides file-level organization but zero encapsulation.

**H2. PPO Update Has No Error Recovery**

`trainer.py:232-293` (`perform_ppo_update`) has no try/except. An OOM or NaN during `agent.learn()` propagates up and kills training. The finally block saves a checkpoint, but the optimizer state may be partially updated.

**H3. Shared Mutable State Without Synchronization**

`metrics_manager.py:103` -- `pending_progress_updates` is a plain `Dict[str, Any]` mutated by TrainingLoopManager (6 locations), Trainer, and read by DisplayManager. No ownership model. Races in parallel training mode.

### Core RL Engine

**H4. `scaler` Parameter Overloaded for Two Purposes**

`ppo_agent.py:33` -- The `scaler` parameter serves as both a `GradScaler` for mixed precision AND an observation normalizer. Disambiguated by `isinstance` checks. No type annotation.

**Fix:** Split into `grad_scaler: Optional[GradScaler]` and `obs_normalizer: Optional[Callable]`.

**H5. `load_model` Returns Default Dict on Failure Instead of Raising**

`ppo_agent.py:488-533` -- On file-not-found or corrupt checkpoint, returns `{"error": "..."}` with zeroed counters. Callers that don't check the `"error"` key silently resume from random initialization.

**H6. Silent Optimizer Fallback**

`ppo_agent.py:71-79` -- `except Exception` catches ALL errors during optimizer initialization and silently falls back to `lr=1e-3`. A CUDA OOM here is masked.

**H7. `PPOAgent` Hard-Codes `PolicyOutputMapper` Instantiation**

`ppo_agent.py:59` -- Creates `PolicyOutputMapper()` internally (13,527 move mappings) rather than accepting it via DI. Every test pays this cost. Cannot mock.

### Shogi Engine

**H8. sys.path Manipulation at Import Time**

`shogi_game_io.py:34` -- `sys.path.insert(0, ...)` mutates global module resolution. Contradicts the claim of zero outbound dependencies.

**H9. O(N^4) Legal Move Generation with Deep Copies**

`shogi_rules_logic.py:465-556` -- Every candidate move triggers `copy.deepcopy(self.board)` and `copy.deepcopy(self.hands)` via `game.make_move(move_tuple, is_simulation=True)`. This is the dominant performance bottleneck during training.

### Evaluation System

**H10. 500+ Lines of Copy-Paste Across Strategy Files**

`_load_evaluation_entity`, `_get_player_action`, `_validate_and_make_move`, `_game_run_game_loop` -- all duplicated verbatim across `single_opponent.py`, `tournament.py`, `ladder.py`, and `benchmark.py`. Termination constants are also copy-pasted (acknowledged in `ladder.py:31` comment).

**H11. 10x Over-Engineered for Project Maturity**

Evaluation subsystem: 9,845 LOC (28 files). Core RL engine: 965 LOC (7 files). Features built with no current use: background tournaments (537 LOC), enhanced opponents with 5 selection strategies (608 LOC), advanced analytics with scipy (589 LOC), parallel executor (395 LOC), performance SLA manager (284 LOC). All disabled by default. `custom.py` evaluator raises `NotImplementedError`.

**H12. Triple ELO Implementation**

Three separate ELO systems: `opponents/elo_registry.py` (128 lines), `analytics/elo_tracker.py` (234 lines), `strategies/ladder.py:54-97` (inline class). Different K-factors, APIs, and persistence. No single source of truth.

**H13. Fake In-Memory Evaluation**

`tournament.py:800-823` -- `evaluate_step_in_memory` falls back to regular evaluation. ~200 lines across `core_manager.py` and `tournament.py` build infrastructure around a no-op.

### Configuration

**H14. No Cross-Field Config Validation**

`minibatch_size > steps_per_epoch` silently produces zero minibatches. No range validation on `gamma`, `clip_epsilon`, or `entropy_coef`. `lr_schedule_kwargs` not validated against `lr_schedule_type`.

**H15. `webui` Key Missing from Config Loader**

`utils/utils.py:132-141` -- `top_keys` set omits `"webui"`. Override YAML files with only a `webui:` section are silently dropped.

### Cross-Cutting

**H16. CI Pipeline Disabled; Local CI Script Broken**

`.github/workflows/ci.yml.disabled` -- No automated quality gate. `scripts/run_local_ci.sh` references nonexistent `requirements.txt` and `requirements-dev.txt`, ignores mypy results (`|| true`), and deletes security scan results.

---

## Medium-Severity Issues (20)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| M1 | 82 `except Exception` blocks total | Codebase-wide | Error suppression |
| M2 | 55+ `print()` bypassing unified_logger | shogi, eval analytics, callbacks | Inconsistent logging |
| M3 | `run()` and `run_async()` 85% duplicated | training_loop_manager.py:80-277 | Double maintenance |
| M4 | Stale reference caching in TrainingLoopManager | training_loop_manager.py:46-50 | Divergent access paths |
| M5 | Callback errors silently swallowed | callback_manager.py:114-122 | Missing evaluations |
| M6 | `step_manager.handle_episode_end` returns stale state on reset failure | step_manager.py:470-481 | Garbage next episode |
| M7 | EvaluationCallback/AsyncEvaluationCallback duplicated | callbacks.py:86-371 | Double maintenance |
| M8 | `pending_progress_updates.pop(0)` is O(n) | metrics_manager.py:56-57 | Inconsistent with deque used elsewhere |
| M9 | Deprecated `torch.cuda.amp` API | ppo_agent.py:11,322 | Will break on PyTorch upgrade |
| M10 | `ExperienceBuffer.merge_from_parallel_buffers` is O(n) per element | experience_buffer.py:285-311 | Parallel bottleneck |
| M11 | `ExperienceBuffer.get_batch()` returns views, not copies | experience_buffer.py:180-198 | Latent mutation hazard |
| M12 | Broad exception catches in evaluation strategies | single_opponent.py (13), benchmark.py (5) | Error masking |
| M13 | Sync/async method duplication in core_manager | core_manager.py:81-298 | 80 lines duplicated |
| M14 | `opponents.py` in utils/ belongs in evaluation/ | utils/opponents.py | Misplaced domain code |
| M15 | `utils.py` is a grab-bag | utils/utils.py | PolicyOutputMapper, load_config, Logger in one file |
| M16 | `DemoConfig` Optional[None] inconsistent with other configs | config_schema.py:646 | Every consumer checks None |
| M17 | `extra = "forbid"` only on AppConfig, not sub-models | config_schema.py:650 | Nested typos silently ignored |
| M18 | Duplicate `evaluation_interval_timesteps` field | config_schema.py:87 + :206 | Two sources of truth |
| M19 | Dead WebSocket `.pyc` files remain in webui/ | webui/__pycache__/ | Stale artifacts confuse developers |
| M20 | Test fixture duplication across conftest files | tests/conftest.py + tests/integration/conftest.py | 250 lines copy-pasted |

---

## Low-Severity Issues (12)

| # | Issue | Location |
|---|-------|----------|
| L1 | `constants.py` is 60% test constants | constants.py:31-166 |
| L2 | No coverage thresholds configured | pytest.ini / pyproject.toml |
| L3 | `model_type` not validated against known models | config_schema.py:75 |
| L4 | `getattr` used for config fields that have Pydantic defaults | ppo_agent.py:63,89-91 |
| L5 | `__init__.py` exports only 2 of 7 core components | core/__init__.py |
| L6 | Gradient norm computation duplicated in mixed/standard paths | ppo_agent.py:397-418 |
| L7 | `validate_environment` has side effect (resets game) | env_manager.py:192 |
| L8 | Unused `pickle` import in model_sync.py | training/parallel/model_sync.py:10 |
| L9 | Unused production dependencies: Jinja2, requests | pyproject.toml:23-24 |
| L10 | scipy as production dep for one optional file | pyproject.toml:18 |
| L11 | SVG piece images directory does not exist | webui/static/images/ |
| L12 | Zero test markers applied despite 5 defined | pytest.ini vs tests/ |

---

## Strengths (Genuine, Evidence-Based)

| Strength | Evidence |
|----------|----------|
| **Shogi engine domain isolation** | `keisei/shogi/` has zero imports from other keisei subsystems (except `shogi_game_io.py:34` sys.path hack) |
| **Protocol-based model interface** | `core/actor_critic_protocol.py` -- structural typing enables model swapping without inheritance |
| **Atomic WebUI state file** | `webui/state_snapshot.py:173-194` -- `tempfile.mkstemp` + `os.replace` is textbook correct |
| **WebUI process isolation** | `streamlit_app.py` imports NO keisei code -- pure JSON consumer |
| **Checkpoint security** | All `torch.load()` calls use `weights_only=True` -- mitigates pickle deserialization attacks |
| **Pre-allocated experience buffer** | `experience_buffer.py:45-72` -- fixed-size tensors with pointer; no per-step allocation |
| **Pydantic config validation** | 9 typed config sections with field validators; catches errors at startup |
| **Comprehensive test fixtures** | `conftest.py` (655 LOC) with config builders, mock agents, WandB mocks |
| **All tests run on CPU** | `device="cpu"` in all test configs; `CUDA_VISIBLE_DEVICES=""` in E2E |
| **E2E tests use real CLI** | `tests/e2e/conftest.py` runs `train.py` as subprocess -- genuine end-to-end |
| **SchedulerFactory** | Clean stateless factory that raises on invalid input (correct error handling pattern) |
| **Optional integration pattern** | W&B, WebUI, CUDA, torch.compile all disable cleanly without code changes |

---

## Priority Recommendations

| # | Action | Severity | Effort | Justification |
|---|--------|----------|--------|---------------|
| 1 | Atomic checkpoint saves (write-then-rename) | Critical | S | Data loss prevention for long training runs |
| 2 | Delete deprecated `evaluation/core/evaluation_config.py` | Critical | S | Active warnings in production |
| 3 | Fix game engine exception swallowing (shogi_game.py:900,956) | Critical | S | Training data correctness |
| 4 | Add cross-field config validators | High | M | Prevent silent waste of GPU hours |
| 5 | Add `"webui"` to `top_keys` in config loader | High | S | WebUI YAML overrides currently broken |
| 6 | Change WebUI default host to `127.0.0.1` | High | S | Security: information disclosure |
| 7 | Split `scaler` into `grad_scaler` + `obs_normalizer` | High | M | API clarity and type safety |
| 8 | Extract shared game-playing code from strategy files | High | L | Eliminate 500+ lines of copy-paste |
| 9 | Re-enable CI (lint + unit test tier minimum) | High | M | Only quality gate is pre-commit hooks |
| 10 | Apply test markers (auto-mark by directory) | High | S | Enable tiered test execution |
| 11 | Replace print() with unified_logger (55+ instances) | Medium | M | Consistent logging |
| 12 | Remove unused deps (Jinja2, requests) | Medium | S | Attack surface reduction |
| 13 | Evaluate/simplify evaluation subsystem | High | XL | 10x size vs core engine; mostly unused |

---

## Comparison with Prior Assessment

| Dimension | Prior (2026-02-06) | Current (2026-02-08) | Change |
|-----------|-------------------|---------------------|--------|
| Architecture | A- | 2.8/5 | Downgraded -- manager pattern is cosmetic, not real encapsulation |
| Security | A- | 3/5 | Downgraded -- 0.0.0.0 default is not "local use only" |
| Error Handling | B- | 2/5 | Downgraded -- 82 broad catches, game engine swallowing |
| Testing | B | 3/5 | Unchanged -- markers still unused, no coverage thresholds |
| Maintainability | B+ | 2.5/5 | Downgraded -- 500+ lines of copy-paste in evaluation |
| Configuration | (not separately graded) | 3/5 | New -- found missing cross-field validation |
| WebUI | (not separately graded) | 4/5 | New -- well-designed, minor issues only |

The prior assessment was diplomatically softened. The evidence doesn't support A- for architecture when the manager pattern provides zero encapsulation, or A- for security when the default binding exposes the dashboard to the network.

---

## Limitations

- **Not assessed:** Functional correctness of Shogi rules (no reference engine comparison)
- **Not assessed:** Actual test coverage percentages (no `pytest-cov` data)
- **Not assessed:** Runtime performance profiling (all performance claims based on static analysis)
- **Not assessed:** `ParallelManager` thread safety (read structure but not full implementation)
- **Confidence gap:** `constants.py` TEST_* constant count came from one agent's reading; may differ slightly from actual

---

## Assessment Methodology

Six parallel assessment agents conducted independent subsystem reviews:
1. Core RL engine (ppo_agent, experience_buffer, protocols, schedulers)
2. Training system (all 9 managers, display, callbacks)
3. Evaluation system (strategies, analytics, opponents, core infrastructure)
4. Shogi engine + Utilities (game logic, utils, checkpoint, agent loading)
5. Configuration + WebUI (config_schema, Streamlit dashboard)
6. Cross-cutting (error handling, security, testing, dependencies, CI)

Each agent read source files directly and provided file:line evidence for all findings. Severity ratings followed the protocol: Critical = data loss/security risk, High = blocks growth/reliability, Medium = ongoing maintenance burden, Low = quality without immediate impact.
