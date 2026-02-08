# Architecture Quality Assessment

**Project:** Keisei Deep Reinforcement Learning Shogi System
**Assessed:** 2026-02-08
**Assessor:** architecture-critic
**Branch:** webui (commit 7d76082)
**Codebase:** ~87 source files, ~21K LOC (non-blank)

## Executive Summary

The Keisei codebase is a **competent but uneven** implementation of a DRL Shogi system. The 9-manager training architecture is well-intentioned but suffers from **tight coupling through the Trainer hub**, **over-engineered evaluation subsystem (39% of LOC)**, and **no active CI pipeline**. The core RL and Shogi engine components are solid. There are 0 critical security issues and 1 confirmed correctness bug (inverted `torch.no_grad()` logic in PPOAgent). The test suite covers 52% of modules, with the evaluation and WebUI subsystems largely untested.

**Quality Score: 5.5 / 10**

| Category | Score | Critical | High | Medium |
|----------|-------|----------|------|--------|
| Training Subsystem | 6/10 | 0 | 2 | 3 |
| Core RL | 7/10 | 0 | 1 | 2 |
| Shogi Engine | 8/10 | 0 | 0 | 3 |
| Evaluation | 4/10 | 0 | 1 | 2 |
| Config | 6/10 | 0 | 1 | 3 |
| WebUI | 6/10 | 0 | 0 | 3 |
| Testing & CI | 4/10 | 0 | 2 | 2 |

---

## Subsystem Assessments

### 1. Training Subsystem

**Quality Score:** 6 / 10
**Critical Issues:** 0
**High Issues:** 2

**Findings:**

1. **TrainingLoopManager is tightly coupled to Trainer** - High
   - **Evidence:** `training_loop_manager.py` contains 69 instances of `self.trainer.*` access, directly reaching into `trainer.metrics_manager`, `trainer.agent`, `trainer.callback_manager`, `trainer.display`, `trainer.webui_manager`, and more.
   - **Impact:** TrainingLoopManager cannot be tested in isolation. It functions as a pseudo-Trainer rather than an independent manager. The "manager-based architecture" claim is partially undermined.
   - **Recommendation:** Pass managers directly to TrainingLoopManager constructor instead of passing the Trainer instance. Use the cached references (`self.agent`, `self.buffer`, etc.) consistently instead of reaching through `self.trainer.*`.

2. **Callbacks require full Trainer reference** - High
   - **Evidence:** `callbacks.py` functions receive the entire Trainer and access arbitrary attributes (`trainer.agent`, `trainer.metrics_manager.get_final_stats()`, `trainer.model_manager.save_checkpoint()`, `trainer.evaluation_manager.opponent_pool.add_checkpoint()`). `callback_manager.py:110-127` passes full Trainer to every callback.
   - **Impact:** Callbacks are unencapsulated — any callback can do anything to any manager. This makes reasoning about callback side effects impossible.
   - **Recommendation:** Define a narrow callback context object with only the fields callbacks actually need.

3. **MetricsManager has 34 methods — scope creep** - Medium
   - **Evidence:** `metrics_manager.py` (477 lines) handles game statistics, PPO metrics formatting, episode logging, Elo rating management, opening move history, square usage tracking, async update queuing, and processing state management.
   - **Impact:** Maintenance burden. Changes to any metric type require modifying this god-object.
   - **Recommendation:** Extract formatting into DisplayManager, keep MetricsManager focused on accumulation.

4. **SessionManager handles evaluation logging** - Medium
   - **Evidence:** `session_manager.py:202-343` contains `log_evaluation_metrics()`, `log_evaluation_performance()`, `log_evaluation_sla_status()` — 140 lines of evaluation concerns in a session-management class.
   - **Impact:** Violates single responsibility. Evaluation logging belongs in the evaluation subsystem.
   - **Recommendation:** Move evaluation logging methods to the evaluation subsystem.

5. **ModelManager contains W&B artifact retry logic** - Medium
   - **Evidence:** `model_manager.py:398-522` handles artifact creation with exponential backoff retry logic — infrastructure concerns mixed with model lifecycle.
   - **Impact:** Model changes risk breaking artifact logic and vice versa.
   - **Recommendation:** Extract W&B artifact management to SessionManager or a dedicated integration layer.

**Strengths:**
- Trainer itself is well-refactored: only 459 lines and 8 methods, focused on orchestration.
- Checkpoint resume flow is correct: global_timestep, episode counts, Elo ratings, and PPO trend history all properly restored through `MetricsManager.restore_from_checkpoint()`.
- Error handling in StepManager is robust: double-failure recovery with fallback observations, specific exception types, proper logging.
- No circular import issues: TYPE_CHECKING used consistently.
- No dead code detected in training subsystem.

---

### 2. Core RL Subsystem

**Quality Score:** 7 / 10
**Critical Issues:** 0
**High Issues:** 1

**Findings:**

1. **Inverted `torch.no_grad()` logic in PPOAgent** - High
   - **Evidence:** `ppo_agent.py:175-195` — when `is_training=True`, the code wraps the forward pass in `torch.no_grad()`. When `is_training=False`, it does NOT use `no_grad()`. This is semantically backwards.
   - **Impact:** During self-play experience collection (`is_training=True`), gradients are disabled. During inference/evaluation (`is_training=False`), gradients are computed unnecessarily. No functional impact (forward pass doesn't require gradients for action selection), but wastes memory during evaluation and is misleading.
   - **Recommendation:** Invert the condition: use `no_grad()` when `is_training=False`.

2. **Observation normalization infrastructure is dead code** - Medium
   - **Evidence:** `ppo_agent.py:155-160, 231-235, 310-315` contain scaler application code, but the scaler is always a `GradScaler` (for mixed precision). No `StandardScaler`, `RunningMeanStd`, or observation normalizer exists anywhere in the codebase. The non-GradScaler code path is never exercised.
   - **Impact:** Misleading code suggests observation normalization is supported when it isn't. Future developers may assume this works.
   - **Recommendation:** Either implement a real observation normalizer or remove the dead infrastructure.

3. **Legal masks are memory-inefficient in ExperienceBuffer** - Medium
   - **Evidence:** `experience_buffer.py:64-67` allocates `buffer_size x 13,527` boolean tensor. For `steps_per_epoch=2048`, this is ~27.7 MB per buffer. Legal masks are typically very sparse (~200 of 13,527 entries are True per position).
   - **Impact:** Memory waste. Not catastrophic for single-GPU training but compounds with larger buffers or multi-agent setups.
   - **Recommendation:** Consider sparse tensor representation or on-the-fly regeneration during training.

**Strengths:**
- Value function clipping IS properly implemented (contrary to CODE_MAP's claim). `ppo_agent.py:356-368` correctly implements clipped value loss with `torch.max()`.
- GAE computation in ExperienceBuffer is correct: proper reverse iteration, episode boundary masking, gamma/lambda accumulation.
- `ActorCriticProtocol` is well-defined (lines 13-90) with comprehensive method coverage.
- `BaseActorCriticModel` eliminates code duplication between CNN and ResNet — good DRY compliance.
- Empty batch handling is explicit and safe (returns zeros metrics dict, not None).
- Scheduler factory supports 4 types with proper input validation and fallbacks.

---

### 3. Shogi Engine

**Quality Score:** 8 / 10
**Critical Issues:** 0
**High Issues:** 0

**Findings:**

1. **Board scanning performance** - Medium
   - **Evidence:** `shogi_rules_logic.py` uses 16 nested `range(9)` loops. `find_king()` (lines 30-34) is O(81) and called for every `is_in_check()`. `generate_all_legal_moves()` (lines 500-576) is O(6561) iterations. `check_if_square_is_attacked()` (lines 250-276) scans all 81 squares for each check.
   - **Impact:** During training, `generate_all_legal_moves()` is called every step. With ~200 legal moves per position and 500 moves per game, this generates millions of board scans per game.
   - **Recommendation:** Cache king positions (invalidate on move). Use bitboard representation for attack detection. These are standard Shogi engine optimizations.

2. **Unbounded move_history growth** - Medium
   - **Evidence:** `shogi_game.py:49, 653` — `move_history` is a plain list appended to every move. Each entry stores the move, captured piece, promotion flags, and state hash. For simulation moves, it also stores `copy.deepcopy(self.board)` and `copy.deepcopy(self.hands)` (lines 635-636).
   - **Impact:** For a 500-move game, approximately 7.5 MB per game history. The deepcopy on every simulation move is the main cost.
   - **Recommendation:** For sennichite checking, only store state hashes in a separate deque. Keep full history only when needed (game recording mode).

3. **Magic number `range(9)` used 16 times instead of `SHOGI_BOARD_SIZE`** - Medium
   - **Evidence:** `constants.py:9` defines `SHOGI_BOARD_SIZE = 9`, but `shogi_rules_logic.py`, `shogi_game.py`, and `shogi_game_io.py` all use `range(9)` directly.
   - **Impact:** Minor maintainability issue. If board size ever changed (hypothetically), 16 locations would need updating.
   - **Recommendation:** Replace with the constant.

**Strengths:**
- Module decomposition is excellent: 7 modules with clear separation (core definitions, game state, rules logic, move execution, I/O, features, engine facade).
- All complex Shogi rules are correctly implemented: uchi-fu-zume with recursion protection, nifu detection, mandatory promotions, drop restrictions.
- No active debug prints (all commented out).
- Comprehensive move validation (`_validate_move_tuple_format()`).
- Sennichite (repetition) detection uses efficient state hashing.
- Error handling raises `ValueError` with SFEN dump on missing king — excellent for debugging.
- No circular dependencies.

---

### 4. Evaluation Subsystem

**Quality Score:** 4 / 10
**Critical Issues:** 0
**High Issues:** 1

**Findings:**

1. **4 of 5 evaluation strategies are dead code** - High
   - **Evidence:** Only `single_opponent` strategy is used in the training pipeline. `train.py` only checks `if eval_config.strategy == "single_opponent"`. `config_schema.py` defaults to `"single_opponent"`. `core_manager.py` creates `strategy="single_opponent"`.
   - Tournament (830 LOC), Ladder (739 LOC), Benchmark (753 LOC), and Custom (391 LOC) strategies are fully implemented, auto-register at import time, but are **never instantiated** in any production workflow.
   - **Impact:** 2,773 lines of dead code. The evaluation subsystem is 9,909 LOC (39.3% of the codebase), but only ~25% of it is actively used. This is the single largest maintenance burden in the project.
   - **Recommendation:** Either promote these strategies to first-class features with tests and documentation, or deprecate them behind a feature flag and plan for removal.

2. **86% of evaluation modules have zero test coverage** - Medium
   - **Evidence:** Of 22 evaluation modules, only 3 have dedicated tests (`test_single_opponent.py`, `test_tournament.py`, `test_ladder.py`). 14 modules have zero coverage: all analytics, all opponents (elo_registry, enhanced_manager, opponent_pool), core evaluators (background_tournament, base_evaluator, evaluation_config, evaluation_context, evaluation_result, parallel_executor), and 2 strategies (benchmark, custom).
   - **Impact:** No confidence in correctness of analytics, opponent management, or advanced strategies.
   - **Recommendation:** If keeping the evaluation infrastructure, add tests. If deprecating, remove it.

3. **One print() in error path instead of unified_logger** - Medium
   - **Evidence:** `core_manager.py:365` uses `print()` for an error fallback message instead of `unified_logger.log_error_to_stderr()`.
   - **Impact:** Inconsistent logging; message may be lost or formatted differently from other errors.
   - **Recommendation:** Replace with unified_logger call.

**Strengths:**
- Factory pattern is professionally implemented (`base_evaluator.py:378-421`).
- Legal mask implementation is correct — starts with zeros, sets True only for actual legal moves, hard-crashes on unmapped moves (no silent corruption).
- Action count (13,527) is verified consistent across all 7 reference points in the codebase.
- Circular dependency between core and utils is properly mitigated with lazy imports in `agent_loading.py`.

---

### 5. Configuration Subsystem

**Quality Score:** 6 / 10
**Critical Issues:** 0
**High Issues:** 1

**Findings:**

1. **No cross-field validation in AppConfig** - High
   - **Evidence:** `config_schema.py` has 21 field-level validators but zero `@model_validator` for cross-field checks. Missing checks include:
     - Device availability (can set `device="cuda"` on CPU-only machine — fails at runtime)
     - Evaluation interval alignment with `steps_per_epoch`
     - Model type vs. architecture-specific parameters (e.g., `se_ratio` is meaningless for non-ResNet)
     - `max_moves_per_game` consistency between env and evaluation configs
   - **Impact:** Users discover configuration errors at runtime, not at config load time. Some errors are cryptic (CUDA not available) or silent (evaluation triggers at irregular intervals).
   - **Recommendation:** Add `@model_validator(mode='after')` to AppConfig for critical cross-field checks.

2. **DemoConfig is completely unused** - Medium
   - **Evidence:** `config_schema.py:582-589` defines `DemoConfig` with 2 fields. `AppConfig` includes it as `demo: Optional[DemoConfig] = None`. But zero references to DemoConfig fields exist in training code. Demo functionality actually lives in `DisplayConfig.display_moves` and `DisplayConfig.turn_tick`.
   - **Impact:** Confusing for users who see two config sections for the same feature. No section in `default_config.yaml` for demo.
   - **Recommendation:** Remove DemoConfig and consolidate into DisplayConfig.

3. **Legacy WebUI config fields are unused** - Medium
   - **Evidence:** `config_schema.py:627-636` defines `max_connections`, `board_update_rate_hz`, and `metrics_update_rate_hz` in WebUIConfig with comments "Legacy fields kept for YAML backward compatibility." These fields are loaded from YAML but never read by any code.
   - **Impact:** Misleading configuration — users may think these fields do something.
   - **Recommendation:** Remove or add deprecation warnings.

4. **torch.compile enabled by default** - Medium
   - **Evidence:** `config_schema.py:119-122` sets `enable_torch_compile=True` by default with `enable_compilation_fallback=True`. Compilation silently falls back to eager mode on unsupported systems.
   - **Impact:** Users get unpredictable performance — sometimes 10-30% faster, sometimes not, with no clear indication of which mode is active.
   - **Recommendation:** Default to `False` for predictable behavior, or log clearly when fallback occurs.

**Strengths:**
- Pydantic configuration with 9 well-organized config classes.
- YAML loading with CLI overrides works correctly.
- WebUI defaults are secure (host=localhost, enabled=false).
- Sensible PPO hyperparameter defaults (lr=3e-4, clip_epsilon=0.2, gamma=0.99).

---

### 6. WebUI Subsystem

**Quality Score:** 6 / 10
**Critical Issues:** 0
**High Issues:** 0

**Findings:**

1. **No process liveness monitoring** - Medium
   - **Evidence:** `streamlit_manager.py` starts a subprocess with `Popen()` but never polls it to check if it's still running. If Streamlit crashes, the main training loop has no way to detect the failure.
   - **Impact:** User believes dashboard is running but it's dead. Training continues normally but monitoring is lost.
   - **Recommendation:** Add periodic `poll()` check in `update_progress()` with re-launch or warning.

2. **KeyboardInterrupt may orphan Streamlit subprocess** - Medium
   - **Evidence:** `train.py:414-415` catches `KeyboardInterrupt` and calls `sys.exit(1)` without explicitly cleaning up the Streamlit subprocess. No `atexit` handler or `finally` block guarantees cleanup.
   - **Impact:** Orphaned Streamlit process continues running after training stops, holding the port.
   - **Recommendation:** Add `atexit.register(trainer.webui_manager.stop)` or use `try/finally`.

3. **Missing SVG assets for board rendering** - Medium
   - **Evidence:** `streamlit_app.py:25` references `static/images/*.svg` but the directory doesn't exist. The code silently falls back to text labels (line 33: `if not _IMAGES_DIR.exists(): return`).
   - **Impact:** Board rendering uses text instead of visual pieces. This is a UX degradation, not a bug.
   - **Recommendation:** Either commit SVG assets or document that text rendering is the intended mode.

**Strengths:**
- Clean subprocess architecture: Streamlit app imports zero keisei code, communicating only via atomic JSON state file.
- Atomic write pattern is correct: `tempfile.mkstemp()` + `os.replace()` is POSIX-atomic.
- Rate limiting prevents excessive I/O (`_min_write_interval = 1.0 / config.update_rate_hz`).
- Graceful degradation: if Streamlit import fails, training continues without WebUI.
- Secure defaults: host=localhost, not 0.0.0.0.

---

### 7. Testing & CI

**Quality Score:** 4 / 10
**Critical Issues:** 0
**High Issues:** 2

**Findings:**

1. **CI pipeline is completely disabled** - High
   - **Evidence:** `.github/workflows/ci.yml.disabled` — the main CI pipeline (flake8, mypy, pytest, bandit, coverage) is disabled. Only active workflows are Claude Code integration and Claude Code Review (both AI-based, not traditional CI).
   - **Impact:** No automated quality gates on push or PR. Code quality regressions can be merged without detection. The local CI script (`scripts/run_local_ci.sh`) exists but requires manual execution.
   - **Recommendation:** Re-enable CI. At minimum: flake8 critical errors + pytest unit tests on every PR.

2. **Test markers defined but never applied** - High
   - **Evidence:** `pyproject.toml:83-89` defines 5 markers (unit, integration, slow, performance, e2e). Zero test functions use `@pytest.mark.unit` or `@pytest.mark.integration`. Only `@pytest.mark.parametrize` and `@pytest.mark.asyncio` are used.
   - **Impact:** `pytest -m unit` returns 0 tests. Directory-based selection (`pytest tests/unit/`) works but markers would enable finer-grained control.
   - **Recommendation:** Apply markers to all test functions, ideally via conftest.py `pytest_collection_modifyitems` hook.

3. **52% of modules have no dedicated test files** - Medium
   - **Evidence:** 23 of ~71 source modules have zero test coverage. Major gaps: evaluation subsystem (14/22 untested), WebUI (3/3 untested), utils (6/10 untested including `unified_logger.py` and `utils.py` with PolicyOutputMapper).
   - **Impact:** No confidence in correctness of large swaths of code.
   - **Recommendation:** Prioritize tests for: unified_logger, PolicyOutputMapper, WebUI state snapshot, evaluation strategies that are kept.

4. **No coverage reporting in CI** - Medium
   - **Evidence:** Coverage is generated locally by `run_local_ci.sh` but never uploaded or gated in CI (CI is disabled). No minimum coverage threshold is configured.
   - **Impact:** Coverage can regress without detection.
   - **Recommendation:** When CI is re-enabled, add coverage gating (suggest >=60% as initial target).

**Strengths:**
- 1,172 test functions across 58 test files — substantial investment in testing.
- No empty or vacuous tests — all contain meaningful assertions.
- Well-organized fixture hierarchy with session-scoped PolicyOutputMapper caching (avoids rebuilding 13,527 moves per test).
- Integration tests verify real multi-component chains (training step, epoch, checkpoint resume).
- Mock usage is appropriate and limited to external boundaries (filesystem, W&B, torch.device).

---

## Cross-Cutting Concerns

### Security
**Rating: ACCEPTABLE for a research/training tool**

- No SQL, no user-facing web input, no authentication required for core functionality.
- WebUI defaults to localhost binding (not 0.0.0.0) — correct for local use.
- WebUI has zero authentication — acceptable for localhost, problematic if exposed to network.
- No secrets in repository (`.env` pattern used for W&B API keys).
- `bandit` security scanner is configured but CI is disabled.
- Subprocess management uses explicit arguments (no `shell=True`).

### Performance
**Rating: ADEQUATE with known bottlenecks**

- Shogi engine uses pure Python with O(81) board scans and `copy.deepcopy()` on every simulation move. Standard for a research project but would need bitboard optimization for production-scale training speed.
- Legal mask memory (~28 MB) is acceptable for single-GPU but scales poorly.
- `torch.no_grad()` inversion wastes memory during evaluation (low impact in practice).
- ExperienceBuffer GAE computation is correct and efficient.
- `torch.compile` enabled by default with silent fallback — unpredictable performance characteristics.

### Maintainability
**Rating: UNEVEN**

- Training managers are well-separated in principle but tightly coupled through Trainer in practice.
- Evaluation subsystem is 39% of LOC but only ~25% is actively used — significant dead code burden.
- Configuration system is well-structured with Pydantic but lacks cross-field validation.
- Logging is mostly unified but 45 `print()` calls remain (most are intentional for profiling/display, 1 should be fixed).
- Documentation (DESIGN.md, CODE_MAP.md) is comprehensive but contains some inaccuracies (e.g., claims value clipping is not implemented when it is, and reports Rich TUI when actual display is simple stderr logging).

---

## Priority Recommendations

1. **Re-enable CI pipeline** - High
   - Why: No automated quality gates means regressions can be merged unchecked.
   - Effort: S (rename ci.yml.disabled to ci.yml, update Python version to 3.13)

2. **Fix inverted `torch.no_grad()` in PPOAgent** - High
   - Why: Semantically wrong; wastes memory during evaluation; confuses future maintainers.
   - Effort: S (one-line condition inversion at `ppo_agent.py:175`)

3. **Add cross-field validation to AppConfig** - High
   - Why: Users discover configuration errors at runtime, not load time. Some are silent.
   - Effort: M (add `@model_validator` with 4-5 checks)

4. **Decide on evaluation strategies: keep or deprecate** - High
   - Why: 2,773 LOC of dead code (tournament/ladder/benchmark/custom) costs maintenance effort. Either test and document them, or remove them.
   - Effort: M (removal) or L (full test coverage + documentation)

5. **Decouple TrainingLoopManager from Trainer** - Medium
   - Why: 69 `self.trainer.*` accesses prevent isolated testing and violate the architecture's own manager-separation claims.
   - Effort: M (refactor constructor to accept managers directly)

6. **Apply test markers or auto-apply via conftest** - Medium
   - Why: `pytest -m unit` doesn't work despite markers being defined.
   - Effort: S (add `pytest_collection_modifyitems` hook)

7. **Remove DemoConfig** - Medium
   - Why: Unused; confuses users; demo functionality lives in DisplayConfig.
   - Effort: S

8. **Add Streamlit subprocess health monitoring** - Medium
   - Why: If Streamlit crashes, user has no indication.
   - Effort: S (add `poll()` check in `update_progress()`)

---

## Limitations

- **No runtime profiling performed.** Performance assessments are based on code analysis, not actual benchmarks. The Shogi engine bottlenecks may or may not be significant depending on training throughput targets.
- **Test execution not performed.** Test quality assessment is based on code inspection, not actually running the test suite.
- **Security assessment is surface-level.** No penetration testing or dependency vulnerability scanning performed (bandit is configured but CI is disabled).
- **Multi-GPU / distributed training paths not assessed.** DDP support is claimed but not verified.
- **WebUI Streamlit app not loaded.** UI quality and functionality not visually assessed.

---

## Appendix: LOC by Subsystem

| Subsystem | LOC | % of Total | Assessment |
|-----------|-----|------------|------------|
| Evaluation | 9,909 | 39.3% | Over-engineered; 75% unused |
| Training | 4,500 | 17.8% | Well-structured, tightly coupled |
| Shogi | 3,164 | 12.5% | Solid; needs perf optimization |
| Core | 2,200 | 8.7% | Good quality; minor issues |
| Utils | 2,403 | 9.5% | Well-organized |
| Config | 650 | 2.6% | Needs cross-validation |
| WebUI | 727 | 2.9% | Clean architecture |
| Other | 1,700 | 6.7% | Constants, entry points, parallel |
| **Total** | **~25,253** | **100%** | |

## Appendix: File Size Ranking (Largest Managers)

| File | Lines | Methods | Concern |
|------|-------|---------|---------|
| model_manager.py | 768 | 20 | W&B artifact logic mixed in |
| training_loop_manager.py | 671 | 16 | 69 Trainer accesses |
| step_manager.py | 575 | 12 | Acceptable — focused |
| metrics_manager.py | 477 | 34 | Too many responsibilities |
| session_manager.py | 472 | 20 | Eval logging mixed in |
| trainer.py | 459 | 8 | Well-refactored |
| callbacks.py | 363 | - | Needs narrow context object |
| callback_manager.py | 267 | - | Acceptable |
| setup_manager.py | 207 | 8 | Some unnecessary indirection |
| env_manager.py | 216 | 8 | Good |
| display_manager.py | 101 | 11 | Good |
