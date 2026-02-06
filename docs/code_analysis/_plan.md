# Deep Code Analysis Plan

**Date:** 2026-02-07
**Scope:** `keisei/` — 87 Python source files, 27,222 total lines
**Lead:** Technical Lead (Claude Opus)
**Engineers:** Senior Engineer subagents (Opus)

---

## Scope Summary

| Subsystem | Files | Lines | Packages |
|-----------|-------|-------|----------|
| Config/Root | 3 | 907 | 1 |
| Core | 7 | 1,256 | 2 |
| Shogi | 8 | 3,449 | 3 |
| Training (main) | 20 | 8,413 | 7 |
| Training (parallel) | 6 | 1,325 | 1 |
| Evaluation | 24 | 9,444 | 8 |
| Utils | 10 | 2,463 | 2 |
| WebUI | 3 | 823 | 1 |
| **Total** | **87** | **27,222** | **25** |

---

## Risk Assessment (Pre-Analysis)

**HIGH RISK** (complex logic, data integrity, concurrency):
- Package 3: Core PPO Engine — heart of the RL algorithm
- Package 5: Shogi Game & IO — 1,797 lines of game state management
- Package 6: Shogi Rules — correctness-critical game logic
- Package 9: Step Manager & Metrics — training/game boundary
- Package 10: Training Model Manager — checkpoints, mixed precision
- Package 12: Training Orchestration — complex state coordination
- Package 14: Parallel System — concurrency, race conditions, IPC
- Package 25: WebUI — external-facing, WebSocket security

**MEDIUM RISK** (substantial logic, potential edge cases):
- Package 1: Config Schema — validation, defaults, type coercion
- Package 8: Callbacks & Environment — lifecycle management
- Package 11: Session Manager — file I/O, directory management
- Package 16-17: Evaluation Core — concurrent evaluation, model loading
- Package 21: Tournament & Ladder strategies — complex game orchestration

**LOWER RISK** (simpler logic, support code):
- Package 2: Core Interfaces — protocol definitions
- Package 4: Shogi Definitions — constants and features
- Package 7: Display System — UI rendering (no data integrity)
- Package 13: Training Support — small utility files
- Package 15: Evaluation Types — data classes
- Packages 18-20, 22-24: Evaluation support, utils

---

## Package Breakdown

### Package 1: Configuration & Root (907 lines, 3 files)
- `keisei/__init__.py` (30)
- `keisei/config_schema.py` (694)
- `keisei/constants.py` (183)

**Rationale:** Configuration is the contract that all other subsystems depend on. Validation bugs here propagate everywhere.

### Package 2: Core — Interfaces & Models (308 lines, 4 files)
- `keisei/core/__init__.py` (6)
- `keisei/core/actor_critic_protocol.py` (89)
- `keisei/core/base_actor_critic.py` (184)
- `keisei/core/neural_network.py` (29)

**Rationale:** Protocol definitions and base classes that all models must implement. Contract correctness is critical.

### Package 3: Core — PPO Engine (948 lines, 3 files) [HIGH RISK]
- `keisei/core/experience_buffer.py` (301)
- `keisei/core/ppo_agent.py` (537)
- `keisei/core/scheduler_factory.py` (110)

**Rationale:** The RL algorithm core. Numerical correctness in PPO updates and experience storage directly affects training quality.

### Package 4: Shogi — Definitions & Features (736 lines, 4 files)
- `keisei/shogi/__init__.py` (23)
- `keisei/shogi/shogi_engine.py` (11)
- `keisei/shogi/features.py` (193)
- `keisei/shogi/shogi_core_definitions.py` (509)

**Rationale:** Foundation types and feature extraction. Errors in piece/move definitions would silently corrupt all training.

### Package 5: Shogi — Game & IO (1,797 lines, 2 files) [HIGH RISK]
- `keisei/shogi/shogi_game.py` (967)
- `keisei/shogi/shogi_game_io.py` (830)

**Rationale:** Central game state management and serialization. The largest Shogi files — complex state transitions.

### Package 6: Shogi — Rules & Move Execution (916 lines, 2 files) [HIGH RISK]
- `keisei/shogi/shogi_rules_logic.py` (695)
- `keisei/shogi/shogi_move_execution.py` (221)

**Rationale:** Rule enforcement and move execution. Bugs here mean illegal game states during training — silent corruption.

### Package 7: Training — Display System (1,472 lines, 4 files)
- `keisei/training/display.py` (628)
- `keisei/training/display_components.py` (610)
- `keisei/training/display_manager.py` (177)
- `keisei/training/adaptive_display.py` (57)

**Rationale:** Rich console output. Lower data-integrity risk but large code surface. Known issue: 16+ long functions.

### Package 8: Training — Events & Environment (881 lines, 3 files)
- `keisei/training/callback_manager.py` (262)
- `keisei/training/callbacks.py` (358)
- `keisei/training/env_manager.py` (261)

**Rationale:** Event system and game environment lifecycle. Callback ordering and environment cleanup are subtle.

### Package 9: Training — Step Manager & Metrics (1,318 lines, 3 files) [HIGH RISK]
- `keisei/training/step_manager.py` (644)
- `keisei/training/metrics_manager.py` (442)
- `keisei/training/utils.py` (232)

**Rationale:** Step execution is the training/game boundary. Known issue: step_manager has 250-line function.

### Package 10: Training — Model Manager (765 lines, 1 file) [HIGH RISK]
- `keisei/training/model_manager.py` (765)

**Rationale:** Checkpoint save/load, mixed precision, model creation. Data loss risk from checkpoint corruption.

### Package 11: Training — Session & Setup (708 lines, 2 files)
- `keisei/training/session_manager.py` (498)
- `keisei/training/setup_manager.py` (210)

**Rationale:** Session lifecycle, directory creation, W&B integration. File I/O edge cases.

### Package 12: Training — Orchestration (1,605 lines, 4 files) [HIGH RISK]
- `keisei/training/__init__.py` (1)
- `keisei/training/trainer.py` (488)
- `keisei/training/training_loop_manager.py` (693)
- `keisei/training/train.py` (423)

**Rationale:** The conductor. Coordinates all 9 managers. Complex initialization ordering and error propagation.

### Package 13: Training — Models & Support (292 lines, 5 files)
- `keisei/training/models/__init__.py` (31)
- `keisei/training/models/resnet_tower.py` (84)
- `keisei/training/elo_rating.py` (65)
- `keisei/training/previous_model_selector.py` (28)
- `keisei/training/train_wandb_sweep.py` (84)

**Rationale:** Small support files. Model factory registration and Elo computation.

### Package 14: Training — Parallel System (1,325 lines, 6 files) [HIGH RISK]
- `keisei/training/parallel/__init__.py` (46)
- `keisei/training/parallel/communication.py` (236)
- `keisei/training/parallel/model_sync.py` (181)
- `keisei/training/parallel/parallel_manager.py` (345)
- `keisei/training/parallel/self_play_worker.py` (463)
- `keisei/training/parallel/utils.py` (54)

**Rationale:** Multi-process self-play. Race conditions, IPC reliability, model sync correctness.
*Note: 6 files (one over limit) — two are trivial init/utils files.*

### Package 15: Evaluation — Core Types (700 lines, 5 files)
- `keisei/evaluation/__init__.py` (14)
- `keisei/evaluation/core/__init__.py` (124)
- `keisei/evaluation/core/evaluation_config.py` (41)
- `keisei/evaluation/core/evaluation_context.py` (125)
- `keisei/evaluation/core/evaluation_result.py` (396)

**Rationale:** Type definitions and configuration for the evaluation subsystem.

### Package 16: Evaluation — Core Evaluators (1,396 lines, 3 files)
- `keisei/evaluation/core/base_evaluator.py` (458)
- `keisei/evaluation/core/background_tournament.py` (537)
- `keisei/evaluation/core/parallel_executor.py` (401)

**Rationale:** Base evaluation logic and parallel execution. Concurrency patterns.

### Package 17: Evaluation — Managers (1,723 lines, 4 files)
- `keisei/evaluation/core/model_manager.py` (540)
- `keisei/evaluation/core_manager.py` (473)
- `keisei/evaluation/enhanced_manager.py` (396)
- `keisei/evaluation/performance_manager.py` (314)

**Rationale:** Evaluation orchestration and model loading. Multiple manager layers suggest potential complexity.

### Package 18: Evaluation — Opponents (853 lines, 5 files)
- `keisei/evaluation/utils/__init__.py` (9)
- `keisei/evaluation/opponents/__init__.py` (19)
- `keisei/evaluation/opponents/elo_registry.py` (131)
- `keisei/evaluation/opponents/enhanced_manager.py` (608)
- `keisei/evaluation/opponents/opponent_pool.py` (86)

**Rationale:** Opponent management and Elo tracking. Enhanced manager is substantial at 608 lines.

### Package 19: Evaluation — Analytics (1,632 lines, 5 files)
- `keisei/evaluation/analytics/__init__.py` (38)
- `keisei/evaluation/analytics/advanced_analytics.py` (589)
- `keisei/evaluation/analytics/elo_tracker.py` (234)
- `keisei/evaluation/analytics/performance_analyzer.py` (391)
- `keisei/evaluation/analytics/report_generator.py` (380)

**Rationale:** Statistical analysis and reporting. Numerical correctness in Elo/analytics calculations.

### Package 20: Evaluation — Strategies: Single & Custom (1,332 lines, 3 files)
- `keisei/evaluation/strategies/__init__.py` (24)
- `keisei/evaluation/strategies/single_opponent.py` (894)
- `keisei/evaluation/strategies/custom.py` (414)

**Rationale:** Single opponent is the largest strategy at 894 lines. Custom strategy has flexible evaluation logic.

### Package 21: Evaluation — Strategies: Tournament & Ladder (1,568 lines, 2 files)
- `keisei/evaluation/strategies/tournament.py` (830)
- `keisei/evaluation/strategies/ladder.py` (738)

**Rationale:** Complex multi-game orchestration with bracket/progression logic.

### Package 22: Evaluation — Strategies: Benchmark (753 lines, 1 file)
- `keisei/evaluation/strategies/benchmark.py` (753)

**Rationale:** Standalone benchmarking strategy. Large single file.

### Package 23: Utils — Core Utilities (1,074 lines, 5 files)
- `keisei/utils/__init__.py` (28)
- `keisei/utils/agent_loading.py` (216)
- `keisei/utils/checkpoint.py` (53)
- `keisei/utils/unified_logger.py` (195)
- `keisei/utils/utils.py` (582)

**Rationale:** Cross-cutting utilities. agent_loading.py has known circular dependency mitigation. utils.py is large.

### Package 24: Utils — Performance & Validation (1,390 lines, 5 files)
- `keisei/utils/compilation_validator.py` (398)
- `keisei/utils/move_formatting.py` (154)
- `keisei/utils/opponents.py` (90)
- `keisei/utils/performance_benchmarker.py` (427)
- `keisei/utils/profiling.py` (321)

**Rationale:** Performance tooling and validation utilities.

### Package 25: WebUI (823 lines, 3 files) [HIGH RISK]
- `keisei/webui/__init__.py` (2)
- `keisei/webui/web_server.py` (116)
- `keisei/webui/webui_manager.py` (705)

**Rationale:** External-facing WebSocket + HTTP server. Security surface, resource management.

---

## Dispatch Order

Priority order balances risk with dependency understanding:

1. **Packages 1-3** (Config, Core) — Foundation understanding
2. **Packages 4-6** (Shogi) — Game engine correctness
3. **Packages 7-14** (Training) — Training infrastructure
4. **Packages 15-22** (Evaluation) — Evaluation subsystem
5. **Packages 23-25** (Utils, WebUI) — Cross-cutting + external surface

---

## Tracking

| Package | Status | Engineer | Findings |
|---------|--------|----------|----------|
| 1 | COMPLETE | Opus | 3 files, 0 CRITICAL, 3 NEEDS_ATTENTION |
| 2 | COMPLETE | Opus | 4 files, 2 CRITICAL, 1 NEEDS_ATTENTION |
| 3 | COMPLETE | Opus | 3 files, 0 CRITICAL, 2 NEEDS_ATTENTION |
| 4 | COMPLETE | Opus | 4 files, 1 CRITICAL, 1 NEEDS_ATTENTION |
| 5 | COMPLETE | Opus | 2 files, 0 CRITICAL, 2 NEEDS_ATTENTION |
| 6 | COMPLETE | Opus | 2 files, 0 CRITICAL, 1 NEEDS_ATTENTION |
| 7 | COMPLETE | Opus | 4 files, 0 CRITICAL, 2 NEEDS_ATTENTION |
| 8 | COMPLETE | Opus | 3 files, 1 CRITICAL, 1 NEEDS_ATTENTION |
| 9 | COMPLETE | Opus | 3 files, 0 CRITICAL, 3 NEEDS_ATTENTION |
| 10 | COMPLETE | Opus | 1 file, 0 CRITICAL, 1 NEEDS_ATTENTION |
| 11 | COMPLETE | Opus | 2 files, 0 CRITICAL, 2 NEEDS_ATTENTION |
| 12 | COMPLETE | Opus | 4 files, 0 CRITICAL, 3 NEEDS_ATTENTION |
| 13 | COMPLETE | Opus | 5 files, 0 CRITICAL, 0 NEEDS_ATTENTION |
| 14 | COMPLETE | Opus | 6 files, 0 CRITICAL, 6 NEEDS_ATTENTION |
| 15 | COMPLETE | Opus | 5 files, 1 CRITICAL, 1 NEEDS_ATTENTION |
| 16 | COMPLETE | Opus | 3 files, 0 CRITICAL, 3 NEEDS_ATTENTION |
| 17 | COMPLETE | Opus | 4 files, 2 CRITICAL, 2 NEEDS_ATTENTION |
| 18 | COMPLETE | Opus | 5 files, 0 CRITICAL, 2 NEEDS_ATTENTION |
| 19 | COMPLETE | Opus | 5 files, 0 CRITICAL, 1 NEEDS_ATTENTION |
| 20 | COMPLETE | Opus | 3 files, 1 CRITICAL, 1 NEEDS_ATTENTION |
| 21 | COMPLETE | Opus | 2 files, 0 CRITICAL, 2 NEEDS_ATTENTION |
| 22 | COMPLETE | Opus | 1 file, 0 CRITICAL, 1 NEEDS_ATTENTION |
| 23 | COMPLETE | Opus | 5 files, 2 CRITICAL, 2 NEEDS_ATTENTION |
| 24 | COMPLETE | Opus | 5 files, 0 CRITICAL, 4 NEEDS_ATTENTION |
| 25 | COMPLETE | Opus | 3 files, 1 CRITICAL, 1 NEEDS_ATTENTION |
