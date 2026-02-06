# Keisei Subsystem Catalog

**Date**: 2026-02-06
**Analyst**: Claude Opus 4.6

---

## 1. Core RL Engine

**Location**: `keisei/core/`

**Responsibility**: Foundational reinforcement learning components implementing PPO with experience replay and GAE.

**Key Components**:
- `ppo_agent.py` - PPO algorithm: action selection, policy updates, loss computation, gradient clipping. Uses `ActorCriticProtocol` for model interface.
- `experience_buffer.py` - Pre-allocated tensor buffer for transitions (obs, actions, rewards, log_probs, values, dones, legal_masks). Implements GAE advantage computation.
- `actor_critic_protocol.py` - Python `Protocol` defining the model interface: `forward()`, `get_action_and_value()`, `evaluate_actions()`, plus PyTorch Module methods.
- `base_actor_critic.py` - Abstract base class providing shared Actor-Critic functionality.
- `neural_network.py` - Simple CNN-based Actor-Critic (legacy, superseded by ResNet).
- `scheduler_factory.py` - Factory for learning rate schedulers (step, cosine, linear, exponential, etc.).

**Dependencies**:
- Inbound: training (uses PPOAgent, ExperienceBuffer), evaluation (uses PPOAgent), utils (agent_loading)
- Outbound: utils (PolicyOutputMapper, unified_logger), root (config_schema)

**Patterns Observed**:
- Protocol-based interfaces (structural typing over inheritance)
- Pre-allocated tensor buffers for memory efficiency
- Factory pattern for scheduler creation
- Deep-copies config to prevent mutation

**Concerns**:
- Circular dependency with `utils` (core imports utils, utils imports core)
- `ppo_agent.py` has a 217-line function (PPO update logic) - could benefit from decomposition

**Confidence**: High - Read source files directly, verified imports and class hierarchies.

---

## 2. Shogi Game Engine

**Location**: `keisei/shogi/`

**Responsibility**: Complete Shogi rules implementation including move generation, validation, execution, board state management, and observation feature extraction.

**Key Components**:
- `shogi_game.py` (765 LOC) - Main game state class: board, hands, turn tracking, move history, game-over detection.
- `shogi_rules_logic.py` (488 LOC) - Legal move generation, check/checkmate detection, king safety validation. Contains a 149-line function for move generation.
- `shogi_move_execution.py` - Move execution: captures, promotions, drops, undo support.
- `shogi_game_io.py` (660 LOC) - Serialization: SFEN notation, USI protocol parsing, board string representation.
- `shogi_core_definitions.py` - Core types: `Piece`, `Color`, `MoveTuple` (NamedTuple), piece promotion maps.
- `features.py` - Observation feature extraction: 46-channel tensor (9x9 board) encoding piece positions, hands, turn, move history.
- `shogi_engine.py` - Thin wrapper providing engine interface.

**Dependencies**:
- Inbound: training (game environment), evaluation (game simulation), utils (move formatting)
- Outbound: None (self-contained domain layer)

**Patterns Observed**:
- Clean domain isolation - no imports from other keisei subsystems
- NamedTuple for immutable move representation (`MoveTuple`)
- Feature extraction separated from game logic
- Serialization in dedicated I/O module

**Concerns**:
- None critical. The engine is well-isolated and self-contained.

**Confidence**: High - Core domain with clear boundaries, verified no outbound dependencies.

---

## 3. Training System (Managers)

**Location**: `keisei/training/`

**Responsibility**: Orchestrates the full training loop via 9 specialized managers coordinated by the `Trainer` class.

**Key Components**:

### Orchestrator
- `trainer.py` (155-line longest function) - Central orchestrator: initializes all managers, runs training, handles cleanup.

### 9 Managers
| Manager | File | LOC | Responsibility |
|---------|------|-----|----------------|
| SessionManager | `session_manager.py` | ~400 | Directories, W&B setup, config saving, seeding |
| ModelManager | `model_manager.py` | 619 | Model creation, checkpoints, mixed precision, torch.compile |
| EnvManager | `env_manager.py` | ~300 | Game setup, PolicyOutputMapper, environment lifecycle |
| StepManager | `step_manager.py` | 546 | Step execution, episode management, experience collection (250-line function) |
| TrainingLoopManager | `training_loop_manager.py` | 571 | Main training loop, PPO updates, timing |
| MetricsManager | `metrics_manager.py` | ~350 | Statistics collection, progress tracking, formatting |
| DisplayManager | `display_manager.py` | ~300 | Rich console UI orchestration |
| CallbackManager | `callback_manager.py` | ~300 | Event system, evaluation scheduling, checkpoints |
| SetupManager | `setup_manager.py` | ~300 | Component initialization, validation, dependency wiring |

### Supporting
- `display.py` (537 LOC, 301-line function) - Rich UI rendering: panels, tables, sparklines, progress bars.
- `display_components.py` (468 LOC) - Individual UI widgets for the dashboard.
- `callbacks.py` - Callback implementations for training events.
- `train.py` (CLI) - Argument parsing, config loading, training/evaluation orchestration.
- `utils.py` - Training-specific utilities and helpers.

**Dependencies**:
- Inbound: CLI entry point (`train.py`)
- Outbound: core (PPOAgent, ExperienceBuffer), shogi (game engine), evaluation (EnhancedEvaluationManager), utils (logging, PolicyOutputMapper), root (AppConfig)

**Patterns Observed**:
- Manager pattern with single responsibility per manager
- Trainer as composition root / orchestrator
- Rich console UI with live display and progress bars
- Callback/observer pattern for event-driven checkpointing and evaluation

**Concerns**:
- `StepManager.step_manager` has a 250-line method - strong candidate for decomposition
- `display.py` has a 301-line rendering function
- Training depends on all other subsystems (expected for orchestrator, but high coupling)

**Confidence**: High - Read Trainer.__init__ and manager imports directly.

---

## 4. Neural Network Models

**Location**: `keisei/training/models/`

**Responsibility**: Neural network architectures implementing the ActorCriticProtocol.

**Key Components**:
- `__init__.py` - `model_factory()` function: creates models by type string ("resnet", "dummy", "testmodel").
- `resnet_tower.py` - `ActorCriticResTower`: Residual network with configurable depth, width, and SE (Squeeze-and-Excitation) blocks. Separate policy head (13,527 actions) and value head.

**Dependencies**:
- Inbound: training (ModelManager creates models), core (PPOAgent uses via protocol)
- Outbound: core (ActorCriticProtocol)

**Patterns Observed**:
- Factory pattern for model creation
- Protocol compliance without inheritance
- Configurable architecture (depth, width, SE ratio)
- Test model support ("dummy", "testmodel") with minimal parameters

**Concerns**:
- Only one real architecture (ResNet). No CNN, Transformer, or other alternatives ready for experimentation.

**Confidence**: High - Read factory and model source.

---

## 5. Parallel Training

**Location**: `keisei/training/parallel/`

**Responsibility**: Multi-process self-play for parallel experience collection.

**Key Components**:
- `parallel_manager.py` - Coordinates parallel worker processes.
- `self_play_worker.py` - Worker process: runs self-play games, collects experiences.
- `model_sync.py` - Synchronizes model weights between main process and workers.
- `communication.py` - Inter-process communication protocol (queues, shared memory).
- `utils.py` - Parallel-specific utilities.

**Dependencies**:
- Inbound: training (ParallelManager used by Trainer)
- Outbound: core (PPOAgent, ExperienceBuffer), shogi (game engine)

**Patterns Observed**:
- Worker pool pattern
- Message-passing for inter-process communication
- Model weight synchronization for distributed self-play

**Concerns**:
- Integration with main training loop is configuration-driven (`ParallelConfig`)
- Not clear how well-tested this is relative to the single-process path

**Confidence**: Medium - Read file listings and module structure; didn't deeply analyze implementation.

---

## 6. Evaluation System

**Location**: `keisei/evaluation/`

**Responsibility**: Comprehensive agent evaluation with pluggable strategies, opponent management, and performance analytics.

**Key Components**:

### Core Infrastructure
- `core_manager.py` - Base `EvaluationManager` with game execution logic.
- `enhanced_manager.py` - `EnhancedEvaluationManager`: extended evaluation with strategy dispatch.
- `evaluation_result.py` - Result data classes for evaluation outcomes.
- `evaluation_config.py` - Evaluation-specific configuration handling.
- `parallel_executor.py` - Parallel evaluation execution.
- `background_tournament.py` - Async background tournament runner.

### Strategies (5 pluggable modes)
| Strategy | File | Purpose |
|----------|------|---------|
| SingleOpponent | `strategies/single_opponent.py` (694 LOC) | 1v1 evaluation against a specific opponent |
| Tournament | `strategies/tournament.py` (673 LOC) | Round-robin tournament between multiple agents |
| Ladder | `strategies/ladder.py` (624 LOC) | ELO-based ladder climbing |
| Benchmark | `strategies/benchmark.py` (651 LOC) | Standardized benchmark suites |
| Custom | `strategies/custom.py` | User-defined evaluation scripts |

### Opponent Management
- `opponents/opponent_pool.py` - Pool of available opponents.
- `opponents/elo_registry.py` - ELO rating tracking and updates.
- `opponents/enhanced_manager.py` - Enhanced opponent lifecycle management.

### Analytics
- `analytics/performance_analyzer.py` - Win rates, move quality, game length analysis.
- `analytics/elo_tracker.py` - Historical ELO tracking with confidence intervals.
- `analytics/report_generator.py` - Human-readable evaluation reports.
- `analytics/advanced_analytics.py` - Statistical tests (requires scipy).

**Dependencies**:
- Inbound: training (CallbackManager triggers evaluations)
- Outbound: core (PPOAgent), shogi (game engine), utils (agent_loading, logging), root (config)

**Patterns Observed**:
- Strategy pattern with factory registration
- Async execution for background tournaments
- Comprehensive analytics pipeline
- Optional enhanced features (graceful degradation via try/except import)

**Concerns**:
- **Largest subsystem** at 7,965 LOC (37% of source) - significantly larger than core RL (965 LOC)
- Some strategy files have 130+ line functions
- May be over-engineered relative to current training capabilities (tournament/ladder features when the agent may not be strong enough yet)
- `advanced_analytics.py` has external dependency on scipy (handled gracefully)

**Confidence**: High - Read __init__.py, strategy files, and analytics code.

---

## 7. Utilities

**Location**: `keisei/utils/`

**Responsibility**: Shared utilities spanning logging, action mapping, agent loading, profiling, and performance benchmarking.

**Key Components**:
- `utils.py` (469 LOC) - `PolicyOutputMapper` (maps 13,527 actions to/from Shogi moves), `load_config()`, `TrainingLogger` factory.
- `unified_logger.py` - Centralized Rich-formatted logging: `log_info_to_stderr()`, `log_error_to_stderr()`, etc.
- `agent_loading.py` (174-line function) - Loads PPOAgent from checkpoint files with config compatibility handling.
- `compilation_validator.py` - Validates `torch.compile()` output: numerical equivalence, performance benchmarks.
- `performance_benchmarker.py` - Timing benchmarks for model forward/backward passes.
- `profiling.py` - cProfile integration, timing decorators.
- `move_formatting.py` - Human-readable move descriptions for display.
- `opponents.py` - Opponent implementations (random, heuristic).
- `checkpoint.py` - Low-level checkpoint save/load utilities.

**Dependencies**:
- Inbound: core (logging), training (all managers), evaluation (agent loading, logging)
- Outbound: core (PPOAgent for agent_loading), shogi (move formatting), root (config)

**Patterns Observed**:
- Centralized logging via unified_logger
- Large action space mapping (13,527 actions) via lookup tables
- Factory functions for logger and config creation

**Concerns**:
- **Circular dependency with core**: `utils.agent_loading` imports `PPOAgent` from `core`, while `core.ppo_agent` imports `PolicyOutputMapper` from `utils`
- `utils.py` is a grab-bag - could benefit from splitting `PolicyOutputMapper` into its own module
- `opponents.py` arguably belongs in `evaluation/opponents/`

**Confidence**: High - Read key source files and verified import chains.

---

## 8. WebUI Streaming System

**Location**: `keisei/webui/`

**Responsibility**: Real-time training visualization via WebSocket for Twitch streaming and demos.

**Key Components**:
- `webui_manager.py` (565 LOC, 148-line function) - `WebUIManager`: async WebSocket server, message serialization, training state broadcasting. Sends board state, metrics, game events.
- `web_server.py` - HTTP server serving static files (dashboard) on port 8766.
- `static/index.html` (755 LOC) - Dashboard HTML with responsive layout.
- `static/app.js` (1,531 LOC) - Main frontend: WebSocket client, board rendering, metrics display.
- `static/advanced_visualizations.js` (919 LOC) - Chart.js visualizations for training metrics.
- `static/images/` - 28 SVG files for Shogi piece graphics.

**Dependencies**:
- Inbound: training (Trainer initializes WebUIManager alongside DisplayManager)
- Outbound: root (WebUIConfig)

**Patterns Observed**:
- Async WebSocket server (Python `websockets` library)
- Parallel to console DisplayManager (doesn't replace it)
- Vanilla JavaScript frontend (no framework dependencies)
- SVG piece graphics for board rendering

**Concerns**:
- 148-line function in webui_manager.py
- Frontend is vanilla JS - harder to maintain at scale than a framework-based approach
- No authentication on WebSocket/HTTP endpoints (acceptable for local/streaming use)

**Confidence**: High - Read Python and frontend source.

---

## 9. Configuration System

**Location**: `keisei/config_schema.py`, `keisei/constants.py`, `default_config.yaml`

**Responsibility**: Type-safe configuration with validation, defaults, and multi-source loading.

**Key Components**:
- `config_schema.py` (602 LOC, 12 classes) - Pydantic v2 BaseModel classes for all 9 config sections:
  - `EnvConfig` - Device, channels, actions, seed, max moves
  - `TrainingConfig` - PPO hyperparameters, model architecture, learning rate schedules
  - `EvaluationConfig` - Strategy selection, opponent types, game counts
  - `LoggingConfig` - Run naming, file paths, log levels
  - `WandBConfig` - Experiment tracking toggle, project/entity names
  - `DisplayConfig` - Rich console settings, layer filters
  - `ParallelConfig` - Worker counts, batch sizes
  - `WebUIConfig` - Enable/disable, port configuration
  - `AppConfig` - Root config composing all sections
- `constants.py` (183 LOC) - Application-wide constants (board dimensions, piece types, etc.)
- `default_config.yaml` - Comprehensive defaults with inline documentation.

**Dependencies**:
- Inbound: Every subsystem reads config
- Outbound: None (leaf dependency)

**Patterns Observed**:
- Pydantic v2 with `Field()` descriptors and `field_validator` decorators
- Nested composition (AppConfig contains all section configs)
- 4-layer precedence: schema defaults -> YAML -> env vars -> CLI overrides
- Cross-field validation documented in YAML comments

**Concerns**:
- `EvaluationStrategy` class duplicates enum-like constants alongside a list - could use `StrEnum`
- No cross-field validators (e.g., `steps_per_epoch` divides `evaluation_interval_timesteps`) - only documented, not enforced

**Confidence**: High - Read full config_schema.py and default_config.yaml.

---

## 10. Test Suite

**Location**: `tests/`

**Responsibility**: Comprehensive testing covering all subsystems with fixtures, markers, and mocking.

**Key Components**:
- `conftest.py` (655 LOC) - Shared fixtures: config builders, mock agents, WandB mocks, test data generators.
- 113 test files organized by subsystem mirror structure.
- Test categories: core/, shogi/, training/, evaluation/, integration/, e2e/, performance/, parallel/, display/, utils/, webui/

**Dependencies**:
- Inbound: CI/CD pipeline
- Outbound: All source subsystems (tests import from all)

**Patterns Observed**:
- Subsystem-mirrored test structure
- Comprehensive fixture library
- WandB mocking infrastructure (prevents real API calls in tests)
- Markers defined but **not consistently applied** (tests collected but `unit` marker matches 0 tests)

**Concerns**:
- **Test markers not applied**: `pytest -m unit` collects 0 tests despite 1,218 available
- Test suite is 1.8x the size of source code - maintenance burden
- Some test files may have low coverage value relative to their size
- 2 collection errors from scipy dependency (resolved by adding scipy to deps)

**Confidence**: High - Ran pytest collection, read conftest.py, verified marker behavior.
