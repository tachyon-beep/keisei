# Keisei Architecture: Discovery Findings

**Date**: 2026-02-06
**Analyst**: Claude Opus 4.6
**Scope**: Full `keisei/` package, `tests/`, configuration files
**Confidence**: High (direct code analysis, metrics verified)

---

## 1. Project Identity

**Keisei** is a production-ready Deep Reinforcement Learning system for mastering Shogi (Japanese chess) from scratch using self-play and the Proximal Policy Optimization (PPO) algorithm. The name "keisei" (桂成) refers to a promoted knight in Shogi.

- **Repository**: `tachyon-beep/shogidrl`
- **Version**: 0.1.0
- **License**: MIT
- **Python**: >=3.12 (currently running 3.13.1)

## 2. Technology Stack

### Core
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| ML Framework | PyTorch | 2.7.0 | Neural networks, RL training |
| GPU Acceleration | CUDA | 12.6 (via cu12 packages) | GPU training |
| Kernel Compilation | Triton | 3.3.0 | Custom GPU kernels |
| Numerical | NumPy | 2.4.2 | Array operations |
| Statistics | SciPy | 1.17.0 | Advanced analytics |
| Config Validation | Pydantic | 2.11.4 | Type-safe configuration |
| Experiment Tracking | W&B | 0.19.11 | Logging, sweeps, artifacts |
| Console UI | Rich | 14.0.0 | Terminal visualization |
| WebUI | websockets | 16.0 | Real-time streaming |

### Development
| Tool | Purpose |
|------|---------|
| pytest 9.0.2 | Test framework |
| black 26.1.0 | Code formatting |
| mypy 1.19.1 | Static type checking |
| flake8 7.3.0 | Linting |
| pylint 4.0.4 | Advanced linting |
| bandit 1.9.3 | Security scanning |
| safety 3.7.0 | Vulnerability scanning |

## 3. Codebase Metrics

### Size
| Category | Files | LOC (non-blank, non-comment) |
|----------|-------|-----|
| Source (`keisei/`) | 87 | 21,586 |
| Tests (`tests/`) | 113 | ~39,000+ |
| Documentation | 281 .md files | - |
| WebUI Frontend | 4 JS/HTML/CSS | ~3,673 |
| **Total Code** | **200+** | **~64,000+** |

### Source Breakdown by Subsystem
| Subsystem | Files | LOC | Classes | Functions | Imports |
|-----------|-------|-----|---------|-----------|---------|
| evaluation/ | 28 | 7,965 | 49 | 288 | 233 |
| training/ | 28 | 6,706 | 40 | 279 | 253 |
| shogi/ | 8 | 2,584 | 7 | 107 | 34 |
| utils/ | 10 | 1,971 | 14 | 105 | 71 |
| core/ | 7 | 965 | 7 | 40 | 36 |
| webui/ | 3 | 656 | 4 | 38 | 26 |
| root (config, constants) | 3 | 739 | 12 | 31 | 5 |
| **TOTAL** | **87** | **21,586** | **133** | **888** | **658** |

### Code Health Indicators
- **Test-to-source ratio**: ~1.8x (tests are 80% larger than source - comprehensive coverage)
- **Average file size**: 248 LOC (healthy - no monolithic files)
- **Largest file**: `shogi_game.py` at 765 LOC (reasonable for a game engine)
- **Functions > 100 lines**: 16 files (some long methods need attention)
- **Longest function**: 301 lines in `display.py` (UI rendering - acceptable for display logic, but could be decomposed)

## 4. Entry Points

| Entry Point | Location | Purpose |
|-------------|----------|---------|
| `train.py` | root | CLI shim -> `keisei.training.train.main()` |
| `keisei/training/train.py` | training | Full CLI (train, evaluate subcommands) |
| `keisei/training/train_wandb_sweep.py` | training | W&B hyperparameter sweeps |
| `keisei/webui/web_server.py` | webui | HTTP server for dashboard static files |

## 5. Directory Organization

The codebase uses a **hybrid domain/layer** organization:

```
keisei/
├── core/           # Domain: RL fundamentals (PPO, buffers, protocols)
├── shogi/          # Domain: Game engine (rules, board, features)
├── training/       # Layer: Training orchestration (managers, display, CLI)
│   ├── models/     # Sub-domain: Neural network architectures
│   └── parallel/   # Sub-domain: Multi-process training
├── evaluation/     # Domain: Agent evaluation (strategies, analytics)
│   ├── core/       # Sub-layer: Evaluation infrastructure
│   ├── strategies/ # Sub-domain: Evaluation strategies
│   ├── opponents/  # Sub-domain: Opponent management
│   └── analytics/  # Sub-domain: Performance analysis
├── utils/          # Cross-cutting: Shared utilities
├── webui/          # Feature: WebSocket streaming dashboard
│   └── static/     # Frontend assets (JS, HTML, CSS, SVG)
├── config_schema.py  # Cross-cutting: Pydantic configuration
└── constants.py      # Cross-cutting: Application constants
```

## 6. Configuration System

**4-layer configuration hierarchy** (highest precedence last):

1. **Schema Defaults**: Pydantic `Field(default=...)` in `config_schema.py`
2. **YAML File**: `default_config.yaml` (comprehensive, documented)
3. **Environment Variables**: `.env` file via `python-dotenv`
4. **CLI Overrides**: `--override training.learning_rate=0.001`

**9 configuration sections**: EnvConfig, TrainingConfig, EvaluationConfig, LoggingConfig, WandBConfig, DisplayConfig, ParallelConfig, WebUIConfig, AppConfig (root)

**Key validation**: Pydantic field validators enforce constraints (learning_rate > 0, valid schedule types, etc.)

## 7. Subsystem Dependency Map

```
                    ┌──────────┐
                    │  config  │  (root: config_schema.py, constants.py)
                    │  schema  │
                    └─────┬────┘
                          │ used by all
          ┌───────────────┼────────────────┐
          │               │                │
    ┌─────▼─────┐   ┌────▼────┐    ┌──────▼──────┐
    │   core    │◄──│  utils  │    │    webui    │
    │ (PPO/buf) │──►│(logging,│    │ (dashboard) │
    └─────┬─────┘   │ mapper) │    └─────────────┘
          │         └────┬────┘
          │              │
    ┌─────▼─────┐   ┌───▼──────┐
    │   shogi   │◄──┤evaluation│
    │  (engine) │   │(strategies│
    └─────┬─────┘   │ analytics)│
          │         └───┬──────┘
          │             │
    ┌─────▼─────────────▼──────┐
    │       training           │
    │ (managers, orchestration)│
    └──────────────────────────┘
```

### Dependency Matrix
| From ↓ / To → | core | shogi | training | evaluation | utils | webui | root |
|---------------|------|-------|----------|------------|-------|-------|------|
| **core** | - | | | | yes | | yes |
| **shogi** | | - | | | | | |
| **training** | yes | yes | - | yes | yes | | yes |
| **evaluation** | yes | yes | | - | yes | | yes |
| **utils** | yes | yes | | | - | | yes |
| **webui** | | | | | | - | yes |

### Circular Dependencies
- **core <-> utils**: `core` imports from `utils` (PolicyOutputMapper, logging) and `utils` imports from `core` (PPOAgent for agent loading). This is a genuine circular coupling that should be addressed.

## 8. Key Design Patterns

1. **Manager Pattern**: 9+ specialized manager classes in training subsystem, each with single responsibility
2. **Protocol-Based Interfaces**: `ActorCriticProtocol` enables duck-typed model compatibility
3. **Strategy Pattern**: 5 pluggable evaluation strategies via factory registration
4. **Factory Pattern**: Scheduler factory, model factory, evaluator factory
5. **Observer/Callback Pattern**: `CallbackManager` with `TrainingCallback` protocol
6. **Async Concurrency**: WebSocket server, background tournaments
7. **Experience Replay**: `ExperienceBuffer` with GAE computation for PPO

## 9. External Integrations

| Integration | Protocol | Purpose | Optional |
|-------------|----------|---------|----------|
| Weights & Biases | HTTPS API | Experiment tracking, sweeps | Yes (can disable) |
| WebSocket Server | ws://0.0.0.0:8765 | Real-time training metrics | Yes (can disable) |
| HTTP Server | http://0.0.0.0:8766 | WebUI dashboard | Yes (can disable) |
| PyTorch CUDA | GPU API | GPU acceleration | Yes (CPU fallback) |
| torch.compile | Compiler API | Model optimization | Yes (auto-fallback) |

## 10. Test Organization

- **Structure**: Mirrors source - `tests/core/`, `tests/shogi/`, `tests/training/`, etc.
- **Markers defined**: unit, integration, slow, performance, e2e
- **Markers in practice**: Most tests are NOT tagged with markers (only `asyncio`, `slow`, `performance` actively used)
- **Fixtures**: Comprehensive `conftest.py` (655 LOC) with config builders, mock agents, WandB mocks
- **1,218 tests collected** (2 collection errors from missing scipy - now resolved)

## 11. Notable Observations

### Strengths
- Clean manager-based architecture with clear separation of concerns
- Protocol-based type safety for model interchangeability
- Comprehensive test suite (1.8x test-to-source ratio)
- Well-documented configuration with inline YAML documentation
- Multiple evaluation strategies for thorough agent assessment
- WebUI for streaming/demo without impacting training performance

### Concerns
- **core <-> utils circular dependency**: Real coupling that could cause import issues
- **16 functions > 100 lines**: Some long methods (max 301 lines in display.py)
- **Test markers unused**: `unit` marker defined but not applied to tests
- **Evaluation subsystem is the largest**: 7,965 LOC (37% of source) - potentially over-engineered relative to core RL
- **requirements.txt stale**: Contains git self-reference and pinned transitive deps that conflict with pyproject.toml
