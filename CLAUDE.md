# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Keisei is a production-ready Deep Reinforcement Learning system for mastering Shogi (Japanese chess) from scratch using self-play and the Proximal Policy Optimization (PPO) algorithm. The project features a modern manager-based architecture with 9 specialized components.

## Common Development Commands

### Training
```bash
# Basic training
python train.py train

# Training with custom config
python train.py train --config my_config.yaml

# Training with CLI overrides
python train.py train --override training.learning_rate=0.001 --override training.total_timesteps=500000

# Resume from checkpoint
python train.py train --resume models/my_model/checkpoint.pt

# Hyperparameter sweep with Weights & Biases
python -m keisei.training.train_wandb_sweep
```

### Testing and Quality Checks
```bash
# Run full CI pipeline locally
./scripts/run_local_ci.sh

# Run test categories by marker
pytest -m unit              # Unit tests (fast, isolated)
pytest -m integration       # Integration tests (multi-component)
pytest -m performance       # Performance benchmarks
pytest -m slow              # Slow tests

# Run specific test directories
pytest tests/unit/            # Unit tests (fast, isolated)
pytest tests/integration/     # Integration tests (multi-component)
pytest tests/e2e/             # End-to-end tests (full workflows)

# Code quality
black keisei/              # Code formatting
mypy keisei/               # Type checking
flake8 keisei/             # Linting
```

### Evaluation
```bash
# Evaluate against random opponent
python train.py evaluate \
  --agent_checkpoint path/to/model.pt \
  --opponent_type random \
  --num_games 100

# Evaluate against another trained agent
python train.py evaluate \
  --agent_checkpoint path/to/model1.pt \
  --opponent_type ppo \
  --opponent_checkpoint path/to/model2.pt \
  --num_games 50 \
  --wandb_log_eval
```

### WebUI (Streamlit Dashboard)
```bash
# Enable Streamlit dashboard during training
python train.py train --override webui.enabled=true

# Custom port (default: 8501)
python train.py train --override webui.enabled=true --override webui.port=8501

# Run Streamlit dashboard standalone (demo mode with sample data)
streamlit run keisei/webui/streamlit_app.py

# Run with a specific state file
streamlit run keisei/webui/streamlit_app.py -- --state-file path/to/state.json

# Access dashboard: http://localhost:8501
```

## High-Level Architecture

The system uses a manager-based architecture with 9 core specialized components orchestrated by the Trainer:

1. **SessionManager**: Handles directories, WandB setup, config saving
2. **ModelManager**: Model creation, checkpoints, mixed precision
3. **EnvManager**: Game setup, policy mapper, environment lifecycle
4. **StepManager**: Step execution, episode management, experience collection
5. **TrainingLoopManager**: Main training loop, PPO updates, callbacks
6. **MetricsManager**: Statistics collection, progress tracking, formatting
7. **DisplayManager**: Barebones stderr logging (throttled one-line summaries)
8. **CallbackManager**: Event system, evaluation scheduling, checkpoints
9. **SetupManager**: Component initialization, validation, dependencies

**Optional Components:**
- **StreamlitManager**: Streamlit training dashboard (parallel to DisplayManager, communicates via atomic JSON state file)

### Key Design Patterns

- **Protocol-based interfaces**: `ActorCriticProtocol` ensures model compatibility
- **Pydantic configuration**: Type-safe config with YAML loading and CLI overrides
- **Manager separation**: Each manager handles a single responsibility
- **Experience buffer**: Efficient storage with GAE computation
- **Streamlit dashboard**: Optional real-time training visualization via atomic JSON state file

### Critical Implementation Details

1. **Action Space**: 13,527 total actions mapped via `PolicyOutputMapper`
2. **Observation Space**: 46-channel tensor (9x9 board representation)
3. **Neural Networks**: Support for CNN and ResNet architectures with SE blocks
4. **Mixed Precision**: Optional AMP for faster training on modern GPUs
5. **Distributed Training**: DDP support for multi-GPU setups

### Important Paths

- **Configuration**: `default_config.yaml`, `config_schema.py`
- **Core RL**: `core/ppo_agent.py`, `core/experience_buffer.py`
- **Game Engine**: `shogi/shogi_game.py`, `shogi/shogi_rules_logic.py`
- **Training**: `training/trainer.py`, `training/train.py`
- **Models**: `training/models/resnet_tower.py`, `core/neural_network.py`
- **Utils**: `utils/unified_logger.py`, `utils/checkpoint.py`
- **WebUI**: `webui/streamlit_manager.py`, `webui/streamlit_app.py`, `webui/state_snapshot.py`

### Development Notes

- Always use the unified logger (`utils/unified_logger.py`) for consistent logging output
- Model checkpoints include optimizer state, training metadata, and configuration
- The game engine supports full Shogi rules including drops, promotions, and special rules
- Experience collection can be parallelized using `training/parallel/` components
- Evaluation system supports 5 strategies: single opponent, tournament, ladder, benchmark, custom
- Streamlit dashboard runs as a subprocess, communicating via atomic JSON state file
- WebUI requires `streamlit` package: `pip install keisei[webui]` or `pip install streamlit`

## Development Environment Setup

### Installation
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate

# Install project with dev dependencies
uv pip install -e ".[dev]"

# Dependencies are defined in pyproject.toml (requirements.txt is stale)
```

### Project Structure
```
keisei/
├── config_schema.py           # Pydantic configuration models
├── constants.py              # Shared constants
├── core/                     # Core RL components (PPO, networks, buffers)
├── shogi/                    # Complete Shogi game implementation
├── training/                 # Manager-based training infrastructure
│   ├── models/              # Neural network architectures
│   └── parallel/            # Multi-process experience collection
├── evaluation/              # Evaluation system with multiple strategies
├── webui/                   # Streamlit training dashboard
│   ├── streamlit_app.py     # Standalone Streamlit dashboard app
│   ├── streamlit_manager.py # Manager that launches/bridges to Streamlit
│   └── state_snapshot.py    # Atomic JSON state file builder
└── utils/                   # Utilities (logging, checkpoints, profiling)
```

### Configuration System
- **Primary config**: `default_config.yaml` with comprehensive documentation
- **Schema validation**: `config_schema.py` using Pydantic models
- **CLI overrides**: `python train.py train --override training.learning_rate=0.001`
- **Environment variables**: Load from `.env` file for W&B API keys
- **Dependency management**: `pyproject.toml` is the source of truth (managed with `uv`)

### Testing Strategy
- **Unit tests** (`tests/unit/`): Fast, isolated component testing
- **Integration tests** (`tests/integration/`): Multi-component interaction testing
- **E2E tests** (`tests/e2e/`): Full workflow tests (CLI, checkpoint resume)
- **Markers**: Defined in `pytest.ini` (unit, integration, e2e, slow, performance) but not yet applied to test functions — use directory-based test selection instead
- **CI/CD**: Local CI via `./scripts/run_local_ci.sh` (GitHub Actions CI is currently disabled)