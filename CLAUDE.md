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
pytest tests/core/           # Core RL component tests
pytest tests/shogi/          # Shogi game engine tests
pytest tests/evaluation/     # Evaluation system tests
pytest tests/training/       # Training manager tests

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

### WebUI (Twitch Streaming)
```bash
# Enable WebUI for streaming/demo
python train.py train --override webui.enabled=true

# Custom WebUI configuration
python train.py train --override webui.enabled=true --override webui.port=8765

# WebUI demo with mock data
python demo_webui.py

# Access WebUI dashboard
# WebSocket: ws://0.0.0.0:8765 (accessible from any IP)
# Dashboard: http://YOUR_SERVER_IP:8766
# Local access: http://localhost:8766
```

## High-Level Architecture

The system uses a manager-based architecture with 9 core specialized components orchestrated by the Trainer:

1. **SessionManager**: Handles directories, WandB setup, config saving
2. **ModelManager**: Model creation, checkpoints, mixed precision
3. **EnvManager**: Game setup, policy mapper, environment lifecycle
4. **StepManager**: Step execution, episode management, experience collection
5. **TrainingLoopManager**: Main training loop, PPO updates, callbacks
6. **MetricsManager**: Statistics collection, progress tracking, formatting
7. **DisplayManager**: Rich console UI, progress bars, logging
8. **CallbackManager**: Event system, evaluation scheduling, checkpoints
9. **SetupManager**: Component initialization, validation, dependencies

**Optional Components:**
- **WebUIManager**: Real-time WebSocket streaming for Twitch/demo (parallel to DisplayManager)

### Key Design Patterns

- **Protocol-based interfaces**: `ActorCriticProtocol` ensures model compatibility
- **Pydantic configuration**: Type-safe config with YAML loading and CLI overrides
- **Manager separation**: Each manager handles a single responsibility
- **Experience buffer**: Efficient storage with GAE computation
- **Rich console UI**: Real-time training visualization and logging

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
- **WebUI**: `webui/webui_manager.py`, `webui/static/index.html`

### Development Notes

- Always use the unified logger (`utils/unified_logger.py`) for consistent Rich-formatted output
- Model checkpoints include optimizer state, training metadata, and configuration
- The game engine supports full Shogi rules including drops, promotions, and special rules
- Experience collection can be parallelized using `training/parallel/` components
- Evaluation system supports multiple opponent types (random, heuristic, trained agents)
- WebUI system runs parallel to console display without impacting training performance
- WebUI requires `websockets` package: `pip install websockets`

## Development Environment Setup

### Installation
```bash
# Create virtual environment
python3 -m venv env
source env/bin/activate  # Linux/Mac
# or
env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
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
├── webui/                   # WebUI streaming system for Twitch/demo
│   ├── static/              # HTML/CSS/JS frontend files
│   ├── webui_manager.py     # WebSocket streaming manager
│   └── web_server.py        # HTTP server for static files
└── utils/                   # Utilities (logging, checkpoints, profiling)
```

### Configuration System
- **Primary config**: `default_config.yaml` with comprehensive documentation
- **Schema validation**: `config_schema.py` using Pydantic models
- **CLI overrides**: `python train.py train --override training.learning_rate=0.001`
- **Environment variables**: Load from `.env` file for W&B API keys

### Testing Strategy
- **Unit tests**: Fast, isolated component testing with pytest markers
- **Integration tests**: Multi-component interaction testing  
- **Performance tests**: Benchmarking and regression detection
- **CI/CD**: Automated testing with GitHub Actions and local CI script