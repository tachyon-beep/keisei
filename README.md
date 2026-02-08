# Keisei: Deep Reinforcement Learning for Shogi

**Keisei** (形勢, "position" in Shogi) is a deep reinforcement learning system that learns to play Shogi from scratch through self-play, using Proximal Policy Optimization (PPO).

No opening books, no hardcoded heuristics — strategies emerge purely from reinforcement learning.

## Features

- **Complete Shogi engine** with full rule support (drops, promotions, repetition)
- **PPO with self-play** — clipped surrogate, GAE, entropy regularization
- **ResNet + SE blocks** — configurable tower depth/width with Squeeze-and-Excitation attention
- **46-channel observation** (9x9 board) with 13,527-action policy space
- **Mixed precision** (AMP) and multi-GPU (DDP) support
- **Pydantic configuration** with YAML files and CLI overrides
- **Streamlit dashboard** for real-time training visualization
- **Weights & Biases** integration for experiment tracking
- **5 evaluation strategies** — single opponent, tournament, ladder, benchmark, custom

## Quick Start

### Prerequisites

- Python 3.12+ (3.13 recommended)
- CUDA-compatible GPU (optional but recommended)
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone https://github.com/tachyon-beep/shogidrl.git
cd keisei

# Create environment and install
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Optional: configure Weights & Biases
echo "WANDB_API_KEY=your_key" > .env
```

### Training

```bash
# Basic training
python train.py train

# With custom config
python train.py train --config examples/enhanced_display_config.yaml

# With CLI overrides
python train.py train --override training.learning_rate=0.001

# Resume from checkpoint
python train.py train --resume models/my_model/checkpoint.pt

# With Streamlit dashboard
python train.py train --override webui.enabled=true
```

### Evaluation

```bash
python train.py evaluate \
  --agent_checkpoint path/to/model.pt \
  --opponent_type random \
  --num_games 100
```

## Architecture

Keisei uses a manager-based architecture with 9 specialized components orchestrated by a central `Trainer`:

| Manager | Responsibility |
|---------|---------------|
| **SessionManager** | Directories, W&B setup, config persistence |
| **ModelManager** | Model creation, checkpoints, mixed precision |
| **EnvManager** | Game environment, policy mapper, lifecycle |
| **StepManager** | Step execution, episode management, experience collection |
| **TrainingLoopManager** | Main loop, PPO updates, callbacks |
| **MetricsManager** | Statistics, progress tracking, formatting |
| **DisplayManager** | Stderr logging (throttled one-line summaries) |
| **CallbackManager** | Event system, evaluation scheduling, checkpoints |
| **SetupManager** | Component initialization, validation, dependencies |

**Optional:** StreamlitManager provides a real-time training dashboard via atomic JSON state file.

## Project Structure

```
keisei/
├── config_schema.py           # Pydantic configuration models
├── constants.py               # Shared constants
├── core/                      # PPO agent, experience buffer, neural networks
├── shogi/                     # Complete Shogi game engine
├── training/                  # Manager-based training infrastructure
│   ├── models/                # Neural network architectures (ResNet, CNN)
│   └── parallel/              # Multi-process experience collection
├── evaluation/                # Multi-strategy evaluation system
├── webui/                     # Streamlit training dashboard
└── utils/                     # Logging, checkpoints, profiling
```

## Configuration

Configuration uses `default_config.yaml` with Pydantic validation. Override any setting via CLI:

```bash
python train.py train \
  --override training.learning_rate=0.001 \
  --override training.mixed_precision=true \
  --override webui.enabled=true
```

See `default_config.yaml` for all available options.

## Development

```bash
# Run tests
pytest tests/unit/              # Fast unit tests
pytest tests/integration/       # Integration tests
pytest tests/e2e/               # End-to-end tests

# Full local CI
./scripts/run_local_ci.sh

# Code quality
black keisei/                   # Formatting
mypy keisei/                    # Type checking
flake8 keisei/                  # Linting
```

See [CLAUDE.md](CLAUDE.md) for detailed development workflow, architecture notes, and contribution guidelines.

## Documentation

- [CLAUDE.md](CLAUDE.md) — Development guide with commands, architecture details, and patterns
- [docs/DESIGN.md](docs/DESIGN.md) — System design document
- [docs/CODE_MAP.md](docs/CODE_MAP.md) — Detailed code organization

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
