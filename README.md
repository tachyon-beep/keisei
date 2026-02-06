# Keisei: Production-Ready Deep Reinforcement Learning Shogi AI

**Keisei** (形勢, Japanese for "position" or "situation" in a game like Shogi or Go) is a production-ready Deep Reinforcement Learning (DRL) system designed to master the game of Shogi from scratch. Built with a modern, manager-based architecture, it learns exclusively through self-play using the Proximal Policy Optimization (PPO) algorithm with advanced features including mixed precision training, distributed computing support, and comprehensive experiment tracking.

## Table of Contents

- [Project Overview](#project-overview)
- [Core Philosophy](#core-philosophy)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
  - [Manager-Based Architecture](#manager-based-architecture)
  - [Configuration System](#configuration-system)
  - [Training Pipeline](#training-pipeline)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running Training](#running-training)
  - [Monitoring & Evaluation](#monitoring--evaluation)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
  - [Neural Network Architectures](#neural-network-architectures)
  - [State & Action Representation](#state--action-representation)
  - [Training Features](#training-features)
- [Development & Testing](#development--testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Keisei implements a sophisticated, production-ready Shogi AI training system featuring:

- **Complete Shogi Implementation**: Full rule support including drops, promotions, and advanced rules (Sennichite, etc.)
- **Modern Architecture**: Manager-based design with 9 specialized components for modularity and maintainability
- **Advanced Training**: Mixed precision, distributed training, gradient clipping, and comprehensive checkpointing
- **Professional Monitoring**: Rich console UI, comprehensive logging, and Weights & Biases integration
- **Robust Evaluation**: Multi-opponent evaluation system with statistical analysis
- **Production Features**: CI/CD pipeline, comprehensive test suite, and extensive documentation

For comprehensive technical details, see:
- [Design Document (`docs/DESIGN.md`)](docs/DESIGN.md) - Complete system architecture and design decisions
- [Code Map (`docs/CODE_MAP.md`)](docs/CODE_MAP.md) - Detailed codebase organization
- [Component Documentation (`docs/components/`)](docs/components/) - Individual module documentation

## Core Philosophy

- **Learn from Scratch**: No hardcoded opening books, human-designed heuristics, or evaluation functions
- **Pure Self-Play**: Strategies emerge solely through reinforcement learning and self-play
- **Production Quality**: Enterprise-grade architecture, monitoring, and operational capabilities
- **Research-Ready**: Modular design supporting advanced research and experimentation

## Key Features

### Core Training Capabilities
- **Advanced PPO Implementation**: Clipped surrogate objectives, entropy regularization, and Generalized Advantage Estimation (GAE)
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) with GradScaler for memory efficiency and speed
- **Distributed Training Support**: DistributedDataParallel (DDP) for multi-GPU training
- **Parallel Self-Play Support**: Multi-process experience collection with `training/parallel`
- **Intelligent Checkpointing**: Automatic model saving with resumption capabilities
- **Gradient Management**: Configurable gradient clipping and optimization stability controls

### Neural Network Architectures
- **Flexible Model Factory**: Support for multiple architectures (CNN, ResNet Tower with SE blocks)
- **Feature Engineering**: Comprehensive 46-channel board representation with configurable feature sets
- **Actor-Critic Design**: Separate policy and value heads with configurable architectures
- **Modern Components**: Support for Squeeze-and-Excitation blocks and advanced regularization

### Production Infrastructure
- **Pydantic Configuration**: Type-safe, validated configuration with YAML loading
- **Manager-Based Architecture**: 9 specialized managers for clean separation of concerns
- **Rich Console UI**: Real-time training visualization with progress bars and metrics
- **Unified Logger**: Consistent Rich-formatted logging via `utils/unified_logger.py`
- **Comprehensive Logging**: Structured logging with file output and console formatting
- **Weights & Biases Integration**: Professional experiment tracking and model artifact management
- **Profiling Utilities**: Lightweight timing and cProfile helpers for development

### Evaluation & Monitoring
- **Multi-Opponent Evaluation**: Test against random, heuristic, and other trained agents
- **Statistical Analysis**: Comprehensive performance metrics and significance testing
- **Real-time Monitoring**: Live training metrics and performance visualization
- **Automated Evaluation**: Scheduled evaluation runs during training

### Development Features
- **CI/CD Pipeline**: Automated testing, linting, and quality checks
- **Comprehensive Test Suite**: Unit tests, integration tests, and performance benchmarks
- **Code Quality Tools**: Black formatting, mypy type checking, and security scanning
- **Extensive Documentation**: Complete API documentation and usage guides
- **Profiling Helpers**: Timing and cProfile utilities for performance debugging

## System Architecture

### Manager-Based Architecture

Keisei employs a sophisticated manager-based architecture that separates concerns across 9 specialized components:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Trainer (Orchestrator)                   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│  │ SessionManager  │ │ ModelManager    │ │ EnvManager      │    │
│  │ - Directories   │ │ - Model Creation│ │ - Game Setup    │    │
│  │ - WandB Setup   │ │ - Checkpoints   │ │ - Policy Mapper │    │
│  │ - Config Save   │ │ - Mixed Prec.   │ │ - Environment   │    │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘    │
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│  │ StepManager     │ │TrainingLoopMgr  │ │ MetricsManager  │    │
│  │ - Step Exec     │ │ - Main Loop     │ │ - Statistics    │    │
│  │ - Episode Mgmt  │ │ - PPO Updates   │ │ - Progress      │    │
│  │ - Experience    │ │ - Callbacks     │ │ - Formatting    │    │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘    │
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│  │ DisplayManager  │ │ CallbackManager │ │ SetupManager    │    │
│  │ - Rich UI       │ │ - Event System  │ │ - Initialization│    │
│  │ - Progress Bars │ │ - Evaluation    │ │ - Validation    │    │
│  │ - Logging       │ │ - Checkpoints   │ │ - Dependencies  │    │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Architecture Benefits:**
- **Modularity**: Each manager handles a single responsibility
- **Testability**: Components can be tested in isolation
- **Maintainability**: Changes localized to specific managers
- **Extensibility**: New features added without affecting other components

### Configuration System

The system uses a sophisticated Pydantic-based configuration architecture:

```python
class AppConfig(BaseModel):
    env: EnvConfig           # Environment settings (device, seeds, action space)
    training: TrainingConfig # PPO parameters, learning rates, optimization
    evaluation: EvaluationConfig # Evaluation schedules and parameters
    logging: LoggingConfig   # File paths, log levels, Rich UI settings
    wandb: WandBConfig      # Experiment tracking configuration
    display: DisplayConfig  # Visualization and move display settings
```

**Configuration Features:**
- **Type Safety**: Full Pydantic validation with detailed error reporting
- **YAML Loading**: Human-readable configuration files with validation
- **CLI Overrides**: Command-line parameter overrides with nested key support
- **Default Management**: Sensible defaults with comprehensive documentation

### Training Pipeline

The training process follows a sophisticated pipeline:

1. **Initialization**: `Trainer` sets up all 9 managers and validates configuration
2. **Session Setup**: `SessionManager` creates directories and initializes experiment tracking
3. **Component Setup**: `SetupManager` initializes models, agents, and environments
4. **Training Loop**: `TrainingLoopManager` orchestrates self-play and learning
5. **Step Execution**: `StepManager` handles individual game steps and experience collection
6. **PPO Updates**: Policy learning with comprehensive loss tracking
7. **Evaluation**: `CallbackManager` triggers periodic evaluation runs
8. **Monitoring**: `DisplayManager` and `MetricsManager` provide real-time feedback

## Getting Started

### Prerequisites

- **Python 3.12+** (recommended for best performance and compatibility)
- **PyTorch** with CUDA support (recommended for GPU acceleration)
- **Git** for version control
- **8GB+ RAM** (16GB+ recommended for larger models)
- **CUDA-compatible GPU** (optional but recommended for training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd keisei
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   # Production dependencies
   pip install -r requirements.txt
   
   # Development dependencies (for testing and development)
   pip install -r requirements-dev.txt
   ```

4. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

5. **(Optional) Configure Weights & Biases:**
   ```bash
   # Create .env file for W&B API key
   echo "WANDB_API_KEY=your_api_key_here" > .env
   ```

### Configuration

Keisei uses YAML configuration files with Pydantic validation. The main configuration file is `default_config.yaml`:

```yaml
env:
  device: "cuda"              # Use "cpu" for CPU-only training
  input_channels: 46          # Observation space channels
  max_moves_per_game: 256     # Game length limit

training:
  learning_rate: 0.0003       # PPO learning rate
  total_timesteps: 100000     # Total training steps
  steps_per_epoch: 2048       # Steps per training epoch
  ppo_epochs: 4               # PPO update epochs per batch
  mixed_precision: false      # Enable AMP for faster training
  distributed: false          # Enable distributed training

evaluation:
  enable_periodic_evaluation: true
  evaluation_interval_timesteps: 50000
  num_games: 20               # Games per evaluation

logging:
  model_dir: "models"         # Model checkpoint directory
  log_file: "training_log.txt"

wandb:
  enabled: true               # Enable experiment tracking
  project: "keisei-shogi"     # W&B project name
```

**Key Configuration Options:**
- **Device**: Set to `"cuda"` for GPU or `"cpu"` for CPU-only training
- **Training Parameters**: Adjust learning rate, batch sizes, and training duration
- **Mixed Precision**: Enable for faster training on modern GPUs
- **Evaluation**: Configure automated evaluation frequency and opponents

### Running Training

1. **Basic Training:**
   ```bash
   python train.py
   ```

2. **Training with Custom Configuration:**
   ```bash
   python train.py --config my_config.yaml
   ```

3. **Training with CLI Overrides:**
   ```bash
   python train.py --config default_config.yaml \
     training.learning_rate=0.001 \
     training.total_timesteps=500000 \
     env.device=cuda
   ```

4. **Resume from Checkpoint:**
   ```bash
   python train.py --resume_checkpoint_path models/my_model/checkpoint.pt
   ```

5. **Weights & Biases Hyperparameter Sweep:**
   ```bash
   python -m keisei.training.train_wandb_sweep
   ```

### Monitoring & Evaluation

#### Real-time Training Monitoring
- **Rich Console UI**: Real-time progress bars, metrics, and log messages
- **Training Logs**: Comprehensive file-based logging in `logs/` directory
- **Weights & Biases Dashboard**: Professional experiment tracking and visualization

#### Evaluation System
```bash
# Evaluate against random opponent
python -m keisei.evaluation.evaluate \
  --agent_checkpoint path/to/model.pt \
  --opponent_type random \
  --num_games 100

# Evaluate against another trained agent
python -m keisei.evaluation.evaluate \
  --agent_checkpoint path/to/model1.pt \
  --opponent_type ppo \
  --opponent_checkpoint path/to/model2.pt \
  --num_games 50 \
  --wandb_log_eval
```

#### Monitoring Features
- **Real-time Metrics**: Win rates, episode lengths, training losses
- **Performance Tracking**: Steps per second, GPU utilization, memory usage
- **Model Artifacts**: Automatic model versioning and artifact management
- **Statistical Analysis**: Confidence intervals, significance testing

## Project Structure

```
keisei/
├── README.md                    # This file
├── pyproject.toml              # Project metadata and dependencies
├── default_config.yaml         # Default training configuration
├── train.py                    # Main training entry point
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
│
├── keisei/                     # Core library package
│   ├── config_schema.py        # Pydantic configuration models
│   ├── constants.py            # Shared board and observation constants
│   │
│   ├── core/                   # Core RL components
│   │   ├── actor_critic_protocol.py # Model interface definition
│   │   ├── base_actor_critic.py # Common ActorCritic helpers
│   │   ├── neural_network.py   # Basic ActorCritic implementation
│   │   ├── ppo_agent.py        # PPO algorithm implementation
│   │   ├── experience_buffer.py # Experience storage and GAE
│   │   ├── scheduler_factory.py # LR scheduler creation helpers
│   │   └── __init__.py
│   │
│   ├── shogi/                  # Shogi game engine
│   │   ├── shogi_core_definitions.py # Piece types, board constants
│   │   ├── shogi_game.py       # Main game logic and state
│   │   ├── shogi_rules_logic.py # Rule validation and enforcement
│   │   ├── shogi_move_execution.py # Move execution and board updates
│   │   ├── shogi_game_io.py    # I/O, SFEN, observation generation
│   │   ├── features.py         # Feature extraction and encoding
│   │   └── __init__.py
│   │
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py          # Main Trainer orchestrator
│   │   ├── train.py            # Training script with CLI
│   │   ├── train_wandb_sweep.py # W&B hyperparameter sweeps
│   │   │
│   │   ├── models/             # Neural network architectures
│   │   │   ├── __init__.py     # Model factory
│   │   │   └── resnet_tower.py # ResNet architecture with SE blocks
│   │   │
│   │   ├── session_manager.py  # Session and directory management
│   │   ├── model_manager.py    # Model operations and checkpointing
│   │   ├── env_manager.py      # Environment lifecycle management
│   │   ├── step_manager.py     # Training step execution
│   │   ├── metrics_manager.py  # Metrics collection and formatting
│   │   ├── training_loop_manager.py # Main training loop
│   │   ├── display_manager.py  # Rich UI and visualization
│   │   ├── callback_manager.py # Training callbacks and events
│   │   ├── setup_manager.py    # Component initialization
│   │   ├── parallel/           # Multi-process experience collection
│   │   │   ├── parallel_manager.py # Worker orchestration
│   │   │   ├── self_play_worker.py # Self-play worker process
│   │   │   ├── communication.py    # IPC utilities
│   │   │   ├── model_sync.py       # Model synchronization
│   │   │   └── utils.py            # Compression helpers
│   │   └── utils.py            # Training utilities
│   │
│   ├── evaluation/             # Evaluation system
│   │   ├── evaluate.py         # Main evaluation orchestrator
│   │   ├── loop.py             # Evaluation game loop
│   │   └── __init__.py
│   │
│   └── utils/                  # Utilities and support
│       ├── agent_loading.py    # Agent loading and initialization
│       ├── checkpoint.py       # Checkpoint management utilities
│       ├── move_formatting.py  # Move display and formatting
│       ├── opponents.py        # Opponent implementations
│       ├── unified_logger.py   # Rich console logging helper
│       ├── profiling.py        # Development profiling tools
│       ├── utils.py            # Core utilities and policy mapping
│       └── __init__.py
│
├── docs/                       # Comprehensive documentation
│   ├── DESIGN.md              # Complete system design document
│   ├── CODE_MAP.md            # Detailed code organization
│   ├── CI_CD.md               # CI/CD pipeline documentation
│   └── components/            # Individual component documentation
│       ├── core_*.md          # Core component docs
│       ├── training_*.md      # Training component docs
│       ├── evaluation_*.md    # Evaluation component docs
│       └── shogi_*.md         # Game engine docs
│
├── tests/                     # Comprehensive test suite
│   ├── conftest.py           # Test configuration and fixtures
│   ├── test_*.py             # Unit and integration tests
│   └── integration/          # Integration test suites
│
├── scripts/                  # Development and utility scripts
│   ├── profile_training.py   # Performance profiling
│   └── run_training.py       # Training utilities
│
├── models/                   # Saved model checkpoints
├── logs/                     # Training logs and outputs
└── wandb/                    # Weights & Biases local artifacts
```

**Key Directories:**
- **`keisei/core/`**: Core reinforcement learning components (PPO, base networks, schedulers)
- **`keisei/shogi/`**: Complete Shogi game implementation with full rule support
- **`keisei/training/`**: Manager-based training infrastructure with parallel self-play support
- **`keisei/evaluation/`**: Comprehensive evaluation system with multi-opponent support
- **`keisei/utils/`**: Unified logger, profiling helpers, and utility functions
- **`docs/`**: Extensive documentation including design documents and component details

## Technical Details

### Neural Network Architectures

Keisei supports multiple neural network architectures through a flexible model factory system:

#### ActorCritic (Basic CNN)
- **Convolutional layers**: Feature extraction from board representation
- **Residual connections**: Improved gradient flow and training stability
- **Separate heads**: Independent policy and value network heads
- **Configurable depth**: Adjustable number of layers and filters

#### ResNet Tower (Advanced)
- **ResNet blocks**: Deep residual network architecture for complex pattern recognition
- **Squeeze-and-Excitation**: Channel attention mechanism for improved feature selection
- **Configurable architecture**: Adjustable tower depth, width, and SE ratio
- **Mixed precision support**: Automatic Mixed Precision (AMP) for faster training

```python
# Model configuration examples
model_configs = {
    "basic_cnn": {
        "model_type": "cnn",
        "input_channels": 46,
        "num_actions_total": 6480
    },
    "resnet_tower": {
        "model_type": "resnet",
        "tower_depth": 10,
        "tower_width": 256,
        "se_ratio": 0.25,
        "input_features": "core46"
    }
}
```

### State & Action Representation

#### State Representation (46-Channel Observation)
The game state is encoded as a multi-channel 3D tensor `(46, 9, 9)`:

- **Player piece planes** (14 channels): Current player's pieces by type and promotion status
- **Opponent piece planes** (14 channels): Opponent's pieces by type and promotion status
- **Hand piece planes** (14 channels): Pieces in hand for both players
- **Game state planes** (4 channels): Turn indicator, repetition count, castling rights, etc.

#### Action Space (6,480 Actions)
Complete coverage of all possible Shogi moves:

- **Board moves**: All possible piece movements with and without promotion
- **Drop moves**: Placing captured pieces back on the board
- **Special moves**: Castling, en passant equivalents in Shogi

```python
class PolicyOutputMapper:
    """Maps between neural network outputs and Shogi game actions."""
    
    def get_total_actions(self) -> int:
        return 6480  # Complete action space coverage
    
    def policy_index_to_shogi_move(self, index: int) -> MoveTuple:
        """Convert policy network output index to Shogi move."""
    
    def shogi_move_to_policy_index(self, move: MoveTuple) -> int:
        """Convert Shogi move to policy network input index."""
```

### Training Features

#### Advanced PPO Implementation
- **Clipped Surrogate Objective**: Prevents large policy updates for training stability
- **Generalized Advantage Estimation (GAE)**: Improved value function learning
- **Entropy Regularization**: Maintains exploration throughout training
- **Value Function Clipping**: Stabilizes critic learning

#### Performance Optimizations
- **Mixed Precision Training**: Faster training with reduced memory usage
- **Gradient Clipping**: Prevents exploding gradients
- **Distributed Training**: Multi-GPU support with DistributedDataParallel
- **Efficient Experience Collection**: Optimized batch processing and memory management

#### Training Stability
- **Comprehensive Checkpointing**: Model state, optimizer state, and training metadata
- **Resume Capabilities**: Seamless training resumption from any checkpoint
- **Error Recovery**: Graceful handling of training interruptions
- **Validation Checks**: Configuration and model validation before training

## Development & Testing

### Quality Assurance
- **Comprehensive Test Suite**: Unit tests, integration tests, and performance benchmarks
- **CI/CD Pipeline**: Automated testing, linting, and security scanning
- **Code Quality**: Black formatting, mypy type checking, and pylint analysis
- **Pre-commit Hooks**: Automated code quality checks before commits

### Development Workflow
```bash
# Run the full CI pipeline locally
./scripts/run_local_ci.sh

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/performance/   # Performance benchmarks

# Code quality checks
black keisei/              # Code formatting
mypy keisei/               # Type checking
pylint keisei/             # Code analysis
```

### Performance Monitoring
- **Training Profiling**: Built-in performance profiling tools
- **Memory Usage**: GPU and CPU memory monitoring
- **Throughput Metrics**: Steps per second and training efficiency
- **Resource Utilization**: Hardware utilization tracking

## Documentation

### Comprehensive Documentation
- **[Design Document](docs/DESIGN.md)**: Complete system architecture and design decisions
- **[Code Map](docs/CODE_MAP.md)**: Detailed codebase organization and component relationships
- **[Component Docs](docs/components/)**: Individual module documentation with examples
- **[CI/CD Guide](docs/CI_CD.md)**: Development workflow and quality assurance

### API Documentation
Each component includes comprehensive documentation:
- **Purpose and Responsibilities**: Clear component roles and boundaries
- **Configuration Options**: All available settings and their effects
- **Usage Examples**: Practical code examples and patterns
- **Testing Strategies**: Component-specific testing approaches

## Contributing

We welcome contributions! Please see our development guidelines:

1. **Fork the repository** and create a feature branch
2. **Install development dependencies**: `pip install -r requirements-dev.txt`
3. **Install pre-commit hooks**: `pre-commit install`
4. **Write tests** for new functionality
5. **Run the CI pipeline**: `./scripts/run_local_ci.sh`
6. **Submit a pull request** with a clear description

### Development Standards
- **Code Quality**: All code must pass black, mypy, and pylint checks
- **Test Coverage**: New features require comprehensive test coverage
- **Documentation**: Update documentation for any API changes
- **Type Safety**: Use type hints throughout the codebase

## License

I don't believe AI generated code is eligible for copyright but for avoidance of doubt: This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
