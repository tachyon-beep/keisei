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
- WebUI requires `streamlit` package: `uv pip install keisei[webui]` or `uv pip install streamlit`

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

# Dependencies are defined in pyproject.toml
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
- **Markers**: Defined in `pyproject.toml` (unit, integration, e2e, slow, performance) but not yet applied to test functions — use directory-based test selection instead
- **CI/CD**: Local CI via `./scripts/run_local_ci.sh` (GitHub Actions CI is currently disabled)

<!-- filigree:instructions:v1.5.1:63b4188e -->
## Filigree Issue Tracker

Use `filigree` for all task tracking in this project. Data lives in `.filigree/`.

### MCP Tools (Preferred)

When MCP is configured, prefer `mcp__filigree__*` tools over CLI commands — they're
faster and return structured data. Key tools:

- `get_ready` / `get_blocked` — find available work
- `get_issue` / `list_issues` / `search_issues` — read issues
- `create_issue` / `update_issue` / `close_issue` — manage issues
- `claim_issue` / `claim_next` — atomic claiming
- `add_comment` / `add_label` — metadata
- `list_labels` / `get_label_taxonomy` — discover labels and reserved namespaces
- `create_plan` / `get_plan` — milestone planning
- `get_stats` / `get_metrics` — project health
- `get_valid_transitions` — workflow navigation
- `observe` / `list_observations` / `dismiss_observation` / `promote_observation` — agent scratchpad
- `trigger_scan` / `trigger_scan_batch` / `get_scan_status` / `preview_scan` / `list_scanners` — automated code scanning
- `get_finding` / `list_findings` / `update_finding` / `batch_update_findings` — scan finding triage
- `promote_finding` / `dismiss_finding` — finding lifecycle (promote to issue or dismiss)

Observations are fire-and-forget notes that expire after 14 days. Use `list_issues --label=from-observation` to find promoted observations.

**Observations are ambient.** While doing other work, use `observe` whenever you
notice something worth noting — a code smell, a potential bug, a missing test, a
design concern. Don't stop what you're doing; just fire off the observation and
carry on. They're ideal for "I don't have time to investigate this right now, but
I want to come back to it." Include `file_path` and `line` when relevant so the
observation is anchored to code. At session end, skim `list_observations` and
either `dismiss_observation` (not worth tracking) or `promote_observation`
(deserves an issue) for anything that's accumulated.

Fall back to CLI (`filigree <command>`) when MCP is unavailable.

### CLI Quick Reference

```bash
# Finding work
filigree ready                              # Show issues ready to work (no blockers)
filigree list --status=open                 # All open issues
filigree list --status=in_progress          # Active work
filigree list --label=bug --label=P1        # Filter by multiple labels (AND)
filigree list --label-prefix=cluster/       # Filter by label namespace prefix
filigree list --not-label=wontfix           # Exclude issues with label
filigree show <id>                          # Detailed issue view

# Creating & updating
filigree create "Title" --type=task --priority=2          # New issue
filigree update <id> --status=in_progress                # Claim work
filigree close <id>                                      # Mark complete
filigree close <id> --reason="explanation"               # Close with reason

# Dependencies
filigree add-dep <issue> <depends-on>       # Add dependency
filigree remove-dep <issue> <depends-on>    # Remove dependency
filigree blocked                            # Show blocked issues

# Comments & labels
filigree add-comment <id> "text"            # Add comment
filigree get-comments <id>                  # List comments
filigree add-label <id> <label>             # Add label
filigree remove-label <id> <label>          # Remove label
filigree labels                             # List all labels by namespace
filigree taxonomy                           # Show reserved namespaces and vocabulary

# Workflow templates
filigree types                              # List registered types with state flows
filigree type-info <type>                   # Full workflow definition for a type
filigree transitions <id>                   # Valid next states for an issue
filigree packs                              # List enabled workflow packs
filigree validate <id>                      # Validate issue against template
filigree guide <pack>                       # Display workflow guide for a pack

# Atomic claiming
filigree claim <id> --assignee <name>            # Claim issue (optimistic lock)
filigree claim-next --assignee <name>            # Claim highest-priority ready issue

# Batch operations
filigree batch-update <ids...> --priority=0      # Update multiple issues
filigree batch-close <ids...>                    # Close multiple with error reporting

# Planning
filigree create-plan --file plan.json            # Create milestone/phase/step hierarchy

# Event history
filigree changes --since 2026-01-01T00:00:00    # Events since timestamp
filigree events <id>                             # Event history for issue
filigree explain-state <type> <state>            # Explain a workflow state

# All commands support --json and --actor flags
filigree --actor bot-1 create "Title"            # Specify actor identity
filigree list --json                             # Machine-readable output

# Project health
filigree stats                              # Project statistics
filigree search "query"                     # Search issues
filigree doctor                             # Health check
```

### File Records & Scan Findings (API)

The dashboard exposes REST endpoints for file tracking and scan result ingestion.
Use `GET /api/files/_schema` for available endpoints and valid field values.

Key endpoints:
- `GET /api/files/_schema` — Discovery: valid enums, endpoint catalog
- `POST /api/v1/scan-results` — Ingest scan results (SARIF-lite format)
- `GET /api/files` — List tracked files with filtering and sorting
- `GET /api/files/{file_id}` — File detail with associations and findings summary
- `GET /api/files/{file_id}/findings` — Findings for a specific file

### Workflow
1. `filigree ready` to find available work
2. `filigree show <id>` to review details
3. `filigree transitions <id>` to see valid state changes
4. `filigree update <id> --status=in_progress` to claim it
5. Do the work, commit code
6. `filigree close <id>` when done

### Session Start
When beginning a new session, run `filigree session-context` to load the project
snapshot (ready work, in-progress items, critical path). This provides the
context needed to pick up where the previous session left off.

### Priority Scale
- P0: Critical (drop everything)
- P1: High (do next)
- P2: Medium (default)
- P3: Low
- P4: Backlog
<!-- /filigree:instructions -->
