# Keisei

Deep Reinforcement Learning for Shogi, powered by a Rust game engine.

Keisei (形成, "formation") trains neural network agents to play Shogi
(Japanese chess) using Proximal Policy Optimization (PPO). The game engine and
RL environment are written in Rust for performance; the training harness is
Python with PyTorch.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Python Training Harness (keisei)               │
│  ┌──────────┐ ┌─────────┐ ┌──────────────────┐  │
│  │ PPO      │ │ Models  │ │ Training Loop    │  │
│  │ Algorithm│ │ ResNet  │ │ Config, DB,      │  │
│  │ + GAE    │ │ MLP     │ │ Checkpoints      │  │
│  │          │ │ Transf. │ │                  │  │
│  └──────────┘ └─────────┘ └──────────────────┘  │
│                     │                            │
│              PyO3 bindings                       │
├─────────────────────────────────────────────────┤
│  Rust Engine (shogi-engine)                     │
│  ┌─────────────────────┐ ┌────────────────────┐ │
│  │ shogi-core          │ │ shogi-gym          │ │
│  │ Board, Pieces,      │ │ VecEnv, Obs,       │ │
│  │ Move Generation,    │ │ Action Mapping,    │ │
│  │ Rules, SFEN, Zobrist│ │ Spectator Data     │ │
│  └─────────────────────┘ └────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### Rust Engine (`shogi-engine/`)

Two workspace crates providing the core game logic:

- **shogi-core** — Full Shogi implementation: board representation, legal move
  generation, rule enforcement (check, checkmate, repetition, impasse), SFEN
  parsing, and Zobrist hashing.
- **shogi-gym** — RL environment layer: vectorized environment (`VecEnv`),
  46-plane board observations, 13,527-action mapping, spectator data for live
  visualization, and step/reset API exposed to Python via PyO3.

### Python Training Harness (`keisei/`)

- **Config** — TOML-based configuration with dataclass validation.
- **Models** — Three neural network architectures (ResNet, MLP, Transformer),
  each outputting a policy head (13,527 actions) and a value head.
- **PPO** — Proximal Policy Optimization with GAE, clipped surrogate objective,
  rollout buffer, and mini-batch updates.
- **Training Loop** — Orchestrates the full training cycle: environment
  interaction, PPO updates, metric logging to SQLite, checkpointing, and
  resume-from-checkpoint support.
- **Database** — SQLite layer (WAL mode) storing training metrics, game
  snapshots, and training state for the spectator WebUI.

## Requirements

- Python >= 3.12
- Rust toolchain (for building the engine)
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

## Getting Started

```bash
# Clone the repository
git clone git@github.com:tachyon-beep/keisei.git
cd keisei

# Install Python dependencies (editable, with dev tools)
uv pip install -e ".[dev]"

# Build the Rust engine (happens automatically via PyO3 on import,
# or build manually)
cd shogi-engine && cargo build --release && cd ..

# Run training
uv run keisei-train --config keisei.toml --epochs 100 --steps-per-epoch 256
```

## Configuration

Training is configured via a TOML file. See `keisei.toml` for the default
configuration:

| Section                      | Key                | Default     | Description                        |
|------------------------------|--------------------|-------------|------------------------------------|
| `[training]`                 | `num_games`        | 8           | Parallel environments (1-10)       |
|                              | `max_ply`          | 500         | Max moves per game before truncation |
|                              | `algorithm`        | `"ppo"`     | Training algorithm                 |
|                              | `checkpoint_interval` | 50       | Epochs between checkpoints         |
| `[training.algorithm_params]`| `learning_rate`    | 3e-4        | Adam learning rate                 |
|                              | `gamma`            | 0.99        | Discount factor                    |
|                              | `clip_epsilon`     | 0.2         | PPO clipping parameter             |
|                              | `epochs_per_batch` | 4           | PPO update epochs per rollout      |
|                              | `batch_size`       | 256         | Mini-batch size                    |
| `[model]`                    | `architecture`     | `"resnet"`  | `resnet`, `mlp`, or `transformer`  |
| `[display]`                  | `moves_per_minute` | 30          | Spectator snapshot rate            |

## Development

```bash
# Run Python tests
uv run pytest

# Run Rust tests
cd shogi-engine && cargo test

# Lint
uv run ruff check .
uv run mypy keisei/
```

## Testing

- **74 Python tests** covering config loading, database operations,
  checkpointing, model architectures, PPO algorithm, and the training loop.
- **188 Rust tests** across shogi-core (move generation, rules, SFEN, position
  logic) and shogi-gym (observations, action mapping, VecEnv, spectator data).

## License

MIT — see [LICENSE](LICENSE).
