# Keisei

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy%20strict-blue.svg)](http://mypy-lang.org/)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](https://pre-commit.com/)

Deep Reinforcement Learning for Shogi, powered by a Rust game engine.

Keisei (形成, "to give form to, to mold, to shape") trains neural network agents to play Shogi
(Japanese chess) using Proximal Policy Optimization (PPO). The game engine and
RL environment are written in Rust for performance; the training harness is
Python with PyTorch.

> **Status:** Early rebuild. The Rust engine is functional; the Python training
> harness is under active development with a KataGo-inspired multi-head
> architecture as the primary training target.

## Prior Art and Attribution

Keisei's neural network design draws heavily from two published systems:

- **[KataGo](https://github.com/lightvector/KataGo)** (David Wu, 2019) — The
  SE-ResNet trunk with global pooling bias, the Win/Draw/Loss value head, and
  the score prediction auxiliary head are adapted from KataGo's architecture.
  KataGo was designed for Go; Keisei adapts these ideas for Shogi's different
  action space, piece-drop mechanics, and promotion rules.
- **[AlphaZero](https://arxiv.org/abs/1712.01815)** (Silver et al., 2018) —
  The general approach of residual convolutional networks with separate policy
  and value heads, trained via self-play, originates from DeepMind's
  AlphaGo/AlphaZero line of work.

**What Keisei adds on top of these foundations:**

- A Rust game engine with vectorized environments exposed to Python via PyO3,
  replacing the typical C++ engine pattern.
- Shogi-specific observation encoding (50-channel, perspective-relative) with
  hand-piece normalization and repetition/check planes.
- A spatial action decomposition (81 squares x 139 move types) tailored to
  Shogi's move semantics, including drops and promotions.
- A dual-contract model system with adapter pattern, allowing the training loop
  to support both simple scalar-value models and KataGo-style multi-head models
  without branching.
- PPO-based training (KataGo uses a custom self-play + training pipeline;
  AlphaZero uses MCTS + supervised learning from self-play).

## Architecture

| Layer | Component | Description |
|-------|-----------|-------------|
| **Python Training Harness** (`keisei`) | PPO / GAE | KataGo-PPO, Value Adapters |
| | Models | SE-ResNet, ResNet, MLP, Transformer |
| | Training Loop | Config, DB, Checkpoints, SL Warmup |
| | | *PyO3 bindings* |
| **Rust Engine** (`shogi-engine`) | shogi-core | Board, Pieces, Move Generation, Rules, SFEN, Zobrist |
| | shogi-gym | VecEnv, Obs (46/50 channel), Action Mapping, Spectator |

### Rust Engine (`shogi-engine/`)

Two workspace crates providing the core game logic:

- **shogi-core** — Full Shogi implementation: board representation, legal move
  generation, rule enforcement (check, checkmate, repetition, impasse), SFEN
  parsing, and Zobrist hashing.
- **shogi-gym** — RL environment layer: vectorized environment (`VecEnv`),
  observation encoding (46 or 50 channels), action mapping (11,259 spatial
  actions), spectator data for live visualization, and step/reset API exposed
  to Python via PyO3.

### Python Training Harness (`keisei/`)

- **Config** — TOML-based configuration with dataclass validation.
- **Models** — Four neural network architectures with a registry-based
  dispatch system. See [Neural Network Architecture](#neural-network-architecture)
  below.
- **PPO** — Two algorithm variants: standard PPO (scalar value) and
  KataGo-style PPO (W/D/L + score heads). Both use GAE, clipped surrogate
  objective, and mini-batch updates.
- **Value Adapters** — Adapter pattern that abstracts over scalar vs. multi-head
  value outputs, so the training loop is model-agnostic.
- **Training Loop** — Orchestrates environment interaction, PPO updates, metric
  logging to SQLite, checkpointing, and resume-from-checkpoint support.
- **Database** — SQLite layer (WAL mode) storing training metrics, game
  snapshots, and training state for the spectator WebUI.

## Neural Network Architecture

Keisei supports four architectures via a registry. The **SE-ResNet** (adapted
from [KataGo](https://github.com/lightvector/KataGo)) is the primary training
target; the others serve as baselines and ablations.

| Architecture   | Value Head     | Channels | Use Case                    |
|----------------|----------------|----------|-----------------------------|
| `se_resnet`    | W/D/L + Score  | 50       | **Primary training target** |
| `resnet`       | Scalar         | 46       | Baseline                    |
| `mlp`          | Scalar         | 46       | Debugging / ablation        |
| `transformer`  | Scalar         | 46       | Experimental                |

The SE-ResNet outputs three heads: a **spatial policy** `(B, 9, 9, 139)` over
81 squares x 139 move types, a **W/D/L value** classification (Win/Draw/Loss —
a KataGo innovation), and a **score** prediction for material balance (auxiliary
task). Two PPO variants match the two model contracts: standard PPO for scalar
models, KataGo-PPO for multi-head.

<details>
<summary><strong>SE-ResNet Architecture Detail</strong></summary>

### SE-ResNet (Primary Architecture)

Adapted from [KataGo](https://github.com/lightvector/KataGo)'s neural network
design (David Wu, 2019).

**Trunk:** An input convolution followed by a configurable number of residual
blocks (default: 40 blocks, 256 channels). Each block is a
`GlobalPoolBiasBlock` — a KataGo-originated design where:

1. A standard `conv -> BN -> ReLU` path processes local spatial features.
2. A **global pooling bias** (mean + max + std of the *block input*, projected
   through a bottleneck FC) is broadcast-added after the first convolution.
   This injects global board context into the local convolutional pathway.
3. A **Squeeze-and-Excitation** (SE) mechanism with scale+shift (not just
   scale) applies channel-wise affine attention after the second convolution.
4. A residual connection adds the block input back before the final ReLU.

**Three output heads:**

| Head       | Shape             | Activation       | Loss             | Purpose                                    |
|------------|-------------------|------------------|------------------|--------------------------------------------|
| **Policy** | `(B, 9, 9, 139)` | Legal-masked softmax | Clipped PPO surrogate | Per-square move-type probabilities     |
| **Value**  | `(B, 3)`          | Softmax (W/D/L)  | Cross-entropy    | Win/Draw/Loss classification               |
| **Score**  | `(B, 1)`          | None (raw)       | MSE              | Material balance prediction (auxiliary)    |

The value and score heads share a global pool `(B, 3C)` computed once from the
trunk output. The scalar value used for GAE bootstrapping is derived as
`P(Win) - P(Loss)` from the softmax of the W/D/L logits.

> **Design note:** The W/D/L value head is a KataGo innovation. It preserves
> the distinction between "50% win / 50% draw" and "50% win / 50% loss" — both
> would map to the same scalar ~0.0, but represent very different positions.
> The score head is an auxiliary task from KataGo that regularizes trunk
> features; its loss weight (`lambda_score=0.02`) is intentionally small.

</details>

<details>
<summary><strong>Observation Encoding (50-channel)</strong></summary>

### Observation Encoding

The board state is encoded as a multi-channel 9x9 tensor by the Rust engine.
All observations are **perspective-relative** (channels 0-13 are always "current
player's pieces", not always "Black's pieces"), following the AlphaZero
convention.

| Channels | Content                                         | Encoding     |
|----------|-------------------------------------------------|--------------|
| 0-13     | Current player's pieces (8 unpromoted + 6 promoted) | Binary (0/1) |
| 14-27    | Opponent's pieces (same layout)                 | Binary (0/1) |
| 28-34    | Current player's hand counts (7 piece types)    | Normalized by max possible count |
| 35-41    | Opponent's hand counts                          | Normalized by max possible count |
| 42       | Player color indicator (1.0=Black, 0.0=White)   | Constant plane |
| 43       | Move count (ply / max_ply)                      | Constant plane |
| 44-47    | Repetition count (binary planes: 1x, 2x, 3x, 4+) | Binary (SE-ResNet only) |
| 48       | Check indicator (1.0 if in check)               | Binary (SE-ResNet only) |
| 49       | Reserved                                        | Zeros        |

The 46-channel encoding (channels 0-45) is used by the scalar-value models.
The 50-channel encoding adds repetition and check awareness for the SE-ResNet,
which are important for Shogi endgame play (repetition can end the game via
sennichite).

</details>

<details>
<summary><strong>Action Space and Legal Masking</strong></summary>

### Action Space

Shogi moves are encoded spatially as `(source_square, move_type)`:

- **81 squares** on the 9x9 board
- **139 move types** per square (directional moves with optional promotion,
  plus piece drops)
- **Total: 11,259** spatial actions (SE-ResNet) or **13,527** flat actions
  (scalar models, includes padding for a different decomposition)

Illegal actions are masked to `-inf` before softmax, guaranteeing zero
probability. The training loop includes runtime guards against all-zero legal
masks (which would produce NaN from softmax).

</details>

<details>
<summary><strong>PPO Training and Loss Function</strong></summary>

### PPO Training

Two algorithm variants:

| Algorithm     | Value Head      | Score Head | Compatible Models |
|---------------|-----------------|------------|-------------------|
| `ppo`         | Scalar MSE      | No         | resnet, mlp, transformer |
| `katago_ppo`  | W/D/L cross-entropy | MSE (normalized) | se_resnet only |

**KataGo-PPO loss function:**

```
L = lambda_policy * L_policy + lambda_value * L_value + lambda_score * L_score - lambda_entropy * H(pi)
```

Default weights: `lambda_policy=1.0`, `lambda_value=1.5`, `lambda_score=0.1`,
`lambda_entropy=0.01`. The higher weight on value reflects the priority of
accurate position evaluation in early training — good advantage estimates
require good value predictions.

Score targets are raw material difference divided by 76.0 (approximate max
material for one side), mapping to roughly [-2.6, +2.6].

</details>

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

# Run training with the KataGo SE-ResNet (primary config)
uv run keisei-train keisei-katago.toml --epochs 100 --steps-per-epoch 256

# Or with the simpler ResNet baseline
uv run keisei-train keisei.toml --epochs 100 --steps-per-epoch 256
```

### CLI Tools

| Command             | Description                              |
|---------------------|------------------------------------------|
| `keisei-train`      | Run RL training                          |
| `keisei-evaluate`   | Evaluate a trained checkpoint            |
| `keisei-serve`      | Launch the spectator WebUI               |
| `keisei-prepare-sl` | Prepare supervised learning datasets     |

## Configuration

Training is configured via TOML files. Two reference configurations are
provided:

**`keisei-katago.toml`** — KataGo SE-ResNet (primary):

| Section                       | Key                   | Default     | Description                          |
|-------------------------------|-----------------------|-------------|--------------------------------------|
| `[training]`                  | `num_games`           | 128         | Parallel environments                |
|                               | `max_ply`             | 512         | Max moves per game before truncation |
|                               | `algorithm`           | `katago_ppo`| Multi-head PPO with W/D/L + score    |
| `[training.algorithm_params]` | `learning_rate`       | 2e-4        | Adam learning rate                   |
|                               | `gamma`               | 0.99        | Discount factor                      |
|                               | `gae_lambda`          | 0.95        | GAE lambda                           |
|                               | `clip_epsilon`        | 0.2         | PPO clipping parameter               |
|                               | `epochs_per_batch`    | 4           | PPO update epochs per rollout        |
|                               | `batch_size`          | 256         | Mini-batch size                      |
|                               | `lambda_value`        | 1.5         | Value loss weight                    |
|                               | `lambda_score`        | 0.1         | Score loss weight                    |
|                               | `lambda_entropy`      | 0.01        | Entropy bonus weight                 |
|                               | `grad_clip`           | 1.0         | Global gradient norm clip            |
| `[model]`                     | `architecture`        | `se_resnet` | SE-ResNet with KataGo-style heads    |
| `[model.params]`              | `num_blocks`          | 40          | Residual blocks in trunk             |
|                               | `channels`            | 256         | Channel width                        |
|                               | `se_reduction`        | 16          | SE bottleneck ratio                  |

**`keisei.toml`** — ResNet baseline:

| Section                       | Key              | Default    | Description                     |
|-------------------------------|------------------|------------|---------------------------------|
| `[training]`                  | `algorithm`      | `ppo`      | Standard scalar-value PPO       |
| `[model]`                     | `architecture`   | `resnet`   | Plain ResNet                    |
| `[model.params]`              | `hidden_size`    | 256        | Channel width                   |
|                               | `num_layers`     | 8          | Residual blocks                 |

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

- **674 Python tests** covering config loading, database operations,
  checkpointing, all four model architectures, both PPO variants, value
  adapters, the training loop, and supervised learning preparation.
- **363 Rust tests** across shogi-core (move generation, rules, SFEN, position
  logic) and shogi-gym (observations, action mapping, VecEnv, spectator data).

## License

MIT — see [LICENSE](LICENSE).

### Third-Party Assets

The shogi piece icon (`images/shogi.svg`, used as the WebUI favicon) is
[Shogi gyokusho](https://commons.wikimedia.org/wiki/File:Shogi_gyokusho%28svg%29.svg)
by [Hari Seldon](https://commons.wikimedia.org/wiki/User:Hari_Seldon), licensed
under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/deed.en).
